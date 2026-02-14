import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import random
import numpy as np
import torch
import torch.nn.functional as F

from src.config import load_config, resolve_config
from src.model_utils import load_model
from src.hook_resolver import resolve_hook_point
from src.sae_loader import load_gemmascope_decoder

# Determinism
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(True)


def _get_main_tensor(outp):
    return outp[0] if isinstance(outp, tuple) else outp


def sae_z_and_act(resid_vec: torch.Tensor, W_enc: torch.Tensor, b_enc: torch.Tensor, thr: torch.Tensor):
    """
    resid_vec: [d_model] at a specific token position
    W_enc: [d_model, n_feats]
    b_enc, thr: [n_feats]
    Returns z: [n_feats], act: [n_feats] where act = relu(z - thr)
    """
    resid_vec = resid_vec.view(-1)
    z = resid_vec @ W_enc + b_enc
    act = torch.relu(z - thr)
    return z, act


def print_top_active_features(resid_vec, W_enc, b_enc, thr, topk=10):
    z, act = sae_z_and_act(resid_vec, W_enc, b_enc, thr)
    vals, idx = torch.topk(act, k=topk)
    print("\nTOP ACTIVE SAE FEATURES (by act = relu(z - thr))")
    print(" fid        act        z        thr     (z-thr)")
    print("------------------------------------------------")
    for v, i in zip(vals.tolist(), idx.tolist()):
        zi = z[i].item()
        thi = thr[i].item()
        print(f"{i:5d}  {v:9.6f}  {zi:8.4f}  {thi:8.4f}  {zi-thi:9.4f}")
    num_active = (act > 0).sum().item()
    print(f"\nActive features count: {num_active} / {act.numel()}")
    return idx, z, act


def run_once_generate(model, tokenizer, prompt, max_new_tokens=20):
    """Run model.generate (greedy, deterministic) and return tokens + per-step scores."""
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            num_beams=1,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    # out.sequences: (1, prompt_len + generated_len)
    # out.scores: tuple of (1, vocab_size) per generated step
    tokens = out.sequences[0]  # (seq_len,)
    scores = torch.stack(out.scores, dim=0)  # (n_steps, vocab_size)
    return {"tokens": tokens, "scores": scores}


def main():
    cfg_path = "configs/targets/gemma2_2b_gemmascope_res16k.yaml"
    run_id = "debug_steer_effect"
    config = resolve_config(load_config(cfg_path), run_id=run_id)

    # Force greedy / no sampling in config
    config["model"]["do_sample"] = False
    config["model"]["temperature"] = 0.0

    model, tokenizer = load_model(config)
    model.eval()
    device = next(model.parameters()).device

    hook_name = config["sae"]["hook_point"]
    layer_idx = config["sae"]["layer_idx"]  # e.g. 20
    # Hook directly on the HF decoder layer (not via resolve_hook_point — that was mapping
    # a TL name but the hook return was not propagating through the model forward).
    hook_module = model.model.layers[layer_idx]
    print(f"Hooking directly: model.model.layers[{layer_idx}] ({type(hook_module).__name__})")

    # Load decoder + full NPZ for encoder/bias/threshold
    decoder_cpu, meta = load_gemmascope_decoder(config)
    npz_path = meta["npz_path"]
    data = np.load(npz_path)

    # All SAE params in float32 (bf16 hides small diffs)
    W_dec = torch.tensor(np.asarray(data["W_dec"], dtype=np.float32), device=device, dtype=torch.float32)
    W_enc = torch.tensor(np.asarray(data["W_enc"], dtype=np.float32), device=device, dtype=torch.float32)
    b_enc = torch.tensor(np.asarray(data["b_enc"], dtype=np.float32), device=device, dtype=torch.float32)
    thr   = torch.tensor(np.asarray(data["threshold"], dtype=np.float32), device=device, dtype=torch.float32)

    # Unembedding weight for predicted Δlogit
    lm_head_w = model.lm_head.weight.detach().float()  # [vocab, d_model]

    print(f"W_dec shape: {W_dec.shape}")  # expect (n_features, d_model)
    print(f"W_enc shape: {W_enc.shape}")  # could be (d_model, n_features) or (n_features, d_model)
    print(f"b_enc shape: {b_enc.shape}")
    print(f"thr   shape: {thr.shape}")
    print(f"lm_head_w shape: {lm_head_w.shape}")  # [vocab, d_model]

    d_model = W_dec.shape[1]

    # Robust z_f computation: handle either W_enc orientation
    def z_and_act(resid_last_f32, fid):
        """Compute pre-activation z and SAE activation for one feature."""
        if W_enc.shape[0] == resid_last_f32.numel():        # (d_model, n_feat)
            z = (resid_last_f32 @ W_enc[:, fid]) + b_enc[fid]
        elif W_enc.shape[1] == resid_last_f32.numel():      # (n_feat, d_model)
            z = (resid_last_f32 @ W_enc[fid, :]) + b_enc[fid]
        else:
            raise ValueError(f"W_enc shape {W_enc.shape} incompatible with d_model={resid_last_f32.numel()}")
        act = torch.relu(z - thr[fid])
        return z, act

    prompt = "One plus one equals"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    fid = 2317
    target_text = " two"
    target_ids = tokenizer.encode(target_text, add_special_tokens=False)
    if len(target_ids) != 1:
        raise ValueError(f"Target {target_text!r} must be single token, got ids={target_ids}")
    tid = target_ids[0]

    ALPHAS = [0.0, 0.25, 0.5, 1.0, 2.0, 5.0, -1.0, -2.0, -5.0]
    pos = -1  # token position to steer (last token)

    _prehook_diag_printed = [False]

    def make_steer_prehook(alpha_val, capture):
        def _prehook(module, inputs):
            capture["ran"] = True

            # inputs is a tuple; first element is hidden_states
            hidden = inputs[0]

            # One-time diagnostic
            if not _prehook_diag_printed[0]:
                print(f"PREHOOK inputs[0] shape: {hidden.shape}")
                print(f"PREHOOK inputs tuple len: {len(inputs)}")
                _prehook_diag_printed[0] = True

            hidden2 = hidden.clone()

            # Capture pre-steering
            capture["resid_pre"] = hidden2[0, pos].detach().clone()

            # Apply steering
            steer_dir = W_dec[fid].to(dtype=hidden2.dtype)
            hidden2[0, pos, :] = hidden2[0, pos, :] + alpha_val * steer_dir

            # Capture post-steering
            capture["resid_post"] = hidden2[0, pos].detach().clone()

            # Return new inputs tuple with hidden replaced
            return (hidden2,) + inputs[1:]
        return _prehook

    def run_once(alpha_val):
        """Single forward pass with steering via pre-hook. Returns dict with logits, tokens, resid_pre, resid_post, full_logits."""
        capture = {"ran": False, "resid_pre": None, "resid_post": None}
        handle = hook_module.register_forward_pre_hook(make_steer_prehook(alpha_val, capture))
        with torch.no_grad():
            out = model(**inputs)
            full_logits = out.logits.float()       # [1, seq, vocab]
            logits = full_logits[0, -1]             # last position
        handle.remove()
        if not capture["ran"]:
            raise RuntimeError("Hook never ran. You're not actually attached to the right hook point/module.")
        if capture["resid_post"] is None:
            raise RuntimeError("Hook ran but resid_post is None. Indexing/pos logic is wrong.")
        resid_pre = capture["resid_pre"].float()
        resid_post = capture["resid_post"].float()
        tokens = logits.argmax(dim=-1, keepdim=True)
        return {"logits": logits, "tokens": tokens,
                "resid_last": resid_post, "resid_pre": resid_pre,
                "resid_post": resid_post,
                "full_logits": full_logits, "hook_ran": capture["ran"]}

    print()
    print("PROMPT:", repr(prompt))
    print("hook_point:", hook_name)
    print("feature_id:", fid)
    print("target token:", repr(target_text), "id:", tid)
    print()

    # ---- Determinism check: run twice, assert exact match ----
    print("=== Determinism check (alpha=1.0, two identical runs) ===")
    out1 = run_once(1.0)
    out2 = run_once(1.0)
    assert (out1["tokens"] == out2["tokens"]).all(), "NON-DETERMINISTIC: greedy token ids differ"
    torch.testing.assert_close(out1["logits"], out2["logits"], rtol=0, atol=0)
    print("PASS: logits and tokens are bitwise identical across two runs.")
    print()

    # ---- Baseline pass (alpha=0): find top active features ----
    r0 = run_once(0.0)
    base_resid_last = r0["resid_last"]
    print("=== Top active SAE features at baseline (alpha=0) ===")
    top_idx, _, _ = print_top_active_features(
        base_resid_last, W_enc, b_enc, thr, topk=20
    )
    fid = int(top_idx[0].item())
    print(f"\nUsing feature_id={fid} (most active) for steering next.\n")

    # ---- Manual vs model logit diagnostic ----
    print("=== Manual vs model logit diagnostic (alpha=5.0) ===")
    r_base = run_once(0.0)
    r_steer = run_once(5.0)
    print("DEBUG r_steer keys:", list(r_steer.keys()))
    print("DEBUG r_steer repr:", r_steer)
    resid_pre = r_base["resid_pre"]
    resid_post = r_steer["resid_post"]
    manual_base = torch.dot(resid_pre, lm_head_w[tid]).item()
    manual_steer = torch.dot(resid_post, lm_head_w[tid]).item()
    manual_dlogit = manual_steer - manual_base
    print(f"MANUAL base/steer/delta: {manual_base:.4f} {manual_steer:.4f} {manual_dlogit:+.4f}")
    print(f"MODEL  logit[-1, tid]:   base={r_base['logits'][tid].item():.4f}  steer={r_steer['logits'][tid].item():.4f}  Δ={r_steer['logits'][tid].item() - r_base['logits'][tid].item():+.4f}")

    # Brute-force correct position
    T = r_steer["full_logits"].shape[1]
    print(f"\nBrute-force position scan (seq_len={T}):")
    for p in range(max(0, T - 4), T):
        base_l = r_base["full_logits"][0, p, tid].item()
        steer_l = r_steer["full_logits"][0, p, tid].item()
        print(f"  pos={p}  base={base_l:.4f}  steer={steer_l:.4f}  Δ={steer_l - base_l:+.4f}")
    print()

    # ---- Run alpha grid, capture resid, compute z/act/logit/projection ----
    results = []
    for alpha in ALPHAS:
        r = run_once(alpha)
        logits = r["logits"]
        resid_last = r["resid_last"]
        z, act = z_and_act(resid_last, fid)
        logit_target = logits[tid].item()
        results.append((alpha, z.item(), act.item(), logit_target, resid_last))

    base_logit = next(lt for a, _, _, lt, _ in results if a == 0.0)
    norm2_wdec = torch.dot(W_dec[fid], W_dec[fid]).item()

    wu_k = lm_head_w[tid]  # [d_model] unembedding for target token

    print(f"{'alpha':>6}  {'z_f':>12}  {'thr':>10}  {'act(fid)':>10}  {'logit':>10}  {'Δlogit':>10}  {'predΔl':>10}  {'proj(Δr,Wdec)':>14}  {'expected':>14}  {'maxΔl':>10}")
    print("-" * 120)
    for alpha, z_val, act_val, logit_val, resid_last in results:
        delta_logit = logit_val - base_logit
        # Projection of residual delta onto W_dec[fid]
        if base_resid_last is not None and alpha != 0.0:
            delta_r = resid_last - base_resid_last
            proj = torch.dot(delta_r, W_dec[fid]).item()
            expected = alpha * norm2_wdec
            pred_dlogit = torch.dot(delta_r, wu_k).item()
            pred_all = lm_head_w @ delta_r  # [vocab]
            pred_max = pred_all.abs().max().item()
        else:
            proj = 0.0
            expected = 0.0
            pred_dlogit = 0.0
            pred_max = 0.0
        print(f"alpha={alpha:+5.1f}  z={z_val:+12.6f}  thr={thr[fid].item():+10.6f}  act={act_val:10.6f}  "
              f"logit={logit_val:+10.4f}  Δlogit={delta_logit:+10.4f}  predΔl={pred_dlogit:+10.4f}  "
              f"proj={proj:+14.6f}  expected~{expected:+14.6f}  maxΔl={pred_max:10.4f}")

    # ---- Top-10 next tokens at alpha=0 and alpha=+5 ----
    def show_top(logits, title):
        top = torch.topk(logits, k=10)
        print(f"\n{title} top-10 next tokens:")
        for v, i in zip(top.values.tolist(), top.indices.tolist()):
            print(f"  logit={v:+.4f}  id={i:<7} text={tokenizer.decode([i])!r}")

    # Re-run for clean top-10 display
    for a_show, label in [(0.0, "BASE (alpha=0)"), (5.0, "STEERED (alpha=+5)")]:
        r = run_once(a_show)
        show_top(r["logits"], label)

    # ---- Generation determinism test ----
    gen_prompt = "What is 17 + 25? Answer with a single number."
    max_new = 20

    print("\n" + "=" * 60)
    print("GENERATION DETERMINISM TEST")
    print("=" * 60)
    print(f"Prompt: {gen_prompt!r}")
    print(f"max_new_tokens: {max_new}")

    out1 = run_once_generate(model, tokenizer, gen_prompt, max_new_tokens=max_new)
    out2 = run_once_generate(model, tokenizer, gen_prompt, max_new_tokens=max_new)

    assert torch.equal(out1["tokens"], out2["tokens"]), \
        "NON-DETERMINISTIC: token ids differ"
    torch.testing.assert_close(out1["scores"], out2["scores"], rtol=0, atol=0)
    print("PASS: tokens and scores are bitwise identical across two runs.")
    print(f"Generated: {tokenizer.decode(out1['tokens'], skip_special_tokens=True)!r}")


if __name__ == "__main__":
    main()
