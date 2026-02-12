import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F

from src.config import load_config, resolve_config
from src.model_utils import load_model
from src.hook_resolver import resolve_hook_point
from src.sae_loader import load_gemmascope_decoder


def _get_main_tensor(outp):
    return outp[0] if isinstance(outp, tuple) else outp


def main():
    cfg_path = "configs/targets/gemma2_2b_gemmascope_res16k.yaml"
    run_id = "debug_steer_effect"
    config = resolve_config(load_config(cfg_path), run_id=run_id)

    model, tokenizer = load_model(config)
    model.eval()
    device = next(model.parameters()).device

    hook_name = config["sae"]["hook_point"]
    hook_module = resolve_hook_point(model, hook_name)

    # Load decoder + full NPZ for encoder/bias/threshold
    decoder_cpu, meta = load_gemmascope_decoder(config)
    npz_path = meta["npz_path"]
    data = np.load(npz_path)

    # All SAE params in float32 (bf16 hides small diffs)
    W_dec = torch.tensor(np.asarray(data["W_dec"], dtype=np.float32), device=device, dtype=torch.float32)
    W_enc = torch.tensor(np.asarray(data["W_enc"], dtype=np.float32), device=device, dtype=torch.float32)
    b_enc = torch.tensor(np.asarray(data["b_enc"], dtype=np.float32), device=device, dtype=torch.float32)
    thr   = torch.tensor(np.asarray(data["threshold"], dtype=np.float32), device=device, dtype=torch.float32)

    print(f"W_dec shape: {W_dec.shape}")  # expect (n_features, d_model)
    print(f"W_enc shape: {W_enc.shape}")  # could be (d_model, n_features) or (n_features, d_model)
    print(f"b_enc shape: {b_enc.shape}")
    print(f"thr   shape: {thr.shape}")

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

    ALPHAS = [0.0, 1.0, 2.0, 5.0, -5.0]
    captured_resid = []

    def make_hook(alpha_val):
        def hook_fn(module, inp, outp):
            t = _get_main_tensor(outp)
            vec = W_dec[fid].to(dtype=t.dtype)
            t2 = t.clone()
            t2[:, -1, :] = t2[:, -1, :] + alpha_val * vec
            captured_resid.append(t2.detach())
            if isinstance(outp, tuple):
                return (t2,) + outp[1:]
            return t2
        return hook_fn

    print()
    print("PROMPT:", repr(prompt))
    print("hook_point:", hook_name)
    print("feature_id:", fid)
    print("target token:", repr(target_text), "id:", tid)
    print()

    # ---- Run alpha grid, capture resid, compute z/act/logit/projection ----
    results = []
    base_resid_last = None
    for alpha in ALPHAS:
        captured_resid.clear()
        handle = hook_module.register_forward_hook(make_hook(alpha))
        with torch.no_grad():
            out = model(**inputs)
            logits = out.logits[0, -1].float()
        handle.remove()
        if not captured_resid:
            raise RuntimeError("Hook did not capture residual")
        resid_last = captured_resid[0][0, -1, :].float()  # (d_model,) in float32
        if alpha == 0.0:
            base_resid_last = resid_last.clone()
        z, act = z_and_act(resid_last, fid)
        logit_target = logits[tid].item()
        results.append((alpha, z.item(), act.item(), logit_target, resid_last))

    base_logit = next(lt for a, _, _, lt, _ in results if a == 0.0)
    norm2_wdec = torch.dot(W_dec[fid], W_dec[fid]).item()

    print(f"{'alpha':>6}  {'z_f':>12}  {'thr':>10}  {'act(fid)':>10}  {'logit':>10}  {'Δlogit':>10}  {'proj(Δr,Wdec)':>14}  {'expected':>14}")
    print("-" * 100)
    for alpha, z_val, act_val, logit_val, resid_last in results:
        delta_logit = logit_val - base_logit
        # Projection of residual delta onto W_dec[fid]
        if base_resid_last is not None and alpha != 0.0:
            delta_r = resid_last - base_resid_last
            proj = torch.dot(delta_r, W_dec[fid]).item()
            expected = alpha * norm2_wdec
        else:
            proj = 0.0
            expected = 0.0
        print(f"alpha={alpha:+5.1f}  z={z_val:+12.6f}  thr={thr[fid].item():+10.6f}  act={act_val:10.6f}  "
              f"logit={logit_val:+10.4f}  Δlogit={delta_logit:+10.4f}  "
              f"proj={proj:+14.6f}  expected~{expected:+14.6f}")

    # ---- Top-10 next tokens at alpha=0 and alpha=+5 ----
    def show_top(logits, title):
        top = torch.topk(logits, k=10)
        print(f"\n{title} top-10 next tokens:")
        for v, i in zip(top.values.tolist(), top.indices.tolist()):
            print(f"  logit={v:+.4f}  id={i:<7} text={tokenizer.decode([i])!r}")

    # Re-run for clean top-10 display
    for a_show, label in [(0.0, "BASE (alpha=0)"), (5.0, "STEERED (alpha=+5)")]:
        captured_resid.clear()
        handle = hook_module.register_forward_hook(make_hook(a_show))
        with torch.no_grad():
            out = model(**inputs)
            logits = out.logits[0, -1].float()
        handle.remove()
        show_top(logits, label)


if __name__ == "__main__":
    main()
