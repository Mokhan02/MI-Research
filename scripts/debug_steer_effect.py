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
    # HF decoder layer often returns tuple; first element is hidden states
    return outp[0] if isinstance(outp, tuple) else outp


def feature_act_from_resid(resid_lastpos: torch.Tensor, fid: int, W_enc_feat: torch.Tensor, b_enc: torch.Tensor, thr: torch.Tensor) -> torch.Tensor:
    """SAE activation for one feature at last position: z = x @ W_enc[fid] + b_enc[fid], a_f = relu(z - thr[fid])."""
    # resid_lastpos: (d_model,); W_enc_feat[fid]: (d_model,)
    z = (resid_lastpos @ W_enc_feat[fid]) + b_enc[fid]
    return torch.relu(z - thr[fid])


def main():
    cfg_path = "configs/targets/gemma2_2b_gemmascope_res16k.yaml"
    run_id = "debug_steer_effect"
    config = resolve_config(load_config(cfg_path), run_id=run_id)

    model, tokenizer = load_model(config)
    model.eval()
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    hook_name = config["sae"]["hook_point"]  # e.g. blocks.20.hook_resid_post
    hook_module = resolve_hook_point(model, hook_name)

    # Load decoder and NPZ for encoder + bias + threshold
    decoder, meta = load_gemmascope_decoder(config)
    npz_path = meta["npz_path"]
    data = np.load(npz_path)
    W_dec = np.asarray(data["W_dec"], dtype=np.float32)   # (n_features, d_model)
    W_enc = np.asarray(data["W_enc"], dtype=np.float32)
    b_enc = np.asarray(data["b_enc"], dtype=np.float32)   # (n_features,)
    thr = np.asarray(data["threshold"], dtype=np.float32)  # (n_features,)
    if W_enc.shape[0] != W_dec.shape[0] and W_enc.shape[1] == W_dec.shape[1]:
        W_enc_feat = W_enc  # (n_features, d_model)
    elif W_enc.shape[0] == W_dec.shape[1]:
        W_enc_feat = W_enc.T  # (d_model, n_features) -> (n_features, d_model)
    else:
        raise ValueError(f"Unexpected W_enc shape {W_enc.shape}, W_dec shape {W_dec.shape}")
    W_enc_feat = torch.tensor(W_enc_feat, device=device, dtype=model_dtype)
    b_enc = torch.tensor(b_enc, device=device, dtype=model_dtype)
    thr = torch.tensor(thr, device=device, dtype=model_dtype)
    decoder = torch.tensor(W_dec, device=device, dtype=model_dtype)

    prompt = "One plus one equals"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    fid = 14885
    target_text = " two"
    target_ids = tokenizer.encode(target_text, add_special_tokens=False)
    if len(target_ids) != 1:
        raise ValueError(f"Target {target_text!r} must be single token, got ids={target_ids}")
    tid = target_ids[0]

    ALPHAS = [0.0, 0.25, 0.5, 1.0, -1.0]
    captured_resid = []

    def make_hook(alpha_val):
        def hook_fn(module, inp, outp):
            t = _get_main_tensor(outp)  # [batch, seq, d_model]
            vec = decoder[fid].to(device=device, dtype=t.dtype)
            t2 = t.clone()
            t2[:, -1, :] = t2[:, -1, :] + alpha_val * vec
            captured_resid.append(t2.detach())
            if isinstance(outp, tuple):
                return (t2,) + outp[1:]
            return t2
        return hook_fn

    print("PROMPT:", repr(prompt))
    print("hook_point:", hook_name)
    print("feature_id:", fid)
    print("target token:", repr(target_text), "id:", tid)
    print()
    print("alpha     act(fid)    logit(' two')")
    print("-" * 40)

    results = []
    for alpha in ALPHAS:
        captured_resid.clear()
        handle = hook_module.register_forward_hook(make_hook(alpha))
        with torch.no_grad():
            out = model(**inputs)
            logits = out.logits[0, -1].float()
        handle.remove()
        if not captured_resid:
            raise RuntimeError("Hook did not capture residual")
        resid_last = captured_resid[0][0, -1, :]  # (d_model,)
        act = feature_act_from_resid(resid_last, fid, W_enc_feat, b_enc, thr)
        logit_two = logits[tid].item()
        results.append((alpha, act.item(), logit_two))
    base_logit_two = next(l for a, _, l in results if a == 0.0)
    for alpha, act_val, logit_two in results:
        delta = logit_two - base_logit_two
        print(f"alpha={alpha:+.2f}  act(fid)={act_val:.6f}  logit({target_text!r})={logit_two:+.4f}  Δlogit={delta:+.4f}")

    # ---- optional: show top tokens at alpha=0 and alpha=1 for comparison ----
    captured_resid.clear()
    handle = hook_module.register_forward_hook(make_hook(0.0))
    with torch.no_grad():
        out_base = model(**inputs)
        base_logits = out_base.logits[0, -1].float()
    handle.remove()
    captured_resid.clear()
    handle = hook_module.register_forward_hook(make_hook(1.0))
    with torch.no_grad():
        out_steer = model(**inputs)
        steered_logits = out_steer.logits[0, -1].float()
    handle.remove()

    def show_top(logits, title):
        top = torch.topk(logits, k=10)
        print(f"\n{title} top-10 next tokens:")
        for v, i in zip(top.values.tolist(), top.indices.tolist()):
            print(f"  logit={v:+.4f}  id={i:<7} text={tokenizer.decode([i])!r}")

    show_top(base_logits, "BASE (alpha=0)")
    show_top(steered_logits, "STEERED (alpha=+1.0)")


if __name__ == "__main__":
    main()
