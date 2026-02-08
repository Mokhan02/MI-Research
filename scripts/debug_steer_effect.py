import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from src.config import load_config, resolve_config
from src.model_utils import load_model
from src.hook_resolver import resolve_hook_point
from src.sae_loader import load_gemmascope_decoder


def _get_main_tensor(outp):
    # HF decoder layer often returns tuple; first element is hidden states
    return outp[0] if isinstance(outp, tuple) else outp


def main():
    cfg_path = "configs/targets/gemma2_2b_gemmascope_res16k.yaml"
    run_id = "debug_steer_effect"
    config = resolve_config(load_config(cfg_path), run_id=run_id)

    model, tokenizer = load_model(config)
    model.eval()
    device = next(model.parameters()).device

    hook_name = config["sae"]["hook_point"]  # e.g. blocks.20.hook_resid_post
    hook_module = resolve_hook_point(model, hook_name)

    decoder, _meta = load_gemmascope_decoder(config)
    decoder = decoder.to(device)

    # ---- single prompt ----
    prompt = "One plus one equals"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # ---- choose feature/alpha to test ----
    fid = 14885
    alpha = 5.0

    # ---- baseline logits for next token ----
    with torch.no_grad():
        out = model(**inputs)
        base_logits = out.logits[0, -1].float()

    # ---- steered logits for next token ----
    def hook_fn(module, inp, outp):
        t = _get_main_tensor(outp)  # [batch, seq, d_model]
        vec = decoder[fid].to(device=device, dtype=t.dtype)
        t2 = t.clone()
        t2[:, -1, :] = t2[:, -1, :] + alpha * vec
        if isinstance(outp, tuple):
            return (t2,) + outp[1:]
        return t2

    handle = hook_module.register_forward_hook(hook_fn)
    with torch.no_grad():
        out2 = model(**inputs)
        steered_logits = out2.logits[0, -1].float()
    handle.remove()

    # ---- compare: overall logit movement ----
    diff = (steered_logits - base_logits).abs()
    print("PROMPT:", repr(prompt))
    print("hook_point:", hook_name)
    print("feature_id:", fid, "alpha:", alpha)
    print("Mean |Δlogit|:", diff.mean().item())
    print("Max  |Δlogit|:", diff.max().item())

    # ---- compare: does it boost the CORRECT answer token? ----
    # Try a few candidate strings because tokenization can differ ("2" vs " 2")
    cands = ["2", " two", " three", " one", " four", " five"]
    pb = F.softmax(base_logits, dim=-1)
    ps = F.softmax(steered_logits, dim=-1)

    print("\nCorrect-token probability probe:")
    for s in cands:
        ids = tokenizer.encode(s, add_special_tokens=False)
        if len(ids) == 1:
            tid = ids[0]
            p0 = pb[tid].item()
            p1 = ps[tid].item()
            ratio = p1 / (p0 + 1e-12)
            print(f"  cand={s!r:6} token={tid:<7} P_base={p0:.6e}  P_steer={p1:.6e}  ratio={ratio:.2f}x")
        else:
            print(f"  cand={s!r:6} -> multi-token ids={ids} (skip)")

    # ---- show top tokens baseline vs steered ----
    def show_top(logits, title):
        top = torch.topk(logits, k=10)
        print(f"\n{title} top-10 next tokens:")
        for v, i in zip(top.values.tolist(), top.indices.tolist()):
            print(f"  logit={v:+.4f}  id={i:<7} text={tokenizer.decode([i])!r}")

    show_top(base_logits, "BASE")
    show_top(steered_logits, "STEERED")


if __name__ == "__main__":
    main()
