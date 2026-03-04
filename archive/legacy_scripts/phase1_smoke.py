"""
Phase 1 Smoke Test: distributional off-target metrics for SAE feature steering.

For each (feature, alpha, prompt): steer via pre-hook on layer input hidden_states,
compute next-token metrics (max|Δlogit|, KL, breadth). Save CSV.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.config import load_config, resolve_config
from src.model_utils import load_model
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

EPS = 0.05
ALPHAS = [-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0]
N_FEATURES = 20
N_PROMPTS = 20
LAYER_IDX = 20


def next_token_metrics(logits_base: torch.Tensor, logits_steer: torch.Tensor, eps: float = EPS, topk: int = 50):
    """
    Compute distributional deltas between base and steered next-token logits ([vocab] each).
    Returns dict with all metrics.
    """
    d = logits_steer - logits_base
    d_abs = d.abs()

    # Logit-level stats
    max_abs = d_abs.max().item()
    mean_abs = d_abs.mean().item()
    p95_abs = torch.quantile(d_abs.float(), 0.95).item()
    l2 = d.float().norm().item()

    # Probability distributions
    p = F.softmax(logits_base, dim=-1)
    q = F.softmax(logits_steer, dim=-1)

    # KL(p||q)
    kl = (p * (torch.log(p + 1e-12) - torch.log(q + 1e-12))).sum().item()

    # Total Variation distance: 0.5 * sum|p - q|
    tv = 0.5 * (p - q).abs().sum().item()

    # Top-k overlap (Jaccard on top-50 tokens)
    topk_base = set(torch.topk(logits_base, k=topk).indices.tolist())
    topk_steer = set(torch.topk(logits_steer, k=topk).indices.tolist())
    topk_inter = len(topk_base & topk_steer)
    topk_jaccard = topk_inter / len(topk_base | topk_steer) if (topk_base | topk_steer) else 1.0

    return {
        "max_abs_dlogit": max_abs,
        "mean_abs_dlogit": mean_abs,
        "p95_abs_dlogit": p95_abs,
        "l2_dlogit": l2,
        "kl_base_to_steer": kl,
        "tv_distance": tv,
        "topk_overlap": topk_inter,
        "topk_jaccard": topk_jaccard,
    }


@torch.no_grad()
def forward_last_logits(model, tokenizer, prompt, prehook_fn=None):
    """Run a single forward pass. Returns (logits_last [vocab], input_ids [seq])."""
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    handle = None
    if prehook_fn is not None:
        handle = prehook_fn()
    out = model(**inputs)
    if handle is not None:
        handle.remove()
    logits_last = out.logits[0, -1, :].float().detach()
    return logits_last, inputs["input_ids"][0]


def pick_top_active_features(model, tokenizer, prompt, W_enc, b_enc, thr, topk=500):
    """Capture layer input hidden_states at last token, compute act for all features, return top feature ids."""
    capture = {"hid": None}

    def cap_prehook():
        def _prehook(module, inputs):
            hidden = inputs[0]
            capture["hid"] = hidden[0, -1].detach().clone().float()
            return inputs
        return model.model.layers[LAYER_IDX].register_forward_pre_hook(_prehook)

    forward_last_logits(model, tokenizer, prompt, prehook_fn=cap_prehook)
    r = capture["hid"]
    assert r is not None, "Failed to capture hidden_states"

    z = r @ W_enc + b_enc
    act = torch.relu(z - thr)
    vals, idx = torch.topk(act, k=min(topk, act.numel()))
    idx = idx[vals > 0]  # strictly active only
    return idx.tolist(), act


def make_steer_prehook(model, alpha, pos, steer_dir):
    """Returns (prehook_fn, capture_dict). prehook_fn() registers and returns a handle."""
    capture = {"ran": False}

    def _mk():
        def _prehook(module, inputs):
            hidden = inputs[0]
            capture["ran"] = True
            hidden2 = hidden.clone()
            hidden2[0, pos, :] = hidden2[0, pos, :] + alpha * steer_dir
            return (hidden2,) + inputs[1:]
        return model.model.layers[LAYER_IDX].register_forward_pre_hook(_prehook)

    return _mk, capture


def load_prompts(path, max_n=None):
    with open(path) as f:
        prompts = [line.strip() for line in f if line.strip()]
    if max_n is not None:
        prompts = prompts[:max_n]
    return prompts


def main():
    cfg_path = "configs/targets/gemma2_2b_gemmascope_res16k.yaml"
    run_id = "phase1_smoke"
    config = resolve_config(load_config(cfg_path), run_id=run_id)
    config["model"]["do_sample"] = False
    config["model"]["temperature"] = 0.0

    model, tokenizer = load_model(config)
    model.eval()
    device = next(model.parameters()).device

    # Load SAE params (same as debug_steer_effect.py)
    _, meta = load_gemmascope_decoder(config)
    npz_path = meta["npz_path"]
    data = np.load(npz_path)
    W_dec = torch.tensor(np.asarray(data["W_dec"], dtype=np.float32), device=device, dtype=torch.float32)
    W_enc = torch.tensor(np.asarray(data["W_enc"], dtype=np.float32), device=device, dtype=torch.float32)
    b_enc = torch.tensor(np.asarray(data["b_enc"], dtype=np.float32), device=device, dtype=torch.float32)
    thr   = torch.tensor(np.asarray(data["threshold"], dtype=np.float32), device=device, dtype=torch.float32)

    print(f"W_dec: {W_dec.shape}, W_enc: {W_enc.shape}, b_enc: {b_enc.shape}, thr: {thr.shape}")

    # Load prompts (mix of arithmetic + control for diversity)
    repo_root = Path(__file__).resolve().parents[1]
    arith = load_prompts(repo_root / "data" / "prompts" / "phase2_arithmetic.txt")
    ctrl = load_prompts(repo_root / "data" / "prompts" / "phase2_control.txt")
    prompts = (arith + ctrl)[:N_PROMPTS]
    print(f"Prompts: {len(prompts)}")

    # Pick candidate features from anchor prompt
    anchor_prompt = prompts[0]
    cand_features, _ = pick_top_active_features(
        model, tokenizer, anchor_prompt, W_enc, b_enc, thr, topk=500
    )
    if len(cand_features) < N_FEATURES:
        raise RuntimeError(f"Only {len(cand_features)} active features found. Need {N_FEATURES}.")
    feature_ids = cand_features[:N_FEATURES]
    print(f"Feature ids (top {N_FEATURES} active): {feature_ids}")

    rows = []
    total = len(prompts) * len(feature_ids) * len(ALPHAS)
    done = 0

    for pi, prompt in enumerate(prompts):
        base_logits, input_ids = forward_last_logits(model, tokenizer, prompt)
        seq_len = input_ids.shape[0]
        pos = seq_len - 1

        for fid in feature_ids:
            steer_dir = W_dec[fid].to(device=device, dtype=torch.float32)

            for alpha in ALPHAS:
                if alpha == 0.0:
                    m = {k: 0.0 for k in ["max_abs_dlogit", "mean_abs_dlogit", "p95_abs_dlogit",
                                           "l2_dlogit", "kl_base_to_steer", "tv_distance"]}
                    m["topk_overlap"] = 50
                    m["topk_jaccard"] = 1.0
                else:
                    mk_hook, cap = make_steer_prehook(model, alpha, pos, steer_dir)
                    steer_logits, _ = forward_last_logits(model, tokenizer, prompt, prehook_fn=mk_hook)
                    assert cap["ran"], f"Prehook never ran for fid={fid} alpha={alpha}"
                    m = next_token_metrics(base_logits, steer_logits)

                row = {
                    "prompt_idx": pi,
                    "feature_id": fid,
                    "alpha": alpha,
                    "seq_len": int(seq_len),
                }
                row.update({k: round(v, 6) if isinstance(v, float) else v for k, v in m.items()})
                rows.append(row)
                done += 1
                if done % 50 == 0:
                    print(f"  progress {done}/{total}")

        print(f"Done prompt {pi+1}/{len(prompts)}")

    df = pd.DataFrame(rows)
    out_dir = repo_root / "outputs" / "phase1_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "phase1_smoke.csv"
    df.to_csv(out_path, index=False)
    print(f"\nWrote {len(df)} rows to {out_path}")
    print(df.head(20))


if __name__ == "__main__":
    main()
