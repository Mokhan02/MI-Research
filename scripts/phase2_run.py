# scripts/phase2_run.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os, json, argparse, random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from src.config import load_config, resolve_config
from src.model_utils import load_model
from src.sae_loader import load_gemmascope_decoder

# --------------------------
# Utils
# --------------------------
def set_determinism(seed: int = 1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

def tv_distance_from_logits(logits_a: torch.Tensor, logits_b: torch.Tensor) -> float:
    p = F.softmax(logits_a, dim=-1)
    q = F.softmax(logits_b, dim=-1)
    return (0.5 * (p - q).abs().sum()).item()

def kl_pq_from_logits(logits_p: torch.Tensor, logits_q: torch.Tensor) -> float:
    p = F.softmax(logits_p, dim=-1)
    q = F.softmax(logits_q, dim=-1)
    return (p * (torch.log(p + 1e-12) - torch.log(q + 1e-12))).sum().item()

def topk_jaccard(logits_a: torch.Tensor, logits_b: torch.Tensor, k: int = 50) -> float:
    top_a = torch.topk(logits_a, k=k).indices
    top_b = torch.topk(logits_b, k=k).indices
    set_a = set(top_a.tolist())
    set_b = set(top_b.tolist())
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union else 1.0

@torch.no_grad()
def forward_last_logits(model, tokenizer, prompt: str, prehook=None):
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    handle = prehook() if prehook else None
    out = model(**inputs)
    if handle is not None:
        handle.remove()
    return out.logits[0, -1, :].float().detach(), inputs["input_ids"][0]

def make_steer_prehook(model, layer_idx: int, alpha: float, pos: int, steer_dir: torch.Tensor):
    ran = {"ok": False}
    def _mk():
        def _prehook(module, inputs):
            hidden = inputs[0]  # [1, seq, d_model]
            ran["ok"] = True
            hidden2 = hidden.clone()
            hidden2[0, pos, :] = hidden2[0, pos, :] + alpha * steer_dir
            return (hidden2,) + inputs[1:]
        layer = model.model.layers[layer_idx]
        return layer.register_forward_pre_hook(_prehook)
    return _mk, ran

def encode_target(tokenizer, target_text):
    if target_text is None or (isinstance(target_text, float) and np.isnan(target_text)):
        return None
    t = target_text
    if not t.startswith(" "):
        t = " " + t
    ids = tokenizer(t, add_special_tokens=False)["input_ids"]
    return ids[0] if len(ids) > 0 else None

# --------------------------
# Main
# --------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/targets/gemma2_2b_gemmascope_res16k.yaml")
    ap.add_argument("--prompt_csv", type=str, default="data/phase2_prompts.csv")
    ap.add_argument("--out_dir", type=str, default="outputs/phase2")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--n_prompts", type=int, default=200)
    ap.add_argument("--n_features", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--alphas", type=str, default="-5,-2,-1,0,1,2,5")
    args = ap.parse_args()

    set_determinism(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # Load prompts
    dfp = pd.read_csv(args.prompt_csv)
    assert "prompt" in dfp.columns, "prompt_csv must have a 'prompt' column"
    if "target" not in dfp.columns:
        dfp["target"] = np.nan
    dfp = dfp.head(args.n_prompts).reset_index(drop=True)

    # Load model via config
    config = resolve_config(load_config(args.config), run_id="phase2_run")
    config["model"]["do_sample"] = False
    config["model"]["temperature"] = 0.0
    model, tokenizer = load_model(config)
    model.eval()
    device = next(model.parameters()).device

    # Load SAE params (same loader as phase1_smoke / debug_steer_effect)
    _, meta = load_gemmascope_decoder(config)
    npz_path = meta["npz_path"]
    data = np.load(npz_path)
    W_dec = torch.tensor(np.asarray(data["W_dec"], dtype=np.float32), device=device, dtype=torch.float32)
    print(f"W_dec: {W_dec.shape}")

    # Choose features
    n_feats_total = W_dec.shape[0]
    rng = np.random.default_rng(args.seed)
    feature_ids = rng.choice(n_feats_total, size=min(args.n_features, n_feats_total), replace=False).tolist()

    alphas = [float(x) for x in args.alphas.split(",")]
    alphas_sorted = sorted(alphas, key=lambda a: abs(a))

    rows = []
    alpha_star_rows = []

    for pi, prow in dfp.iterrows():
        prompt = prow["prompt"]
        target_id = encode_target(tokenizer, prow.get("target", np.nan))

        base_logits, input_ids = forward_last_logits(model, tokenizer, prompt)
        seq_len = int(input_ids.shape[0])
        pos = seq_len - 1

        if target_id is None:
            target_id = int(torch.argmax(base_logits).item())

        for fid in feature_ids:
            steer_dir = W_dec[fid].to(device=device, dtype=torch.float32)
            alpha_star = None

            for alpha in alphas_sorted:
                if alpha == 0.0:
                    steer_logits = base_logits
                    hook_ran = True
                else:
                    mk_hook, cap = make_steer_prehook(model, args.layer, alpha, pos, steer_dir)
                    steer_logits, _ = forward_last_logits(model, tokenizer, prompt, prehook=mk_hook)
                    hook_ran = cap["ok"]

                d = steer_logits - base_logits
                delta_target = float(d[target_id].item())
                max_abs = float(d.abs().max().item())
                tv = tv_distance_from_logits(base_logits, steer_logits)
                kl = kl_pq_from_logits(base_logits, steer_logits)
                jac = topk_jaccard(base_logits, steer_logits, k=args.topk)

                rows.append({
                    "prompt_idx": pi,
                    "feature_id": fid,
                    "alpha": alpha,
                    "seq_len": seq_len,
                    "target_id": int(target_id),
                    "delta_logit_target": delta_target,
                    "tv_distance": tv,
                    "kl_base_to_steer": kl,
                    "topk_jaccard": jac,
                    "max_abs_dlogit": max_abs,
                    "hook_ran": bool(hook_ran),
                })

                if alpha_star is None and alpha != 0.0 and delta_target >= args.tau:
                    alpha_star = abs(alpha)

            alpha_star_rows.append({
                "prompt_idx": pi,
                "feature_id": fid,
                "alpha_star": alpha_star if alpha_star is not None else np.nan,
            })

        if (pi + 1) % 10 == 0:
            print(f"done prompts {pi+1}/{len(dfp)}")

    df = pd.DataFrame(rows)
    df_alpha = pd.DataFrame(alpha_star_rows)

    # Write raw rows
    run_path = os.path.join(args.out_dir, "run_rows.csv")
    df.to_csv(run_path, index=False)

    # Aggregate per feature
    feat_alpha = (df_alpha.groupby("feature_id")
                  .agg(alpha_star_mean=("alpha_star", "mean"),
                       alpha_star_median=("alpha_star", "median"),
                       success_rate=("alpha_star", lambda s: float(np.isfinite(s).mean())))
                  .reset_index())

    alpha_ref = max(alphas, key=lambda a: abs(a))
    df_ref = df[df.alpha == alpha_ref].copy()
    feat_ref = (df_ref.groupby("feature_id")
                .agg(tv_mean=("tv_distance", "mean"),
                     kl_mean=("kl_base_to_steer", "mean"),
                     maxabs_mean=("max_abs_dlogit", "mean"),
                     jacc_mean=("topk_jaccard", "mean"),
                     target_delta_mean=("delta_logit_target", "mean"))
                .reset_index())

    feat_summary = feat_alpha.merge(feat_ref, on="feature_id", how="left")
    summary_path = os.path.join(args.out_dir, "feature_summary.csv")
    feat_summary.to_csv(summary_path, index=False)

    # Meta
    meta_out = vars(args)
    meta_out["n_rows"] = int(len(df))
    meta_out["alpha_ref"] = alpha_ref
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta_out, f, indent=2)

    print(f"\nWrote:\n- {run_path}\n- {summary_path}\n- {os.path.join(args.out_dir, 'meta.json')}")
    print(feat_summary.sort_values(["alpha_star_mean", "tv_mean"], ascending=[True, True]).head(15))


if __name__ == "__main__":
    main()
