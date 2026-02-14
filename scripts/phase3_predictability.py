# scripts/phase3_predictability.py
"""
Predict phase2 steerability (alpha_star_best, tv_at_alpha_star_best) from
pre-steering features: W_dec geometry + baseline activation usage.

Step 0: Y = alpha_star_best, tv_at_alpha_star_best, success_best, monotone_best
        (best direction = higher success, tie-break lower tv_at_alpha_star)
Step 1: X = geometry (max_cosine_to_any, density_tau, mean_topk_cos) + usage (act_freq, mean_act, mean_z_minus_thr)
Step 2: Merge and write outputs/phase3_predictability/<run>_merged.csv
Step 3: Spearman correlations, OLS, logistic(is_steerable), stratify by act_freq

Usage:
  python scripts/phase3_predictability.py --phase2_dir outputs/phase2_arith_run1_tau01 --config configs/targets/gemma2_2b_gemmascope_res16k.yaml
  python scripts/phase3_predictability.py --phase2_dir ... --skip_baseline   # geometry only
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from src.config import load_config, resolve_config
from src.model_utils import load_model
from src.sae_loader import load_gemmascope_decoder


# --------------------------
# Step 0: Labels from feature_summary
# --------------------------
def make_y_labels(summary: pd.DataFrame) -> pd.DataFrame:
    """One row per feature: alpha_star_best, tv_at_alpha_star_best, success_best, monotone_best."""
    rows = []
    for _, row in summary.iterrows():
        su = float(row.get("success_rate_up", 0) or 0)
        sd = float(row.get("success_rate_down", 0) or 0)
        tv_up = row.get("tv_at_alpha_star_up", np.nan)
        tv_dn = row.get("tv_at_alpha_star_down", np.nan)
        # Best direction: higher success first; tie-break lower tv_at_alpha_star
        if su > sd:
            direction = "up"
        elif sd > su:
            direction = "down"
        else:
            direction = "up" if (pd.isna(tv_dn) or (not pd.isna(tv_up) and float(tv_up) <= float(tv_dn))) else "down"

        if direction == "up":
            alpha_star_best = row.get("alpha_star_mean_up", np.nan)
            tv_at_alpha_star_best = row.get("tv_at_alpha_star_up", np.nan)
            success_best = row.get("success_rate_up", np.nan)
            monotone_best = row.get("monotone_frac_up", np.nan)
        else:
            alpha_star_best = row.get("alpha_star_mean_down", np.nan)
            tv_at_alpha_star_best = row.get("tv_at_alpha_star_down", np.nan)
            success_best = row.get("success_rate_down", np.nan)
            monotone_best = row.get("monotone_frac_down", np.nan)

        rows.append({
            "feature_id": row["feature_id"],
            "alpha_star_best": alpha_star_best,
            "tv_at_alpha_star_best": tv_at_alpha_star_best,
            "success_best": success_best,
            "monotone_best": monotone_best,
        })
    return pd.DataFrame(rows)


# --------------------------
# Step 1a: Geometry from W_dec (chunked)
# --------------------------
def compute_geometry_chunked(W_dec: np.ndarray, tau: float, topk: int = 50, chunk_size: int = 256) -> dict:
    """W_dec: (n_feats, d_model). Returns max_cosine_to_any, density_tau, mean_topk_cos per feature."""
    n_feats, d_model = W_dec.shape
    Wn = W_dec / (np.linalg.norm(W_dec, axis=1, keepdims=True) + 1e-12)

    max_cosine = np.full(n_feats, -np.inf, dtype=np.float32)
    count_ge_tau = np.zeros(n_feats, dtype=np.float32)
    sum_topk = np.zeros(n_feats, dtype=np.float32)
    count_topk = np.zeros(n_feats, dtype=np.float32)

    for i0 in tqdm(range(0, n_feats, chunk_size), desc="Geometry chunks"):
        i1 = min(i0 + chunk_size, n_feats)
        block = Wn[i0:i1]
        S = block @ Wn.T
        # Self-similarity to -inf so we don't pick self
        for j in range(i1 - i0):
            S[j, i0 + j] = -np.inf

        for j in range(i1 - i0):
            row = S[j]
            max_cosine[i0 + j] = float(np.max(row))
            count_ge_tau[i0 + j] = float(np.sum(row >= tau))
            top_vals = np.sort(row)[::-1][:topk]
            valid = top_vals > -np.inf
            if np.any(valid):
                sum_topk[i0 + j] = np.sum(top_vals[valid])
                count_topk[i0 + j] = np.sum(valid)

    density_tau = count_ge_tau / max(1, n_feats - 1)
    mean_topk_cos = np.where(count_topk > 0, sum_topk / count_topk, np.nan)

    return {
        "max_cosine_to_any": max_cosine,
        "density_tau": density_tau,
        "mean_topk_cos": mean_topk_cos,
    }


# --------------------------
# Step 1b: Baseline activation usage (recompute or load cache)
# --------------------------
@torch.no_grad()
def compute_baseline_usage(
    model, tokenizer, W_enc: torch.Tensor, b_enc: torch.Tensor, thr: torch.Tensor,
    prompts: list, device,
) -> np.ndarray:
    """Returns (n_feats,) arrays: act_freq, mean_act, mean_z_minus_thr. W_enc: (d_model, n_feats) or (n_feats, d_model)."""
    n_feats = b_enc.shape[0]
    d_model = W_enc.shape[0] if W_enc.shape[0] != n_feats else W_enc.shape[1]

    resids = []
    for prompt in tqdm(prompts, desc="Baseline forwards"):
        inp = tokenizer(prompt, return_tensors="pt").to(device)
        out = model(**inp, output_hidden_states=True)
        r = out.hidden_states[-1][0, -1, :].float().cpu().numpy()
        resids.append(r)
    resids = np.stack(resids, axis=0)

    if W_enc.shape[0] == d_model and W_enc.shape[1] == n_feats:
        z = resids @ W_enc.cpu().numpy() + b_enc.cpu().numpy()
    else:
        z = resids @ W_enc.cpu().numpy().T + b_enc.cpu().numpy()
    thr_np = thr.cpu().numpy()
    act = np.maximum(z - thr_np, 0.0)

    act_freq = (act > 0).mean(axis=0)
    mean_act = np.zeros(n_feats)
    mean_z_minus_thr = np.zeros(n_feats)
    for i in range(n_feats):
        active = act[:, i] > 0
        if np.any(active):
            mean_act[i] = act[active, i].mean()
            mean_z_minus_thr[i] = (z[active, i] - thr_np[i]).mean()
        else:
            mean_act[i] = np.nan
            mean_z_minus_thr[i] = np.nan

    return act_freq, mean_act, mean_z_minus_thr


# --------------------------
# Step 3: Stats
# --------------------------
def run_correlations(df: pd.DataFrame, y_col: str, x_cols: list):
    from scipy.stats import spearmanr
    print(f"\n--- Spearman correlations vs {y_col} ---")
    for x in x_cols:
        if x not in df.columns or y_col not in df.columns:
            continue
        valid = df[[x, y_col]].dropna()
        if len(valid) < 10:
            continue
        r, p = spearmanr(valid[x], valid[y_col])
        print(f"  {x}: r={r:.3f}  p={p:.4f}  n={len(valid)}")


def run_ols(df: pd.DataFrame, y_col: str, x_cols: list):
    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("\n--- OLS skipped (sklearn not installed) ---")
        return
    valid = df[x_cols + [y_col]].dropna()
    if len(valid) < 20:
        print("\n--- OLS skipped (too few rows after dropna) ---")
        return
    X = valid[x_cols].values
    y = valid[y_col].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    reg = LinearRegression().fit(Xs, y)
    print(f"\n--- OLS {y_col} ~ {' + '.join(x_cols)} ---")
    print(f"  R2 = {reg.score(Xs, y):.4f}")
    for name, coef in zip(x_cols, reg.coef_):
        print(f"  {name}: coef = {coef:.4f}")


def run_logistic(df: pd.DataFrame, x_cols: list, threshold: float = 0.5):
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        return
    df = df.copy()
    df["is_steerable"] = (df["success_best"] >= threshold).astype(float)
    valid = df[x_cols + ["is_steerable"]].dropna()
    if len(valid) < 20 or valid["is_steerable"].nunique() < 2:
        print("\n--- Logistic skipped (too few rows or no variance in is_steerable) ---")
        return
    X = valid[x_cols].values
    y = valid["is_steerable"].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=500).fit(Xs, y)
    print(f"\n--- Logistic is_steerable (success_best >= {threshold}) ~ {' + '.join(x_cols)} ---")
    print(f"  accuracy = {(clf.predict(Xs) == y).mean():.4f}")
    for name, coef in zip(x_cols, clf.coef_[0]):
        print(f"  {name}: coef = {coef:.4f}")


def run_stratified_correlations(df: pd.DataFrame, y_col: str, x_col: str, q: int = 3):
    try:
        from scipy.stats import spearmanr
    except ImportError:
        return
    if "act_freq" not in df.columns or x_col not in df.columns or y_col not in df.columns:
        return
    df = df.dropna(subset=["act_freq", x_col, y_col])
    if len(df) < 30:
        return
    df = df.copy()
    df["act_freq_q"] = pd.qcut(df["act_freq"], q=q, labels=False, duplicates="drop")
    print(f"\n--- Spearman({x_col}, {y_col}) by act_freq quantile ---")
    for qid in sorted(df["act_freq_q"].unique()):
        sub = df[df["act_freq_q"] == qid]
        if len(sub) < 10:
            continue
        r, p = spearmanr(sub[x_col], sub[y_col])
        print(f"  q{qid} (n={len(sub)}): r={r:.3f}  p={p:.4f}")


# --------------------------
# Main
# --------------------------
def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--phase2_dir", type=str, help="Phase2 output dir (contains feature_summary.csv)")
    g.add_argument("--phase2_summary", type=str, help="Path to feature_summary.csv directly")
    ap.add_argument("--config", type=str, default="configs/targets/gemma2_2b_gemmascope_res16k.yaml")
    ap.add_argument("--out_dir", type=str, default="outputs/phase3_predictability")
    ap.add_argument("--tau", type=float, default=0.1, help="Cosine threshold for density_tau")
    ap.add_argument("--chunk_size", type=int, default=256)
    ap.add_argument("--topk", type=int, default=50, help="Top-k for mean_topk_cos")
    ap.add_argument("--n_prompts", type=int, default=100, help="Max prompts for baseline activation computation")
    ap.add_argument("--skip_baseline", action="store_true", help="Skip baseline activation computation (geometry only)")
    ap.add_argument("--prompt_csv", type=str, default=None, help="Prompts for baseline; default arithmetic_only.csv")
    ap.add_argument("--baseline_cache", type=str, default=None, help="Path to load/save baseline_usage.csv")
    ap.add_argument("--run_name", type=str, default="arithmetic", help="Suffix for merged CSV and cache")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # Load phase2 feature_summary
    if args.phase2_summary:
        summary_path = Path(args.phase2_summary)
        phase2_dir = summary_path.parent
    else:
        phase2_dir = Path(args.phase2_dir)
        summary_path = phase2_dir / "feature_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Need {summary_path}")
    summary = pd.read_csv(summary_path)
    feature_ids = summary["feature_id"].astype(int).tolist()
    n_feats_summary = len(feature_ids)
    print(f"Loaded feature_summary: {n_feats_summary} features")

    # Y labels
    y_df = make_y_labels(summary)
    print("Y labels: alpha_star_best, tv_at_alpha_star_best, success_best, monotone_best")

    # Load SAE
    config = resolve_config(load_config(args.config), run_id="phase3")
    _, meta = load_gemmascope_decoder(config)
    npz_path = meta["npz_path"]
    data = np.load(npz_path)
    W_dec = np.asarray(data["W_dec"], dtype=np.float32)
    n_feats_total, d_model = W_dec.shape
    print(f"W_dec: {n_feats_total} x {d_model}")

    # Geometry (all features)
    geo = compute_geometry_chunked(W_dec, tau=args.tau, topk=args.topk, chunk_size=args.chunk_size)
    geo_df = pd.DataFrame({
        "feature_id": np.arange(n_feats_total),
        "max_cosine_to_any": geo["max_cosine_to_any"],
        "density_tau": geo["density_tau"],
        "mean_topk_cos": geo["mean_topk_cos"],
    })

    # Baseline usage: load cache or recompute
    if args.skip_baseline:
        usage_df = pd.DataFrame({"feature_id": np.arange(n_feats_total), "act_freq": np.nan, "mean_act": np.nan, "mean_z_minus_thr": np.nan})
        print("Skipping baseline (--skip_baseline)")
    else:
        cache_path = Path(args.baseline_cache) if args.baseline_cache else Path(args.out_dir) / f"baseline_usage_{args.run_name}.csv"
        if cache_path.exists():
            usage_df = pd.read_csv(cache_path)
            print(f"Loaded baseline usage from {cache_path}")
        else:
            model, tokenizer = load_model(config)
            model.eval()
            device = next(model.parameters()).device
            W_enc = torch.tensor(np.asarray(data["W_enc"], dtype=np.float32), device=device)
            b_enc = torch.tensor(np.asarray(data["b_enc"], dtype=np.float32), device=device)
            thr = torch.tensor(np.asarray(data["threshold"], dtype=np.float32), device=device)
            if W_enc.shape[0] != d_model:
                W_enc = W_enc.T
            prompt_csv = args.prompt_csv or "data/arithmetic_only.csv"
            if not Path(prompt_csv).exists():
                prompt_csv = str(Path(__file__).resolve().parents[1] / "data" / "arithmetic_only.csv")
            dfp = pd.read_csv(prompt_csv, dtype={"prompt": "string"})
            prompts = dfp["prompt"].dropna().astype(str).tolist()[: args.n_prompts]
            act_freq, mean_act, mean_z_minus_thr = compute_baseline_usage(
                model, tokenizer, W_enc, b_enc, thr, prompts, device
            )
            usage_df = pd.DataFrame({
                "feature_id": np.arange(n_feats_total),
                "act_freq": act_freq,
                "mean_act": mean_act,
                "mean_z_minus_thr": mean_z_minus_thr,
            })
            usage_df.to_csv(cache_path, index=False)
            print(f"Saved baseline usage -> {cache_path}")

    # Merge: only features in summary
    merge = summary[["feature_id"]].copy()
    merge = merge.merge(geo_df, on="feature_id", how="left")
    merge = merge.merge(usage_df, on="feature_id", how="left")
    merge = merge.merge(y_df, on="feature_id", how="left")
    out_path = Path(args.out_dir) / f"{args.run_name}_features_merged.csv"
    merge.to_csv(out_path, index=False)
    print(f"Wrote {out_path}  ({len(merge)} rows)")

    # Step 3: Stats
    x_cols = ["max_cosine_to_any", "density_tau", "act_freq", "mean_act"]
    x_cols = [c for c in x_cols if c in merge.columns]

    run_correlations(merge, "alpha_star_best", x_cols)
    run_ols(merge, "alpha_star_best", x_cols)
    run_logistic(merge, x_cols, threshold=0.5)
    run_stratified_correlations(merge, "alpha_star_best", "max_cosine_to_any", q=3)

    print("\nTop 10 by alpha_star_best (lower = better):")
    print(merge.sort_values("alpha_star_best").head(10)[["feature_id", "alpha_star_best", "tv_at_alpha_star_best", "max_cosine_to_any", "act_freq"]].to_string(index=False))


if __name__ == "__main__":
    main()
