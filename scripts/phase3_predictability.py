# scripts/phase3_predictability.py
"""
Phase 3: Predict steerability (alpha*) from pre-steering geometric features.

PRIMARY ANALYSIS (main paper):
  Continuous Spearman rank correlation between geometric metrics
  (max_cosine_similarity, neighbor_density, coactivation_correlation) and
  log(alpha*).  Censored features (alpha* never hit threshold at alpha=10) are
  assigned alpha*=10 (tied maximum rank) and flagged in all plots.
  Sensitivity analysis reports correlations with AND without censored features.
  Bootstrap (Fisher-z) 95% CIs are computed for every Spearman r.

SECONDARY ANALYSIS (supplementary):
  Binary logistic-regression classifier (steerable vs not) as a practical
  utility check.  NOT the primary result.

Inputs
------
  --input_csv   Path to a pre-merged CSV that already contains feature_id,
                alpha_star_best (or alpha_star_feature_up/down), censored_up/
                censored_down, and geometry columns.  When supplied the script
                skips all geometry computation.
  OR
  --phase2_dir / --phase2_summary   Phase-2 output dir (feature_summary.csv)
                plus --config for SAE geometry recomputation.

Outputs
-------
  plots/scatter_max_cosine_similarity.png
  plots/scatter_neighbor_density.png
  plots/scatter_coactivation_correlation.png
  plots/correlation_summary.png   (1×3 panel)
  outputs/correlation_results.csv
  outputs/summary_stats.csv
  outputs/classification_supplementary.csv

Usage
-----
  # From pre-merged CSV (analysis only, no model/SAE needed):
  python scripts/phase3_predictability.py \\
      --input_csv outputs/phase3_predictability/salad_features_merged.csv \\
      --output_dir outputs/phase3_analysis

  # From phase-2 outputs (computes geometry from SAE decoder):
  python scripts/phase3_predictability.py \\
      --phase2_dir outputs/phase2_salad_run1 \\
      --config configs/targets/gemma2_2b_gemmascope_res16k.yaml \\
      --output_dir outputs/phase3_analysis
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import logging
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm.auto import tqdm
import wandb
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GEOMETRY_METRICS = [
    "max_cosine_similarity",
    "neighbor_density",
    "coactivation_correlation",
]
ALPHA_STAR_MAX = 10.0
RNG_SEED = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ===================================================================
# Data Preparation  (preserved from original, extended)
# ===================================================================

def make_y_labels(summary: pd.DataFrame) -> pd.DataFrame:
    """One row per feature: alpha_star_best, tv_at_alpha_star_best,
    success_best, monotone_best, is_censored."""
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

        a_up = pd.to_numeric(row.get("alpha_star_feature_up", row.get("alpha_star_mean_up", np.nan)), errors="coerce")
        a_dn = pd.to_numeric(row.get("alpha_star_feature_down", row.get("alpha_star_mean_down", np.nan)), errors="coerce")
        if direction == "up":
            alpha_star_best = a_up if pd.notna(a_up) else a_dn
            tv_at_alpha_star_best = row.get("tv_at_alpha_star_up", np.nan)
            success_best = row.get("success_rate_up", np.nan)
            monotone_best = row.get("monotone_frac_up", np.nan)
            censored_best = bool(row.get("censored_up", False))
        else:
            alpha_star_best = a_dn if pd.notna(a_dn) else a_up
            tv_at_alpha_star_best = row.get("tv_at_alpha_star_down", np.nan)
            success_best = row.get("success_rate_down", np.nan)
            monotone_best = row.get("monotone_frac_down", np.nan)
            censored_best = bool(row.get("censored_down", False))

        fid = int(row["feature_id"])
        rows.append({
            "feature_id": fid,
            "alpha_star_best": alpha_star_best,
            "tv_at_alpha_star_best": tv_at_alpha_star_best,
            "success_best": success_best,
            "monotone_best": monotone_best,
            "is_censored": censored_best,
        })
    return pd.DataFrame(rows)


# ===================================================================
# Geometry Computation  (preserved from original)
# ===================================================================

def compute_geometry_chunked(W_dec: np.ndarray, tau: float, topk: int = 50,
                             chunk_size: int = 256, device: str = "auto") -> dict:
    """W_dec: (n_feats, d_model).
    Returns max_cosine_similarity, neighbor_density, density_tau per feature.

    Uses PyTorch on GPU when available for a large speedup (~100x vs NumPy
    for a 16k-feature SAE).  Falls back to NumPy transparently on CPU-only
    machines.
    """
    # ---- resolve device ----
    if device == "auto":
        _dev = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        _dev = device

    n_feats = W_dec.shape[0]

    if _dev == "cuda":
        # ---- GPU path (PyTorch) ----
        logger.info("compute_geometry_chunked: using GPU (%s), n_feats=%d", _dev, n_feats)
        Wt = torch.from_numpy(W_dec).float().to(_dev)  # (n_feats, d_model)
        norms = Wt.norm(dim=1, keepdim=True).clamp(min=1e-12)
        Wn = Wt / norms  # (n_feats, d_model) — unit vectors

        max_cosine = torch.full((n_feats,), -torch.inf, device=_dev)
        count_ge_tau = torch.zeros(n_feats, device=_dev)
        sum_topk = torch.zeros(n_feats, device=_dev)
        count_topk = torch.zeros(n_feats, device=_dev)

        for i0 in tqdm(range(0, n_feats, chunk_size), desc="Geometry chunks (GPU)"):
            i1 = min(i0 + chunk_size, n_feats)
            block = Wn[i0:i1]            # (chunk, d_model)
            S = block @ Wn.T             # (chunk, n_feats)
            # Mask self-similarities
            for j in range(i1 - i0):
                S[j, i0 + j] = -torch.inf

            chunk_max, _ = S.max(dim=1)
            max_cosine[i0:i1] = chunk_max

            count_ge_tau[i0:i1] = (S >= tau).float().sum(dim=1)

            # top-k mean (excluding -inf self entries are already -inf ranked last)
            k = min(topk, n_feats - 1)
            topk_vals, _ = torch.topk(S, k=k, dim=1)  # (chunk, k)
            valid_mask = topk_vals > -torch.inf        # (chunk, k)
            valid_sums = (topk_vals * valid_mask.float()).sum(dim=1)
            valid_counts = valid_mask.float().sum(dim=1)
            sum_topk[i0:i1] = valid_sums
            count_topk[i0:i1] = valid_counts

        density_tau = (count_ge_tau / max(1, n_feats - 1)).cpu().numpy().astype(np.float32)
        mean_topk_cos = torch.where(
            count_topk > 0, sum_topk / count_topk, torch.full_like(sum_topk, float("nan"))
        ).cpu().numpy().astype(np.float32)
        max_cosine_np = max_cosine.cpu().numpy().astype(np.float32)

    else:
        # ---- CPU fallback (NumPy) — original implementation ----
        logger.info("compute_geometry_chunked: using CPU/NumPy, n_feats=%d", n_feats)
        Wn = W_dec / (np.linalg.norm(W_dec, axis=1, keepdims=True) + 1e-12)

        max_cosine_np = np.full(n_feats, -np.inf, dtype=np.float32)
        count_ge_tau = np.zeros(n_feats, dtype=np.float32)
        sum_topk = np.zeros(n_feats, dtype=np.float32)
        count_topk = np.zeros(n_feats, dtype=np.float32)

        for i0 in tqdm(range(0, n_feats, chunk_size), desc="Geometry chunks"):
            i1 = min(i0 + chunk_size, n_feats)
            block = Wn[i0:i1]
            S = block @ Wn.T
            for j in range(i1 - i0):
                S[j, i0 + j] = -np.inf
            for j in range(i1 - i0):
                row = S[j]
                max_cosine_np[i0 + j] = float(np.max(row))
                count_ge_tau[i0 + j] = float(np.sum(row >= tau))
                top_vals = np.sort(row)[::-1][:topk]
                valid = top_vals > -np.inf
                if np.any(valid):
                    sum_topk[i0 + j] = np.sum(top_vals[valid])
                    count_topk[i0 + j] = np.sum(valid)

        density_tau = count_ge_tau / max(1, n_feats - 1)
        mean_topk_cos = np.where(count_topk > 0, sum_topk / count_topk, np.nan)

    return {
        "max_cosine_similarity": max_cosine_np,
        "neighbor_density": mean_topk_cos,
        "density_tau": density_tau,
    }


def compute_coactivation_chunked(activations: np.ndarray,
                                  feature_ids: np.ndarray,
                                  topk: int = 50) -> np.ndarray:
    """Compute mean absolute Pearson correlation with top-k co-active features.

    Parameters
    ----------
    activations : (n_samples, n_feats_total)
    feature_ids : 1-d array of feature indices to compute for
    topk : number of top correlated features to average over

    Returns
    -------
    coact : (len(feature_ids),) array of mean |Pearson r| with top-k features
    """
    n_samples, n_feats = activations.shape
    if n_samples < 5:
        logger.warning("Too few samples (%d) for coactivation; returning NaN", n_samples)
        return np.full(len(feature_ids), np.nan)

    # Center activations once
    act_centered = activations - activations.mean(axis=0, keepdims=True)
    act_std = act_centered.std(axis=0, keepdims=True) + 1e-12
    act_normed = act_centered / act_std  # (n_samples, n_feats)

    coact = np.full(len(feature_ids), np.nan, dtype=np.float32)
    for i, fid in enumerate(tqdm(feature_ids, desc="Coactivation")):
        if fid < 0 or fid >= n_feats:
            continue
        v = act_normed[:, fid]  # (n_samples,)
        # Pearson r = mean of element-wise product of z-scored vectors
        corrs = (v[:, None] * act_normed).mean(axis=0)  # (n_feats,)
        corrs[fid] = 0.0  # exclude self
        abs_corrs = np.abs(corrs)
        k = min(topk, n_feats - 1)
        top_idx = np.argpartition(abs_corrs, -k)[-k:]
        coact[i] = abs_corrs[top_idx].mean()
    return coact


# ===================================================================
# Baseline Activation Usage  (preserved from original)
# ===================================================================

import torch

@torch.no_grad()
def compute_baseline_usage(
    model, tokenizer, W_enc: torch.Tensor, b_enc: torch.Tensor, thr: torch.Tensor,
    prompts: list, device,
) -> tuple:
    """Returns (act_freq, mean_act, mean_z_minus_thr, raw_activations).
    W_enc: (d_model, n_feats) or (n_feats, d_model)."""
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

    return act_freq, mean_act, mean_z_minus_thr, act


# ===================================================================
# PRIMARY ANALYSIS: Continuous Spearman Correlation
# ===================================================================

def bootstrap_spearman_ci(x: np.ndarray, y: np.ndarray,
                          n_bootstrap: int = 1000, seed: int = RNG_SEED,
                          alpha: float = 0.05) -> tuple:
    """Bootstrap 95% CI for Spearman r using Fisher z-transformation.

    Returns (r, p, ci_lower, ci_upper).
    """
    r, p = spearmanr(x, y)
    rng = np.random.RandomState(seed)
    n = len(x)
    z_samples = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        r_boot, _ = spearmanr(x[idx], y[idx])
        if np.isnan(r_boot):
            continue
        z_boot = np.arctanh(np.clip(r_boot, -0.9999, 0.9999))
        z_samples.append(z_boot)
    if len(z_samples) < 10:
        return r, p, np.nan, np.nan
    z_samples = np.array(z_samples)
    lo = np.percentile(z_samples, 100 * alpha / 2)
    hi = np.percentile(z_samples, 100 * (1 - alpha / 2))
    return r, p, float(np.tanh(lo)), float(np.tanh(hi))


def make_scatter_plot(df: pd.DataFrame, metric: str, output_dir: Path):
    """Publication-quality scatter: metric vs log(alpha*), censored flagged."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if metric not in df.columns:
        logger.warning("Column '%s' not found -- skipping scatter plot", metric)
        return

    sub = df.dropna(subset=[metric, "log_alpha_star"])
    if len(sub) == 0:
        logger.warning("No valid rows for scatter plot of %s", metric)
        return

    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=300)
    nc = sub[~sub["is_censored"]]
    ce = sub[sub["is_censored"]]

    ax.scatter(nc[metric], nc["log_alpha_star"], c="steelblue", marker="o",
               s=28, alpha=0.7, label="Non-censored", edgecolors="white", linewidths=0.3)
    if len(ce) > 0:
        ax.scatter(ce[metric], ce["log_alpha_star"], c="firebrick", marker="^",
                   s=40, alpha=0.8, label=r"Censored ($\alpha^*$=10)", edgecolors="white", linewidths=0.3)

    # Trend line (lowess or linear) for non-censored only
    if len(nc) >= 10:
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess
            lw = sm_lowess(nc["log_alpha_star"].values, nc[metric].values, frac=0.6)
            ax.plot(lw[:, 0], lw[:, 1], color="darkorange", lw=2, label="LOWESS (non-censored)")
        except ImportError:
            z = np.polyfit(nc[metric].values, nc["log_alpha_star"].values, 1)
            xline = np.linspace(nc[metric].min(), nc[metric].max(), 100)
            ax.plot(xline, np.polyval(z, xline), color="darkorange", lw=2, label="Linear fit (non-censored)")

    # Spearman annotation (all features)
    r_all, p_all = spearmanr(sub[metric], sub["log_alpha_star"])
    ax.text(0.03, 0.97, f"Spearman r = {r_all:.3f}\np = {p_all:.2e}\nn = {len(sub)}",
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="grey", boxstyle="round,pad=0.3"))

    pretty = metric.replace("_", " ").title()
    ax.set_xlabel(pretty, fontsize=11)
    ax.set_ylabel(r"log($\alpha^*$)", fontsize=11)
    ax.set_title(f"{pretty} vs log($\\alpha^*$)", fontsize=12)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = output_dir / "plots" / f"scatter_{metric}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved scatter plot -> %s", out)


def make_panel_figure(df: pd.DataFrame, output_dir: Path):
    """1×3 panel figure with all scatter plots side by side."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    available = [m for m in GEOMETRY_METRICS if m in df.columns]
    if len(available) == 0:
        logger.warning("No geometry metrics available for panel figure")
        return

    fig, axes = plt.subplots(1, len(available), figsize=(5.5 * len(available), 4.5), dpi=300)
    if len(available) == 1:
        axes = [axes]

    for ax, metric in zip(axes, available):
        sub = df.dropna(subset=[metric, "log_alpha_star"])
        if len(sub) == 0:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
            continue

        nc = sub[~sub["is_censored"]]
        ce = sub[sub["is_censored"]]

        ax.scatter(nc[metric], nc["log_alpha_star"], c="steelblue", marker="o",
                   s=24, alpha=0.7, label="Non-censored", edgecolors="white", linewidths=0.3)
        if len(ce) > 0:
            ax.scatter(ce[metric], ce["log_alpha_star"], c="firebrick", marker="^",
                       s=36, alpha=0.8, label=r"Censored ($\alpha^*$=10)", edgecolors="white", linewidths=0.3)

        if len(nc) >= 10:
            try:
                from statsmodels.nonparametric.smoothers_lowess import lowess as sm_lowess
                lw = sm_lowess(nc["log_alpha_star"].values, nc[metric].values, frac=0.6)
                ax.plot(lw[:, 0], lw[:, 1], color="darkorange", lw=2)
            except ImportError:
                z = np.polyfit(nc[metric].values, nc["log_alpha_star"].values, 1)
                xline = np.linspace(nc[metric].min(), nc[metric].max(), 100)
                ax.plot(xline, np.polyval(z, xline), color="darkorange", lw=2)

        r_all, p_all = spearmanr(sub[metric], sub["log_alpha_star"])
        ax.text(0.03, 0.97, f"r = {r_all:.3f}\np = {p_all:.2e}",
                transform=ax.transAxes, va="top", ha="left", fontsize=8,
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="grey", boxstyle="round,pad=0.3"))

        pretty = metric.replace("_", " ").title()
        ax.set_xlabel(pretty, fontsize=10)
        ax.set_ylabel(r"log($\alpha^*$)", fontsize=10)
        ax.set_title(f"{pretty}", fontsize=11)
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out = output_dir / "plots" / "correlation_summary.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved panel figure -> %s", out)


def run_primary_analysis(df: pd.DataFrame, output_dir: Path, n_bootstrap: int):
    """Spearman correlation analysis: full + sensitivity (no censored)."""
    logger.info("=== PRIMARY ANALYSIS: Continuous Spearman Correlation ===")

    available = [m for m in GEOMETRY_METRICS if m in df.columns]
    missing = [m for m in GEOMETRY_METRICS if m not in df.columns]
    if missing:
        logger.warning("Missing geometry columns (will be skipped): %s", missing)
    if not available:
        logger.error("No geometry metrics available -- cannot run primary analysis")
        return

    results = []
    for metric in available:
        sub_all = df.dropna(subset=[metric, "log_alpha_star"])
        sub_nc = sub_all[~sub_all["is_censored"]]
        n_total = len(sub_all)
        n_censored = int(sub_all["is_censored"].sum())

        if n_total < 5:
            logger.warning("Fewer than 5 valid rows for %s -- skipping", metric)
            continue
        if len(sub_nc) < 10:
            logger.warning("Fewer than 10 non-censored features for %s "
                           "-- results may be unreliable", metric)

        # Full (with censored assigned alpha*=10)
        r_full, p_full, ci_lo_full, ci_hi_full = bootstrap_spearman_ci(
            sub_all[metric].values, sub_all["log_alpha_star"].values,
            n_bootstrap=n_bootstrap, seed=RNG_SEED)

        # Sensitivity: non-censored only
        if len(sub_nc) >= 5:
            r_nc, p_nc, ci_lo_nc, ci_hi_nc = bootstrap_spearman_ci(
                sub_nc[metric].values, sub_nc["log_alpha_star"].values,
                n_bootstrap=n_bootstrap, seed=RNG_SEED)
        else:
            r_nc, p_nc, ci_lo_nc, ci_hi_nc = np.nan, np.nan, np.nan, np.nan

        results.append({
            "metric": metric,
            "r_full": r_full,
            "p_full": p_full,
            "ci_lower_full": ci_lo_full,
            "ci_upper_full": ci_hi_full,
            "r_nocensored": r_nc,
            "p_nocensored": p_nc,
            "ci_lower_nocensored": ci_lo_nc,
            "ci_upper_nocensored": ci_hi_nc,
            "n_total": n_total,
            "n_censored": n_censored,
        })

        logger.info("  %s: r_full=%.3f (p=%.2e) [%.3f, %.3f]  |  "
                     "r_nocens=%.3f (p=%.2e) [%.3f, %.3f]  n=%d (cens=%d)",
                     metric, r_full, p_full, ci_lo_full, ci_hi_full,
                     r_nc, p_nc, ci_lo_nc, ci_hi_nc, n_total, n_censored)

    res_df = pd.DataFrame(results)
    out_csv = output_dir / "outputs" / "correlation_results.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(out_csv, index=False)
    logger.info("Saved correlation results -> %s", out_csv)

    # Print table
    print("\n" + "=" * 80)
    print("PRIMARY RESULTS: Spearman Rank Correlation with log(alpha*)")
    print("=" * 80)
    print(res_df.to_string(index=False, float_format="%.4f"))
    print("=" * 80 + "\n")

    # Scatter plots
    for metric in available:
        make_scatter_plot(df, metric, output_dir)
    make_panel_figure(df, output_dir)


def run_summary_statistics(df: pd.DataFrame, output_dir: Path):
    """Report summary stats for features and geometry metrics."""
    logger.info("=== SUMMARY STATISTICS ===")

    n_total = len(df)
    n_censored = int(df["is_censored"].sum())
    pct_censored = 100.0 * n_censored / max(1, n_total)
    print(f"\nTotal features: {n_total}")
    print(f"Censored features: {n_censored} ({pct_censored:.1f}%)")

    rows = [{"statistic": "n_total", "value": n_total},
            {"statistic": "n_censored", "value": n_censored},
            {"statistic": "pct_censored", "value": round(pct_censored, 2)}]

    available = [m for m in GEOMETRY_METRICS if m in df.columns]
    for metric in available:
        vals = df[metric].dropna()
        for stat_name, stat_val in [("mean", vals.mean()), ("std", vals.std()),
                                     ("min", vals.min()), ("max", vals.max())]:
            rows.append({"statistic": f"{metric}_{stat_name}", "value": round(stat_val, 6)})
        print(f"  {metric}: mean={vals.mean():.4f}  std={vals.std():.4f}  "
              f"min={vals.min():.4f}  max={vals.max():.4f}")

    stats_df = pd.DataFrame(rows)
    out_csv = output_dir / "outputs" / "summary_stats.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(out_csv, index=False)
    logger.info("Saved summary stats -> %s", out_csv)


# ===================================================================
# SECONDARY ANALYSIS: Binary Classification (SUPPLEMENTARY ONLY)
# ===================================================================

def run_binary_classification(df: pd.DataFrame, output_dir: Path):
    """Practical utility check: logistic regression steerable vs not.
    Uses ONLY non-censored features.  Threshold: alpha* <= 5 → steerable."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import cross_val_score
    except ImportError:
        logger.warning("sklearn not installed -- skipping binary classification")
        return

    logger.info("=== SECONDARY ANALYSIS: Binary Classification (SUPPLEMENTARY) ===")

    sub = df[~df["is_censored"]].copy()
    sub["steerable"] = (sub["alpha_star_best"] <= 5.0).astype(int)

    available = [m for m in GEOMETRY_METRICS if m in sub.columns]
    if not available:
        logger.warning("No geometry metrics for classification -- skipping")
        return

    sub_valid = sub.dropna(subset=available + ["steerable"])
    if len(sub_valid) < 20:
        logger.warning("Too few non-censored features (%d) for classification", len(sub_valid))
        return
    if sub_valid["steerable"].nunique() < 2:
        logger.warning("No class variance in steerable label -- skipping classification")
        return

    results = []

    # Individual metrics
    for metric in available:
        X = sub_valid[[metric]].values
        y = sub_valid["steerable"].values
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        clf = LogisticRegression(max_iter=500, random_state=RNG_SEED)
        scores = cross_val_score(clf, Xs, y, cv=5, scoring="roc_auc")
        results.append({
            "model": metric,
            "mean_auc": round(scores.mean(), 4),
            "std_auc": round(scores.std(), 4),
            "n_samples": len(sub_valid),
            "n_steerable": int(y.sum()),
        })

    # Combined model
    X_all = sub_valid[available].values
    y = sub_valid["steerable"].values
    scaler = StandardScaler()
    Xs_all = scaler.fit_transform(X_all)
    clf = LogisticRegression(max_iter=500, random_state=RNG_SEED)
    scores_all = cross_val_score(clf, Xs_all, y, cv=5, scoring="roc_auc")
    results.append({
        "model": "combined (" + " + ".join(available) + ")",
        "mean_auc": round(scores_all.mean(), 4),
        "std_auc": round(scores_all.std(), 4),
        "n_samples": len(sub_valid),
        "n_steerable": int(y.sum()),
    })

    res_df = pd.DataFrame(results)
    out_csv = output_dir / "outputs" / "classification_supplementary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(out_csv, index=False)
    logger.info("Saved classification results -> %s", out_csv)

    print("\n" + "-" * 60)
    print("SUPPLEMENTARY RESULTS: Binary Classification (practical utility check)")
    print(f"  Threshold: alpha* <= 5 = steerable | Non-censored features only")
    print("-" * 60)
    print(res_df.to_string(index=False))
    print("-" * 60 + "\n")


# ===================================================================
# Legacy Stats  (preserved from original for backward compatibility)
# ===================================================================

def run_correlations(df: pd.DataFrame, y_col: str, x_cols: list):
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


# ===================================================================
# Column Mapping Helpers
# ===================================================================

# Map legacy column names to the canonical names used in the analysis.
_COLUMN_ALIASES = {
    "max_cosine_to_any": "max_cosine_similarity",
    "mean_topk_cos": "neighbor_density",
}


def _apply_column_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """Rename legacy geometry columns to canonical names if present."""
    renames = {}
    for old, new in _COLUMN_ALIASES.items():
        if old in df.columns and new not in df.columns:
            renames[old] = new
    if renames:
        logger.info("Renaming legacy columns: %s", renames)
        df = df.rename(columns=renames)
    return df


# ===================================================================
# Main
# ===================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Phase 3: Predict steerability from geometric features. "
                    "Primary analysis: Spearman correlation. "
                    "Secondary: binary classification (supplementary)."
    )
    # Input group
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--input_csv", type=str,
                   help="Pre-merged CSV with feature_id, alpha_star, and geometry columns (analysis only)")
    g.add_argument("--phase2_dir", type=str,
                   help="Phase2 output dir (contains feature_summary.csv)")
    g.add_argument("--phase2_summary", type=str,
                   help="Path to feature_summary.csv directly")

    # Shared
    ap.add_argument("--output_dir", type=str, default="outputs/phase3_analysis",
                    help="Root output directory for all results")
    ap.add_argument("--n_bootstrap", type=int, default=1000,
                    help="Number of bootstrap samples for Spearman CI")

    # Compute-from-scratch options (only used when --phase2_dir / --phase2_summary)
    ap.add_argument("--config", type=str, default="configs/targets/gemma2_2b_gemmascope_res16k.yaml")
    ap.add_argument("--tau", type=float, default=0.1, help="Cosine threshold for density_tau")
    ap.add_argument("--chunk_size", type=int, default=256)
    ap.add_argument("--topk", type=int, default=50, help="Top-k for neighbor_density and coactivation")
    ap.add_argument("--n_prompts", type=int, default=100,
                    help="Max prompts for baseline activation computation")
    ap.add_argument("--skip_baseline", action="store_true",
                    help="Skip baseline activation computation (geometry only, no coactivation)")
    ap.add_argument("--prompt_csv", type=str, default=None,
                    help="Prompts for baseline; default data/prompts/salad_alpha.csv")
    ap.add_argument("--baseline_cache", type=str, default=None,
                    help="Path to load/save baseline_usage.csv")
    ap.add_argument("--run_name", type=str, default="salad",
                    help="Suffix for merged CSV and cache")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)
    (output_dir / "outputs").mkdir(parents=True, exist_ok=True)

    # Initialize W&B (spec: sae-refusal-steering project)
    try:
        import subprocess as _sp
        _git_hash = _sp.check_output(
            ["git", "rev-parse", "HEAD"], stderr=_sp.DEVNULL
        ).decode().strip()
    except Exception:
        _git_hash = "unknown"

    wandb.init(
        project="sae-refusal-steering",
        name=f"phase3_{args.run_name}",
        tags=[
            f"config_{Path(args.config).stem}",
            "phase3_predictability",
        ],
        config={
            # Ablation metadata (Section 10)
            "experiment_type": "full_run",
            "pipeline_version": "phase3_v2",
            "git_commit": _git_hash,
            "run_tag": f"phase3_{args.run_name}",

            "phase": "phase3_predictability",
            "n_bootstrap": args.n_bootstrap,
            "tau": args.tau,
            "topk": args.topk,
            "run_name": args.run_name,
            "input_csv": args.input_csv,
            "phase2_dir": args.phase2_dir,
        },
    )

    # -----------------------------------------------------------------
    # MODE A: Load pre-merged CSV (analysis only)
    # -----------------------------------------------------------------
    if args.input_csv:
        logger.info("Loading pre-merged CSV: %s", args.input_csv)
        csv_path = Path(args.input_csv)
        if not csv_path.exists():
            raise FileNotFoundError(f"Input CSV not found: {csv_path}")
        merge = pd.read_csv(csv_path)
        merge = _apply_column_aliases(merge)

        # Validate required columns
        if "feature_id" not in merge.columns:
            raise ValueError("Input CSV must contain 'feature_id' column")

        # Handle alpha_star: might be alpha_star_best or need to compute from up/down
        if "alpha_star_best" not in merge.columns:
            if "alpha_star_feature_up" in merge.columns or "alpha_star_feature_down" in merge.columns:
                logger.info("Computing alpha_star_best from directional columns")
                # We need to use make_y_labels logic; check for required columns
                y_df = make_y_labels(merge)
                for col in y_df.columns:
                    if col != "feature_id":
                        merge[col] = y_df[col].values
            else:
                raise ValueError(
                    "Input CSV must contain 'alpha_star_best' or "
                    "'alpha_star_feature_up'/'alpha_star_feature_down' columns"
                )

        # Handle is_censored
        if "is_censored" not in merge.columns:
            if "censored_up" in merge.columns or "censored_down" in merge.columns:
                logger.info("Deriving is_censored from censored_up/censored_down")
                y_df = make_y_labels(merge)
                merge["is_censored"] = y_df["is_censored"].values
            else:
                # Infer: censored if alpha_star_best >= ALPHA_STAR_MAX
                logger.info("Inferring is_censored from alpha_star_best >= %.1f", ALPHA_STAR_MAX)
                merge["is_censored"] = merge["alpha_star_best"] >= ALPHA_STAR_MAX

        # Check geometry columns
        available_geo = [m for m in GEOMETRY_METRICS if m in merge.columns]
        missing_geo = [m for m in GEOMETRY_METRICS if m not in merge.columns]
        if missing_geo:
            logger.warning("Missing geometry columns in input CSV: %s. "
                           "These will be skipped in the analysis.", missing_geo)
        if not available_geo:
            raise ValueError(
                f"Input CSV must contain at least one geometry column. "
                f"Expected any of: {GEOMETRY_METRICS}. "
                f"Found columns: {list(merge.columns)}"
            )

        logger.info("Loaded %d features (%d censored)", len(merge), int(merge["is_censored"].sum()))

    # -----------------------------------------------------------------
    # MODE B: Compute from phase-2 outputs (original flow, preserved)
    # -----------------------------------------------------------------
    else:
        from src.config import load_config, resolve_config
        from src.sae_loader import load_gemmascope_decoder

        if args.phase2_summary:
            summary_path = Path(args.phase2_summary)
            phase2_dir = summary_path.parent
        else:
            phase2_dir = Path(args.phase2_dir)
            summary_path = phase2_dir / "feature_summary.csv"
        if not summary_path.exists():
            raise FileNotFoundError(f"Need {summary_path}")
        summary = pd.read_csv(summary_path)
        summary["feature_id"] = summary["feature_id"].astype(int)
        feature_ids = summary["feature_id"].tolist()
        n_feats_summary = len(feature_ids)
        logger.info("Loaded feature_summary: %d features", n_feats_summary)

        # Y labels (with censoring)
        y_df = make_y_labels(summary)
        logger.info("Y labels: alpha_star_best, tv_at_alpha_star_best, "
                     "success_best, monotone_best, is_censored")

        # Load SAE
        config = resolve_config(load_config(args.config), run_id="phase3")
        _, meta = load_gemmascope_decoder(config)
        npz_path = meta["npz_path"]
        data = np.load(npz_path)
        W_dec = np.asarray(data["W_dec"], dtype=np.float32)
        n_feats_total, d_model = W_dec.shape
        logger.info("W_dec: %d x %d", n_feats_total, d_model)

        # Geometry (all features)
        geo = compute_geometry_chunked(W_dec, tau=args.tau, topk=args.topk,
                                       chunk_size=args.chunk_size)
        geo_df = pd.DataFrame({
            "feature_id": np.arange(n_feats_total),
            "max_cosine_similarity": geo["max_cosine_similarity"],
            "neighbor_density": geo["neighbor_density"],
            "density_tau": geo["density_tau"],
        })

        # Baseline usage + coactivation
        if args.skip_baseline:
            usage_df = pd.DataFrame({
                "feature_id": np.arange(n_feats_total),
                "act_freq": np.nan, "mean_act": np.nan, "mean_z_minus_thr": np.nan,
            })
            # No coactivation without activations
            geo_df["coactivation_correlation"] = np.nan
            logger.info("Skipping baseline (--skip_baseline), coactivation will be NaN")
        else:
            cache_path = (Path(args.baseline_cache) if args.baseline_cache
                          else Path(args.output_dir) / f"baseline_usage_{args.run_name}.csv")
            if cache_path.exists():
                usage_df = pd.read_csv(cache_path)
                logger.info("Loaded baseline usage from %s", cache_path)
                # Coactivation not available from cache
                geo_df["coactivation_correlation"] = np.nan
                logger.info("Coactivation not available from cache -- will be NaN")
            else:
                from src.model_utils import load_model
                model, tokenizer = load_model(config)
                model.eval()
                device = next(model.parameters()).device
                W_enc = torch.tensor(np.asarray(data["W_enc"], dtype=np.float32), device=device)
                b_enc = torch.tensor(np.asarray(data["b_enc"], dtype=np.float32), device=device)
                thr = torch.tensor(np.asarray(data["threshold"], dtype=np.float32), device=device)
                if W_enc.shape[0] != d_model:
                    W_enc = W_enc.T
                prompt_csv = args.prompt_csv or "data/prompts/salad_alpha.csv"
                if not Path(prompt_csv).exists():
                    prompt_csv = str(Path(__file__).resolve().parents[1] / "data" / "prompts" / "salad_alpha.csv")
                dfp = pd.read_csv(prompt_csv, dtype={"prompt": "string"})
                prompts = dfp["prompt"].dropna().astype(str).tolist()[: args.n_prompts]
                act_freq, mean_act, mean_z_minus_thr, raw_act = compute_baseline_usage(
                    model, tokenizer, W_enc, b_enc, thr, prompts, device
                )
                usage_df = pd.DataFrame({
                    "feature_id": np.arange(n_feats_total),
                    "act_freq": act_freq,
                    "mean_act": mean_act,
                    "mean_z_minus_thr": mean_z_minus_thr,
                })
                usage_df.to_csv(cache_path, index=False)
                logger.info("Saved baseline usage -> %s", cache_path)

                # Compute coactivation from raw activations
                logger.info("Computing coactivation correlation from baseline activations")
                feature_ids_arr = np.array(feature_ids)
                coact_vals = compute_coactivation_chunked(
                    raw_act, feature_ids_arr, topk=args.topk)
                # Map to full geo_df: only computed for features in summary
                coact_full = np.full(n_feats_total, np.nan, dtype=np.float32)
                for idx_i, fid in enumerate(feature_ids_arr):
                    if 0 <= fid < n_feats_total:
                        coact_full[fid] = coact_vals[idx_i]
                geo_df["coactivation_correlation"] = coact_full

        # Merge: only features in summary
        merge = summary[["feature_id"]].copy()
        merge = merge.merge(geo_df, on="feature_id", how="left")
        merge = merge.merge(usage_df, on="feature_id", how="left")
        merge = merge.merge(y_df, on="feature_id", how="left")
        merged_path = output_dir / f"{args.run_name}_features_merged.csv"
        merge.to_csv(merged_path, index=False)
        logger.info("Wrote merged CSV: %s (%d rows)", merged_path, len(merge))

    # -----------------------------------------------------------------
    # Censored feature handling
    # -----------------------------------------------------------------
    merge["is_censored"] = merge["is_censored"].fillna(False).astype(bool)

    # Assign censored features alpha*=10 (tied maximum rank)
    censored_mask = merge["is_censored"]
    if censored_mask.any():
        merge.loc[censored_mask, "alpha_star_best"] = ALPHA_STAR_MAX
        logger.info("Assigned alpha*=%.1f to %d censored features",
                     ALPHA_STAR_MAX, censored_mask.sum())

    # Compute log(alpha*)
    merge["log_alpha_star"] = np.log1p(merge["alpha_star_best"].astype(float))

    # -----------------------------------------------------------------
    # Run analyses
    # -----------------------------------------------------------------
    run_summary_statistics(merge, output_dir)
    run_primary_analysis(merge, output_dir, n_bootstrap=args.n_bootstrap)
    run_binary_classification(merge, output_dir)

    # Legacy stats (preserved)
    x_cols_legacy = ["max_cosine_similarity", "neighbor_density",
                     "density_tau", "act_freq", "mean_act"]
    x_cols_legacy = [c for c in x_cols_legacy if c in merge.columns]
    if x_cols_legacy:
        run_correlations(merge, "alpha_star_best", x_cols_legacy)
        run_ols(merge, "alpha_star_best", x_cols_legacy)
        run_logistic(merge, x_cols_legacy, threshold=0.5)
        if "max_cosine_similarity" in merge.columns:
            run_stratified_correlations(merge, "alpha_star_best",
                                        "max_cosine_similarity", q=3)

    print("\nTop 10 by alpha_star_best (lower = better):")
    show_cols = ["feature_id", "alpha_star_best", "is_censored"]
    show_cols += [c for c in GEOMETRY_METRICS if c in merge.columns]
    if "act_freq" in merge.columns:
        show_cols.append("act_freq")
    top10 = merge.sort_values("alpha_star_best").head(10)
    print(top10[show_cols].to_string(index=False))

    # ==================================================================
    # W&B Logging
    # ==================================================================

    # Correlation results as metrics
    corr_csv = output_dir / "outputs" / "correlation_results.csv"
    if corr_csv.exists():
        corr_df = pd.read_csv(corr_csv)
        corr_metrics = {}
        for _, row in corr_df.iterrows():
            m = row["metric"]
            corr_metrics[f"spearman_r_full/{m}"] = row["r_full"]
            corr_metrics[f"spearman_p_full/{m}"] = row["p_full"]
            corr_metrics[f"spearman_r_nocens/{m}"] = row.get("r_nocensored", float("nan"))
        wandb.log(corr_metrics)

    # Feature metrics table (geometry + steerability)
    table_cols = ["feature_id", "alpha_star_best", "is_censored", "log_alpha_star"]
    table_cols += [m for m in GEOMETRY_METRICS if m in merge.columns]
    for extra in ["act_freq", "mean_act", "density_tau"]:
        if extra in merge.columns:
            table_cols.append(extra)
    feature_table = wandb.Table(dataframe=merge[table_cols].copy())
    wandb.log({"feature_metrics": feature_table})

    # Scatter plots as images
    for png in (output_dir / "plots").glob("*.png"):
        wandb.log({png.stem: wandb.Image(str(png))})

    # Artifacts
    artifact = wandb.Artifact(
        name=f"phase3_{args.run_name}",
        type="analysis_outputs",
        metadata={"n_features": len(merge), "n_bootstrap": args.n_bootstrap},
    )
    for fpath in [corr_csv, output_dir / "outputs" / "summary_stats.csv",
                  output_dir / "outputs" / "classification_supplementary.csv"]:
        if fpath.exists():
            artifact.add_file(str(fpath))
    for png in (output_dir / "plots").glob("*.png"):
        artifact.add_file(str(png))
    wandb.log_artifact(artifact)

    wandb.finish()
    logger.info("Phase 3 analysis complete. Outputs in %s", output_dir)


if __name__ == "__main__":
    main()
