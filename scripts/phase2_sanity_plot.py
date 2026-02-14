# scripts/phase2_sanity_plot.py
"""
Sanity-check plot: for one or more features, plot:
  - Panel 1: delta_logit_target(alpha) — one thin line per prompt, mean in red
  - Panel 2: tv_distance(alpha)        — same layout

Also prints prompt-wise sign consistency (monotone fraction).

Usage:
  python scripts/phase2_sanity_plot.py --csv outputs/phase2/run_rows.csv --features 13851,541,13499
  python scripts/phase2_sanity_plot.py --csv outputs/phase2/run_rows.csv --features 13851 --out_dir outputs/phase2/plots
"""
import argparse, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def monotone_fraction(df_dir, prompts, alphas_ordered, sign=1):
    """Fraction of prompts with >= 2/3 monotone steps in expected direction."""
    keyed = df_dir.set_index(["prompt_idx", "alpha"])
    n_mono = 0
    n_total = 0
    for pidx in prompts:
        deltas = []
        for a in alphas_ordered:
            if (pidx, a) not in keyed.index:
                continue
            row = keyed.loc[(pidx, a)]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            deltas.append(row["delta_logit_target"])
        if len(deltas) < 2:
            continue
        n_total += 1
        n_steps = len(deltas) - 1
        mono_steps = sum(
            1 for i in range(n_steps)
            if sign * (deltas[i + 1] - deltas[i]) >= 0
        )
        if mono_steps >= (2 * n_steps / 3):
            n_mono += 1
    return n_mono / n_total if n_total > 0 else float("nan")


def plot_feature(df_feat, fid, out_dir):
    """Two-panel plot: delta_logit_target and tv_distance vs alpha."""
    prompts = df_feat["prompt_idx"].unique()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    # --- Panel 1: delta_logit_target ---
    for pidx in prompts:
        sub = df_feat[df_feat["prompt_idx"] == pidx].sort_values("alpha")
        ax1.plot(sub["alpha"], sub["delta_logit_target"], color="steelblue", alpha=0.12, lw=0.6)

    mean_delta = df_feat.groupby("alpha")["delta_logit_target"].mean().sort_index()
    ax1.plot(mean_delta.index, mean_delta.values, color="red", lw=2.5, label="mean")
    ax1.axhline(0, color="gray", lw=0.5, ls="--")
    ax1.axvline(0, color="gray", lw=0.5, ls="--")
    ax1.set_ylabel("delta_logit_target (fp32)")
    ax1.set_title(f"Feature {fid}  ({len(prompts)} prompts)")
    ax1.legend(loc="best", fontsize=9)

    # --- Panel 2: tv_distance ---
    for pidx in prompts:
        sub = df_feat[df_feat["prompt_idx"] == pidx].sort_values("alpha")
        ax2.plot(sub["alpha"], sub["tv_distance"], color="darkorange", alpha=0.12, lw=0.6)

    mean_tv = df_feat.groupby("alpha")["tv_distance"].mean().sort_index()
    ax2.plot(mean_tv.index, mean_tv.values, color="darkred", lw=2.5, label="mean TV")
    ax2.axhline(0, color="gray", lw=0.5, ls="--")
    ax2.axvline(0, color="gray", lw=0.5, ls="--")
    ax2.set_xlabel("alpha")
    ax2.set_ylabel("TV distance")
    ax2.legend(loc="best", fontsize=9)

    fig.tight_layout()
    path = os.path.join(out_dir, f"sanity_feat_{fid}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="outputs/phase2/run_rows.csv")
    ap.add_argument("--features", type=str, required=True,
                    help="Comma-separated feature IDs, e.g. 13851,541,13499")
    ap.add_argument("--out_dir", type=str, default="outputs/phase2/plots")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.csv)
    feature_ids = [int(x) for x in args.features.split(",")]

    for fid in feature_ids:
        df_feat = df[df["feature_id"] == fid]
        if len(df_feat) == 0:
            print(f"  WARNING: feature {fid} not found in {args.csv}, skipping")
            continue

        prompts = df_feat["prompt_idx"].unique()
        pos = df_feat[df_feat["alpha"] > 0]
        neg = df_feat[df_feat["alpha"] < 0]
        pos_alphas = sorted(pos["alpha"].unique(), key=lambda a: abs(a))
        neg_alphas = sorted(neg["alpha"].unique(), key=lambda a: abs(a))

        mono_up = monotone_fraction(pos, prompts, pos_alphas, sign=+1)
        mono_down = monotone_fraction(neg, prompts, neg_alphas, sign=-1)

        print(f"Feature {fid}: {len(df_feat)} rows, {len(prompts)} prompts")
        print(f"  monotone_frac_up   = {mono_up:.3f}")
        print(f"  monotone_frac_down = {mono_down:.3f}")

        plot_feature(df_feat, fid, args.out_dir)


if __name__ == "__main__":
    main()
