# scripts/make_feature_sets.py
"""
Build S_good.txt and S_bad.txt from a phase2 feature_summary.csv.

S_good: high monotonicity, high success rate, low TV (bottom 30%).
S_bad:  low monotonicity, low success rate, TV-matched to S_good.

Usage:
  python scripts/make_feature_sets.py --src outputs/phase2_arith_p75_f100_tau01
  python scripts/make_feature_sets.py --src outputs/phase2_arith_p75_f100_tau01 \
      --mono_min 0.7 --succ_min 0.4 --tv_pct 0.40
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, required=True, help="Phase2 output directory")
    ap.add_argument("--mono_min", type=float, default=0.80, help="Min monotone_frac for S_good")
    ap.add_argument("--succ_min", type=float, default=0.50, help="Min success_rate for S_good")
    ap.add_argument("--tv_pct", type=float, default=0.30, help="TV percentile cutoff for S_good (0.30 = bottom 30%%)")
    ap.add_argument("--bad_mono_max", type=float, default=0.20, help="Max monotone_frac for S_bad")
    ap.add_argument("--bad_succ_max", type=float, default=0.20, help="Max success_rate for S_bad")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    src = Path(args.src)
    s = pd.read_csv(src / "feature_summary.csv")

    # --- S_good ---
    tv_cut = s["tv_mean"].quantile(args.tv_pct)

    good = s[
        (s["monotone_frac_up"] >= args.mono_min) &
        (s["monotone_frac_down"] >= args.mono_min) &
        (s["success_rate_up"] >= args.succ_min) &
        (s["success_rate_down"] >= args.succ_min) &
        (s["tv_mean"] <= tv_cut)
    ].copy()

    good_ids = good["feature_id"].astype(int).tolist()
    good_path = src / "S_good.txt"
    with open(good_path, "w") as f:
        for fid in good_ids:
            f.write(str(fid) + "\n")

    print(f"tv_cut(p{int(args.tv_pct*100)}) = {tv_cut:.4f}")
    print(f"S_good size = {len(good_ids)}")
    print(f"Wrote {good_path}")
    if len(good) > 0:
        print("\nTop 10 S_good by tv_mean:")
        cols = ["feature_id", "tv_mean", "success_rate_up", "success_rate_down",
                "monotone_frac_up", "monotone_frac_down"]
        cols = [c for c in cols if c in good.columns]
        print(good.sort_values("tv_mean").head(10)[cols].to_string(index=False))

    # --- S_bad ---
    bad_pool = s[
        (s["monotone_frac_up"] <= args.bad_mono_max) &
        (s["monotone_frac_down"] <= args.bad_mono_max) &
        (s["success_rate_up"] <= args.bad_succ_max) &
        (s["success_rate_down"] <= args.bad_succ_max)
    ].copy()

    n = len(good_ids)

    # TV binning to roughly match S_good distribution
    s["tv_bin"] = pd.qcut(s["tv_mean"], q=5, duplicates="drop")
    good_df = s[s["feature_id"].isin(good_ids)].copy()
    bins = good_df["tv_bin"].value_counts()

    picked = []
    for b, k in bins.items():
        cand = bad_pool[bad_pool["tv_bin"] == b]
        take = cand.sample(min(k, len(cand)), random_state=args.seed) if len(cand) > 0 else cand
        picked.append(take)

    bad = pd.concat(picked, axis=0).head(n).copy() if picked else pd.DataFrame()
    bad_ids = bad["feature_id"].astype(int).tolist() if len(bad) > 0 else []

    bad_path = src / "S_bad.txt"
    with open(bad_path, "w") as f:
        for fid in bad_ids:
            f.write(str(fid) + "\n")

    print(f"\nS_bad size = {len(bad_ids)}  (target = {n})")
    print(f"Wrote {bad_path}")
    if len(bad) > 0:
        print("\nTop 10 S_bad by tv_mean:")
        cols = ["feature_id", "tv_mean", "success_rate_up", "success_rate_down",
                "monotone_frac_up", "monotone_frac_down"]
        cols = [c for c in cols if c in bad.columns]
        print(bad.sort_values("tv_mean").head(10)[cols].to_string(index=False))

    if len(bad_ids) < n:
        print(f"\nWARNING: S_bad ({len(bad_ids)}) smaller than S_good ({n}). "
              f"Consider loosening --bad_mono_max / --bad_succ_max.")


if __name__ == "__main__":
    main()
