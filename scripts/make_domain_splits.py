"""
Create domain-specific prompt splits so selection and evaluation do not share prompts.

Outputs per domain (planets, capitals, neutral):
  - {domain}_select.csv   Used ONLY for feature selection (contrast: task − neutral).
  - {domain}_alpha.csv    Used ONLY for α-grid / α* runs (must have target for phase2_run).
  - {domain}_holdout.csv  Used ONLY for final reporting.

If selection and evaluation share prompts, selection is contaminated. Run this first.

Usage:
  python scripts/make_domain_splits.py --prompts_dir data/prompts --out_dir data/prompts
  # If you have CSVs with prompt,target (e.g. from make_phase2_csvs), put in data/ and use --data_dir data
"""
import argparse
import random
from pathlib import Path

import pandas as pd


DOMAINS = ["planets", "capitals", "neutral"]
# neutral = control prompts (no task target)
SPLIT_FRAC = (0.4, 0.3, 0.3)  # select, alpha, holdout


def load_prompts_txt(path: Path) -> list[str]:
    if not path.exists():
        return []
    lines = path.read_text().strip().splitlines()
    return [l.strip() for l in lines if l.strip()]


def load_domain_data(prompts_dir: Path, data_dir, domain: str):
    """Load prompt,target if available from data_dir (e.g. planets_only.csv). Else from .txt with target=''."""
    if data_dir:
        for name in (f"{domain}_only.csv", f"phase2_{domain}.csv"):
            p = data_dir / name
            if p.exists():
                df = pd.read_csv(p, dtype={"prompt": "string", "target": "string"})
                if "prompt" in df.columns:
                    if "target" not in df.columns:
                        df["target"] = ""
                    df["prompt"] = df["prompt"].astype(str).str.strip()
                    return df[["prompt", "target"]].dropna(subset=["prompt"])
    if domain == "neutral":
        txt_name = "phase2_control.txt"
    else:
        txt_name = f"phase2_{domain}.txt"
    path = prompts_dir / txt_name
    prompts = load_prompts_txt(path)
    if not prompts:
        return None
    return pd.DataFrame({"prompt": prompts, "target": [""] * len(prompts)})


def main():
    ap = argparse.ArgumentParser(description="Create select/alpha/holdout CSVs per domain")
    ap.add_argument("--prompts_dir", type=str, default="data/prompts", help="Dir with phase2_<domain>.txt")
    ap.add_argument("--data_dir", type=str, default=None, help="Optional: dir with <domain>_only.csv (prompt,target) to preserve targets")
    ap.add_argument("--out_dir", type=str, default="data/prompts", help="Where to write *_select.csv, *_alpha.csv, *_holdout.csv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    prompts_dir = Path(args.prompts_dir)
    data_dir = Path(args.data_dir) if args.data_dir else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)

    for domain in DOMAINS:
        df = load_domain_data(prompts_dir, data_dir, domain)
        if df is None or len(df) == 0:
            print(f"[Skip] {domain}: no source found")
            continue

        n = len(df)
        indices = list(range(n))
        random.shuffle(indices)
        s1 = int(n * SPLIT_FRAC[0])
        s2 = int(n * (SPLIT_FRAC[0] + SPLIT_FRAC[1]))
        i_select = indices[:s1]
        i_alpha = indices[s1:s2]
        i_holdout = indices[s2:]

        for name, idx_list in [("select", i_select), ("alpha", i_alpha), ("holdout", i_holdout)]:
            out_df = df.iloc[idx_list].reset_index(drop=True)
            out_path = out_dir / f"{domain}_{name}.csv"
            out_df.to_csv(out_path, index=False)
            print(f"Wrote {out_path}  ({len(out_df)} rows)")

    print("Done. Use *_select for feature selection, *_alpha for alpha* runs, *_holdout for final reporting.")


if __name__ == "__main__":
    main()
