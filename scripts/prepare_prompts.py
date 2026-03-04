"""
Download SALADBench base_set from HuggingFace, sample N prompts (stratified by
top-level safety category), and split into select / alpha / holdout CSVs.

Also writes a neutral set from the HH-harmless subset (benign questions).

Usage:
  PYTHONPATH=. python scripts/prepare_prompts.py --out_dir data/prompts --n_prompts 300 --seed 42
"""
import argparse
import random
from pathlib import Path
from collections import defaultdict

import pandas as pd


def download_salad_data():
    from datasets import load_dataset
    ds = load_dataset("OpenSafetyLab/Salad-Data", name="base_set", split="train")
    return ds


def stratified_sample(df: pd.DataFrame, n: int, category_col: str, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    groups = defaultdict(list)
    for idx, row in df.iterrows():
        groups[row[category_col]].append(idx)

    n_cats = len(groups)
    per_cat = max(1, n // n_cats)
    sampled = []
    for cat, indices in sorted(groups.items()):
        k = min(per_cat, len(indices))
        sampled.extend(rng.sample(indices, k))

    # If we haven't reached n yet, sample remaining from the leftovers
    remaining = [i for i in df.index if i not in set(sampled)]
    if len(sampled) < n and remaining:
        extra = min(n - len(sampled), len(remaining))
        sampled.extend(rng.sample(remaining, extra))

    sampled = sampled[:n]
    rng.shuffle(sampled)
    return df.loc[sampled].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description="Prepare SALADBench prompts for the pipeline")
    ap.add_argument("--out_dir", type=str, default="data/prompts")
    ap.add_argument("--n_prompts", type=int, default=300, help="Total unsafe prompts to sample")
    ap.add_argument("--n_neutral", type=int, default=150, help="Neutral (benign) prompts to sample")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--select_frac", type=float, default=0.40, help="Fraction for select split")
    ap.add_argument("--alpha_frac", type=float, default=0.33, help="Fraction for alpha split")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading SALADBench base_set...")
    ds = download_salad_data()
    df_all = pd.DataFrame(ds)
    print(f"  Total rows: {len(df_all)}")

    # Rename columns to pipeline convention
    df_all = df_all.rename(columns={"question": "prompt", "1-category": "category"})
    if "prompt" not in df_all.columns:
        raise ValueError(f"Expected 'question' column. Got: {list(df_all.columns)}")

    # Sample unsafe prompts (stratified by top-level category)
    df_unsafe = stratified_sample(df_all, args.n_prompts, "category", args.seed)
    df_unsafe = df_unsafe[["prompt", "category"]].copy()
    print(f"  Sampled {len(df_unsafe)} unsafe prompts across {df_unsafe['category'].nunique()} categories")

    # Split: select / alpha / holdout
    n_select = int(len(df_unsafe) * args.select_frac)
    n_alpha = int(len(df_unsafe) * args.alpha_frac)
    df_select = df_unsafe.iloc[:n_select].reset_index(drop=True)
    df_alpha = df_unsafe.iloc[n_select:n_select + n_alpha].reset_index(drop=True)
    df_holdout = df_unsafe.iloc[n_select + n_alpha:].reset_index(drop=True)

    df_select.to_csv(out_dir / "salad_select.csv", index=False)
    df_alpha.to_csv(out_dir / "salad_alpha.csv", index=False)
    df_holdout.to_csv(out_dir / "salad_holdout.csv", index=False)
    print(f"  Wrote salad_select.csv ({len(df_select)}), salad_alpha.csv ({len(df_alpha)}), salad_holdout.csv ({len(df_holdout)})")

    # Neutral set: reuse existing benign neutrals if present; only generate if missing
    existing_neutral = out_dir / "neutral_select.csv"
    if existing_neutral.exists():
        print(f"  Keeping existing neutral CSVs in {out_dir} (benign prompts already present)")
    else:
        print(f"  No existing neutral_select.csv found; generating neutral set from benign completions")
        neutral_prompts = [
            "The weather today is", "I enjoy reading about", "My favorite food is",
            "A good book to read is", "The best way to learn is", "Music helps me",
            "Exercise is important because", "Cooking at home is", "Traveling teaches us",
            "A healthy breakfast includes", "The ocean is", "Learning a new language",
            "Gardening can be", "Morning routines help", "Friendship is important because",
            "A walk in the park", "Reading before bed", "Clean water is essential",
            "Teamwork makes", "Art inspires us", "Volunteering helps", "Sleep is vital for",
            "Science explains", "History teaches", "Nature provides", "Kindness matters because",
            "Libraries are", "Education opens", "Recycling helps", "Sunlight is good for",
            "Writing improves", "Animals deserve", "Technology enables", "Communication is key",
            "Practice makes", "Patience is", "Curiosity drives", "Laughter is",
            "Breathing exercises", "Meditation helps", "Hydration is", "Creativity flows",
            "Balance in life", "Honesty builds", "Gratitude improves", "Planning ahead",
            "Fresh air is", "Walking daily", "Smiling can", "Learning never stops",
        ]
        rng = random.Random(args.seed + 1)
        rng.shuffle(neutral_prompts)
        n_total = len(neutral_prompts)
        n_ns = int(n_total * args.select_frac)
        n_na = int(n_total * args.alpha_frac)
        pd.DataFrame({"prompt": neutral_prompts[:n_ns]}).to_csv(out_dir / "neutral_select.csv", index=False)
        pd.DataFrame({"prompt": neutral_prompts[n_ns:n_ns + n_na]}).to_csv(out_dir / "neutral_alpha.csv", index=False)
        pd.DataFrame({"prompt": neutral_prompts[n_ns + n_na:]}).to_csv(out_dir / "neutral_holdout.csv", index=False)
        print(f"  Wrote neutral_select.csv ({n_ns}), neutral_alpha.csv ({n_na}), neutral_holdout.csv ({n_total - n_ns - n_na})")

    print("Done.")


if __name__ == "__main__":
    main()
