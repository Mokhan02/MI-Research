# scripts/phase3_predict_cv.py
"""
Cross-validated predictability: logistic AUC for is_steerable (alpha_star <= 5/10),
Ridge R2 and Spearman for tv_at_alpha_star_best.

Usage:
  python scripts/phase3_predict_cv.py
  python scripts/phase3_predict_cv.py --csv outputs/phase3_predictability_arith/arithmetic_features_merged.csv
"""
import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, r2_score
from scipy.stats import spearmanr

DEFAULT_CSV = "outputs/phase3_predictability_arith/arithmetic_features_merged.csv"

# Pick predictors you trust (mean_act is often missing — include only if present & non-null enough)
BASE_FEATURES = ["max_cosine_to_any", "density_tau", "mean_topk_cos", "act_freq", "mean_z_minus_thr"]


def load_data(csv_path: str):
    df = pd.read_csv(csv_path)

    # Targets
    df["is_steerable_5"] = (df["alpha_star_best"] <= 5).astype(int)
    df["is_steerable_10"] = (df["alpha_star_best"] <= 10).astype(int)

    # Choose feature columns that actually exist
    feats = [c for c in BASE_FEATURES if c in df.columns]

    # Drop rows with missing predictors or missing outcomes we need
    cls_cols = feats + ["alpha_star_best", "is_steerable_5", "is_steerable_10"]
    reg_cols = feats + ["tv_at_alpha_star_best"]

    df_cls = df.dropna(subset=cls_cols).copy()
    df_reg = df.dropna(subset=reg_cols).copy()

    return df_cls, df_reg, feats


def cv_logistic_auc(df, feats, label_col, n_splits=5, seed=0):
    X = df[feats].values
    y = df[label_col].values

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs"
        ))
    ])

    aucs = []
    for tr, te in cv.split(X, y):
        model.fit(X[tr], y[tr])
        p = model.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p))

    return np.array(aucs)


def cv_regression(df, feats, y_col, n_splits=5, seed=0):
    X = df[feats].values
    y = df[y_col].values

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0))
    ])

    r2s = []
    spears = []
    for tr, te in cv.split(X):
        model.fit(X[tr], y[tr])
        pred = model.predict(X[te])

        r2s.append(r2_score(y[te], pred))

        r, _ = spearmanr(y[te], pred)
        spears.append(r)

    return np.array(r2s), np.array(spears)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=DEFAULT_CSV, help="Merged features CSV from phase3_predictability")
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df_cls, df_reg, feats = load_data(args.csv)
    print(f"Loaded classification rows: {len(df_cls)}  regression rows: {len(df_reg)}")
    print("Predictors:", feats)

    # --- Classification: steerable within budget ---
    for label in ["is_steerable_5", "is_steerable_10"]:
        aucs = cv_logistic_auc(df_cls, feats, label, n_splits=args.n_splits, seed=args.seed)
        print(f"\nCV AUC for {label}:")
        print(f"  fold AUCs: {np.round(aucs, 3)}")
        print(f"  mean±std: {aucs.mean():.3f} ± {aucs.std():.3f}")

    # --- Regression: risk at first success ---
    r2s, spears = cv_regression(df_reg, feats, "tv_at_alpha_star_best", n_splits=args.n_splits, seed=args.seed)
    print("\nCV regression for tv_at_alpha_star_best:")
    print(f"  R2 fold: {np.round(r2s, 3)}")
    print(f"  R2 mean±std: {r2s.mean():.3f} ± {r2s.std():.3f}")
    print(f"  Spearman fold: {np.round(spears, 3)}")
    print(f"  Spearman mean±std: {np.nanmean(spears):.3f} ± {np.nanstd(spears):.3f}")


if __name__ == "__main__":
    main()
