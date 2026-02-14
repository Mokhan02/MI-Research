# scripts/phase3_predict_cv_np.py
"""
Cross-validated predictability using only numpy + pandas (no sklearn/scipy).
Logistic AUC for is_steerable (alpha_star <= 5/10), Ridge R2 and Spearman for tv_at_alpha_star_best.

Usage:
  python scripts/phase3_predict_cv_np.py
  python scripts/phase3_predict_cv_np.py --csv outputs/phase3_predictability_arith/arithmetic_features_merged.csv
"""
import argparse
import numpy as np
import pandas as pd

DEFAULT_CSV = "outputs/phase3_predictability_arith/arithmetic_features_merged.csv"
FEATURES = ["max_cosine_to_any", "density_tau", "mean_topk_cos", "act_freq", "mean_z_minus_thr"]


def zscore(X):
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    return (X - mu) / sd


def add_intercept(X):
    return np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)


def stratified_kfold_indices(y, k=5, seed=0):
    rng = np.random.default_rng(seed)
    y = np.asarray(y).astype(int)
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    rng.shuffle(idx0)
    rng.shuffle(idx1)
    folds = [[] for _ in range(k)]
    for i, ix in enumerate(idx0):
        folds[i % k].append(ix)
    for i, ix in enumerate(idx1):
        folds[i % k].append(ix)
    return [np.array(f, dtype=int) for f in folds]


def auc_score(y_true, y_score):
    # AUC via rank statistic (handles ties reasonably with average ranks)
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score)
    order = np.argsort(y_score)
    y_true = y_true[order]
    y_score = y_score[order]
    n1 = y_true.sum()
    n0 = len(y_true) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    # ranks with tie handling
    ranks = np.empty_like(y_score, dtype=float)
    i = 0
    r = 1.0
    while i < len(y_score):
        j = i
        while j + 1 < len(y_score) and y_score[j + 1] == y_score[i]:
            j += 1
        avg_rank = (r + (r + (j - i))) / 2.0
        ranks[i : j + 1] = avg_rank
        r += j - i + 1
        i = j + 1
    sum_ranks_pos = ranks[y_true == 1].sum()
    return (sum_ranks_pos - n1 * (n1 + 1) / 2) / (n1 * n0)


def sigmoid(z):
    z = np.clip(z, -40, 40)
    return 1.0 / (1.0 + np.exp(-z))


def fit_logreg_gd(X, y, l2=1.0, lr=0.1, steps=2000):
    # X includes intercept already
    n, d = X.shape
    w = np.zeros(d)
    for _ in range(steps):
        p = sigmoid(X @ w)
        grad = (X.T @ (p - y)) / n
        grad[1:] += l2 * w[1:] / n  # don't regularize intercept
        w -= lr * grad
    return w


def fit_ridge_closed_form(X, y, l2=1.0):
    # X includes intercept; don't regularize intercept
    n, d = X.shape
    A = X.T @ X
    reg = np.eye(d) * l2
    reg[0, 0] = 0.0
    b = X.T @ y
    return np.linalg.solve(A + reg, b)


def r2_score(y, pred):
    y = np.asarray(y)
    pred = np.asarray(pred)
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def spearman_r(y, pred):
    # rank-based correlation without scipy
    def rank(a):
        order = np.argsort(a)
        r = np.empty_like(order, dtype=float)
        r[order] = np.arange(len(a), dtype=float)
        return r

    ry = rank(y)
    rp = rank(pred)
    ry -= ry.mean()
    rp -= rp.mean()
    denom = np.sqrt(np.sum(ry**2) * np.sum(rp**2))
    return float(np.sum(ry * rp) / denom) if denom > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=DEFAULT_CSV)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    feats = [c for c in FEATURES if c in df.columns]

    # classification datasets
    df["is_steerable_5"] = (df["alpha_star_best"] <= 5).astype(int)
    df["is_steerable_10"] = (df["alpha_star_best"] <= 10).astype(int)

    df_cls = df.dropna(subset=feats + ["alpha_star_best"]).copy()
    df_reg = df.dropna(subset=feats + ["tv_at_alpha_star_best"]).copy()

    print(f"Predictors: {feats}")
    print(f"Rows cls: {len(df_cls)}  Rows reg: {len(df_reg)}")

    # --------- CV Logistic (AUC) ----------
    for label in ["is_steerable_5", "is_steerable_10"]:
        y = df_cls[label].to_numpy().astype(int)
        X = df_cls[feats].to_numpy().astype(float)
        X = zscore(X)
        X = add_intercept(X)

        folds = stratified_kfold_indices(y, k=5, seed=42)
        aucs = []
        for i in range(5):
            te = folds[i]
            tr = np.setdiff1d(np.arange(len(y)), te)
            w = fit_logreg_gd(X[tr], y[tr], l2=1.0, lr=0.2, steps=3000)
            p = sigmoid(X[te] @ w)
            aucs.append(auc_score(y[te], p))
        aucs = np.array(aucs)
        print(f"\nCV AUC {label}: folds={np.round(aucs,3)}  mean±std={np.nanmean(aucs):.3f}±{np.nanstd(aucs):.3f}")

    # --------- CV Ridge regression (R2 + Spearman) ----------
    y = df_reg["tv_at_alpha_star_best"].to_numpy().astype(float)
    X = df_reg[feats].to_numpy().astype(float)
    X = zscore(X)
    X = add_intercept(X)

    rng = np.random.default_rng(42)
    idx = np.arange(len(y))
    rng.shuffle(idx)
    folds = np.array_split(idx, 5)

    r2s, sps = [], []
    for i in range(5):
        te = folds[i]
        tr = np.setdiff1d(idx, te)
        w = fit_ridge_closed_form(X[tr], y[tr], l2=1.0)
        pred = X[te] @ w
        r2s.append(r2_score(y[te], pred))
        sps.append(spearman_r(y[te], pred))

    r2s = np.array(r2s)
    sps = np.array(sps)
    print(f"\nCV Ridge tv_at_alpha_star_best:")
    print(f"  R2 folds={np.round(r2s,3)} mean±std={np.nanmean(r2s):.3f}±{np.nanstd(r2s):.3f}")
    print(f"  Spearman folds={np.round(sps,3)} mean±std={np.nanmean(sps):.3f}±{np.nanstd(sps):.3f}")


if __name__ == "__main__":
    main()
