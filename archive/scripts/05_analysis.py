"""
Analysis and visualization stage.

Merges all experiment outputs, runs statistical analyses,
and generates plots and summaries used in the final paper.
"""

import argparse
import logging
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
import sys
import os
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import r2_score, roc_auc_score

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import load_config, resolve_config, save_resolved_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_all_data(output_dir: str) -> pd.DataFrame:
    """Load and merge all experiment outputs."""
    # Load pre-steering metrics
    presteering_path = os.path.join(output_dir, "presteering_metrics.csv")
    if not os.path.exists(presteering_path):
        raise FileNotFoundError(f"Pre-steering metrics not found at {presteering_path}")
    presteering_df = pd.read_csv(presteering_path)
    
    # Load steerability results
    steerability_path = os.path.join(output_dir, "steerability_results.csv")
    if not os.path.exists(steerability_path):
        raise FileNotFoundError(f"Steerability results not found at {steerability_path}")
    steerability_df = pd.read_csv(steerability_path)
    
    # Load off-target results
    offtarget_path = os.path.join(output_dir, "offtarget_results.csv")
    if not os.path.exists(offtarget_path):
        raise FileNotFoundError(f"Off-target results not found at {offtarget_path}")
    offtarget_df = pd.read_csv(offtarget_path)
    
    # Merge on feature_idx
    # Handle duplicate columns (scorer, alpha_star) by using suffixes
    df = presteering_df.merge(steerability_df, on="feature_idx", how="inner", suffixes=("_pre", "_steer"))
    df = df.merge(offtarget_df, on="feature_idx", how="inner", suffixes=("", "_off"))
    
    # Use alpha_star from steerability (may have _steer suffix if conflict)
    if "alpha_star" not in df.columns:
        if "alpha_star_steer" in df.columns:
            df["alpha_star"] = df["alpha_star_steer"]
        elif "alpha_star_x" in df.columns:
            df["alpha_star"] = df["alpha_star_x"]
        elif "alpha_star_y" in df.columns:
            df["alpha_star"] = df["alpha_star_y"]
    
    logger.info(f"Merged data: {len(df)} features")
    assert len(df) == len(presteering_df), "Data loss during merge"
    
    return df


def correlation_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute correlations between pre-steering metrics and log(alpha_star).
    
    Returns DataFrame with correlation results.
    """
    # Filter out censored features for correlation analysis
    df_non_censored = df[df["censored"] == 0].copy()
    
    if len(df_non_censored) == 0:
        logger.warning("No non-censored features for correlation analysis")
        return pd.DataFrame()
    
    # Compute log(alpha_star)
    df_non_censored["log_alpha_star"] = np.log(df_non_censored["alpha_star"] + 1e-8)
    
    # Pre-steering metrics to correlate
    metrics = [
        "max_cosine_similarity",
        "neighbor_density",
        "coactivation_correlation",
    ]
    
    results = []
    for metric in metrics:
        if metric not in df_non_censored.columns:
            logger.warning(f"Metric {metric} not found in data")
            continue
        
        # Spearman correlation (rank-based)
        spearman_r, spearman_p = stats.spearmanr(
            df_non_censored[metric],
            df_non_censored["log_alpha_star"]
        )
        
        # Pearson correlation
        pearson_r, pearson_p = stats.pearsonr(
            df_non_censored[metric],
            df_non_censored["log_alpha_star"]
        )
        
        results.append({
            "metric": metric,
            "spearman_r": float(spearman_r),
            "spearman_p": float(spearman_p),
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "n_features": len(df_non_censored),
        })
    
    return pd.DataFrame(results)


def predictive_modeling(df: pd.DataFrame) -> dict:
    """
    Train predictive models for log(alpha_star) from pre-steering metrics.
    
    Returns dict with model performance metrics.
    """
    # Filter out censored features
    df_non_censored = df[df["censored"] == 0].copy()
    
    if len(df_non_censored) < 10:
        logger.warning("Too few non-censored features for predictive modeling")
        return {}
    
    # Prepare features and target
    feature_cols = [
        "max_cosine_similarity",
        "neighbor_density",
        "coactivation_correlation",
    ]
    feature_cols = [col for col in feature_cols if col in df_non_censored.columns]
    
    if len(feature_cols) == 0:
        logger.warning("No valid feature columns for predictive modeling")
        return {}
    
    X = df_non_censored[feature_cols].values
    y = np.log(df_non_censored["alpha_star"].values + 1e-8)
    
    # Linear regression
    lr = LinearRegression()
    lr_scores = cross_val_score(lr, X, y, cv=5, scoring="r2")
    lr.fit(X, y)
    lr_r2 = r2_score(y, lr.predict(X))
    
    # Random forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_scores = cross_val_score(rf, X, y, cv=5, scoring="r2")
    rf.fit(X, y)
    rf_r2 = r2_score(y, rf.predict(X))
    
    results = {
        "linear_regression": {
            "cv_r2_mean": float(np.mean(lr_scores)),
            "cv_r2_std": float(np.std(lr_scores)),
            "train_r2": float(lr_r2),
        },
        "random_forest": {
            "cv_r2_mean": float(np.mean(rf_scores)),
            "cv_r2_std": float(np.std(rf_scores)),
            "train_r2": float(rf_r2),
        },
        "n_features": len(df_non_censored),
        "feature_columns": feature_cols,
    }
    
    return results


def create_plots(df: pd.DataFrame, output_dir: str):
    """Create visualization plots."""
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Filter out censored for most plots
    df_non_censored = df[df["censored"] == 0].copy()
    df_non_censored["log_alpha_star"] = np.log(df_non_censored["alpha_star"] + 1e-8)
    
    # Plot 1: Scatter plots of each metric vs log(alpha_star)
    metrics = [
        "max_cosine_similarity",
        "neighbor_density",
        "coactivation_correlation",
    ]
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]
    
    for idx, metric in enumerate(metrics):
        if metric not in df_non_censored.columns:
            continue
        
        ax = axes[idx]
        ax.scatter(
            df_non_censored[metric],
            df_non_censored["log_alpha_star"],
            alpha=0.5,
            s=20,
        )
        
        # Fit trend line
        if len(df_non_censored) > 1:
            z = np.polyfit(df_non_censored[metric], df_non_censored["log_alpha_star"], 1)
            p = np.poly1d(z)
            x_trend = np.linspace(
                df_non_censored[metric].min(),
                df_non_censored[metric].max(),
                100
            )
            ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, label="Trend")
        
        ax.set_xlabel(metric.replace("_", " ").title())
        ax.set_ylabel("log(alpha_star)")
        ax.set_title(f"{metric.replace('_', ' ').title()} vs Steerability")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "metrics_vs_steerability.png"), dpi=150)
    plt.close()
    logger.info("Saved metrics_vs_steerability.png")
    
    # Plot 2: Risk vs Steerability
    if len(df_non_censored) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # R_mag vs alpha_star
        axes[0].scatter(
            df_non_censored["alpha_star"],
            df_non_censored["R_mag_at_alpha_star"],
            alpha=0.5,
            s=20,
        )
        axes[0].set_xlabel("alpha_star")
        axes[0].set_ylabel("R_mag")
        axes[0].set_title("Risk Magnitude vs Steerability")
        axes[0].grid(True, alpha=0.3)
        
        # R_breadth vs alpha_star
        axes[1].scatter(
            df_non_censored["alpha_star"],
            df_non_censored["R_breadth_at_alpha_star"],
            alpha=0.5,
            s=20,
        )
        axes[1].set_xlabel("alpha_star")
        axes[1].set_ylabel("R_breadth")
        axes[1].set_title("Risk Breadth vs Steerability")
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "risk_vs_steerability.png"), dpi=150)
        plt.close()
        logger.info("Saved risk_vs_steerability.png")
    
    # Plot 3: Predicted vs Actual (if we have predictions)
    # This would be generated if we save model predictions
    logger.info("Plots saved to plots/ directory")


def main():
    parser = argparse.ArgumentParser(description="Run analysis and generate plots")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Config file path")
    parser.add_argument("--run-id", type=str, required=True, help="Unique run identifier")
    args = parser.parse_args()
    
    # Load and resolve config
    config = load_config(args.config)
    config = resolve_config(config, args.run_id)
    save_resolved_config(config, args.run_id)
    
    output_dir = config["experiment"]["output_dir"]
    
    # Load and merge all data
    df = load_all_data(output_dir)
    
    # Correlation analysis
    logger.info("Computing correlations...")
    corr_results = correlation_analysis(df)
    if len(corr_results) > 0:
        corr_path = os.path.join(output_dir, "correlation_results.csv")
        corr_results.to_csv(corr_path, index=False)
        logger.info(f"Saved correlation results to {corr_path}")
        logger.info("\nCorrelation Results:")
        logger.info(corr_results.to_string())
    
    # Predictive modeling
    logger.info("Training predictive models...")
    model_results = predictive_modeling(df)
    if len(model_results) > 0:
        model_path = os.path.join(output_dir, "model_results.yaml")
        with open(model_path, "w") as f:
            yaml.dump(model_results, f, default_flow_style=False)
        logger.info(f"Saved model results to {model_path}")
        logger.info("\nModel Performance:")
        logger.info(f"  Linear Regression CV R²: {model_results['linear_regression']['cv_r2_mean']:.4f} ± {model_results['linear_regression']['cv_r2_std']:.4f}")
        logger.info(f"  Random Forest CV R²: {model_results['random_forest']['cv_r2_mean']:.4f} ± {model_results['random_forest']['cv_r2_std']:.4f}")
    
    # Create plots
    logger.info("Generating plots...")
    create_plots(df, output_dir)
    
    # Save merged dataset
    merged_path = os.path.join(output_dir, "merged_results.csv")
    df.to_csv(merged_path, index=False)
    logger.info(f"Saved merged dataset to {merged_path}")
    
    # Save master.csv (as requested)
    master_path = os.path.join(output_dir, "master.csv")
    df.to_csv(master_path, index=False)
    logger.info(f"Saved master dataset to {master_path}")
    
    # Summary statistics
    logger.info("\nSummary Statistics:")
    logger.info(f"  Total features: {len(df)}")
    logger.info(f"  Censored features: {df['censored'].sum()} ({100*df['censored'].mean():.1f}%)")
    if len(df[df["censored"] == 0]) > 0:
        df_non_censored = df[df["censored"] == 0]
        logger.info(f"  alpha_star (non-censored): mean={df_non_censored['alpha_star'].mean():.4f}, median={df_non_censored['alpha_star'].median():.4f}")
        logger.info(f"  R_mag (at alpha_star): mean={df_non_censored['R_mag_at_alpha_star'].mean():.4f}, median={df_non_censored['R_mag_at_alpha_star'].median():.4f}")
        logger.info(f"  R_breadth (at alpha_star): mean={df_non_censored['R_breadth_at_alpha_star'].mean():.4f}, median={df_non_censored['R_breadth_at_alpha_star'].median():.4f}")
    
    logger.info("Analysis complete!")


if __name__ == "__main__":
    main()
