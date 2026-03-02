"""
Off-target evaluation stage.

Measures how steering a single feature affects performance
on unrelated benchmarks, capturing collateral or spillover effects.
"""

import argparse
import logging
import numpy as np
import pandas as pd
import torch
import yaml
from pathlib import Path
import sys
import os
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import load_config, resolve_config, save_resolved_config
from src.model_utils import load_model, load_sae, create_steering_hook
from src.scoring import load_scorer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_off_target_prompts(config: dict) -> dict:
    """
    Load off-target benchmark prompts.
    
    Returns dict mapping benchmark name to list of prompts.
    """
    data_dir = config.get("data", {}).get("data_dir", "data")
    prompts_dir = os.path.join(data_dir, "prompts")
    
    # Try to load off-target benchmarks
    benchmarks = {}
    benchmark_names = ["gpqa", "truthfulqa", "darkbench"]  # From experimental setup
    
    for bench_name in benchmark_names:
        bench_path = os.path.join(prompts_dir, f"{bench_name}.txt")
        if os.path.exists(bench_path):
            with open(bench_path, "r") as f:
                prompts = [line.strip() for line in f if line.strip()]
            benchmarks[bench_name] = prompts
            logger.info(f"Loaded {len(prompts)} prompts for {bench_name}")
        else:
            # Placeholder prompts
            logger.warning(
                f"Off-target prompts not found at {bench_path}. "
                f"Using placeholder prompts for {bench_name}."
            )
            benchmarks[bench_name] = [f"{bench_name} prompt {i}." for i in range(50)]
    
    if len(benchmarks) == 0:
        raise ValueError("No off-target benchmarks found")
    
    return benchmarks


def run_model_with_steering(
    model,
    tokenizer,
    prompts: list,
    sae,
    feature_idx: int,
    alpha: float,
    config: dict,
) -> list:
    """
    Run model on prompts with steering intervention.
    
    Same as in 03_steerability.py - placeholder implementation.
    """
    # TODO: Replace with actual model forward pass with hook
    # Placeholder: return deterministic outputs sensitive to alpha
    outputs = []
    np.random.seed(config["experiment"]["seed"])
    for prompt in prompts:
        # Deterministic placeholder: output changes based on alpha and feature_idx
        base_hash = hash(f"{prompt}_{feature_idx}") % 10000
        alpha_effect = int(alpha * 100) % 1000
        combined_hash = (base_hash + alpha_effect) % 10000
        output = f"Steered output (alpha={alpha:.2f}, f={feature_idx}): {prompt[:30]}... [score_proxy={combined_hash/10000.0:.3f}]"
        outputs.append(output)
    return outputs


def compute_offtarget_deltas(
    config: dict,
    model,
    tokenizer,
    sae,
    feature_idx: int,
    benchmarks: dict,
    scorers: dict,
    alpha: float,
) -> dict:
    """
    Compute off-target deltas for a feature at a given alpha.
    
    Δ = score_steered - score_baseline for each benchmark.
    
    Returns dict mapping benchmark name to delta.
    """
    deltas = {}
    
    # Compute baseline scores (alpha=0)
    baseline_scores = {}
    for bench_name, prompts in benchmarks.items():
        baseline_outputs = run_model_with_steering(
            model, tokenizer, prompts, sae, feature_idx, 0.0, config
        )
        scores = scorers[bench_name].score(prompts, baseline_outputs)
        baseline_scores[bench_name] = float(np.mean(scores))
    
    # Compute steered scores
    steered_scores = {}
    for bench_name, prompts in benchmarks.items():
        steered_outputs = run_model_with_steering(
            model, tokenizer, prompts, sae, feature_idx, alpha, config
        )
        scores = scorers[bench_name].score(prompts, steered_outputs)
        steered_scores[bench_name] = float(np.mean(scores))
    
    # Compute deltas
    for bench_name in benchmarks.keys():
        delta = steered_scores[bench_name] - baseline_scores[bench_name]
        deltas[bench_name] = delta
    
    return deltas


def compute_risk_metrics(deltas: dict, tau: float) -> tuple:
    """
    Compute risk metrics from off-target deltas.
    
    R_mag = mean(|Δ|)
    R_breadth = fraction(|Δ| > tau)
    
    Returns:
        (R_mag, R_breadth)
    """
    abs_deltas = [abs(d) for d in deltas.values()]
    
    R_mag = float(np.mean(abs_deltas))
    R_breadth = float(np.mean([abs(d) > tau for d in deltas.values()]))
    
    return R_mag, R_breadth


def main():
    parser = argparse.ArgumentParser(description="Evaluate off-target effects")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Config file path")
    parser.add_argument("--run-id", type=str, required=True, help="Unique run identifier")
    args = parser.parse_args()
    
    # Load and resolve config
    config = load_config(args.config)
    config = resolve_config(config, args.run_id)
    save_resolved_config(config, args.run_id)
    
    # Load selected features
    output_dir = config["experiment"]["output_dir"]
    features_path = os.path.join(output_dir, "selected_features.npy")
    if not os.path.exists(features_path):
        raise FileNotFoundError(
            f"Selected features not found at {features_path}. "
            "Run 01_choose_features.py first."
        )
    selected_features = np.load(features_path)
    logger.info(f"Loaded {len(selected_features)} selected features")
    
    # Load steerability results
    steerability_path = os.path.join(output_dir, "steerability_results.csv")
    if not os.path.exists(steerability_path):
        raise FileNotFoundError(
            f"Steerability results not found at {steerability_path}. "
            "Run 03_steerability.py first."
        )
    steerability_df = pd.read_csv(steerability_path)
    logger.info(f"Loaded steerability results for {len(steerability_df)} features")
    
    logger.warning("=" * 60)
    logger.warning("PLACEHOLDER MODE: Using placeholder scorer and SAE")
    logger.warning("=" * 60)
    
    # Load model and SAE (with fallback)
    # In placeholder mode, skip model loading
    try:
        model, tokenizer = load_model(config)
    except (OSError, RuntimeError) as e:
        logger.warning(f"Model loading failed: {e}")
        logger.warning("Using placeholder model (outputs will be deterministic placeholders)")
        model = None
        tokenizer = None
    
    try:
        sae = load_sae(config)
    except (ValueError, NotImplementedError) as e:
        logger.warning(f"SAE loading failed: {e}")
        logger.warning("Using placeholder SAE")
        # TODO: Replace with actual SAE loading
        class PlaceholderSAE:
            def __init__(self, n_features, hidden_dim=2048):
                self.n_features = n_features
                self.hook_point = config.get("sae", {}).get("hook_point", "TBD")
                np.random.seed(config["experiment"]["seed"])
                self.decoder_weights = torch.from_numpy(
                    np.random.randn(n_features, hidden_dim).astype(np.float32)
                )
                if config.get("sae", {}).get("normalize_decoder", True):
                    self.decoder_weights = torch.nn.functional.normalize(
                        self.decoder_weights, dim=-1
                    )
            
            def get_decoder_direction(self, feature_idx: int) -> torch.Tensor:
                return self.decoder_weights[feature_idx]
        
        n_features_total = config.get("sae", {}).get("n_features_total", 1000)
        sae = PlaceholderSAE(n_features_total)
    
    # Load off-target benchmarks and scorers
    benchmarks = load_off_target_prompts(config)
    scorers = {
        name: load_scorer(config, "off_target") for name in benchmarks.keys()
    }
    
    tau = config["steering"]["tau"]
    alpha0 = config["steering"]["alpha0"]
    
    logger.info(f"Evaluating off-target effects for {len(selected_features)} features")
    logger.info(f"Using {len(benchmarks)} off-target benchmarks")
    logger.info(f"tau = {tau}, alpha0 = {alpha0}")
    
    # Create mapping from feature_idx to alpha_star
    alpha_star_map = dict(zip(steerability_df["feature_idx"], steerability_df["alpha_star"]))
    
    # Evaluate off-target effects at two regimes:
    # (i) alpha = alpha_star(f) - steerability threshold
    # (ii) alpha = alpha0 - fixed low coefficient
    
    results = []
    for feature_idx in tqdm(selected_features, desc="Computing off-target effects"):
        feature_idx = int(feature_idx)
        alpha_star = alpha_star_map[feature_idx]
        
        # Regime 1: alpha = alpha_star(f)
        deltas_star = compute_offtarget_deltas(
            config, model, tokenizer, sae, feature_idx, benchmarks, scorers, alpha_star
        )
        R_mag_star, R_breadth_star = compute_risk_metrics(deltas_star, tau)
        
        # Regime 2: alpha = alpha0
        deltas_alpha0 = compute_offtarget_deltas(
            config, model, tokenizer, sae, feature_idx, benchmarks, scorers, alpha0
        )
        R_mag_alpha0, R_breadth_alpha0 = compute_risk_metrics(deltas_alpha0, tau)
        
        # Store results
        result = {
            "feature_idx": feature_idx,
            "alpha_star": float(alpha_star),
            # Regime 1 metrics
            "R_mag_at_alpha_star": float(R_mag_star),
            "R_breadth_at_alpha_star": float(R_breadth_star),
            # Regime 2 metrics
            "R_mag_at_alpha0": float(R_mag_alpha0),
            "R_breadth_at_alpha0": float(R_breadth_alpha0),
            "scorer": "placeholder",  # Mark as placeholder
        }
        
        # Also store individual deltas for each benchmark
        for bench_name, delta in deltas_star.items():
            result[f"delta_{bench_name}_at_alpha_star"] = float(delta)
        for bench_name, delta in deltas_alpha0.items():
            result[f"delta_{bench_name}_at_alpha0"] = float(delta)
        
        results.append(result)
    
    df = pd.DataFrame(results)
    
    # Shape assertions
    assert len(df) == len(selected_features), "Mismatch in number of features"
    assert set(df["feature_idx"]) == set(selected_features), "Feature index mismatch"
    assert (df["R_mag_at_alpha_star"] >= 0).all(), "R_mag must be non-negative"
    assert (df["R_breadth_at_alpha_star"] >= 0).all() and (df["R_breadth_at_alpha_star"] <= 1).all(), \
        "R_breadth must be in [0, 1]"
    
    # Save results
    results_path = os.path.join(output_dir, "offtarget_results.csv")
    df.to_csv(results_path, index=False)
    logger.info(f"Saved results to {results_path}")
    
    # Save risk.csv (as requested)
    risk_path = os.path.join(output_dir, "risk.csv")
    risk_df = df[[
        "feature_idx",
        "alpha_star",
        "R_mag_at_alpha_star",
        "R_breadth_at_alpha_star",
        "R_mag_at_alpha0",
        "R_breadth_at_alpha0",
        "scorer",
    ]]
    risk_df.to_csv(risk_path, index=False)
    logger.info(f"Saved risk metrics to {risk_path}")
    
    # Log summary statistics
    logger.info(f"Summary (at alpha_star):")
    logger.info(f"  R_mag range: [{df['R_mag_at_alpha_star'].min():.4f}, {df['R_mag_at_alpha_star'].max():.4f}]")
    logger.info(f"  R_mag mean: {df['R_mag_at_alpha_star'].mean():.4f}")
    logger.info(f"  R_breadth range: [{df['R_breadth_at_alpha_star'].min():.4f}, {df['R_breadth_at_alpha_star'].max():.4f}]")
    logger.info(f"  R_breadth mean: {df['R_breadth_at_alpha_star'].mean():.4f}")
    
    logger.info(f"Summary (at alpha0={alpha0}):")
    logger.info(f"  R_mag range: [{df['R_mag_at_alpha0'].min():.4f}, {df['R_mag_at_alpha0'].max():.4f}]")
    logger.info(f"  R_mag mean: {df['R_mag_at_alpha0'].mean():.4f}")
    logger.info(f"  R_breadth range: [{df['R_breadth_at_alpha0'].min():.4f}, {df['R_breadth_at_alpha0'].max():.4f}]")
    logger.info(f"  R_breadth mean: {df['R_breadth_at_alpha0'].mean():.4f}")
    
    # Save metadata
    metadata = {
        "n_features": len(df),
        "tau": tau,
        "alpha0": alpha0,
        "benchmarks": list(benchmarks.keys()),
    }
    metadata_path = os.path.join(output_dir, "04_offtarget_metadata.yaml")
    with open(metadata_path, "w") as f:
        yaml.dump(metadata, f, default_flow_style=False)
    logger.info(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
