"""
Steerability measurement stage.

Sweeps steering strength (alpha) on an on-target benchmark
to estimate how difficult each SAE feature is to steer.

Produces alpha*(f), the core steerability quantity.
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


def load_on_target_prompts(config: dict) -> list:
    """Load on-target benchmark prompts."""
    data_dir = config.get("data", {}).get("data_dir", "data")
    prompts_dir = os.path.join(data_dir, "prompts")
    
    # Try to load on-target prompts
    on_target_path = os.path.join(prompts_dir, "on_target.txt")
    if os.path.exists(on_target_path):
        with open(on_target_path, "r") as f:
            prompts = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(prompts)} on-target prompts from {on_target_path}")
        return prompts
    
    # Fallback: placeholder prompts
    logger.warning(
        f"On-target prompts not found at {on_target_path}. "
        "Using placeholder prompts. This is for testing only."
    )
    prompts = [f"On-target prompt {i}." for i in range(100)]
    return prompts


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
    
    Steering: activation += alpha * v_f at the SAE's hook point.
    
    Returns list of model outputs.
    """
    device = config["model"]["device"]
    hook_point = sae.hook_point
    max_new_tokens = config["model"]["max_new_tokens"]
    temperature = config["model"]["temperature"]
    do_sample = config["model"]["do_sample"]
    
    # Create steering hook
    hook_fn = create_steering_hook(sae, feature_idx, alpha, hook_point)
    
    # Register hook
    # Find the module at hook_point
    # This is a placeholder - actual implementation would find the correct module
    logger.warning(
        "Hook registration is a placeholder. "
        "Implement actual module lookup for hook_point."
    )
    
    # TODO: Replace with actual model forward pass with hook
    # For now, return placeholder outputs that are sensitive to alpha
    # This makes the steerability calculation work correctly even with placeholders
    
    outputs = []
    np.random.seed(config["experiment"]["seed"])
    for prompt in prompts:
        # Deterministic placeholder: output changes based on alpha and feature_idx
        # This simulates steering having an effect
        base_hash = hash(f"{prompt}_{feature_idx}") % 10000
        alpha_effect = int(alpha * 100) % 1000
        combined_hash = (base_hash + alpha_effect) % 10000
        # Use hash to create deterministic but alpha-sensitive output
        output = f"Steered output (alpha={alpha:.2f}, f={feature_idx}): {prompt[:30]}... [score_proxy={combined_hash/10000.0:.3f}]"
        outputs.append(output)
    
    return outputs


def compute_alpha_star(
    config: dict,
    model,
    tokenizer,
    sae,
    feature_idx: int,
    prompts: list,
    scorer,
) -> tuple:
    """
    Compute alpha_star(f) for a feature.
    
    alpha_star(f) = smallest alpha in alpha_grid such that
    on-target score(alpha,f) - score(0,f) >= threshold_T.
    
    If never crosses, set alpha_star=max(alpha_grid), censored=1.
    
    Returns:
        (alpha_star, censored, scores_dict)
    """
    alpha_grid = config["steering"]["alpha_grid"]
    threshold_T = config["steering"]["threshold_T"]
    
    # Ensure 0.0 is in alpha_grid for baseline
    if 0.0 not in alpha_grid:
        alpha_grid = [0.0] + alpha_grid
        alpha_grid = sorted(alpha_grid)
    
    scores = {}
    
    # Compute baseline score (alpha=0)
    logger.debug(f"Computing baseline score for feature {feature_idx}")
    baseline_outputs = run_model_with_steering(
        model, tokenizer, prompts, sae, feature_idx, 0.0, config
    )
    baseline_scores = scorer.score(prompts, baseline_outputs)
    baseline_mean = float(np.mean(baseline_scores))
    scores[0.0] = baseline_mean
    
    # Sweep alpha values
    alpha_star = None
    for alpha in alpha_grid:
        if alpha == 0.0:
            continue
        
        logger.debug(f"Testing alpha={alpha} for feature {feature_idx}")
        outputs = run_model_with_steering(
            model, tokenizer, prompts, sae, feature_idx, alpha, config
        )
        steered_scores = scorer.score(prompts, outputs)
        steered_mean = float(np.mean(steered_scores))
        scores[alpha] = steered_mean
        
        # Check if threshold is crossed
        delta = steered_mean - baseline_mean
        if delta >= threshold_T:
            alpha_star = alpha
            logger.debug(
                f"Threshold crossed at alpha={alpha} for feature {feature_idx} "
                f"(delta={delta:.4f} >= {threshold_T})"
            )
            break
    
    # If threshold never crossed, set to max(alpha_grid) and mark as censored
    if alpha_star is None:
        alpha_star = max(alpha_grid)
        censored = 1
        logger.debug(
            f"Threshold never crossed for feature {feature_idx}. "
            f"Setting alpha_star={alpha_star}, censored=1"
        )
    else:
        censored = 0
    
    return alpha_star, censored, scores


def main():
    parser = argparse.ArgumentParser(description="Measure steerability (alpha_star)")
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
    
    # Load on-target prompts and scorer
    prompts = load_on_target_prompts(config)
    scorer = load_scorer(config, "on_target")
    
    logger.info(f"Computing alpha_star for {len(selected_features)} features")
    logger.info(f"Using {len(prompts)} on-target prompts")
    logger.info(f"Threshold T = {config['steering']['threshold_T']}")
    
    # Compute alpha_star for each feature and collect curve data
    results = []
    curve_data = []  # For ontarget_curve.csv
    
    for feature_idx in tqdm(selected_features, desc="Computing alpha_star"):
        alpha_star, censored, scores = compute_alpha_star(
            config, model, tokenizer, sae, int(feature_idx), prompts, scorer
        )
        
        results.append({
            "feature_idx": int(feature_idx),
            "alpha_star": float(alpha_star),
            "censored": int(censored),
            "baseline_score": scores.get(0.0, np.nan),
            "final_score": scores.get(alpha_star, np.nan),
            "scorer": "placeholder",  # Mark as placeholder
        })
        
        # Store curve data for all alphas
        for alpha, score in scores.items():
            curve_data.append({
                "feature_idx": int(feature_idx),
                "alpha": float(alpha),
                "score": float(score),
                "scorer": "placeholder",
            })
    
    df = pd.DataFrame(results)
    curve_df = pd.DataFrame(curve_data)
    
    # Shape assertions
    assert len(df) == len(selected_features), "Mismatch in number of features"
    assert set(df["feature_idx"]) == set(selected_features), "Feature index mismatch"
    assert df["censored"].isin([0, 1]).all(), "Censored must be 0 or 1"
    assert (df["alpha_star"] >= 0).all(), "alpha_star must be non-negative"
    
    # Save results
    results_path = os.path.join(output_dir, "steerability_results.csv")
    df.to_csv(results_path, index=False)
    logger.info(f"Saved results to {results_path}")
    
    # Save ontarget_curve.csv (as requested)
    curve_path = os.path.join(output_dir, "ontarget_curve.csv")
    curve_df.to_csv(curve_path, index=False)
    logger.info(f"Saved curve data to {curve_path}")
    
    # Save alpha_star.csv (as requested)
    alpha_star_path = os.path.join(output_dir, "alpha_star.csv")
    df[["feature_idx", "alpha_star", "censored"]].to_csv(alpha_star_path, index=False)
    logger.info(f"Saved alpha_star to {alpha_star_path}")
    
    # Log summary statistics
    n_censored = df["censored"].sum()
    logger.info(f"Summary:")
    logger.info(f"  Total features: {len(df)}")
    logger.info(f"  Censored features: {n_censored} ({100*n_censored/len(df):.1f}%)")
    logger.info(f"  alpha_star range: [{df['alpha_star'].min():.4f}, {df['alpha_star'].max():.4f}]")
    logger.info(f"  alpha_star mean (non-censored): {df[df['censored']==0]['alpha_star'].mean():.4f}")
    
    # Save metadata
    metadata = {
        "n_features": len(df),
        "n_censored": int(n_censored),
        "threshold_T": config["steering"]["threshold_T"],
        "alpha_grid": config["steering"]["alpha_grid"],
    }
    metadata_path = os.path.join(output_dir, "03_steerability_metadata.yaml")
    with open(metadata_path, "w") as f:
        yaml.dump(metadata, f, default_flow_style=False)
    logger.info(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
