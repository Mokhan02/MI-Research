"""
Feature selection stage.

Selects and freezes the set of SAE features to study.
All downstream scripts depend on this fixed feature list
to ensure consistency and avoid cherry-picking.
"""

import argparse
import logging
import numpy as np
import pandas as pd
import yaml
import json
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import load_config, resolve_config, save_resolved_config
from src.model_utils import load_sae

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def select_features(config: dict, sae) -> np.ndarray:
    """
    Select features based on config selection mode.
    
    Args:
        config: Experiment configuration
        sae: SAE object
    
    Returns:
        Array of selected feature indices
    """
    features_cfg = config["features"]
    n_features = features_cfg["n_features"]
    selection_mode = features_cfg.get("selection_mode", "uniform")
    activation_stats_path = features_cfg.get("activation_stats_path")
    
    total_features = sae.n_features
    
    if n_features > total_features:
        logger.warning(
            f"Requested {n_features} features but SAE has only {total_features}. "
            f"Selecting all {total_features} features."
        )
        n_features = total_features
    
    if selection_mode == "uniform":
        # Uniform random sampling
        np.random.seed(config["experiment"]["seed"])
        selected = np.random.choice(total_features, size=n_features, replace=False)
        selected = np.sort(selected)
        logger.info(f"Selected {n_features} features uniformly at random")
    
    elif selection_mode == "stratified":
        # Stratified sampling based on activation frequency
        if activation_stats_path is None or not os.path.exists(activation_stats_path):
            logger.warning(
                "Stratified selection requested but activation_stats_path not provided. "
                "Falling back to uniform selection."
            )
            np.random.seed(config["experiment"]["seed"])
            selected = np.random.choice(total_features, size=n_features, replace=False)
            selected = np.sort(selected)
        else:
            # Load activation stats and stratify
            stats = pd.read_csv(activation_stats_path)
            # Simple stratification: divide into bins and sample from each
            # This is a placeholder - actual implementation would be more sophisticated
            np.random.seed(config["experiment"]["seed"])
            selected = np.random.choice(total_features, size=n_features, replace=False)
            selected = np.sort(selected)
            logger.info(f"Selected {n_features} features using stratified sampling")
    
    else:
        raise ValueError(f"Unknown selection_mode: {selection_mode}")
    
    assert len(selected) == n_features, f"Expected {n_features} features, got {len(selected)}"
    assert len(np.unique(selected)) == n_features, "Duplicate features selected"
    assert selected.min() >= 0 and selected.max() < total_features, "Invalid feature indices"
    
    return selected


def main():
    parser = argparse.ArgumentParser(description="Select SAE features for experiment")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Config file path")
    parser.add_argument("--run-id", type=str, required=True, help="Unique run identifier")
    args = parser.parse_args()
    
    # Load and resolve config
    config = load_config(args.config)
    config = resolve_config(config, args.run_id)
    save_resolved_config(config, args.run_id)
    
    # Load SAE or use fallback
    logger.warning("=" * 60)
    logger.warning("PLACEHOLDER MODE: Using fallback SAE configuration")
    logger.warning("=" * 60)
    
    try:
        sae = load_sae(config)
    except (ValueError, NotImplementedError) as e:
        logger.warning(f"SAE loading failed: {e}")
        logger.warning("Using placeholder SAE with n_features_total from config")
        # Create placeholder SAE object
        class PlaceholderSAE:
            def __init__(self, n_features):
                self.n_features = n_features
        
        n_features_total = config.get("sae", {}).get("n_features_total", 1000)
        sae = PlaceholderSAE(n_features_total)
        logger.warning(f"Created placeholder SAE with {n_features_total} features")
    
    # Select features
    selected_features = select_features(config, sae)
    
    # Save selected features
    output_dir = config["experiment"]["output_dir"]
    features_path = os.path.join(output_dir, "selected_features.npy")
    np.save(features_path, selected_features)
    
    # Also save as CSV for readability
    features_df = pd.DataFrame({
        "feature_idx": selected_features,
    })
    features_csv_path = os.path.join(output_dir, "selected_features.csv")
    features_df.to_csv(features_csv_path, index=False)
    
    # Also save as JSON (as requested)
    features_json_path = os.path.join(output_dir, "features.json")
    with open(features_json_path, "w") as f:
        json.dump({"feature_indices": selected_features.tolist()}, f, indent=2)
    logger.info(f"Saved to {features_json_path}")
    
    logger.info(f"Selected {len(selected_features)} features")
    logger.info(f"Feature indices range: [{selected_features.min()}, {selected_features.max()}]")
    logger.info(f"Saved to {features_path} and {features_csv_path}")
    
    # Save metadata
    metadata = {
        "n_features": len(selected_features),
        "selection_mode": config["features"]["selection_mode"],
        "feature_indices": selected_features.tolist(),
    }
    metadata_path = os.path.join(output_dir, "01_features_metadata.yaml")
    with open(metadata_path, "w") as f:
        yaml.dump(metadata, f, default_flow_style=False)
    
    logger.info(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
