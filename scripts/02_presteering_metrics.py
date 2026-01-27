"""
Pre-steering metrics stage.

Computes feature-level geometry and activation metrics
before any steering or intervention is applied.

These metrics are used as predictors later and must be
measured on the unmodified model.
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
from src.model_utils import load_model, load_sae
from src.geometry import (
    compute_max_cosine_similarity,
    compute_neighbor_density,
    compute_coactivation_correlation,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_reference_corpus(config: dict):
    """
    Load reference corpus for computing pre-steering metrics.
    
    Returns list of text strings.
    """
    # Placeholder: in practice, this would load from data/prompts/
    # For now, return a small placeholder corpus
    data_dir = config.get("data", {}).get("data_dir", "data")
    prompts_dir = os.path.join(data_dir, "prompts")
    
    # Try to load reference corpus
    reference_path = os.path.join(prompts_dir, "reference_corpus.txt")
    if os.path.exists(reference_path):
        with open(reference_path, "r") as f:
            texts = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(texts)} texts from {reference_path}")
        return texts
    
    # Fallback: generate placeholder texts
    logger.warning(
        f"Reference corpus not found at {reference_path}. "
        "Using placeholder corpus. This is for testing only."
    )
    texts = [f"Sample text {i} for reference corpus." for i in range(100)]
    return texts


def get_feature_activations(model, tokenizer, sae, texts: list, hook_point: str, device: str):
    """
    Get feature activations for all features on reference corpus.
    
    Returns:
        activations: (n_samples, n_features) tensor
    """
    # This is a placeholder implementation
    # In practice, this would:
    # 1. Hook into the model at hook_point
    # 2. Run forward passes on texts
    # 3. Extract SAE feature activations
    
    logger.warning(
        "get_feature_activations is a placeholder. "
        "Implement actual activation extraction from SAE."
    )
    
    # Placeholder: return random activations
    n_samples = len(texts)
    n_features = sae.n_features
    activations = torch.randn(n_samples, n_features, device=device)
    activations = torch.relu(activations)  # ReLU to simulate sparse activations
    
    return activations


def compute_presteering_metrics(
    config: dict,
    sae,
    selected_features: np.ndarray,
) -> pd.DataFrame:
    """
    Compute pre-steering geometric and usage metrics for selected features.
    
    Returns DataFrame with one row per feature.
    """
    # Get decoder weights
    decoder_weights = torch.stack([
        sae.get_decoder_direction(i) for i in range(sae.n_features)
    ])  # (n_features, hidden_dim)
    
    # Load reference corpus and get activations
    # In placeholder mode, skip model loading and use placeholder activations
    texts = load_reference_corpus(config)
    device = config["model"]["device"]
    
    # Try to get activations, but use placeholder if model loading fails
    try:
        model, tokenizer = load_model(config)
        activations = get_feature_activations(
            model, tokenizer, sae, texts, sae.hook_point, device
        )  # (n_samples, n_features)
    except (OSError, RuntimeError) as e:
        logger.warning(f"Model loading failed: {e}")
        logger.warning("Using placeholder activations for co-activation correlation")
        # Generate placeholder activations deterministically
        n_samples = len(texts)
        n_features = sae.n_features
        np.random.seed(config["experiment"]["seed"])
        activations = torch.from_numpy(
            np.random.randn(n_samples, n_features).astype(np.float32)
        )
        activations = torch.relu(activations)  # ReLU to simulate sparse activations
        activations = activations.to(device)
    
    logger.info(f"Computing pre-steering metrics for {len(selected_features)} features")
    
    metrics = []
    for feature_idx in tqdm(selected_features, desc="Computing metrics"):
        # Geometric metrics
        max_cos_sim = compute_max_cosine_similarity(decoder_weights, feature_idx)
        neighbor_density = compute_neighbor_density(decoder_weights, feature_idx, k=50)
        
        # Usage metrics (co-activation correlation)
        coactivation_corr = compute_coactivation_correlation(
            activations, feature_idx, top_k=50
        )
        
        # Optional: activation frequency
        feature_activations = activations[:, feature_idx]
        activation_freq = (feature_activations > 0).float().mean().item()
        mean_activation = feature_activations.mean().item()
        
        metrics.append({
            "feature_idx": int(feature_idx),
            "max_cosine_similarity": float(max_cos_sim),
            "neighbor_density": float(neighbor_density),
            "coactivation_correlation": float(coactivation_corr),
            "activation_frequency": float(activation_freq),
            "mean_activation": float(mean_activation),
        })
    
    df = pd.DataFrame(metrics)
    
    # Shape assertions
    assert len(df) == len(selected_features), "Mismatch in number of features"
    assert set(df["feature_idx"]) == set(selected_features), "Feature index mismatch"
    
    logger.info(f"Computed metrics for {len(df)} features")
    logger.info(f"Metric ranges:")
    logger.info(f"  max_cosine_similarity: [{df['max_cosine_similarity'].min():.4f}, {df['max_cosine_similarity'].max():.4f}]")
    logger.info(f"  neighbor_density: [{df['neighbor_density'].min():.4f}, {df['neighbor_density'].max():.4f}]")
    logger.info(f"  coactivation_correlation: [{df['coactivation_correlation'].min():.4f}, {df['coactivation_correlation'].max():.4f}]")
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Compute pre-steering metrics")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Config file path")
    parser.add_argument("--run-id", type=str, required=True, help="Unique run identifier")
    args = parser.parse_args()
    
    # Load and resolve config
    config = load_config(args.config)
    config = resolve_config(config, args.run_id)
    save_resolved_config(config, args.run_id)
    
    logger.warning("=" * 60)
    logger.warning("PLACEHOLDER MODE: Using placeholder geometry metrics")
    logger.warning("=" * 60)
    
    # Load selected features from previous stage
    output_dir = config["experiment"]["output_dir"]
    features_path = os.path.join(output_dir, "selected_features.npy")
    if not os.path.exists(features_path):
        raise FileNotFoundError(
            f"Selected features not found at {features_path}. "
            "Run 01_choose_features.py first."
        )
    selected_features = np.load(features_path)
    logger.info(f"Loaded {len(selected_features)} selected features")
    
    # Load SAE or use placeholder
    try:
        sae = load_sae(config)
    except (ValueError, NotImplementedError) as e:
        logger.warning(f"SAE loading failed: {e}")
        logger.warning("Using placeholder SAE for geometry metrics")
        # Create placeholder SAE with random decoder weights
        class PlaceholderSAE:
            def __init__(self, n_features, hidden_dim=2048):
                self.n_features = n_features
                self.hook_point = config.get("sae", {}).get("hook_point", "TBD")
                # Generate deterministic placeholder decoder weights
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
        logger.warning(f"Created placeholder SAE with {n_features_total} features")
    
    # Compute metrics
    metrics_df = compute_presteering_metrics(config, sae, selected_features)
    
    # Add placeholder marker
    metrics_df["scorer"] = "placeholder"
    
    # Save metrics
    metrics_path = os.path.join(output_dir, "presteering_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Saved metrics to {metrics_path}")
    
    # Also save as pre_metrics.csv (as requested)
    pre_metrics_path = os.path.join(output_dir, "pre_metrics.csv")
    metrics_df.to_csv(pre_metrics_path, index=False)
    logger.info(f"Saved to {pre_metrics_path}")
    
    # Save metadata
    metadata = {
        "n_features": len(metrics_df),
        "metrics_computed": list(metrics_df.columns),
    }
    metadata_path = os.path.join(output_dir, "02_presteering_metadata.yaml")
    with open(metadata_path, "w") as f:
        yaml.dump(metadata, f, default_flow_style=False)
    logger.info(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    main()
