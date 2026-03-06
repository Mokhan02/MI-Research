"""Geometric metrics for SAE features."""

import torch
import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def compute_max_cosine_similarity(
    decoder_weights: torch.Tensor,
    feature_idx: int,
) -> float:
    """
    Compute maximum cosine similarity between feature f and any other feature.
    
    Measures proximity to nearest neighbors in representation space.
    
    Args:
        decoder_weights: (n_features, hidden_dim) tensor of decoder directions
        feature_idx: Index of feature f
    
    Returns:
        Maximum cosine similarity value
    """
    v_f = decoder_weights[feature_idx]  # (hidden_dim,)
    other_weights = torch.cat([
        decoder_weights[:feature_idx],
        decoder_weights[feature_idx + 1:]
    ], dim=0)  # (n_features - 1, hidden_dim)
    
    # Compute cosine similarities
    v_f_norm = torch.nn.functional.normalize(v_f.unsqueeze(0), dim=1)
    other_norm = torch.nn.functional.normalize(other_weights, dim=1)
    
    similarities = torch.mm(v_f_norm, other_norm.t()).squeeze(0)  # (n_features - 1,)
    max_sim = similarities.max().item()
    
    return max_sim


def compute_neighbor_density(
    decoder_weights: torch.Tensor,
    feature_idx: int,
    k: int = 50,
) -> float:
    """
    Compute average cosine similarity to k nearest neighbors.
    
    Captures local crowding in representation space.
    
    Args:
        decoder_weights: (n_features, hidden_dim) tensor
        feature_idx: Index of feature f
        k: Number of nearest neighbors to consider
    
    Returns:
        Mean cosine similarity to k nearest neighbors
    """
    v_f = decoder_weights[feature_idx]  # (hidden_dim,)
    other_weights = torch.cat([
        decoder_weights[:feature_idx],
        decoder_weights[feature_idx + 1:]
    ], dim=0)  # (n_features - 1, hidden_dim)
    
    # Compute all cosine similarities
    v_f_norm = torch.nn.functional.normalize(v_f.unsqueeze(0), dim=1)
    other_norm = torch.nn.functional.normalize(other_weights, dim=1)
    
    similarities = torch.mm(v_f_norm, other_norm.t()).squeeze(0)  # (n_features - 1,)
    
    # Get k nearest neighbors (highest similarities)
    k_actual = min(k, len(similarities))
    topk_similarities, _ = torch.topk(similarities, k_actual)
    mean_sim = topk_similarities.mean().item()
    
    return mean_sim


def compute_coactivation_correlation(
    activations: torch.Tensor,
    feature_idx: int,
    top_k: int = 50,
) -> float:
    """
    Compute mean activation correlation with top-k co-active features.
    
    Captures functional entanglement.
    
    Args:
        activations: (n_samples, n_features) tensor of feature activations
        feature_idx: Index of feature f
        top_k: Number of top co-active features to consider
    
    Returns:
        Mean correlation with top-k co-active features
    """
    if activations.shape[1] <= 1:
        return 0.0
    
    # Get activations for feature f
    a_f = activations[:, feature_idx]  # (n_samples,)
    
    # Get activations for all other features
    other_indices = torch.cat([
        torch.arange(feature_idx),
        torch.arange(feature_idx + 1, activations.shape[1])
    ])
    other_activations = activations[:, other_indices]  # (n_samples, n_features - 1)
    
    # Compute correlations
    a_f_centered = a_f - a_f.mean()
    other_centered = other_activations - other_activations.mean(dim=0, keepdim=True)
    
    # Pearson correlation
    numerator = (a_f_centered.unsqueeze(1) * other_centered).mean(dim=0)
    denominator = a_f_centered.std() * other_centered.std(dim=0)
    correlations = numerator / (denominator + 1e-8)  # (n_features - 1,)
    
    # Get top-k most correlated features
    k_actual = min(top_k, len(correlations))
    topk_correlations, _ = torch.topk(correlations.abs(), k_actual)
    mean_corr = topk_correlations.mean().item()
    
    return mean_corr

