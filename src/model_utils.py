"""Model and SAE loading utilities."""

import torch
import numpy as np
from typing import Dict, Any, Optional, Callable
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def load_model(config: Dict[str, Any]):
    """Load model and tokenizer from config."""
    model_cfg = config["model"]
    model_id = model_cfg["model_id"]
    device = model_cfg.get("device", "cpu")
    dtype_str = model_cfg.get("dtype", "float32")
    
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(dtype_str, torch.float32)
    
    logger.info(f"Loading model {model_id} on {device} with dtype {dtype_str}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device if device != "cpu" else None,
    )
    if device == "cpu":
        model = model.to(device)
    
    model.eval()
    
    logger.info("Model loaded successfully")
    return model, tokenizer


def load_sae(config: Dict[str, Any]):
    """
    Load SAE from config.
    
    For now, this is a placeholder that raises NotImplementedError.
    In practice, this would load from neuronpedia or other SAE sources.
    """
    sae_cfg = config["sae"]
    source = sae_cfg.get("source", "neuronpedia")
    sae_id = sae_cfg.get("sae_id", "TBD")
    hook_point = sae_cfg.get("hook_point", "TBD")
    normalize_decoder = sae_cfg.get("normalize_decoder", True)
    
    if sae_id == "TBD" or hook_point == "TBD":
        raise ValueError(
            "SAE ID and hook_point must be specified in config. "
            "This is a placeholder implementation."
        )
    
    logger.warning(
        "SAE loading is a placeholder. Implement actual SAE loading "
        f"for source={source}, sae_id={sae_id}"
    )
    
    # Placeholder structure - would be replaced with actual SAE loading
    class PlaceholderSAE:
        def __init__(self):
            self.n_features = 1000  # Placeholder
            self.hook_point = hook_point
            self.normalize_decoder = normalize_decoder
            # Placeholder decoder directions
            self.decoder_weights = torch.randn(1000, 2048)  # Placeholder dimensions
            if normalize_decoder:
                self.decoder_weights = torch.nn.functional.normalize(
                    self.decoder_weights, dim=-1
                )
        
        def get_decoder_direction(self, feature_idx: int) -> torch.Tensor:
            """Get decoder direction vector v_f for feature f."""
            return self.decoder_weights[feature_idx]
    
    sae = PlaceholderSAE()
    logger.info(f"Loaded SAE with {sae.n_features} features at hook_point={hook_point}")
    return sae


def create_steering_hook(
    sae,
    feature_idx: int,
    alpha: float,
    hook_point: str,
) -> Callable:
    """
    Create a hook function that applies steering intervention.
    
    Steering: activation += alpha * v_f at the SAE's hook point.
    
    Args:
        sae: SAE object with get_decoder_direction method
        feature_idx: Index of feature to steer
        alpha: Steering coefficient
        hook_point: Layer name where hook is applied
    
    Returns:
        Hook function that modifies activations in-place
    """
    v_f = sae.get_decoder_direction(feature_idx)
    
    def steering_hook(module, input, output):
        """Hook that adds alpha * v_f to activations."""
        if isinstance(output, tuple):
            # Handle tuple outputs (e.g., from attention layers)
            activations = output[0]
        else:
            activations = output
        
        # Ensure v_f is on same device and has compatible shape
        v_f_device = v_f.to(activations.device)
        
        # Handle shape compatibility
        # v_f might be 1D, activations might be 2D or 3D
        if len(activations.shape) == 3:  # (batch, seq, hidden)
            v_f_expanded = v_f_device.unsqueeze(0).unsqueeze(0)
        elif len(activations.shape) == 2:  # (batch, hidden) or (seq, hidden)
            v_f_expanded = v_f_device.unsqueeze(0)
        else:
            v_f_expanded = v_f_device
        
        # Apply steering: activation += alpha * v_f
        activations.add_(alpha * v_f_expanded)
        
        if isinstance(output, tuple):
            return (activations,) + output[1:]
        return activations
    
    return steering_hook

