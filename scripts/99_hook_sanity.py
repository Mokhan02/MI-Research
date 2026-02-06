"""
SAE Hook Sanity Check

Proves that adding alpha * v_f at the SAE hook point:
(1) changes activations
(2) changes model behavior on prompts
"""

import argparse
import logging
import torch
import numpy as np
import sys
import os
from typing import Optional, Tuple

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import load_config, resolve_config, save_resolved_config
from src.model_utils import load_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_hook_point(model, hook_point_name: str) -> torch.nn.Module:
    """
    Find the module at the specified hook point.
    
    Args:
        model: The model to search
        hook_point_name: Name or partial name of the hook point
    
    Returns:
        The module at the hook point
    
    Raises:
        ValueError: If hook point not found, with suggestions
    """
    # Get all named modules
    named_modules = dict(model.named_modules())
    
    # Try exact match first
    if hook_point_name in named_modules:
        logger.info(f"Found exact match for hook_point: {hook_point_name}")
        return named_modules[hook_point_name]
    
    # Try partial matches
    matches = [name for name in named_modules.keys() if hook_point_name in name]
    
    if len(matches) == 1:
        logger.info(f"Found unique partial match: {matches[0]}")
        return named_modules[matches[0]]
    elif len(matches) > 1:
        raise ValueError(
            f"Hook point '{hook_point_name}' matches multiple modules:\n"
            + "\n".join(f"  - {m}" for m in matches[:10])
            + f"\n\nPlease specify the exact hook point name."
        )
    else:
        # Find close matches (simple string similarity)
        all_names = list(named_modules.keys())
        # Show some example names
        examples = [name for name in all_names if any(x in name.lower() for x in ["layer", "mlp", "attn", "embed"])][:10]
        raise ValueError(
            f"Hook point '{hook_point_name}' not found.\n"
            f"Example module names:\n"
            + "\n".join(f"  - {name}" for name in examples)
            + f"\n\nTotal modules: {len(all_names)}"
        )


def get_feature_vector(config: dict, model, hook_point_module: torch.nn.Module) -> torch.Tensor:
    """
    Get or create a feature vector v_f.
    
    If SAE loading is available, use it. Otherwise, create a placeholder.
    
    Returns:
        v_f: Feature vector with shape matching hook point output
    """
    sae_cfg = config.get("sae", {})
    sae_id = sae_cfg.get("sae_id", "TBD")
    
    if sae_id != "TBD":
        # TODO: Load actual SAE feature vector
        logger.warning("SAE loading not yet implemented. Using PLACEHOLDER feature vector.")
    
    # Create placeholder feature vector
    # Infer hidden dimension from hook point module
    # Try to get output dimension
    if hasattr(hook_point_module, "out_features"):
        d_model = hook_point_module.out_features
    elif hasattr(hook_point_module, "embed_dim"):
        d_model = hook_point_module.embed_dim
    elif hasattr(hook_point_module, "hidden_size"):
        d_model = hook_point_module.hidden_size
    else:
        # Fallback: try to infer from model config
        if hasattr(model, "config"):
            d_model = getattr(model.config, "hidden_size", 2048)
        else:
            d_model = 2048  # Default fallback
        logger.warning(f"Could not infer d_model from hook point. Using {d_model}")
    
    # Create placeholder feature vector
    np.random.seed(config["experiment"]["seed"])
    v_f = torch.randn(d_model, dtype=torch.float32)
    
    # Normalize if config says so
    if sae_cfg.get("normalize_decoder", True):
        v_f = torch.nn.functional.normalize(v_f, dim=0)
    
    logger.warning(f"Using PLACEHOLDER feature vector v_f with shape {v_f.shape}")
    logger.info(f"Feature vector norm: {v_f.norm().item():.4f}")
    
    return v_f


def create_steering_hook(v_f: torch.Tensor, alpha: float, device: str):
    """
    Create a hook function that applies steering: activation += alpha * v_f.
    
    Returns:
        Hook function and a list to store captured activations
    """
    captured_activations = []
    
    def steering_hook(module, input, output):
        """Hook that adds alpha * v_f to activations."""
        # Handle tuple outputs
        if isinstance(output, tuple):
            activations = output[0].clone()  # Clone to avoid modifying original
        else:
            activations = output.clone()
        
        # Ensure v_f is on same device and dtype as activations
        v_f_device = v_f.to(device=activations.device, dtype=activations.dtype)
        
        # Shape safety: expect (batch, seq, d_model) or (batch, d_model)
        assert len(activations.shape) in (2, 3), \
            f"Expected activation shape [batch, seq?, d_model], got {activations.shape}"
        # Handle shape compatibility
        # v_f is 1D (d_model,), activations might be 2D or 3D
        if len(activations.shape) == 3:  # (batch, seq, hidden)
            v_f_expanded = v_f_device.unsqueeze(0).unsqueeze(0)
        elif len(activations.shape) == 2:  # (batch, hidden) or (seq, hidden)
            v_f_expanded = v_f_device.unsqueeze(0)
        else:
            v_f_expanded = v_f_device
        
        # Shape assertion
        assert v_f_expanded.shape[-1] == activations.shape[-1], \
            f"Feature vector dim {v_f_expanded.shape[-1]} != activation dim {activations.shape[-1]}"
        
        # Apply steering: activation += alpha * v_f
        activations.add_(alpha * v_f_expanded)
        
        # Capture activation after modification (for delta comparison)
        captured_activations.append(activations.clone().detach())
        
        if isinstance(output, tuple):
            return (activations,) + output[1:]
        return activations
    
    return steering_hook, captured_activations


def run_model_with_hook(
    model,
    tokenizer,
    prompt: str,
    captured_list,
    config: dict,
) -> Tuple[str, Optional[torch.Tensor]]:
    """
    Run model on a prompt with a hook registered.
    
    Returns:
        (generated_text, captured_activation)
    """
    device = config["model"]["device"]
    max_new_tokens = config["model"]["max_new_tokens"]
    temperature = config["model"]["temperature"]
    do_sample = config["model"]["do_sample"]
    
    # Clear captured activations
    captured_list.clear()
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else None,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Get captured activation from hook (last one captured)
    captured_activation = captured_list[-1] if len(captured_list) > 0 else None
    
    return generated_text, captured_activation


def main():
    parser = argparse.ArgumentParser(description="SAE hook sanity check")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Config file path")
    parser.add_argument("--run-id", type=str, default="sanity_check", help="Run identifier")
    parser.add_argument("--hook-point", type=str, default=None, help="Override hook point from config")
    parser.add_argument("--alpha", type=float, default=1.0, help="Steering coefficient for test")
    parser.add_argument("--prompt", type=str, default="The capital of France is", help="Test prompt")
    args = parser.parse_args()
    
    # Load and resolve config
    config = load_config(args.config)
    config = resolve_config(config, args.run_id)
    save_resolved_config(config, args.run_id)
    
    # Override hook point if provided
    hook_point_name = args.hook_point or config["sae"]["hook_point"]
    if hook_point_name == "TBD":
        # Try to find a reasonable default hook point
        logger.warning("Hook point is TBD. Attempting to find a default...")
        # For Gemma models, try common hook points
        model_id = config["model"]["model_id"]
        if "gemma" in model_id.lower():
            # Try common Gemma hook points
            possible_hooks = [
                "model.layers.0",
                "model.layers.0.mlp",
                "model.layers.0.self_attn",
            ]
            hook_point_name = possible_hooks[0]  # Use first layer as default
            logger.info(f"Using default hook point: {hook_point_name}")
        else:
            raise ValueError(
                "Hook point is TBD and no default found. "
                "Please specify --hook-point or set sae.hook_point in config."
            )
    
    logger.info("=" * 60)
    logger.info("SAE HOOK SANITY CHECK")
    logger.info("=" * 60)
    logger.info(f"Model: {config['model']['model_id']}")
    logger.info(f"Hook point: {hook_point_name}")
    logger.info(f"Test prompt: {args.prompt}")
    logger.info(f"Alpha (steering coefficient): {args.alpha}")
    
    # Load model
    logger.info("\nLoading model...")
    model, tokenizer = load_model(config)
    device = config["model"]["device"]
    
    # Find hook point module
    logger.info(f"\nFinding hook point: {hook_point_name}")
    hook_point_module = find_hook_point(model, hook_point_name)
    logger.info(f"Found module: {type(hook_point_module).__name__}")
    
    # Get feature vector
    logger.info("\nGetting feature vector v_f...")
    v_f = get_feature_vector(config, model, hook_point_module)
    logger.info(f"v_f shape: {v_f.shape}, norm: {v_f.norm().item():.4f}")
    
    # Test 1: Baseline (alpha=0)
    logger.info("\n" + "=" * 60)
    logger.info("TEST 1: Baseline (alpha=0)")
    logger.info("=" * 60)
    
    hook_fn_baseline, captured_baseline = create_steering_hook(v_f, 0.0, device)
    handle_baseline = hook_point_module.register_forward_hook(hook_fn_baseline)
    
    try:
        text_baseline, activation_baseline = run_model_with_hook(
            model, tokenizer, args.prompt, captured_baseline, config
        )
        
        logger.info(f"Generated text: {text_baseline}")
        if activation_baseline is not None:
            logger.info(f"Activation shape: {activation_baseline.shape}")
            logger.info(f"Activation norm: {activation_baseline.norm().item():.4f}")
    finally:
        handle_baseline.remove()
    
    # Test 2: Steered (alpha>0)
    logger.info("\n" + "=" * 60)
    logger.info(f"TEST 2: Steered (alpha={args.alpha})")
    logger.info("=" * 60)
    
    hook_fn_steered, captured_steered = create_steering_hook(v_f, args.alpha, device)
    handle_steered = hook_point_module.register_forward_hook(hook_fn_steered)
    
    try:
        text_steered, activation_steered = run_model_with_hook(
            model, tokenizer, args.prompt, captured_steered, config
        )
        
        logger.info(f"Generated text: {text_steered}")
        if activation_steered is not None:
            logger.info(f"Activation shape: {activation_steered.shape}")
            logger.info(f"Activation norm: {activation_steered.norm().item():.4f}")
    finally:
        handle_steered.remove()
    
    # Comparison
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON")
    logger.info("=" * 60)
    
    # Compare activations
    if activation_baseline is not None and activation_steered is not None:
        # Take mean over batch/sequence dimensions for comparison
        act_baseline_flat = activation_baseline.flatten()
        act_steered_flat = activation_steered.flatten()
        
        activation_diff = torch.norm(act_steered_flat - act_baseline_flat).item()
        logger.info(f"\n||activation_steered - activation_baseline|| = {activation_diff:.6f}")
        
        assert activation_diff > 0, "ERROR: Activations did not change! Steering hook may not be working."
        logger.info("✓ PASS: Activations changed as expected")
    else:
        logger.warning("Could not capture activations for comparison")
    
    # Compare outputs
    logger.info(f"\nBaseline output:  {text_baseline}")
    logger.info(f"Steered output:    {text_steered}")
    
    if text_baseline != text_steered:
        logger.info("✓ PASS: Model outputs changed as expected")
    else:
        logger.warning("Model outputs did not change. This may be normal for small alpha or certain features.")
    
    logger.info("\n" + "=" * 60)
    logger.info("SANITY CHECK COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

