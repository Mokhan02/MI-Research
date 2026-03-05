"""
Phase 1 Target Verification Script

Verifies that the locked target (Gemma-2-2b + GemmaScope SAE) is correctly configured.

Run this BEFORE running experiments to confirm:
1. Model loads and hook point exists
2. Hook point naming matches SAE metadata (not guessed)
3. Decoder shape is correct (n_features x d_model)

All values (hook_point, d_model, n_features_total) are read from the target config.
"""

import argparse
import logging
import sys
import os
import yaml

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from transformers import AutoModelForCausalLM, AutoConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default path to Phase 1 target config (relative to repo root)
DEFAULT_TARGET_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "configs", "targets", "gemma2_2b_gemmascope_res16k.yaml"
)


def load_target_config(config_path: str) -> dict:
    """Load target config YAML."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def verify_model_modules(model_id: str, target_config: dict):
    """
    Verify 1: Model loads and list all modules.

    Checks if hook point from config exists in HuggingFace module names.
    Uses target_config["sae"]["hook_point"] and model.config.hidden_size (d_model).
    """
    hook_point = target_config["sae"]["hook_point"]
    layer_idx = target_config["sae"].get("layer_idx", 20)

    logger.info("=" * 60)
    logger.info("VERIFICATION 1: Model Loading and Module Listing")
    logger.info("=" * 60)

    logger.info(f"Loading model: {model_id}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype="auto"
        )
        logger.info("✓ Model loaded successfully")
    except Exception as e:
        logger.error(f"✗ Model loading failed: {e}")
        return False

    # Get model config (d_model from actual model)
    hf_config = AutoConfig.from_pretrained(model_id)
    d_model = hf_config.hidden_size
    logger.info(f"\nModel config (from loaded model):")
    logger.info(f"  hidden_size (d_model): {d_model}")
    logger.info(f"  num_hidden_layers: {hf_config.num_hidden_layers}")
    logger.info(f"  vocab_size: {hf_config.vocab_size}")

    # List all module names
    named_modules = dict(model.named_modules())
    logger.info(f"\nTotal modules: {len(named_modules)}")

    # Check if config hook point exists in model
    logger.info("\n" + "=" * 60)
    logger.info(f"Config hook_point: '{hook_point}'")
    if hook_point in named_modules:
        logger.info(f"  ✓ Hook point EXISTS in model modules")
    else:
        logger.warning(f"  ✗ Hook point '{hook_point}' NOT found in model modules")
        logger.warning("  You may need a hook resolver to map this logical name to an actual HF module output")

    # Check for common patterns (using layer_idx from config)
    patterns = [
        f"layers.{layer_idx}",
        f"model.layers.{layer_idx}",
        f"blocks.{layer_idx}",
        "attn",
        "mlp",
        "resid",
    ]
    logger.info(f"\nModule name patterns (layer_idx={layer_idx} from config):")
    for pattern in patterns:
        hits = [n for n in named_modules.keys() if pattern in n.lower()]
        if hits:
            logger.info(f"\n  Pattern '{pattern}': {len(hits)} matches")
            for hit in hits[:15]:
                logger.info(f"    - {hit}")
            if len(hits) > 15:
                logger.info(f"    ... and {len(hits) - 15} more")
        else:
            logger.info(f"\n  Pattern '{pattern}': No matches")

    # Check specifically for layer_idx
    logger.info("\n" + "=" * 60)
    logger.info(f"Layer {layer_idx} module names (from config):")
    layer_modules = [n for n in named_modules.keys() if str(layer_idx) in n and ("layer" in n.lower() or "block" in n.lower())]
    if layer_modules:
        for mod in layer_modules[:20]:
            logger.info(f"  - {mod}")
    else:
        logger.warning(f"  No layer {layer_idx} modules found with 'layer' or 'block' in name")
        logger.warning("  Check the pattern list above for actual naming convention")

    # Check for TransformerLens-style hook names (any hook_resid_post)
    logger.info("\n" + "=" * 60)
    logger.info("Checking for TransformerLens-style hook names:")
    tl_hooks = [n for n in named_modules.keys() if "hook_resid_post" in n]
    if tl_hooks:
        logger.info(f"  Found {len(tl_hooks)} hook_resid_post modules:")
        for hook in tl_hooks[:10]:
            logger.info(f"    - {hook}")
    else:
        logger.warning("  ✗ No TransformerLens-style hook names found in model")
        logger.warning(f"  Config hook_point '{hook_point}' is likely a logical name (e.g. TransformerLens)")
        logger.warning("  You will need a hook resolver to map logical hook points to actual module outputs")

    return True


def verify_hook_point_naming(target_config: dict):
    """
    Verification 2: Hook point naming from SAE metadata.

    DO NOT GUESS. This must come from SAE release metadata.
    Reports hook_point from target_config["sae"]["hook_point"].
    """
    hook_point = target_config["sae"]["hook_point"]
    logger.info("\n" + "=" * 60)
    logger.info("VERIFICATION 2: Hook Point Naming from SAE Metadata")
    logger.info("=" * 60)

    logger.warning("⚠️  MANUAL VERIFICATION REQUIRED")
    logger.info("\nYou must check SAE metadata to determine:")
    logger.info("  1. Was the SAE trained with TransformerLens naming?")
    logger.info("  2. Or was it trained on HuggingFace module outputs?")
    logger.info("  3. What is the exact hook point name used during training?")
    logger.info("\nSources to check:")
    logger.info("  - Neuronpedia SAE page metadata")
    logger.info("  - GemmaScope tooling/repository documentation")
    logger.info("  - SAE checkpoint README or config files")
    logger.info(f"\nCurrent hook_point from config: '{hook_point}'")
    logger.info("If TransformerLens-style, you will need a hook resolver to map to HF module.")
    logger.info("\nOnce confirmed, update configs/targets/gemma2_2b_gemmascope_res16k.yaml")


def verify_decoder_shape(target_config: dict, d_model: int | None):
    """
    Verification 3: Decoder shape confirmation.

    Must confirm decoder is (n_features_total, d_model) or (d_model, n_features_total).
    Uses n_features_total from config; d_model from model.config.hidden_size or config.architecture.d_model.
    """
    n_features_total = target_config["sae"]["n_features_total"]
    logger.info("\n" + "=" * 60)
    logger.info("VERIFICATION 3: Decoder Shape Confirmation")
    logger.info("=" * 60)

    logger.warning("⚠️  MANUAL VERIFICATION REQUIRED")
    logger.info("\nOnce SAE weights are loaded, you MUST confirm:")
    if d_model is not None:
        logger.info(f"  1. Decoder shape is ({n_features_total}, {d_model}) (n_features x d_model)")
        logger.info(f"     OR ({d_model}, {n_features_total}) if transposed (d_model x n_features)")
        logger.info(f"  2. d_model dimension == {d_model} (matches model.config.hidden_size)")
    else:
        logger.info(f"  1. Decoder shape is (n_features_total, d_model) OR (d_model, n_features_total)")
        logger.info(f"     with n_features_total = {n_features_total} (from config), d_model = model.config.hidden_size")
        logger.info("  2. d_model dimension must match model.config.hidden_size")
    logger.info(f"  3. n_features == {n_features_total} (matches SAE dictionary size)")
    logger.info("\nNo confirmation = no experiment")
    logger.info("\nAdd verification code to load_sae() in src/model_utils.py:")
    if d_model is not None:
        logger.info(f"  assert decoder.shape == ({n_features_total}, {d_model}) or decoder.shape == ({d_model}, {n_features_total})")
        logger.info(f"  assert decoder.shape[-1] == {d_model}  # d_model dimension")
    else:
        logger.info(f"  n_features_total = config['sae']['n_features_total']  # {n_features_total}")
        logger.info("  d_model = model.config.hidden_size  # from loaded model")
        logger.info("  assert decoder.shape == (n_features_total, d_model) or decoder.shape == (d_model, n_features_total)")
        logger.info("  assert decoder.shape[-1] == d_model  # d_model dimension")


def main():
    parser = argparse.ArgumentParser(description="Verify Phase 1 target configuration")
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Model ID to verify (default: from target config)"
    )
    parser.add_argument(
        "--skip-model-load",
        action="store_true",
        help="Skip model loading (if already verified)"
    )
    args = parser.parse_args()

    # Load target config first (fixed path)
    config_path = os.path.abspath(DEFAULT_TARGET_CONFIG_PATH)
    if not os.path.isfile(config_path):
        logger.error(f"Target config not found: {config_path}")
        return 1
    target_config = load_target_config(config_path)
    model_id = args.model_id or target_config["model"]["model_id"]
    hook_point = target_config["sae"]["hook_point"]
    n_features_total = target_config["sae"]["n_features_total"]
    d_model_from_config = target_config.get("architecture", {}).get("d_model")

    logger.info("Phase 1 Target Verification")
    logger.info("=" * 60)
    logger.info(f"Target config: {config_path}")
    logger.info(f"Model (from config): {target_config['model']['model_id']}")
    logger.info(f"Hook point (from config): '{hook_point}'")
    logger.info(f"n_features_total (from config): {n_features_total}")
    logger.info(f"d_model (from config): {d_model_from_config}")

    # d_model: from loaded model when we run Verification 1, else from config
    d_model = d_model_from_config

    # Verification 1: Model loading
    if not args.skip_model_load:
        success = verify_model_modules(model_id, target_config)
        if not success:
            logger.error("\n✗ Verification 1 FAILED - Model loading failed")
            return 1
        # Use actual d_model from model for Verification 3
        hf_config = AutoConfig.from_pretrained(model_id)
        d_model = hf_config.hidden_size
        logger.info("\n✓ Verification 1 PASSED - Model loads and modules listed")
    else:
        logger.info("\n⏭️  Verification 1 SKIPPED (--skip-model-load)")
        if d_model is None:
            logger.warning("  d_model not in config.architecture; decoder shape instructions may be incomplete.")

    # Verification 2: Hook point naming (manual)
    verify_hook_point_naming(target_config)

    # Verification 3: Decoder shape (manual) - use d_model from model or config
    verify_decoder_shape(target_config, d_model if not args.skip_model_load else d_model_from_config)

    logger.info("\n" + "=" * 60)
    logger.info("VERIFICATION SUMMARY")
    logger.info("=" * 60)
    logger.info("✓ Verification 1: Model loading (automated)")
    logger.info("⚠️  Verification 2: Hook point naming (MANUAL - check SAE metadata)")
    logger.info("⚠️  Verification 3: Decoder shape (MANUAL - verify after loading SAE)")
    logger.info("\nTarget is NOT fully locked until all 3 are confirmed.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
