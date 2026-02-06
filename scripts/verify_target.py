"""
Phase 1 Target Verification Script

Verifies that the locked target (Gemma-2-2b + GemmaScope SAE) is correctly configured.

Run this BEFORE running experiments to confirm:
1. Model loads and hook point exists
2. Hook point naming matches SAE metadata (not guessed)
3. Decoder shape is correct (n_features x d_model)
"""

import argparse
import logging
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from transformers import AutoModelForCausalLM, AutoConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def verify_model_modules(model_id: str):
    """
    Verify 1: Model loads and list all modules.
    
    Checks if hook point naming exists in HuggingFace module names.
    Note: TransformerLens-style names (blocks.20.hook_resid_post) won't exist
    in HF models - you'll need a hook resolver.
    """
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
    
    # Get model config
    config = AutoConfig.from_pretrained(model_id)
    logger.info(f"\nModel config:")
    logger.info(f"  hidden_size (d_model): {config.hidden_size}")
    logger.info(f"  num_hidden_layers: {config.num_hidden_layers}")
    logger.info(f"  vocab_size: {config.vocab_size}")
    
    # List all module names
    named_modules = dict(model.named_modules())
    logger.info(f"\nTotal modules: {len(named_modules)}")
    
    # Check for common patterns
    patterns = [
        "layers.20",
        "model.layers.20",
        "blocks.20",
        "attn",
        "mlp",
        "resid",
    ]
    
    logger.info("\nModule name patterns:")
    for pattern in patterns:
        hits = [n for n in named_modules.keys() if pattern in n.lower()]
        if hits:
            logger.info(f"\n  Pattern '{pattern}': {len(hits)} matches")
            for hit in hits[:15]:  # Show first 15
                logger.info(f"    - {hit}")
            if len(hits) > 15:
                logger.info(f"    ... and {len(hits) - 15} more")
        else:
            logger.info(f"\n  Pattern '{pattern}': No matches")
    
    # Check specifically for layer 20
    logger.info("\n" + "=" * 60)
    logger.info("Layer 20 module names:")
    layer_20_modules = [n for n in named_modules.keys() if "20" in n and ("layer" in n.lower() or "block" in n.lower())]
    if layer_20_modules:
        for mod in layer_20_modules[:20]:
            logger.info(f"  - {mod}")
    else:
        logger.warning("  No layer 20 modules found with 'layer' or 'block' in name")
        logger.warning("  Check the pattern list above for actual naming convention")
    
    # Check for TransformerLens-style hook names
    logger.info("\n" + "=" * 60)
    logger.info("Checking for TransformerLens-style hook names:")
    tl_hooks = [n for n in named_modules.keys() if "hook_resid_post" in n]
    if tl_hooks:
        logger.info(f"  Found {len(tl_hooks)} hook_resid_post modules:")
        for hook in tl_hooks[:10]:
            logger.info(f"    - {hook}")
    else:
        logger.warning("  ✗ No TransformerLens-style hook names found")
        logger.warning("  This means 'blocks.20.hook_resid_post' is NOT a native HF module name")
        logger.warning("  You will need a hook resolver to map logical hook points to actual module outputs")
    
    return True


def verify_hook_point_naming():
    """
    Verification 2: Hook point naming from SAE metadata.
    
    DO NOT GUESS. This must come from SAE release metadata.
    """
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
    logger.info("\nCurrent provisional hook_point: 'blocks.20.hook_resid_post'")
    logger.info("This is TransformerLens-style and likely needs mapping to HF module.")
    logger.info("\nOnce confirmed, update configs/targets/gemma2_2b_gemmascope_res16k.yaml")


def verify_decoder_shape():
    """
    Verification 3: Decoder shape confirmation.
    
    Must confirm decoder is [n_features, d_model] (or transposed).
    """
    logger.info("\n" + "=" * 60)
    logger.info("VERIFICATION 3: Decoder Shape Confirmation")
    logger.info("=" * 60)
    
    logger.warning("⚠️  MANUAL VERIFICATION REQUIRED")
    logger.info("\nOnce SAE weights are loaded, you MUST confirm:")
    logger.info("  1. Decoder shape is [16384, 2048] (n_features x d_model)")
    logger.info("     OR [2048, 16384] if transposed (d_model x n_features)")
    logger.info("  2. d_model dimension == 2048 (matches model.config.hidden_size)")
    logger.info("  3. n_features == 16384 (matches SAE dictionary size)")
    logger.info("\nNo confirmation = no experiment")
    logger.info("\nAdd verification code to load_sae() in src/model_utils.py:")
    logger.info("  assert decoder.shape == (16384, 2048) or decoder.shape == (2048, 16384)")
    logger.info("  assert decoder.shape[-1] == 2048  # d_model dimension")


def main():
    parser = argparse.ArgumentParser(description="Verify Phase 1 target configuration")
    parser.add_argument(
        "--model-id",
        type=str,
        default="google/gemma-2-2b",
        help="Model ID to verify"
    )
    parser.add_argument(
        "--skip-model-load",
        action="store_true",
        help="Skip model loading (if already verified)"
    )
    args = parser.parse_args()
    
    logger.info("Phase 1 Target Verification")
    logger.info("=" * 60)
    logger.info(f"Target: Gemma-2-2b + GemmaScope SAE (gemmascope-res-16k)")
    logger.info(f"Config: configs/targets/gemma2_2b_gemmascope_res16k.yaml")
    
    # Verification 1: Model loading
    if not args.skip_model_load:
        success = verify_model_modules(args.model_id)
        if not success:
            logger.error("\n✗ Verification 1 FAILED - Model loading failed")
            return 1
        logger.info("\n✓ Verification 1 PASSED - Model loads and modules listed")
    else:
        logger.info("\n⏭️  Verification 1 SKIPPED (--skip-model-load)")
    
    # Verification 2: Hook point naming (manual)
    verify_hook_point_naming()
    
    # Verification 3: Decoder shape (manual)
    verify_decoder_shape()
    
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
