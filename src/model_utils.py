"""Model loading utilities.

SAE loading lives in src/sae_loader.py (load_gemmascope_decoder).
"""

import torch
from typing import Dict, Any
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def resolve_device(requested: str = "auto") -> str:
    """Resolve device string to the best available: MPS > CUDA > CPU.

    Accepts "auto", "mps", "cuda", or "cpu".  When "auto", picks the first
    available accelerator.
    """
    if requested == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but unavailable – falling back to auto-detect")
        return resolve_device("auto")
    if requested == "mps" and not torch.backends.mps.is_available():
        logger.warning("MPS requested but unavailable – falling back to auto-detect")
        return resolve_device("auto")
    return requested


def load_model(config: Dict[str, Any]):
    """Load model and tokenizer from config."""
    model_cfg = config["model"]
    model_id = model_cfg["model_id"]
    device = resolve_device(model_cfg.get("device", "auto"))
    dtype_str = model_cfg.get("dtype", "float32")

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(dtype_str, torch.float32)

    # MPS doesn't support bfloat16 – silently downgrade to float16
    if device == "mps" and dtype == torch.bfloat16:
        logger.warning("MPS does not support bfloat16 – using float16 instead")
        dtype = torch.float16

    logger.info(f"Loading model {model_id} on {device} with dtype {dtype}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device if device not in ("cpu", "mps") else None,
    )
    if device in ("cpu", "mps"):
        model = model.to(device)

    model.eval()

    logger.info(f"Model loaded successfully on {device}")
    return model, tokenizer

