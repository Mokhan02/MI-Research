"""Model loading utilities.

SAE loading lives in src/sae_loader.py (load_gemmascope_decoder).
"""

import torch
from typing import Dict, Any
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

