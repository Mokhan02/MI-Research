"""
Hook resolver: map TransformerLens-style hook names to HuggingFace module names.

Resolves blocks.{L}.hook_resid_post -> model.layers.{L} so TL-style configs work with HF models.
"""

import re
import logging
from typing import Tuple

import torch

logger = logging.getLogger(__name__)


def parse_tl_blocks_hook(hook_name: str) -> Tuple[int | None, str | None]:
    """
    Parse TransformerLens-style blocks hook name.

    Args:
        hook_name: e.g. "blocks.20.hook_resid_post"

    Returns:
        (layer_idx, kind) if pattern matches, else (None, None).
        kind is the part after the last dot (e.g. "hook_resid_post").
    """
    m = re.match(r"blocks\.(\d+)\.(.+)$", hook_name.strip())
    if not m:
        return None, None
    layer_idx = int(m.group(1))
    kind = m.group(2)
    return layer_idx, kind


def resolve_hook_point(model: torch.nn.Module, hook_name: str) -> torch.nn.Module:
    """
    Resolve hook name to the actual HuggingFace module.

    Mapping rules:
    - If hook_name matches blocks.{L}.hook_resid_post, map to model.layers.{L} and return that module.
    - Otherwise, attempt exact match in model.named_modules() and return it.
    - If not found, raise ValueError listing close candidates that contain the layer index.

    Returns:
        The module to attach the hook to.

    Raises:
        ValueError: If hook point cannot be resolved.
    """
    named_modules = dict(model.named_modules())

    # Exact match
    if hook_name in named_modules:
        logger.info(f"Resolved hook point (exact match): '{hook_name}'")
        return named_modules[hook_name]

    # TransformerLens: blocks.{L}.hook_resid_post -> model.layers.{L}
    layer_idx, kind = parse_tl_blocks_hook(hook_name)
    if layer_idx is not None and kind is not None:
        if "hook_resid_post" in kind or kind == "hook_resid_post":
            hf_name = f"model.layers.{layer_idx}"
            if hf_name in named_modules:
                logger.info(f"Resolved hook point (TL -> HF): '{hook_name}' -> '{hf_name}'")
                return named_modules[hf_name]
            # Try without "model." prefix (some architectures)
            alt = f"layers.{layer_idx}"
            for name, mod in named_modules.items():
                if name == alt or name.endswith("." + alt):
                    logger.info(f"Resolved hook point (TL -> HF): '{hook_name}' -> '{name}'")
                    return mod

    # Close candidates containing layer index
    layer_str = str(layer_idx) if layer_idx is not None else ""
    if layer_str:
        candidates = [n for n in named_modules.keys() if layer_str in n and ("layer" in n.lower() or "block" in n.lower())]
        if candidates:
            raise ValueError(
                f"Hook point '{hook_name}' not found. "
                f"Resolved layer index {layer_idx} but 'model.layers.{layer_idx}' not in model. "
                f"Close candidates: {candidates[:15]}"
            )
    # Generic: list some names containing "layer"
    candidates = [n for n in named_modules.keys() if "layer" in n.lower()][:20]
    raise ValueError(
        f"Hook point '{hook_name}' not found. "
        f"Example module names: {candidates}"
    )
