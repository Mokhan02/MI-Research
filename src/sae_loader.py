"""
SAE decoder loader: supports NPZ (GemmaScope) and safetensors (e.g. Llama Scope).

Downloads from HuggingFace Hub and loads decoder weights; uses heuristics to find
the decoder matrix when key names vary.
"""

import logging
import os
from typing import Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _download_from_hub(repo_id: str, filename: str, cache_dir: str | None = None) -> str:
    """Download a file from HuggingFace Hub; return local path."""
    from huggingface_hub import hf_hub_download
    local_path = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=cache_dir)
    logger.info(f"Downloaded: {repo_id}/{filename} -> {local_path}")
    return local_path


def download_sae_npz(repo_id: str, file_path: str, cache_dir: str | None = None) -> str:
    """
    Download params.npz from HuggingFace Hub into cache; return local file path.

    Args:
        repo_id: HuggingFace repo (e.g. google/gemma-scope-2b-pt-res).
        file_path: Path within repo (e.g. layer_20/width_16k/average_l0_71/params.npz).
        cache_dir: Optional cache directory; if None, uses HuggingFace default.

    Returns:
        Absolute path to the downloaded NPZ file on disk.
    """
    return _download_from_hub(repo_id, file_path, cache_dir)


def load_decoder_from_npz(
    npz_path: str,
    n_features_total: int,
    d_model: int,
) -> Tuple[torch.Tensor, str, bool]:
    """
    Load NPZ, infer decoder array by heuristics, return as [n_features, d_model] on CPU float32.

    Heuristics:
    - Prefer keys containing "W_dec" or "decoder".
    - Else pick the 2D array whose one dimension equals n_features_total or d_model.
    - Return tensor as [n_features_total, d_model]; transpose if necessary.

    Returns:
        decoder: (n_features_total, d_model) float32 on CPU.
        chosen_key: Name of the array key used.
        transposed: True if the array was transposed to get (n_features, d_model).

    Raises:
        ValueError: If no suitable 2D array is found.
    """
    data = np.load(npz_path)
    keys = list(data.keys())
    logger.info(f"NPZ keys: {keys}")

    chosen_key = None
    transposed = False

    # Prefer W_dec / W_D / decoder (covers GemmaScope, OpenMOSS/LlamaScope, etc.)
    preferred = [k for k in keys if "W_dec" in k or "W_D" in k or "decoder" in k.lower()]
    if preferred:
        chosen_key = preferred[0]
        logger.info(f"Chosen key (W_dec/decoder): {chosen_key}")
    else:
        # Find 2D array with one dim n_features_total or d_model
        candidates = []
        for k in keys:
            arr = data[k]
            if not hasattr(arr, "shape") or len(arr.shape) != 2:
                continue
            a, b = arr.shape
            if (a == n_features_total and b == d_model) or (a == d_model and b == n_features_total):
                candidates.append(k)
        if not candidates:
            raise ValueError(
                f"No 2D array with one dimension in {{{n_features_total}, {d_model}}} found. "
                f"Keys: {keys}. Inspect NPZ and add support for the decoder key."
            )
        # Prefer (n_features, d_model) order
        for k in candidates:
            arr = np.asarray(data[k], dtype=np.float32)
            if arr.shape[0] == n_features_total and arr.shape[1] == d_model:
                chosen_key = k
                break
        else:
            chosen_key = candidates[0]
        logger.info(f"Chosen key (shape heuristic): {chosen_key}")

    if chosen_key is None:
        raise ValueError(f"No suitable 2D array. Keys: {keys}.")

    arr = np.asarray(data[chosen_key], dtype=np.float32)
    if arr.shape[0] == d_model and arr.shape[1] == n_features_total:
        arr = arr.T
        transposed = True
        logger.info(f"Transposed decoder to (n_features, d_model): {arr.shape}")
    elif arr.shape[0] != n_features_total or arr.shape[1] != d_model:
        raise ValueError(
            f"Decoder shape {arr.shape} is not (n_features_total={n_features_total}, d_model={d_model}) "
            f"nor (d_model, n_features_total). Key={chosen_key}."
        )
    else:
        logger.info(f"Decoder shape (no transpose): {arr.shape}")

    out = torch.from_numpy(arr).to(torch.float32)
    assert out.shape == (n_features_total, d_model)
    return out, chosen_key, transposed


def load_decoder_from_safetensors(
    st_path: str,
    n_features_total: int,
    d_model: int,
) -> Tuple[torch.Tensor, str, bool]:
    """
    Load safetensors file, infer decoder by heuristics, return [n_features, d_model] float32 CPU.

    Prefers keys containing "W_dec" or "decoder"; else 2D tensor with shape
    (n_features_total, d_model) or (d_model, n_features_total). Transposes to (n_features, d_model) if needed.

    Returns:
        decoder, chosen_key, transposed.
    """
    from safetensors.torch import load_file
    state = load_file(st_path)
    keys = list(state.keys())
    logger.info(f"Safetensors keys: {keys}")

    chosen_key = None
    transposed = False
    preferred = [k for k in keys if "W_dec" in k or "W_D" in k or "decoder" in k.lower()]
    if preferred:
        chosen_key = preferred[0]
        logger.info(f"Chosen key (W_dec/decoder): {chosen_key}")
    else:
        candidates = []
        for k in keys:
            t = state[k]
            if not isinstance(t, torch.Tensor) or t.dim() != 2:
                continue
            a, b = t.shape
            if (a == n_features_total and b == d_model) or (a == d_model and b == n_features_total):
                candidates.append(k)
        if not candidates:
            raise ValueError(
                f"No 2D tensor with one dimension in {{{n_features_total}, {d_model}}}. Keys: {keys}"
            )
        for k in candidates:
            if state[k].shape[0] == n_features_total and state[k].shape[1] == d_model:
                chosen_key = k
                break
        else:
            chosen_key = candidates[0]
        logger.info(f"Chosen key (shape heuristic): {chosen_key}")

    if chosen_key is None:
        raise ValueError(f"No suitable 2D tensor. Keys: {keys}.")

    t = state[chosen_key].to(torch.float32)
    if t.shape[0] == d_model and t.shape[1] == n_features_total:
        t = t.T
        transposed = True
        logger.info(f"Transposed decoder to (n_features, d_model): {t.shape}")
    elif t.shape[0] != n_features_total or t.shape[1] != d_model:
        raise ValueError(
            f"Decoder shape {t.shape} not (n_features_total={n_features_total}, d_model={d_model}). Key={chosen_key}."
        )
    if t.device.type != "cpu":
        t = t.cpu()
    assert t.shape == (n_features_total, d_model)
    return t, chosen_key, transposed


def load_gemmascope_decoder(cfg: dict) -> Tuple[torch.Tensor, dict]:
    """
    Load GemmaScope decoder from config: download NPZ, load decoder, assert shape.

    Reads:
        cfg.sae.weights_repo
        cfg.sae.weights_path (required; add to config if missing)
        cfg.sae.n_features_total
        cfg.architecture.d_model (required; no hardcode)

    Returns:
        decoder: (n_features_total, d_model) float32 on CPU.
        metadata: dict with npz_keys, chosen_key, transposed, decoder_shape.
    """
    sae = cfg.get("sae", {})
    sae_id = sae.get("sae_id")
    if sae_id in ("TBD", "", None):
        raise ValueError(
            "sae.sae_id must be set to a concrete identifier (e.g. 20-gemmascope-res-16k/2263). "
            f"Got: {sae_id!r}. No mercy: every run must be reproducible."
        )
    arch = cfg.get("architecture", {})
    repo_id = sae.get("weights_repo")
    file_path = sae.get("weights_path")
    n_features_total = sae.get("n_features_total")
    d_model = arch.get("d_model")

    if not repo_id:
        raise ValueError("cfg.sae.weights_repo is required")
    if not file_path:
        raise ValueError("cfg.sae.weights_path is required")
    if n_features_total is None:
        raise ValueError("cfg.sae.n_features_total is required")
    if d_model is None:
        raise ValueError("cfg.architecture.d_model is required (or load model and use model.config.hidden_size)")

    cache_dir = sae.get("cache_dir")
    is_safetensors = file_path.rstrip("/").endswith(".safetensors")
    local_path = _download_from_hub(repo_id, file_path, cache_dir=cache_dir)

    if is_safetensors:
        decoder, chosen_key, transposed = load_decoder_from_safetensors(
            local_path, n_features_total, d_model
        )
        metadata = {
            "npz_path": local_path,
            "weights_format": "safetensors",
            "chosen_key": chosen_key,
            "transposed": transposed,
            "decoder_shape": list(decoder.shape),
            "n_features_total": n_features_total,
            "d_model": d_model,
        }
    else:
        decoder, chosen_key, transposed = load_decoder_from_npz(
            local_path, n_features_total, d_model
        )
        data = np.load(local_path)
        npz_keys = list(data.keys())
        metadata = {
            "npz_path": local_path,
            "weights_format": "npz",
            "npz_keys": npz_keys,
            "chosen_key": chosen_key,
            "transposed": transposed,
            "decoder_shape": list(decoder.shape),
            "n_features_total": n_features_total,
            "d_model": d_model,
        }
    assert decoder.shape[0] == n_features_total and decoder.shape[1] == d_model
    logger.info(f"Decoder loaded: shape={decoder.shape}, chosen_key={chosen_key}, transposed={transposed}")
    return decoder, metadata
