"""
GemmaScope SAE loader (Neuronpedia source).

Downloads and loads decoder weights from NPZ files in the GemmaScope weights repo.
Does not assume array key names; uses heuristics to find the decoder matrix.

Also supports loading the full SAE (encoder, decoder, bias, threshold) for
activation-scaled steering (Arad et al., 2025).
"""

import logging
import os
from typing import Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


# Canonical NPZ key names across GemmaScope releases
_KEY_ALIASES = {
    "W_dec": ["W_dec", "decoder"],
    "W_enc": ["W_enc", "encoder"],
    "b_enc": ["b_enc", "encoder_bias"],
    "threshold": ["threshold", "thr"],
}


def _find_npz_key(data, canonical: str) -> str | None:
    """Find the first matching key in data for a canonical parameter name."""
    aliases = _KEY_ALIASES.get(canonical, [canonical])
    for alias in aliases:
        for key in data.keys():
            if alias in key or alias.lower() in key.lower():
                return key
    return None


def download_sae_weights(repo_id: str, file_path: str, cache_dir: str | None = None) -> str:
    """
    Download SAE weights file (NPZ or PT) from HuggingFace Hub; return local path.

    Args:
        repo_id: HuggingFace repo (e.g. adamkarvonen/qwen3-8b-saes).
        file_path: Path within repo (e.g. saes_.../resid_post_layer_18/trainer_0/ae.pt).
        cache_dir: Optional cache directory.

    Returns:
        Absolute path to the downloaded file on disk.
    """
    from huggingface_hub import hf_hub_download

    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=file_path,
        cache_dir=cache_dir,
    )
    logger.info(f"Downloaded SAE weights: {repo_id}/{file_path} -> {local_path}")
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
    from huggingface_hub import hf_hub_download

    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=file_path,
        cache_dir=cache_dir,
    )
    logger.info(f"Downloaded SAE NPZ: {repo_id}/{file_path} -> {local_path}")
    return local_path


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

    # Prefer W_dec / decoder
    preferred = [k for k in keys if "W_dec" in k or "decoder" in k.lower()]
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


def load_decoder_from_pt(
    pt_path: str,
    n_features_total: int,
    d_model: int,
) -> Tuple[torch.Tensor, str, bool]:
    """
    Load decoder from a dictionary_learning-style ae.pt file.

    The raw state dict uses key 'decoder.weight' with shape (n_features, d_model)
    or (d_model, n_features). Returns (decoder, chosen_key, transposed) in the
    same contract as load_decoder_from_npz.

    Raises:
        ValueError: If no suitable decoder tensor is found.
    """
    state = torch.load(pt_path, map_location="cpu", weights_only=True)

    # Prefer 'decoder.weight'; fall back to any 2-D tensor matching the shape
    chosen_key = None
    for k in ("decoder.weight", "W_dec", "decoder"):
        if k in state:
            chosen_key = k
            break

    if chosen_key is None:
        candidates = [k for k, v in state.items() if isinstance(v, torch.Tensor) and v.dim() == 2]
        for k in candidates:
            s = state[k].shape
            if set(s) == {n_features_total, d_model} or (n_features_total in s and d_model in s):
                chosen_key = k
                break

    if chosen_key is None:
        raise ValueError(
            f"No decoder tensor found in PT state dict. Keys: {list(state.keys())}. "
            f"Expected a 2-D tensor with dims {n_features_total} and {d_model}."
        )

    arr = state[chosen_key].to(torch.float32)
    transposed = False
    if arr.shape == (d_model, n_features_total):
        arr = arr.T
        transposed = True
    elif arr.shape != (n_features_total, d_model):
        raise ValueError(
            f"Decoder shape {tuple(arr.shape)} is not ({n_features_total}, {d_model}) "
            f"nor ({d_model}, {n_features_total}). Key={chosen_key}."
        )

    logger.info(f"PT decoder loaded: key={chosen_key}, shape={tuple(arr.shape)}, transposed={transposed}")
    assert arr.shape == (n_features_total, d_model)
    return arr, chosen_key, transposed


def load_gemmascope_decoder(cfg: dict) -> Tuple[torch.Tensor, dict]:
    """
    Load SAE decoder from config. Supports NPZ (GemmaScope) and PT (dictionary_learning) formats.

    Dispatches on the file extension of cfg.sae.weights_path:
        .npz  -> download_sae_npz + load_decoder_from_npz  (existing path, unchanged)
        .pt   -> download_sae_weights + load_decoder_from_pt

    Reads:
        cfg.sae.weights_repo
        cfg.sae.weights_path
        cfg.sae.n_features_total
        cfg.architecture.d_model

    Returns:
        decoder: (n_features_total, d_model) float32 on CPU.
        metadata: dict with weights_path, chosen_key, transposed, decoder_shape.
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
    ext = file_path.rsplit(".", 1)[-1].lower()

    if ext == "pt":
        local_path = download_sae_weights(repo_id, file_path, cache_dir=cache_dir)
        decoder, chosen_key, transposed = load_decoder_from_pt(local_path, n_features_total, d_model)
        state_keys = list(torch.load(local_path, map_location="cpu", weights_only=True).keys())
        metadata = {
            "weights_path": local_path,
            "weights_format": "pt",
            "state_keys": state_keys,
            "chosen_key": chosen_key,
            "transposed": transposed,
            "decoder_shape": list(decoder.shape),
            "n_features_total": n_features_total,
            "d_model": d_model,
        }
    else:
        # NPZ path — unchanged
        npz_path = download_sae_npz(repo_id, file_path, cache_dir=cache_dir)
        decoder, chosen_key, transposed = load_decoder_from_npz(npz_path, n_features_total, d_model)
        data = np.load(npz_path)
        metadata = {
            "weights_path": npz_path,
            "weights_format": "npz",
            "npz_path": npz_path,  # kept for backwards compat
            "npz_keys": list(data.keys()),
            "chosen_key": chosen_key,
            "transposed": transposed,
            "decoder_shape": list(decoder.shape),
            "n_features_total": n_features_total,
            "d_model": d_model,
        }

    assert decoder.shape == (n_features_total, d_model), (
        f"Expected ({n_features_total}, {d_model}), got {decoder.shape}"
    )
    logger.info(f"Decoder loaded: shape={decoder.shape}, chosen_key={chosen_key}, transposed={transposed}")
    return decoder, metadata


def load_gemmascope_full(cfg: dict) -> dict:
    """
    Load full GemmaScope SAE: W_dec, W_enc, b_enc, threshold.

    Uses the same config keys as load_gemmascope_decoder. Returns a dict with:
        W_dec:      (n_features, d_model) float32 CPU
        W_enc:      (d_model, n_features) float32 CPU  (always this orientation)
        b_enc:      (n_features,) float32 CPU
        threshold:  (n_features,) float32 CPU
        metadata:   dict with npz_path, keys used, shapes
    """
    decoder, meta = load_gemmascope_decoder(cfg)
    weights_path = meta["weights_path"]
    n_features = meta["n_features_total"]
    d_model = meta["d_model"]

    if meta.get("weights_format") == "pt":
        state = torch.load(weights_path, map_location="cpu", weights_only=True)

        # encoder.weight: (n_features, d_model) or (d_model, n_features)
        enc_raw = state.get("encoder.weight")
        if enc_raw is None:
            enc_raw = state.get("W_enc")
        if enc_raw is None:
            raise ValueError(
                f"Cannot find encoder weights in PT state dict. Keys: {list(state.keys())}. "
                "Expected 'encoder.weight' or 'W_enc'."
            )
        W_enc = enc_raw.to(torch.float32)
        if W_enc.shape == (n_features, d_model):
            W_enc = W_enc.T
        assert W_enc.shape == (d_model, n_features), (
            f"W_enc shape {tuple(W_enc.shape)} != expected ({d_model}, {n_features})"
        )

        b_enc_raw = state.get("encoder.bias")
        if b_enc_raw is None:
            b_enc_raw = state.get("b_enc")
        if b_enc_raw is None:
            raise ValueError(
                f"Cannot find encoder bias in PT state dict. Keys: {list(state.keys())}. "
                "Expected 'encoder.bias' or 'b_enc'."
            )
        b_enc = b_enc_raw.to(torch.float32)
        assert b_enc.shape == (n_features,), f"b_enc shape {tuple(b_enc.shape)} != ({n_features},)"

        thr_raw = state.get("threshold")
        if thr_raw is None:
            raise ValueError(
                f"Cannot find threshold in PT state dict. Keys: {list(state.keys())}. "
                "Expected 'threshold'."
            )
        threshold = thr_raw.to(torch.float32)
        assert threshold.shape == (n_features,), f"threshold shape {tuple(threshold.shape)} != ({n_features},)"

        meta["enc_key"] = "encoder.weight"
        meta["benc_key"] = "encoder.bias"
        meta["thr_key"] = "threshold"

    else:
        # NPZ path — unchanged
        data = np.load(weights_path)

        enc_key = _find_npz_key(data, "W_enc")
        if enc_key is None:
            raise ValueError(
                f"Cannot find encoder weights in NPZ. Keys: {list(data.keys())}. "
                "Expected a key containing 'W_enc' or 'encoder'."
            )
        W_enc = torch.from_numpy(np.asarray(data[enc_key], dtype=np.float32))
        if W_enc.shape == (n_features, d_model):
            W_enc = W_enc.T
        assert W_enc.shape == (d_model, n_features), (
            f"W_enc shape {W_enc.shape} != expected ({d_model}, {n_features})"
        )

        benc_key = _find_npz_key(data, "b_enc")
        if benc_key is None:
            raise ValueError(
                f"Cannot find encoder bias in NPZ. Keys: {list(data.keys())}. "
                "Expected a key containing 'b_enc' or 'encoder_bias'."
            )
        b_enc = torch.from_numpy(np.asarray(data[benc_key], dtype=np.float32))
        assert b_enc.shape == (n_features,), f"b_enc shape {b_enc.shape} != ({n_features},)"

        thr_key = _find_npz_key(data, "threshold")
        if thr_key is None:
            raise ValueError(
                f"Cannot find threshold in NPZ. Keys: {list(data.keys())}. "
                "Expected a key containing 'threshold' or 'thr'."
            )
        threshold = torch.from_numpy(np.asarray(data[thr_key], dtype=np.float32))
        assert threshold.shape == (n_features,), f"threshold shape {threshold.shape} != ({n_features},)"

        meta["enc_key"] = enc_key
        meta["benc_key"] = benc_key
        meta["thr_key"] = thr_key

    logger.info(
        f"Full SAE loaded: W_dec={decoder.shape}, W_enc={W_enc.shape}, "
        f"b_enc={b_enc.shape}, threshold={threshold.shape}"
    )

    return {
        "W_dec": decoder,
        "W_enc": W_enc,
        "b_enc": b_enc,
        "threshold": threshold,
        "metadata": meta,
    }
