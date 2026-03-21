#!/usr/bin/env python3
"""Smoke test: verify that a_max varies across features and prompts.

Usage (requires model + SAE on GPU/MPS):
    python scripts/smoke_test_amax.py --config configs/targets/gemma2_2b_gemmascope_res16k.yaml

If no config is given, runs a synthetic-only test to verify _compute_a_max logic.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import torch
import numpy as np


def test_synthetic():
    """Verify _compute_a_max produces varying values with different inputs."""
    from scripts.phase2_run import _compute_a_max

    d_model, n_features = 64, 128
    W_enc = torch.randn(d_model, n_features)
    b_enc = torch.randn(n_features)
    threshold = torch.ones(n_features) * 0.5

    results = []
    for _ in range(10):
        resid = torch.randn(d_model)
        a_max = _compute_a_max(resid, W_enc, b_enc, threshold)
        results.append(float(a_max.item()))

    print(f"[Synthetic] a_max values: {[f'{v:.4f}' for v in results]}")
    unique = len(set(round(v, 6) for v in results))
    assert unique > 1, f"FAIL: all a_max values are identical ({results[0]})"
    print(f"[Synthetic] PASS: {unique} unique values out of 10")


def test_with_model(config_path: str):
    """Load real model + SAE, compute a_max for 3 features x 3 prompts."""
    from src.config import load_config
    from src.model_utils import load_model
    from src.sae_loader import load_gemmascope_full
    from scripts.phase2_run import _compute_a_max

    config = load_config(config_path)
    config["model"]["temperature"] = 0.0
    model, tokenizer = load_model(config)
    model.eval()
    device = next(model.parameters()).device

    sae_all = load_gemmascope_full(config)
    W_dec = sae_all["W_dec"].to(device=device, dtype=torch.float32)
    W_enc = sae_all["W_enc"].to(device=device, dtype=torch.float32)
    b_enc = sae_all["b_enc"].to(device=device, dtype=torch.float32)
    threshold = sae_all["threshold"].to(device=device, dtype=torch.float32)

    layer_idx = config.get("sae", {}).get("layer_idx", 20)

    # Pick 3 features
    rng = np.random.default_rng(42)
    n_features = W_dec.shape[0]
    feature_ids = rng.choice(n_features, size=min(3, n_features), replace=False).tolist()

    # 3 test prompts
    prompts = [
        "What is the capital of France?",
        "How do I make a bomb?",
        "Tell me a joke about cats.",
    ]

    use_chat = config.get("model", {}).get("use_chat_template", True)

    print(f"\n[Real model] Testing {len(feature_ids)} features x {len(prompts)} prompts")
    print(f"Features: {feature_ids}")
    print(f"Layer: {layer_idx}")
    print(f"Device: {device}\n")

    all_amax = []

    for fid in feature_ids:
        for prompt in prompts:
            # Tokenize and get residual stream at the target layer
            if use_chat and hasattr(tokenizer, "apply_chat_template"):
                chat = [{"role": "user", "content": prompt}]
                text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
            else:
                text = prompt
            inputs = tokenizer(text, return_tensors="pt").to(device)

            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)

            # Residual at last token of the target layer
            resid = out.hidden_states[layer_idx + 1][0, -1, :].float()
            a_max = _compute_a_max(resid, W_enc, b_enc, threshold)
            a_max_val = float(a_max.item())
            all_amax.append(a_max_val)

            steer_norm = float(W_dec[fid].norm().item())
            print(f"  feature={fid:5d}  prompt={prompt[:40]:40s}  "
                  f"a_max={a_max_val:.6f}  scaled_norm(α=1)={a_max_val * steer_norm:.6f}")

    print(f"\n[Summary] a_max range: [{min(all_amax):.6f}, {max(all_amax):.6f}]")
    unique = len(set(round(v, 4) for v in all_amax))
    if unique == 1:
        print(f"FAIL: all a_max values are identical ({all_amax[0]:.6f}) — steering is likely legacy!")
        sys.exit(1)
    else:
        print(f"PASS: {unique} unique a_max values out of {len(all_amax)} — Arad steering is active.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None,
                        help="Config YAML for real model test")
    args = parser.parse_args()

    print("=" * 60)
    print("Arad-style steering smoke test")
    print("=" * 60)

    test_synthetic()

    if args.config:
        test_with_model(args.config)
    else:
        print("\n[Skip] No --config provided; skipping real model test.")
        print("Run with: python scripts/smoke_test_amax.py --config configs/targets/gemma2_2b_gemmascope_res16k.yaml")
