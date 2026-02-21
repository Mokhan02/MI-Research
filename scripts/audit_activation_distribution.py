"""
Audit activation distribution before setting tau_act.

tau_act=0.0 is usually too permissive: act_freq becomes "fraction where act > 0",
which is noisy and can drift across seeds. Run this on a small baseline (e.g. neutral_select)
to see percentiles of activations for a random subset of features, then set tau_act
to cut out near-zero fuzz (e.g. 25th or 50th percentile, or a fixed value from this run).

Usage:
  uv run python scripts/audit_activation_distribution.py --config configs/targets/gemma2_2b_gemmascope_res16k.yaml --prompt_csv data/prompts/neutral_select.csv --n_features 200 --n_prompts 100
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config, resolve_config
from src.model_utils import load_model
from src.hook_resolver import resolve_hook_point
from src.sae_loader import load_gemmascope_decoder


def main():
    ap = argparse.ArgumentParser(description="Print activation distribution for a random subset of features")
    ap.add_argument("--config", type=str, default="configs/targets/gemma2_2b_gemmascope_res16k.yaml")
    ap.add_argument("--prompt_csv", type=str, default="data/prompts/neutral_select.csv")
    ap.add_argument("--n_features", type=int, default=200, help="Number of random features to sample")
    ap.add_argument("--n_prompts", type=int, default=100, help="Cap prompts to this many")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--token_span", type=str, default="last_n", choices=["last", "last_n", "all"])
    ap.add_argument("--last_n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = load_config(args.config)
    resolve_config(config, "audit_act")
    if not Path(args.prompt_csv).exists():
        print(f"Prompt CSV not found: {args.prompt_csv}")
        sys.exit(1)
    df = pd.read_csv(args.prompt_csv, dtype={"prompt": "string"})
    prompts = df["prompt"].dropna().astype(str).str.strip().tolist()[: args.n_prompts]
    if not prompts:
        print("No prompts")
        sys.exit(1)

    print("Loading model and SAE...")
    model, tokenizer = load_model(config)
    model.eval()
    device = next(model.parameters()).device
    hook_module = resolve_hook_point(model, config["sae"]["hook_point"])
    decoder, _ = load_gemmascope_decoder(config)
    W_dec = decoder.to(device=device, dtype=torch.float32)
    n_total, d_model = W_dec.shape
    feature_idx = np.random.choice(n_total, size=min(args.n_features, n_total), replace=False)

    captured = []
    def hook_fn(module, input, output):
        out = output[0] if isinstance(output, tuple) else output
        captured.append(out.detach())
    handle = hook_module.register_forward_hook(hook_fn)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    acts_list = []
    with torch.no_grad():
        for i in range(0, len(prompts), args.batch_size):
            batch = prompts[i : i + args.batch_size]
            captured.clear()
            inp = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            model(**inp)
            resid = captured[0]
            attn = inp.get("attention_mask")
            if attn is not None:
                last_idx = (attn.sum(dim=1) - 1).clamp(min=0)
                batch_dim = torch.arange(resid.shape[0], device=device)
                if args.token_span == "last":
                    pos = resid[batch_dim, last_idx, :]
                elif args.token_span == "last_n":
                    start = (last_idx - args.last_n + 1).clamp(min=0)
                    pos = torch.stack([
                        resid[b, start[b]:last_idx[b] + 1, :].mean(dim=0) for b in range(resid.shape[0])
                    ])
                else:
                    pos = resid.mean(dim=1)
            else:
                pos = resid[:, -1, :]
            pos = pos.float()
            acts = pos @ W_dec[torch.as_tensor(feature_idx, device=device)].T
            acts_list.append(acts.cpu().numpy())
    handle.remove()

    acts = np.concatenate(acts_list, axis=0)
    # acts: (n_prompts, n_sampled_features)
    flat = acts.flatten()
    flat_positive = flat[flat > 0]

    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print(f"\n--- Activation distribution (n_prompts={acts.shape[0]}, n_features_sampled={len(feature_idx)}) ---")
    print(f"  token_span={args.token_span}, last_n={args.last_n}")
    print(f"  All values:  min={flat.min():.4f}  max={flat.max():.4f}  mean={flat.mean():.4f}  std={flat.std():.4f}")
    print(f"  Fraction > 0: {(flat > 0).mean():.2%}")
    if len(flat_positive) > 0:
        pct = np.percentile(flat_positive, percentiles)
        print(f"  Conditional on act > 0:")
        for p, v in zip(percentiles, pct):
            print(f"    p{p}: {v:.4f}")
        print(f"\n  Recommendation: set tau_act above near-zero fuzz, e.g. 25th percentile of positive activations = {pct[3]:.4f}")
        print(f"  Or use a small fixed value (e.g. 0.01--0.05) and re-audit if you change layer/model.")
    else:
        print("  No positive activations in sample.")
    print()


if __name__ == "__main__":
    main()
