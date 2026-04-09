# scripts/compute_output_scores.py
"""
Compute Arad et al. output scores for a set of candidate SAE features.

For each feature:
  1. Project W_dec[fid] through final layer norm + unembedding → top-k logit-lens tokens.
  2. Run one forward pass on a neutral prompt with activation-scaled steering (s=amp_factor).
  3. Compute output_score = (1 - min_rank/vocab_size) * max_prob(logit_lens_tokens).

Inputs:
  --config:           YAML config (model + SAE)
  --features_path:    JSON with {"feature_ids": [...]} (e.g. from phase2_select_contrast)
  --out_path:         Output JSON: {feature_id: output_score, ...}
  --amp_factor:       Steering factor for intervention (default 10, matching Arad)
  --logit_lens_k:     Top-k logit lens tokens per feature (default 20)
  --neutral_prompt:   Neutral prompt for scoring (default "In my experience,")
  --threshold:        Minimum output score to pass filter (default 0.01)
  --filtered_out:     Optional path to write filtered selected_features JSON

Reference: Arad, Mueller, Belinkov (2025) §3.2, Eq. 9-10.
           https://arxiv.org/abs/2505.20063
           https://github.com/technion-cs-nlp/saes-are-good-for-steering
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import torch
import numpy as np
from tqdm.auto import tqdm

from src.config import load_config, resolve_config
from src.model_utils import load_model
from src.sae_loader import load_gemmascope_full


@torch.no_grad()
def compute_logit_lens_topk(W_dec: torch.Tensor, final_norm, lm_head,
                            k: int = 20) -> torch.Tensor:
    """
    For each feature, project decoder direction through final layer norm + lm_head.
    Returns top-k token indices per feature: (n_features, k).
    """
    dtype = next(lm_head.parameters()).dtype
    normed = final_norm(W_dec.to(dtype=dtype))
    logits = lm_head(normed)
    probs = torch.softmax(logits, dim=-1)
    topk = torch.topk(probs, k=k, dim=-1)
    return topk.indices  # (n_features, k)


@torch.no_grad()
def compute_output_score(model, tokenizer, layer_idx: int, feature_id: int,
                         logit_lens_indices: list[int], neutral_prompt: str,
                         W_dec: torch.Tensor, W_enc: torch.Tensor,
                         b_enc: torch.Tensor, threshold: torch.Tensor,
                         amp_factor: float, device) -> float:
    """
    Compute output score for a single feature (Arad et al. Eq. 9-10).

    1. Forward pass with activation-scaled steering at last token.
    2. Get output distribution at last position.
    3. Score = (1 - min_rank_of_ll_tokens / vocab_size) * max_prob_of_ll_tokens.
    """
    steer_dir = W_dec[feature_id].to(device=device, dtype=torch.float32)

    def _prehook(module, inputs):
        hidden = inputs[0]
        hidden2 = hidden.clone()
        resid = hidden2[0, -1, :].float()
        z = resid @ W_enc + b_enc
        act = torch.relu(z - threshold)
        a_max = act.max()
        hidden2[0, -1, :] = hidden2[0, -1, :] + amp_factor * a_max * steer_dir
        return (hidden2,) + inputs[1:]

    layer_module = model.model.layers[layer_idx]
    handle = layer_module.register_forward_pre_hook(_prehook)

    inputs = tokenizer(neutral_prompt, return_tensors="pt").to(device)
    outputs = model(**inputs)
    handle.remove()

    logits = outputs.logits[0, -1, :]
    probs = torch.softmax(logits.float(), dim=0)
    vocab_size = probs.shape[0]

    sorted_indices = torch.argsort(probs, descending=True)
    ranks = []
    for ll_token in logit_lens_indices:
        rank = (sorted_indices == ll_token).nonzero(as_tuple=True)[0].item()
        ranks.append(rank)

    min_rank = min(ranks)
    rank_score = 1.0 - (min_rank / vocab_size)
    top_prob = probs[logit_lens_indices].max().item()

    return rank_score * top_prob


def main():
    ap = argparse.ArgumentParser(description="Compute output scores for SAE features (Arad et al.)")
    ap.add_argument("--config", type=str, default="configs/targets/gemma2_2b_gemmascope_res16k.yaml")
    ap.add_argument("--features_path", type=str, required=True,
                    help="JSON with feature_ids list (from phase2_select_contrast or manual)")
    ap.add_argument("--out_path", type=str, default="outputs/output_scores.json")
    ap.add_argument("--amp_factor", type=float, default=10.0,
                    help="Steering factor for output score (default 10, matching Arad)")
    ap.add_argument("--logit_lens_k", type=int, default=20)
    ap.add_argument("--neutral_prompt", type=str, default="In my experience,")
    ap.add_argument("--threshold", type=float, default=0.01,
                    help="Minimum output score to include in filtered set")
    ap.add_argument("--filtered_out", type=str, default=None,
                    help="Path to write filtered selected_features JSON")
    ap.add_argument("--top_k", type=int, default=None,
                    help="If set with --filtered_out, write only top K by score (for sanity checks)")
    ap.add_argument("--layer", type=int, default=None,
                    help="Override layer index (default: from config sae.layer_idx)")
    args = ap.parse_args()

    config = resolve_config(load_config(args.config), run_id="output_scores")
    config["model"]["do_sample"] = False
    config["model"]["temperature"] = 0.0
    model, tokenizer = load_model(config)
    model.eval()
    device = next(model.parameters()).device

    layer_idx = args.layer if args.layer is not None else config["sae"]["layer_idx"]

    sae_all = load_gemmascope_full(config)
    W_dec = sae_all["W_dec"].to(device=device, dtype=torch.float32)
    W_enc = sae_all["W_enc"].to(device=device, dtype=torch.float32)
    b_enc = sae_all["b_enc"].to(device=device, dtype=torch.float32)
    thr = sae_all["threshold"].to(device=device, dtype=torch.float32)

    with open(args.features_path) as f:
        data = json.load(f)
    feature_ids = data["feature_ids"]
    print(f"Loaded {len(feature_ids)} candidate features from {args.features_path}")

    final_norm = model.model.norm
    lm_head = model.lm_head
    print("Computing logit lens top-k for candidate features only...")
    # Subset to candidates before the lm_head projection to avoid OOM on wide SAEs
    # (e.g. 65k × 256k vocab in float32 would be ~67 GB; 300 × 256k is negligible)
    W_dec_candidates = W_dec[feature_ids]  # (n_candidates, d_model)
    ll_topk_candidates = compute_logit_lens_topk(W_dec_candidates, final_norm, lm_head, k=args.logit_lens_k)
    fid_to_local = {fid: i for i, fid in enumerate(feature_ids)}

    scores = {}
    for fid in tqdm(feature_ids, desc="Output scores"):
        ll_indices = ll_topk_candidates[fid_to_local[fid]].tolist()
        score = compute_output_score(
            model, tokenizer, layer_idx, fid, ll_indices,
            args.neutral_prompt, W_dec, W_enc, b_enc, thr,
            args.amp_factor, device)
        scores[str(fid)] = score

    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_path, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"Wrote output scores for {len(scores)} features -> {args.out_path}")

    n_above = sum(1 for v in scores.values() if v >= args.threshold)
    print(f"Features with output_score >= {args.threshold}: {n_above}/{len(scores)}")

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 10 by output score:")
    for fid_str, sc in sorted_scores[:10]:
        print(f"  feature {fid_str}: {sc:.6f}")

    if args.filtered_out:
        filtered_ids = [int(fid_str) for fid_str, sc in sorted_scores if sc >= args.threshold]
        if args.top_k is not None:
            filtered_ids = filtered_ids[: args.top_k]
            print(f"Top-K: using first {len(filtered_ids)} features by output score")
        payload = {
            "feature_ids": filtered_ids,
            "n_features": len(filtered_ids),
            "source": args.features_path,
            "filter": f"output_score >= {args.threshold}" + (f", top_k={args.top_k}" if args.top_k else ""),
            "amp_factor": args.amp_factor,
        }
        Path(args.filtered_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.filtered_out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote {len(filtered_ids)} filtered features -> {args.filtered_out}")


if __name__ == "__main__":
    main()
