"""
Phase 2 prefilter: identify candidate arithmetic features by activation lift.

For each prompt (arithmetic vs control), compute residual stream at the hook point,
project onto SAE features: acts = resid @ W_dec.T. Aggregate mean_act_arith, mean_act_ctrl,
lift = mean_act_arith - mean_act_ctrl. Output top-K features by lift for steering.
"""

import argparse
import csv
import logging
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import load_config, resolve_config
from src.model_utils import load_model
from src.hook_resolver import resolve_hook_point
from src.sae_loader import load_gemmascope_decoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _load_prompts(path: str) -> list[str]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Prompts file not found: {path}")
    with open(path, "r") as f:
        prompts = [line.strip() for line in f if line.strip()]
    if not prompts:
        raise ValueError(f"No prompts in {path}")
    return prompts


def main():
    parser = argparse.ArgumentParser(description="Phase 2 prefilter: arithmetic activation lift vs control")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Config file path")
    parser.add_argument("--run-id", type=str, default="phase2_prefilter", help="Run identifier")
    parser.add_argument("--max-arith", type=int, default=None, help="Cap arithmetic prompts (None = all)")
    parser.add_argument("--max-ctrl", type=int, default=None, help="Cap control prompts (None = all)")
    parser.add_argument("--top-k", type=int, default=50, help="Number of top features by lift to output")
    parser.add_argument("--eps", type=float, default=1e-6, help="Epsilon for ratio (mean_arith+eps)/(mean_ctrl+eps)")
    parser.add_argument("--out", type=str, default=None, help="Output CSV path (default: outputs/phase2_smoke/prefilter_lift.csv)")
    args = parser.parse_args()

    config = load_config(args.config)
    config = resolve_config(config, args.run_id)

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    arith_path = os.path.join(repo_root, "data", "prompts", "phase2_arithmetic.txt")
    ctrl_path = os.path.join(repo_root, "data", "prompts", "phase2_control.txt")
    arith_prompts = _load_prompts(arith_path)
    ctrl_prompts = _load_prompts(ctrl_path)
    if args.max_arith is not None:
        arith_prompts = arith_prompts[: args.max_arith]
    if args.max_ctrl is not None:
        ctrl_prompts = ctrl_prompts[: args.max_ctrl]
    logger.info(f"Arithmetic prompts: {len(arith_prompts)}, control prompts: {len(ctrl_prompts)}")

    logger.info("Loading model and tokenizer...")
    model, tokenizer = load_model(config)
    device = config["model"]["device"]
    hook_point_name = config["sae"]["hook_point"]
    if hook_point_name == "TBD":
        raise ValueError("sae.hook_point must be set in config")
    hook_point_module = resolve_hook_point(model, hook_point_name)

    logger.info("Loading SAE decoder...")
    decoder, _ = load_gemmascope_decoder(config)
    n_features_total = config["sae"]["n_features_total"]
    d_model = config["architecture"]["d_model"]
    assert decoder.shape == (n_features_total, d_model), (
        f"decoder.shape {decoder.shape} != ({n_features_total}, {d_model})"
    )
    W_dec = decoder.to(device)  # (n_features, d_model)

    captured = []

    def hook_fn(module, input, output):
        out = output[0] if isinstance(output, tuple) else output
        captured.append(out.detach())

    handle = hook_point_module.register_forward_hook(hook_fn)

    def get_feature_acts_for_prompts(prompts: list[str]) -> torch.Tensor:
        """Run forward on each prompt, capture resid at last position, project to features. Return (n_prompts, n_features)."""
        acts_list = []
        with torch.no_grad():
            for prompt in prompts:
                captured.clear()
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                model(**inputs)
                if not captured:
                    raise RuntimeError("Hook did not capture activations")
                resid = captured[0]  # (batch, seq, d_model)
                last_pos = resid[:, -1, :]  # (batch, d_model)
                acts = last_pos @ W_dec.T  # (batch, n_features)
                acts_list.append(acts)
        return torch.cat(acts_list, dim=0)  # (n_prompts, n_features)

    try:
        logger.info("Computing feature activations on arithmetic prompts...")
        arith_acts = get_feature_acts_for_prompts(arith_prompts)  # (n_arith, n_features)
        logger.info("Computing feature activations on control prompts...")
        ctrl_acts = get_feature_acts_for_prompts(ctrl_prompts)  # (n_ctrl, n_features)
    finally:
        handle.remove()

    mean_act_arith = arith_acts.float().mean(dim=0).cpu()  # (n_features,)
    mean_act_ctrl = ctrl_acts.float().mean(dim=0).cpu()  # (n_features,)
    lift = mean_act_arith - mean_act_ctrl  # (n_features,)
    ratio = (mean_act_arith + args.eps) / (mean_act_ctrl + args.eps)  # (n_features,)

    _, top_indices = torch.topk(lift, min(args.top_k, lift.shape[0]), largest=True)
    top_indices = top_indices.tolist()

    out_dir = os.path.join(repo_root, "outputs", "phase2_smoke")
    os.makedirs(out_dir, exist_ok=True)
    out_path = args.out or os.path.join(out_dir, "prefilter_lift.csv")

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["feature_id", "mean_act_arith", "mean_act_ctrl", "lift", "ratio"])
        for fid in top_indices:
            w.writerow([
                fid,
                round(mean_act_arith[fid].item(), 6),
                round(mean_act_ctrl[fid].item(), 6),
                round(lift[fid].item(), 6),
                round(ratio[fid].item(), 4),
            ])
    logger.info(f"Wrote top-{len(top_indices)} features by lift to {out_path}")

    logger.info("Top 10 feature_ids by lift (use for sanity.feature_ids): %s", top_indices[:10])


if __name__ == "__main__":
    main()
