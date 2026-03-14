"""
Contrast-based feature selection: top K by Δ = task − neutral (not top K act_freq on task).

Uses domain-specific *_select.csv prompts only (no overlap with alpha or holdout).
Outputs:
  - feature_summary_{task_domain}.csv  (act_freq, delta_freq, delta_mean, composite_score, strategy)
  - feature_summary_neutral.csv
  - selected_features_{task_domain}.json  (top K by composite_score or delta_freq)
  - randK_matched_actfreq.json  (random K control set)

Activation rule: active = (act > tau_act) on token_span (config: tau_act, token_span).

Scoring modes (--scoring):
  delta_freq:  Legacy: rank by delta_freq = act_freq_task - act_freq_neutral.
  composite:   Bhargav & Zhu (arXiv 2511.00029) composite score:
               score_f = w1 * (|norm_diff_mean_f| / max) + w2 * (1 - norm_variance_f)
               with thresholds on score, |norm_diff_mean|, and variance.
               Also writes strategy (suppress / amplify) from sign of diff_mean.

Pooling modes (--pooling):
  mean:  Mean over token span (default, legacy).
  max:   Max over token span (CorrSteer recommendation).
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import load_config, resolve_config
from src.model_utils import load_model
from src.hook_resolver import resolve_hook_point
from src.sae_loader import load_gemmascope_decoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_prompts_from_csv(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Prompt CSV not found: {path}")
    df = pd.read_csv(path, dtype={"prompt": "string"})
    prompts = df["prompt"].dropna().astype(str).str.strip().tolist()
    prompts = [p for p in prompts if p]
    if not prompts:
        raise ValueError(f"No prompts in {path}")
    return prompts


def get_feature_acts(
    model, tokenizer, hook_module, W_dec, prompts: list[str],
    batch_size: int, device, token_span: str = "last", last_n: int = 1,
    pooling: str = "mean",
) -> np.ndarray:
    """Return (n_prompts, n_features) raw pre-threshold activations (resid @ W_dec.T at hook).

    pooling: how to aggregate across tokens in the span.
      "mean" — average residual over the span, then project (legacy).
      "max"  — project each token separately, then take max activation per feature (CorrSteer).
    """
    captured = []

    def hook_fn(module, input, output):
        out = output[0] if isinstance(output, tuple) else output
        captured.append(out.detach())

    handle = hook_module.register_forward_hook(hook_fn)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    acts_list = []
    try:
        with torch.no_grad():
            for i in range(0, len(prompts), batch_size):
                batch = prompts[i : i + batch_size]
                captured.clear()
                inputs = tokenizer(
                    batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                ).to(device)
                model(**inputs)
                if not captured:
                    raise RuntimeError("Hook did not capture")
                resid = captured[0]  # (batch, seq, d_model)
                attn = inputs.get("attention_mask")

                if pooling == "max":
                    resid_f = resid.float()
                    all_acts = resid_f @ W_dec.T  # (batch, seq, n_features)
                    if attn is not None:
                        mask = attn.unsqueeze(-1).float()
                        all_acts = all_acts * mask + (-1e9) * (1 - mask)
                    acts = all_acts.max(dim=1).values  # (batch, n_features)
                else:
                    if attn is not None:
                        last_idx = (attn.sum(dim=1) - 1).clamp(min=0)
                        batch_dim = torch.arange(resid.shape[0], device=resid.device)
                        if token_span == "last":
                            pos = resid[batch_dim, last_idx, :]
                        elif token_span == "last_n":
                            start = (last_idx - last_n + 1).clamp(min=0)
                            pos = torch.stack([
                                resid[b, start[b]:last_idx[b] + 1, :].mean(dim=0) for b in range(resid.shape[0])
                            ])
                        else:
                            pos = resid.mean(dim=1)
                    else:
                        pos = resid[:, -last_n:, :].mean(dim=1) if token_span == "last_n" else resid[:, -1, :]
                    pos = pos.float()
                    acts = pos @ W_dec.T

                acts_list.append(acts.cpu().numpy())
        return np.concatenate(acts_list, axis=0)
    finally:
        handle.remove()


def act_freq_and_mean(acts: np.ndarray, tau_act: float):
    """acts: (n_prompts, n_features). Return (act_freq, mean_act) per feature."""
    active = acts > tau_act
    act_freq = active.mean(axis=0).astype(np.float64)
    mean_act = np.zeros(acts.shape[1], dtype=np.float64)
    for j in range(acts.shape[1]):
        if np.any(active[:, j]):
            mean_act[j] = np.mean(acts[active[:, j], j])
        else:
            mean_act[j] = np.nan
    return act_freq, mean_act


def composite_score(
    task_acts: np.ndarray,
    neutral_acts: np.ndarray,
    tau_act: float,
    w1: float = 0.5,
    w2: float = 0.5,
    min_score: float = 0.0,
    min_diff_mean: float = 0.0,
    max_variance: float = float("inf"),
):
    """Bhargav & Zhu-style composite feature scoring.

    Returns (scores, strategies, diff_mean, variance) arrays of shape (n_features,).
    strategy[f] ∈ {"amplify", "suppress"} from sign of diff_mean.
    """
    n_features = task_acts.shape[1]
    active_task = task_acts > tau_act
    active_neutral = neutral_acts > tau_act

    mean_task = np.nanmean(np.where(active_task, task_acts, np.nan), axis=0)
    mean_neutral = np.nanmean(np.where(active_neutral, neutral_acts, np.nan), axis=0)
    mean_task = np.nan_to_num(mean_task, nan=0.0)
    mean_neutral = np.nan_to_num(mean_neutral, nan=0.0)

    diff_mean = mean_task - mean_neutral
    abs_diff = np.abs(diff_mean)
    max_abs = abs_diff.max() if abs_diff.max() > 0 else 1.0
    norm_diff = abs_diff / max_abs

    variance = np.nanvar(np.where(active_task, task_acts, np.nan), axis=0)
    variance = np.nan_to_num(variance, nan=0.0)
    max_var = variance.max() if variance.max() > 0 else 1.0
    norm_var = variance / max_var

    scores = w1 * norm_diff + w2 * (1.0 - norm_var)

    mask = (scores >= min_score) & (abs_diff >= min_diff_mean) & (variance <= max_variance)
    scores = np.where(mask, scores, 0.0)

    strategies = np.where(diff_mean >= 0, "amplify", "suppress")
    return scores, strategies, diff_mean, variance


def main():
    ap = argparse.ArgumentParser(description="Contrast-based feature selection (task − neutral)")
    ap.add_argument("--config", type=str, default="configs/targets/gemma2_2b_gemmascope_res16k.yaml")
    ap.add_argument("--domain", type=str, default="planets",
                    help="Task domain (selection uses {domain}_select.csv vs neutral_select.csv)")
    ap.add_argument("--prompts_dir", type=str, default="data/prompts")
    ap.add_argument("--out_dir", type=str, default="outputs/phase2_select")
    ap.add_argument("--top-k", type=int, default=100, help="Top K by composite_score or delta_freq")
    ap.add_argument("--rand-k", type=int, default=100, help="Size of random control set")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--tau-act", type=float, default=None, help="Activation threshold (default: config features.tau_act)")
    ap.add_argument("--token-span", type=str, default=None, choices=["last", "last_n", "all"])
    ap.add_argument("--scoring", type=str, default="composite", choices=["delta_freq", "composite"],
                    help="Scoring strategy: delta_freq (legacy) or composite (Bhargav & Zhu)")
    ap.add_argument("--pooling", type=str, default="mean", choices=["mean", "max"],
                    help="Token-level pooling: mean (legacy) or max (CorrSteer)")
    ap.add_argument("--comp-w1", type=float, default=0.5, help="Composite weight on |norm_diff_mean|")
    ap.add_argument("--comp-w2", type=float, default=0.5, help="Composite weight on (1 - norm_variance)")
    ap.add_argument("--comp-min-score", type=float, default=0.0, help="Min composite score threshold")
    ap.add_argument("--comp-min-diff", type=float, default=0.0, help="Min |diff_mean| threshold")
    ap.add_argument("--comp-max-var", type=float, default=float("inf"), help="Max variance threshold")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = load_config(args.config)
    config = resolve_config(config, "phase2_select")
    repo_root = Path(__file__).resolve().parents[1]
    prompts_dir = Path(args.prompts_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tau_act = args.tau_act if args.tau_act is not None else float(config.get("features", {}).get("tau_act", 0.0))
    token_span = args.token_span or config.get("features", {}).get("token_span", "last")
    last_n = int(config.get("features", {}).get("token_span_last_n", 1))
    logger.info("Activation rule: active = (act > %.4f) on token_span=%s (last_n=%d)", tau_act, token_span, last_n)

    task_select = load_prompts_from_csv(prompts_dir / f"{args.domain}_select.csv")
    neutral_select = load_prompts_from_csv(prompts_dir / "neutral_select.csv")
    logger.info("Task (%s) select prompts: %d; neutral select prompts: %d", args.domain, len(task_select), len(neutral_select))

    logger.info("Loading model and SAE...")
    model, tokenizer = load_model(config)
    model.eval()
    device = next(model.parameters()).device
    hook_point_name = config["sae"]["hook_point"]
    if hook_point_name in ("TBD", "", None):
        raise ValueError("sae.hook_point must be set")
    hook_module = resolve_hook_point(model, hook_point_name)
    decoder, _ = load_gemmascope_decoder(config)
    n_features_total = config["sae"]["n_features_total"]
    d_model = config["architecture"]["d_model"]
    W_dec = decoder.to(device=device, dtype=torch.float32)
    assert W_dec.shape == (n_features_total, d_model), f"W_dec {W_dec.shape}"

    logger.info("Pooling mode: %s", args.pooling)
    task_acts = get_feature_acts(
        model, tokenizer, hook_module, W_dec, task_select,
        args.batch_size, device, token_span=token_span, last_n=last_n,
        pooling=args.pooling,
    )
    neutral_acts = get_feature_acts(
        model, tokenizer, hook_module, W_dec, neutral_select,
        args.batch_size, device, token_span=token_span, last_n=last_n,
        pooling=args.pooling,
    )

    act_freq_task, mean_act_task = act_freq_and_mean(task_acts, tau_act)
    act_freq_neutral, mean_act_neutral = act_freq_and_mean(neutral_acts, tau_act)
    delta_freq = act_freq_task - act_freq_neutral
    delta_mean = np.nan_to_num(mean_act_task, nan=0) - np.nan_to_num(mean_act_neutral, nan=0)

    # --- Composite scoring (Bhargav & Zhu) ---
    if args.scoring == "composite":
        comp_scores, strategies, diff_mean_raw, variance_raw = composite_score(
            task_acts, neutral_acts, tau_act,
            w1=args.comp_w1, w2=args.comp_w2,
            min_score=args.comp_min_score,
            min_diff_mean=args.comp_min_diff,
            max_variance=args.comp_max_var,
        )
    else:
        comp_scores = np.zeros(n_features_total)
        strategies = np.full(n_features_total, "n/a")
        diff_mean_raw = delta_mean
        variance_raw = np.zeros(n_features_total)

    # Feature summary for task domain
    summary_task = pd.DataFrame({
        "feature_id": np.arange(n_features_total),
        "act_freq_task": act_freq_task,
        "act_freq_neutral": act_freq_neutral,
        "delta_freq": delta_freq,
        "mean_act_task": mean_act_task,
        "mean_act_neutral": mean_act_neutral,
        "delta_mean": delta_mean,
        "composite_score": comp_scores,
        "strategy": strategies,
        "diff_mean_raw": diff_mean_raw,
        "variance": variance_raw,
    })
    summary_task_path = out_dir / f"feature_summary_{args.domain}.csv"
    summary_task.to_csv(summary_task_path, index=False)
    logger.info("Wrote %s", summary_task_path)

    summary_neutral = pd.DataFrame({
        "feature_id": np.arange(n_features_total),
        "act_freq": act_freq_neutral,
        "mean_act": mean_act_neutral,
    })
    summary_neutral_path = out_dir / "feature_summary_neutral.csv"
    summary_neutral.to_csv(summary_neutral_path, index=False)
    logger.info("Wrote %s", summary_neutral_path)

    # Top K by the selected scoring strategy
    top_k = min(args.top_k, n_features_total)
    if args.scoring == "composite":
        rank_metric = comp_scores
        selection_label = "contrast_composite"
    else:
        rank_metric = delta_freq
        selection_label = "contrast_delta_freq"

    order = np.argsort(-rank_metric)
    top_ids = order[:top_k].tolist()
    top_strategies = [strategies[i] for i in top_ids]

    payload = {
        "feature_ids": top_ids,
        "strategies": top_strategies,
        "n_features": len(top_ids),
        "domain": args.domain,
        "selection": selection_label,
        "scoring": args.scoring,
        "pooling": args.pooling,
        "tau_act": tau_act,
        "token_span": token_span,
        "seed": args.seed,
    }
    selected_path = out_dir / f"selected_features_{args.domain}.json"
    with open(selected_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Wrote %s (top %d by %s)", selected_path, len(top_ids), selection_label)

    topk_delta_path = out_dir / f"topK_{args.domain}_{args.scoring}.json"
    with open(topk_delta_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Wrote %s", topk_delta_path)

    # Random K control (optionally matched on act_freq later; here plain random)
    rand_ids = np.random.choice(n_features_total, size=min(args.rand_k, n_features_total), replace=False).tolist()
    rand_path = out_dir / "randK_matched_actfreq.json"
    with open(rand_path, "w") as f:
        json.dump({
            "feature_ids": rand_ids,
            "n_features": len(rand_ids),
            "note": "random K control (not yet matched on act_freq)",
            "seed": args.seed,
        }, f, indent=2)
    logger.info("Wrote %s", rand_path)

    # Control audit: compare task-selected vs random on act_freq_task, ||W_dec||, mean_act
    # If distributions differ a lot, match on norm too or control in analysis
    W_norms = np.linalg.norm(W_dec.cpu().numpy(), axis=1)
    task_act_freq = np.array([act_freq_task[i] for i in top_ids])
    task_mean_act = np.array([mean_act_task[i] for i in top_ids])
    task_norms = np.array([W_norms[i] for i in top_ids])
    rand_act_freq = np.array([act_freq_task[i] for i in rand_ids])
    rand_mean_act = np.array([mean_act_task[i] for i in rand_ids])
    rand_norms = np.array([W_norms[i] for i in rand_ids])
    audit_path = out_dir / "control_audit.txt"
    with open(audit_path, "w") as f:
        f.write("Control audit: task-selected (top K by delta_freq) vs random K\n")
        f.write(f"  act_freq_task:  task mean={task_act_freq.mean():.4f} std={task_act_freq.std():.4f}  rand mean={rand_act_freq.mean():.4f} std={rand_act_freq.std():.4f}\n")
        f.write(f"  mean_act_task:  task mean={np.nanmean(task_mean_act):.4f}  rand mean={np.nanmean(rand_mean_act):.4f}\n")
        f.write(f"  ||W_dec||:     task mean={task_norms.mean():.4f} std={task_norms.std():.4f}  rand mean={rand_norms.mean():.4f} std={rand_norms.std():.4f}\n")
        f.write("  If ||W_dec|| differs a lot, match on norm or control for it in analysis.\n")
    logger.info("Wrote %s", audit_path)
    logger.info("Top 10 by delta_freq: %s", top_ids[:10])


if __name__ == "__main__":
    main()
