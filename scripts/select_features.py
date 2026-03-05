"""
Feature selection for steering experiments.

Two selection modes (--selection-mode):
  contrast_delta  : top K by Δ_freq = act_freq_task − act_freq_neutral.
                    Finds features that *activate* more on task prompts, but
                    these may be content detectors rather than refusal drivers.
  logit_attr      : top K by activation-weighted logit attribution toward
                    refusal tokens.  Ranks features by how much they *push the
                    output distribution toward refusal* when they fire on task
                    prompts.  Requires delta_freq > 0 as a pre-filter.

Uses domain-specific *_select.csv prompts only (no overlap with alpha or holdout).
Outputs:
  - feature_summary_{task_domain}.csv  (act_freq_task, act_freq_neutral, delta_freq, delta_mean, logit_attr)
  - feature_summary_neutral.csv
  - selected_features_{task_domain}.json  (top K by chosen criterion)
  - randK_matched_actfreq.json  (random K control set)

Activation rule: active = (act > tau_act) on token_span (config: tau_act, token_span).
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
) -> np.ndarray:
    """Return (n_prompts, n_features) raw pre-threshold activations (resid @ W_dec.T at hook)."""
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


def main():
    ap = argparse.ArgumentParser(description="Contrast-based feature selection (task − neutral)")
    ap.add_argument("--config", type=str, default="configs/targets/gemma2_2b_gemmascope_res16k.yaml")
    ap.add_argument("--domain", type=str, default="salad",
                    help="Task domain (selection uses {domain}_select.csv vs neutral_select.csv)")
    ap.add_argument("--prompts_dir", type=str, default="data/prompts")
    ap.add_argument("--out_dir", type=str, default="outputs/phase2_select")
    ap.add_argument("--selection-mode", type=str, default="logit_attr",
                    choices=["contrast_delta", "logit_attr"],
                    help="Ranking criterion: contrast_delta (Δ act_freq) or logit_attr (activation-weighted logit attribution toward refusal tokens)")
    ap.add_argument("--refusal-tokens", type=str, default=None,
                    help="Comma-separated refusal seed strings for logit_attr mode (default: config or built-in list)")
    ap.add_argument("--top-k", type=int, default=100, help="Top K features to save as selected")
    ap.add_argument("--rand-k", type=int, default=100, help="Size of random control set")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--tau-act", type=float, default=None, help="Activation threshold (default: config features.tau_act)")
    ap.add_argument("--token-span", type=str, default=None, choices=["last", "last_n", "all"])
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

    task_acts = get_feature_acts(
        model, tokenizer, hook_module, W_dec, task_select,
        args.batch_size, device, token_span=token_span, last_n=last_n,
    )
    neutral_acts = get_feature_acts(
        model, tokenizer, hook_module, W_dec, neutral_select,
        args.batch_size, device, token_span=token_span, last_n=last_n,
    )

    act_freq_task, mean_act_task = act_freq_and_mean(task_acts, tau_act)
    act_freq_neutral, mean_act_neutral = act_freq_and_mean(neutral_acts, tau_act)
    delta_freq = act_freq_task - act_freq_neutral
    delta_mean = np.nan_to_num(mean_act_task, nan=0) - np.nan_to_num(mean_act_neutral, nan=0)

    # --- Logit attribution toward refusal tokens ---
    # For each feature j: logit_attr_j = mean_over_prompts(act_j) * (W_dec[j] @ W_unembed[refusal_tok].mean)
    # This measures how much the feature pushes next-token logits toward refusal when it fires.
    DEFAULT_REFUSAL_SEEDS = [" I", " Sorry", " cannot", " apologize", " unfortunately",
                             " shouldn", " won", " don", " isn"]
    if args.refusal_tokens:
        refusal_seeds = [s.strip() for s in args.refusal_tokens.split(",")]
    else:
        refusal_seeds = config.get("features", {}).get("refusal_tokens", DEFAULT_REFUSAL_SEEDS)

    refusal_tok_ids = []
    for s in refusal_seeds:
        # Ensure leading space for word-initial tokens
        if not s.startswith(" "):
            s = " " + s
        ids = tokenizer.encode(s, add_special_tokens=False)
        if ids:
            refusal_tok_ids.append(ids[0])
    refusal_tok_ids = list(set(refusal_tok_ids))
    logger.info("Refusal seed tokens (%d unique): %s", len(refusal_tok_ids),
                [(tid, tokenizer.decode([tid])) for tid in refusal_tok_ids[:15]])

    W_unembed = model.lm_head.weight.detach().float()  # (vocab, d_model)
    refusal_dir = W_unembed[refusal_tok_ids].mean(dim=0)  # (d_model,)
    refusal_dir = refusal_dir / refusal_dir.norm().clamp(min=1e-12)  # unit vector

    # Per-feature projection onto refusal direction in output space
    logit_attr_per_feat = (W_dec.float() @ refusal_dir.to(W_dec.device)).cpu().numpy()  # (n_features,)

    # Activation-weighted attribution: mean activation on task prompts × logit projection
    mean_act_task_clean = np.nan_to_num(mean_act_task, nan=0.0)
    weighted_logit_attr = mean_act_task_clean * logit_attr_per_feat  # (n_features,)
    logger.info("Logit attr stats: proj min=%.4f max=%.4f; weighted min=%.4f max=%.4f",
                logit_attr_per_feat.min(), logit_attr_per_feat.max(),
                weighted_logit_attr.min(), weighted_logit_attr.max())

    # Feature summary for task domain
    summary_task = pd.DataFrame({
        "feature_id": np.arange(n_features_total),
        "act_freq_task": act_freq_task,
        "act_freq_neutral": act_freq_neutral,
        "delta_freq": delta_freq,
        "mean_act_task": mean_act_task,
        "mean_act_neutral": mean_act_neutral,
        "delta_mean": delta_mean,
        "logit_attr_proj": logit_attr_per_feat,
        "weighted_logit_attr": weighted_logit_attr,
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

    # --- Rank and select top K ---
    top_k = min(args.top_k, n_features_total)
    selection_mode = args.selection_mode

    if selection_mode == "logit_attr":
        # Rank by weighted_logit_attr, but require delta_freq > 0 (must activate more on task)
        ranking_score = weighted_logit_attr.copy()
        ranking_score[delta_freq <= 0] = -np.inf
        order = np.argsort(-ranking_score)
        # Drop any that got -inf (shouldn't appear in top K unless <K candidates)
        top_ids = [int(i) for i in order[:top_k] if np.isfinite(ranking_score[i])]
        logger.info("Selection mode: logit_attr (weighted logit attribution, delta_freq>0 filter)")
        logger.info("Top 5 weighted_logit_attr scores: %s",
                     [(tid, f"{weighted_logit_attr[tid]:.4f}") for tid in top_ids[:5]])
    else:
        # Original: rank by delta_freq
        order = np.argsort(-delta_freq)
        top_ids = order[:top_k].tolist()
        logger.info("Selection mode: contrast_delta (Δ act_freq)")

    payload = {
        "feature_ids": top_ids,
        "n_features": len(top_ids),
        "domain": args.domain,
        "selection": selection_mode,
        "tau_act": tau_act,
        "token_span": token_span,
        "seed": args.seed,
    }
    if selection_mode == "logit_attr":
        payload["refusal_seeds"] = refusal_seeds

    selected_path = out_dir / f"selected_features_{args.domain}.json"
    with open(selected_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Wrote %s (top %d by %s)", selected_path, len(top_ids), selection_mode)

    # Also write topK_{domain}_{mode}.json for reproducibility
    topk_path = out_dir / f"topK_{args.domain}_{selection_mode}.json"
    with open(topk_path, "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Wrote %s", topk_path)

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
        f.write(f"Control audit: task-selected (top K by {selection_mode}) vs random K\n")
        f.write(f"  act_freq_task:  task mean={task_act_freq.mean():.4f} std={task_act_freq.std():.4f}  rand mean={rand_act_freq.mean():.4f} std={rand_act_freq.std():.4f}\n")
        f.write(f"  mean_act_task:  task mean={np.nanmean(task_mean_act):.4f}  rand mean={np.nanmean(rand_mean_act):.4f}\n")
        f.write(f"  ||W_dec||:     task mean={task_norms.mean():.4f} std={task_norms.std():.4f}  rand mean={rand_norms.mean():.4f} std={rand_norms.std():.4f}\n")
        f.write("  If ||W_dec|| differs a lot, match on norm or control for it in analysis.\n")
    logger.info("Wrote %s", audit_path)
    logger.info("Top 10 by delta_freq: %s", top_ids[:10])


if __name__ == "__main__":
    main()
