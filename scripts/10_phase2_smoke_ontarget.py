"""
Phase 2: On-Target Steerability Smoke Test.

For each (feature_id, alpha, prompt): run generation with steering (max 64 tokens),
compute on-target score by keyword matching, save results to CSV.
Deterministic. No plotting. Fail loudly on invalid decoder or feature index.
"""

import argparse
import csv
import logging
import os
import sys
from collections import defaultdict

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import load_config, resolve_config, save_resolved_config
from src.model_utils import load_model
from src.hook_resolver import resolve_hook_point
from src.sae_loader import load_gemmascope_decoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Alpha grid for Phase 2 smoke test
ALPHA_GRID = [0.0, 0.25, 0.5, 1.0, 2.0, 5.0]

# One topic per run. Keywords are bucket-specific for on-target score.
TOPIC_KEYWORDS = {
    "capitals": ["paris", "france", "capital", "city"],
    "arithmetic": ["equals", "=", "sum", "plus", "minus", "times", "divided", "0", "1", "3", "4", "5", "6", "7", "8", "9"],
    "planets": [k for k in ["jupiter", "planet", "solar", "saturn", "mars", "earth"] if k != "p"],
}
VALID_TOPICS = list(TOPIC_KEYWORDS.keys())


def _steering_hook_fn(v_f: torch.Tensor, alpha: float, device: str):
    """Returns a hook that does activation += alpha * v_f. Caller registers it."""
    def hook(module, input, output):
        if isinstance(output, tuple):
            activations = output[0].clone()
        else:
            activations = output.clone()
        v_f_dev = v_f.to(device=activations.device, dtype=activations.dtype)
        if len(activations.shape) == 3:
            v_f_exp = v_f_dev.unsqueeze(0).unsqueeze(0)
        else:
            v_f_exp = v_f_dev.unsqueeze(0)
        activations.add_(alpha * v_f_exp)
        if isinstance(output, tuple):
            return (activations,) + output[1:]
        return activations
    return hook


def _on_target_score(completion: str, prompt: str, keywords: list) -> float:
    """
    Score only on keywords not present in the prompt (avoids prompt echo).
    Score is 1.0 if any remaining keyword appears in completion, else 0.0.
    """
    prompt_lower = prompt.lower()
    completion_lower = completion.lower()
    keywords_not_in_prompt = [kw for kw in keywords if kw not in prompt_lower]
    if not keywords_not_in_prompt:
        return 0.0  # all keywords were in the prompt; nothing to score on
    return 1.0 if any(kw in completion_lower for kw in keywords_not_in_prompt) else 0.0


def main():
    parser = argparse.ArgumentParser(description="Phase 2 on-target steerability smoke test (one topic per run)")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Config file path")
    parser.add_argument("--run-id", type=str, default="phase2_smoke", help="Run identifier")
    parser.add_argument("--topic", type=str, required=True, choices=VALID_TOPICS,
                        help="Topic bucket: capitals, arithmetic, or planets. One topic per run.")
    args = parser.parse_args()

    config = load_config(args.config)
    config = resolve_config(config, args.run_id)
    save_resolved_config(config, args.run_id)

    sanity = config.get("sanity", {}) or {}
    feature_ids = sanity.get("feature_ids", None)
    feature_id_single = sanity.get("feature_id", None)

    if feature_ids is None:
        if feature_id_single is None:
            raise ValueError("No feature ids provided. Set sanity.feature_ids (list[int]) or sanity.feature_id (int).")
        feature_ids = [feature_id_single]

    # defensive: allow comma-separated string if someone messes up YAML later
    if isinstance(feature_ids, str):
        feature_ids = [int(x.strip()) for x in feature_ids.split(",") if x.strip()]

    if not isinstance(feature_ids, list) or not all(isinstance(x, int) for x in feature_ids):
        raise TypeError(f"sanity.feature_ids must be a list[int]. Got: {type(feature_ids)} -> {feature_ids}")

    topic = args.topic
    keywords = TOPIC_KEYWORDS[topic]
    run_id = config["experiment"]["run_id"]
    seed = config["experiment"]["seed"]
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Prompts from topic-specific file (one topic per run)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    prompts_path = os.path.join(repo_root, "data", "prompts", f"phase2_{topic}.txt")
    if not os.path.isfile(prompts_path):
        raise FileNotFoundError(f"Prompts file not found: {prompts_path}")
    with open(prompts_path, "r") as f:
        prompts = [line.strip() for line in f if line.strip()]
    if not prompts:
        raise ValueError(f"No prompts in {prompts_path}")
    logger.info(f"Topic: {topic}, prompts: {len(prompts)}, keywords: {keywords}")

    logger.info("Loading model and tokenizer...")
    model, tokenizer = load_model(config)
    device = config["model"]["device"]
    hook_point_name = config["sae"]["hook_point"]
    if hook_point_name == "TBD":
        raise ValueError("sae.hook_point must be set in config for Phase 2 smoke test")
    hook_point_module = resolve_hook_point(model, hook_point_name)

    logger.info("Loading SAE decoder...")
    decoder, _ = load_gemmascope_decoder(config)
    n_features_total = config["sae"]["n_features_total"]
    d_model = config["architecture"]["d_model"]
    assert decoder.shape == (n_features_total, d_model), (
        f"decoder.shape {decoder.shape} != ({n_features_total}, {d_model})"
    )

    for fid in feature_ids:
        assert 0 <= fid < decoder.shape[0], f"feature_id {fid} out of range [0, {decoder.shape[0]})"

    max_new_tokens = 64
    temperature = config["model"].get("temperature", 0.0)
    do_sample = config["model"].get("do_sample", False)

    out_dir = os.path.join(repo_root, "outputs", "phase2_smoke")
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"ontarget_curve_{topic}.csv")
    fieldnames = ["feature_id", "alpha", "score", "prompt_idx", "run_id", "topic"]
    total = len(feature_ids) * len(ALPHA_GRID) * len(prompts)
    done = 0
    rows = []

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for feature_id in feature_ids:
            v_f = decoder[feature_id].clone()
            for alpha in ALPHA_GRID:
                hook_fn = _steering_hook_fn(v_f, alpha, device)
                handle = hook_point_module.register_forward_hook(hook_fn)
                try:
                    for prompt_idx, prompt in enumerate(prompts):
                        inputs = tokenizer(prompt, return_tensors="pt").to(device)
                        with torch.no_grad():
                            outputs = model.generate(
                                **inputs,
                                max_new_tokens=max_new_tokens,
                                temperature=temperature if do_sample else None,
                                do_sample=do_sample,
                                pad_token_id=tokenizer.pad_token_id,
                            )
                        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
                        score = _on_target_score(generated, prompt, keywords)
                        row = {
                            "feature_id": feature_id,
                            "alpha": alpha,
                            "score": score,
                            "prompt_idx": prompt_idx,
                            "run_id": run_id,
                            "topic": topic,
                        }
                        rows.append(row)
                        writer.writerow(row)
                        done += 1
                        if done % 25 == 0:
                            logger.info(f"progress {done}/{total} (fid={feature_id}, alpha={alpha}, prompt_idx={prompt_idx})")
                        if done % 50 == 0:
                            f.flush()
                finally:
                    handle.remove()

    logger.info(f"Wrote {len(rows)} rows to {csv_path}")

    # Delta-over-baseline summary: mean(score) per (feature_id, alpha), delta vs alpha=0
    by_key = defaultdict(list)  # (feature_id, alpha) -> [scores]
    for r in rows:
        by_key[(r["feature_id"], r["alpha"])].append(r["score"])
    summary_rows = []
    for (fid, alpha), scores in sorted(by_key.items()):
        mean_score = sum(scores) / len(scores) if scores else 0.0
        baseline_scores = by_key.get((fid, 0.0), [])
        mean_baseline = sum(baseline_scores) / len(baseline_scores) if baseline_scores else 0.0
        delta = mean_score - mean_baseline
        summary_rows.append({
            "feature_id": fid,
            "alpha": alpha,
            "mean_score": round(mean_score, 4),
            "delta_over_baseline": round(delta, 4),
        })
    summary_path = os.path.join(out_dir, f"ontarget_summary_{topic}.csv")
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["feature_id", "alpha", "mean_score", "delta_over_baseline"])
        w.writeheader()
        w.writerows(summary_rows)
    logger.info(f"Wrote summary ({len(summary_rows)} rows) to {summary_path}")


if __name__ == "__main__":
    main()
