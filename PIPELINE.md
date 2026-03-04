# Pipeline

## What This Project Is

We test whether geometric properties of SAE decoder features — max cosine similarity, neighbor density, and coactivation correlation — predict how easily a feature can steer model behavior (measured by alpha\*, the minimum steering multiplier needed to shift outputs past a pre-registered threshold). Experiments run on Gemma 2-2B with GemmaScope SAE (layer 20, width 16k), using SALADBench refusal-rate as the steerability benchmark.

## Pipeline (Run In Order)

### 1. `scripts/prepare_prompts.py`
Downloads SALADBench from HuggingFace, samples prompts stratified by safety category, and splits into select/alpha/holdout CSVs. Also writes a neutral set from HH-harmless.
- **Inputs:** HuggingFace SALADBench dataset
- **Outputs:** `data/prompts/salad_select.csv`, `salad_alpha.csv`, `salad_holdout.csv`, `neutral_select.csv`

### 2. `scripts/select_features.py`
Contrast-based feature selection: runs SAE on task vs neutral prompts and selects features with the largest activation frequency difference (delta_freq).
- **Inputs:** `data/prompts/salad_select.csv`, `data/prompts/neutral_select.csv`, SAE config
- **Outputs:** `outputs/phase2_select/selected_features_salad.json`, `feature_summary_salad.csv`

### 3. `scripts/run_sweep.py`
Steers each selected feature at multiple alpha values and records per-prompt metrics (delta logit, TV distance, refusal rate). Computes alpha\* per feature and flags censored features.
- **Inputs:** `selected_features_salad.json`, `data/prompts/salad_alpha.csv`, SAE config
- **Outputs:** `outputs/<run_name>/run_rows.csv`, `feature_summary.csv`, `alpha_star.csv`

### 4. `scripts/analyze_geometry.py`
Primary analysis: Spearman rank correlation between SAE decoder geometry and log(alpha\*), with bootstrap CIs and censored-feature sensitivity analysis. Secondary (supplementary): binary classification of steerable vs non-steerable.
- **Inputs:** `feature_summary.csv` from run_sweep (or pre-merged CSV via `--input_csv`)
- **Outputs:** `outputs/phase3_analysis/plots/`, `outputs/correlation_results.csv`, `outputs/summary_stats.csv`, `outputs/classification_supplementary.csv`

## Pre-Flight Checklist

Run these before any experiment:

- **`scripts/check_hooks.py`** — Verify SAE hooks fire correctly at the expected layer
- **`scripts/audit_activations.py`** — Inspect activation distributions and set `tau_act` (must NOT be 0.0)
- **`scripts/verify_target.py`** — Confirm target tokens are valid and in the tokenizer vocabulary
- **Pre-register threshold T** in `configs/base.yaml` before running `run_sweep.py` — never adjust T after seeing results

## Folder Structure

```
configs/          Config YAML files (base.yaml, targets/)
data/prompts/     Prompt CSVs for SALADBench and neutral baselines
scripts/          All executable scripts (pipeline + utilities)
src/              Shared library code (config, model loading, SAE, hooks, refusal scoring)
outputs/          All experiment outputs (gitignored except .gitkeep)
archive/          Archived legacy scripts and data from planets/capitals/arithmetic experiments
docs/             Design documents and proposals
```

## What To Watch Out For

- **`tau_act` must not be 0.0** — use `audit_activations.py` to pick a sensible threshold; 0.0 counts near-zero noise as activations
- **Threshold T must be pre-registered** — set it in `configs/base.yaml` before `run_sweep.py`; never adjust after seeing results
- **Censored features** (alpha\* never hit T even at alpha=10) are flagged with `censored=True` and assigned alpha\*=10 for tied maximum rank; `analyze_geometry.py` runs sensitivity analysis both with and without them
- **Run a micro-sweep first** (10 features) before committing to a full experiment to catch config or hook issues early
