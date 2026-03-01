# SAE Steerability Experiment Pipeline - Team Guide

## What This Codebase Does

This is an experiment pipeline that studies **SAE feature steerability** in language models. The core question: *Can we predict how difficult it is to steer a feature and what side effects it will have, based solely on the feature's geometric properties measured before any intervention?*

**Real pipeline (use only these):**
1. **Feature selection** — `phase2_select_contrast.py` (contrast: task − neutral); outputs `selected_features_planets.json`, feature summaries.
2. **Steerability** — `phase2_run.py` (real SAE decoder, α-grid, delta_logit_target, α*, censoring); uses config + prompt CSV with matched features/prompts.
3. **Prediction** — `phase3_predictability.py` / `phase3_predict_cv*.py` (correlate geometry with α*).
4. **Off-target** — To be run on Phase 2 outputs with real SAE (no script in main pipeline yet; old 04 was placeholder).

**Do not run scripts in `scripts/legacy/`.** They are the old 01–05 pipeline: fake SAE (`load_sae`), fake steering, fake scores. Any results from them would be fabricated. They are archived so nobody runs them by accident.

## Quick Start

### Prerequisites
- Python 3.11-3.12
- `uv` or `PYTHONPATH=.` for running scripts

### Setup and run (real pipeline)
```bash
# Install dependencies
uv sync

# 1. Contrast feature selection (uses real model + SAE)
PYTHONPATH=. python scripts/phase2_select_contrast.py --config configs/targets/gemma2_2b_gemmascope_res16k.yaml --domain planets --out_dir outputs/phase2_select --top-k 100

# 2. Measure steerability (real SAE, matched features + prompts)
PYTHONPATH=. python scripts/phase2_run.py --config configs/targets/gemma2_2b_gemmascope_res16k.yaml --out_dir outputs/phase2 --n_prompts 100 --fixed_features_path outputs/phase2_select/selected_features_planets.json
```

Results: `outputs/phase2_select/` (feature summaries, selected_features_planets.json), `outputs/phase2/` (run_rows.csv, alpha_star.csv, curves).

## Run gates (before full pipeline)

See **GATES.md** for the seven gates in priority order. Summary:

1. **Domain prompt splits** — Use `make_domain_splits.py` to create select/alpha/holdout CSVs per domain (no prompt overlap between selection and evaluation).
2. **Contrast-based selection** — Top K by Δ = task − neutral (`phase2_select_contrast.py`), not top K act_freq on task.
3. **Active = act > τ_act on token_span** — Set `tau_act` and `token_span` in config; keep consistent.
4. **Persist** — feature_summary_*.csv and selected_features_*.json from selection.
5. **Control set** — randK_matched_actfreq.json (or random K) for comparisons.
6. **Micro sweep first** — `phase2_run --micro_sweep` (10 features, ~25 prompts, small α grid) before full K=100.
7. **α* and no-effect** — Success = delta_logit_target ≥ T (T pre-registered). Censored = no-effect (do not set α* = max).

See **LANDMINES.md** for six traps to fix or audit before running (tau_act, token_span, T pre-registration, directionality, control matching, neutral isomorphism).

## Configuration

All parameters are in `configs/base.yaml`:

```yaml
model:
  model_id: "gpt2"  # Change to your model
  device: "cpu"     # or "cuda"

sae:
  sae_id: "TBD"           # TODO: Set real SAE ID
  hook_point: "TBD"       # TODO: Set actual layer name
  n_features_total: 1000   # Fallback for placeholder mode

features:
  n_features: 300
  selection_mode: "uniform"  # or "stratified"

steering:
  alpha_grid: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0, 5.0, 10.0]
  threshold_T: 0.10  # On-target score improvement threshold
  alpha0: 1.0        # Fixed low coefficient for risk regime 2
  tau: 0.05          # Small change threshold for R_breadth
```

## For Real Experiments

- Use **configs/targets/gemma2_2b_gemmascope_res16k.yaml** (real SAE: GemmaScope, `sae_id` set).
- **SAE loading** is in `src/sae_loader.py` (`load_gemmascope_decoder`). The real pipeline uses this; do not use `load_sae` in `src/model_utils.py` (placeholder).
- **Feature selection** and **steerability** use matched features and prompts: run `phase2_select_contrast` first, then `phase2_run` with `--fixed_features_path` pointing at its output.
- Prompt sets: domain splits in `data/prompts/` (`*_select.csv`, `*_alpha.csv`, `*_holdout.csv`); benchmark CSV in config (`benchmark.prompt_csv`).

## Pipeline Stages (real pipeline)

### Feature selection: `phase2_select_contrast.py`
Contrast-based: top K by Δ = act_freq_task − act_freq_neutral. Uses real model + SAE. Outputs: `selected_features_planets.json`, `feature_summary_planets.csv`, control set.

### Steerability: `phase2_run.py`
Real SAE decoder, α-grid, delta_logit_target, α* with censoring. Uses prompt CSV with `prompt` + `target`; features from `--fixed_features_path` or random sample. Outputs: `run_rows.csv`, `alpha_star.csv`, `curves_per_feature.parquet`.

### Prediction: `phase3_predictability.py` / `phase3_predict_cv*.py`
Correlate pre-steering geometry (and usage) with α*. Inputs: Phase 2 outputs + geometry from W_dec or pre-steering metrics.

### Archived (do not run): `scripts/legacy/`
01_choose_features, 02_presteering_metrics, 03_steerability, 04_offtarget, 05_analysis, 10/11/11b/make_feature_sets — fake SAE, fake steering, fake scores. Kept only for reference.

## Key Files

- `configs/targets/gemma2_2b_gemmascope_res16k.yaml` - Target config (model, SAE, benchmark, steering)
- `scripts/phase2_select_contrast.py`, `scripts/phase2_run.py` - Core pipeline
- `scripts/phase3_predictability.py`, `scripts/phase3_predict_cv.py` - Phase 3
- `src/sae_loader.py` - Real SAE (load_gemmascope_decoder)
- `src/model_utils.py` - load_model (real); load_sae is placeholder, unused by real pipeline
- `outputs/<run_id>/` - Experiment results

## Design Principles

- **Stage-based**: Each script is independent, communicates via files
- **Deterministic**: Seed-controlled, reproducible
- **No hidden state**: Everything explicit in config/artifacts
- **Placeholder-friendly**: Gracefully handles missing components
- **Reproducible**: Every script saves resolved config

## Troubleshooting

**"SAE loading failed"**: Expected in placeholder mode. Set `sae_id` and implement `load_sae()` for real runs.

**"Model loading failed"**: Check disk space and model access. For placeholder mode, scripts handle this gracefully.

**"Hook registration is a placeholder"**: Expected. Implement actual hook registration for real experiments.

**Missing prompt files**: Scripts fall back to placeholder prompts. Add real prompts to `data/prompts/` for real runs.

## Questions?

See the research doc for experimental design details:
https://docs.google.com/document/d/1c-gnQnJlBCQvK3M607sx0QAn8XebIgnVtDrnkB0pQQ0/edit

