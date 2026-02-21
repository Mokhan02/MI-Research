# SAE Steerability Experiment Pipeline - Team Guide

## What This Codebase Does

This is an experiment pipeline that studies **SAE feature steerability** in language models. The core question: *Can we predict how difficult it is to steer a feature and what side effects it will have, based solely on the feature's geometric properties measured before any intervention?*

The pipeline runs 5 sequential stages:
1. **Feature Selection** - Pick 300 SAE features to study
2. **Pre-Steering Metrics** - Compute geometric/usage metrics (cosine similarity, neighbor density, co-activation)
3. **Steerability Measurement** - Find the minimum steering coefficient α* needed to achieve target behavioral change
4. **Off-Target Risk** - Measure collateral effects on unrelated benchmarks
5. **Analysis** - Merge data, run correlations, train predictive models, generate plots

## Quick Start

### Prerequisites
- Python 3.11-3.12
- `uv` package manager (install from https://github.com/astral-sh/uv)

### Setup
```bash
# Clone the repo
git clone <repo-url>
cd MI-Research

# Install dependencies
uv sync

# Run the pipeline
RUN_ID="my_experiment"
uv run python scripts/01_choose_features.py --run-id $RUN_ID
uv run python scripts/02_presteering_metrics.py --run-id $RUN_ID
uv run python scripts/03_steerability.py --run-id $RUN_ID
uv run python scripts/04_offtarget.py --run-id $RUN_ID
uv run python scripts/05_analysis.py --run-id $RUN_ID
```

Results will be in `outputs/$RUN_ID/`:
- `master.csv` - Complete merged dataset
- `alpha_star.csv` - Steerability metric per feature
- `risk.csv` - Off-target risk metrics
- `plots/` - Visualizations
- `config_resolved.yaml` - Reproducible config snapshot

## Current Status: Placeholder Mode

**The pipeline runs end-to-end right now**, but uses **placeholder components**:

✅ **What Works:**
- All 5 scripts execute successfully
- Pipeline structure and data flow
- Analysis and visualization
- File I/O and artifact management

⚠️ **What's Placeholder:**
- **SAE Loading**: Uses random decoder weights (since `sae_id="TBD"` in config)
- **Model Hooks**: Logs warnings, doesn't actually hook into model
- **Benchmark Scoring**: Hash-based proxy scores (not real benchmark scores)
- **Activation Extraction**: Random placeholder activations

**This means:** The pipeline completes and produces outputs, but the data is synthetic. This is useful for:
- Testing pipeline structure
- Verifying code logic
- Debugging before real experiments
- Training team members on the workflow

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

To run with **real SAE and model data**, you need to:

1. **Set SAE parameters** in `configs/base.yaml`:
   - `sae_id`: Your SAE identifier
   - `hook_point`: Layer name (e.g., "model.layers.5")

2. **Implement SAE loading** in `src/model_utils.py:
   - Replace `load_sae()` placeholder with actual SAE loading code
   - Currently raises `NotImplementedError` if `sae_id == "TBD"`

3. **Implement hook registration** in `scripts/03_steerability.py` and `04_offtarget.py`:
   - Replace placeholder hook registration with actual module lookup
   - Currently logs: "Hook registration is a placeholder"

4. **Implement benchmark scorers** in `src/scoring.py`:
   - Replace `PlaceholderScorer` with official scorers
   - Currently returns deterministic hash-based proxy scores

5. **Add prompt sets** in `data/prompts/`:
   - `on_target.txt` - For steerability measurement
   - `gpqa.txt`, `truthfulqa.txt`, `darkbench.txt` - For off-target evaluation
   - `reference_corpus.txt` - For pre-steering metrics

## Pipeline Stages Explained

### Stage 1: Feature Selection (`01_choose_features.py`)
Selects 300 SAE features uniformly or via stratified sampling. Outputs: `selected_features.npy`, `features.json`

### Stage 2: Pre-Steering Metrics (`02_presteering_metrics.py`)
Computes geometric metrics **before any intervention**:
- Max cosine similarity (nearest neighbor)
- Neighbor density (local crowding)
- Co-activation correlation (entanglement)
Outputs: `pre_metrics.csv`

### Stage 3: Steerability (`03_steerability.py`)
Sweeps steering coefficient α, finds minimum α* where `score(α) - score(0) ≥ threshold_T`. Outputs: `alpha_star.csv`, `ontarget_curve.csv`

### Stage 4: Off-Target Risk (`04_offtarget.py`)
Measures side effects at two regimes:
- At α*(f) - when target effect happens
- At fixed α₀=1.0 - gentle steering
Computes R_mag (mean absolute change) and R_breadth (fraction above threshold). Outputs: `risk.csv`

### Stage 5: Analysis (`05_analysis.py`)
Merges all data, runs correlations, trains predictive models, generates plots. Outputs: `master.csv`, `correlation_results.csv`, `plots/`

## Key Files

- `configs/base.yaml` - Central configuration
- `scripts/01-05_*.py` - Pipeline stages
- `scripts/99_hook_sanity.py` - Hook mechanism verification
- `src/` - Reusable utilities (config, model loading, geometry, scoring)
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

