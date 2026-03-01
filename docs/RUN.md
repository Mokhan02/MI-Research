# How to run the pipeline

Minimal path to run the proposal pipeline (Phase 2 steerability → Phase 3 prediction). Use from repo root with `PYTHONPATH=.` or `uv run`.

## Prerequisites

- **Config:** `configs/targets/gemma2_2b_gemmascope_res16k.yaml` (model, SAE, benchmark, steering.alpha_grid, steering.threshold_T).
- **Prompt CSVs:** Must have columns `prompt` and `target`; every row must have a non-empty target (used for delta_logit_target).
- **Data layout:** Domain splits in `data/prompts/`:
  - `planets_select.csv`, `planets_alpha.csv`, `planets_holdout.csv`
  - `neutral_select.csv`, `neutral_alpha.csv`, `neutral_holdout.csv`
  - (Optional) same for `capitals`.

Config’s `benchmark.prompt_csv` points at `data/prompts/planets_alpha.csv` by default.

## 1. Feature selection (contrast: task − neutral)

Uses **select** splits only. Writes selected features and control set.

```bash
PYTHONPATH=. python scripts/phase2_select_contrast.py \
  --config configs/targets/gemma2_2b_gemmascope_res16k.yaml \
  --domain planets \
  --prompts_dir data/prompts \
  --out_dir outputs/phase2_select \
  --top-k 100 \
  --rand-k 100
```

**Outputs (in `out_dir`):**

- `feature_summary_planets.csv`, `feature_summary_neutral.csv`
- `selected_features_planets.json` — **use this for steering**
- `topK_planets_delta.json`, `randK_matched_actfreq.json`
- `control_audit.txt`

## 2. Phase 2: measure steerability (α-grid, α*)

Uses **alpha** split prompts and optional fixed feature list. One row per (prompt, feature, α); then α* and curves per feature.

```bash
PYTHONPATH=. python scripts/phase2_run.py \
  --config configs/targets/gemma2_2b_gemmascope_res16k.yaml \
  --out_dir outputs/phase2 \
  --n_prompts 100 \
  --fixed_features_path outputs/phase2_select/selected_features_planets.json
```

- Omit `--fixed_features_path` to sample `--n_features` (default 300) at random.
- Override prompt set: `--prompt_csv data/prompts/planets_alpha.csv` (or leave default from config).
- Micro sweep (fewer features/prompts/alphas): add `--micro_sweep`.

**Outputs (in `out_dir`):**

- `run_rows.csv` — raw (prompt_idx, feature_id, alpha, delta_logit_target, …)
- `alpha_star.csv` — α* and censored flags per feature
- `curves_per_feature.parquet`, `selected_features.json`

## 3. Phase 3: predict steerability from pre-steering metrics

Needs Phase 1 metrics (e.g. from `02_presteering_metrics.py`) merged with Phase 2 α*; then correlate / regress. See `scripts/phase3_predictability.py` and `phase3_predict_cv*.py` for args and inputs.

## Quick test (no GPU / small run)

- Use `--n_prompts 20` and `--n_features 5` (and no `--fixed_features_path`) or `--micro_sweep` to reduce load.
- Ensure `data/prompts/planets_alpha.csv` exists and has `prompt,target` with non-empty targets.

## Troubleshooting

| Issue | Check |
|-------|--------|
| "No prompts with explicit target" | CSV must have `target` column and no empty targets. |
| "sae.sae_id must be set" | In config, set `sae.sae_id` to a concrete id (e.g. `20-gemmascope-res-16k/2263`). |
| Missing `planets_alpha.csv` | Run `make_domain_splits.py` (and ensure source CSVs have targets) or point `--prompt_csv` to another CSV with prompt+target. |
| phase2_select fails on missing select CSV | Ensure `data/prompts/planets_select.csv` and `neutral_select.csv` exist. |
