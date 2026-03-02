# Phase 1 Legacy Scripts (Archived)

These scripts produced **synthetic / placeholder data** and have been
superseded by the Phase 2 pipeline (`phase2_run.py`, `phase2_select_contrast.py`)
and Phase 3 analysis (`phase3_predictability.py`, `phase3_predict_cv.py`).

## Archived scripts

| Script | Purpose |
|---|---|
| `01_choose_features.py` | Feature selection using placeholder `load_sae()` |
| `02_presteering_metrics.py` | Pre-steering geometry via placeholder SAE |
| `03_steerability.py` | Steering runs using placeholder `create_steering_hook()` |
| `04_offtarget.py` | Off-target evaluation using placeholder scorer |
| `05_analysis.py` | Correlation analysis on synthetic outputs |

## Why archived

- All five scripts depend on `model_utils.load_sae()` and `src/scoring.py`,
  both of which are placeholders that return synthetic data.
- Phase 2 scripts use the real SAE loader (`src/sae_loader.load_gemmascope_decoder()`)
  and run on the actual gemma-2-2b model with gemma-scope-2b-pt-res SAE weights.
