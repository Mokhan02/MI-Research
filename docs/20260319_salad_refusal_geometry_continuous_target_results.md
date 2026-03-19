# 20260319 Salad Refusal Steering: Geometry vs Continuous Steerability

## Run Overview

This report summarizes the end-to-end “SALAD refusal steering” experiment we ran to evaluate whether **SAE geometry metrics** predict **steering effectiveness**.

### Pipeline stages

1. **Generate SALADBench CSV prompts**
   - `scripts/prepare_salad_bench.py`
   - Outputs:
     - `data/prompts/salad_select.csv` (120 rows)
     - `data/prompts/salad_alpha.csv` (99 rows)
     - `data/prompts/salad_holdout.csv` (81 rows)

2. **Contrast-based SAE feature selection**
   - `scripts/phase2_select_contrast.py`
   - Scoring mode: `--scoring composite`
   - Selected top-K before output-score filtering: `--top-k 300`
   - Output:
     - `outputs/phase2_select_salad/selected_features_salad.json` (300 features)

3. **Output-score filtering (top 100)**
   - `scripts/compute_output_scores.py`
   - Output:
     - `outputs/phase2_select_salad/selected_features_filtered.json` (100 features)

4. **Full steering run (refusal mode)**
   - `scripts/phase2_run.py --mode refusal`
   - Fixed features:
     - `outputs/phase2_select_salad/selected_features_filtered.json`
   - Prompt set:
     - `--prompt_csv data/prompts/salad_alpha.csv`
   - Run size:
     - `--n_prompts 75`
     - `--n_features 100`
   - Output:
     - `outputs/phase2_fullrun/`

5. **Phase 3 analysis: geometry vs steerability**
   - `scripts/phase3_predictability.py`
   - Output:
     - `outputs/phase3_fullrun_T005/`
     - `outputs/phase3_fullrun_T004/`
     - `outputs/phase3_fullrun_T003/`

## Model + SAE + Target Settings

### Model
- `google/gemma-2-2b-it`

### SAE
- GemmaScope SAE: `gemmascope-res-16k`
- Layer index: `layer_idx=20`
- Hook point: `blocks.20.hook_resid_post`
- Decoder normalization: `normalize_decoder: true`

### Feature selection thresholding (alpha*)

The Phase 2 outputs are aggregated into an `alpha*`-based target using a refusal-drop threshold `threshold_T`.

We explored:
- `threshold_T = 0.10` (initial)
- `threshold_T = 0.05` (T005)
- `threshold_T = 0.04` (T004)
- `threshold_T = 0.03` (T003)

### Steering metrics

Two supervision regimes were compared:

1. **Thresholded alpha***:
   - Supervision derived from when mean refusal drop crosses `threshold_T`.
   - Problem observed: high censoring → label “pile-up” at the fallback alpha.

2. **Continuous steerability target (final)**
   - Continuous target: `refusal_drop_max`
   - Interpretation: larger values correspond to a larger maximum refusal drop across the steering alpha grid.

## Code/Execution Fixes

During Phase 2 refusal mode, the run initially crashed with:
- `UnboundLocalError: cannot access local variable 'br' ...`

We fixed `scripts/phase2_run.py` by removing a premature `br` reference in the refusal-mode loop. After the fix, Phase 2 successfully completed and Phase 3 produced geometry metrics on GPU.

## Key Phase 2 Outcome: Censoring Regime

At the initial stricter threshold (`threshold_T = 0.10`), the run produced:
- `frac_uncensored_up = 0`
- `frac_uncensored_down = 0`

At lower thresholds, censoring decreased:
- `threshold_T = 0.05` (T005):
  - non-censored features became available (reported `frac_uncensored_up/down` around `0.07`)
- `threshold_T = 0.04` (T004):
  - non-censored features increased (reported `frac_uncensored_down ~ 0.25`, `frac_uncensored_up ~ 0.17`)

Even at `threshold_T = 0.04`, a large fraction of features remained censored:
- Phase 3 reported `Censored features: 83 (83%)` for T004

## Phase 3: Geometry vs Alpha* (Thresholded Labels)

When correlating SAE geometry metrics with **thresholded `alpha_star_best`**, the results were weak:
- Many censored features mapped to the fallback `alpha*` value (10.0)
- This reduced label variance and produced unstable/undefined Spearman correlations (`NaN` / constant-input warnings)

Net: thresholded alpha* was not a robust supervision signal for this particular run regime.

## Phase 3 Final Result: Geometry vs Continuous Steerability

We therefore recomputed the analysis using the continuous target **`refusal_drop_max`**.

Using:
- geometry metrics from the merged Phase 3 CSV (e.g. `outputs/phase3_fullrun_T005/salad_fullrun_T005_features_merged.csv`)
- continuous target from `outputs/phase2_fullrun/feature_summary.csv`

### Spearman correlations (all 100 features)

1. `max_cosine_similarity` vs `refusal_drop_max`
   - `r = 0.2653`, `p = 0.00764`, `n = 100`
   - Bootstrap 95% CI: **[0.0729, 0.4418]**

2. `neighbor_density` vs `refusal_drop_max`
   - `r = 0.2779`, `p = 0.00511`, `n = 100`
   - Bootstrap 95% CI: **[0.1023, 0.4441]**

3. `coactivation_correlation` vs `refusal_drop_max`
   - `r = 0.0639`, `p = 0.5273`, `n = 100`
   - Bootstrap 95% CI: **[-0.1174, 0.2402]**

### Bootstrap interpretation

- Both **max_cosine_similarity** and **neighbor_density** show confidence intervals fully above zero, supporting a positive monotonic relationship.
- **coactivation_correlation** shows a CI spanning zero, consistent with no clear predictive relationship.

## Consistency Check (T004 vs T005)

We verified that the “all features” continuous-target correlations are effectively identical across thresholded alpha* variants:
- For `max_cosine_similarity` and `neighbor_density`, the all-features Spearman results remained stable between T004 and T005.

## Non-censored subset caveat

For the non-censored subset (where `is_censored == False`), sample sizes were small (e.g. `n=7` at T005, `n=17` at T004).

As a result, subset-only Spearman correlations were low-power and not consistently significant. The primary evidence is therefore the full-feature continuous-target correlations.

## Conclusion

Despite weak/unstable results from thresholded alpha* labels due to censoring, the run produced a **clear and statistically supported signal**:

**SAE geometry metrics (`max_cosine_similarity`, `neighbor_density`) predict continuous steering effectiveness (`refusal_drop_max`).**

This provides a stronger and more robust test than thresholded alpha* in this regime.

## Next Steps (Recommended)

1. Report geometry-vs-continuous-target results (with bootstrap CIs) as the primary claim; treat alpha* threshold correlations as exploratory.
2. Increase the effective sample of steerability outcomes by:
   - increasing top-k after output filtering (e.g., evaluate 150/200 features),
   - or using continuous AUC-style targets derived directly from `run_rows.csv`.
3. Validate robustness across:
   - neighboring SAE layers (e.g., 18/22/24),
   - alternative refusal prompt subsets,
   - and a second dataset/domain.

