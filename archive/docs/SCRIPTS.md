# Script audit: what’s needed vs redundant

Aligned to **docs/PROPOSAL.md** and the runnable pipeline in **docs/RUN.md**.

---

## Core pipeline (keep — use these)

| Script | Role |
|--------|------|
| **phase2_select_contrast.py** | Contrast feature selection (task − neutral); writes `selected_features_planets.json`, feature summaries, control set. |
| **phase2_run.py** | Phase 2: α-grid steering, delta_logit_target, α*, censoring; writes run_rows.csv, alpha_star.csv, curves. |
| **phase3_predictability.py** | Phase 3: geometry from W_dec + optional usage; correlate with α*; regression/classification. |
| **phase3_predict_cv.py** | Phase 3: CV over features (Ridge, LogReg), Spearman, R², AUROC. |
| **phase3_predict_cv_np.py** | Phase 3: same idea, no PyTorch (CSV-only). |
| **04_offtarget.py** | Phase 4: off-target effects at α* and at fixed α₀. |
| **02_presteering_metrics.py** | Phase 1: max_cos_sim, neighbor_density, coactivation (expects `selected_features.npy` from 01; can be adapted to read JSON). |
| **make_domain_splits.py** | Build *_select / *_alpha / *_holdout from source CSVs. |
| **make_phase2_csvs.py** | Build planets_only.csv, capitals_only.csv with prompt+target and " Answer:" scaffolding. |
| **archive/scripts/utilities/compute_alignment_mean_target.py** | Optional: cos(w_dec, u_target) per feature. |
| **audit_activation_distribution.py** | Optional: activation percentiles to set tau_act. |

---

## Sanity / one-off (keep but optional)

| Script | Role |
|--------|------|
| **verify_target.py** | Check config: model, hook point, d_model, decoder shape. Run once per target. |
| **99_hook_sanity.py** | Hook + decoder sanity; run once to confirm steering path. |
| **phase1_smoke.py** | Quick smoke test (model + SAE load, one forward). |
| **10_phase2_smoke_ontarget.py** | On-target smoke with keyword scoring (topic-specific). |
| **phase2_sanity_plot.py** | Plot run_rows / alpha_star for inspection. |
| **debug_steer_effect.py** | Debug steering and logits. |

---

## Redundant or superseded (safe to remove or archive)

These implement an **old pipeline** (01→02→03→04→05) or narrow variants that the proposal replaces.

| Script | Why redundant |
|--------|----------------|
| **01_choose_features.py** | Feature selection by uniform/stratified sample; **replaced by** contrast selection in **phase2_select_contrast.py** (proposal: top K by Δ = task − neutral). |
| **03_steerability.py** | Old steerability: uses `load_sae`, benchmark scorer, `on_target.txt`. **Replaced by** **phase2_run.py** (config prompt CSV, delta_logit_target, α*, censoring). |
| **05_analysis.py** | Expects presteering_metrics.csv, steerability_results.csv, offtarget_results.csv (old I/O). **Replaced by** **phase3_predictability.py** + **phase3_predict_cv*.py** (run_rows, alpha_star.csv). |
| **11_phase2_prefilter_features.py** | Arithmetic activation lift vs control; topic-specific prefilter. **Replaced by** contrast selection in **phase2_select_contrast.py**. |
| **11b_phase2_prefilter_logit.py** | Logit-based prefilter for one token. **Replaced by** contrast + phase2_run. |
| **make_feature_sets.py** | Builds S_good.txt / S_bad.txt from feature_summary.csv (monotone_frac, success_rate, tv_mean). **Tied to old pipeline output**; not used in proposal. |

---

## Minimal runnable set (proposal-only)

If you want the repo to contain **only** what the proposal needs:

1. **Data:** make_phase2_csvs.py, make_domain_splits.py  
2. **Phase 2:** phase2_select_contrast.py → phase2_run.py  
3. **Phase 3:** phase3_predictability.py, phase3_predict_cv.py (and optionally phase3_predict_cv_np.py)  
4. **Phase 4:** 04_offtarget.py  
5. **Phase 1 (metrics):** 02_presteering_metrics.py (after adapting to read features from selected_features_planets.json instead of 01’s selected_features.npy)  
6. **Optional:** `archive/scripts/utilities/compute_alignment_mean_target.py`, audit_activation_distribution.py, verify_target.py, 99_hook_sanity.py  

You can **move** 01, 03, 05, 10, 11, 11b, make_feature_sets, phase1_smoke, debug_steer_effect, phase2_sanity_plot into an `archive/` or `scripts/legacy/` folder, or delete them if you’re sure you won’t need the old pipeline.
