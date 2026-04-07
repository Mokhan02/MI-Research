# Repo Audit Report

Pulled `origin/main` to `0baf31f` before auditing. I read the `transformer-lens-interpretability` and `sparse-autoencoder-training` skills first, then reviewed the entire repo. `python3 -m compileall` passed for `src/`, `scripts/`, and `archive/`, so there are no syntax-level blockers.

## Highest-Impact Findings

1. `[P1]` Phase 3 computes SAE usage and coactivation from the model's final hidden state, not the SAE layer residual, so `act_freq`, `mean_act`, `mean_z_minus_thr`, and coactivation are mismeasured for the target SAE layer. See `scripts/phase3_predictability.py:277` and `scripts/phase3_predictability.py:290`. This directly affects the main Phase 3 correlation analysis.

2. `[P1]` The active README tells users to run contrast selection with `--pooling max`, but `max` pooling ignores `token_span` and takes the max over the full sequence, not the configured `last_n` span. See `README.md:119` and `scripts/phase2_select_contrast.py:102`. That changes which features get selected for the main run.

3. `[P1]` SALAD prep claims to build a neutral set from HH-harmless and exposes `--n_neutral`, but the code ignores `--n_neutral` and uses a fixed hand-written list of 50 neutral prompts. See `scripts/prepare_salad_bench.py:52` and `scripts/prepare_salad_bench.py:93`. That weakens the task-vs-neutral contrast used in feature selection.

4. `[P2]` Phase 3 silently drops `coactivation_correlation` on reruns if `baseline_usage_*.csv` already exists, because the cache stores usage stats only. See `scripts/phase3_predictability.py:944`. The same command can therefore produce different predictor sets depending on cache state.

5. `[P2]` The documented Step 7 command is stale: the README uses `--run_dir` and `--out_dir`, but the script accepts `--phase2_dir` and `--output_dir`. See `README.md:164` and `scripts/phase3_predictability.py:762`. A user following the README literally will fail before analysis starts.

## Active Pipeline

The supported pipeline in the current branch is:

1. `scripts/prepare_salad_bench.py`
2. `scripts/phase2_select_contrast.py`
3. `scripts/compute_output_scores.py`
4. `scripts/phase2_run.py`
5. `scripts/phase3_predictability.py`

`archive/` is explicitly reference-only and should not be used for new runs.

## Active Source Files

### `src/__init__.py`

- Purpose: package marker for reusable experiment utilities.
- Pipeline position: shared support layer.
- Issues: none.

### `src/config.py`

- Purpose: loads YAML, resolves run-specific output paths, validates steering config, applies deterministic generation defaults.
- Pipeline position: shared setup for every active script.
- Issues:
- `resolve_config` does a shallow `dict.copy()`, so nested dicts are still shared. Nothing obviously breaks in the current flow, but it is not a deep immutable config resolution step.

### `src/model_utils.py`

- Purpose: resolves device, loads HF causal LM and tokenizer, sets pad token, applies dtype handling.
- Pipeline position: model bootstrap for selection, scoring, steering, and analysis.
- Issues:
- Current code is fine for the Gemma configs in this repo.
- It is not fully architecture-agnostic; downstream scripts assume Gemma-style submodules anyway.

### `src/sae_loader.py`

- Purpose: downloads GemmaScope NPZs from Hugging Face, infers decoder key, loads `W_dec`, `W_enc`, `b_enc`, and `threshold`.
- Pipeline position: core SAE loading for selection, output filtering, steering, and some analysis.
- Issues:
- Overall robust.
- Active analysis code in `phase3_predictability.py` bypasses some of its key-finding flexibility by reopening the NPZ and indexing `data["W_dec"]` directly.

### `src/hook_resolver.py`

- Purpose: maps TransformerLens-style hook names like `blocks.20.hook_resid_post` to HF modules like `model.layers.20`.
- Pipeline position: hook resolution for contrast selection and archived diagnostics.
- Issues:
- Only really supports the current `hook_resid_post` mapping pattern. That is acceptable for the active Gemma target configs.

### `src/refusal_scorer.py`

- Purpose: keyword-based refusal detector for SALAD-style generations.
- Pipeline position: refusal-mode scoring in `phase2_run.py`, plus rescoring utilities.
- Issues:
- It is a heuristic classifier, so measurement noise is expected.
- This is a scientific validity limitation, not a runtime blocker.

## Active Scripts

### `scripts/prepare_salad_bench.py`

- Purpose: downloads SALADBench `base_set`, samples prompts by safety category, writes `salad_select.csv`, `salad_alpha.csv`, and `salad_holdout.csv`, and creates neutral CSVs.
- Pipeline position: Step 0, prompt preparation.
- Issues:
- `--n_neutral` is unused.
- Neutral prompts are not actually pulled from HH-harmless despite the docstring.
- If `neutral_select.csv` already exists, it is reused without validating provenance or size.

### `scripts/phase2_select_contrast.py`

- Purpose: computes task-vs-neutral feature activations and ranks features by either `delta_freq` or composite score.
- Pipeline position: Step 1, feature selection.
- Issues:
- With `--pooling max`, token span is ignored and activations are maxed over the whole sequence.
- W&B config metadata is stale in places: it logs `contrast_delta_freq` even when composite scoring is used.
- Random control output is labeled `randK_matched_actfreq.json`, but the note says it is not actually matched on activation frequency.

### `scripts/compute_output_scores.py`

- Purpose: applies an Arad-style output relevance filter to candidate features using logit lens tokens and activation-scaled steering on a neutral prompt.
- Pipeline position: Step 2, optional causal relevance filtering before the main run.
- Issues:
- Hard-wired to Gemma-like internals: `model.model.layers[layer_idx]` and `model.model.norm`.
- Fine for the current target configs, but not portable.

### `scripts/phase2_run.py`

- Purpose: the main steering sweep. Supports `logit` and `refusal` modes, runs alpha grids, computes per-feature summaries, writes `run_rows.csv`, `feature_summary.csv`, `alpha_star.csv`, and curves.
- Pipeline position: Step 3, main experiment run.
- Issues:
- Main experiment path looks broadly runnable.
- Scientific interpretation in refusal mode depends strongly on the keyword refusal scorer and the coherence filter.
- Generation steering uses last-token prehooks only for refusal mode, which is intentional here but should be treated as part of the experimental definition.

### `scripts/phase3_predictability.py`

- Purpose: merges steerability outputs with geometry and usage metrics, then runs Spearman correlations, plots, and optional supplementary classification.
- Pipeline position: Step 4, downstream analysis.
- Issues:
- Uses final hidden state rather than the SAE hook-layer residual for baseline activation statistics.
- Cached baseline usage files omit coactivation, so reruns with cache produce a different predictor set.
- README invocation does not match the CLI.

### `scripts/smoke_test_amax.py`

- Purpose: sanity check that Arad-style `a_max` varies across prompts and features.
- Pipeline position: optional preflight diagnostic.
- Issues:
- No major issues.
- Useful to run before expensive steering sweeps.

## Config Files

### `configs/base.yaml`

- Purpose: default config schema with steering, model, feature, and benchmark defaults.
- Pipeline position: shared config reference.
- Issues:
- Mostly a template; target configs override what matters for the active runs.

### `configs/targets/gemma2_2b_gemmascope_res16k.yaml`

- Purpose: primary locked target config for the active SALAD pipeline.
- Pipeline position: default production config for selection, filtering, steering, and analysis.
- Issues:
- Coherent and aligned with the active code path.

### `configs/targets/gemma2_2b_base_gemmascope_res16k.yaml`

- Purpose: base-model ablation using Gemma-2 2B base instead of instruct.
- Pipeline position: alternate experiment config.
- Issues:
- No obvious config-level blockers.

### `configs/targets/gemma2_2b_gemmascope_layer22.yaml`

- Purpose: later-layer ablation for 2B IT, layer 22.
- Pipeline position: alternate experiment config.
- Issues:
- No obvious config-level blockers.

### `configs/targets/gemma2_2b_gemmascope_layer24.yaml`

- Purpose: later-layer ablation for 2B IT, layer 24.
- Pipeline position: alternate experiment config.
- Issues:
- No obvious config-level blockers.

### `configs/targets/gemma2_2b_gemmascope_res65k.yaml`

- Purpose: wider-SAE ablation for 2B IT, width 65k.
- Pipeline position: alternate experiment config.
- Issues:
- No obvious config-level blockers, but much heavier geometry computation is implied.

### `configs/targets/gemma2_9b_gemmascope_layer20.yaml`

- Purpose: 9B IT absolute-position layer match.
- Pipeline position: scale-up ablation config.
- Issues:
- No code changes accompany the 9B addition, so practical runtime risk is memory and throughput, not syntax.

### `configs/targets/gemma2_9b_gemmascope_layer36.yaml`

- Purpose: 9B IT proportional-depth layer match.
- Pipeline position: scale-up ablation config.
- Issues:
- Same note as the other 9B config.

## Archive and Legacy Files

These do not sit in the supported path, but I audited them because you asked for the entire repo.

### `archive/src/geometry.py`

- Purpose: old geometry helpers for max cosine, neighbor density, and coactivation.
- Pipeline position: first-generation pre-steering metrics stage.
- Issues:
- No direct effect on the current pipeline.

### `archive/src/scoring.py`

- Purpose: placeholder scorer for old benchmark scripts.
- Pipeline position: legacy scoring layer.
- Issues:
- Explicitly placeholder-only and not valid for real experiments.

### `archive/dead_src/geometry.py`

- Purpose: dead duplicate of archived geometry helpers.
- Pipeline position: none.
- Issues:
- None beyond being dead code.

### `archive/dead_src/scoring.py`

- Purpose: dead duplicate of archived placeholder scoring.
- Pipeline position: none.
- Issues:
- None beyond being dead code.

### `archive/scripts/01_choose_features.py`

- Purpose: old random or stratified feature chooser.
- Pipeline position: old Phase 1 feature selection stage.
- Issues:
- Placeholder-heavy.
- Superseded by `scripts/phase2_select_contrast.py`.

### `archive/scripts/02_presteering_metrics.py`

- Purpose: old pre-steering metric extraction.
- Pipeline position: old predictor stage.
- Issues:
- Explicit placeholder activations and fallback SAE logic.
- Not suitable for scientific use.

### `archive/scripts/03_steerability.py`

- Purpose: old alpha-star sweep on an on-target benchmark.
- Pipeline position: old steering stage.
- Issues:
- Placeholder hook registration and placeholder outputs.
- Not suitable for real runs.

### `archive/scripts/04_offtarget.py`

- Purpose: old off-target spillover evaluation.
- Pipeline position: old off-target analysis stage.
- Issues:
- Placeholder steering and scoring paths.

### `archive/scripts/05_analysis.py`

- Purpose: old merge, stats, and plotting stage for the first-generation pipeline.
- Pipeline position: old downstream analysis.
- Issues:
- Assumes the legacy output schema.

### `archive/scripts/10_phase2_smoke_ontarget.py`

- Purpose: old on-target smoke test.
- Pipeline position: legacy diagnostic.
- Issues:
- Not on the active path.

### `archive/scripts/11_phase2_prefilter_features.py`

- Purpose: old arithmetic-vs-control activation-lift prefilter.
- Pipeline position: legacy feature prefilter.
- Issues:
- Domain-specific and superseded by contrast selection.

### `archive/scripts/11b_phase2_prefilter_logit.py`

- Purpose: old token-logit dot-product prefilter.
- Pipeline position: legacy feature prefilter.
- Issues:
- Superseded.

### `archive/scripts/99_hook_sanity.py`

- Purpose: older hook and steering sanity checker.
- Pipeline position: legacy preflight diagnostic.
- Issues:
- No direct effect on the current pipeline.

### `archive/scripts/audit_activation_distribution.py`

- Purpose: audits positive activation percentiles to help choose `tau_act`.
- Pipeline position: legacy utility, conceptually useful before feature selection.
- Issues:
- Not wired into the active README.

### `archive/scripts/debug_steer_effect.py`

- Purpose: one-off steering debugger.
- Pipeline position: legacy diagnostic.
- Issues:
- No direct effect on the current pipeline.

### `archive/scripts/make_domain_splits.py`

- Purpose: builds legacy domain splits from older prompt files.
- Pipeline position: old data-prep stage.
- Issues:
- Superseded by the SALAD prompt pipeline.

### `archive/scripts/make_feature_sets.py`

- Purpose: old feature-set packaging helper.
- Pipeline position: legacy bookkeeping.
- Issues:
- Not relevant to the active pipeline.

### `archive/scripts/make_phase2_csvs.py`

- Purpose: converts old text prompt sets into CSVs with prompt and target columns.
- Pipeline position: old prompt-prep stage.
- Issues:
- Superseded by SALAD prep and archived with the old data.

### `archive/scripts/phase1_smoke.py`

- Purpose: old next-token perturbation smoke test over selected features and alpha values.
- Pipeline position: legacy diagnostic.
- Issues:
- Useful historically, not part of the current run path.

### `archive/scripts/phase2_sanity_plot.py`

- Purpose: plots per-feature alpha curves from `run_rows.csv`.
- Pipeline position: post-run legacy diagnostic.
- Issues:
- Assumes logit-mode columns.

### `archive/scripts/phase3_predict_cv.py`

- Purpose: cross-validated predictability analysis using sklearn.
- Pipeline position: legacy supplementary analysis.
- Issues:
- Assumes older merged CSV column names.

### `archive/scripts/phase3_predict_cv_np.py`

- Purpose: numpy-only version of the old CV analysis.
- Pipeline position: legacy supplementary analysis.
- Issues:
- Assumes older merged CSV column names.

### `archive/scripts/verify_target.py`

- Purpose: older manual verification script for the primary target config.
- Pipeline position: legacy preflight.
- Issues:
- Still useful as a checklist, but the active pipeline no longer depends on it.

### `archive/scripts/utilities/check_feature_overlap.py`

- Purpose: ad hoc overlap check between a hard-coded pilot feature set and a full run.
- Pipeline position: none.
- Issues:
- Hard-coded paths and feature IDs make it a one-off notebook-style script.

### `archive/scripts/utilities/compute_alignment_mean_target.py`

- Purpose: computes decoder alignment to target token unembedding vectors.
- Pipeline position: auxiliary analysis.
- Issues:
- Standalone analysis tool only.

### `archive/scripts/utilities/rescore_pilot.py`

- Purpose: rescored an existing run with an updated refusal scorer and rewrote summaries.
- Pipeline position: post-hoc repair utility.
- Issues:
- Tied to old run directory conventions.

### `archive/scripts/utilities/rescore_soft.py`

- Purpose: rescored old refusal runs with a soft continuous refusal metric that treats degenerate text as uncertain.
- Pipeline position: post-hoc salvage and analysis utility.
- Issues:
- Useful historically, but not part of the active branch workflow.

### `archive/phase1_legacy/01_choose_features.py`

- Purpose: duplicate of the first-generation feature selection script.
- Pipeline position: obsolete.
- Issues:
- Same placeholder issues as its archived numbered counterpart.

### `archive/phase1_legacy/02_presteering_metrics.py`

- Purpose: duplicate of the first-generation pre-steering metrics script.
- Pipeline position: obsolete.
- Issues:
- Same placeholder issues as its archived numbered counterpart.

### `archive/phase1_legacy/03_steerability.py`

- Purpose: duplicate of the first-generation steerability script.
- Pipeline position: obsolete.
- Issues:
- Same placeholder issues as its archived numbered counterpart.

### `archive/phase1_legacy/04_offtarget.py`

- Purpose: duplicate of the first-generation off-target script.
- Pipeline position: obsolete.
- Issues:
- Same placeholder issues as its archived numbered counterpart.

### `archive/phase1_legacy/05_analysis.py`

- Purpose: duplicate of the first-generation analysis script.
- Pipeline position: obsolete.
- Issues:
- Same schema assumptions as its archived numbered counterpart.

## Bottom Line

The current root pipeline is real and mostly coherent. The two main experiment-affecting problems are:

1. Phase 3 uses the wrong residual source for usage and coactivation metrics.
2. Feature selection behavior does not match the README-described `pooling=max` semantics most users would assume.

The archive is largely reference material from an older placeholder-heavy pipeline and does not threaten the main run unless someone mistakenly uses it.

## Residual Risk

I did not execute the actual model-loading and steering runs, so there can still be runtime issues tied to:

- Hugging Face auth and gated model access
- available GPU memory
- exact Gemma module layout on the installed transformers version

Those are runtime risks, not issues I could prove from static audit alone.
