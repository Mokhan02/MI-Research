# Research Proposal — Canonical Reference

**Use this document as the single source of truth.** All implementation and design choices must align with it.

---

## Motivation

- **Problem:** Feature steering with SAEs is often unreliable—some features respond to tiny coefficients, others need large ones and cause unintended side effects. There is no principled way to predict this *before* steering.
- **Gap:** Prior work treats steerability as an outcome rather than a predictable property; coefficient selection stays ad hoc and risky for safety-critical use.
- **Proposal:** Define **steerability** as a measurable, pre-intervention property of an SAE feature.
  - **Operationalization:** Minimum steering coefficient required to induce a **fixed on-target behavioral change**.
- **Hypothesis:** SAE features differ in geometry and usage. **Clustered/entangled** features → harder to steer cleanly (larger α, more collateral). **Geometrically isolated** features → more responsive and safer. These structural properties predict steerability.

**Paradigm:** "Predict-before-steering"—use pre-steering geometry and usage to estimate intervention difficulty and off-target risk in advance; reframe steering from trial-and-error to measurable and partially predictable.

---

## Key Contributions

1. **Steerability as a pre-intervention property:** A measurable, feature-level property defined *prior* to any steering; operationalized as min α for a fixed on-target change; places features on a common scale.
2. **Predicting steerability from geometry and usage:** Pre-steering properties (local density, feature similarity, co-activation) predict the magnitude of steering required; variance is structured, not arbitrary.

---

## Five-Phase Pipeline (Strict)

### Phase 1: Pre-steering feature characterization

- **Objective:** For each feature `f`, compute geometric and usage metrics on a reference corpus *before* any steering.
- **Metrics (required):**
  1. **Max cosine similarity:** \(\max_{g \neq f} \cos(v_f, v_g)\) — proximity to nearest neighbor.
  2. **Neighbor density:** Mean cosine similarity to k-NN (e.g. k=50) — local crowding.
  3. **Co-activation correlation:** Mean activation correlation with top-k co-active features on reference corpus — functional entanglement.
- **Optional:** activation frequency, mean activation, sparsity.
- **Scripts:** `02_presteering_metrics.py`, `src/geometry` (max_cosine_similarity, neighbor_density, coactivation_correlation).

---

### Phase 2: Measuring steerability

- **Objective:** Quantify steerability by activation steering and on-target behavioral change.
- **Steering:** \(h' = h + \alpha \cdot v_f\) (decoder direction \(v_f\)).
- **Coefficient grid:** \(\{0, 0.1, 0.2, 0.5, 1, 2, 5, 10\}\) (log-spaced); optionally ±α for directionality.
- **On-target metric:** \(B(\alpha, f) = \mathbb{E}_{x \in S_{\text{target}}}[\text{score}(M_{\alpha,f}(x))]\).
- **Steerability definition:**  
  \(\alpha^*(f) = \min\{\alpha : B(\alpha,f) - B(0) \geq T\}\)  
  with **threshold T** fixed and pre-registered (e.g. T = 0.10 absolute improvement).
- **Censoring:** If no α in the grid reaches T, treat \(\alpha^*(f)\) as **right-censored** (> max α), not "α* = max."
- **Scripts:** `phase2_run.py` (α-grid, delta_logit_target or benchmark score, threshold_T, censored α*).

---

### Phase 3: Predicting steerability (primary analysis)

- **Objective:** Test whether pre-steering metrics predict \(\alpha^*(f)\).
- **Targets:** Predict \(\log \alpha^*(f)\) (regression) and/or \(\alpha^*(f) \leq \alpha_{\max}\) (classification).
- **Methods:** Rank-based correlations (e.g. Spearman), regression/classification trained on pre-steering metrics only; cross-validation over features; permutation and ablation baselines.
- **Outputs:** Scatter (metric vs \(\log \alpha^*\)), predicted vs actual, CIs, permutation p-values.
- **Scripts:** `phase3_predictability.py`, `phase3_predict_cv.py`, `phase3_predict_cv_np.py`.

---

### Phase 4: Off-target behavioral analysis (secondary)

- **Objective:** Evaluate off-target effects under two regimes.
- **Regime (i):** At \(\alpha = \alpha^*(f)\) — risk at "minimum effective push."
- **Regime (ii):** At fixed low \(\alpha = \alpha_0\) (e.g. 1) — "unsafe even when gentle."
- **Metrics:**  
  - **Magnitude:** \(R_{\text{mag}}(\alpha,f) = \frac{1}{m}\sum_j |\Delta U_j(\alpha,f)|\).  
  - **Breadth:** \(R_{\text{breadth}}(\alpha,f) = \frac{1}{m}\sum_j \mathbf{1}\{|\Delta U_j| > \tau\}\) (τ small, TBD).
- **Scripts:** `04_offtarget.py`.

---

### Phase 5: Safe steering coefficient rules

- Develop rules for choosing α from pre-steering estimates and risk.
- (To be elaborated in implementation.)

---

## Experimental setup (locked where decided)

- **Model:** Gemma-2-2b.
- **SAE:** Existing checkpoint on Gemma (e.g. GemmaScope resid_post, 16k); single layer for MVP (e.g. layer 20).
- **Feature set:** N = 200–500 (e.g. 300); stratified by activation frequency if available.
- **Prompts:**
  - **(A) Reference corpus:** 50k–200k tokens for Phase 1 geometry/usage.
  - **(B) On-target:** One primary behavior (e.g. SALADBench, or planets/capitals-style); 300–1k prompts; behavior score in [0,1].
  - **(C) Off-target:** 2–3 benchmarks (e.g. GPQA, TruthfulQA); 100–300 prompts each.
- **Config:** `configs/targets/gemma2_2b_gemmascope_res16k.yaml`.

---

## Pre-steering metrics (inputs) — exact definitions

| Metric | Definition |
|--------|------------|
| Max cosine similarity | \(\max_{g \neq f} \cos(v_f, v_g)\) |
| Neighbor density | Mean cosine similarity to k-NN (e.g. k=50) |
| Co-activation correlation | Mean correlation with top-k co-active features on reference corpus |

**Baselines and controls:** Permutation test (shuffle metric–feature); ablations (drop one metric); random-direction control (steer along random vector matched in norm to \(v_f\)).

---

## Primary analyses and reporting

- **Primary:** Predict \(\log \alpha^*(f)\); Spearman, R², AUROC (for classification).
- **Uncertainty:** Bootstrap CIs over features; permutation p-values.
- **Visualizations:** Metric vs \(\log \alpha^*\); predicted vs actual (CV); risk vs \(\alpha^*\); optional UMAP by steerability.

---

## Ideal results

- **Primary:** Pre-steering geometric/usage properties predict the minimum coefficient for a fixed on-target change.
- **Secondary:** Features requiring larger α exhibit greater or broader off-target effects (including at fixed low α).

---

## Script ↔ Proposal mapping

| Proposal | Script(s) | Notes |
|----------|-----------|--------|
| Phase 1 metrics | `02_presteering_metrics.py`, `src/geometry` | Max cos sim, neighbor density, co-activation |
| Phase 2 steerability | `phase2_run.py` | α-grid, T, α*, censoring |
| Phase 3 prediction | `phase3_predictability.py`, `phase3_predict_cv*.py` | Correlate metrics with α* |
| Phase 4 off-target | `04_offtarget.py` | At α* and at fixed α₀ |
| Alignment / geometry | `archive/scripts/utilities/compute_alignment_mean_target.py` | cos(w_dec, u_target) — optional add-on |
| Data/splits | `make_domain_splits.py`, `make_phase2_csvs.py` | Select / alpha / holdout; targets |
| Feature selection | `phase2_select_contrast.py` | Contrast (task − neutral); controls |

Any script or config that diverges from this pipeline should be brought back in line with this document.
