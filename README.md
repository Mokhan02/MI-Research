# Steerability as a Predictable Property of SAE Features

This repository contains the pipeline for studying whether pre-steering geometric and usage properties of sparse autoencoder (SAE) features predict their **steerability** — defined as the minimum steering coefficient required to induce a fixed on-target behavioral change — and whether harder-to-steer features produce greater off-target effects under constrained steering.

## Overview

Feature steering with SAEs is increasingly used for model control and alignment, but coefficient selection today is largely ad hoc: researchers pick a handful of values and see what happens. This project operationalizes steerability as a measurable, pre-intervention property of individual SAE features and tests whether it can be predicted from geometry and usage statistics alone — *before* any steering is applied.

The core thesis: features that are densely clustered in representation space or highly co-activated with neighboring features are more entangled, require larger steering coefficients to induce targeted effects, and cause greater collateral behavioral changes when forced.

## Quick Overview

The pipeline has five phases:

| Phase | Question | Script |
|-------|----------|--------|
| 1 | What are the geometric and usage properties of each feature? | `pipeline/phase1_feature_metrics.py` |
| 2 | What steering coefficient does each feature require? | `pipeline/phase2_steerability.py` |
| 3 | Do pre-steering metrics predict steerability? | `pipeline/phase3_prediction.py` |
| 4 | What off-target effects does steering induce? | `pipeline/phase4_off_target.py` |
| 5 | Can we derive safe steering rules? | `pipeline/phase5_safe_rules.py` |

## Setup

Add your API keys to `.env`:
```bash
HF_TOKEN=...
WANDB_API_KEY=...
```

Ensure `uv` is installed, then:
```bash
uv run pipeline/run_pipeline.sh
```

Already-completed phases are automatically skipped. Pass `--force` to rerun everything.

---

## Experimental Pipeline

### Models and SAE

The pipeline operates on a single model × layer × feature set (MVP):

| Component | Role |
|-----------|------|
| Base model | The LLM being analyzed (TBD) |
| SAE checkpoint | Sparse autoencoder trained on the chosen layer (TBD) |
| Intervention layer | Single transformer layer for MVP (TBD) |
| Feature set | N = 200–500 SAE features, stratified by activation frequency |

### Datasets

Three prompt sets are used across phases:

| Dataset | Purpose | Size |
|---------|---------|------|
| Reference corpus | Phase 1: compute geometry and usage statistics | 50k–200k tokens |
| On-target benchmark | Phase 2: measure behavioral change under steering | 300–1,000 prompts |
| Off-target benchmarks | Phase 4: measure collateral behavioral effects | 100–300 prompts each |

**On-target benchmark options (choose one for MVP):**
- SALADBench — refusal behavior
- BBQ / ToxiGen — bias
- HalluLens — hallucination

**Off-target benchmarks:**
- GPQA — reasoning
- TruthfulQA — truthfulness
- DarkBench — sycophancy

---

## Pipeline Details

### Phase 1: Pre-Steering Feature Characterization

For each SAE feature `f`, three geometric and usage metrics are computed on a reference corpus *before any steering intervention*:

| Metric | Definition | What it captures |
|--------|-----------|-----------------|
| Max cosine similarity | `max_{g≠f} cos(v_f, v_g)` | Proximity to nearest neighbor in decoder space |
| Neighborhood density | Mean cosine similarity to k-NN (default k=50) | Local crowding in representation space |
| Co-activation correlation | Mean activation correlation with top-k co-active features | Functional entanglement across the corpus |

Optional add-ons: activation frequency, mean activation, sparsity.

These metrics are computed once and remain fixed throughout the study.

### Phase 2: Measuring Steerability

Steering is applied as:
```
h' = h + α · v_f
```

where `v_f` is the decoder direction of feature `f` and `α` is a steering coefficient drawn from a log-spaced grid:
```
α ∈ {0.1, 0.2, 0.5, 1, 2, 5, 10}
```

**Steerability** `α*(f)` is defined as the minimum coefficient required to induce a fixed behavioral delta `T` (e.g., T = 0.10 absolute improvement) relative to the unsteered baseline:
```
α*(f) = min{α : B(α, f) − B(0) ≥ T}
```

Features that do not reach the threshold within the tested range are right-censored: `α*(f) > 10`.

### Phase 3: Predicting Steerability (Primary Analysis)

Tests whether pre-steering metrics predict `log α*(f)` using:
- Spearman rank correlations between each metric and `log α*(f)`
- Regression and classification models trained on pre-steering metrics only
- Cross-validation across features
- Permutation-based null baselines and single-metric ablations

**Key outputs:** scatter plots (metric vs. `log α*(f)`), predicted vs. actual plots, AUROC for steerability classification.

### Phase 4: Off-Target Behavioral Analysis (Secondary Analysis)

Off-target effects are measured under two regimes:

1. **At `α*(f)`** — collateral effects at the minimum coefficient needed for the target behavior
2. **At a fixed low `α₀`** (e.g., `α₀ = 1`) — isolates structurally risky features independent of intervention strength

For each off-target benchmark `U_j`, two quantities are computed:

| Metric | Formula | What it captures |
|--------|---------|-----------------|
| Magnitude | `(1/m) Σ_j |ΔU_j(α, f)|` | Average severity of collateral changes |
| Breadth | `(1/m) Σ_j 1{|ΔU_j(α, f)| > τ}` | Number of behaviors meaningfully affected |

### Phase 5: Safe Steering Rules

Derives principled guidelines for safe coefficient selection based on the relationship between pre-steering geometry and observed off-target risk across Phases 2–4.

---

## Baselines and Controls

| Baseline | Purpose |
|----------|---------|
| Permutation test | Shuffle metric–feature assignments; expected predictive performance under null |
| Single-metric ablations | Quantify individual contribution of each pre-steering metric |
| Random direction control | Steer along random vectors matched in norm to `v_f`; tests whether any direction causes effects |
| Second model / layer (stretch) | Generalization beyond the MVP model and layer |

---

## Statistical Methods

- **Primary target:** predict `log α*(f)` (regression) and `α*(f) ≤ α_max` (classification)
- **Metrics:** Spearman ρ, R² on `log α*`, AUROC for classification
- **Uncertainty:** bootstrap confidence intervals over features; permutation-based p-values for key correlations

---

## Output Structure

All results are saved under `outputs/`:
```
outputs/
  phase1_metrics/
    feature_metrics.csv          # Per-feature geometry and usage stats
    metric_distributions.png     # Histograms of each metric across features

  phase2_steerability/
    steerability_scores.csv      # α*(f) per feature, with censoring flags
    steering_curves.png          # On-target score vs. α for sampled features

  phase3_prediction/
    correlation_results.csv      # Spearman ρ and p-values per metric
    regression_cv_results.csv    # Cross-validated regression performance
    scatter_plots/               # Metric vs. log α*(f) per metric
    predicted_vs_actual.png      # CV predictions vs. ground truth

  phase4_off_target/
    off_target_at_threshold.csv  # Magnitude and breadth at α*(f)
    off_target_at_fixed_alpha.csv
    risk_vs_steerability.png     # Off-target risk vs. α*(f) scatter

  phase5_safe_rules/
    safe_steering_guidelines.md  # Derived coefficient recommendations
```

---

## Hypotheses

| Hypothesis | Prediction | What would confirm it |
|------------|-----------|----------------------|
| Geometry predicts steerability | Features with higher neighbor density and co-activation require larger `α*` | Significant positive Spearman ρ between density/co-activation and `log α*(f)` |
| Entangled features cause more collateral effects | Features with high pre-steering entanglement show higher off-target magnitude and breadth | Strong correlation between geometric metrics and off-target risk |
| Steerability and off-target risk are related but distinct | Harder-to-steer features are riskier, but some features are risky even at low `α` | Risk vs. `α*(f)` shows a positive trend with meaningful residuals |

---

## Limitations

- **Computational cost.** Running multiple benchmarks across a grid of coefficients for hundreds of features is expensive. This limits the number of features, layers, and models that can be studied in a single sweep.
- **Single model and layer.** The MVP focuses on one model and one intervention layer. The relationship between feature geometry and steerability may differ across architectures or depths.
- **Behavior-specific thresholds.** Steerability is defined relative to a fixed behavioral delta `T`. What counts as a "meaningful" change differs across benchmarks (e.g., a 0.1 improvement on refusal is not equivalent to 0.1 on bias), so comparisons across on-target behaviors require care.
- **Generalization.** It is unclear whether features found to be predictable in one setting will behave similarly in another. Stretch experiments on a second model or layer are included to probe this.

---

## Related Work

| Paper | Key Finding | Gap Addressed |
|-------|------------|---------------|
| [SAEs Are Good for Steering](https://arxiv.org/abs/2505.20063) | Steering is more effective on causally influential features | No pre-intervention predictor of difficulty or risk |
| [Entanglement and the CACE Principle](https://medium.com/@ib.lahlou) | Interventions on entangled representations cause widespread changes | No concrete, testable framework for predicting steering risk |
| [Geometry of Categorical Concepts in LLMs](https://arxiv.org/abs/2406.01506) | Clean concept geometry corresponds to better semantic alignment | Geometry studied at concept level, not linked to steering |
| [Evaluating Feature Steering](https://www.anthropic.com) | Features respond very differently to steering strength | Analysis is retrospective, no pre-steering predictive framework |

---

## Team

Girish, Akshaj, Mo, Aashna, Shlok

Research doc: https://docs.google.com/document/d/1c-gnQnJlBCQvK3M607sx0QAn8XebIgnVtDrnkB0pQQ0/edit?tab=t.itpz4hcr2hqi#heading=h.4uarl9d0uyvr

