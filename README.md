# Steerability as a Predictable Property of SAE Features

This repository contains the pipeline for studying whether pre-steering geometric and usage properties of sparse autoencoder (SAE) features predict their **steerability** — defined as the minimum steering coefficient required to induce a fixed on-target behavioral change — and whether harder-to-steer features produce greater off-target effects under constrained steering.

## Overview

Feature steering with SAEs is increasingly used for model control and alignment, but coefficient selection today is largely ad hoc: researchers pick a handful of values and see what happens. This project operationalizes steerability as a measurable, pre-intervention property of individual SAE features and tests whether it can be predicted from geometry and usage statistics alone — *before* any steering is applied.

The core thesis: features that are densely clustered in representation space or highly co-activated with neighboring features are more entangled, require larger steering coefficients to induce targeted effects, and cause greater collateral behavioral changes when forced.

## Quick Start

### Prerequisites

- Python 3.11–3.12
- [uv](https://docs.astral.sh/uv/) for dependency management
- GPU recommended (MPS on Apple Silicon, CUDA on Linux/Windows) — CPU works but is slow
- HuggingFace token for gated model access

### Installation

```bash
uv sync
export HF_TOKEN=your_token_here
```

### Run the pipeline

```bash
# 1. Contrast feature selection (real model + SAE)
PYTHONPATH=. uv run python scripts/phase2_select_contrast.py \
  --config configs/targets/gemma2_2b_gemmascope_res16k.yaml \
  --domain planets --out_dir outputs/phase2_select --top-k 100

# 2. Measure steerability (real SAE, matched features + prompts)
PYTHONPATH=. uv run python scripts/phase2_run.py \
  --config configs/targets/gemma2_2b_gemmascope_res16k.yaml \
  --out_dir outputs/phase2 --n_prompts 100 \
  --fixed_features_path outputs/phase2_select/selected_features_planets.json

# 3. Predict α* from geometry
PYTHONPATH=. uv run python scripts/phase3_predictability.py \
  --config configs/targets/gemma2_2b_gemmascope_res16k.yaml \
  --phase2_dir outputs/phase2 --out_dir outputs/phase3
```

Results land in `outputs/phase2_select/` (feature summaries), `outputs/phase2/` (run_rows.csv, alpha_star.csv, curves), and `outputs/phase3/` (correlation results, scatter plots).

### Device selection

All configs default to `device: "auto"`, which picks the best available backend:

1. **MPS** (Apple Silicon) — used automatically on macOS with M-series chips
2. **CUDA** — used automatically when an NVIDIA GPU is available
3. **CPU** — fallback

Override in config YAML or pass `--device cpu` where supported.

## Pipeline

| Phase | Question | Script |
|-------|----------|--------|
| 1 | Select features by contrast (task − neutral) | `scripts/phase2_select_contrast.py` |
| 2 | What steering coefficient does each feature require? | `scripts/phase2_run.py` |
| 3 | Do pre-steering metrics predict steerability? | `scripts/phase3_predictability.py` |
| 4 | What off-target effects does steering induce? | *(not yet implemented)* |

**Do not run scripts in `scripts/legacy/`.** They are the old pipeline with fake SAE/steering/scores, kept only for reference.

### Models and SAE

| Component | Value |
|-----------|-------|
| Model | `google/gemma-2-2b-it` (instruction-tuned) or `google/gemma-2-2b` (base) |
| SAE | GemmaScope res-16k, layer 20 (`google/gemma-scope-2b-pt-res`) |
| Hook point | `blocks.20.hook_resid_post` (TransformerLens-style → `model.layers.20`) |
| Features | 16,384 total; 300 selected by contrast; τ_act=2.0, token_span=last_n (n=8) |
| Benchmark | SALADBench (refusal), 100 prompts |
| Steering grid | α ∈ {0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0} |
| Threshold | T = 0.10 (minimum Δrefusal for α*) |

### Phase 1: Feature Selection (`phase2_select_contrast.py`)

Contrast-based: top K by Δ = act_freq_task − act_freq_neutral. Uses real model + SAE decoder. Outputs `selected_features_<domain>.json` and `feature_summary_<domain>.csv`.

### Phase 2: Measuring Steerability (`phase2_run.py`)

Steering is applied as: `h' = h + α · v_f`

where `v_f` is the SAE decoder direction and `α` sweeps the grid. **Steerability** `α*(f)` = min α where Δrefusal ≥ T. Features that never reach T are right-censored.

Outputs: `run_rows.csv`, `alpha_star.csv`, `curves_per_feature.parquet`.

### Phase 3: Predicting Steerability (`phase3_predictability.py`)

Computes pre-steering geometric metrics (max cosine similarity, neighborhood density, co-activation correlation) and correlates them with log(α*) via Spearman rank correlation and regression.

### Phase 4: Off-Target Analysis

To be implemented. Will measure collateral behavioral effects at α*(f) and at fixed low α₀.

## Configuration

Configs live in `configs/`. The target configs inherit from `configs/base.yaml`:

- `configs/targets/gemma2_2b_gemmascope_res16k.yaml` — Gemma-2-2b-IT (main)
- `configs/targets/gemma2_2b_base_gemmascope_res16k.yaml` — Gemma-2-2b base

## Run Gates

See **GATES.md** for the seven gates before a full run. Key points:

1. Domain prompt splits — no overlap between selection and evaluation
2. Contrast-based selection — top K by Δ, not top K act_freq
3. Micro sweep first — `phase2_run --micro_sweep` (10 features, ~25 prompts) before full K=100
4. α* with censoring — never set α* = max for no-effect features

See **LANDMINES.md** for six traps to audit (tau_act, token_span, T pre-registration, directionality, control matching, neutral isomorphism).

## Key Files

| File | Purpose |
|------|---------|
| `src/model_utils.py` | Model loading with auto device detection (MPS/CUDA/CPU) |
| `src/sae_loader.py` | Real SAE decoder loading (`load_gemmascope_decoder`) |
| `src/hook_resolver.py` | TransformerLens ↔ HuggingFace hook name mapping |
| `src/refusal_scorer.py` | Keyword-based refusal classifier |
| `src/config.py` | Config loading and validation |

## Hypotheses

| Hypothesis | What would confirm it |
|------------|----------------------|
| Geometry predicts steerability | Significant positive Spearman ρ between density/co-activation and log α*(f) |
| Entangled features cause more collateral effects | Strong correlation between geometric metrics and off-target risk |
| Steerability and off-target risk are related but distinct | Risk vs. α*(f) shows positive trend with meaningful residuals |

## Limitations

- **Computational cost** — multiple benchmarks × coefficient grid × hundreds of features is expensive
- **Single model and layer** — MVP focuses on Gemma-2-2b layer 20; may not generalize
- **Behavior-specific thresholds** — T = 0.10 on refusal ≠ 0.10 on bias
- **Generalization** — unclear if results transfer across architectures or depths

## Related Work

| Paper | Key Finding |
|-------|------------|
| [SAEs Are Good for Steering](https://arxiv.org/abs/2505.20063) | Steering is more effective on causally influential features |
| [Geometry of Categorical Concepts in LLMs](https://arxiv.org/abs/2406.01506) | Clean concept geometry corresponds to better semantic alignment |

## Team

Girish, Akshaj, Mo, Aashna, Shlok

[Research doc](https://docs.google.com/document/d/1c-gnQnJlBCQvK3M607sx0QAn8XebIgnVtDrnkB0pQQ0/edit?tab=t.itpz4hcr2hqi#heading=h.4uarl9d0uyvr)
