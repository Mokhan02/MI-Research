# Steering pipeline changes and their paper/source

This document lists **every code/config change** made to align our SAE steering with the literature, **why** each was made, and **which paper or source** it comes from. Use it to trace “we did X because paper Y says Z.”

---

## Sources

| Short name | Full reference | What they contribute |
|------------|----------------|----------------------|
| **Arad et al. 2025** | “SAEs Are Good for Steering – If You Select the Right Features”, EMNLP 2025, [arxiv 2505.20063](https://arxiv.org/pdf/2505.20063) | Steering operator (activation-scaled in SAE latent space), output-score feature selection, last-token steering for generation. Same model (Gemma-2-2B) as ours. |
| **Bhargav & Zhu** | “Feature-Guided SAE Steering for Refusal-Rate Control using Contrasting Prompts”, [arxiv 2511.00029](https://arxiv.org/pdf/2511.00029) | Contrast-based feature selection with **composite score** (magnitude + consistency/variance). Steering vector \( \alpha \cdot \max(\text{activations}) \cdot w_f \), i.e. activation-scaled like Arad. |
| **Teammate PDF** | “Why Your SAE Steering Effects Are Weak and How to Fix Them: Feature Selection, Output Scores, and Geometry in SAE-Based Model Steering” | Synthesis: recommends output-score selection (Arad), wider SAEs and later layers, Bhargav-style composite scoring, multi-feature steering, and max-pooling (CorrSteer). Frames our geometry–steerability research. |
| **CorrSteer** | Cited in teammate PDF | Recommends **max-pooling** over token positions for activation aggregation in feature selection (instead of mean-pooling residuals first). |

---

## Change list (file → what changed → paper → why)

### 1. Activation-scaled steering (load full SAE + hooks)

| Where | What changed |
|-------|----------------|
| `src/sae_loader.py` | Added `load_gemmascope_full()`: loads `W_enc`, `b_enc`, `threshold` from the same NPZ as `W_dec`. |
| `scripts/phase2_run.py` | Steering is activation-scaled only: `a_max = max(relu(resid @ W_enc + b_enc - threshold))`, then `hidden += alpha * a_max * W_dec[fid]`. Hooks: `make_steer_prehook_amax` (logit), `make_steer_prehook_amax_lastpos` (generation / refusal). Full SAE weights (encoder + threshold) are always loaded. |

| Paper | Why |
|-------|-----|
| **Arad et al. 2025** | They steer in SAE latent space: increase feature \(i\) by \(s \cdot a_{\max}\), then decode. Effective residual step is \(s \cdot a_{\max} \cdot W_{\text{dec}}^{(i)}\). Our previous fixed \( \alpha \cdot W_{\text{dec}}^{(i)} \) used the wrong units and ignored context; we align with their Eq. (6) so results are comparable. |
| **Bhargav & Zhu** | Same idea: steering magnitude proportional to \(\max(\text{activations}) \cdot w_f\). |

---

### 2. Output-score feature selection

| Where | What changed |
|-------|----------------|
| `scripts/compute_output_scores.py` | New script: for each feature, (1) logit-lens top-20 tokens from \(W_{\text{dec}}[fid]\) through final norm + lm_head, (2) one forward with activation-scaled steering at last token, (3) score = \((1 - \text{min\_rank}/\text{vocab\_size}) \times \max\text{prob}(\text{logit\_lens tokens})\). Writes scores and optional filtered feature list. |

| Paper | Why |
|-------|-----|
| **Arad et al. 2025** | They define **output score** as the causal effect of a feature on model output (rank-weighted probability of logit-lens tokens under intervention). High output score = feature actually drives behaviour. They report ~2–3× improvement from selecting by output score; we add this so our feature set is selected the same way. |

---

### 3. Composite scoring in contrast selection

| Where | What changed |
|-------|----------------|
| `scripts/phase2_select_contrast.py` | New `--scoring composite` (default): per-feature score = \(w_1 \cdot \frac{|\Delta\bar{a}|}{\max} + w_2 \cdot (1 - \hat{\sigma}^2)\). Uses mean activation difference (task − neutral) and variance of task activations. Optional thresholds and per-feature strategy (amplify/suppress). `--scoring delta_freq` keeps legacy ranking. |

| Paper | Why |
|-------|-----|
| **Bhargav & Zhu** | They use a **dual-component composite score**: (1) normalized differential activation magnitude, (2) inverse variance (consistency across contrasting prompts). We implement this so contrast selection is not only “who activates more on task” but also “who is consistent,” matching their refusal-rate control setup. |
| **Teammate PDF** | Recommends Bhargav-style composite scoring as a Tier 2 improvement for better feature quality. |

---

### 4. Max-pooling for activations

| Where | What changed |
|-------|----------------|
| `scripts/phase2_select_contrast.py` | New `--pooling max`: for each prompt we get activations per token (resid @ W_dec.T), then take **max over tokens** per feature. Default remains `--pooling mean` (mean-pool residual then project). |

| Paper | Why |
|-------|-----|
| **CorrSteer** (via teammate PDF) | CorrSteer recommends **max-pooling** over the sequence for activation aggregation: the strongest activation instance per feature is more informative than averaging over tokens (which dilutes signal). Teammate PDF lists this as a Tier 2 recommendation. |

---

### 5. Multi-feature steering

| Where | What changed |
|-------|----------------|
| `scripts/phase2_run.py` | New `--multi_steer_top_n N`: takes the first N features from the selected-features file, sums their decoder directions, and runs one alpha sweep with the combined vector. Hooks: `make_steer_prehook_multi_amax`, `make_steer_prehook_multi_amax_lastpos`. Outputs `run_rows_multi.csv` and `curves_multi_topN.csv`. |

| Paper | Why |
|-------|-----|
| **Teammate PDF** | Recommends **multi-feature steering** (combine several top features) as Tier 2 to get stronger, more reliable steering effects. We add it as an option so you can compare single-feature vs combined-feature runs. |

---

### 6. Configs for wider SAE and later layers

| Where | What changed |
|-------|----------------|
| `configs/targets/gemma2_2b_gemmascope_res65k.yaml` | Layer 20, **65K** features. `weights_path`: `layer_20/width_65k/average_l0_61/params.npz` (verified on HF). |
| `configs/targets/gemma2_2b_gemmascope_layer22.yaml` | **Layer 22**, 16K. `weights_path`: `layer_22/width_16k/average_l0_72/params.npz`. |
| `configs/targets/gemma2_2b_gemmascope_layer24.yaml` | **Layer 24**, 16K. `weights_path`: `layer_24/width_16k/average_l0_73/params.npz`. |

| Paper | Why |
|-------|-----|
| **Teammate PDF** | Recommends **wider SAEs** (e.g. 65K) for more monosemantic features and **later layers** for more behavioural/abstract features that steer better. We add these configs so you can run layer/width ablations without guessing paths. |

---

## One-line summary

- **Arad et al.** → activation-scaled steering operator + output-score selection (Tier 1).
- **Bhargav & Zhu** → composite scoring in contrast selection (Tier 2).
- **CorrSteer / Teammate PDF** → max-pooling option (Tier 2).
- **Teammate PDF** → multi-feature steering (Tier 2), wider SAE + later-layer configs (Tier 3).

The main write-up that compares our pipeline to Arad et al. and gives full pipeline commands is **`archive/notes/steering_arad_vs_phase2.md`**. This file is the **change log + paper mapping** so you can cite and justify each modification.
