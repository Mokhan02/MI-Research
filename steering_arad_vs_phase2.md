## Arad et al. SAE Steering vs Current Phase 2 Pipeline

This note explains how Arad et al. (“SAEs Are Good for Steering – If You Select the Right Features” [`arxiv.org/pdf/2505.20063`](https://arxiv.org/pdf/2505.20063)) implement SAE-based steering on Gemma-2, how that differs from the current `phase2_run.py` setup, and what concrete changes are needed to match their setup more closely.

**For a single list of all changes with the paper/source and rationale for each, see [docs/steering_changes_and_papers.md](docs/steering_changes_and_papers.md).**

---

## 1. Steering in Arad et al.

Arad et al. treat the SAE as the primary object and perform steering in **SAE latent space**, not directly in model residual space.

Given a model hidden state \(x^l \in \mathbb{R}^d\) at layer \(l\) for a particular token:

- **Encode with the SAE**

  \[
  z = W_{\text{enc}} x^l + b_{\text{enc}}, \quad
  a = \sigma(z) \approx \text{JumpReLU}(z - \text{threshold})
  \]

  Here \(a \in \mathbb{R}^k\) are feature activations.

- **Compute a per-token scaling factor \(a_{\max}\)**

  \[
  a_{\max} = \max_j a_j
  \]

- **Apply steering in feature space (their Eq. (6))**

  For a chosen feature index \(i\) and steering factor \(s\):

  \[
  \tilde{a}_j =
  \begin{cases}
  a_j & j \neq i \\
  a_j + s \cdot a_{\max} & j = i
  \end{cases}
  \]

  Intuition: “take the strongest currently active feature and add \(s\) copies of it into feature \(i\).”

- **Decode back to model space and continue the forward pass**

  \[
  \Phi(x^l) = W_{\text{dec}} \tilde{a} + b_{\text{dec}}, \quad
  x^l \leftarrow \Phi(x^l)
  \]

Key properties:

- **Direction** is the SAE decoder column \(W_{\text{dec}}^{(i)}\).
- **Magnitude** is **context-dependent** via \(s \cdot a_{\max}\); different prompts/tokens get different perturbation sizes.
- All of their “input/output feature” analysis and steerability results are measured under this latent-space intervention.

---

## 2. What `phase2_run.py` currently does

In `scripts/phase2_run.py`, both logit and refusal modes implement steering **directly in residual space** using only `W_dec`:

- Load GemmaScope decoder:

  - `W_dec` of shape `[n_features, d_model]` (optionally row-normalized).

- For a chosen feature `fid`, define a fixed direction:

  ```python
  steer_dir = W_dec[fid]  # [d_model]
  ```

- Apply steering in a forward pre-hook on `model.model.layers[layer_idx]`:

  - **Logit mode** (single position, last token):

    ```python
    hidden2[0, pos, :] = hidden2[0, pos, :] + alpha * steer_dir
    ```

  - **Refusal mode** (all positions):

    ```python
    hidden2[:, :, :] = hidden2[:, :, :] + alpha * steer_dir
    ```

Effective operation:

\[
x^l \leftarrow x^l + \alpha \cdot W_{\text{dec}}^{(i)}
\]

with \(\alpha\) chosen from a fixed grid (e.g. \(0, 1, 2, 5, 10, 20, 40\)), **independent of the SAE activations**.

What is correct:

- The **direction** \(W_{\text{dec}}^{(i)}\) is the right one: increasing only activation \(a_i\) by a constant \(c\) while holding other activations fixed yields a decoded shift of \(c \cdot W_{\text{dec}}^{(i)}\).

What is missing relative to Arad et al.:

- No computation of SAE activations \(a\) or \(a_{\max}\) from the current hidden state.
- `alpha` is treated as a raw residual-space offset, not as a feature-space steering factor scaled by current activity.

---

## 3. Why this mismatch matters

### 3.1 Magnitude calibration

In Arad et al., the effective residual-space step is:

\[
\Delta x^l = (s \cdot a_{\max}) \cdot W_{\text{dec}}^{(i)}.
\]

In the current code, it is:

\[
\Delta x^l = \alpha \cdot W_{\text{dec}}^{(i)}.
\]

Unless \(\alpha\) happens to match \(s \cdot a_{\max}\) for typical tokens, the explored perturbation sizes are different. Features can look “hard to steer” or “saturating at max alpha” simply because the grid is in the wrong units.

### 3.2 Context dependence

- **Arad et al.**: intervention strength varies with how active the SAE is for that token (via \(a_{\max}\)).
- **Current pipeline**: same push for every token and prompt, regardless of whether the SAE is “quiet” or “very active” on that example.

### 3.3 Theoretical alignment

Their claims about “input features vs output features” and their strong steering results are all evaluated under the SAE-latent intervention. Using a fixed vector moves you closer to classic representation-steering, which is a different regime.

---

## 4. How to align Phase 2 steering with Arad et al.

Goal: move from

\[
x^l \leftarrow x^l + \alpha \cdot W_{\text{dec}}^{(i)}
\]

to

\[
x^l \leftarrow x^l + \alpha \cdot a_{\max}(x^l) \cdot W_{\text{dec}}^{(i)},
\]

where \(a_{\max}(x^l)\) is computed from the **live** SAE encoder at the hook point. Then interpret `alpha` as the paper’s steering factor \(s\).

### 4.1 Load full SAE parameters in `phase2_run.py`

Currently, `phase2_run.py` only uses:

```python
W_dec, _ = load_gemmascope_decoder(config)
```

`archive/scripts/debug_steer_effect.py` already shows how to load:

- `W_enc`
- `b_enc`
- `threshold`

from the same GemmaScope NPZ used for the decoder. The first step is to replicate that loading logic in `phase2_run.py` so that `W_enc`, `b_enc`, and `thr` are on the same device as the model.

### 4.2 Implement SAE encoding for a single residual vector

For a residual vector `resid_vec` at a given token:

1. Compute pre-activation:

   \[
   z = W_{\text{enc}} x + b_{\text{enc}}
   \]

2. Apply thresholded nonlinearity:

   \[
   a = \text{ReLU}(z - \text{threshold})
   \]

3. Take:

   \[
   a_{\max} = \max_j a_j
   \]

You already have a helper like this in `debug_steer_effect.py`. That helper can be reused or refactored into a shared utility and called from `phase2_run.py`.

### 4.3 Update steering hooks to use `a_max`

Instead of:

```python
hidden2[0, pos, :] = hidden2[0, pos, :] + alpha * steer_dir
```

do (conceptually):

```python
resid_vec = hidden2[0, pos, :]
z, act = sae_z_and_act(resid_vec, W_enc, b_enc, thr)
a_max = act.max()
hidden2[0, pos, :] = resid_vec + alpha * a_max * steer_dir
```

For refusal mode (all positions), apply the same logic per position along the sequence dimension.

This matches their Equation (6): you’re effectively increasing feature \(i\) by \(\alpha \cdot a_{\max}\), then decoding that change via `W_dec[fid]`.

### 4.4 Treat `alpha` as steering factor `s`

Once steering is implemented as `alpha * a_max * W_dec[fid]`, the parameter `alpha` in your CLI/grid is directly comparable to Arad et al.’s steering factor \(s\). You can then reason about alpha grids using their experiments (e.g., ranges that preserve coherence while still moving the concept strongly).

---

## 5. Feature selection considerations

The paper’s strongest gains come not just from *how* they steer but from *which* SAE features they steer:

- They define **input scores** and **output scores** using a combination of logit-lens and interventions.
- Features with high **output scores** but low input scores are the most effective steering directions.

Your current setup uses **contrast-based selection** (unsafe vs neutral prompts on SALADBench), which is good for **task relevance** but does not distinguish between input- and output-type features in the sense of the paper.

To fully mirror their recipe on Gemma-2-2B:

- Use contrast selection to get a candidate pool of safety-related features.
- Compute **output scores** for this pool (per [Arad et al. 2025](https://arxiv.org/pdf/2505.20063)), using a neutral prompt and a single forward pass per feature.
- Filter out low-output-score features, and only steer with the remaining high-output-score subset.

This should make SAE-based steering more comparable, in both methodology and expected performance, to the results they report on Gemma-2.

---

## 6. Implementation status

All Tier 1 changes from the alignment plan have been implemented:

### 6.1 Activation-scaled steering (implemented)

`phase2_run.py` now defaults to Arad-style activation-scaled steering:

- `src/sae_loader.py` has `load_gemmascope_full()` which loads `W_enc`, `b_enc`, and `threshold` from the GemmaScope NPZ alongside `W_dec`.
- New hooks in `phase2_run.py`:
  - `make_steer_prehook_amax` — single-position steering (logit mode).
  - `make_steer_prehook_amax_lastpos` — last-token-only steering (refusal/generation mode, matching Arad's `AmlifySAEHook`).
- Both compute `a_max = max(relu(resid @ W_enc + b_enc - threshold))` from the live hidden state.
- The `alpha` parameter is now the steering factor \(s\); effective perturbation is \(\alpha \cdot a_{\max} \cdot W_{\text{dec}}^{(i)}\).
- Pass `--legacy_steering` to revert to the old fixed-vector mode for comparison.

### 6.2 Output-score feature selection (implemented)

`scripts/compute_output_scores.py` computes Arad et al. output scores:

- Projects `W_dec[fid]` through final layer norm + lm_head → top-20 logit-lens tokens.
- One forward pass per feature on a neutral prompt with activation-scaled steering at `amp_factor=10`.
- Score = `(1 - min_rank/vocab_size) * max_prob(logit_lens_tokens)`.
- `--filtered_out` writes a `selected_features_*.json` with only features above the threshold.

### 6.3 Full pipeline

```bash
# 1. Contrast-based candidate pool
uv run python scripts/phase2_select_contrast.py \
  --config configs/targets/gemma2_2b_gemmascope_res16k.yaml \
  --domain salad --out_dir outputs/phase2_select_salad --top-k 100

# 2. Output-score filter (Arad et al.)
uv run python scripts/compute_output_scores.py \
  --config configs/targets/gemma2_2b_gemmascope_res16k.yaml \
  --features_path outputs/phase2_select_salad/selected_features_salad.json \
  --out_path outputs/phase2_select_salad/output_scores.json \
  --filtered_out outputs/phase2_select_salad/selected_features_salad_filtered.json \
  --threshold 0.01

# 3. Steering run with filtered features (activation-scaled by default)
uv run python scripts/phase2_run.py \
  --config configs/targets/gemma2_2b_gemmascope_res16k.yaml \
  --mode refusal \
  --out_dir outputs/phase2_salad_arad \
  --n_prompts 99 \
  --alphas 0,1,2,5,10,20,40 \
  --fixed_features_path outputs/phase2_select_salad/selected_features_salad_filtered.json
```

### 6.4 Composite scoring — Bhargav & Zhu (implemented)

`scripts/phase2_select_contrast.py` now supports `--scoring composite` (default):

- Computes a dual-component score per feature:

  \[
  \text{score}_f = w_1 \cdot \frac{|\Delta\bar{a}_f|}{\max |\Delta\bar{a}|} + w_2 \cdot (1 - \hat{\sigma}^2_f)
  \]

  where \(\Delta\bar{a}_f\) = mean activation difference (task − neutral) and \(\hat{\sigma}^2_f\) = normalized variance of task activations.

- Thresholds: `--comp-min-score`, `--comp-min-diff`, `--comp-max-var`.
- Weights: `--comp-w1` (default 0.5) and `--comp-w2` (default 0.5).
- Outputs per-feature `strategy` (amplify / suppress) from the sign of the differential.
- Pass `--scoring delta_freq` to revert to legacy ranking by \(\Delta\text{freq}\).

### 6.5 Max-pooling for activations (implemented)

`scripts/phase2_select_contrast.py` now supports `--pooling max`:

- Projects every token's residual through \(W_{\text{dec}}^T\) to get per-token feature activations.
- Takes `max` over all non-padding tokens per feature (instead of mean-pooling the residuals first).
- Recommended by CorrSteer literature: max-pool captures the strongest activation instance, avoiding dilution from inactive positions.
- Pass `--pooling mean` (default) to keep the legacy mean-pool behavior.

### 6.6 Multi-feature steering (implemented)

`scripts/phase2_run.py` now supports `--multi_steer_top_n N`:

- Takes the first N features from `--fixed_features_path` (assumed ordered by score, top first).
- Sums their decoder directions into a single combined steering vector.
- Sweeps over the alpha grid with this combined direction instead of iterating per feature.
- New hooks: `make_steer_prehook_multi_amax` (logit) and `make_steer_prehook_multi_amax_lastpos` (refusal).
- Writes results to `run_rows_multi.csv` and `curves_multi_topN.csv`.
- Usage:

  ```bash
  uv run python scripts/phase2_run.py \
    --mode refusal \
    --fixed_features_path outputs/phase2_select_salad/selected_features_salad_filtered.json \
    --multi_steer_top_n 5 \
    --out_dir outputs/phase2_salad_multi5
  ```

### 6.7 Config variants for wider SAE / later layers (implemented)

Three new config files for ablation studies. The `weights_path` (average_l0_*) values have been verified against [google/gemma-scope-2b-pt-res](https://huggingface.co/google/gemma-scope-2b-pt-res) on HuggingFace:

| Config | Layer | Width | weights_path (verified) |
|--------|-------|-------|-------------------------|
| `gemma2_2b_gemmascope_res65k.yaml` | 20 | 65K | `layer_20/width_65k/average_l0_61/params.npz` |
| `gemma2_2b_gemmascope_layer22.yaml` | 22 | 16K | `layer_22/width_16k/average_l0_72/params.npz` |
| `gemma2_2b_gemmascope_layer24.yaml` | 24 | 16K | `layer_24/width_16k/average_l0_73/params.npz` |

Other `average_l0_*` options exist per layer/width on the repo if you want sparser/denser SAEs.

---

## 7. Full pipeline (updated)

```bash
# 1. Contrast-based candidate pool (composite scoring + max-pool)
uv run python scripts/phase2_select_contrast.py \
  --config configs/targets/gemma2_2b_gemmascope_res16k.yaml \
  --domain salad --out_dir outputs/phase2_select_salad --top-k 100 \
  --scoring composite --pooling max

# 2. Output-score filter (Arad et al.)
uv run python scripts/compute_output_scores.py \
  --config configs/targets/gemma2_2b_gemmascope_res16k.yaml \
  --features_path outputs/phase2_select_salad/selected_features_salad.json \
  --out_path outputs/phase2_select_salad/output_scores.json \
  --filtered_out outputs/phase2_select_salad/selected_features_salad_filtered.json \
  --threshold 0.01

# 3. Single-feature steering run (activation-scaled by default)
uv run python scripts/phase2_run.py \
  --config configs/targets/gemma2_2b_gemmascope_res16k.yaml \
  --mode refusal \
  --out_dir outputs/phase2_salad_arad \
  --n_prompts 99 \
  --alphas 0,1,2,5,10,20,40 \
  --fixed_features_path outputs/phase2_select_salad/selected_features_salad_filtered.json

# 4. Multi-feature steering (optional: top 5 features combined)
uv run python scripts/phase2_run.py \
  --config configs/targets/gemma2_2b_gemmascope_res16k.yaml \
  --mode refusal \
  --out_dir outputs/phase2_salad_multi5 \
  --n_prompts 99 \
  --alphas 0,1,2,5,10,20,40 \
  --fixed_features_path outputs/phase2_select_salad/selected_features_salad_filtered.json \
  --multi_steer_top_n 5

# 5. Layer ablation (repeat steps 1-3 with different configs)
uv run python scripts/phase2_select_contrast.py \
  --config configs/targets/gemma2_2b_gemmascope_layer22.yaml \
  --domain salad --out_dir outputs/phase2_select_salad_l22 --top-k 100 \
  --scoring composite --pooling max
# ... then output scores + steering run with layer22 config
```

---

## 8. Summary

- **Arad et al.** steer by modifying SAE activations: increase feature \(i\) by \(s \cdot a_{\max}\) (where \(a_{\max}\) is the max activation at that token), then decode with \(W_{\text{dec}}\).
- **Phase 2** now implements the same operator by default. Pass `--legacy_steering` for the old fixed-vector mode.
- **Feature selection** now supports three complementary methods:
  - Contrast-based candidate pool (with composite scoring from Bhargav & Zhu).
  - Output-score filtering (from Arad et al.) for causal effectiveness.
  - Max-pooling over tokens for stronger activation signals.
- **Multi-feature steering** lets you combine top-N features into a single intervention.
- **Layer/width ablation configs** enable testing whether later layers or wider SAEs improve steerability.
- The full pipeline is: composite contrast selection → output-score filter → activation-scaled steering (single or multi-feature).

