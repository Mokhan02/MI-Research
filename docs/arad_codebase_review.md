# Arad et al. Codebase Review: SAEs Are Good for Steering

Review of the [technion-cs-nlp/saes-are-good-for-steering](https://github.com/technion-cs-nlp/saes-are-good-for-steering) repo and how it maps to our pipeline and alignment plan.

---

## 1. Repo layout

| Path | Purpose |
|------|--------|
| `src/output_score.py` | Compute output score per feature (one forward + intervention, neutral prompt). |
| `src/input_score.py` | Compute input score from Neuronpedia activation data + logit-lens overlap. |
| `src/steer.py` | Run steered generation for many features and prefixes; cache generated texts. |
| `src/sae_utils.py` | `AmlifySAEHook` (steering hook) and `init_hook`. |
| `src/utils.py` | SAE loading (sae-lens), logit-lens cache, feature-file parsing. |
| `src/plot.py` | Plotting / analysis. |
| `data/` | features, output_scores, input_scores, generated_texts, steering_factors, axbench_*. |

Dependencies: `torch`, `transformers`, `accelerate`, **`sae-lens`** (SAE loading), `tqdm`, `numpy`, `pandas`, `matplotlib`.

---

## 2. SAE loading (utils.py)

- They use **sae-lens** only: `SAE.from_pretrained(release=..., sae_id=...)`.
- Gemma 2: `release = "gemma-scope-{2b|9b}-{pt|it}-res-canonical"`, `sae_id = "layer_{layer}/width_{width}/canonical"`.
- Width: `"16k"` default; `"131k"` for `gemma2_9b_it_131`.
- No raw NPZ loading; everything goes through sae-lens (encoder, decoder, thresholds, hook names).

Implication for us: we load from GemmaScope NPZ directly. To mirror them exactly we could add a path that uses sae-lens for Gemma 2; our alignment plan (load `W_enc`, `b_enc`, `threshold` from NPZ) is still valid and matches the paper’s math.

---

## 3. Steering hook (sae_utils.py — AmlifySAEHook)

This is the core of their steering. It is a **forward hook** (on layer output), not a pre-hook.

### 3.1 Steps (per forward pass)

1. **Input**: `output` = layer output tensor (residual stream after the block), shape `(batch, seq, d_model)`.
2. **Encode**: `feature_acts = self.sae.encode(output_tensor)` → `(batch, seq, n_features)`.
3. **Max activation (last token only)**:  
   `max_act_value = torch.max(feature_acts[:, -1, :]).item()`  
   So \(a_{\max}\) is over all features at the **last token** only.
4. **Modify activations (last token, selected feature)**:  
   `feature_acts[:, -1, feature] += max_act_value * self.amp_factor`  
   So they implement Eq. (6): add \(s \cdot a_{\max}\) to feature \(i\) at the last position only.
5. **Decode**: `sae_out = self.sae.decode(feature_acts)`.
6. **Reconstruction error**: They compute the SAE error on the **unmodified** activations:  
   `x_reconstruct_clean = sae.decode(sae.encode(output_tensor))`,  
   `sae_error = output_tensor - x_reconstruct_clean`,  
   then **return** `sae_out + sae_error`.

So the layer output is **replaced** by:  
`decode(modified_acts) + (original_resid - decode(original_acts))`.  
That is, the residual at that layer is the SAE reconstruction of the steered activations plus the part of the residual the SAE did not capture. For a linear decoder (and no \(b_{\text{dec}}\)), the change relative to no steering is approximately \(s \cdot a_{\max} \cdot W_{\text{dec}}^{(i)}\) at the last token; with a nonlinear decoder they do the exact nonlinear intervention.

### 3.2 Where steering is applied

- Only the **last token** position is modified in the activation array (`feature_acts[:, -1, feature]`).
- During **generation**, each new step has a new “last” token, so every generated position is steered once when it is the last token. So effectively they steer at the current decoding position only, not at all positions in the prompt.

### 3.3 Summary vs our plan

| Aspect | Arad code | Our plan (steering_arad_vs_phase2.md) |
|--------|-----------|----------------------------------------|
| Scaling | \(s \cdot a_{\max}\) at last token | Same: \(\alpha \cdot a_{\max}\) (we can do last-token or all positions). |
| Application | Replace residual: decode(modified_acts) + error | We proposed **add** \(\alpha \cdot a_{\max} \cdot W_{\text{dec}}^{(i)}\) (linearized). |
| Error term | Yes: residual − decode(original_acts) added back | No: we only add a vector. |

So they do a **full replace** in SAE space (decode modified activations and add reconstruction error); we planned a **linearized add**. For a linear decoder the effect is the same; for nonlinear decoder + \(b_{\text{dec}}\), their version is the exact paper. We can either implement the full replace (need encoder + decoder + error) or keep the add for simplicity; the scaling \(a_{\max}\) is what matters most.

---

## 4. Output score (output_score.py)

- **Model**: Base Gemma-2 (e.g. `google/gemma-2-2b`), not IT, in the script.
- **Neutral prompt**: Single sentence, e.g. `"From my experience,"` (hardcoded).
- **Hook**: Same `AmlifySAEHook` with **amp_factor=10** (hardcoded).
- **Logit lens**: From `cache_logit_lens`: apply final layer norm to `sae.W_dec`, then `lm_head`, then per-feature top-20 token indices.
- **Score**: One forward with the hook; take logits at last position;  
  `rank_output_score = 1 - (min_rank_of_logit_lens_tokens / vocab_size)`;  
  `top_token_score = max(prob[logit_lens_indices])`;  
  **output_score = rank_output_score * top_token_score** (paper Eq. 9–10: rank-weighted probability).
- **Caching**: Results saved to JSON by `layer_feature_key` (e.g. `"20_1234"`).

So: one forward per feature, one neutral prompt, no concept-specific data. We can mirror this in `scripts/compute_output_scores.py` (or equivalent): same formula, our model/SAE loading, and our steering hook once we have activation-scaled steering.

---

## 5. Input score (input_score.py)

- **Data**: Expects Neuronpedia-style per-feature JSONs: `{layer}_{feature}.json` with an `"activations"` list; each entry has `"tokens"` and `"values"` (activation per token).
- **Logic**: For up to 100 activation records, find the token with max activation; count how often that token is in the feature’s top-20 logit-lens tokens. Input score = that fraction (paper Eq. 8).
- **Dependency**: Requires pre-downloaded Neuronpedia feature data; not self-contained.

For our pipeline, input score is optional (PDF and paper say output score is the main lever); we can add it later or skip if we don’t use Neuronpedia.

---

## 6. Steer.py (generation)

- Loads features from JSON (layer → list of feature IDs).
- For each feature: attach `AmlifySAEHook`, run generation on **50 neutral prefixes** (same list as in the paper appendix), temperature 0.7, max_new_tokens=20, then remove hook and cache generated texts.
- Used for their “generation success” and qualitative results, not for output-score computation.

We already have refusal-mode generation in `phase2_run.py`; once our hook is activation-scaled (and optionally full replace), we’re aligned for this kind of evaluation.

---

## 7. Feature and config format

- **Features file**: JSON object mapping layer (int or string) to list of feature IDs, e.g. `{"20": [1, 2, 3], "21": [4, 5]}`.
- **Model types**: `gemma2_2b`, `gemma2_9b`, `gemma2_9b_it_131` (and similar); base vs IT and width are selected in `get_sae()`.

We use a single layer and a flat list in `selected_features_*.json`; we can adapt by grouping by layer if we add multi-layer support.

---

## 8. What to take from their codebase

1. **Steering formula**: Encode → \(a_{\max}\) at last token → add \(s \cdot a_{\max}\) to chosen feature at last token → decode → add SAE error. We currently only add \(\alpha \cdot W_{\text{dec}}^{(i)}\); we should at least move to \(\alpha \cdot a_{\max} \cdot W_{\text{dec}}^{(i)}\) (and optionally full replace + error as in their code).
2. **Output score**: One forward per feature, neutral prompt, amp_factor=10, rank-weighted probability of top logit-lens token(s). We should implement this in our pipeline and use it to filter features (Tier 1.2 of the plan).
3. **Logit lens**: LN(decoder weights) then lm_head, then top-20 indices per feature. We should use the same for output (and optionally input) score.
4. **Where to steer**: They steer only at the **last token** per forward. For generation that steers every new token when it’s the last. We can match that in refusal mode (last token only per step) or keep all-position add; the plan can note this difference.
5. **SAE source**: They use sae-lens; we use NPZ. Both are fine; our plan to load encoder + threshold from NPZ is enough to implement activation-scaled steering and output score.

---

## 9. References

- Repo: [technion-cs-nlp/saes-are-good-for-steering](https://github.com/technion-cs-nlp/saes-are-good-for-steering)
- Paper: [Arad et al., SAEs Are Good for Steering – If You Select the Right Features (EMNLP 2025)](https://arxiv.org/abs/2505.20063)
- Our alignment plan: [steering_arad_vs_phase2.md](../steering_arad_vs_phase2.md) and the SAE steering alignment plan in `.cursor/plans/`
