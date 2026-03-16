## IT Refusal Ablations – Steering & Feature Selection

This note records the full setup and results for the IT refusal ablations on Gemma‑2‑2B with GemmaScope SAEs. It is meant as a stable reference for which design choices matter most for SAE steering in this setting.

The experiments compare:

- **Steering method**: activation‑scaled (Arad‑style) vs legacy fixed‑vector.
- **Output‑score filter**: with vs without Arad‑style output‑score feature filtering.
- **Contrast scoring**: composite vs delta_freq contrast selection.

All runs use the same **model/config**, **prompt set**, **alpha grid**, and **evaluation protocol**; only the ablated ingredient changes.

---

## 1. Shared experimental setup

- **Model**: `google/gemma-2-2b-it` (instruction‑tuned).
- **SAE**: GemmaScope resid‑post, layer 20, width 16k (see `configs/targets/gemma2_2b_gemmascope_res16k.yaml`).
- **Hook point**: `blocks.20.hook_resid_post` (maps to `model.layers[20]`).
- **Steering hooks** (in `scripts/phase2_run.py`):
  - **Activation‑scaled (default)**:
    - Compute SAE activations from the live residual:  
      \(a = \mathrm{ReLU}(W_{\text{enc}} x + b_{\text{enc}} - \text{threshold})\).
    - \(a_{\max} = \max_j a_j\).
    - Intervention: \(x \leftarrow x + \alpha \cdot a_{\max} \cdot W_{\text{dec}}^{(i)}\) at the last token.
  - **Legacy fixed‑vector** (using `--legacy_steering`):
    - Intervention: \(x \leftarrow x + \alpha \cdot W_{\text{dec}}^{(i)}\), no dependence on SAE activations.

- **Refusal scoring**:
  - Mode: `--mode refusal` in `phase2_run.py`.
  - Generation: chat template enabled, max_new_tokens = 64, temperature 0.
  - Refusal metric: keyword‑based `_refusal_score(text)` (SALADBench‑style).
  - Baseline: for each prompt, generate once at \(\alpha = 0\), record `base_refusal`.

- **Prompts and alphas**:
  - **Prompts**: 30 prompts (first 30 from the SALADBench alpha CSV in the config).
  - **Alphas**: \(\{0, 1, 5, 20\}\) (symmetric grid not needed for refusal; we only sweep positive alphas).

- **Outputs per run (single‑feature mode)**:
  - `run_rows.csv`: per (prompt, feature, alpha) row with `refusal_score`, `base_refusal`, `delta_refusal`.
  - `feature_summary.csv`: per feature, with:
    - `base_refusal_rate` (mean refusal at \(\alpha=0\)),
    - `refusal_rate_at_max_alpha`,
    - `refusal_drop_at_max_alpha`,
    - `alpha_star_feature_up`, `alpha_star_feature_down`,
    - `censored_up`, `censored_down`.
  - `curves_per_feature.parquet`: per (feature, alpha) curve with `mean_refusal`, `std_refusal`, `n_prompts`.
  - `alpha_star.csv`: subset of `feature_summary` columns for quick inspection.
  - `selected_features.json`: the actual feature IDs used in the run.

**Alpha\*** (`alpha_star_feature_up`) is defined as:

> The smallest positive alpha where the mean refusal rate drops by at least a fixed threshold \(T\) relative to the baseline for that feature’s prompts. If no such alpha exists on the grid, the feature is **censored** (no observed effect) and `censored_up == True`.

---

## 2. Runs included in the ablation

All runs use **25 features**, **30 prompts**, and **the same alpha grid**; only the way the 25 features and/or steering rule are chosen is changed.

- **Baseline (legacy steering)**  
  - Out dir: `outputs/phase2_salad_ablation_steering_legacy`  
  - Steering: **legacy fixed‑vector** (`--legacy_steering`).  
  - Features: top 25 from **composite‑contrast + output‑score–filtered** list.

- **Run 1A – Activation‑scaled, filtered**  
  - Out dir: `outputs/phase2_salad_ablation_steering_amax`  
  - Steering: **activation‑scaled** (Arad‑style, default).  
  - Features: top 25 from **composite‑contrast + output‑score–filtered** list  
    (`outputs/phase2_select_salad/selected_features_salad_filtered.json`).

- **Run 2 – Contrast‑only (no output‑score)**  
  - Out dir: `outputs/phase2_salad_ablation_no_output_score`  
  - Steering: activation‑scaled.  
  - Features: top 25 from **contrast‑only** composite ranking  
    (`outputs/phase2_select_salad/selected_features_salad.json`, no output‑score filter).

- **Run 3 – Delta\_freq + output‑score**  
  - Out dir: `outputs/phase2_salad_ablation_deltafreq`  
  - Steering: activation‑scaled.  
  - Features:
    1. Run `phase2_select_contrast.py` with `--scoring delta_freq --pooling max --top-k 100` → `outputs/phase2_select_salad_deltafreq/selected_features_salad.json`.
    2. Run `compute_output_scores.py` on that list, `--top_k 25` →  
       `outputs/phase2_select_salad_deltafreq/selected_features_salad_filtered_top25.json`.
    3. Use that filtered top‑25 list in `phase2_run.py`.

---

## 3. Aggregate metrics across runs

Using `alpha_star.csv`, `feature_summary.csv`, and `curves_per_feature.parquet`, we computed:

- **`frac_uncensored_up`**: fraction of features with `censored_up == False`.  
- **`mean_alpha_star_up`**: mean `alpha_star_feature_up` over uncensored features.  
- **`mean_refusal_alpha_5.0`**: mean refusal at alpha = 5 (moderate steering) across features.  
- **`baseline_refusal`**: mean `base_refusal_rate` across features.

The code used:

```python
import pandas as pd
from pathlib import Path

base = Path("outputs")

runs = {
    "legacy": base / "phase2_salad_ablation_steering_legacy",
    "amax_filtered": base / "phase2_salad_ablation_steering_amax",
    "contrast_only": base / "phase2_salad_ablation_no_output_score",
    "deltafreq_filtered": base / "phase2_salad_ablation_deltafreq",
}

alpha_target = 5.0
rows = []
for name, rdir in runs.items():
    alpha_star = pd.read_csv(rdir / "alpha_star.csv")
    feat_summary = pd.read_csv(rdir / "feature_summary.csv")
    curves = pd.read_parquet(rdir / "curves_per_feature.parquet")

    n_feats = len(alpha_star)
    uncensored = alpha_star[alpha_star["censored_up"] == False]
    frac_unc = len(uncensored) / n_feats if n_feats > 0 else float("nan")
    mean_alpha_up = uncensored["alpha_star_feature_up"].mean()

    base_refusal = feat_summary["base_refusal_rate"].mean()

    curves_a = curves[curves["alpha"] == alpha_target]
    refusal_at_a = curves_a["mean_refusal"].mean() if len(curves_a) > 0 else float("nan")

    rows.append({
        "run": name,
        "n_features": n_feats,
        "frac_uncensored_up": frac_unc,
        "mean_alpha_star_up": mean_alpha_up,
        f"mean_refusal_alpha_{alpha_target}": refusal_at_a,
        "baseline_refusal": base_refusal,
    })

df = pd.DataFrame(rows)
print(df.to_string(index=False, float_format=lambda x: f"{x:0.3f}"))
```

**Resulting table:**

```text
               run  n_features  frac_uncensored_up  mean_alpha_star_up  mean_refusal_alpha_5.0  baseline_refusal
            legacy          25               0.000                 NaN                   0.849             0.867
     amax_filtered          25               1.000               4.680                   0.312             0.867
     contrast_only          25               0.960               5.125                   0.340             0.867
deltafreq_filtered          25               1.000               4.360                   0.333             0.867
```

All runs share essentially the same **baseline refusal** (~0.867), so differences are attributable to the steering configuration, not prompt sampling.

---

## 4. Findings by ablation

### 4.1 Ablation 1 – Steering method (activation‑scaled vs legacy)

- **Legacy steering** (`legacy`):
  - `frac_uncensored_up = 0.0`: all 25 features are censored (`censored_up == True`).
  - No `alpha_star_feature_up` exists; the mean is undefined (`NaN`).
  - Mean refusal at alpha 5 is **0.849**, essentially identical to baseline 0.867.
  - Qualitatively, the refusal curve is flat: pushing along raw `W_dec[f]` with a fixed magnitude does not reduce refusal on this task for this model.

- **Activation‑scaled steering** (`amax_filtered`):
  - `frac_uncensored_up = 1.0`: **all 25 features** have a defined alpha\*_up (strong effect).
  - `mean_alpha_star_up ≈ 4.68`: most features achieve the target refusal drop at relatively moderate alphas.
  - `mean_refusal_alpha_5.0 ≈ 0.312`: large reduction relative to baseline 0.867.
  - Refusal curves are generally smooth and monotone in alpha; many features achieve strong drops without obvious overshoot at alpha = 5–20.

**Conclusion (Ablation 1):**

- Activation‑scaled steering (Arad‑style: \(\alpha \cdot a_{\max} \cdot W_{\text{dec}}^{(i)}\)) is **necessary** for effective refusal control in this setup.
- Legacy fixed‑vector steering (\(\alpha \cdot W_{\text{dec}}^{(i)}\)) fails to produce meaningful refusal reduction for any of the 25 features.
- This strongly supports the Arad recipe that the SAE activations (via \(a_{\max}\)) should calibrate steering magnitude.

---

### 4.2 Ablation 2 – Output‑score filter (contrast‑only vs filtered)

This ablation compares:

- **`amax_filtered`**: activation‑scaled steering on top‑25 features that passed an **Arad output‑score** filter.
- **`contrast_only`**: activation‑scaled steering on top‑25 features from the **contrast ranking only** (composite scoring), **without** any output‑score filtering.

Key metrics:

- `amax_filtered`:  
  - `frac_uncensored_up = 1.0`  
  - `mean_alpha_star_up ≈ 4.68`  
  - `mean_refusal_alpha_5.0 ≈ 0.312`

- `contrast_only`:  
  - `frac_uncensored_up ≈ 0.96` (24/25 features uncensored)  
  - `mean_alpha_star_up ≈ 5.13` (slightly higher alphas needed on average)  
  - `mean_refusal_alpha_5.0 ≈ 0.340` (still a large drop from baseline, slightly weaker than `amax_filtered`)

Qualitative observations:

- The **contrast‑only** top‑25 set is already very strong: almost all features are steerable (uncensored) and achieve sizable refusal reductions.
- Adding the **output‑score filter** mildly improves things:
  - Slightly lower mean alpha\*_up (4.68 vs 5.13).
  - Slightly lower mean refusal at alpha 5 (0.312 vs 0.340).
- However, the **bulk of the effect** comes from activation‑scaled steering plus contrast selection; output‑score filtering provides an incremental benefit rather than being critical.

**Conclusion (Ablation 2):**

- In this IT setting, **output‑score filtering is not required** to obtain a strong pool of steerable features.
- Output scores help refine the feature set (slightly lower alphas and slightly better refusal at moderate alpha), but even a pure contrast‑only top‑25 list works very well.
- This suggests that, for practical steering on this task, one could simplify the pipeline by skipping output‑score filtering, especially when compute is tight, without losing most of the steering benefit.

---

### 4.3 Ablation 3 – Contrast scoring (composite vs delta\_freq)

This ablation compares:

- **`amax_filtered`**: candidate features selected via **composite scoring** (Bhargav & Zhu‑style) on contrast activations, then output‑score filtered.
- **`deltafreq_filtered`**: candidate features selected via **delta\_freq** (legacy contrast score), then output‑score filtered and truncated to top‑25.

Key metrics:

- `amax_filtered` (composite + output‑score):
  - `frac_uncensored_up = 1.0`
  - `mean_alpha_star_up ≈ 4.68`
  - `mean_refusal_alpha_5.0 ≈ 0.312`

- `deltafreq_filtered` (delta\_freq + output‑score):
  - `frac_uncensored_up = 1.0`
  - `mean_alpha_star_up ≈ 4.36`
  - `mean_refusal_alpha_5.0 ≈ 0.333`

Qualitative observations:

- Both scoring schemes produce **fully uncensored** top‑25 sets; every feature meets the refusal drop threshold somewhere on the alpha grid.
- Delta\_freq + output‑score yields slightly **lower mean alpha\*_up** (4.36 vs 4.68), but slightly **higher mean refusal at alpha 5** (0.333 vs 0.312).
- The differences are small; either scoring method can feed a strong pool of steerable features once activation‑scaled steering is in place.

**Conclusion (Ablation 3):**

- For this IT refusal task, **composite vs delta\_freq contrast does not qualitatively change steerability** once activation‑scaled steering and output‑score filtering are applied.
- Composite scoring may still be preferable for interpretability or generalization reasons (e.g., favoring features with stable, high differential activation and low variance), but empirically both schemes work well here.

---

## 5. Overall conclusions and recommended config

Putting all three ablations together:

- **Steering method (critical):**
  - Activation‑scaled steering (\(\alpha \cdot a_{\max} \cdot W_{\text{dec}}^{(i)}\)) is **essential**; legacy fixed‑vector steering shows no effect.

- **Output‑score filtering (helpful but not critical in this setup):**
  - Contrast‑only top‑25 features are already highly steerable.
  - Output‑score filtering slightly improves average alpha\* and refusal at moderate alpha but is not the main source of the gain.

- **Contrast scoring (secondary choice):**
  - Composite vs delta\_freq contrast both yield fully uncensored top‑25 sets when combined with activation‑scaled steering and, optionally, output‑score filtering.
  - The choice between them is a second‑order design decision for this task.

**Recommended configuration for larger IT runs:**

- **Steering:** activation‑scaled (Arad‑style) as implemented in `phase2_run.py` (no `--legacy_steering`).
- **Feature selection:**
  - Primary: **composite contrast** (`--scoring composite --pooling max`) to rank features.
  - Optional refinement: **output‑score filtering** via `compute_output_scores.py` to slightly clean up the top‑K set and align more closely with Arad et al.’s methodology.
- **Evaluation:** carry the **legacy steering** run as a baseline row in all future comparison tables (anchor for the prior setup), and log:
  - `frac_uncensored_up`,
  - `mean_alpha_star_up`,
  - `mean_refusal_alpha_{alpha_target}`,
  - `baseline_refusal`,
  together with representative refusal curves from `curves_per_feature`.

This document, together with `steering_arad_vs_phase2.md`, should provide a complete record of both **how** the Arad‑style steering was implemented and **which design choices actually mattered** on the IT refusal benchmark.

