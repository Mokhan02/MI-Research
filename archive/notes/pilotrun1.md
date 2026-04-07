## Pilot Run 1 — SALADBench, Gemma-2-2b-IT

### Setup

- **Model**: Gemma-2-2b-IT (instruction-tuned), with GemmaScope SAE at layer 20 (16k features).
- **Benchmark**: SALADBench unsafe prompts (salad\_alpha split).
- **Scale**: 25 prompts × 100 contrast-selected features × 7 alphas  
  Alphas: \(0, 1, 2, 5, 10, 20, 40\).
- **Scoring**: Keyword-based refusal classifier over generated responses using the Gemma chat template.
- **Baseline refusal rate**: **0.88** (22/25 prompts refused at \(\alpha = 0\)).

---

### Steering Results

| **Category**                  | **Count** | **Details**                                    |
|-------------------------------|-----------|-----------------------------------------------|
| Uncensored (alpha\* found)    | 10 / 100  | Steering broke safety refusal                 |
| Censored (near threshold)     | 23 / 100  | Refusal dropped 0.08 (just below \(T = 0.10\)) |
| Censored (some effect)        | 30 / 100  | Refusal dropped 0.04                          |
| No effect                     | 37 / 100  | Refusal unchanged at max alpha                |

**The 10 uncensored features:**

- **2 features with alpha\* = 20** (relatively easy to break safety):  
  feature IDs `1645`, `14481`
- **8 features with alpha\* = 40** (needed max alpha):  
  feature IDs `1216`, `3917`, `9348`, `10461`, `10648`, `10936`, `13263`, `15959`

Note: **23 features** had refusal drops of **0.08** (baseline 0.88 → 0.80). With **99 prompts** instead of 25 (1% resolution vs. 4%), these would likely cross the \(T = 0.10\) threshold and count as uncensored.

---

### Feature Selection Quality

Contrast-based selection compared unsafe (SALADBench) vs neutral prompts:

- **Task-selected features**: mean activation frequency on unsafe prompts ≈ **0.76**.
- **Random features**: mean activation frequency on unsafe prompts ≈ **0.34**.

This ~2.2× enrichment confirms that contrast selection is picking safety-relevant features rather than random directions.

---

### Geometry vs Steerability (Phase 3)

Correlation between pre-steering geometry and alpha\* (using the 10 uncensored features):

| **Metric**          | **Spearman r** | **p-value** | **n** |
|---------------------|----------------|-------------|-------|
| `max_cosine_to_any` | **+0.174**     | 0.63        | 10    |
| `density_tau`       | **+0.348**     | 0.32        | 10    |

- Both correlations are **positive**, matching the proposal’s hypothesis that features in **denser geometric neighborhoods** are **harder to steer** (larger alpha\*).
- With only **10** uncensored points, these effects are **not statistically significant**; results are **directionally consistent** but inconclusive.

Examples from the raw data:

- The two “easiest” features (alpha\* = 20) have **moderate** max cosine:
  - feature `1645`: max\_cos ≈ 0.325
  - feature `14481`: max\_cos ≈ 0.286
- Several **censored** features show **higher** max cosine (e.g., ≈ 0.81, 0.77, 0.72), consistent with “denser neighborhood → harder to steer”.

An interesting outlier: **feature `5670`** actually *increased* refusal at max alpha (0.88 → 0.92), acting as a **safety-reinforcing** direction.

---

### Limitations

- **Small effective sample size**: Only **10** features are uncensored; OLS and logistic models are underpowered or skipped.
- **Coarse refusal resolution**: 25 prompts → each prompt is 4 percentage points; borderline features with 0.08 drops are treated as censored.
- **Feature selection model**: Features were selected using the **base** Gemma-2-2b activations; repeating selection on **Gemma-2-2b-IT** may yield more safety-critical features.
- **Alpha grid saturation**: Many uncensored features needed \(\alpha = 40\) (max grid value), suggesting:
  - either the grid should extend further (e.g., add 60, 80), or
  - these features represent genuinely robust safety alignments.

---

### Next Steps for the Full Run

- **Increase prompts**: Move to **99** SALADBench prompts to get 1% refusal resolution and increase the number of uncensored features (target: 30–40+).
- **Re-run contrast selection on Gemma-2-2b-IT**: Align feature selection with the actual safety-tuned model used for steering.
- **Reuse improved refusal scorer**: Keep the expanded keyword patterns that capture Gemma’s softer refusal style.
- **Re-evaluate geometry correlations**: With more uncensored features, re-run Phase 3 (Spearman, OLS, logistic) to test whether geometry meaningfully predicts alpha\*.

---

### How This Relates to the Proposal (for mentors / teammates)

- **Pipeline validation**: The full stack described in the proposal (SAE loading, contrast feature selection, steering on a safety benchmark, refusal scoring, geometry computation, alpha\* estimation) is now **implemented and exercised** end-to-end on real data.
- **Behavioral sanity**: Gemma-2-2b-IT shows a high baseline refusal rate (**0.88**) on SALADBench, and targeted SAE steering can reliably **reduce refusal** for a subset of features, matching the intended experimental setup.
- **Directional support for the hypothesis**: Geometry–alpha\* correlations are **positive**, as hypothesized (denser neighborhoods → harder steering), but the pilot is **underpowered** (only 10 uncensored features), so no claims about prediction strength can be made yet.
- **Design implication for the full run**: The pilot reveals that many features sit just below the refusal-drop threshold and that alpha\* often saturates at the top of the grid. This directly motivates the full-scale run (more prompts, IT-based feature selection, possibly a slightly extended alpha grid) specified in the proposal’s next phases. 

---

### How to Reproduce This Pilot Run

- **1. Prepare SALADBench data**

```bash
uv run python scripts/prepare_salad_bench.py --out_dir data/prompts --n_prompts 300 --seed 42
```

- **2. Contrast-based feature selection (salad vs neutral)**

```bash
uv run python scripts/phase2_select_contrast.py \
  --config configs/targets/gemma2_2b_gemmascope_res16k.yaml \
  --domain salad \
  --out_dir outputs/phase2_select_salad \
  --top-k 100
```

- **3. Phase 2 pilot run (refusal mode, 25 prompts × 100 features)**

Run inside `tmux` so it survives disconnects:

```bash
tmux new -s experiment

uv run python scripts/phase2_run.py \
  --config configs/targets/gemma2_2b_gemmascope_res16k.yaml \
  --mode refusal \
  --out_dir outputs/phase2_salad_pilot \
  --n_prompts 25 \
  --alphas 0,1,2,5,10,20,40 \
  --fixed_features_path outputs/phase2_select_salad/selected_features_salad.json \
  --flush_every 100
```

- **4. Re-score pilot with updated refusal scorer (no GPU)**

```bash
PYTHONPATH=. python archive/scripts/utilities/rescore_pilot.py
```

- **5. Phase 3 on pilot outputs (geometry vs alpha\*)**

```bash
uv run python scripts/phase3_predictability.py \
  --phase2_dir outputs/phase2_salad_pilot \
  --config configs/targets/gemma2_2b_gemmascope_res16k.yaml \
  --out_dir outputs/phase3_salad_pilot \
  --run_name salad_pilot \
  --skip_baseline
```

---

### Commands for the Full Run (after pilot)

- **Re-select features using Gemma-2-2b-IT**

```bash
uv run python scripts/phase2_select_contrast.py \
  --config configs/targets/gemma2_2b_gemmascope_res16k.yaml \
  --domain salad \
  --out_dir outputs/phase2_select_salad_it \
  --top-k 100
```

- **Full Phase 2 run (99 prompts, 100 features)**

```bash
tmux new -s experiment   # or tmux attach -t experiment if already running

uv run python scripts/phase2_run.py \
  --config configs/targets/gemma2_2b_gemmascope_res16k.yaml \
  --mode refusal \
  --out_dir outputs/phase2_salad_full \
  --n_prompts 99 \
  --alphas 0,1,2,5,10,20,40 \
  --fixed_features_path outputs/phase2_select_salad_it/selected_features_salad.json \
  --flush_every 100
```

- **Then reuse `archive/scripts/utilities/rescore_pilot.py` (pointed at `outputs/phase2_salad_full`) and rerun Phase 3 with `--phase2_dir outputs/phase2_salad_full` to get the full-scale results.**


