# Run gates (priority order)

Treat these as gates: if you can't check one off, don't run the full pipeline.

## 1) Domain-specific prompt splits

Selection and evaluation must not share prompts (otherwise selection is contaminated).

- **Create three CSVs per domain** (planets, capitals, neutral):
  - `*_select.csv` — used **only** for feature selection
  - `*_alpha.csv` — used **only** for α-grid / α* runs
  - `*_holdout.csv` — used **only** for final reporting

**How:** Run `scripts/make_domain_splits.py`. Optionally use `--data_dir data` if you have `planets_only.csv` etc. with targets so alpha/holdout get targets.

**Landmine:** Neutral prompts must be **structurally isomorphic** to task prompts (same style, length, scaffolding). Otherwise contrast is confounded (see LANDMINES.md).

```bash
uv run python scripts/make_domain_splits.py --prompts_dir data/prompts --out_dir data/prompts [--data_dir data]
```

## 2) Contrast-based feature selection

Do **not** use "top K act_freq on task". Use:

- **Top K by Δ = task − neutral**
  - `delta_freq = act_freq_task - act_freq_neutral` (preferred), or
  - `delta_mean = mean_act_task - mean_act_neutral`

**How:** Run `scripts/phase2_select_contrast.py` (uses `*_select.csv` only). It writes top K by delta_freq.

```bash
uv run python scripts/phase2_select_contrast.py --domain planets --top-k 100 --out_dir outputs/phase2_select
```

## 3) Define "active" precisely

- **τ_act**: Do **not** leave at 0.0 (noisy). Run `scripts/audit_activation_distribution.py` and set from percentiles or pilot.
- **Token span**: Prefer `last_n` with N=8 or 16. `last` with N=1 is formatting-sensitive (see LANDMINES.md).

Rule: **"Active" means act > τ_act on token window Y.** Keep consistent across domains and runs.

**Where:** `configs/...` under `features`: `tau_act`, `token_span`, `token_span_last_n`. See **LANDMINES.md** for tau_act and token_span traps.

## 4) Persist selection artifacts

Every selection run must write:

- `feature_summary_planets.csv` (or capitals)
- `feature_summary_neutral.csv`
- `selected_features_planets.json` (the list you will steer with)

**How:** `phase2_select_contrast.py` writes these to `--out_dir`.

## 5) Matched-random control feature set

For each task domain you need:

- `topK_planets_delta.json` — task contrast features
- `randK_matched_actfreq.json` — random K control
- **Audit:** `phase2_select_contrast.py` writes `control_audit.txt` comparing task vs rand for act_freq_task, ||W_dec||, mean_act. If ||W_dec|| differs a lot, match on norm or control in analysis (see LANDMINES.md).

## 6) α-grid micro sweep first

Before a full K=100 run:

- Run a **micro sweep**: 10 features, 20–30 prompts, α = [0, 0.5, 1, 2, 5].
- Only if task features outperform random, launch the full run.

**How:** `phase2_run.py --micro_sweep` (sets n_features=10, n_prompts=25, alphas=0,0.5,1,2,5). Use `*_alpha.csv` as prompt_csv.

```bash
uv run python scripts/phase2_run.py --micro_sweep --prompt_csv data/prompts/planets_alpha.csv --fixed_features_path outputs/phase2_select/selected_features_planets.json --out_dir outputs/phase2_micro
```

## 7) α* rule and no-effect handling

- **Success metric:** `delta_logit_target` (steer_t − base_t) ≥ T (up) or ≤ −T (down). This is the only official success definition for alpha*.
- **No-effect:** Features that never reach threshold in the alpha range are **censored** (censored_up/censored_down = True). Do **not** set alpha* = max(alpha); treat them as "no-effect" in correlations and reporting. (Treating censored as α* = max would create fake patterns.)

**Where:** `summarize_feature_directional()` in `phase2_run.py` implements this; alpha_star.csv and feature_summary include `censored_up`, `censored_down`.
