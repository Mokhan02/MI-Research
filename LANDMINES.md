# Landmines to fix or audit before you run

## 1) tau_act: 0.0 is almost certainly too permissive

With SAEs, many features have tiny positive activations. With **tau_act=0.0**, act_freq is "fraction of tokens where act > 0," which is noisy and can drift across seeds.

**Do this before running selection:**
- Run `scripts/audit_activation_distribution.py` on a baseline (e.g. neutral_select.csv).
- Inspect percentiles of activations (especially conditional on act > 0).
- Set **tau_act** to cut out near-zero fuzz: e.g. 25th percentile of positive activations, or a small fixed value (0.01–0.05). Re-audit if you change layer/model.

Config default is now 0.01; still run the audit and adjust.

---

## 2) token_span: "last" with N=1 is a trap

Last token only is cheap but **extremely sensitive to formatting** (trailing punctuation, "Answer:", etc.). You can select "response formatting" features instead of task semantics.

**Safer default:** `token_span: "last_n"`, `token_span_last_n: 8` (or 16).  
If you keep N=1, **enforce identical prompt formatting** across task and neutral.

Config default is now last_n=8.

---

## 3) Success threshold T must be pre-registered

**delta_logit_target ≥ T** is the success metric. If T is chosen after you see results, that’s p-hacking.

**Before running:**
- Choose T from baseline logit std or a **pilot run** (e.g. micro sweep).
- **Freeze T** in config and do not change it for the main run.

**Aggregation:** We use **feature-level**: mean(delta_logit_target) over prompts, then threshold. That is *not* the same as "per-prompt success then aggregate fraction." Be explicit which you use (we use: alpha* = smallest |alpha| where mean(delta) ≥ T).

---

## 4) Directionality: helpful vs harmful

- **Alpha* (official):** "Helpful" direction only — push **toward** the correct target (delta_logit_target ≥ T for positive alpha).
- **Harmful direction** (push away from target) should be tracked **separately** as off-target risk, not folded into alpha*.

We report both up and down (success_rate_up/down, alpha_star_feature_up/down). For interpretability, treat "up" as toward target when that’s the intended direction, and treat strong wrong-way effects as off-target.

---

## 5) Matched-random control: matching on act_freq may not be enough

act_freq correlates with norm, generality, etc. Matching only on act_freq can leave **||W_dec||** (and mean activation magnitude) different between task-selected and control, which can confound geometry correlations.

**Minimum audit:** After selection, run the **control audit** (written by `phase2_select_contrast.py` to `control_audit.txt`). Compare:
- act_freq_task  
- ||W_dec||  
- mean_act_task  

If norms differ a lot, match on norm too (or control for norm in analysis).

---

## 6) Neutral prompts must be structurally isomorphic to task prompts

If task prompts have targets/labels and neutral doesn’t, or lengths/punctuation/scaffolding differ, the contrast is confounded (you’re comparing different sentence types).

**Rule:** Neutral prompts must be **structurally isomorphic**: same style, similar length, same "Answer:" (or equivalent) scaffolding. No extra named entities or formatting only in one set.

Audit your neutral_select vs task _select CSVs before running selection.
