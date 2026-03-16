## 2026-03-16 – IT Refusal Ablations (planned runs)

This report is a stub for the next round of IT refusal ablations. It records the **exact commands** to re-create the planned runs with the current codebase.

Once the runs complete, empirical results and W&B links can be appended here.

### Environment

- Python and dependencies are managed via `uv` from the project root:

```bash
uv sync
```

- Secrets are provided via `.env` in the project root:

```text
HF_TOKEN=your_hf_token_here
WANDB_API_KEY=your_wandb_api_key_here
WANDB_PROJECT=sae-refusal-steering
WANDB_ENTITY=your_wandb_entity_here
WANDB_MODE=online
```

### Commands for this ablation

All commands are intended to be run from the project root (`MI-Research/`) with `PYTHONPATH=.` so that `src/` is importable.

1. **Contrast feature selection**

```bash
PYTHONPATH=. uv run python scripts/phase2_select_contrast.py \
  --config configs/targets/gemma2_2b_gemmascope_res16k.yaml \
  --domain planets \
  --out_dir outputs/phase2_select \
  --top-k 100
```

2. **Steerability measurement (phase 2)**

```bash
PYTHONPATH=. uv run python scripts/phase2_run.py \
  --config configs/targets/gemma2_2b_gemmascope_res16k.yaml \
  --out_dir outputs/phase2 \
  --n_prompts 100 \
  --fixed_features_path outputs/phase2_select/selected_features_planets.json
```

3. **Predictability analysis (phase 3)**

```bash
PYTHONPATH=. uv run python scripts/phase3_predictability.py \
  --config configs/targets/gemma2_2b_gemmascope_res16k.yaml \
  --phase2_dir outputs/phase2 \
  --out_dir outputs/phase3
```

### Notes

- After the runs finish, append:
  - A brief description of each run (any deviations from the commands above).
  - Key scalar metrics (e.g., baseline refusal, max refusal drop, alpha* distribution).
  - Pointers to the corresponding W&B runs and artifacts.

