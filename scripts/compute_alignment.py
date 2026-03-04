import argparse, json
import numpy as np
import pandas as pd
import torch

from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.config import load_config, resolve_config

def load_feature_ids(path: str):
    if path.endswith(".json"):
        obj = json.load(open(path))
        if isinstance(obj, list):
            return [int(x) for x in obj]
        for k in ["feature_ids", "selected_features", "ids"]:
            if k in obj:
                return [int(x) for x in obj[k]]
        raise ValueError(f"Unrecognized json schema in {path}")
    return [int(x.strip()) for x in open(path) if x.strip()]

def load_wdec_from_npz(weights_repo: str, weights_path: str):
    local = hf_hub_download(repo_id=weights_repo, filename=weights_path)
    npz = np.load(local)
    if "W_dec" in npz:
        W = npz["W_dec"]
    else:
        # fallback: pick first 2D array
        key = None
        for k in npz.files:
            if npz[k].ndim == 2:
                key = k
                break
        if key is None:
            raise ValueError(f"No 2D arrays found in NPZ. Keys={npz.files}")
        W = npz[key]
    return W  # [n_feat, d_model]

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--prompt_csv", required=True)          # must contain target
    ap.add_argument("--feature_ids_file", required=True)    # txt or json
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--agg", default="mean_max", choices=["mean", "max", "mean_max"])
    args = ap.parse_args()

    config = resolve_config(load_config(args.config), run_id="alignment_per_example")
    model_id = config["model"]["model_id"]
    device = args.device or config["model"].get("device", "cuda")
    dtype = config["model"].get("dtype", "bfloat16")
    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}.get(dtype, torch.bfloat16)

    df = pd.read_csv(args.prompt_csv)
    if "target" not in df.columns:
        raise ValueError("prompt_csv must have 'target' column")
    targets = [t for t in df["target"].astype(str).tolist() if t.strip()]
    if not targets:
        raise ValueError("No non-empty targets found")

    print("Loading HF model + tokenizer...")
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype, device_map=None).to(device)
    model.eval()

    U = model.get_output_embeddings().weight.detach().float().cpu().numpy()  # [vocab, d_model]

    # Per-example target vectors (single-token only)
    target_ids = []
    skipped = 0
    for t in targets:
        tid = tok.encode(t, add_special_tokens=False)
        if len(tid) != 1:
            skipped += 1
            continue
        target_ids.append(tid[0])
    if not target_ids:
        raise ValueError(f"No single-token targets after tokenization. Skipped multi-token={skipped}/{len(targets)}")

    Ut = U[target_ids]  # [n_targets, d_model]
    Ut = Ut / (np.linalg.norm(Ut, axis=1, keepdims=True) + 1e-12)

    sae_cfg = config["sae"]
    W_dec = load_wdec_from_npz(sae_cfg["weights_repo"], sae_cfg["weights_path"]).astype(np.float32)  # [n_feat, d_model]
    if bool(sae_cfg.get("normalize_decoder", False)):
        W_dec = W_dec / (np.linalg.norm(W_dec, axis=1, keepdims=True) + 1e-12)

    feats = load_feature_ids(args.feature_ids_file)

    rows = []
    for f in feats:
        w = W_dec[int(f)]
        w = w / (np.linalg.norm(w) + 1e-12)
        # cosine against EACH target vector
        cos_all = Ut @ w  # [n_targets]
        row = {"feature_id": int(f)}
        if args.agg in ("mean", "mean_max"):
            row["cos_wdec_u_target_perex_mean"] = float(cos_all.mean())
        if args.agg in ("max", "mean_max"):
            row["cos_wdec_u_target_perex_max"] = float(cos_all.max())
        rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv} rows={len(out)} targets_used={len(target_ids)}/{len(targets)} skipped_multi={skipped}")

if __name__ == "__main__":
    main()
