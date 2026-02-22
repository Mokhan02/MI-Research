# scripts/compute_alignment_mean_target.py
import argparse, json
import numpy as np
import pandas as pd
import torch

from src.config import load_config, resolve_config
from src.model_utils import load_model_and_tokenizer
from src.sae_loader import load_sae_decoder

def load_feature_ids(path: str):
    if path.endswith(".json"):
        obj = json.load(open(path))
        if isinstance(obj, list):
            return [int(x) for x in obj]
        for k in ["feature_ids", "selected_features", "ids"]:
            if k in obj:
                return [int(x) for x in obj[k]]
        raise ValueError(f"Unrecognized json schema in {path}")
    # txt: one id per line
    return [int(x.strip()) for x in open(path) if x.strip()]

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--prompt_csv", required=True)          # must contain target column
    ap.add_argument("--feature_ids_file", required=True)    # txt or json
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    config = resolve_config(load_config(args.config), run_id="alignment")
    if args.device:
        config["model"]["device"] = args.device

    df = pd.read_csv(args.prompt_csv)
    if "target" not in df.columns:
        raise ValueError("prompt_csv must have 'target' column")
    targets = [t for t in df["target"].astype(str).tolist() if t.strip()]
    if not targets:
        raise ValueError("No non-empty targets found")

    model, tok = load_model_and_tokenizer(config["model"])
    # unembedding / lm_head weights: [vocab, d_model]
    # HF models usually: model.get_output_embeddings().weight
    U = model.get_output_embeddings().weight.detach().float().cpu().numpy()

    # mean target direction in residual space
    ids = []
    for t in targets:
        tid = tok.encode(t, add_special_tokens=False)
        if len(tid) != 1:
            # we need single-token targets for this analysis; skip otherwise
            continue
        ids.append(tid[0])
    if not ids:
        raise ValueError("No single-token targets after tokenization. Fix targets or choose another method.")
    u_mean = U[ids].mean(axis=0)
    u_mean = u_mean / (np.linalg.norm(u_mean) + 1e-12)

    W_dec = load_sae_decoder(config["sae"]).detach().float().cpu().numpy()  # [n_feat, d_model]
    feats = load_feature_ids(args.feature_ids_file)

    rows = []
    for f in feats:
        w = W_dec[f]
        w = w / (np.linalg.norm(w) + 1e-12)
        cos = float(np.dot(w, u_mean))
        rows.append({"feature_id": int(f), "cos_wdec_u_target_mean": cos})

    out = pd.DataFrame(rows)
    out.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv} rows={len(out)}  (targets used={len(ids)}/{len(targets)} single-token)")

if __name__ == "__main__":
    main()
