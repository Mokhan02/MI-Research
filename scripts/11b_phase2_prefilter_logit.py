# scripts/11b_phase2_prefilter_logit.py
import argparse
from pathlib import Path
import pandas as pd
import torch

from src.config import load_config, resolve_config
from src.model_utils import load_model
from src.sae_loader import load_gemmascope_decoder

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out", default="outputs/phase2_smoke/prefilter_logit.csv")
    ap.add_argument("--token-text", default=" two")   # target answer token (single token!)
    ap.add_argument("--top-k", type=int, default=200)
    args = ap.parse_args()

    cfg = resolve_config(load_config(args.config), run_id=args.run_id)
    model, tok = load_model(cfg)
    model.eval()
    device = next(model.parameters()).device

    # Decoder: [n_features, d_model]
    decoder, meta = load_gemmascope_decoder(cfg)
    decoder = decoder.to(device=device, dtype=torch.float32)

    # Unembed: Gemma2 ties lm_head? Either way, use lm_head weight.
    W_U = model.get_output_embeddings().weight.to(device=device, dtype=torch.float32)  # [vocab, d_model]

    # Ensure single-token target
    ids = tok.encode(args.token_text, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"token-text {args.token_text!r} is not single-token: ids={ids}")
    tid = ids[0]

    u = W_U[tid]  # [d_model]
    # Dot with each feature direction
    scores = torch.matmul(decoder, u)  # [n_features]

    topv, topi = torch.topk(scores, k=min(args.top_k, scores.numel()))
    df = pd.DataFrame({
        "feature_id": topi.detach().cpu().numpy(),
        "logit_dot": topv.detach().cpu().numpy(),
        "token_text": args.token_text,
        "token_id": tid,
    })
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")
    print("Top 10 feature_ids:", df["feature_id"].head(10).tolist())

if __name__ == "__main__":
    main()
