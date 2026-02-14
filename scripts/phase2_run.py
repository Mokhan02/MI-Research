# scripts/phase2_run.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os, json, argparse, random, time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from itertools import product
from tqdm.auto import tqdm

from src.config import load_config, resolve_config
from src.model_utils import load_model
from src.sae_loader import load_gemmascope_decoder

# --------------------------
# Utils
# --------------------------
def set_determinism(seed: int = 1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

def nanmean_or_nan(x):
    """Mean of finite values, or NaN if none. No RuntimeWarning on empty slice."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) > 0 else np.nan

def tv_distance_from_logits(logits_a: torch.Tensor, logits_b: torch.Tensor) -> float:
    p = F.softmax(logits_a, dim=-1)
    q = F.softmax(logits_b, dim=-1)
    return (0.5 * (p - q).abs().sum()).item()

def kl_pq_from_logits(logits_p: torch.Tensor, logits_q: torch.Tensor) -> float:
    p = F.softmax(logits_p, dim=-1)
    q = F.softmax(logits_q, dim=-1)
    return (p * (torch.log(p + 1e-12) - torch.log(q + 1e-12))).sum().item()

def topk_jaccard(logits_a: torch.Tensor, logits_b: torch.Tensor, k: int = 50) -> float:
    top_a = torch.topk(logits_a, k=k).indices
    top_b = torch.topk(logits_b, k=k).indices
    set_a = set(top_a.tolist())
    set_b = set(top_b.tolist())
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union else 1.0

@torch.no_grad()
def forward_last_logits(model, tokenizer, prompt: str, prehook=None):
    """Returns (logits_last_fp32, input_ids, resid_last_fp32)."""
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    handle = prehook() if prehook else None
    out = model(**inputs, output_hidden_states=True)
    if handle is not None:
        handle.remove()
    logits = out.logits[0, -1, :].float().detach()
    # Final hidden state at last position (after all layers + final norm)
    resid_last = out.hidden_states[-1][0, -1, :].float().detach()
    return logits, inputs["input_ids"][0], resid_last

def make_steer_prehook(model, layer_idx: int, alpha: float, pos: int, steer_dir: torch.Tensor):
    ran = {"ok": False}
    def _mk():
        def _prehook(module, inputs):
            hidden = inputs[0]  # [1, seq, d_model]
            ran["ok"] = True
            hidden2 = hidden.clone()
            hidden2[0, pos, :] = hidden2[0, pos, :] + alpha * steer_dir
            return (hidden2,) + inputs[1:]
        layer = model.model.layers[layer_idx]
        return layer.register_forward_pre_hook(_prehook)
    return _mk, ran

def encode_target(tokenizer, t):
    if t is None or (isinstance(t, float) and np.isnan(t)) or (isinstance(t, str) and t.strip() == ""):
        return None

    # If it's numeric, force string
    if isinstance(t, (int, np.integer, float, np.floating)):
        # if it's float but integer-valued, make it clean
        if isinstance(t, (float, np.floating)) and float(t).is_integer():
            t = str(int(t))
        else:
            t = str(t)

    # If it's pandas scalar
    if isinstance(t, (pd.Timestamp, pd.Timedelta)):
        t = str(t)

    # Now must be str
    t = str(t)

    # Tokenizers often expect leading space for "word" tokens
    if not t.startswith(" "):
        t = " " + t

    ids = tokenizer.encode(t, add_special_tokens=False)
    if len(ids) != 1:
        # Multi-token target: return first token (best effort)
        pass
    return ids[0]

def _find_alpha_star(df_direction, prompts, threshold_fn, alphas_ordered):
    """
    Scan alphas in order of increasing |alpha|.  For each prompt, find the
    first alpha where threshold_fn(delta) is True.

    Returns: dict  prompt_idx -> (alpha_star, tv_at_crossing)
    Both values come from the SAME run_rows row.
    """
    # Index df by (prompt_idx, alpha) for fast lookup
    keyed = df_direction.set_index(["prompt_idx", "alpha"])
    hits = {}
    for pidx in prompts:
        for a in alphas_ordered:
            if (pidx, a) not in keyed.index:
                continue
            row = keyed.loc[(pidx, a)]
            # row could be a Series (single match) or DataFrame (shouldn't happen
            # with unique (prompt, feature, alpha) but be safe)
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            if threshold_fn(row["delta_logit_target"]):
                hits[pidx] = (abs(a), row["tv_distance"])
                break
    return hits

def _monotone_fraction(df_direction, prompts, alphas_ordered, sign=1):
    """
    For each prompt, check how many consecutive alpha steps show
    delta moving in the expected direction (sign=+1: increasing, sign=-1: decreasing).
    Returns fraction of prompts with >= 2/3 monotone steps.
    """
    keyed = df_direction.set_index(["prompt_idx", "alpha"])
    n_mono_prompts = 0
    n_total = 0
    for pidx in prompts:
        deltas = []
        for a in alphas_ordered:
            if (pidx, a) not in keyed.index:
                continue
            row = keyed.loc[(pidx, a)]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            deltas.append(row["delta_logit_target"])
        if len(deltas) < 2:
            continue
        n_total += 1
        n_steps = len(deltas) - 1
        monotone_steps = sum(
            1 for i in range(n_steps)
            if sign * (deltas[i + 1] - deltas[i]) >= 0
        )
        if monotone_steps >= (2 * n_steps / 3):
            n_mono_prompts += 1
    return n_mono_prompts / n_total if n_total > 0 else np.nan

def summarize_feature_directional(df_feat, tau: float):
    """
    df_feat: run_rows filtered to a single feature_id (all prompts, all alphas).

    Directional convention:
      - success_up   uses positive alphas:  delta_logit_target >= +tau
      - success_down uses negative alphas:  delta_logit_target <= -tau

    alpha_star is found by explicit scan in order of increasing |alpha|.
    tv_at_alpha_star comes from the exact same (prompt, feature, alpha) row.

    Also computes sign consistency (monotone_frac_up/down): fraction of
    prompts where >= 2/3 of consecutive alpha steps are monotone in the
    expected direction.
    """
    pos = df_feat[df_feat["alpha"] > 0].copy()
    neg = df_feat[df_feat["alpha"] < 0].copy()
    prompts = df_feat["prompt_idx"].unique()

    # Explicit alpha orderings by increasing |alpha|
    pos_alphas = sorted(pos["alpha"].unique(), key=lambda a: abs(a))
    neg_alphas = sorted(neg["alpha"].unique(), key=lambda a: abs(a))

    result = dict(
        success_rate_up=0.0,
        success_rate_down=0.0,
        alpha_star_mean_up=np.nan,
        alpha_star_mean_down=np.nan,
        tv_at_alpha_star_up=np.nan,
        tv_at_alpha_star_down=np.nan,
        monotone_frac_up=np.nan,
        monotone_frac_down=np.nan,
    )

    # --- UP direction (positive alphas, delta >= +tau) ---
    if len(pos) > 0:
        mx = pos.groupby("prompt_idx")["delta_logit_target"].max()
        succ_up = (mx >= tau)
        result["success_rate_up"] = float(succ_up.mean())

        hits_up = _find_alpha_star(
            pos, prompts,
            threshold_fn=lambda d: d >= tau,
            alphas_ordered=pos_alphas,
        )
        if hits_up:
            result["alpha_star_mean_up"] = nanmean_or_nan([v[0] for v in hits_up.values()])
            result["tv_at_alpha_star_up"] = nanmean_or_nan([v[1] for v in hits_up.values()])

        result["monotone_frac_up"] = _monotone_fraction(
            pos, prompts, pos_alphas, sign=+1,
        )

    # --- DOWN direction (negative alphas, delta <= -tau) ---
    if len(neg) > 0:
        mn = neg.groupby("prompt_idx")["delta_logit_target"].min()
        succ_down = (mn <= -tau)
        result["success_rate_down"] = float(succ_down.mean())

        hits_down = _find_alpha_star(
            neg, prompts,
            threshold_fn=lambda d: d <= -tau,
            alphas_ordered=neg_alphas,  # [-1, -2, -5, ...] by increasing |alpha|
        )
        if hits_down:
            result["alpha_star_mean_down"] = nanmean_or_nan([v[0] for v in hits_down.values()])
            result["tv_at_alpha_star_down"] = nanmean_or_nan([v[1] for v in hits_down.values()])

        result["monotone_frac_down"] = _monotone_fraction(
            neg, prompts, neg_alphas, sign=-1,
        )

    return result

# --------------------------
# Main
# --------------------------
def main():
    # Preprocess sys.argv: merge --alphas <value> into --alphas=<value>
    # so argparse doesn't choke on values starting with '-'
    for i in range(len(sys.argv) - 1):
        if sys.argv[i] == "--alphas":
            sys.argv[i] = "--alphas=" + sys.argv[i + 1]
            del sys.argv[i + 1]
            break

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/targets/gemma2_2b_gemmascope_res16k.yaml")
    ap.add_argument("--prompt_csv", type=str, default="data/phase2_prompts.csv")
    ap.add_argument("--out_dir", type=str, default="outputs/phase2")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--n_prompts", type=int, default=200)
    ap.add_argument("--n_features", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--alphas", type=str, default="-40,-20,-10,-5,-2,-1,0,1,2,5,10,20,40",
                        help="Comma-separated alpha values, e.g. -40,-20,-10,-5,-2,-1,0,1,2,5,10,20,40")
    args = ap.parse_args()

    set_determinism(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # Load prompts
    dfp = pd.read_csv(args.prompt_csv, dtype={"prompt": "string", "target": "string"})
    assert "prompt" in dfp.columns, "prompt_csv must have a 'prompt' column"
    if "target" not in dfp.columns:
        dfp["target"] = np.nan
    dfp = dfp.head(args.n_prompts).reset_index(drop=True)

    # Load model via config
    config = resolve_config(load_config(args.config), run_id="phase2_run")
    config["model"]["do_sample"] = False
    config["model"]["temperature"] = 0.0
    model, tokenizer = load_model(config)
    model.eval()
    device = next(model.parameters()).device

    # Load SAE params (same loader as phase1_smoke / debug_steer_effect)
    _, meta = load_gemmascope_decoder(config)
    npz_path = meta["npz_path"]
    data = np.load(npz_path)
    W_dec = torch.tensor(np.asarray(data["W_dec"], dtype=np.float32), device=device, dtype=torch.float32)
    print(f"W_dec: {W_dec.shape}")

    # Choose features
    n_feats_total = W_dec.shape[0]
    rng = np.random.default_rng(args.seed)
    feature_ids = rng.choice(n_feats_total, size=min(args.n_features, n_feats_total), replace=False).tolist()

    alphas = [float(x) for x in args.alphas.split(",")]
    alphas_sorted = sorted(alphas, key=lambda a: abs(a))

    # lm_head weight for fp32 target logit computation
    lm_head_w = model.lm_head.weight.detach().float()  # [vocab, d_model]

    rows = []

    # Pre-compute per-prompt baselines (one forward pass each)
    print(f"Computing baselines for {len(dfp)} prompts...")
    prompt_cache = {}
    for pi, prow in tqdm(dfp.iterrows(), total=len(dfp), desc="Baselines"):
        prompt = prow["prompt"]
        target_id = encode_target(tokenizer, prow.get("target", np.nan))
        base_logits, input_ids, base_resid = forward_last_logits(model, tokenizer, prompt)
        seq_len = int(input_ids.shape[0])
        pos = seq_len - 1
        if target_id is None:
            target_id = int(torch.argmax(base_logits).item())
        w_t = lm_head_w[target_id]
        base_t = torch.dot(base_resid, w_t)
        prompt_cache[pi] = dict(
            prompt=prompt, base_logits=base_logits, base_resid=base_resid,
            seq_len=seq_len, pos=pos, target_id=target_id, w_t=w_t, base_t=base_t,
        )

    # Flat iterator over all (prompt, feature, alpha) jobs
    all_jobs = list(product(dfp.index, feature_ids, alphas_sorted))
    total_tasks = len(all_jobs)
    start_time = time.time()
    last_flush_pi = -1
    partial_path = os.path.join(args.out_dir, "run_rows_partial.csv")

    print(f"Total tasks: {total_tasks} ({len(dfp)} prompts x {len(feature_ids)} features x {len(alphas_sorted)} alphas)")

    for task_i, (pi, fid, alpha) in enumerate(tqdm(all_jobs, total=total_tasks, desc="Steering")):
        pc = prompt_cache[pi]
        steer_dir = W_dec[fid].to(device=device, dtype=torch.float32)

        if alpha == 0.0:
            steer_logits = pc["base_logits"]
            steer_resid = pc["base_resid"]
            hook_ran = True
        else:
            mk_hook, cap = make_steer_prehook(model, args.layer, alpha, pc["pos"], steer_dir)
            t0 = time.time()
            steer_logits, _, steer_resid = forward_last_logits(model, tokenizer, pc["prompt"], prehook=mk_hook)
            t1 = time.time()
            hook_ran = cap["ok"]
            if (t1 - t0) > 2.0:
                print(f"[SLOW] forward {t1-t0:.2f}s  prompt={pi} feat={fid} alpha={alpha}")

        # fp32 delta_logit_target from residuals
        steer_t = torch.dot(steer_resid, pc["w_t"])
        delta_target = float((steer_t - pc["base_t"]).item())

        # Off-target metrics still from full logits
        d = steer_logits - pc["base_logits"]
        max_abs = float(d.abs().max().item())
        tv = tv_distance_from_logits(pc["base_logits"], steer_logits)
        kl = kl_pq_from_logits(pc["base_logits"], steer_logits)
        jac = topk_jaccard(pc["base_logits"], steer_logits, k=args.topk)

        # Stricter metrics: rank change of target token
        tid = pc["target_id"]
        base_rank = int((pc["base_logits"] >= pc["base_logits"][tid]).sum().item())
        steer_rank = int((steer_logits >= steer_logits[tid]).sum().item())
        target_is_top1 = bool(steer_rank == 1)

        rows.append({
            "prompt_idx": pi,
            "feature_id": fid,
            "alpha": alpha,
            "seq_len": pc["seq_len"],
            "target_id": int(tid),
            "delta_logit_target": delta_target,
            "tv_distance": tv,
            "kl_base_to_steer": kl,
            "topk_jaccard": jac,
            "max_abs_dlogit": max_abs,
            "hook_ran": bool(hook_ran),
            "base_rank": base_rank,
            "steer_rank": steer_rank,
            "target_is_top1": target_is_top1,
        })

        # Heartbeat every 500 tasks
        if (task_i + 1) % 500 == 0:
            elapsed = time.time() - start_time
            rate = (task_i + 1) / elapsed
            remaining = (total_tasks - task_i - 1) / rate if rate > 0 else 0
            print(f"[Heartbeat] {task_i+1}/{total_tasks}  "
                  f"{rate:.1f} tasks/s  ETA {remaining/60:.1f}min")

        # Periodic disk flush every 5 prompts worth of work
        if pi != last_flush_pi and pi % 5 == 0:
            last_flush_pi = pi
            pd.DataFrame(rows).to_csv(partial_path, index=False)

    df = pd.DataFrame(rows)

    # Write raw rows
    run_path = os.path.join(args.out_dir, "run_rows.csv")
    df.to_csv(run_path, index=False)

    # -----------------------------------------------------------
    # Aggregate per feature via summarize_feature_directional
    # -----------------------------------------------------------
    summary_rows = []
    for fid, df_feat in df.groupby("feature_id"):
        d = summarize_feature_directional(df_feat, tau=args.tau)

        # Non-zero alphas for off-target metrics (exclude baseline alpha=0)
        df_nonzero = df_feat[df_feat["alpha"] != 0.0]

        # Rank-based: at max positive alpha, what fraction of prompts
        # have target as top-1?
        max_pos_alpha = max([a for a in alphas if a > 0], default=None)
        if max_pos_alpha is not None:
            at_max = df_feat[df_feat["alpha"] == max_pos_alpha]
            top1_rate_up = float(at_max["target_is_top1"].mean()) if len(at_max) > 0 else np.nan
        else:
            top1_rate_up = np.nan

        # Mean rank improvement at max positive alpha
        if max_pos_alpha is not None and len(at_max) > 0:
            mean_rank_improve = float((at_max["base_rank"] - at_max["steer_rank"]).mean())
        else:
            mean_rank_improve = np.nan

        row = {
            "feature_id": fid,
            # Off-target metrics (non-zero alphas)
            "tv_mean": df_nonzero["tv_distance"].mean() if len(df_nonzero) > 0 else np.nan,
            "kl_mean": df_nonzero["kl_base_to_steer"].mean() if len(df_nonzero) > 0 else np.nan,
            "maxabs_mean": df_nonzero["max_abs_dlogit"].mean() if len(df_nonzero) > 0 else np.nan,
            "jacc_mean": df_nonzero["topk_jaccard"].mean() if len(df_nonzero) > 0 else np.nan,
            "target_delta_mean": df_feat["delta_logit_target"].mean(),
            # Rank-based stricter metrics
            "top1_rate_up": top1_rate_up,
            "mean_rank_improve": mean_rank_improve,
            # Directional success/alpha*/tv_at_crossing/monotone
            **d,
        }
        summary_rows.append(row)

    feat_summary = pd.DataFrame(summary_rows)
    alpha_ref = max(alphas, key=lambda a: abs(a))
    summary_path = os.path.join(args.out_dir, "feature_summary.csv")
    feat_summary.to_csv(summary_path, index=False)

    # Meta
    meta_out = vars(args)
    meta_out["n_rows"] = int(len(df))
    meta_out["alpha_ref"] = alpha_ref
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta_out, f, indent=2)

    print(f"\nWrote:\n- {run_path}\n- {summary_path}\n- {os.path.join(args.out_dir, 'meta.json')}")
    print(feat_summary.sort_values(["alpha_star_mean_up", "tv_mean"], ascending=[True, True]).head(15))


if __name__ == "__main__":
    main()
