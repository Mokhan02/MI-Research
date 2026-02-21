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

def find_alpha_star_feature_level(df_feat: pd.DataFrame, T: float, direction: int = +1):
    """
    Feature-level alpha*: smallest |alpha| where mean(delta_logit_target) crosses threshold.
    direction = +1 (up) uses alpha>0 and delta>=T
    direction = -1 (down) uses alpha<0 and delta<=-T
    Returns dict with alpha_star (magnitude), censored, curve table.
    """
    assert direction in (+1, -1)

    df = df_feat.copy()
    df = df[df["alpha"] != 0]  # alpha=0 is baseline, mean delta should be ~0

    if direction == +1:
        df = df[df["alpha"] > 0]
        curve = df.groupby("alpha")["delta_logit_target"].mean()
        curve = curve.reindex(sorted(curve.index, key=lambda a: abs(a)))
        crossed = curve[curve >= T]
    else:
        df = df[df["alpha"] < 0]
        curve = df.groupby("alpha")["delta_logit_target"].mean()
        curve = curve.reindex(sorted(curve.index, key=lambda a: abs(a)))
        crossed = curve[curve <= -T]

    if len(curve) == 0:
        return {"alpha_star": np.nan, "censored": True, "curve": curve}

    if len(crossed) == 0:
        amax = float(np.max(np.abs(curve.index.to_numpy())))
        return {"alpha_star": amax, "censored": True, "curve": curve}

    a_star = float(np.abs(crossed.index[0]))
    return {"alpha_star": a_star, "censored": False, "curve": curve}


def find_alpha_star_per_prompt(df_direction: pd.DataFrame, prompts, threshold_fn, alphas_ordered):
    """
    Diagnostic: per-prompt alpha* (smallest |alpha| where threshold_fn(delta) is True).
    Returns dict prompt_idx -> alpha_star (magnitude only, no TV).
    """
    keyed = df_direction.set_index(["prompt_idx", "alpha"])
    hits = {}
    for pidx in prompts:
        for a in alphas_ordered:
            if (pidx, a) not in keyed.index:
                continue
            row = keyed.loc[(pidx, a)]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            if threshold_fn(row["delta_logit_target"]):
                hits[pidx] = float(np.abs(a))
                break
    return hits


def monotone_fraction(df_direction, prompts, alphas_ordered, sign=1):
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

def summarize_feature_directional(df_feat: pd.DataFrame, T: float):
    """
    df_feat: run_rows filtered to a single feature_id (all prompts, all alphas).

    Success metric: delta_logit_target (steer_t - base_t) >= +T (up) or <= -T (down).
    T must be pre-registered (set from baseline or pilot, then frozen). Aggregation:
    feature-level mean(delta) then threshold (alpha* = smallest |alpha| where mean(delta) >= T).
    Directionality: alpha* is "helpful" (toward target); track wrong-way effects as off-target.

    No-effect (censored): When mean(delta) never crosses threshold, censored_up/down = True.
    Do NOT set alpha* = max(alpha); treat as no-effect in correlations (see LANDMINES.md).
    """
    prompts = df_feat["prompt_idx"].unique()

    pos = df_feat[df_feat["alpha"] > 0].copy()
    neg = df_feat[df_feat["alpha"] < 0].copy()

    pos_alphas = sorted(pos["alpha"].unique(), key=lambda a: abs(a))
    neg_alphas = sorted(neg["alpha"].unique(), key=lambda a: abs(a))

    out = dict(
        success_rate_up=np.nan,
        success_rate_down=np.nan,

        # OFFICIAL: feature-level alpha*
        alpha_star_feature_up=np.nan,
        alpha_star_feature_down=np.nan,
        censored_up=True,
        censored_down=True,

        # DIAGNOSTIC: per-prompt alpha* mean
        alpha_star_prompt_mean_up=np.nan,
        alpha_star_prompt_mean_down=np.nan,

        monotone_frac_up=np.nan,
        monotone_frac_down=np.nan,
    )

    if len(pos) > 0:
        mx = pos.groupby("prompt_idx")["delta_logit_target"].max()
        out["success_rate_up"] = float((mx >= T).mean())

        feat_up = find_alpha_star_feature_level(df_feat, T=T, direction=+1)
        out["alpha_star_feature_up"] = feat_up["alpha_star"]
        out["censored_up"] = feat_up["censored"]

        hits_up = find_alpha_star_per_prompt(
            pos, prompts, threshold_fn=lambda d: d >= T, alphas_ordered=pos_alphas
        )
        if hits_up:
            out["alpha_star_prompt_mean_up"] = float(np.nanmean(list(hits_up.values())))

        out["monotone_frac_up"] = monotone_fraction(pos, prompts, pos_alphas, sign=+1)

    if len(neg) > 0:
        mn = neg.groupby("prompt_idx")["delta_logit_target"].min()
        out["success_rate_down"] = float((mn <= -T).mean())

        feat_down = find_alpha_star_feature_level(df_feat, T=T, direction=-1)
        out["alpha_star_feature_down"] = feat_down["alpha_star"]
        out["censored_down"] = feat_down["censored"]

        hits_down = find_alpha_star_per_prompt(
            neg, prompts, threshold_fn=lambda d: d <= -T, alphas_ordered=neg_alphas
        )
        if hits_down:
            out["alpha_star_prompt_mean_down"] = float(np.nanmean(list(hits_down.values())))

        out["monotone_frac_down"] = monotone_fraction(neg, prompts, neg_alphas, sign=-1)

    return out

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
    ap.add_argument("--prompt_csv", type=str, default=None, help="Override benchmark.prompt_csv from config")
    ap.add_argument("--out_dir", type=str, default="outputs/phase2")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--n_prompts", type=int, default=None, help="Override benchmark.prompt_count from config")
    ap.add_argument("--n_features", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--alphas", type=str, default=None,
                        help="Override steering.alpha_grid from config (comma-separated, e.g. -10,-5,-1,0,1,5,10)")
    ap.add_argument("--fixed_features_path", type=str, default=None,
                        help="Path to selected_features.json (frozen feature set). If set, load feature_ids and skip sampling.")
    ap.add_argument("--feature_ids_file", type=str, default=None,
                        help="Path to txt with one feature_id per line. If set, overrides random sampling (ignored if --fixed_features_path set).")
    ap.add_argument("--heartbeat_every", type=int, default=500)
    ap.add_argument("--flush_every", type=int, default=2000)
    ap.add_argument("--resume", action="store_true", help="Skip tasks already in run_rows.csv")
    ap.add_argument("--micro_sweep", action="store_true",
                    help="Micro sweep: 10 features, 25 prompts, alpha=[0,.5,1,2,5]. Run this before full K=100.")
    args = ap.parse_args()

    if args.micro_sweep:
        args.n_features = 10
        args.n_prompts = 25
        args.alphas = "0,0.5,1,2,5"
        print("[Micro sweep] n_features=10, n_prompts=25, alphas=[0,.5,1,2,5]")

    set_determinism(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # Load config first (needed for defaults and SAE assertion)
    config = resolve_config(load_config(args.config), run_id="phase2_run")
    sae_id = config.get("sae", {}).get("sae_id")
    if sae_id in ("TBD", "", None):
        raise ValueError(
            "config sae.sae_id must be set to a concrete identifier (e.g. 20-gemmascope-res-16k/2263). "
            f"Got: {sae_id!r}. No mercy: every run must be reproducible."
        )

    # CLI overrides config; config must be sufficient to run
    bench = config.get("benchmark", {})
    prompt_csv = args.prompt_csv or bench.get("prompt_csv")
    if not prompt_csv:
        raise ValueError("Either pass --prompt_csv or set benchmark.prompt_csv in config")
    n_prompts = args.n_prompts if args.n_prompts is not None else bench.get("prompt_count")
    if n_prompts is None:
        n_prompts = 200
    n_prompts = int(n_prompts)

    # Steering threshold and alpha grid (CLI overrides config)
    threshold_T = float(config["steering"]["threshold_T"])
    if args.alphas is not None:
        alphas = [float(x) for x in args.alphas.split(",")]
    else:
        # Use config steering.alpha_grid; extend with negatives for directional (up/down) runs
        grid = list(config["steering"]["alpha_grid"])
        positive = [x for x in grid if x > 0]
        alphas = sorted(set([0.0] + positive + [-x for x in positive]), key=lambda a: abs(a))

    # Load prompts
    dfp = pd.read_csv(prompt_csv, dtype={"prompt": "string", "target": "string"})
    assert "prompt" in dfp.columns, "prompt_csv must have a 'prompt' column"
    if "target" not in dfp.columns:
        dfp["target"] = np.nan
    # Official: every prompt must have an explicit target; drop prompts without one
    dfp["_has_target"] = dfp["target"].notna() & (dfp["target"].astype(str).str.strip() != "")
    dfp = dfp[dfp["_has_target"]].drop(columns=["_has_target"]).reset_index(drop=True)
    dfp = dfp.head(n_prompts).reset_index(drop=True)
    if len(dfp) == 0:
        raise ValueError("No prompts with explicit target in prompt_csv. Official runs require a non-null target per prompt.")
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

    # Choose features (fixed set > txt file > random sampling)
    n_feats_total = W_dec.shape[0]
    if args.fixed_features_path:
        with open(args.fixed_features_path) as f:
            data = json.load(f)
        feature_ids = list(data["feature_ids"])
        print(f"Loaded {len(feature_ids)} feature IDs from {args.fixed_features_path} (skip sampling)")
    elif args.feature_ids_file:
        with open(args.feature_ids_file) as f:
            feature_ids = [int(x.strip()) for x in f if x.strip()]
        feature_ids = feature_ids[:args.n_features]
        print(f"Loaded {len(feature_ids)} feature IDs from {args.feature_ids_file}")
    else:
        rng = np.random.default_rng(args.seed)
        feature_ids = rng.choice(n_feats_total, size=min(args.n_features, n_feats_total), replace=False).tolist()

    # Freeze selected features for reproducibility (run start)
    selected_features_path = Path(args.out_dir) / "selected_features.json"
    with open(selected_features_path, "w") as f:
        json.dump({"feature_ids": feature_ids, "n_features": len(feature_ids), "seed": args.seed}, f, indent=2)
    print(f"Wrote {selected_features_path}")

    alphas_sorted = sorted(alphas, key=lambda a: abs(a))

    # lm_head weight for fp32 target logit computation
    lm_head_w = model.lm_head.weight.detach().float()  # [vocab, d_model]

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
    run_csv = Path(args.out_dir) / "run_rows.csv"
    start_time = time.time()
    buffer = []

    # Resume: skip tasks already in run_rows.csv
    done = set()
    if args.resume and run_csv.exists():
        prev = pd.read_csv(run_csv, usecols=["prompt_idx", "feature_id", "alpha"])
        done = set(map(tuple, prev.values))
        print(f"[Resume] loaded {len(done)} completed tasks from {run_csv}")

    print(f"Total tasks: {total_tasks} ({len(dfp)} prompts x {len(feature_ids)} features x {len(alphas_sorted)} alphas)")

    completed_this_session = 0
    for task_i, (pi, fid, alpha) in enumerate(tqdm(all_jobs, total=total_tasks, desc="Steering")):
        if (pi, fid, alpha) in done:
            continue

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

        row = {
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
        }
        buffer.append(row)
        completed_this_session += 1

        # Flush buffer to disk
        if len(buffer) >= args.flush_every:
            df_out = pd.DataFrame(buffer)
            header = not run_csv.exists()
            df_out.to_csv(run_csv, mode="a", header=header, index=False)
            buffer.clear()
            print(f"[Flush] wrote {args.flush_every} rows -> {run_csv}")

        # Heartbeat
        done_tasks = len(done) + completed_this_session
        if done_tasks > 0 and done_tasks % args.heartbeat_every == 0:
            elapsed = time.time() - start_time
            rate = completed_this_session / max(1e-9, elapsed)
            eta = (total_tasks - done_tasks) / max(1e-9, rate)
            print(f"[Heartbeat] {done_tasks}/{total_tasks}  {rate:.1f} tasks/s  ETA {eta/60:.1f}min")

    # Flush remaining buffer
    if buffer:
        df_out = pd.DataFrame(buffer)
        header = not run_csv.exists()
        df_out.to_csv(run_csv, mode="a", header=header, index=False)
        print(f"[Flush] wrote {len(buffer)} rows -> {run_csv}")

    df = pd.read_csv(run_csv)

    # Sanity gate: alpha=0 must yield ~0 deltas (steer_t - base_t at alpha=0)
    df_alpha0 = df[df["alpha"] == 0.0]
    if len(df_alpha0) > 0:
        ok = (df_alpha0["delta_logit_target"].abs() < 1e-4)
        frac_ok = float(ok.mean())
        if frac_ok < 0.95:
            raise RuntimeError(
                f"Alpha=0 sanity failed: only {frac_ok:.1%} of alpha=0 rows have |delta|<1e-4. "
                "Check hook application or computation path."
            )
        print(f"[Sanity] alpha=0: {frac_ok:.1%} of {len(df_alpha0)} rows have |delta|<1e-4")

    # -----------------------------------------------------------
    # Aggregate per feature via summarize_feature_directional
    # -----------------------------------------------------------
    summary_rows = []
    curve_rows = []  # for curves_per_feature.parquet
    for fid, df_feat in df.groupby("feature_id"):
        d = summarize_feature_directional(df_feat, T=threshold_T)

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

        # Curves per feature: (feature_id, alpha, mean_delta, std_delta, n_prompts)
        for alpha_val, grp in df_feat.groupby("alpha"):
            curve_rows.append({
                "feature_id": fid,
                "alpha": float(alpha_val),
                "mean_delta": float(grp["delta_logit_target"].mean()),
                "std_delta": float(grp["delta_logit_target"].std()) if len(grp) > 1 else 0.0,
                "n_prompts": int(len(grp)),
            })

    feat_summary = pd.DataFrame(summary_rows)
    alpha_ref = max(alphas, key=lambda a: abs(a))
    summary_path = os.path.join(args.out_dir, "feature_summary.csv")
    feat_summary.to_csv(summary_path, index=False)

    # alpha_star.csv (official feature-level alpha* at run end)
    alpha_star_path = os.path.join(args.out_dir, "alpha_star.csv")
    alpha_star_df = feat_summary[["feature_id", "alpha_star_feature_up", "alpha_star_feature_down", "censored_up", "censored_down"]].copy()
    alpha_star_df.to_csv(alpha_star_path, index=False)

    # curves_per_feature.parquet
    curves_path = os.path.join(args.out_dir, "curves_per_feature.parquet")
    pd.DataFrame(curve_rows).to_parquet(curves_path, index=False)

    # Meta (include threshold_T for reproducibility)
    meta_out = vars(args).copy()
    meta_out["n_rows"] = int(len(df))
    meta_out["alpha_ref"] = alpha_ref
    meta_out["threshold_T"] = threshold_T
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta_out, f, indent=2)

    print(f"\nWrote:\n- {run_csv}\n- {summary_path}\n- {alpha_star_path}\n- {curves_path}\n- {selected_features_path}\n- {os.path.join(args.out_dir, 'meta.json')}")
    print(feat_summary.sort_values(["alpha_star_feature_up", "tv_mean"], ascending=[True, True]).head(15))


if __name__ == "__main__":
    main()
