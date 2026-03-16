# scripts/phase2_run.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os, json, argparse, random, time, subprocess
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from itertools import product
from tqdm.auto import tqdm

import wandb
from dotenv import load_dotenv

load_dotenv()

def _git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"

from src.config import load_config, resolve_config
from src.model_utils import load_model
from src.sae_loader import load_gemmascope_decoder
from src.refusal_scorer import refusal_score as _refusal_score

# --------------------------
# Utils
# --------------------------
def set_determinism(seed: int = 1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
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

@torch.no_grad()
def generate_steered(model, tokenizer, prompt: str, max_new_tokens: int = 64, prehook=None, use_chat_template: bool = False):
    """Generate a short response with optional steering hook. Returns generated text (excluding prompt)."""
    device = next(model.parameters()).device
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        chat = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(device)
    else:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
    handle = prehook() if prehook else None
    out_ids = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=1.0,
    )
    if handle is not None:
        handle.remove()
    gen_ids = out_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True)


def make_steer_prehook_all_pos(model, layer_idx: int, alpha: float, steer_dir: torch.Tensor):
    """Steering hook that adds alpha * steer_dir at ALL positions (needed for generation)."""
    ran = {"ok": False}
    def _mk():
        def _prehook(module, inputs):
            hidden = inputs[0]
            ran["ok"] = True
            hidden2 = hidden.clone()
            hidden2[:, :, :] = hidden2[:, :, :] + alpha * steer_dir
            return (hidden2,) + inputs[1:]
        layer = model.model.layers[layer_idx]
        return layer.register_forward_pre_hook(_prehook)
    return _mk, ran


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
    ap.add_argument("--mode", type=str, default="logit", choices=["logit", "refusal"],
                    help="logit: delta_logit_target (factual QA). refusal: generate + keyword refusal scoring (SALADBench).")
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
    dfp = pd.read_csv(prompt_csv, dtype={"prompt": "string"})
    assert "prompt" in dfp.columns, "prompt_csv must have a 'prompt' column"
    if args.mode == "logit":
        if "target" not in dfp.columns:
            dfp["target"] = np.nan
        dfp["_has_target"] = dfp["target"].notna() & (dfp["target"].astype(str).str.strip() != "")
        dfp = dfp[dfp["_has_target"]].drop(columns=["_has_target"]).reset_index(drop=True)
        if len(dfp) == 0:
            raise ValueError("No prompts with explicit target in prompt_csv. Logit mode requires a non-null target per prompt.")
    dfp = dfp.head(n_prompts).reset_index(drop=True)
    if len(dfp) == 0:
        raise ValueError("No usable prompts in prompt_csv.")
    # Initialize W&B (spec: sae-refusal-steering project)
    model_cfg = config.get("model", {})
    sae_cfg = config.get("sae", {})
    steering_cfg = config.get("steering", {})
    bench_cfg = config.get("benchmark", {})
    gen_cfg = config.get("generation", {})

    experiment_type = "full_run"
    if args.micro_sweep:
        experiment_type = "ablation_steering"

    wandb.init(
        project="sae-refusal-steering",
        name=f"phase2_run_{args.mode}",
        tags=[
            model_cfg.get("model_id", "unknown"),
            f"layer_{args.layer}",
            "activation_scaled",
            "contrast_delta" if args.fixed_features_path else "random",
            experiment_type,
        ],
        config={
            # Ablation metadata (Section 10)
            "experiment_type": experiment_type,
            "pipeline_version": "phase2_run_v2",
            "git_commit": _git_commit_hash(),
            "run_tag": f"phase2_run_{args.mode}",

            # Model configuration (Section 2)
            "model_name": model_cfg.get("model_id"),
            "model_size": model_cfg.get("model_id", "").split("-")[-1] if model_cfg.get("model_id") else None,
            "model_checkpoint": model_cfg.get("model_id"),
            "layer": args.layer,
            "hook_point": sae_cfg.get("hook_point"),
            "sae_width": sae_cfg.get("n_features_total"),
            "sae_repo": sae_cfg.get("weights_repo"),
            "sae_layer": sae_cfg.get("sae_id"),

            # Steering configuration (Section 2)
            "steering_method": "activation_scaled",
            "activation_scale_method": "fixed_alpha",
            "alpha_values": alphas,
            "alpha_max": max(alphas, key=abs),
            "steering_token_position": "last_token" if args.mode == "logit" else "all_tokens",

            # Feature selection configuration (Section 2)
            "feature_selection_method": "contrast_delta" if args.fixed_features_path else "random",
            "contrast_scoring_method": "delta_freq" if args.fixed_features_path else "none",
            "feature_pool_size": args.n_features,
            "selected_feature_count": args.n_features,
            "fixed_features_path": args.fixed_features_path,

            # Dataset / Prompt configuration (Section 2)
            "prompt_dataset": bench_cfg.get("name", "custom"),
            "prompt_count": n_prompts,
            "prompt_subset_strategy": "head",
            "generation_temperature": 0.0,
            "max_new_tokens": gen_cfg.get("max_new_tokens", 64),
            "chat_template": model_cfg.get("use_chat_template", True),

            # Evaluation configuration (Section 2)
            "refusal_scoring_method": "keyword" if args.mode == "refusal" else "logit_delta",
            "refusal_keywords_version": "v1",
            "threshold_T": threshold_T,

            # Legacy / extra
            "mode": args.mode,
            "seed": args.seed,
            "micro_sweep": args.micro_sweep,
            "prompt_csv": prompt_csv,
        },
    )

    config["model"]["do_sample"] = False
    config["model"]["temperature"] = 0.0
    model, tokenizer = load_model(config)
    model.eval()
    device = next(model.parameters()).device

    # Load SAE decoder (use returned tensor so NPZ key name is not assumed)
    W_dec, _ = load_gemmascope_decoder(config)
    W_dec = W_dec.to(device=device, dtype=torch.float32)
    if config.get("sae", {}).get("normalize_decoder", False):
        W_dec = W_dec / (W_dec.norm(dim=1, keepdim=True).clamp(min=1e-12))
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

    # One row per (prompt, feature, alpha): no duplicate feature_ids
    feature_ids = list(dict.fromkeys(feature_ids))

    # Freeze selected features for reproducibility (run start)
    selected_features_path = Path(args.out_dir) / "selected_features.json"
    with open(selected_features_path, "w") as f:
        json.dump({"feature_ids": feature_ids, "n_features": len(feature_ids), "seed": args.seed}, f, indent=2)
    print(f"Wrote {selected_features_path}")

    alphas_sorted = sorted(alphas, key=lambda a: abs(a))

    if args.mode == "logit":
        _run_logit_mode(model, tokenizer, dfp, W_dec, feature_ids, alphas, alphas_sorted, threshold_T, args, config)
    else:
        _run_refusal_mode(model, tokenizer, dfp, W_dec, feature_ids, alphas, alphas_sorted, threshold_T, args, config)


# ===========================================================
# Logit mode (factual QA: delta_logit_target)
# ===========================================================
def _run_logit_mode(model, tokenizer, dfp, W_dec, feature_ids, alphas, alphas_sorted, threshold_T, args, config):
    device = next(model.parameters()).device

    lm_head_w = model.lm_head.weight.detach().float()  # [vocab, d_model]

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

    all_jobs = list(dict.fromkeys(product(dfp.index, feature_ids, alphas_sorted)))
    total_tasks = len(all_jobs)
    run_csv = Path(args.out_dir) / "run_rows.csv"
    start_time = time.time()
    buffer = []

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
        steer_norm = float(steer_dir.norm().item())

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

        steer_t = torch.dot(steer_resid, pc["w_t"])
        delta_target = float((steer_t - pc["base_t"]).item())

        d = steer_logits - pc["base_logits"]
        max_abs = float(d.abs().max().item())
        tv = tv_distance_from_logits(pc["base_logits"], steer_logits)
        kl = kl_pq_from_logits(pc["base_logits"], steer_logits)
        jac = topk_jaccard(pc["base_logits"], steer_logits, k=args.topk)

        tid = pc["target_id"]
        base_rank = int((pc["base_logits"] >= pc["base_logits"][tid]).sum().item())
        steer_rank = int((steer_logits >= steer_logits[tid]).sum().item())
        target_is_top1 = bool(steer_rank == 1)

        row = {
            "prompt_idx": pi, "feature_id": fid, "alpha": alpha,
            "seq_len": pc["seq_len"], "target_id": int(tid),
            "delta_logit_target": delta_target,
            "tv_distance": tv, "kl_base_to_steer": kl,
            "topk_jaccard": jac, "max_abs_dlogit": max_abs,
            "hook_ran": bool(hook_ran),
            "base_rank": base_rank, "steer_rank": steer_rank,
            "target_is_top1": target_is_top1,
            "steering_vector_norm": steer_norm,
            "scaled_vector_norm": abs(alpha) * steer_norm,
        }
        buffer.append(row)
        completed_this_session += 1

        if len(buffer) >= args.flush_every:
            df_out = pd.DataFrame(buffer)
            header = not run_csv.exists()
            df_out.to_csv(run_csv, mode="a", header=header, index=False)
            buffer.clear()
            print(f"[Flush] wrote {args.flush_every} rows -> {run_csv}")

        done_tasks = len(done) + completed_this_session
        if done_tasks > 0 and done_tasks % args.heartbeat_every == 0:
            elapsed = time.time() - start_time
            rate = completed_this_session / max(1e-9, elapsed)
            eta = (total_tasks - done_tasks) / max(1e-9, rate)
            print(f"[Heartbeat] {done_tasks}/{total_tasks}  {rate:.1f} tasks/s  ETA {eta/60:.1f}min")
            wandb.log({
                "tasks_done": done_tasks,
                "tasks_total": total_tasks,
                "tasks_per_sec": rate,
                "eta_min": eta / 60,
                "delta_logit_target": delta_target,
                "tv_distance": tv,
                "kl_base_to_steer": kl,
            })

    if buffer:
        df_out = pd.DataFrame(buffer)
        header = not run_csv.exists()
        df_out.to_csv(run_csv, mode="a", header=header, index=False)
        print(f"[Flush] wrote {len(buffer)} rows -> {run_csv}")

    df = pd.read_csv(run_csv)
    key_cols = ["prompt_idx", "feature_id", "alpha"]
    dupes = df.duplicated(subset=key_cols, keep="first")
    if dupes.any():
        n_dup = int(dupes.sum())
        df = df.drop_duplicates(subset=key_cols, keep="first").reset_index(drop=True)
        print(f"[Fix] Dropped {n_dup} duplicate rows; {len(df)} rows remain.")

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

    summary_rows = []
    curve_rows = []
    for fid, df_feat in df.groupby("feature_id"):
        d = summarize_feature_directional(df_feat, T=threshold_T)

        df_nonzero = df_feat[df_feat["alpha"] != 0.0]
        max_pos_alpha = max([a for a in alphas if a > 0], default=None)
        if max_pos_alpha is not None:
            at_max = df_feat[df_feat["alpha"] == max_pos_alpha]
            top1_rate_up = float(at_max["target_is_top1"].mean()) if len(at_max) > 0 else np.nan
        else:
            top1_rate_up = np.nan

        if max_pos_alpha is not None and len(at_max) > 0:
            mean_rank_improve = float((at_max["base_rank"] - at_max["steer_rank"]).mean())
        else:
            mean_rank_improve = np.nan

        row = {
            "feature_id": fid,
            "tv_mean": df_nonzero["tv_distance"].mean() if len(df_nonzero) > 0 else np.nan,
            "kl_mean": df_nonzero["kl_base_to_steer"].mean() if len(df_nonzero) > 0 else np.nan,
            "maxabs_mean": df_nonzero["max_abs_dlogit"].mean() if len(df_nonzero) > 0 else np.nan,
            "jacc_mean": df_nonzero["topk_jaccard"].mean() if len(df_nonzero) > 0 else np.nan,
            "target_delta_mean": df_feat["delta_logit_target"].mean(),
            "top1_rate_up": top1_rate_up,
            "mean_rank_improve": mean_rank_improve,
            **d,
        }
        summary_rows.append(row)

        for alpha_val, grp in df_feat.groupby("alpha"):
            curve_rows.append({
                "feature_id": fid, "alpha": float(alpha_val),
                "mean_delta": float(grp["delta_logit_target"].mean()),
                "std_delta": float(grp["delta_logit_target"].std()) if len(grp) > 1 else 0.0,
                "n_prompts": int(len(grp)),
            })

    _write_outputs(summary_rows, curve_rows, alphas, threshold_T, args,
                   sort_cols=["alpha_star_feature_up", "tv_mean"],
                   alpha_star_cols=["feature_id", "alpha_star_feature_up", "alpha_star_feature_down", "censored_up", "censored_down"],
                   n_rows=len(df))


# ===========================================================
# Refusal mode (SALADBench: generate + keyword refusal scoring)
# ===========================================================
def _run_refusal_mode(model, tokenizer, dfp, W_dec, feature_ids, alphas, alphas_sorted, threshold_T, args, config):
    """
    For each (prompt, feature, alpha):
      1. Generate a short response with steering applied at all positions.
      2. Score the response for refusal (keyword-based).
    alpha* = smallest |alpha| where refusal_rate drops below baseline by T.
    """
    device = next(model.parameters()).device
    max_new_tokens = int(config.get("generation", {}).get("max_new_tokens", 64))

    use_chat = config.get("model", {}).get("use_chat_template", True)
    print(f"[Refusal mode] use_chat_template={use_chat}, computing baseline generations for {len(dfp)} prompts...")
    baseline_refusal = {}
    for pi, prow in tqdm(dfp.iterrows(), total=len(dfp), desc="Baseline gen"):
        text = generate_steered(model, tokenizer, prow["prompt"], max_new_tokens=max_new_tokens, use_chat_template=use_chat)
        baseline_refusal[pi] = {
            "prompt": prow["prompt"],
            "base_text": text,
            "base_refusal": _refusal_score(text),
        }

    all_jobs = list(dict.fromkeys(product(dfp.index, feature_ids, alphas_sorted)))
    total_tasks = len(all_jobs)
    run_csv = Path(args.out_dir) / "run_rows.csv"
    start_time = time.time()
    buffer = []

    done = set()
    if args.resume and run_csv.exists():
        prev = pd.read_csv(run_csv, usecols=["prompt_idx", "feature_id", "alpha"])
        done = set(map(tuple, prev.values))
        print(f"[Resume] loaded {len(done)} completed tasks from {run_csv}")

    print(f"Total tasks: {total_tasks} ({len(dfp)} prompts x {len(feature_ids)} features x {len(alphas_sorted)} alphas)")

    completed_this_session = 0
    for task_i, (pi, fid, alpha) in enumerate(tqdm(all_jobs, total=total_tasks, desc="Steering (refusal)")):
        if (pi, fid, alpha) in done:
            continue

        br = baseline_refusal[pi]
        steer_dir = W_dec[fid].to(device=device, dtype=torch.float32)
        steer_norm = float(steer_dir.norm().item())

        if alpha == 0.0:
            gen_text = br["base_text"]
            hook_ran = True
        else:
            mk_hook, cap = make_steer_prehook_all_pos(model, args.layer, alpha, steer_dir)
            gen_text = generate_steered(model, tokenizer, br["prompt"], max_new_tokens=max_new_tokens, prehook=mk_hook, use_chat_template=use_chat)
            hook_ran = cap["ok"]

        ref_score = _refusal_score(gen_text)
        delta_refusal = ref_score - br["base_refusal"]

        row = {
            "prompt_idx": pi, "feature_id": fid, "alpha": alpha,
            "refusal_score": ref_score,
            "base_refusal": br["base_refusal"],
            "delta_refusal": delta_refusal,
            "hook_ran": bool(hook_ran),
            "gen_text": gen_text[:500],
            "steering_vector_norm": steer_norm,
            "scaled_vector_norm": abs(alpha) * steer_norm,
        }
        buffer.append(row)
        completed_this_session += 1

        if len(buffer) >= args.flush_every:
            df_out = pd.DataFrame(buffer)
            header = not run_csv.exists()
            df_out.to_csv(run_csv, mode="a", header=header, index=False)
            buffer.clear()
            print(f"[Flush] wrote {args.flush_every} rows -> {run_csv}")

        done_tasks = len(done) + completed_this_session
        if done_tasks > 0 and done_tasks % args.heartbeat_every == 0:
            elapsed = time.time() - start_time
            rate = completed_this_session / max(1e-9, elapsed)
            eta = (total_tasks - done_tasks) / max(1e-9, rate)
            print(f"[Heartbeat] {done_tasks}/{total_tasks}  {rate:.1f} tasks/s  ETA {eta/60:.1f}min")
            wandb.log({
                "tasks_done": done_tasks,
                "tasks_total": total_tasks,
                "tasks_per_sec": rate,
                "eta_min": eta / 60,
                "refusal_score": ref_score,
                "delta_refusal": delta_refusal,
            })

    if buffer:
        df_out = pd.DataFrame(buffer)
        header = not run_csv.exists()
        df_out.to_csv(run_csv, mode="a", header=header, index=False)
        print(f"[Flush] wrote {len(buffer)} rows -> {run_csv}")

    df = pd.read_csv(run_csv)
    key_cols = ["prompt_idx", "feature_id", "alpha"]
    dupes = df.duplicated(subset=key_cols, keep="first")
    if dupes.any():
        n_dup = int(dupes.sum())
        df = df.drop_duplicates(subset=key_cols, keep="first").reset_index(drop=True)
        print(f"[Fix] Dropped {n_dup} duplicate rows; {len(df)} rows remain.")

    # Sanity: alpha=0 delta_refusal should be ~0
    df_alpha0 = df[df["alpha"] == 0.0]
    if len(df_alpha0) > 0:
        ok = (df_alpha0["delta_refusal"].abs() < 1e-6)
        frac_ok = float(ok.mean())
        print(f"[Sanity] alpha=0: {frac_ok:.1%} of {len(df_alpha0)} rows have |delta_refusal|<1e-6")

    # Aggregate per feature: alpha* is min |alpha| where mean refusal drops by >= T
    summary_rows = []
    curve_rows = []
    for fid, df_feat in df.groupby("feature_id"):
        prompts = sorted(df_feat["prompt_idx"].unique())
        pos_alphas = sorted([a for a in alphas if a > 0])
        neg_alphas = sorted([a for a in alphas if a < 0], key=lambda a: -a)

        # Baseline refusal rate for this feature's prompts
        base_rates = df_feat[df_feat["alpha"] == 0.0]["refusal_score"]
        base_refusal_rate = float(base_rates.mean()) if len(base_rates) > 0 else np.nan

        # alpha* up: find smallest positive alpha where mean refusal drops by >= T
        alpha_star_up = np.nan
        censored_up = True
        for a in pos_alphas:
            grp = df_feat[df_feat["alpha"] == a]
            if len(grp) == 0:
                continue
            mean_ref = float(grp["refusal_score"].mean())
            drop = base_refusal_rate - mean_ref
            if drop >= threshold_T:
                alpha_star_up = a
                censored_up = False
                break

        # alpha* down: find smallest (most negative) alpha where mean refusal drops by >= T
        alpha_star_down = np.nan
        censored_down = True
        for a in neg_alphas:
            grp = df_feat[df_feat["alpha"] == a]
            if len(grp) == 0:
                continue
            mean_ref = float(grp["refusal_score"].mean())
            drop = base_refusal_rate - mean_ref
            if drop >= threshold_T:
                alpha_star_down = a
                censored_down = False
                break

        # Refusal rate at max positive alpha
        max_pos = max(pos_alphas) if pos_alphas else None
        if max_pos is not None:
            at_max = df_feat[df_feat["alpha"] == max_pos]
            refusal_rate_at_max = float(at_max["refusal_score"].mean()) if len(at_max) > 0 else np.nan
        else:
            refusal_rate_at_max = np.nan

        # Refusal rates at specific alphas (Section 4)
        refusal_at_alpha = {}
        for a_probe in [1, 5, 20]:
            grp_probe = df_feat[df_feat["alpha"] == float(a_probe)]
            refusal_at_alpha[a_probe] = float(grp_probe["refusal_score"].mean()) if len(grp_probe) > 0 else np.nan

        # Refusal drop max (Section 3): max(baseline - min_refusal_across_alphas)
        all_alpha_means = df_feat.groupby("alpha")["refusal_score"].mean()
        min_refusal = float(all_alpha_means.min()) if len(all_alpha_means) > 0 else np.nan
        refusal_drop_max = base_refusal_rate - min_refusal if not np.isnan(min_refusal) else np.nan

        # Best alpha = the alpha with lowest mean refusal
        if len(all_alpha_means) > 0:
            best_alpha_val = float(all_alpha_means.idxmin())
            best_refusal_val = float(all_alpha_means.min())
        else:
            best_alpha_val = np.nan
            best_refusal_val = np.nan

        # Threshold-independent effect size: min refusal rate and its alpha
        # (same values as best_*, but named explicitly for clarity)
        min_refusal_rate = best_refusal_val
        alpha_at_min_refusal = best_alpha_val

        # Steering diagnostics (Section 7)
        steer_dir_feat = W_dec[fid].to(device=device, dtype=torch.float32)
        feat_decoder_norm = float(steer_dir_feat.norm().item())

        # Determine steering direction
        if not censored_up and not censored_down:
            steering_dir = "up" if alpha_star_up <= abs(alpha_star_down) else "down"
        elif not censored_up:
            steering_dir = "up"
        elif not censored_down:
            steering_dir = "down"
        else:
            steering_dir = "none"

        row = {
            "feature_id": fid,
            "base_refusal_rate": base_refusal_rate,
            "refusal_rate_at_max_alpha": refusal_rate_at_max,
            "refusal_drop_at_max_alpha": base_refusal_rate - refusal_rate_at_max if not np.isnan(refusal_rate_at_max) else np.nan,
            "refusal_rate_alpha_1": refusal_at_alpha.get(1, np.nan),
            "refusal_rate_alpha_5": refusal_at_alpha.get(5, np.nan),
            "refusal_rate_alpha_20": refusal_at_alpha.get(20, np.nan),
            "refusal_drop_max": refusal_drop_max,
            "refusal_drop_alpha_5": base_refusal_rate - refusal_at_alpha.get(5, np.nan) if not np.isnan(refusal_at_alpha.get(5, np.nan)) else np.nan,
            "best_alpha": best_alpha_val,
            "best_refusal_rate": best_refusal_val,
            "min_refusal_rate": min_refusal_rate,
            "alpha_at_min_refusal": alpha_at_min_refusal,
            "alpha_star_feature_up": alpha_star_up,
            "alpha_star_feature_down": alpha_star_down,
            "censored_up": censored_up,
            "censored_down": censored_down,
            "steering_direction": steering_dir,
            # Geometry / diagnostics (Sections 5, 7)
            "decoder_vector_norm": feat_decoder_norm,
            "feature_l2_norm": feat_decoder_norm,
            "steering_vector_norm": feat_decoder_norm,
        }
        summary_rows.append(row)

        for alpha_val, grp in df_feat.groupby("alpha"):
            curve_rows.append({
                "feature_id": fid, "alpha": float(alpha_val),
                "mean_refusal": float(grp["refusal_score"].mean()),
                "std_refusal": float(grp["refusal_score"].std()) if len(grp) > 1 else 0.0,
                "n_prompts": int(len(grp)),
            })

    _write_outputs(summary_rows, curve_rows, alphas, threshold_T, args,
                   sort_cols=["alpha_star_feature_up", "base_refusal_rate"],
                   alpha_star_cols=["feature_id", "alpha_star_feature_up", "alpha_star_feature_down", "censored_up", "censored_down"],
                   n_rows=len(df))


# ===========================================================
# Shared output writer
# ===========================================================
def _write_outputs(summary_rows, curve_rows, alphas, threshold_T, args, sort_cols, alpha_star_cols, n_rows):
    selected_features_path = Path(args.out_dir) / "selected_features.json"
    run_csv = Path(args.out_dir) / "run_rows.csv"

    feat_summary = pd.DataFrame(summary_rows)
    alpha_ref = max(alphas, key=lambda a: abs(a))
    summary_path = os.path.join(args.out_dir, "feature_summary.csv")
    feat_summary.to_csv(summary_path, index=False)

    alpha_star_path = os.path.join(args.out_dir, "alpha_star.csv")
    available_cols = [c for c in alpha_star_cols if c in feat_summary.columns]
    alpha_star_df = feat_summary[available_cols].copy()
    alpha_star_df.to_csv(alpha_star_path, index=False)

    curves_df = pd.DataFrame(curve_rows)
    curves_path = os.path.join(args.out_dir, "curves_per_feature.parquet")
    try:
        curves_df.to_parquet(curves_path, index=False)
    except ImportError:
        curves_path = os.path.join(args.out_dir, "curves_per_feature.csv")
        curves_df.to_csv(curves_path, index=False)
        print(f"[Note] Wrote {curves_path} (parquet engine not found; install pyarrow for .parquet)")

    meta_out = vars(args).copy()
    meta_out["n_rows"] = int(n_rows)
    meta_out["alpha_ref"] = alpha_ref
    meta_out["threshold_T"] = threshold_T
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta_out, f, indent=2)

    print(f"\nWrote:\n- {run_csv}\n- {summary_path}\n- {alpha_star_path}\n- {curves_path}\n- {selected_features_path}\n- {os.path.join(args.out_dir, 'meta.json')}")
    existing_sort = [c for c in sort_cols if c in feat_summary.columns]
    if existing_sort:
        print(feat_summary.sort_values(existing_sort, ascending=True).head(15))
    else:
        print(feat_summary.head(15))

    # ==================================================================
    # W&B Logging (Sections 3–12 of instrumentation spec)
    # ==================================================================

    # --- Section 3: Run-Level Metrics ---
    run_metrics = {"n_features": len(feat_summary), "n_rows": n_rows}

    if "base_refusal_rate" in feat_summary.columns:
        run_metrics["baseline_refusal_rate"] = float(feat_summary["base_refusal_rate"].mean())

    for a_key, col in [(1, "refusal_rate_alpha_1"), (5, "refusal_rate_alpha_5"), (20, "refusal_rate_alpha_20")]:
        if col in feat_summary.columns:
            vals = feat_summary[col].dropna()
            if len(vals) > 0:
                run_metrics[f"mean_refusal_alpha_{a_key}"] = float(vals.mean())

    if "refusal_drop_max" in feat_summary.columns:
        drops = feat_summary["refusal_drop_max"].dropna()
        if len(drops) > 0:
            run_metrics["max_refusal_drop"] = float(drops.max())
            run_metrics["mean_refusal_drop"] = float(drops.mean())
            run_metrics["median_refusal_drop"] = float(drops.median())

    if "alpha_star_feature_up" in feat_summary.columns:
        uncensored_up = feat_summary[~feat_summary.get("censored_up", pd.Series(True)).astype(bool)]
        n_uncensored_up = len(uncensored_up)
        run_metrics["frac_uncensored_up"] = n_uncensored_up / max(1, len(feat_summary))
        if n_uncensored_up > 0:
            vals = uncensored_up["alpha_star_feature_up"].dropna()
            run_metrics["mean_alpha_star_up"] = float(vals.mean())
            run_metrics["median_alpha_star_up"] = float(vals.median())
            run_metrics["std_alpha_star_up"] = float(vals.std())

    if "alpha_star_feature_down" in feat_summary.columns:
        uncensored_down = feat_summary[~feat_summary.get("censored_down", pd.Series(True)).astype(bool)]
        run_metrics["frac_uncensored_down"] = len(uncensored_down) / max(1, len(feat_summary))
        if len(uncensored_down) > 0:
            vals_dn = uncensored_down["alpha_star_feature_down"].dropna()
            if len(vals_dn) > 0:
                run_metrics["mean_alpha_star_down"] = float(vals_dn.mean())
                run_metrics["median_alpha_star_down"] = float(vals_dn.median())
                run_metrics["std_alpha_star_down"] = float(vals_dn.std())

    if "best_alpha" in feat_summary.columns:
        run_metrics["best_alpha"] = float(feat_summary["best_alpha"].dropna().mode().iloc[0]) if len(feat_summary["best_alpha"].dropna()) > 0 else np.nan
    if "best_refusal_rate" in feat_summary.columns:
        run_metrics["best_refusal_rate"] = float(feat_summary["best_refusal_rate"].dropna().min()) if len(feat_summary["best_refusal_rate"].dropna()) > 0 else np.nan

    if "min_refusal_rate" in feat_summary.columns:
        mr = feat_summary["min_refusal_rate"].dropna()
        if len(mr) > 0:
            run_metrics["mean_min_refusal_rate"] = float(mr.mean())
            run_metrics["median_min_refusal_rate"] = float(mr.median())

    wandb.log(run_metrics)

    # --- Section 4 + 5 + 6 + 7: feature_metrics Table ---
    fm_cols = [c for c in [
        "feature_id", "alpha_star_feature_up", "alpha_star_feature_down",
        "censored_up", "censored_down", "steering_direction",
        "base_refusal_rate",
        "refusal_rate_alpha_1", "refusal_rate_alpha_5", "refusal_rate_alpha_20",
        "refusal_drop_max", "refusal_drop_alpha_5",
        "best_alpha", "best_refusal_rate",
        "min_refusal_rate", "alpha_at_min_refusal",
        # Geometry (Section 5)
        "decoder_vector_norm", "feature_l2_norm",
        # Steering diagnostics (Section 7)
        "steering_vector_norm",
        # Logit-mode extras
        "tv_mean", "kl_mean", "maxabs_mean", "jacc_mean",
        "target_delta_mean", "top1_rate_up", "mean_rank_improve",
        "success_rate_up", "success_rate_down",
    ] if c in feat_summary.columns]

    feature_metrics_table = wandb.Table(dataframe=feat_summary[fm_cols])
    wandb.log({"feature_metrics": feature_metrics_table})

    # --- Section 8: feature_curves Table ---
    curves_df = pd.DataFrame(curve_rows)
    curve_table_cols = [c for c in [
        "feature_id", "alpha",
        "mean_refusal", "std_refusal",
        "mean_delta", "std_delta",
        "n_prompts",
    ] if c in curves_df.columns]
    feature_curves_table = wandb.Table(dataframe=curves_df[curve_table_cols])
    wandb.log({"feature_curves": feature_curves_table})

    # --- Section 9: prompt_results Table ---
    # Read full run_rows and log as table (truncated to avoid OOM)
    full_df = pd.read_csv(run_csv)
    pr_cols = [c for c in [
        "prompt_idx", "feature_id", "alpha",
        "base_refusal", "refusal_score", "delta_refusal",
        "gen_text",
        "delta_logit_target", "tv_distance", "kl_base_to_steer",
        "hook_ran",
        "steering_vector_norm", "scaled_vector_norm",
    ] if c in full_df.columns]
    # Cap at 10k rows for the table to avoid W&B limits
    prompt_results_table = wandb.Table(dataframe=full_df[pr_cols].head(10000))
    wandb.log({"prompt_results": prompt_results_table})

    # --- Section 12: Artifacts ---
    artifact = wandb.Artifact(
        name=f"phase2_run_{args.mode}",
        type="run_outputs",
        metadata={"n_rows": n_rows, "threshold_T": threshold_T},
    )
    for fpath in [run_csv, summary_path, alpha_star_path, curves_path,
                  selected_features_path, os.path.join(args.out_dir, "meta.json")]:
        fpath_str = str(fpath)
        if os.path.exists(fpath_str):
            artifact.add_file(fpath_str)
    wandb.log_artifact(artifact)

    wandb.finish()


if __name__ == "__main__":
    main()
