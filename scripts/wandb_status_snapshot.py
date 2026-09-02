#!/usr/bin/env python3
"""Project-close status readout from W&B run summaries.

Two modes:

    # 1. Pull every run summary from the 16 `coupled-muon-*` projects into a
    #    parquet snapshot (needs WANDB_API_KEY or ~/.netrc; ~4 min).
    python scripts/wandb_status_snapshot.py --fetch docs/data/wandb_all_runs_<date>.parquet

    # 2. Print the per-plan-step readout from an existing snapshot (no network).
    python scripts/wandb_status_snapshot.py --analyze docs/data/wandb_all_runs_2026-09-02.parquet

The readout reproduces every number in `docs/FINAL_STATUS.md`: per-project
completion, own-best-LR paired Coupled-vs-Muon deltas with 95% percentile
bootstrap CIs on the median seed-wise difference, the Phase-2.1 B1/B2
qk-policy tables (with and without the probe-based `diverged` flag), the D1
MLA readout and D-gate, K-curve, NS-policy and runtime ratios.

Conventions (mirrors docs/figures/_stats.py and part2_analysis.py):
  * dedupe on the full config axis set, keep the latest row with a final loss;
  * "valid" = finished, finite final_val_loss, seed parsed, not `diverged`;
  * own-best LR per (rung, optimizer) = argmin over LR of the median across
    valid seeds; arms are then paired by seed at each arm's own best LR.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import time

import numpy as np
import pandas as pd

ENT = "liuyc1025-university-of-cambridge"
SEED_RE = re.compile(r"[·_\-.]s(\d+)(?:\b|$)")

NUMERIC = [
    "lr",
    "seed",
    "final_val_loss",
    "final_train_loss",
    "final_step",
    "final_tokens",
    "runtime",
    "wall_clock_s",
    "params_total",
    "params_active",
    "val_loss",
    "tokens",
    "coupled_steps",
    "ns_steps",
    "kv_lora_rank",
    "q_lora_rank",
    "factorize_rank",
    "num_experts",
    "top_k",
    "hidden",
    "n_layers",
    "total_tokens",
    "rope_partial_frac",
]
AXES = [
    "project",
    "L",
    "opt",
    "lr",
    "seed",
    "coupled_steps",
    "ns_steps",
    "ns_coefficients",
    "lr_prefactor",
    "final_polish",
    "couple_qk",
    "couple_vo",
    "couple_updown",
    "pair_policy",
    "use_multi_head",
    "qk_no_rope_policy",
    "qk_partial_rope_policy",
    "couple_router_to_muon",
    "kv_lora_rank",
    "q_lora_rank",
    "attn_type",
]


# ----------------------------------------------------------------------------- fetch
def fetch(out_path: str) -> None:
    import wandb

    api = wandb.Api(timeout=120)
    projects = sorted(p.name for p in api.projects(ENT) if p.name.startswith("coupled-muon"))
    rows = []
    for pname in projects:
        t0 = time.time()
        runs = list(api.runs(f"{ENT}/{pname}", per_page=500))
        for r in runs:
            cfg = r.config or {}
            sm = dict(r.summary)
            opt = cfg.get("optimizer", {}) or {}
            model = cfg.get("model", {}) or {}
            attn = model.get("attn", {}) or {}
            moe = model.get("moe", {}) or {}
            qk = opt.get("qk_coupling", {}) or {}
            m = SEED_RE.search(r.name or "")
            seed = cfg.get("seed")
            if seed is None and m:
                seed = int(m.group(1))
            rows.append(
                dict(
                    project=pname,
                    run_id=r.id,
                    name=r.name,
                    state=r.state,
                    created_at=str(r.created_at),
                    group=r.group,
                    job_type=r.job_type,
                    tags="|".join(r.tags or []),
                    rung=cfg.get("name"),
                    opt=opt.get("type"),
                    lr=opt.get("lr"),
                    seed=seed,
                    coupled_steps=opt.get("coupled_steps"),
                    ns_steps=opt.get("ns_steps"),
                    ns_coefficients=opt.get("ns_coefficients"),
                    lr_prefactor=opt.get("lr_prefactor"),
                    final_polish=opt.get("final_polish"),
                    couple_qk=opt.get("couple_qk"),
                    couple_vo=opt.get("couple_vo"),
                    couple_updown=opt.get("couple_updown"),
                    pair_policy=opt.get("pair_policy"),
                    use_multi_head=opt.get("use_multi_head"),
                    qk_full_rope=qk.get("full_rope"),
                    qk_no_rope_policy=qk.get("no_rope_policy"),
                    qk_partial_rope_policy=qk.get("partial_rope_policy"),
                    couple_router_to_muon=opt.get("couple_router_to_muon"),
                    pos_emb=(model.get("pos_emb", {}) or {}).get("type"),
                    rope_partial_frac=attn.get("rope_partial_frac"),
                    qk_norm=attn.get("qk_norm"),
                    attn_type=attn.get("attn_type"),
                    kv_lora_rank=attn.get("kv_lora_rank"),
                    q_lora_rank=attn.get("q_lora_rank"),
                    mlp_type=(model.get("mlp", {}) or {}).get("type"),
                    factorize_rank=(model.get("mlp", {}) or {}).get("factorize_rank"),
                    norm_type=(model.get("norm", {}) or {}).get("type"),
                    moe_enabled=moe.get("enabled"),
                    num_experts=moe.get("num_experts"),
                    top_k=moe.get("top_k"),
                    balancing_type=moe.get("balancing_type"),
                    shared_expert=moe.get("shared_expert"),
                    hidden=model.get("hidden"),
                    n_layers=model.get("n_layers"),
                    total_tokens=(cfg.get("train", {}) or {}).get("total_tokens"),
                    final_val_loss=sm.get("final_val_loss"),
                    final_train_loss=sm.get("final_train_loss"),
                    diverged=sm.get("diverged"),
                    final_step=sm.get("final_step"),
                    final_tokens=sm.get("final_tokens"),
                    runtime=sm.get("_runtime"),
                    wall_clock_s=sm.get("wall_clock_s"),
                    params_total=sm.get("params_total"),
                    params_active=sm.get("params_active"),
                    val_loss=sm.get("val_loss"),
                    tokens=sm.get("tokens"),
                )
            )
        print(f"{pname}: {len(runs)} runs ({time.time() - t0:.1f}s)", file=sys.stderr)
    df = pd.DataFrame(rows)
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(lambda v: None if v is None else str(v))
    df.to_parquet(out_path, index=False)
    print(f"wrote {out_path}: {df.shape}", file=sys.stderr)


# ----------------------------------------------------------------------------- helpers
def _letter(r: str) -> str:
    r = str(r)
    if r.startswith("H_prime"):
        return "H'"
    if r.startswith("I_prime"):
        return "I'"
    if r.startswith("A0"):
        return "A0"
    if r.startswith("O_mla"):
        return "O"
    return r.split("_")[0]


def load(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_parquet(path)
    df["diverged"] = df["diverged"].astype(str).str.lower().eq("true")
    df["L"] = df["rung"].astype(str).str.replace("ladder_", "", regex=False).map(_letter)
    df["has_final"] = df["final_val_loss"].map(
        lambda x: isinstance(x, (int, float)) and math.isfinite(float(x))
    )
    work = df.copy()
    for c in AXES:
        work[c] = work[c].astype(object)
    non_c = work["opt"] != "coupled_muon_v2"
    for c in [
        "coupled_steps",
        "final_polish",
        "couple_qk",
        "couple_vo",
        "couple_updown",
        "pair_policy",
        "use_multi_head",
        "qk_no_rope_policy",
        "qk_partial_rope_policy",
    ]:
        work.loc[non_c, c] = None
    work.loc[work["opt"] == "adamw", ["ns_steps", "ns_coefficients", "lr_prefactor"]] = None
    for c in AXES:
        work[c] = work[c].where(work[c].notna(), None)
    work["_ts"] = pd.to_datetime(work["created_at"], errors="coerce", utc=True)
    work = work.sort_values(["has_final", "_ts"], ascending=[False, False])
    work["_rank"] = work.groupby(AXES, dropna=False).cumcount()
    dedup = work[work["_rank"] == 0].copy()
    valid = dedup[
        dedup["has_final"] & ~dedup["diverged"] & dedup["seed"].notna() & (dedup["state"] == "finished")
    ].copy()
    return dedup, valid


def boot_median(x, n=10000, seed=0):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return (np.nan, np.nan, np.nan, 0)
    if len(x) == 1:
        return (float(x[0]), float(x[0]), float(x[0]), 1)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    m = np.median(x[idx], axis=1)
    return (float(np.median(x)), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975)), len(x))


def best_lr(sub: pd.DataFrame):
    by = sub.groupby("lr")["final_val_loss"].agg(["median", "size"]).reset_index()
    if by.empty:
        return None
    r = by.loc[by["median"].idxmin()]
    return float(r["lr"]), float(r["median"]), int(r["size"])


def paired(a: pd.DataFrame, b: pd.DataFrame):
    m = a[["seed", "final_val_loss"]].merge(b[["seed", "final_val_loss"]], on="seed", suffixes=("_a", "_b"))
    return boot_median((m["final_val_loss_a"] - m["final_val_loss_b"]).to_numpy())


def pooled_std(a: pd.DataFrame, b: pd.DataFrame) -> float:
    s = [x["final_val_loss"].std(ddof=1) for x in (a, b) if len(x) >= 2]
    return float(np.sqrt(np.mean(np.square(s)))) if s else float("nan")


def compare(label: str, arms: dict[str, pd.DataFrame]) -> dict:
    out = {}
    for k, sub in arms.items():
        bl = best_lr(sub)
        out[k] = (bl, sub[sub["lr"] == bl[0]] if bl else sub.iloc[0:0])
    print(
        label
        + " | "
        + " | ".join(
            f"{k}: lr={bl[0]:g} med={bl[1]:.4f} n={bl[2]}" if bl else f"{k}: none"
            for k, (bl, _) in out.items()
        )
    )
    names = list(out)
    res = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = out[names[i]][1], out[names[j]][1]
            med, lo, hi, n = paired(a, b)
            std = pooled_std(a, b)
            flag = "" if (np.isnan(lo) or lo <= 0 <= hi) else " **"
            print(
                f"    {names[i]} - {names[j]}: {med:+.4f} [{lo:+.4f},{hi:+.4f}] "
                f"n={n} pooled_std={std:.4f}{flag}"
            )
            res[(names[i], names[j])] = (med, lo, hi, n, std)
    return res


# ----------------------------------------------------------------------------- analyze
def analyze(path: str) -> None:
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 60)
    pd.set_option("display.max_rows", 500)
    dedup, valid = load(path)

    print("=== PER-PROJECT COMPLETION (deduped) ===")
    print(
        dedup.groupby("project")
        .agg(
            runs=("run_id", "size"),
            finished=("state", lambda s: (s == "finished").sum()),
            has_final=("has_final", "sum"),
            flagged_diverged=("diverged", "sum"),
            rungs=("L", lambda s: ",".join(sorted(set(map(str, s))))),
            opts=("opt", lambda s: ",".join(sorted(set(map(str, s))))),
        )
        .to_string()
    )

    opts3 = ["coupled_muon_v2", "muon", "adamw"]
    print("\n=== STAGE 1 + 2 dense (A0-repro, ladder) ===")
    lad = valid[valid["project"].isin(["coupled-muon-A0-repro", "coupled-muon-ladder"])]
    for rung in ["A0", "B", "C", "D", "E", "G", "H"]:
        vr = lad[lad["L"] == rung]
        compare(f"[dense] {rung}", {o: vr[vr["opt"] == o] for o in opts3 if (vr["opt"] == o).any()})

    print("\n=== STAGE 3 sparse MoE (pt1-prod, pt2-screen, AdamW anchors) ===")
    moe = valid[
        valid["project"].isin(
            [
                "coupled-muon-moe-pt1-prod",
                "coupled-muon-moe-pt2-screen",
                "coupled-muon-moe-anchor-adamw",
                "coupled-muon-moe-anchor-adamw-Iprime",
            ]
        )
    ]
    for rung in ["I", "I'", "J", "K", "L", "M"]:
        vr = moe[moe["L"] == rung]
        compare(f"[MoE] {rung}", {o: vr[vr["opt"] == o] for o in opts3 if (vr["opt"] == o).any()})

    print(
        "\n=== PHASE 2.1 B1 qk-policy: per (rung, policy, lr) — n / probe-flagged / median incl. flagged ==="
    )
    qk = dedup[dedup["project"] == "coupled-muon-qk-policy"].copy()
    qk["pol"] = np.where(qk["L"] == "H'", qk["qk_partial_rope_policy"], qk["qk_no_rope_policy"])
    qk.loc[qk["opt"] == "muon", "pol"] = "MUON(H')"
    lad_all = dedup[(dedup["project"] == "coupled-muon-ladder") & (dedup["opt"] != "adamw")].copy()
    lad_all["pol"] = np.where(lad_all["opt"] == "muon", "MUON(ladder)", "legacy(ladder)")
    both = pd.concat([qk, lad_all[lad_all["L"].isin(["C", "D", "E"])]])
    ok = both[both["final_val_loss"].notna() & (both["final_val_loss"] < 4.0)]
    t = (
        ok.groupby(["L", "pol", "lr"])
        .agg(n=("run_id", "size"), flagged=("diverged", "sum"), med=("final_val_loss", "median"))
        .round(4)
    )
    print(t.to_string())
    print("\n--- B1/B2 own-best-LR paired readout (protocol: flagged rows excluded) ---")
    qkv = valid[valid["project"] == "coupled-muon-qk-policy"]
    b2 = valid[valid["project"] == "coupled-muon-qk-policy-confirm"]
    for rung in ["C", "D", "E"]:
        arms = {}
        for pol in ["current_flat2d_fallback", "qk_off", "headwise_no_rope"]:
            sub = qkv[(qkv["L"] == rung) & (qkv["qk_no_rope_policy"] == pol)]
            if len(sub):
                arms[pol] = sub
        if (b2["L"] == rung).any():
            arms["B2 confirm"] = b2[b2["L"] == rung]
        arms["muon(ladder)"] = lad[(lad["L"] == rung) & (lad["opt"] == "muon")]
        arms["legacy coupled(ladder)"] = lad[(lad["L"] == rung) & (lad["opt"] == "coupled_muon_v2")]
        compare(f"[B no-RoPE] {rung}", arms)
    hp = qkv[qkv["L"] == "H'"]
    arms = {
        pol: hp[(hp["opt"] == "coupled_muon_v2") & (hp["qk_partial_rope_policy"] == pol)]
        for pol in ["current_flat2d_fallback", "qk_off", "headwise_no_rope", "partial_rope_split"]
    }
    arms["B2 confirm"] = b2[b2["L"] == "H'"]
    arms["muon(H')"] = hp[hp["opt"] == "muon"]
    compare("[B partial-RoPE] H'", {k: v for k, v in arms.items() if len(v)})

    print("\n=== PHASE 2.1 D1 MLA (rows at the canonical 4768-step global batch only) ===")
    mv = valid[(valid["project"] == "coupled-muon-mla") & (valid["final_step"] == 4768)]
    res = compare("[D1] O_mla", {o: mv[mv["opt"] == o] for o in opts3})
    if ("coupled_muon_v2", "muon") in res:
        med, lo, hi, n, std = res[("coupled_muon_v2", "muon")]
        delta = -med
        print(
            f"D-GATE: delta(Muon-Coupled)={delta:+.4f}; 1.5*pooled_std={1.5 * std:.4f}; "
            f"CI excludes 0={'yes' if (lo > 0) or (hi < 0) else 'no'} -> "
            f"{'POSITIVE' if (delta > 1.5 * std and hi < 0) else 'NEGATIVE (Muon >= Coupled)'}"
        )
    n9536 = int((valid[valid["project"] == "coupled-muon-mla"]["final_step"] == 9536).sum())
    print(f"(excluded {n9536} re-run rows trained at half global batch / 9536 steps)")

    print("\n=== MLA kv_lora_rank sweep (coupled only; coupled-muon-lora-rank) ===")
    lr_ = valid[valid["project"] == "coupled-muon-lora-rank"]
    print(lr_.groupby(["kv_lora_rank", "lr"])["final_val_loss"].agg(["median", "size"]).round(4).to_string())

    print("\n=== A3 K-curve (coupled_steps) vs plain Muon at same LR ===")
    mref = valid[
        (valid["project"].isin(["coupled-muon-ablation", "coupled-muon-ns-policy"]))
        & (valid["opt"] == "muon")
        & (valid["lr"] == 0.003)
        & (valid["ns_steps"] == 5)
        & ((valid["ns_coefficients"] == "bernstein") | valid["ns_coefficients"].isna())
    ]
    kc = valid[valid["project"] == "coupled-muon-k-curve"]
    for K in sorted(kc["coupled_steps"].dropna().unique()):
        sub = kc[kc["coupled_steps"] == K]
        med, lo, hi, n = paired(sub, mref)
        print(
            f"A0 K={int(K)}: n={len(sub)} med={sub['final_val_loss'].median():.4f}  "
            f"coupled-muon {med:+.4f} [{lo:+.4f},{hi:+.4f}] n={n}"
        )
    kci = valid[valid["project"] == "coupled-muon-k-curve-I"]
    mi = moe[(moe["L"] == "I") & (moe["opt"] == "muon") & (moe["lr"] == 0.01)]
    for K in sorted(kci["coupled_steps"].dropna().unique()):
        sub = kci[kci["coupled_steps"] == K]
        med, lo, hi, n = paired(sub, mi)
        print(
            f"I  K={int(K)}: n={len(sub)} med={sub['final_val_loss'].median():.4f}  "
            f"coupled-muon {med:+.4f} [{lo:+.4f},{hi:+.4f}] n={n}"
        )

    print("\n=== A2 NS-coefficient policy × K (A0, lr=3e-3), coupled - muon paired by seed ===")
    ns = valid[valid["project"] == "coupled-muon-ns-policy"]
    for pol in ["bernstein", "cesista", "polar_express"]:
        for K in [3, 5, 8]:
            c = ns[(ns.opt == "coupled_muon_v2") & (ns.ns_coefficients == pol) & (ns.ns_steps == K)]
            m = ns[(ns.opt == "muon") & (ns.ns_coefficients == pol) & (ns.ns_steps == K)]
            med, lo, hi, n = paired(c, m)
            print(
                f"{pol:14s} K={K}: coupled n={len(c)} med={c.final_val_loss.median():.4f} | "
                f"muon n={len(m)} med={m.final_val_loss.median():.4f} | "
                f"delta {med:+.4f} [{lo:+.4f},{hi:+.4f}] n={n}"
            )

    print("\n=== C1 MoE pair_policy at I ===")
    pp = valid[valid["project"] == "coupled-muon-pair-policy"].copy()
    pp["pol"] = pp.apply(
        lambda r: {
            ("True", "True", "True"): "all",
            ("True", "True", "False"): "attention_only",
            ("False", "False", "True"): "ffn_only",
        }.get((str(r["couple_qk"]), str(r["couple_vo"]), str(r["couple_updown"])), "?"),
        axis=1,
    )
    compare(
        "[C1 5-seed @lr=1e-2, 5B tok] I",
        {p: pp[pp["pol"] == p] for p in ["all", "attention_only", "ffn_only"]},
    )
    pl = valid[valid["project"] == "coupled-muon-pair-policy-lr"]
    arms = {
        p: pl[pl["pair_policy"] == p]
        for p in ["all", "attention_only", "ffn_only"]
        if (pl["pair_policy"] == p).any()
    }
    arms["muon(I,pt1)"] = moe[(moe["L"] == "I") & (moe["opt"] == "muon")]
    compare("[C1-lr single-seed] I", arms)

    print("\n=== runtime ratio coupled/muon at own-best LR ===")
    for proj in [
        "coupled-muon-A0-repro",
        "coupled-muon-ladder",
        "coupled-muon-moe-pt1-prod",
        "coupled-muon-moe-pt2-screen",
        "coupled-muon-mla",
    ]:
        v = valid[valid["project"] == proj]
        for rung in sorted(v["L"].unique()):
            vr = v[v["L"] == rung]
            r = {}
            for o in ["coupled_muon_v2", "muon"]:
                s = vr[vr["opt"] == o]
                bl = best_lr(s)
                if bl:
                    r[o] = s[s["lr"] == bl[0]]["runtime"].median()
            if len(r) == 2:
                print(
                    f"{rung}: coupled={r['coupled_muon_v2'] / 3600:.2f}h muon={r['muon'] / 3600:.2f}h "
                    f"ratio={r['coupled_muon_v2'] / r['muon']:.3f}"
                )

    df = pd.read_parquet(path)
    print(
        f"\nTotal: {len(df)} runs, {df['runtime'].sum() / 3600:.0f} run-hours (2-GPU DDP cells), "
        f"{df['final_tokens'].sum() / 1e12:.2f}T tokens trained"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--fetch", metavar="OUT.parquet")
    g.add_argument("--analyze", metavar="IN.parquet")
    a = ap.parse_args()
    if a.fetch:
        fetch(a.fetch)
    else:
        analyze(a.analyze)


if __name__ == "__main__":
    main()
