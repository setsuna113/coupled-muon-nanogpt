"""Shared Part II analysis helpers.

The Part II report is deliberately stricter than the older Stage 1/2 scripts:
seed must be known before paired analyses, duplicate W&B rows are resolved
explicitly, and Phase-2 ablation axes stay in the grouping key.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from _common import (
    DATA_DIR,
    MOE_RUNG_ORDER,
    PART2_PROJECTS,
    PROJECT_FACTFF,
    PROJECT_K_CURVE,
    PROJECT_LORA_RANK,
    PROJECT_LR_PREFACTOR,
    PROJECT_MLA,
    PROJECT_MLA_350M,
    PROJECT_NS_POLICY,
    PROJECT_PAIR_POLICY,
    PROJECT_STAGE3_ADAMW_I,
    PROJECT_STAGE3_ADAMW_IPRIME,
    PROJECT_STAGE3_PT1,
    PROJECT_STAGE3_PT2,
)
from _stats import best_lr_per_arm, paired_bootstrap

ACTIVE_ABLATION_AXES = [
    "ns_coefficients",
    "lr_prefactor",
    "coupled_steps",
    "ns_steps",
    "ns_gram_form",
    "couple_qk",
    "couple_vo",
    "couple_updown",
    "couple_router_to_muon",
    "couple_mla",
    "use_multi_head",
    "attn_type",
    "kv_lora_rank",
    "q_lora_rank",
    "mlp_type",
    "moe_balancing_type",
]

BASE_CELL_KEYS = ["project", "rung", "optimizer", "lr", "seed"]
STAGE3_PROJECTS = [
    PROJECT_STAGE3_PT1,
    PROJECT_STAGE3_PT2,
    PROJECT_STAGE3_ADAMW_I,
    PROJECT_STAGE3_ADAMW_IPRIME,
]


def normalize_inactive_axes(df: pd.DataFrame) -> pd.DataFrame:
    """Clear config toggles that are inert for a row's optimizer/architecture."""
    if df.empty:
        return df.copy()
    work = df.copy()
    if "optimizer" in work.columns:
        non_coupled = ~work["optimizer"].eq("coupled_muon_v2")
        for col in [
            "couple_qk",
            "couple_vo",
            "couple_updown",
            "couple_router_to_muon",
            "couple_mla",
            "use_multi_head",
        ]:
            if col in work.columns:
                work[col] = work[col].astype("object")
                work.loc[non_coupled, col] = None
        for col in ["coupled_steps", "final_polish", "ns_steps", "ns_coefficients", "ns_gram_form", "lr_prefactor"]:
            if col in work.columns:
                work[col] = work[col].astype("object")
                work.loc[work["optimizer"].eq("adamw"), col] = None
    if "has_mla" in work.columns:
        non_mla = ~work["has_mla"].fillna(False).astype(bool)
        for col in ["couple_mla", "kv_lora_rank", "q_lora_rank"]:
            if col in work.columns:
                work[col] = work[col].astype("object")
                work.loc[non_mla, col] = None
    return work


def _project_cache(project: str) -> Path:
    return DATA_DIR / f"{project.replace('-', '_')}_summary.parquet"


def load_part2_summary(projects: Iterable[str] = PART2_PROJECTS) -> pd.DataFrame:
    frames = []
    for project in projects:
        path = _project_cache(project)
        if path.exists():
            frames.append(pd.read_parquet(path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _finite(x) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def dedupe_runs(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep the latest final row within each analysis cell.

    Cell identity is project/rung/optimizer/LR/seed plus every active ablation
    axis present in the dataframe. This prevents, for example, an NS-policy row
    from being merged with the default production rows just because LR/seed
    match.
    """
    if df.empty:
        return df.copy(), df.copy()

    work = normalize_inactive_axes(df)
    for col in BASE_CELL_KEYS + ACTIVE_ABLATION_AXES:
        if col not in work.columns:
            work[col] = None

    work["_has_seed"] = work["seed"].notna()
    work["_has_final"] = work["final_val_loss"].map(_finite)
    work["_is_finished"] = work["state"].eq("finished") if "state" in work else False
    work["_updated_ts"] = pd.to_datetime(work.get("updated_at"), errors="coerce", utc=True)
    work["_created_ts"] = pd.to_datetime(work.get("created_at"), errors="coerce", utc=True)

    group_cols = [*BASE_CELL_KEYS, *ACTIVE_ABLATION_AXES]
    sort_cols = ["_has_final", "_is_finished", "_updated_ts", "_created_ts", "run_id"]
    work = work.sort_values(sort_cols, ascending=[False, False, False, False, False], na_position="last")
    work["_dedupe_rank"] = work.groupby(group_cols, dropna=False).cumcount()
    discarded = work[work["_dedupe_rank"] > 0].copy()
    kept = work[work["_dedupe_rank"] == 0].copy()
    drop_cols = ["_has_seed", "_has_final", "_is_finished", "_updated_ts", "_created_ts", "_dedupe_rank"]
    return kept.drop(columns=drop_cols), discarded.drop(columns=drop_cols)


def valid_final_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    work = df.copy()
    mask = work["seed"].notna() & work["final_val_loss"].map(_finite)
    if "diverged" in work.columns:
        mask &= ~work["diverged"].fillna(False).astype(bool)
    if "state" in work.columns:
        mask &= work["state"].eq("finished")
    return work[mask].copy()


def stage3_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    return df[df["project"].isin(STAGE3_PROJECTS)].copy()


def best_lr_run_rows(df: pd.DataFrame, *, group_keys: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = valid_final_rows(df)
    if valid.empty:
        return valid, pd.DataFrame()
    bests = best_lr_per_arm(valid, group_keys=group_keys)
    frames = []
    for _, row in bests.iterrows():
        mask = valid["lr"].eq(row["lr"])
        for key in group_keys:
            mask &= valid[key].eq(row[key])
        frames.append(valid[mask].copy())
    out = pd.concat(frames, ignore_index=True) if frames else valid.iloc[0:0].copy()
    return out, bests


def stage3_forest_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Paired deltas at each arm's own best LR."""
    clean = stage3_rows(valid_final_rows(df))
    if clean.empty:
        return pd.DataFrame(columns=["rung", "comparison", "median", "lo", "hi", "n"])

    best_rows, _ = best_lr_run_rows(clean, group_keys=["rung", "optimizer"])
    rows = []
    order = {r: i for i, r in enumerate(MOE_RUNG_ORDER)}
    for rung in sorted(best_rows["rung"].dropna().unique(), key=lambda r: order.get(r, 999)):
        sub = best_rows[best_rows["rung"].eq(rung)]
        coupled = sub[sub["optimizer"].eq("coupled_muon_v2")]
        muon = sub[sub["optimizer"].eq("muon")]
        adamw = sub[sub["optimizer"].eq("adamw")]
        comparisons = [
            ("Coupled - Muon", coupled, muon),
            ("Muon - AdamW", muon, adamw),
            ("Coupled - AdamW", coupled, adamw),
        ]
        for label, left, right in comparisons:
            if len(left) == 0 or len(right) == 0:
                continue
            ci = paired_bootstrap(left, right, on=["rung", "seed"])
            rows.append({
                "rung": rung,
                "comparison": label,
                "median": ci.median,
                "lo": ci.lo,
                "hi": ci.hi,
                "n": ci.n,
            })
    return pd.DataFrame(rows)


def stage3_best_lr_table(df: pd.DataFrame) -> pd.DataFrame:
    clean = stage3_rows(valid_final_rows(df))
    if clean.empty:
        return pd.DataFrame()
    _, bests = best_lr_run_rows(clean, group_keys=["rung", "optimizer"])
    rows = []
    for _, row in bests.iterrows():
        sub = clean[
            clean["rung"].eq(row["rung"])
            & clean["optimizer"].eq(row["optimizer"])
            & clean["lr"].eq(row["lr"])
        ]
        rows.append({
            "rung": row["rung"],
            "optimizer": row["optimizer"],
            "best_lr": row["lr"],
            "median_loss": float(sub["final_val_loss"].median()),
            "n_seeds": int(sub["seed"].nunique()),
            "seeds": ",".join(str(int(s)) for s in sorted(sub["seed"].dropna().unique())),
            "project": ",".join(sorted(sub["project"].dropna().unique())),
        })
    out = pd.DataFrame(rows)
    order = {r: i for i, r in enumerate(MOE_RUNG_ORDER)}
    opt_order = {"adamw": 0, "muon": 1, "coupled_muon_v2": 2}
    return out.sort_values(
        by=["rung", "optimizer"],
        key=lambda s: s.map(order if s.name == "rung" else opt_order).fillna(999),
    )


def ns_policy_table(df: pd.DataFrame) -> pd.DataFrame:
    clean = valid_final_rows(df[df["project"].eq(PROJECT_NS_POLICY)] if not df.empty else df)
    if clean.empty:
        return pd.DataFrame()
    rows = []
    keys = ["optimizer", "ns_coefficients", "coupled_steps"]
    for keys_val, sub in clean.groupby(keys, dropna=False):
        optimizer, policy, k = keys_val
        rows.append({
            "optimizer": optimizer,
            "policy": policy,
            "K": int(k) if pd.notna(k) else None,
            "median_loss": float(sub["final_val_loss"].median()),
            "n_seeds": int(sub["seed"].nunique()),
            "seeds": ",".join(str(int(s)) for s in sorted(sub["seed"].dropna().unique())),
        })
    out = pd.DataFrame(rows)
    policy_order = {"bernstein": 0, "cesista": 1, "polar_express": 2}
    opt_order = {"muon": 0, "coupled_muon_v2": 1}
    def _sort_key(s: pd.Series) -> pd.Series:
        if s.name == "optimizer":
            return s.map(opt_order).fillna(999)
        if s.name == "policy":
            return s.map(policy_order).fillna(999)
        return s

    return out.sort_values(by=["optimizer", "policy", "K"], key=_sort_key)


def planned_cells() -> pd.DataFrame:
    rows: list[dict] = []

    def add(project, rungs, optimizers, lrs, seeds, **fixed):
        for rung in rungs:
            for opt in optimizers:
                for lr in lrs:
                    for seed in seeds:
                        row = {"project": project, "rung": rung, "optimizer": opt, "lr": float(lr), "seed": seed}
                        row.update(fixed)
                        rows.append(row)

    add(PROJECT_STAGE3_PT1, ["I", "I'"], ["muon", "coupled_muon_v2"], [3e-3, 1e-2, 3e-2], range(5))
    add(PROJECT_STAGE3_PT2, ["J", "K", "L", "M", "N"], ["muon", "coupled_muon_v2"], [3e-3, 1e-2, 3e-2], range(3))
    add(PROJECT_STAGE3_ADAMW_I, ["I"], ["adamw"], [3e-5, 1e-4, 3e-4, 1e-3, 3e-3], range(5))
    add(PROJECT_STAGE3_ADAMW_IPRIME, ["I'"], ["adamw"], [3e-5, 1e-4, 3e-4, 1e-3, 3e-3], range(5))

    for opt in ["muon", "coupled_muon_v2"]:
        for policy in ["bernstein", "polar_express", "cesista"]:
            for k in [3, 5, 8]:
                add(
                    PROJECT_NS_POLICY,
                    ["A0"],
                    [opt],
                    [3e-3],
                    [0, 1],
                    ns_coefficients=policy,
                    coupled_steps=k,
                    ns_steps=k,
                )

    for k in [1, 2, 3, 5, 8, 12]:
        add(
            PROJECT_K_CURVE,
            ["A0"],
            ["coupled_muon_v2"],
            [3e-3],
            range(3),
            ns_coefficients="bernstein",
            coupled_steps=k,
        )
    for prefactor in ["moonlight", "bernstein_ratio", "cesista"]:
        add(
            PROJECT_LR_PREFACTOR,
            ["A0"],
            ["coupled_muon_v2"],
            [1e-3, 3e-3, 1e-2],
            range(2),
            lr_prefactor=prefactor,
        )
    add(PROJECT_MLA, ["O_mla"], ["adamw", "muon", "coupled_muon_v2"], [3e-3, 1e-2, 3e-2], range(5))
    add(PROJECT_FACTFF, ["A0", "P_factff"], ["muon", "coupled_muon_v2"], [3e-3], range(3))
    for rank in [16, 32, 64, 128, 256]:
        add(PROJECT_LORA_RANK, ["O_mla"], ["coupled_muon_v2"], [1e-2], range(3), kv_lora_rank=rank)
    for label, flags in [
        ("attn_only", {"couple_qk": True, "couple_vo": True, "couple_updown": False}),
        ("ffn_only", {"couple_qk": False, "couple_vo": False, "couple_updown": True}),
        ("all", {"couple_qk": True, "couple_vo": True, "couple_updown": True}),
    ]:
        add(PROJECT_PAIR_POLICY, ["I"], ["coupled_muon_v2"], [1e-2], range(5), pair_policy=label, **flags)
    add(PROJECT_MLA_350M, ["Z_350m_mla"], ["muon", "coupled_muon_v2"], [3e-3, 1e-2, 3e-2], range(5))
    return pd.DataFrame(rows)


def missing_cells(expected: pd.DataFrame, observed: pd.DataFrame) -> pd.DataFrame:
    kept, _ = dedupe_runs(observed)
    valid = valid_final_rows(kept)
    if expected.empty:
        return expected.copy()

    chunks = []
    for project, exp_sub in expected.groupby("project", dropna=False):
        obs_sub = valid[valid["project"].eq(project)] if not valid.empty else valid
        merge_cols = [c for c in BASE_CELL_KEYS if c in exp_sub.columns]
        for c in ACTIVE_ABLATION_AXES + ["pair_policy"]:
            if c in exp_sub.columns and exp_sub[c].notna().any():
                merge_cols.append(c)
        left = exp_sub.copy()
        right = obs_sub.copy()
        for c in merge_cols:
            if c not in right.columns:
                right[c] = np.nan
        marker = right[merge_cols].drop_duplicates()
        merged = left.merge(marker.assign(_observed=True), on=merge_cols, how="left")
        chunks.append(merged[merged["_observed"].isna()].drop(columns=["_observed"]))
    return pd.concat(chunks, ignore_index=True, sort=False) if chunks else expected.iloc[0:0].copy()


def completion_table(summary: pd.DataFrame) -> pd.DataFrame:
    expected = planned_cells()
    kept, discarded = dedupe_runs(summary)
    missing = missing_cells(expected, kept)
    rows = []
    for project, exp_sub in expected.groupby("project", sort=False):
        raw = summary[summary["project"].eq(project)] if not summary.empty else summary
        valid = valid_final_rows(kept[kept["project"].eq(project)] if not kept.empty else kept)
        miss = missing[missing["project"].eq(project)]
        states = raw["state"].value_counts().to_dict() if len(raw) and "state" in raw else {}
        rows.append({
            "project": project,
            "expected": int(len(exp_sub)),
            "wandb_runs": int(len(raw)),
            "final_cells": int(len(valid)),
            "missing_final": int(len(miss)),
            "duplicates_discarded": int(len(discarded[discarded["project"].eq(project)])) if len(discarded) else 0,
            "states": ", ".join(f"{k}:{v}" for k, v in sorted(states.items())) or "none",
        })
    return pd.DataFrame(rows)


def fmt_num(x: float | int | None, digits: int = 4) -> str:
    if x is None:
        return "--"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "--"
    if not math.isfinite(v):
        return "--"
    return f"{v:.{digits}f}"


def fmt_lr(x: float | None) -> str:
    if x is None or not _finite(x):
        return "--"
    return f"{float(x):.0e}"
