"""Refresh the parquet snapshots under docs/data/.

Usage:
    uv run python docs/figures/fetch_all.py            # use cache if present
    uv run python docs/figures/fetch_all.py --refresh  # re-fetch all
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from anywhere: add docs/figures to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import (
    DATA_DIR,
    LONG_TO_SHORT,
    PROJECT_ABLATION,
    PROJECT_STAGE1,
    PROJECT_STAGE2,
    RUNG_ORDER,
)
from _fetch import history_df, manifest_update, summary_df
from _stats import best_lr_per_arm


def _stage1_history(refresh: bool) -> None:
    """Pull loss curves for best-LR cells of A0 (one curve per (opt, seed))."""
    s1 = summary_df(PROJECT_STAGE1, refresh=refresh)
    bests = best_lr_per_arm(s1, group_keys=["rung", "optimizer"])
    keys_curve = ["val_loss", "loss"]
    run_ids: list[str] = []
    for _, row in bests.iterrows():
        sub = s1[(s1["rung"] == row["rung"]) & (s1["optimizer"] == row["optimizer"]) & (s1["lr"] == row["lr"])]
        run_ids.extend(sub["run_id"].tolist())
    print(f"  Stage 1 curve history: {len(run_ids)} runs at best LR")
    history_df(
        PROJECT_STAGE1,
        run_ids=run_ids,
        keys=keys_curve,
        samples=300,
        cache_tag="bestlr_curves",
        refresh=refresh,
    )

    # Probe history for one converged seed per opt at best LR.
    probe_keys = [
        "probe/coupled_pair/qk.0/kappa",
        "probe/coupled_pair/qk.0/sigma_max",
        "probe/coupled_pair/vo.0/kappa",
        "probe/coupled_pair/vo.0/sigma_max",
        "probe/coupled_pair/updown.0/kappa",
        "probe/coupled_pair/updown.0/sigma_max",
        "probe/attn_logit/global_max",
    ]
    probe_run_ids = []
    for _, row in bests.iterrows():
        sub = s1[
            (s1["rung"] == row["rung"])
            & (s1["optimizer"] == row["optimizer"])
            & (s1["lr"] == row["lr"])
            & (~s1["diverged"].astype(bool))
        ].sort_values("seed", na_position="last")
        if len(sub):
            probe_run_ids.append(sub.iloc[0]["run_id"])
    print(f"  Stage 1 probe history: {len(probe_run_ids)} runs (1 seed per opt at best LR)")
    history_df(
        PROJECT_STAGE1,
        run_ids=probe_run_ids,
        keys=probe_keys,
        samples=200,
        cache_tag="bestlr_probes",
        refresh=refresh,
    )


def _stage2_history(refresh: bool) -> None:
    s2 = summary_df(PROJECT_STAGE2, refresh=refresh)
    bests = best_lr_per_arm(s2, group_keys=["rung", "optimizer"])
    keys_curve = ["val_loss", "loss"]
    run_ids: list[str] = []
    for _, row in bests.iterrows():
        sub = s2[(s2["rung"] == row["rung"]) & (s2["optimizer"] == row["optimizer"]) & (s2["lr"] == row["lr"])]
        run_ids.extend(sub["run_id"].tolist())
    print(f"  Stage 2 curve history: {len(run_ids)} runs at best LR")
    history_df(
        PROJECT_STAGE2,
        run_ids=run_ids,
        keys=keys_curve,
        samples=300,
        cache_tag="bestlr_curves",
        refresh=refresh,
    )


def _ablation_history(refresh: bool) -> None:
    a = summary_df(PROJECT_ABLATION, refresh=refresh)
    # B.2 (final_polish on/off + post-stage-1 sigma_max trace): pick a single
    # coupled run per final_polish value at lr=3e-3.
    sub = a[(a["optimizer"] == "coupled_muon_v2") & (a["lr"] == 3e-3)]
    pick_ids = []
    for fp_val in [True, False]:
        sub2 = sub[sub["final_polish"] == fp_val].sort_values("seed", na_position="last")
        if len(sub2):
            pick_ids.append(sub2.iloc[0]["run_id"])
    if pick_ids:
        history_df(
            PROJECT_ABLATION,
            run_ids=pick_ids,
            keys=[
                "probe/ns_internal/coupled_pre_polish/sigma_max",
                "probe/ns_internal/coupled_post_polish/sigma_max",
            ],
            samples=200,
            cache_tag="final_polish_sigma",
            refresh=refresh,
        )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--refresh", action="store_true", help="Bypass parquet cache")
    p.add_argument("--summary-only", action="store_true", help="Skip history fetches")
    args = p.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Cache dir: {DATA_DIR}")

    print(f"\n[1/3] Stage 1 ({PROJECT_STAGE1})")
    s1 = summary_df(PROJECT_STAGE1, refresh=args.refresh)
    manifest_update({
        "project": PROJECT_STAGE1,
        "n_runs": int(len(s1)),
        "n_diverged": int(s1["diverged"].astype(bool).sum()),
        "rungs": sorted({x for x in s1["rung"].dropna()}),
    })
    if not args.summary_only:
        _stage1_history(args.refresh)

    print(f"\n[2/3] Stage 2 ({PROJECT_STAGE2})")
    s2 = summary_df(PROJECT_STAGE2, refresh=args.refresh)
    manifest_update({
        "project": PROJECT_STAGE2,
        "n_runs": int(len(s2)),
        "n_diverged": int(s2["diverged"].astype(bool).sum()),
        "rungs": sorted({x for x in s2["rung"].dropna()}),
    })
    if not args.summary_only:
        _stage2_history(args.refresh)

    print(f"\n[3/3] Ablation ({PROJECT_ABLATION})")
    a = summary_df(PROJECT_ABLATION, refresh=args.refresh)
    manifest_update({
        "project": PROJECT_ABLATION,
        "n_runs": int(len(a)),
    })
    if not args.summary_only:
        _ablation_history(args.refresh)

    print("\nDone. Snapshot in docs/data/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
