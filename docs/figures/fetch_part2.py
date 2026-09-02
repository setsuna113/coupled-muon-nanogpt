"""Refresh Part II sparse-MoE / Phase-2 W&B snapshots.

Usage:
    uv run python docs/figures/fetch_part2.py --refresh --summary-only
    uv run python docs/figures/fetch_part2.py --refresh --with-history
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import DATA_DIR, PART2_PROJECTS
from _fetch import history_df, summary_df
from part2_analysis import (
    best_lr_run_rows,
    completion_table,
    dedupe_runs,
    missing_cells,
    planned_cells,
    stage3_rows,
    valid_final_rows,
)

PROBE_HISTORY_KEYS = [
    "val_loss",
    "loss",
    "probe/attn_logit/global_max",
    "probe/moe_load/layer.0/router_entropy",
    "probe/moe_load/layer.1/router_entropy",
    "probe/moe_load/layer.2/router_entropy",
    "probe/moe_load/layer.3/router_entropy",
    "probe/moe_load/layer.0/imbalance",
    "probe/moe_load/layer.1/imbalance",
    "probe/moe_load/layer.2/imbalance",
    "probe/moe_load/layer.3/imbalance",
    "probe/moe_load/layer.0/grad_norm_var",
    "probe/moe_load/layer.1/grad_norm_var",
    "probe/moe_load/layer.2/grad_norm_var",
    "probe/moe_load/layer.3/grad_norm_var",
]


def _write_manifest(summary: pd.DataFrame, errors: dict[str, str]) -> None:
    expected = planned_cells()
    kept, discarded = dedupe_runs(summary)
    completion = completion_table(summary)
    missing = missing_cells(expected, kept)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_parquet(DATA_DIR / "part2_summary.parquet", index=False)
    kept.to_parquet(DATA_DIR / "part2_summary_deduped.parquet", index=False)
    completion.to_csv(DATA_DIR / "part2_completion.csv", index=False)
    missing.to_csv(DATA_DIR / "part2_missing_cells.csv", index=False)
    discarded.to_csv(DATA_DIR / "part2_discarded_duplicates.csv", index=False)

    manifest = {
        "snapshot_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "projects": completion.to_dict(orient="records"),
        "fetch_errors": errors,
        "missing_cells_csv": "docs/data/part2_missing_cells.csv",
        "discarded_duplicates_csv": "docs/data/part2_discarded_duplicates.csv",
        "dedupe_rule": (
            "project/rung/optimizer/lr/seed plus active ablation axes; "
            "keep final non-null finished rows, prefer latest W&B timestamp"
        ),
    }
    (DATA_DIR / "part2_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("  wrote docs/data/part2_manifest.json")
    print(f"  wrote docs/data/part2_summary.parquet ({len(summary)} raw rows)")
    print(f"  wrote docs/data/part2_summary_deduped.parquet ({len(kept)} kept, {len(discarded)} duplicate rows)")
    print(f"  wrote docs/data/part2_missing_cells.csv ({len(missing)} missing final cells)")


def _fetch_histories(summary: pd.DataFrame, *, refresh: bool) -> None:
    kept, _ = dedupe_runs(summary)
    stage3 = stage3_rows(valid_final_rows(kept))
    best_rows, _ = best_lr_run_rows(stage3, group_keys=["rung", "optimizer"])
    if best_rows.empty:
        print("  no Stage 3 best-LR rows available for history fetch")
        return
    for project, sub in best_rows.groupby("project"):
        run_ids = sorted(set(sub["run_id"].dropna()))
        if not run_ids:
            continue
        print(f"  history {project}: {len(run_ids)} best-LR runs")
        history_df(
            project,
            run_ids=run_ids,
            keys=PROBE_HISTORY_KEYS,
            samples=300,
            cache_tag="part2_bestlr_probes",
            refresh=refresh,
        )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--refresh", action="store_true", help="Bypass cached project summaries")
    p.add_argument("--summary-only", action="store_true", help="Compatibility flag; summaries are the default")
    p.add_argument("--with-history", action="store_true", help="Also fetch selected best-LR history/probe curves")
    args = p.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    errors: dict[str, str] = {}
    print(f"Cache dir: {DATA_DIR}")
    for project in PART2_PROJECTS:
        print(f"\n[fetch] {project}", flush=True)
        try:
            df = summary_df(project, refresh=args.refresh)
        except Exception as exc:
            errors[project] = f"{type(exc).__name__}: {exc}"
            print(f"  ERROR: {errors[project]}", flush=True)
            continue
        frames.append(df)

    summary = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    _write_manifest(summary, errors)
    if args.with_history:
        _fetch_histories(summary, refresh=args.refresh)
    print("\nDone. Part II snapshot in docs/data/.")
    # Missing future Phase-2 projects are part of the report evidence, not a
    # CLI failure; the manifest records them explicitly.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
