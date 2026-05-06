"""Aggregate per-run metrics.jsonl + config.yaml across a sweep, compute
bootstrap 95 % CIs on the median final val_loss across seeds.

Usage:
    python -m coupled_muon_nanogpt.analysis.bootstrap \\
        --results results/ \\
        --group config,optimizer.type,optimizer.lr \\
        --out summary.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def aggregate_runs(
    results_root: str | Path,
    group_by: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Walk `results/<run_id>/`, parse each metrics.jsonl + config.yaml, and
    produce a list of {key=value, ..., final_val_loss=, run_id=, seed=} dicts.

    `group_by`: dotted config keys plus the literal "config" (the run.name).
    Used by the CLI to bucket rows for bootstrap; this function returns a flat
    list and lets callers do the grouping themselves.
    """
    from omegaconf import OmegaConf

    rows: list[dict[str, Any]] = []
    root = Path(results_root)
    if not root.exists():
        return rows

    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        cfg_path = run_dir / "config.yaml"
        metrics_path = run_dir / "metrics.jsonl"
        if not cfg_path.exists() or not metrics_path.exists():
            continue

        cfg = OmegaConf.load(cfg_path)
        # Find the last `event:"val"` row.
        last_val: dict[str, Any] | None = None
        with metrics_path.open() as f:
            for line in f:
                try:
                    j = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if j.get("event") == "val":
                    last_val = j
        if last_val is None:
            continue

        row: dict[str, Any] = {
            "run_id": run_dir.name,
            "name": str(cfg.get("name", "")),
            "optimizer.type": str(cfg.optimizer.type),
            "optimizer.lr": float(cfg.optimizer.lr),
            "optimizer.coupled_steps": int(cfg.optimizer.get("coupled_steps", 0)),
            "optimizer.couple_qk": bool(cfg.optimizer.get("couple_qk", False)),
            "optimizer.couple_vo": bool(cfg.optimizer.get("couple_vo", False)),
            "optimizer.couple_updown": bool(cfg.optimizer.get("couple_updown", False)),
            "moe.enabled": bool(cfg.model.moe.enabled),
            "moe.balancing_type": str(cfg.model.moe.get("balancing_type", "")),
            "final_val_loss": float(last_val.get("val_loss", float("nan"))),
            "final_tokens": int(last_val.get("tokens", 0)),
            "final_step": int(last_val.get("step", 0)),
        }
        rows.append(row)
    return rows


def bootstrap_median_ci(
    losses: list[float],
    *,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap of the median. Returns (median, lo, hi)."""
    arr = np.asarray([x for x in losses if math.isfinite(x)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), float(arr[0]), float(arr[0])
    rng = np.random.default_rng(seed)
    medians = np.empty(n_resamples)
    n = arr.size
    for i in range(n_resamples):
        sample = arr[rng.integers(0, n, size=n)]
        medians[i] = np.median(sample)
    lo = float(np.quantile(medians, alpha / 2))
    hi = float(np.quantile(medians, 1 - alpha / 2))
    return float(np.median(arr)), lo, hi


def _group_key(row: dict[str, Any], keys: list[str]) -> tuple:
    return tuple(row.get(k, None) for k in keys)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=str, default="results")
    p.add_argument(
        "--group",
        type=str,
        default="name,optimizer.type,optimizer.lr",
        help="comma-separated dotted keys for grouping (default: name, optimizer.type, optimizer.lr)",
    )
    p.add_argument("--out", type=str, default="summary.csv")
    p.add_argument("--n-resamples", type=int, default=10_000)
    p.add_argument("--alpha", type=float, default=0.05)
    args = p.parse_args()

    keys = [k.strip() for k in args.group.split(",") if k.strip()]
    rows = aggregate_runs(args.results)
    if not rows:
        print(f"No completed runs found under {args.results}/")
        return

    groups: dict[tuple, list[dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(_group_key(r, keys), []).append(r)

    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([*keys, "n_seeds", "median_loss", "ci_lo", "ci_hi", "final_tokens_median"])
        for k, group in sorted(groups.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
            losses = [r["final_val_loss"] for r in group]
            med, lo, hi = bootstrap_median_ci(losses, n_resamples=args.n_resamples, alpha=args.alpha)
            tok_med = float(np.median([r["final_tokens"] for r in group])) if group else 0.0
            w.writerow([*k, len(group), f"{med:.6f}", f"{lo:.6f}", f"{hi:.6f}", int(tok_med)])
    print(f"Wrote {args.out} ({len(groups)} groups, {len(rows)} runs)")


if __name__ == "__main__":
    main()
