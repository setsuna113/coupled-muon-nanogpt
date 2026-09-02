#!/usr/bin/env python3
"""Archive per-run W&B histories (validation curve, subsampled train loss,
attention-logit probe) for every `coupled-muon-*` run into
``docs/data/histories/<project>.parquet``.

The committed summary snapshot (``docs/data/wandb_all_runs_<date>.parquet``)
holds only final values; the report caches hold curves for best-LR cells
only. Free-tier W&B retention makes the raw histories the one asset that
can disappear, so this pulls them once, project by project, with
one unsampled ``scan_history`` pass per run (whole-project ``history()``
calls time out). Idempotent: projects whose parquet already exists are skipped.

    WANDB_API_KEY=... python scripts/fetch_wandb_histories.py [--project NAME]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ENT = "liuyc1025-university-of-cambridge"
OUT = Path(__file__).resolve().parent.parent / "docs" / "data" / "histories"
VAL_KEYS = ["tokens", "val_loss"]
TRAIN_KEYS = ["step", "tokens", "loss", "lr", "opt_step_s", "tok_per_s"]
PROBE_KEYS = ["tokens", "probe/attn_logit/global_max"]
TRAIN_EVERY = 100  # keep one train-loss row per 100 steps


def rows_for(run):
    """One unsampled pass over the run's history (``scan_history`` with no
    ``keys`` filter is the only call that returns every logged row), split
    into the three row kinds by which keys are present."""
    out = []
    try:
        for r in run.scan_history(page_size=2000):
            base = {"run_id": run.id, "name": run.name}
            if r.get("val_loss") is not None:
                out.append({**base, "kind": "val", **{k: r.get(k) for k in VAL_KEYS}})
            if r.get("probe/attn_logit/global_max") is not None:
                out.append({**base, "kind": "attn_logit", **{k: r.get(k) for k in PROBE_KEYS}})
            if r.get("loss") is not None and (r.get("step") or 0) % TRAIN_EVERY == 0:
                out.append({**base, "kind": "train", **{k: r.get(k) for k in TRAIN_KEYS}})
    except Exception as e:  # noqa: BLE001
        print(f"  {run.name}: scan failed: {e}", file=sys.stderr)
    return out


def main() -> None:
    import wandb

    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=None)
    a = ap.parse_args()
    api = wandb.Api(timeout=180)
    OUT.mkdir(parents=True, exist_ok=True)
    projects = (
        [a.project]
        if a.project
        else sorted(p.name for p in api.projects(ENT) if p.name.startswith("coupled-muon"))
    )
    for pname in projects:
        path = OUT / f"{pname}.parquet"
        if path.exists():
            print(f"{pname}: exists, skip", file=sys.stderr)
            continue
        t0 = time.time()
        runs = list(api.runs(f"{ENT}/{pname}", per_page=500))
        rows = []
        for i, run in enumerate(runs):
            rows += rows_for(run)
            if (i + 1) % 25 == 0:
                print(
                    f"  {pname}: {i + 1}/{len(runs)} runs, {len(rows)} rows, {time.time() - t0:.0f}s",
                    file=sys.stderr,
                )
        df = pd.DataFrame(rows)
        for c in df.columns:
            if c not in ("run_id", "name", "kind"):
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df.to_parquet(path, index=False)
        print(
            f"{pname}: {len(runs)} runs -> {len(df)} rows -> {path} ({time.time() - t0:.0f}s)",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
