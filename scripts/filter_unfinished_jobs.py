#!/usr/bin/env python3
"""Filter a sweep JSONL down to only-unfinished jobs.

For each job in the input JSONL, computes the expected `run_id` via the same
hash function `train.py` uses (`utils.run_id` over the resolved config + seed)
and writes the job to the output JSONL only if its result dir is missing or
the `[saved_ckpt]` completion marker is absent from `<run_id>/stdout.log`.

Use this to resume an interrupted sweep without re-running cleanly-finished
cells. Combine with `--force-redo-indices` to override completion for cells
you know are corrupted (e.g. `metrics.jsonl` truncated by a duplicate
dispatcher), where stdout.log still shows `[saved_ckpt]` but on-disk metrics
were overwritten.

Usage:
    uv run python scripts/filter_unfinished_jobs.py \\
        --in $RDIR/stage3_jobs.jsonl \\
        --out $RDIR/stage3_remaining.jsonl \\
        --results-dir $RDIR \\
        --force-redo-indices 0,1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from coupled_muon_nanogpt.train import load_config
from coupled_muon_nanogpt.utils import run_id


def has_saved_ckpt(stdout_log: Path) -> bool:
    if not stdout_log.exists():
        return False
    try:
        with stdout_log.open() as f:
            for line in f:
                if line.startswith("[saved_ckpt]"):
                    return True
    except OSError:
        return False
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--out", dest="out_path", required=True)
    p.add_argument("--results-dir", required=True, type=Path)
    p.add_argument(
        "--force-redo-indices",
        default="",
        help="Comma-separated 0-based job indices to force-redo even if finished.",
    )
    args = p.parse_args()

    force_redo: set[int] = set()
    if args.force_redo_indices.strip():
        force_redo = {int(x) for x in args.force_redo_indices.split(",") if x.strip()}

    repo_root = Path(__file__).resolve().parent.parent
    rdir = args.results_dir.resolve()

    with open(args.in_path) as f:
        jobs = [json.loads(line) for line in f if line.strip()]

    finished = partial = missing = forced = 0
    out_jobs: list[dict] = []

    for i, job in enumerate(jobs):
        base_path = repo_root / job["base"]
        overrides = [f"{k}={v}" for k, v in job["overrides"].items()]
        cfg = load_config(str(base_path), overrides)
        rid = run_id(cfg, int(job["seed"]))

        if i in force_redo:
            forced += 1
            out_jobs.append(job)
            continue

        run_dir = rdir / rid
        if not run_dir.exists():
            missing += 1
            out_jobs.append(job)
            continue

        if has_saved_ckpt(run_dir / "stdout.log"):
            finished += 1
            continue

        partial += 1
        out_jobs.append(job)

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for j in out_jobs:
            f.write(json.dumps(j) + "\n")

    print(f"Total jobs:         {len(jobs)}")
    print(f"  Finished (skip):  {finished}")
    print(f"  Partial (rerun):  {partial}")
    print(f"  Missing (rerun):  {missing}")
    print(f"  Forced (rerun):   {forced}")
    print(f"Wrote {len(out_jobs)} unfinished jobs to {out_path}")


if __name__ == "__main__":
    main()
