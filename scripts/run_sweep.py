#!/usr/bin/env python3
"""Sequential multi-seed/LR sweep launcher.

Reads a JSONL sweep file (output of `python -m coupled_muon_nanogpt.sweep.expand`)
and runs each job sequentially via `torchrun`. Each job's stdout is teed to
`results/<run_id>/stdout.log`. Failures are logged but do not abort the rest of
the sweep — partial completion produces partial-but-real data, which is more
useful than discarding the whole sweep.

Usage:
    python scripts/run_sweep.py sweep_jobs.jsonl
    python scripts/run_sweep.py sweep_jobs.jsonl --nproc-per-node 2 --start-from 12
    python scripts/run_sweep.py sweep_jobs.jsonl --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


def _job_to_argv(job: dict, nproc: int, repo_root: Path) -> list[str]:
    overrides = []
    for k, v in job["overrides"].items():
        # OmegaConf.from_dotlist expects "key=value"; YAML yields proper types.
        overrides += ["--override", f"{k}={v}"]
    cmd = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={nproc}",
        "-m",
        "coupled_muon_nanogpt.train",
        "--config",
        str(repo_root / job["base"]),
        "--seed",
        str(job["seed"]),
        *overrides,
    ]
    return cmd


def _resolved_run_id(cmd: list[str]) -> str | None:
    """Best-effort run-id parse from train.py's first stdout line."""
    return None  # We capture it from the actual stdout instead.


def _tee_stream(proc: subprocess.Popen, log_path: Path) -> str | None:
    """Stream proc.stdout to both terminal and log_path; return discovered run_id."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    run_id = None
    with log_path.open("w") as f:
        for line in proc.stdout:  # type: ignore[union-attr]
            sys.stdout.write(line)
            f.write(line)
            if line.startswith("[run_id]"):
                run_id = line.split(maxsplit=1)[1].strip()
    return run_id


def main():
    p = argparse.ArgumentParser()
    p.add_argument("jobs", help="JSONL file produced by sweep.expand")
    p.add_argument("--nproc-per-node", type=int, default=1)
    p.add_argument("--start-from", type=int, default=0, help="resume from job index N")
    p.add_argument("--dry-run", action="store_true", help="print commands without running")
    p.add_argument(
        "--log-dir",
        type=str,
        default="results/_sweep_logs",
        help="directory for per-job stdout logs (used until the run-id is parsed)",
    )
    args = p.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    log_root = repo_root / args.log_dir

    with open(args.jobs) as f:
        jobs = [json.loads(line) for line in f if line.strip()]

    if args.start_from >= len(jobs):
        print(f"start-from={args.start_from} ≥ {len(jobs)} jobs; nothing to run.")
        return

    print(f"Running {len(jobs) - args.start_from} jobs sequentially "
          f"(nproc={args.nproc_per_node}; first={args.start_from})")
    successes, failures = 0, 0
    t0 = time.time()
    for i, job in enumerate(jobs):
        if i < args.start_from:
            continue
        cmd = _job_to_argv(job, args.nproc_per_node, repo_root)
        printable = " ".join(shlex.quote(c) for c in cmd)
        print(f"\n[{i + 1}/{len(jobs)}] {printable}")
        if args.dry_run:
            continue
        provisional_log = log_root / f"job_{i:04d}.log"
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(repo_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                text=True,
                env=env,
            )
            run_id = _tee_stream(proc, provisional_log)
            ret = proc.wait()
            if run_id:
                final_log = repo_root / "results" / run_id / "stdout.log"
                final_log.parent.mkdir(parents=True, exist_ok=True)
                provisional_log.replace(final_log)
            if ret != 0:
                print(f"[{i + 1}/{len(jobs)}] FAILED with exit code {ret}")
                failures += 1
            else:
                successes += 1
        except Exception as exc:  # noqa: BLE001
            print(f"[{i + 1}/{len(jobs)}] EXCEPTION: {exc}")
            failures += 1

    elapsed = time.time() - t0
    print(
        f"\nDone in {elapsed / 3600:.2f} h. "
        f"{successes}/{len(jobs) - args.start_from} succeeded, {failures} failed."
    )


if __name__ == "__main__":
    main()
