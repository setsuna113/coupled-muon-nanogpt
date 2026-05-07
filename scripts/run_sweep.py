#!/usr/bin/env python3
"""Multi-seed/LR sweep launcher with optional parallel dispatch.

Reads a JSONL sweep file (output of `python -m coupled_muon_nanogpt.sweep.expand`)
and runs each job via `torchrun`. Sequential by default; pass `--num-workers N`
to dispatch N cells concurrently to disjoint GPU slices.

Usage:
    # Sequential on 2×H200 (one cell at a time, 2-GPU DDP per cell)
    python scripts/run_sweep.py sweep_jobs.jsonl --nproc-per-node 2

    # Parallel on 8×H100 (4 cells concurrently, 2-GPU DDP per cell)
    python scripts/run_sweep.py sweep_jobs.jsonl --nproc-per-node 2 --num-workers 4

    # Resume from job index 12
    python scripts/run_sweep.py sweep_jobs.jsonl --num-workers 4 --start-from 12

Per-cell GPU pinning uses CUDA_VISIBLE_DEVICES: worker w gets GPUs
[w*K, w*K+K-1] where K = --gpus-per-worker (defaults to --nproc-per-node).
torchrun's --standalone selects a free rendezvous port (localhost:0), so
concurrent torchrun groups don't collide.

Per-cell stdout lands in `results/<run_id>/stdout.log` once the run-id is
parsed; before that, in `results/_sweep_logs/job_NNNN.log`. Terminal output
streams the cell's stdout only in single-worker mode; in parallel mode the
terminal sees only worker progress markers (full per-cell output is in the
file).
"""
from __future__ import annotations

import argparse
import json
import os
import queue as _queue
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path


def _job_to_argv(
    job: dict,
    nproc: int,
    repo_root: Path,
    extra_overrides: list[str] | None = None,
) -> list[str]:
    overrides = []
    for k, v in job["overrides"].items():
        overrides += ["--override", f"{k}={v}"]
    for ov in extra_overrides or []:
        overrides += ["--override", ov]
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


def _stream(proc: subprocess.Popen, log_path: Path, to_stdout: bool) -> str | None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    run_id = None
    with log_path.open("w") as f:
        for line in proc.stdout:  # type: ignore[union-attr]
            if to_stdout:
                sys.stdout.write(line)
            f.write(line)
            if line.startswith("[run_id]"):
                run_id = line.split(maxsplit=1)[1].strip()
    return run_id


def _run_one(
    i: int,
    job: dict,
    args: argparse.Namespace,
    repo_root: Path,
    log_root: Path,
    env: dict,
    total: int,
    tag: str,
) -> bool | None:
    cmd = _job_to_argv(job, args.gpus_per_worker, repo_root, args.extra_override)
    printable = " ".join(shlex.quote(c) for c in cmd)
    prefix = f"[{tag} {i + 1}/{total}]" if tag else f"[{i + 1}/{total}]"
    print(f"\n{prefix} {printable}", flush=True)
    if args.dry_run:
        return None
    provisional_log = log_root / f"job_{i:04d}.log"
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
        run_id = _stream(proc, provisional_log, to_stdout=(args.num_workers == 1))
        ret = proc.wait()
        if run_id:
            final_log = Path(args.results_dir) / run_id / "stdout.log"
            final_log.parent.mkdir(parents=True, exist_ok=True)
            provisional_log.replace(final_log)
        if ret != 0:
            print(f"{prefix} FAILED exit={ret}", flush=True)
            return False
        print(f"{prefix} OK", flush=True)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"{prefix} EXCEPTION: {exc}", flush=True)
        return False


def _worker_loop(
    worker_id: int,
    gpu_ids: list[int],
    job_q: _queue.Queue,
    args: argparse.Namespace,
    repo_root: Path,
    log_root: Path,
    total: int,
    stats: list[int],
    lock: threading.Lock,
):
    env = {
        **os.environ,
        "PYTHONUNBUFFERED": "1",
        "CUDA_VISIBLE_DEVICES": ",".join(str(g) for g in gpu_ids),
    }
    tag = f"w{worker_id}gpu{gpu_ids[0]}-{gpu_ids[-1]}"
    while True:
        try:
            i, job = job_q.get_nowait()
        except _queue.Empty:
            return
        ok = _run_one(i, job, args, repo_root, log_root, env, total, tag)
        if ok is None:
            continue
        with lock:
            if ok:
                stats[0] += 1
            else:
                stats[1] += 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("jobs", help="JSONL file produced by sweep.expand")
    p.add_argument("--nproc-per-node", type=int, default=1,
                   help="GPUs per cell (DDP world size). Ladder yamls assume 2.")
    p.add_argument("--num-workers", type=int, default=1,
                   help="Cells to run concurrently (default 1 = sequential).")
    p.add_argument("--gpus-per-worker", type=int, default=None,
                   help="GPUs visible to each worker (default = --nproc-per-node).")
    p.add_argument("--start-from", type=int, default=0, help="Resume from job index N.")
    p.add_argument("--dry-run", action="store_true", help="Print commands without running.")
    p.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help=(
            "Override the per-cell `run.output_dir` AND the sweep dispatcher's "
            "log root. Default: keep each cell's config default (usually "
            "`./results`) and put dispatcher logs under `<results>/_sweep_logs`."
        ),
    )
    p.add_argument(
        "--extra-override",
        action="append",
        default=[],
        help="Repeatable. Extra KEY=VALUE override appended to every cell.",
    )
    p.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Override sweep-dispatcher log dir (else `<results-dir>/_sweep_logs`).",
    )
    args = p.parse_args()
    if args.gpus_per_worker is None:
        args.gpus_per_worker = args.nproc_per_node
    if args.results_dir is not None:
        # Inject as a per-cell override and use it for the dispatcher's logs.
        args.extra_override.append(f"run.output_dir={args.results_dir}")
    else:
        args.results_dir = "results"
    if args.log_dir is None:
        args.log_dir = str(Path(args.results_dir) / "_sweep_logs")
    if args.gpus_per_worker != args.nproc_per_node:
        print(
            f"WARNING: --gpus-per-worker={args.gpus_per_worker} != "
            f"--nproc-per-node={args.nproc_per_node}; torchrun will use "
            f"{args.nproc_per_node} processes inside a {args.gpus_per_worker}-GPU mask.",
            file=sys.stderr,
        )

    repo_root = Path(__file__).resolve().parent.parent
    log_root = repo_root / args.log_dir

    with open(args.jobs) as f:
        jobs = [json.loads(line) for line in f if line.strip()]

    if args.start_from >= len(jobs):
        print(f"start-from={args.start_from} ≥ {len(jobs)} jobs; nothing to run.")
        return

    pending = [(i, j) for i, j in enumerate(jobs) if i >= args.start_from]
    total_gpus = args.num_workers * args.gpus_per_worker
    print(
        f"Running {len(pending)} jobs across {args.num_workers} worker(s) "
        f"× {args.gpus_per_worker} GPU = {total_gpus} GPUs total "
        f"(first={args.start_from})",
        flush=True,
    )

    t0 = time.time()
    if args.num_workers == 1:
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        successes, failures = 0, 0
        for i, job in pending:
            ok = _run_one(i, job, args, repo_root, log_root, env, len(jobs), "")
            if ok is None:
                continue
            if ok:
                successes += 1
            else:
                failures += 1
        elapsed = time.time() - t0
        print(
            f"\nDone in {elapsed / 3600:.2f} h. "
            f"{successes}/{len(pending)} succeeded, {failures} failed.",
            flush=True,
        )
        return

    # Parallel path: N worker threads pull from a shared queue; each owns a
    # disjoint GPU slice via CUDA_VISIBLE_DEVICES.
    job_q: _queue.Queue = _queue.Queue()
    for i, j in pending:
        job_q.put((i, j))

    stats = [0, 0]
    lock = threading.Lock()
    threads: list[threading.Thread] = []
    for w in range(args.num_workers):
        gpus = list(range(w * args.gpus_per_worker, (w + 1) * args.gpus_per_worker))
        t = threading.Thread(
            target=_worker_loop,
            args=(w, gpus, job_q, args, repo_root, log_root, len(jobs), stats, lock),
            daemon=False,
        )
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    elapsed = time.time() - t0
    print(
        f"\nDone in {elapsed / 3600:.2f} h. "
        f"{stats[0]}/{len(pending)} succeeded, {stats[1]} failed.",
        flush=True,
    )


if __name__ == "__main__":
    main()
