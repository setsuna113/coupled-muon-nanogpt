#!/usr/bin/env python3
"""Light-touch GPU keep-alive to prevent cluster idle-reaping.

Pods on Inspire (and similar SLURM/k8s setups) get torn down when their GPUs
sit at 0% for too long. This script holds each visible GPU at a configurable
duty cycle (default ~30%) by firing small matmuls on a fixed work/sleep
schedule. Memory footprint is tiny (~64 MiB per GPU at the default matrix
size) and the kernels yield SMs between bursts so a real training job
running alongside is barely affected.

Typical usage — run as a sidecar to the main dispatcher; if the dispatcher
crashes (data symlink wipe, NCCL hang, etc.) the keep-alive keeps the rig
warm long enough for you to notice and recover:

    nohup uv run python scripts/gpu_keepalive.py --util 0.3 > keepalive.log 2>&1 & disown

CUDA_VISIBLE_DEVICES is honored, so you can limit to specific GPUs by
exporting it before launch.
"""
from __future__ import annotations

import argparse
import signal
import sys
import time

import torch


def _parse_gpus(spec: str) -> list[int]:
    if spec == "all":
        return list(range(torch.cuda.device_count()))
    return [int(x) for x in spec.split(",") if x.strip()]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--util", type=float, default=0.3,
                   help="Target duty cycle (fraction of period spent on GPU work). Default 0.3.")
    p.add_argument("--period", type=float, default=10.0,
                   help="Duty-cycle period in seconds (NVML sampling window ~1s, so 10s averages well). Default 10.")
    p.add_argument("--gpus", default="all",
                   help="Comma list of GPU indices, or 'all'. Default 'all'.")
    p.add_argument("--matrix-size", type=int, default=4096,
                   help="N for the N×N fp16 matmul. Bigger = more SM occupancy. Default 4096.")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress periodic heartbeat to stdout.")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("[keepalive] CUDA not available — exiting.", file=sys.stderr)
        return 1

    if not (0.0 < args.util <= 1.0):
        print(f"[keepalive] --util must be in (0,1], got {args.util}", file=sys.stderr)
        return 2

    gpu_ids = _parse_gpus(args.gpus)
    if not gpu_ids:
        print("[keepalive] no GPUs selected — exiting.", file=sys.stderr)
        return 3

    # Allocate two persistent matrices per GPU so we don't churn the allocator.
    tensors: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for g in gpu_ids:
        torch.cuda.set_device(g)
        a = torch.randn(args.matrix_size, args.matrix_size, device=f"cuda:{g}", dtype=torch.float16)
        b = torch.randn(args.matrix_size, args.matrix_size, device=f"cuda:{g}", dtype=torch.float16)
        tensors[g] = (a, b)

    work_s = args.util * args.period
    sleep_s = args.period - work_s
    mem_per_gpu_mib = 2 * (args.matrix_size ** 2) * 2 / (1024 ** 2)
    print(
        f"[keepalive] gpus={gpu_ids} matrix={args.matrix_size} "
        f"util={args.util:.2f} period={args.period:.1f}s "
        f"work={work_s:.2f}s sleep={sleep_s:.2f}s "
        f"mem~{mem_per_gpu_mib:.0f}MiB/gpu",
        flush=True,
    )

    # Graceful shutdown on SIGTERM/SIGINT
    stop = {"flag": False}

    def _handle(signum, _frame):
        print(f"[keepalive] received signal {signum}, exiting cleanly.", flush=True)
        stop["flag"] = True

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)

    cycles = 0
    t_start_wall = time.time()
    while not stop["flag"]:
        # Burst: matmuls on each GPU for `work_s` seconds.
        t0 = time.time()
        while (not stop["flag"]) and (time.time() - t0 < work_s):
            for g in gpu_ids:
                a, b = tensors[g]
                # In-place + reassign to avoid a Python-side ref to a new
                # buffer each iter (allocator stays cool).
                _ = torch.matmul(a, b)
            # Force completion so the next sleep is actually idle.
            for g in gpu_ids:
                torch.cuda.synchronize(g)
        if stop["flag"]:
            break
        # Idle: sleep for the remainder of the period.
        remaining = sleep_s
        while remaining > 0 and not stop["flag"]:
            chunk = min(remaining, 0.5)  # wake periodically to check signal
            time.sleep(chunk)
            remaining -= chunk
        cycles += 1
        if (not args.quiet) and (cycles % 6 == 0):  # heartbeat every ~minute
            up = time.time() - t_start_wall
            print(f"[keepalive] alive {up/60:.1f} min, {cycles} cycles", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
