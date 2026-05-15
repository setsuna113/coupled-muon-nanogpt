#!/usr/bin/env python3
"""Generate Rig B pt3 jobs from completed pt1B results.

pt1B runs the production MoE tier:
  I, I' x {muon, coupled_muon_v2} x {3e-3, 1e-2, 3e-2} x 5 seeds.

pt3 is adaptive. For each rung/optimizer arm, this script reads completed
pt1B metrics, chooses the median-best LR, and then:
  - if the best LR is at an edge, adds the missing outside LR for that arm;
  - otherwise, spends the remaining budget on extra seeds at the best LR.

The emitted JSONL is ready for scripts/run_sweep.py and contains explicit
optimizer.lr overrides. No LR is inferred by the launcher itself.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


PT1_LRS = (3.0e-3, 1.0e-2, 3.0e-2)
LOW_EDGE_LR = 1.0e-3
HIGH_EDGE_LR = 1.0e-1
PT3_PROJECT = "coupled-muon-moe-pt3-telescope"

BASE_BY_NAME = {
    "ladder_I_moe": "configs/ladder/I_moe.yaml",
    "ladder_I_prime_relu2_moe": "configs/ladder/I_prime_relu2_moe.yaml",
}

ARM_ORDER = (
    ("ladder_I_moe", "coupled_muon_v2"),
    ("ladder_I_moe", "muon"),
    ("ladder_I_prime_relu2_moe", "coupled_muon_v2"),
    ("ladder_I_prime_relu2_moe", "muon"),
)


@dataclass(frozen=True)
class Run:
    name: str
    opt: str
    lr: float
    seed: int
    val_loss: float
    run_id: str


@dataclass(frozen=True)
class ArmDecision:
    name: str
    opt: str
    best_lr: float
    best_median: float
    second_median: float
    edge_lr: float | None

    @property
    def edge_strength(self) -> float:
        if self.edge_lr is None:
            return float("-inf")
        return self.second_median - self.best_median


def _seed_from_run_id(run_id: str) -> int:
    m = re.search(r"-s(\d+)-[0-9a-f]{12}$", run_id)
    if not m:
        raise ValueError(f"cannot parse seed from run_id={run_id!r}")
    return int(m.group(1))


def _has_saved_ckpt(run_dir: Path) -> bool:
    stdout = run_dir / "stdout.log"
    if not stdout.exists():
        return False
    with stdout.open(errors="replace") as f:
        return any(line.startswith("[saved_ckpt]") for line in f)


def _last_val_loss(metrics_path: Path) -> float:
    last: float | None = None
    with metrics_path.open(errors="replace") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("event") == "val":
                last = float(row["val_loss"])
    if last is None or not math.isfinite(last):
        raise ValueError(f"no finite final val_loss in {metrics_path}")
    return last


def read_completed_runs(pt1_dir: Path) -> list[Run]:
    runs: list[Run] = []
    for run_dir in sorted(pt1_dir.iterdir()):
        if not run_dir.is_dir() or not _has_saved_ckpt(run_dir):
            continue
        cfg_path = run_dir / "config.yaml"
        metrics_path = run_dir / "metrics.jsonl"
        if not cfg_path.exists() or not metrics_path.exists():
            continue
        cfg = OmegaConf.load(cfg_path)
        name = str(cfg.name)
        opt = str(cfg.optimizer.type)
        lr = float(cfg.optimizer.lr)
        if name not in BASE_BY_NAME or opt not in {"muon", "coupled_muon_v2"}:
            continue
        runs.append(
            Run(
                name=name,
                opt=opt,
                lr=lr,
                seed=_seed_from_run_id(run_dir.name),
                val_loss=_last_val_loss(metrics_path),
                run_id=run_dir.name,
            )
        )
    return runs


def decide_arms(runs: list[Run], *, expected_seeds_per_lr: int) -> list[ArmDecision]:
    by_arm_lr: dict[tuple[str, str, float], list[Run]] = {}
    for run in runs:
        by_arm_lr.setdefault((run.name, run.opt, run.lr), []).append(run)

    decisions: list[ArmDecision] = []
    errors: list[str] = []
    for name, opt in ARM_ORDER:
        medians: dict[float, float] = {}
        for lr in PT1_LRS:
            group = by_arm_lr.get((name, opt, lr), [])
            seeds = sorted(r.seed for r in group)
            if len(group) != expected_seeds_per_lr:
                errors.append(
                    f"{name}/{opt}/lr={lr:.0e}: expected "
                    f"{expected_seeds_per_lr} seeds, found {len(group)} {seeds}"
                )
                continue
            medians[lr] = statistics.median(r.val_loss for r in group)

        if len(medians) != len(PT1_LRS):
            continue
        ranked = sorted(medians.items(), key=lambda kv: kv[1])
        best_lr, best_median = ranked[0]
        second_median = ranked[1][1]
        edge_lr = None
        if math.isclose(best_lr, min(PT1_LRS)):
            edge_lr = LOW_EDGE_LR
        elif math.isclose(best_lr, max(PT1_LRS)):
            edge_lr = HIGH_EDGE_LR
        decisions.append(
            ArmDecision(
                name=name,
                opt=opt,
                best_lr=best_lr,
                best_median=best_median,
                second_median=second_median,
                edge_lr=edge_lr,
            )
        )

    if errors:
        joined = "\n  ".join(errors)
        raise SystemExit(f"pt1B is not complete/consistent enough for pt3:\n  {joined}")
    return decisions


def _job(name: str, opt: str, lr: float, seed: int) -> dict[str, Any]:
    return {
        "base": BASE_BY_NAME[name],
        "overrides": {
            "run.wandb_project": PT3_PROJECT,
            "optimizer.type": opt,
            "optimizer.lr": lr,
        },
        "seed": seed,
    }


def build_jobs(decisions: list[ArmDecision], *, max_jobs: int) -> tuple[list[dict[str, Any]], list[str]]:
    jobs: list[dict[str, Any]] = []
    notes: list[str] = []

    edge_arms = sorted(
        [d for d in decisions if d.edge_lr is not None],
        key=lambda d: d.edge_strength,
        reverse=True,
    )

    if edge_arms:
        notes.append("mode=edge_telescope")
        for seed in range(5):
            for d in edge_arms:
                if len(jobs) >= max_jobs:
                    notes.append(f"max_jobs={max_jobs} reached while adding edge telescope cells")
                    return jobs, notes
                assert d.edge_lr is not None
                jobs.append(_job(d.name, d.opt, d.edge_lr, seed))
                notes.append(
                    f"edge {d.name}/{d.opt}: pt1 best lr={d.best_lr:.0e}, "
                    f"adding lr={d.edge_lr:.0e}, seed={seed}"
                )

    if len(jobs) < max_jobs:
        notes.append("mode=bonus_seeds" if not edge_arms else "fill=bonus_seeds")
        for seed in range(5, 50):
            for d in decisions:
                if len(jobs) >= max_jobs:
                    return jobs, notes
                jobs.append(_job(d.name, d.opt, d.best_lr, seed))
                notes.append(
                    f"bonus {d.name}/{d.opt}: best lr={d.best_lr:.0e}, seed={seed}"
                )
    return jobs, notes


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pt1-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--expected-completed", type=int, default=60)
    p.add_argument("--expected-seeds-per-lr", type=int, default=5)
    p.add_argument("--max-jobs", type=int, default=19)
    args = p.parse_args()

    runs = read_completed_runs(args.pt1_dir)
    if len(runs) != args.expected_completed:
        raise SystemExit(
            f"expected {args.expected_completed} completed pt1B runs, found {len(runs)} "
            f"under {args.pt1_dir}"
        )

    decisions = decide_arms(runs, expected_seeds_per_lr=args.expected_seeds_per_lr)
    jobs, notes = build_jobs(decisions, max_jobs=args.max_jobs)
    if not jobs:
        raise SystemExit("generated zero pt3 jobs")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for job in jobs:
            f.write(json.dumps(job, sort_keys=True) + "\n")

    manifest_path = args.manifest or args.out.with_suffix(args.out.suffix + ".manifest")
    with manifest_path.open("w") as f:
        f.write(f"pt1_dir: {args.pt1_dir}\n")
        f.write(f"out: {args.out}\n")
        f.write(f"jobs: {len(jobs)}\n")
        f.write("\narm_decisions:\n")
        for d in decisions:
            edge = "none" if d.edge_lr is None else f"{d.edge_lr:.8g}"
            f.write(
                f"- {d.name}/{d.opt}: best_lr={d.best_lr:.8g}, "
                f"median={d.best_median:.8g}, second={d.second_median:.8g}, "
                f"edge_lr={edge}\n"
            )
        f.write("\njob_notes:\n")
        for note in notes:
            f.write(f"- {note}\n")

    print(f"Wrote {len(jobs)} pt3 jobs to {args.out}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
