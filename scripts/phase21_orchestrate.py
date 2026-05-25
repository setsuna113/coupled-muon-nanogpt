#!/usr/bin/env python3
"""Three-machine Phase-2.1 chain orchestrator.

Each physical node runs this script with its role. The script coordinates via
small lock/done files under ``$GLOBAL/phase2.1-control`` and always filters a
JSONL against the target results directory before launching ``run_sweep.py``.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence


GLOBAL_DEFAULT = Path("/inspire/hdd/global_user/yanjunchi-24040/yancheng")
CONTROL_NAME = "phase2.1-control"

RESULT_DIRS = {
    "a1": "phase2.1-rigA-a1-adamw-I-topup",
    "a3": "phase2.1-rigA-a3-kcurve-I",
    "b2_no_rope": "phase2.1-rigB-b2-no-rope-confirm",
    "b2_partial_rope": "phase2.1-rigB-b2-partial-rope-confirm",
    "d1": "phase2.1-rigA-d1-mla",
    "d3": "phase2.1-rigB-d3-lora-rank",
    "d4": "phase2.1-rigC-d4-factff",
    "e1": "phase2.1-rigA-e1-mla-350m",
}

EXPECTED = {
    "a1": 10,
    "a3": 9,
    "b2_no_rope": 10,
    "b2_partial_rope": 5,
    "d1": 45,
    "d3": 30,
    "d4": 12,
    "e1": 30,
}

D1_SHARDS = {
    "h100": (0, 1, 2, 3),
    "h200x2": (4,),
    "h200x4": (5, 6),
}

RUN_ARGS = {
    "h100": {
        "b2_no_rope": (2, 4, 2),
        "d1": (2, 4, 2),
        "d3": (2, 4, 2),
    },
    "h200x2": {
        "b2_partial_rope": (2, 1, 2),
        "d1": (2, 1, 2),
        "d4": (2, 1, 2),
    },
    "h200x4": {
        "a1": (2, 2, 2),
        "a3": (2, 2, 2),
        "d1": (2, 2, 2),
        "e1": (4, 1, 4),
    },
}

JOB_FILES = {
    "a1": "p21_a1_adamw_I.jsonl",
    "a3": "p21_a3_kcurve_I.jsonl",
    "b2_no_rope": "p21_b2_no_rope.jsonl",
    "b2_partial_rope": "p21_b2_partial_rope.jsonl",
    "d1": "p21_d1_mla.jsonl",
    "d3": "p21_d3_lora_rank.jsonl",
    "d4": "p21_d4_factff.jsonl",
    "e1": "p21_e1_mla_350m.jsonl",
}


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %z")


def log(message: str) -> None:
    print(f"[{timestamp()}] {message}", flush=True)


def count_done(results_dir: Path) -> int:
    if not results_dir.exists():
        return 0
    total = 0
    for stdout_log in results_dir.glob("*/stdout.log"):
        try:
            with stdout_log.open(errors="replace") as f:
                if any("[saved_ckpt]" in line for line in f):
                    total += 1
        except OSError:
            continue
    return total


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open() as f:
        return sum(1 for line in f if line.strip())


class Orchestrator:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.role = args.role
        self.repo_root = args.repo_root.resolve()
        self.global_root = args.global_root.resolve()
        self.control_root = args.control_root.resolve() if args.control_root else self.global_root / CONTROL_NAME
        self.jobs_dir = self.control_root / "jobs"
        self.locks_dir = self.control_root / "locks"
        self.done_dir = self.control_root / "done"
        self.logs_dir = self.control_root / "logs"

    def result_dir(self, key: str) -> Path:
        return self.global_root / RESULT_DIRS[key]

    def job_path(self, key: str) -> Path:
        return self.jobs_dir / JOB_FILES[key]

    def done_file(self, key: str) -> Path:
        return self.done_dir / f"{key}.done"

    def setup(self) -> None:
        self.control_root.mkdir(parents=True, exist_ok=True)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.locks_dir.mkdir(parents=True, exist_ok=True)
        self.done_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(self.repo_root)
        self._load_env()
        if not self.args.skip_git_pull:
            self.run(["git", "pull"], label="git pull")

    def _load_env(self) -> None:
        env_sh = self.args.env_sh
        if env_sh is None:
            for candidate in (self.repo_root / "../my_env.sh", self.repo_root / "my_env.sh"):
                if candidate.exists():
                    env_sh = candidate
                    break
        if env_sh is None:
            log("No env file sourced; set --env-sh if this node needs one.")
            return
        env_sh = env_sh.expanduser().resolve()
        if not env_sh.exists():
            log(f"Env file not found: {env_sh}; continuing with current environment.")
            return
        cmd = ["bash", "-lc", f"set -a; source {shlex_quote(str(env_sh))}; env -0"]
        proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE)
        for item in proc.stdout.split(b"\0"):
            if not item or b"=" not in item:
                continue
            key, value = item.split(b"=", 1)
            os.environ[key.decode()] = value.decode(errors="replace")
        log(f"Sourced env from {env_sh}")

    @contextmanager
    def lock(self, name: str) -> Iterator[None]:
        path = self.locks_dir / f"{name}.lock"
        while True:
            try:
                path.mkdir()
                (path / "holder.txt").write_text(f"role={self.role}\npid={os.getpid()}\ntime={timestamp()}\n")
                break
            except FileExistsError:
                age = time.time() - path.stat().st_mtime
                if age > self.args.lock_stale_seconds:
                    log(f"Removing stale lock {path} (age {age:.0f}s)")
                    shutil.rmtree(path, ignore_errors=True)
                    continue
                log(f"Waiting for lock {path}")
                time.sleep(self.args.sleep_seconds)
        try:
            yield
        finally:
            shutil.rmtree(path, ignore_errors=True)

    def run(self, cmd: Sequence[str], *, label: str) -> None:
        printable = " ".join(shlex_quote(part) for part in cmd)
        log(f"{label}: {printable}")
        if self.args.dry_run:
            return
        env = {**os.environ, "WANDB_MODE": os.environ.get("WANDB_MODE", "offline")}
        repo_src = str(self.repo_root / "src")
        env["PYTHONPATH"] = (
            repo_src
            if not env.get("PYTHONPATH")
            else repo_src + os.pathsep + env["PYTHONPATH"]
        )
        subprocess.run(cmd, cwd=self.repo_root, env=env, check=True)

    def mark_done(self, key: str, *, extra: str = "") -> None:
        self.done_dir.mkdir(parents=True, exist_ok=True)
        body = f"role={self.role}\ntime={timestamp()}\n"
        if extra:
            body += extra.rstrip() + "\n"
        self.done_file(key).write_text(body)
        log(f"Marked {key} done")

    def ensure_count_done(self, key: str) -> bool:
        done = count_done(self.result_dir(key))
        expected = EXPECTED[key]
        if done == expected:
            if not self.done_file(key).exists():
                self.mark_done(key, extra=f"completed={done}/{expected}")
            return True
        return False

    def wait_for_done(self, key: str) -> None:
        expected = EXPECTED[key]
        while True:
            if self.done_file(key).exists() or self.ensure_count_done(key):
                log(f"Gate satisfied: {key}")
                return
            done = count_done(self.result_dir(key))
            log(f"Waiting for {key}: {done}/{expected} completed")
            time.sleep(self.args.sleep_seconds)

    def phase21_script(self, filename: str) -> Path:
        sibling = Path(__file__).resolve().with_name(filename)
        if sibling.exists():
            return sibling
        return self.repo_root / "scripts" / filename

    def ensure_static_jobs(self) -> None:
        if self.job_path("a1").exists() and self.job_path("a3").exists():
            return
        with self.lock("static_jobs"):
            if self.job_path("a1").exists() and self.job_path("a3").exists():
                return
            self.run(
                [
                    sys.executable,
                    str(self.phase21_script("phase21_materialize.py")),
                    "--mode",
                    "static",
                    "--global-root",
                    str(self.global_root),
                    "--control-root",
                    str(self.control_root),
                ],
                label="materialize static A jobs",
            )

    def ensure_winners(self) -> None:
        if (self.control_root / "p21_winners.json").exists() and self.job_path("d1").exists():
            return
        with self.lock("winners"):
            if (self.control_root / "p21_winners.json").exists() and self.job_path("d1").exists():
                return
            self.run(
                [
                    sys.executable,
                    str(self.phase21_script("phase21_materialize.py")),
                    "--mode",
                    "full",
                    "--global-root",
                    str(self.global_root),
                    "--control-root",
                    str(self.control_root),
                ],
                label="materialize Phase-2.1 winners/jobs",
            )

    def run_filtered_sweep(self, key: str, *, job_path: Path | None = None, done_key: str | None = None) -> None:
        done_key = done_key or key
        nproc, workers, gpus = RUN_ARGS[self.role][key]
        source_jobs = job_path or self.job_path(key)
        if not source_jobs.exists():
            raise SystemExit(f"missing jobs file: {source_jobs}")
        remaining = self.jobs_dir / f"{source_jobs.stem}.{self.role}.remaining.jsonl"
        results_dir = self.result_dir(key)
        results_dir.mkdir(parents=True, exist_ok=True)
        self.run(
            [
                sys.executable,
                str(self.repo_root / "scripts" / "filter_unfinished_jobs.py"),
                "--in",
                str(source_jobs),
                "--out",
                str(remaining),
                "--results-dir",
                str(results_dir),
            ],
            label=f"filter {key}",
        )
        pending = count_jsonl(remaining)
        if pending == 0:
            log(f"{key}: no unfinished jobs in {source_jobs}")
            if done_key == key:
                done = count_done(results_dir)
                self.mark_done(done_key, extra=f"exact_jobs_complete=1 completed_dirs={done}/{EXPECTED[key]}")
            else:
                self.mark_done(done_key, extra="shard had no unfinished jobs")
            return
        self.run(
            [
                sys.executable,
                str(self.repo_root / "scripts" / "run_sweep.py"),
                str(remaining),
                "--nproc-per-node",
                str(nproc),
                "--num-workers",
                str(workers),
                "--gpus-per-worker",
                str(gpus),
                "--results-dir",
                str(results_dir),
            ],
            label=f"launch {key}",
        )
        if done_key == key:
            post_remaining = self.jobs_dir / f"{source_jobs.stem}.{self.role}.post.remaining.jsonl"
            self.run(
                [
                    sys.executable,
                    str(self.repo_root / "scripts" / "filter_unfinished_jobs.py"),
                    "--in",
                    str(source_jobs),
                    "--out",
                    str(post_remaining),
                    "--results-dir",
                    str(results_dir),
                ],
                label=f"verify {key}",
            )
            still_pending = count_jsonl(post_remaining)
            if still_pending:
                raise SystemExit(f"{key} launcher exited with {still_pending} jobs still unfinished")
            done = count_done(results_dir)
            self.mark_done(key, extra=f"exact_jobs_complete=1 completed_dirs={done}/{EXPECTED[key]}")
        else:
            post_remaining = self.jobs_dir / f"{source_jobs.stem}.{self.role}.post.remaining.jsonl"
            self.run(
                [
                    sys.executable,
                    str(self.repo_root / "scripts" / "filter_unfinished_jobs.py"),
                    "--in",
                    str(source_jobs),
                    "--out",
                    str(post_remaining),
                    "--results-dir",
                    str(results_dir),
                ],
                label=f"verify shard {done_key}",
            )
            still_pending = count_jsonl(post_remaining)
            if still_pending:
                raise SystemExit(f"{done_key} launcher exited with {still_pending} shard jobs still unfinished")
            self.mark_done(done_key, extra=f"shard_jobs={count_jsonl(source_jobs)}")

    def write_d1_shard(self) -> Path:
        full = self.job_path("d1")
        if not full.exists():
            raise SystemExit(f"missing D1 jobs file: {full}")
        shard = self.jobs_dir / f"p21_d1_mla.{self.role}.jsonl"
        residues = set(D1_SHARDS[self.role])
        jobs: list[dict] = []
        with full.open() as f:
            for index, line in enumerate(f):
                if line.strip() and index % 7 in residues:
                    jobs.append(json.loads(line))
        with shard.open("w") as f:
            for job in jobs:
                f.write(json.dumps(job, sort_keys=True) + "\n")
        log(f"Wrote D1 shard {shard}: residues={sorted(residues)} jobs={len(jobs)}")
        return shard

    def run_d1_shard(self) -> None:
        self.ensure_winners()
        self.wait_for_done("b2_no_rope")
        self.wait_for_done("b2_partial_rope")
        self.wait_for_done("a3")
        shard_key = f"d1_{self.role}"
        if self.done_file(shard_key).exists():
            log(f"D1 shard already marked done: {shard_key}")
            return
        shard = self.write_d1_shard()
        self.run_filtered_sweep("d1", job_path=shard, done_key=shard_key)
        self.ensure_d1_global_done()

    def ensure_d1_global_done(self) -> None:
        while True:
            done = count_done(self.result_dir("d1"))
            if done == EXPECTED["d1"]:
                if not self.done_file("d1").exists():
                    self.mark_done("d1", extra=f"completed={done}/{EXPECTED['d1']}")
                return
            shard_done = all(self.done_file(f"d1_{role}").exists() for role in D1_SHARDS)
            if shard_done:
                raise SystemExit(f"all D1 shards exited, but D1 completion is {done}/{EXPECTED['d1']}")
            log(f"Waiting for D1 global completion: {done}/{EXPECTED['d1']}")
            time.sleep(self.args.sleep_seconds)

    def ensure_d_gate(self) -> dict[str, object]:
        gate_path = self.control_root / "p21_d_gate.json"
        if gate_path.exists():
            return json.loads(gate_path.read_text())
        self.wait_for_done("d1")
        with self.lock("d_gate"):
            if gate_path.exists():
                return json.loads(gate_path.read_text())
            self.run(
                [
                    sys.executable,
                    str(self.phase21_script("phase21_materialize.py")),
                    "--mode",
                    "d-gate",
                    "--global-root",
                    str(self.global_root),
                    "--control-root",
                    str(self.control_root),
                    "--d1-dir",
                    str(self.result_dir("d1")),
                ],
                label="compute D-gate",
            )
        gate = json.loads(gate_path.read_text())
        done_name = "d_gate_positive" if gate.get("positive") else "d_gate_negative"
        if not self.done_file(done_name).exists():
            self.mark_done(done_name, extra=f"delta_p50={gate.get('delta_p50')} threshold={gate.get('threshold')}")
        return gate

    def conditional_if_positive(self, key: str) -> None:
        gate = self.ensure_d_gate()
        if not gate.get("positive"):
            log(f"D-gate negative; skipping {key}")
            return
        self.run_filtered_sweep(key)

    def run_h100(self) -> None:
        self.ensure_winners()
        self.run_filtered_sweep("b2_no_rope")
        self.run_d1_shard()
        self.conditional_if_positive("d3")

    def run_h200x2(self) -> None:
        self.ensure_winners()
        self.run_filtered_sweep("b2_partial_rope")
        self.run_d1_shard()
        self.conditional_if_positive("d4")

    def run_h200x4(self) -> None:
        self.ensure_static_jobs()
        self.run_filtered_sweep("a1")
        self.run_filtered_sweep("a3")
        self.run_d1_shard()
        self.conditional_if_positive("e1")

    def run_role(self) -> None:
        self.setup()
        log(f"Starting role={self.role} repo={self.repo_root} global={self.global_root}")
        if self.role == "h100":
            self.run_h100()
        elif self.role == "h200x2":
            self.run_h200x2()
        elif self.role == "h200x4":
            self.run_h200x4()
        else:
            raise SystemExit(f"unknown role: {self.role}")
        log(f"Role {self.role} complete")


def shlex_quote(value: str) -> str:
    import shlex

    return shlex.quote(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("h100", "h200x2", "h200x4"), required=True)
    parser.add_argument("--global-root", type=Path, default=GLOBAL_DEFAULT)
    parser.add_argument("--control-root", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--env-sh", type=Path, default=None)
    parser.set_defaults(skip_git_pull=True)
    parser.add_argument("--skip-git-pull", dest="skip_git_pull", action="store_true")
    parser.add_argument("--no-skip-git-pull", dest="skip_git_pull", action="store_false")
    parser.add_argument("--sleep-seconds", type=int, default=300)
    parser.add_argument("--lock-stale-seconds", type=int, default=6 * 3600)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    Orchestrator(parse_args()).run_role()


if __name__ == "__main__":
    main()
