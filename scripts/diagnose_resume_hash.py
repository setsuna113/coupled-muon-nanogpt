#!/usr/bin/env python3
"""Diagnose why `filter_unfinished_jobs.py` flagged finished cells as missing.

`run_id` (utils.py) hashes the *entire resolved config*, including
`run.output_dir`. `run_sweep.py` feeds `train.py` the raw `--results-dir`
string; `filter_unfinished_jobs.py` feeds the symlink-*resolved* path. If the
results dir crosses a symlink, those differ and every recomputed hash misses
the on-disk dir — so a clean resume looks like a full rerun.

This script recomputes each job's `run_id` both ways and reports how many
match a directory on disk. If neither form explains the miss, it diffs a
freshly-resolved config against the `config.yaml` that `train.py` persisted
in an existing finished cell, so the drifted key is obvious.

Usage:
    uv run python scripts/diagnose_resume_hash.py --results-dir $RDIR
    uv run python scripts/diagnose_resume_hash.py --results-dir $RDIR --jobs $RDIR/_all.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from coupled_muon_nanogpt.train import load_config
from coupled_muon_nanogpt.utils import _to_jsonable, run_id


def diff(a, b, path=""):
    """Recursive leaf-level diff between two jsonable structures."""
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            out += diff(a.get(k, "<MISSING>"), b.get(k, "<MISSING>"),
                        f"{path}.{k}" if path else str(k))
    elif a != b:
        out.append((path, a, b))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", required=True, type=Path)
    p.add_argument("--jobs", type=Path, default=None,
                   help="JSONL of jobs (default: <results-dir>/_all.jsonl)")
    args = p.parse_args()

    repo = Path(__file__).resolve().parent.parent
    rdir = args.results_dir
    jobs_path = args.jobs or (rdir / "_all.jsonl")
    raw = str(rdir)
    res = str(rdir.resolve())

    print(f"raw      : {raw}")
    print(f"resolved : {res}")
    print(f"identical: {raw == res}\n")

    jobs = [json.loads(line) for line in open(jobs_path) if line.strip()]
    hit = {"raw": 0, "res": 0}
    first_miss = None

    for job in jobs:
        base = repo / job["base"]
        ov = [f"{k}={v}" for k, v in job["overrides"].items()]
        rids = {}
        for tag, od in (("raw", raw), ("res", res)):
            cfg = load_config(str(base), ov + [f"run.output_dir={od}"])
            rids[tag] = run_id(cfg, int(job["seed"]))
            if (rdir / rids[tag]).exists():
                hit[tag] += 1
        if (first_miss is None
                and not (rdir / rids["raw"]).exists()
                and not (rdir / rids["res"]).exists()):
            first_miss = (job, rids)

    print(f"jobs={len(jobs)}  match_with_raw={hit['raw']}  "
          f"match_with_resolved={hit['res']}\n")

    if hit["raw"] == len(jobs):
        print("=> CAUSE: the filter's .resolve() breaks the hash.")
        print("   Re-run filter_unfinished_jobs.py with the raw path forced:")
        print(f'   --extra-override "run.output_dir={raw}"')
        return
    if hit["res"] == len(jobs):
        print("=> All cells match the RESOLVED path — filter logic is consistent.")
        print("   The miss is elsewhere; inspect the filter output paths.")
        return

    print("=> Neither output_dir form matches all cells. Probing for config drift...")
    if first_miss is None:
        print("   (no fully-missing job; partial mismatch — inspect manually)")
        return

    job, rids = first_miss
    name_prefix = rids["raw"].rsplit("-s", 1)[0]
    seed = job["seed"]
    cands = sorted(rdir.glob(f"{name_prefix}-s{seed}-*"))
    print(f"   probe job : base={job['base']} seed={seed}")
    print(f"   computed run_id (raw output_dir) : {rids['raw']}")
    print(f"   on-disk dirs {name_prefix}-s{seed}-* : {[c.name for c in cands]}")
    if not cands:
        print("   no dir with this name+seed -> sweep was REDEFINED (genuinely new cells).")
        return

    stored_path = cands[0] / "config.yaml"
    if not stored_path.exists():
        print(f"   {stored_path} missing -> cannot diff.")
        return

    from omegaconf import OmegaConf
    stored = OmegaConf.to_container(OmegaConf.load(stored_path), resolve=True)
    base = repo / job["base"]
    ov = [f"{k}={v}" for k, v in job["overrides"].items()]
    fresh = _to_jsonable(load_config(str(base), ov + [f"run.output_dir={raw}"]))
    deltas = diff(stored, fresh)
    print(f"\n   diff  stored({cands[0].name}/config.yaml)  vs  freshly-resolved:")
    if not deltas:
        print("     (no differences — run_id should match; check the name field)")
    for path, sv, fv in deltas:
        print(f"     {path}:  stored={sv!r}  fresh={fv!r}")


if __name__ == "__main__":
    main()
