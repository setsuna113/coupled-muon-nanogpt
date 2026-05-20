#!/usr/bin/env python3
"""Diagnose why `filter_unfinished_jobs.py` flags finished cells as missing.

`run_id` (utils.py) hashes the *entire resolved config*. If any config input
drifted since the cells were trained, every recomputed hash misses its
on-disk dir and a resume becomes a full rerun.

For each job in <results-dir>/_all.jsonl this script:
  1. recomputes run_id and checks for an on-disk match;
  2. for every miss, finds the closest sibling dir (same name+seed, fewest
     config.yaml differences) and records which keys differ;
  3. prints an aggregate histogram of drifted keys across all misses, so the
     single responsible input (a base.yaml key, an LR grid, ...) is obvious.

It also reports *which* base.yaml `load_config` actually reads — decisive
when an editable-vs-installed package makes a `git checkout` look ineffective.

Usage:
    uv run python scripts/diagnose_resume_hash.py --results-dir $RDIR
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from coupled_muon_nanogpt import train as train_mod
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

    # Which base.yaml does load_config actually read? (editable vs installed)
    base_yaml = (Path(train_mod.__file__).resolve().parent.parent.parent
                 / "configs" / "base.yaml")
    qk_in_base = (base_yaml.exists()
                  and any("qk_coupling" in ln
                          for ln in base_yaml.read_text().splitlines()))
    print(f"train.py         : {train_mod.__file__}")
    print(f"load_config reads: {base_yaml}  (exists={base_yaml.exists()})")
    print(f"  -> still has qk_coupling block: {qk_in_base}")
    print(f"results-dir      : {rdir}\n")

    from omegaconf import OmegaConf
    jobs = [json.loads(ln) for ln in open(jobs_path) if ln.strip()]

    matched = no_sibling = 0
    drift_keys: Counter = Counter()
    examples: dict[str, tuple] = {}

    for job in jobs:
        base = repo / job["base"]
        ov = [f"{k}={v}" for k, v in job["overrides"].items()]
        cfg = load_config(str(base), ov + [f"run.output_dir={rdir}"])
        rid = run_id(cfg, int(job["seed"]))
        if (rdir / rid).exists():
            matched += 1
            continue

        fresh = _to_jsonable(cfg)
        name = rid.rsplit("-s", 1)[0]
        seed = job["seed"]
        best = None
        for s in sorted(rdir.glob(f"{name}-s{seed}-*")):
            cy = s / "config.yaml"
            if not cy.exists():
                continue
            stored = OmegaConf.to_container(OmegaConf.load(cy), resolve=True)
            d = diff(stored, fresh)
            if best is None or len(d) < len(best[1]):
                best = (s, d)
        if best is None:
            no_sibling += 1
            continue
        for path, sv, fv in best[1]:
            drift_keys[path] += 1
            examples.setdefault(path, (best[0].name, sv, fv))

    n_miss = len(jobs) - matched
    print(f"jobs={len(jobs)}  matched={matched}  missing={n_miss}  "
          f"(of missing: {no_sibling} have no sibling dir at all)\n")

    if n_miss == 0:
        print("=> All cells matched. Nothing to diagnose.")
        return

    print("Drifted keys across the missing cells "
          "(count : key  — how many missing cells differ there):")
    for path, cnt in drift_keys.most_common():
        ex_dir, sv, fv = examples[path]
        print(f"  [{cnt:3d}/{n_miss}] {path}")
        print(f"            e.g. stored({ex_dir})={sv!r}  fresh={fv!r}")

    print()
    universal = [k for k, c in drift_keys.items() if c == n_miss]
    if universal:
        print(f"=> {len(universal)} key(s) differ in EVERY missing cell:")
        for k in universal:
            print(f"     {k}")
        print("   That is the drifted input. Restore it to the value the cells")
        print("   were trained with, then re-run filter_unfinished_jobs.py.")
    else:
        print("=> No single universal key — drift is heterogeneous; read the")
        print("   per-key counts above (likely a sweep-grid redefinition).")


if __name__ == "__main__":
    main()
