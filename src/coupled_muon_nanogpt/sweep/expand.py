"""Sweep YAML → list of resolved run configs.

Sweep YAML format (single base):

    base: configs/ladder/A0_llama60m.yaml
    grid:
      optimizer.type: [adamw, muon, coupled_muon_v2]
      optimizer.lr: [1e-3, 3e-3, 1e-2]
      seed: [0, 1, 2]
    fixed:
      run.wandb_project: "coupled-muon-ladder"

Sweep YAML format (multiple bases — for the d.3 cross-rung sweep):

    bases: [configs/ladder/A1_llama125m.yaml, configs/ladder/B_gelu2mat_125m.yaml, ...]
    grid:
      ...

Emits a JSONL file where each line is one resolved (base, overrides, seed)
triple, ready for the launcher to consume.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from omegaconf import OmegaConf


def expand(sweep_path: str | Path) -> list[dict]:
    s = OmegaConf.load(sweep_path)
    # resolve=False so OmegaConf interpolations like ${name} or ${optimizer.lr}
    # in `fixed` (used by sweep YAMLs to set wandb_group per ablation cell)
    # survive expansion as literal strings; train.py resolves them after
    # merging with the base config.
    grid = OmegaConf.to_container(s.get("grid", {}), resolve=False) or {}
    fixed = OmegaConf.to_container(s.get("fixed", {}), resolve=False) or {}
    if "bases" in s:
        bases = [str(b) for b in OmegaConf.to_container(s.bases, resolve=True)]
    elif "base" in s:
        bases = [str(s.base)]
    else:
        raise ValueError(f"sweep YAML must define `base:` or `bases:` ({sweep_path})")

    keys = list(grid.keys())
    values = [list(v) for v in grid.values()]
    out = []
    for base in bases:
        for combo in itertools.product(*values):
            overrides = dict(zip(keys, combo, strict=True))
            seed = int(overrides.pop("seed", 0))
            merged = {**fixed, **overrides}
            out.append({"base": base, "overrides": merged, "seed": seed})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("sweep")
    p.add_argument("--out", type=str, default="sweep_jobs.jsonl")
    args = p.parse_args()
    jobs = expand(args.sweep)
    with open(args.out, "w") as f:
        for j in jobs:
            f.write(json.dumps(j) + "\n")
    print(f"Wrote {len(jobs)} jobs to {args.out}")


if __name__ == "__main__":
    main()
