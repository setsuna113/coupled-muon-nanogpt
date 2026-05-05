"""Sweep YAML → list of resolved run configs.

Sweep YAML format:

    base: configs/ladder/A0_llama60m.yaml
    grid:
      optimizer.type: [adamw, muon, coupled_muon_v2]
      optimizer.lr: [1e-3, 3e-3, 1e-2]
      seed: [0, 1, 2]
    fixed:
      run.wandb_project: "coupled-muon-ladder"

Emits a JSONL file where each line is one resolved (config_overrides, seed)
combination, ready for the launcher to consume.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from omegaconf import OmegaConf


def expand(sweep_path: str | Path) -> list[dict]:
    s = OmegaConf.load(sweep_path)
    grid = OmegaConf.to_container(s.get("grid", {}), resolve=True) or {}
    fixed = OmegaConf.to_container(s.get("fixed", {}), resolve=True) or {}
    base = str(s.base)

    keys = list(grid.keys())
    values = [list(v) for v in grid.values()]
    out = []
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
