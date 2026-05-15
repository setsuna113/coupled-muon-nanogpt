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

    bases: [configs/ladder/B_gelu2mat.yaml, configs/ladder/C_learnedpos.yaml, ...]
    grid:
      ...

Sweep YAML format with tied keys (constrained Cartesian product):

    tie:
      - [optimizer.coupled_steps, optimizer.ns_steps]
    grid:
      optimizer.coupled_steps: [3, 5, 8]
      optimizer.ns_steps:      [3, 5, 8]
      ...

Keys listed inside a single `tie:` group are **zipped** rather than
Cartesian-producted — so the example above yields three (3,3)/(5,5)/(8,8)
combinations, not nine. The S1 NS-coefficient sweep relies on this to
keep `coupled_steps == ns_steps`. All keys inside a tie group must (a)
appear in `grid:` and (b) have the same number of values; otherwise
`expand()` raises `ValueError`.

Emits a JSONL file where each line is one resolved (base, overrides, seed)
triple, ready for the launcher to consume.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from omegaconf import OmegaConf


def _build_axes(
    grid: dict,
    tie_groups: list[list[str]],
    sweep_path: str | Path,
) -> tuple[list[list[str]], list[list[tuple]]]:
    """Split `grid` into Cartesian axes, zipping each `tie` group into one axis.

    Returns parallel lists ``(keys_per_axis, values_per_axis)`` where each
    axis emits a tuple matching its keys when iterated.
    """
    consumed: set[str] = set()
    keys_per_axis: list[list[str]] = []
    values_per_axis: list[list[tuple]] = []

    for tie in tie_groups:
        if not tie:
            raise ValueError(
                f"empty `tie:` group in {sweep_path}; each tie must list ≥ 2 keys"
            )
        for k in tie:
            if k not in grid:
                raise ValueError(
                    f"`tie:` key {k!r} is not in `grid` ({sweep_path}); "
                    f"every tied key must have a grid entry"
                )
            if k in consumed:
                raise ValueError(
                    f"`tie:` key {k!r} appears in multiple tie groups ({sweep_path})"
                )
        tied_values = [list(grid[k]) for k in tie]
        lengths = {len(v) for v in tied_values}
        if len(lengths) != 1:
            raise ValueError(
                f"`tie:` keys {tie} have mismatched value lengths "
                f"{[len(v) for v in tied_values]} in {sweep_path}; "
                f"tied keys must be zip-compatible"
            )
        consumed.update(tie)
        keys_per_axis.append(list(tie))
        values_per_axis.append(list(zip(*tied_values, strict=True)))

    for k, v in grid.items():
        if k in consumed:
            continue
        keys_per_axis.append([k])
        values_per_axis.append([(item,) for item in v])

    return keys_per_axis, values_per_axis


def expand(sweep_path: str | Path) -> list[dict]:
    s = OmegaConf.load(sweep_path)
    # resolve=False so OmegaConf interpolations like ${name} or ${optimizer.lr}
    # in `fixed` (used by sweep YAMLs to set wandb_group per ablation cell)
    # survive expansion as literal strings; train.py resolves them after
    # merging with the base config.
    grid = OmegaConf.to_container(s.grid, resolve=False) if "grid" in s else {}
    fixed = OmegaConf.to_container(s.fixed, resolve=False) if "fixed" in s else {}
    tie_groups = OmegaConf.to_container(s.tie, resolve=False) if "tie" in s else []
    if "bases" in s:
        bases = [str(b) for b in OmegaConf.to_container(s.bases, resolve=True)]
    elif "base" in s:
        bases = [str(s.base)]
    else:
        raise ValueError(f"sweep YAML must define `base:` or `bases:` ({sweep_path})")

    keys_per_axis, values_per_axis = _build_axes(grid, tie_groups, sweep_path)
    out = []
    for base in bases:
        for combo in itertools.product(*values_per_axis):
            overrides: dict = {}
            for axis_keys, axis_values in zip(keys_per_axis, combo, strict=True):
                for k, v in zip(axis_keys, axis_values, strict=True):
                    overrides[k] = v
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
