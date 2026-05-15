"""Sweep expander tests covering Cartesian, tied, and error cases."""
from __future__ import annotations

from pathlib import Path

import pytest

from coupled_muon_nanogpt.sweep.expand import expand


def _write(tmp: Path, body: str) -> Path:
    p = tmp / "sweep.yaml"
    p.write_text(body)
    return p


def test_no_tie_reproduces_full_cartesian(tmp_path):
    sweep = _write(
        tmp_path,
        """
base: configs/ladder/A0_llama60m.yaml
grid:
  optimizer.type: [muon, coupled_muon_v2]
  optimizer.lr: [1e-3, 3e-3]
  seed: [0, 1]
""",
    )
    jobs = expand(sweep)
    assert len(jobs) == 2 * 2 * 2
    triples = {
        (j["overrides"]["optimizer.type"], j["overrides"]["optimizer.lr"], j["seed"])
        for j in jobs
    }
    assert len(triples) == 8


def test_tie_zips_listed_keys(tmp_path):
    sweep = _write(
        tmp_path,
        """
base: configs/ladder/A0_llama60m.yaml
tie:
  - [optimizer.coupled_steps, optimizer.ns_steps]
grid:
  optimizer.type: [muon, coupled_muon_v2]
  optimizer.coupled_steps: [3, 5, 8]
  optimizer.ns_steps:      [3, 5, 8]
  seed: [0, 1]
""",
    )
    jobs = expand(sweep)
    assert len(jobs) == 2 * 3 * 2
    for j in jobs:
        assert j["overrides"]["optimizer.coupled_steps"] == j["overrides"]["optimizer.ns_steps"]


def test_s1_yaml_expands_to_36_cells():
    repo = Path(__file__).resolve().parents[1]
    sweep = repo / "configs" / "sweeps" / "ns_coeff_K_sweep.yaml"
    jobs = expand(sweep)
    assert len(jobs) == 36, f"S1 sweep expected 36 cells, got {len(jobs)}"
    for j in jobs:
        assert j["overrides"]["optimizer.coupled_steps"] == j["overrides"]["optimizer.ns_steps"]


def test_tie_length_mismatch_raises(tmp_path):
    sweep = _write(
        tmp_path,
        """
base: configs/ladder/A0_llama60m.yaml
tie:
  - [optimizer.coupled_steps, optimizer.ns_steps]
grid:
  optimizer.coupled_steps: [3, 5, 8]
  optimizer.ns_steps:      [3, 5]
""",
    )
    with pytest.raises(ValueError, match="mismatched value lengths"):
        expand(sweep)


def test_tie_key_missing_from_grid_raises(tmp_path):
    sweep = _write(
        tmp_path,
        """
base: configs/ladder/A0_llama60m.yaml
tie:
  - [optimizer.coupled_steps, optimizer.ns_steps]
grid:
  optimizer.coupled_steps: [3, 5, 8]
""",
    )
    with pytest.raises(ValueError, match="not in `grid`"):
        expand(sweep)


def test_tie_key_in_two_groups_raises(tmp_path):
    sweep = _write(
        tmp_path,
        """
base: configs/ladder/A0_llama60m.yaml
tie:
  - [optimizer.coupled_steps, optimizer.ns_steps]
  - [optimizer.ns_steps, seed]
grid:
  optimizer.coupled_steps: [3, 5]
  optimizer.ns_steps:      [3, 5]
  seed: [0, 1]
""",
    )
    with pytest.raises(ValueError, match="multiple tie groups"):
        expand(sweep)


def test_fixed_passthrough_unchanged(tmp_path):
    sweep = _write(
        tmp_path,
        """
base: configs/ladder/A0_llama60m.yaml
grid:
  optimizer.lr: [1e-3, 3e-3]
fixed:
  run.wandb_project: my-proj
  run.wandb_group: "${name}/lr${optimizer.lr}"
""",
    )
    jobs = expand(sweep)
    assert len(jobs) == 2
    for j in jobs:
        assert j["overrides"]["run.wandb_project"] == "my-proj"
        # OmegaConf interpolation must survive expansion as a literal string.
        assert j["overrides"]["run.wandb_group"] == "${name}/lr${optimizer.lr}"
