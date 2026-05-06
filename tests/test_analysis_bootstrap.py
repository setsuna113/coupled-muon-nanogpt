"""Stage 4.3: bootstrap median CI sanity tests + aggregate_runs smoke."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from coupled_muon_nanogpt.analysis.bootstrap import aggregate_runs, bootstrap_median_ci


def test_bootstrap_ci_shrinks_with_more_seeds():
    rng = np.random.default_rng(0)
    samples_small = rng.normal(loc=2.0, scale=0.1, size=3).tolist()
    samples_large = rng.normal(loc=2.0, scale=0.1, size=30).tolist()
    _, lo_s, hi_s = bootstrap_median_ci(samples_small, n_resamples=2000, seed=1)
    _, lo_l, hi_l = bootstrap_median_ci(samples_large, n_resamples=2000, seed=1)
    width_small = hi_s - lo_s
    width_large = hi_l - lo_l
    assert width_large < width_small, (
        f"CI did not shrink: small={width_small:.3f} large={width_large:.3f}"
    )


def test_bootstrap_ci_brackets_known_median():
    samples = [1.0, 2.0, 3.0, 4.0, 5.0]
    med, lo, hi = bootstrap_median_ci(samples, n_resamples=5000, seed=0)
    assert med == 3.0
    assert lo <= 3.0 <= hi


def test_bootstrap_ci_handles_empty_and_singleton():
    med, lo, hi = bootstrap_median_ci([], n_resamples=100, seed=0)
    assert all(np.isnan(x) for x in (med, lo, hi))
    med, lo, hi = bootstrap_median_ci([42.0], n_resamples=100, seed=0)
    assert med == lo == hi == 42.0


def test_aggregate_runs_finds_completed_jsonl(tmp_path: Path):
    """Build two fake run directories and ensure aggregate_runs picks them up."""
    for i, lr in enumerate([1e-3, 3e-3]):
        run_dir = tmp_path / f"run_s{i}"
        run_dir.mkdir()
        cfg = OmegaConf.create(
            {
                "name": "fake",
                "model": {"moe": {"enabled": False}},
                "optimizer": {
                    "type": "coupled_muon_v2",
                    "lr": lr,
                    "coupled_steps": 4,
                    "couple_qk": True,
                    "couple_vo": True,
                    "couple_updown": True,
                },
            }
        )
        (run_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg))
        # Three step events then a val event.
        with (run_dir / "metrics.jsonl").open("w") as f:
            for s in range(3):
                f.write(json.dumps({"event": "step", "step": s, "tokens": 100 * s, "loss": 5.0 - s}) + "\n")
            f.write(json.dumps({"event": "val", "step": 2, "tokens": 200, "val_loss": 4.5 - i}) + "\n")
    rows = aggregate_runs(tmp_path)
    assert len(rows) == 2
    losses = sorted(r["final_val_loss"] for r in rows)
    assert losses == [3.5, 4.5]
