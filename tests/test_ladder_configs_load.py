"""Verify every ladder/smoke config builds a model and runs forward+backward+step.

This is the d.2 ladder coverage gate: every rung from A0 to N should at least
be architecturally buildable on synthetic input. Catches misconfigured YAMLs
before they consume hours of GPU time.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from coupled_muon_nanogpt.optim.factory import build_optimizer
from coupled_muon_nanogpt.train import build_model, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
LADDER_DIR = REPO_ROOT / "configs" / "ladder"
SMOKE_DIR = REPO_ROOT / "configs" / "smoke"


def _all_configs() -> list[Path]:
    return sorted(LADDER_DIR.glob("*.yaml")) + sorted(SMOKE_DIR.glob("*.yaml"))


@pytest.mark.parametrize("cfg_path", _all_configs(), ids=lambda p: p.stem)
def test_config_builds_and_steps(cfg_path: Path):
    # Override to a tiny model so the test fits in seconds — we are testing the
    # config-resolves-to-buildable-architecture path, not training.
    cfg = load_config(
        str(cfg_path),
        overrides=[
            "model.hidden=64",
            "model.n_layers=2",
            "model.attn.n_heads=4",
            "model.attn.n_kv_heads=null",
            "model.mlp.intermediate=128",
            "train.seq_len=64",
            "train.local_batch_size=2",
            "train.grad_accum_steps=1",
        ],
    )

    model = build_model(cfg)
    optimizer = build_optimizer(model, cfg)

    x = torch.randint(0, int(cfg.model.vocab_size), (2, 32))
    y = torch.randint(0, int(cfg.model.vocab_size), (2, 32))
    out = model(x, targets=y)
    out["total_loss"].backward()
    optimizer.step()
    optimizer.zero_grad()

    assert torch.isfinite(out["total_loss"]).item(), f"non-finite loss for {cfg_path.name}"
