"""Phase-2 ladder configs — narrower test than test_ladder_configs_load.py.

The shared file already iterates over every YAML under configs/ladder/, so the
new O_mla / P_factff / Z_350m_dense / Z_350m_mla configs are picked up. Here
we add targeted assertions specific to the Phase-2 rungs (pair counts, MLA
attention type, factff FFN type).
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from coupled_muon_nanogpt.model.attention import MultiLatentAttention  # noqa: F401 — import-validation only
from coupled_muon_nanogpt.optim.factory import build_optimizer, classify_parameters
from coupled_muon_nanogpt.train import build_model, load_config


REPO_ROOT = Path(__file__).resolve().parents[1]
LADDER_DIR = REPO_ROOT / "configs" / "ladder"


@pytest.mark.parametrize(
    "rung", ["O_mla", "P_factff", "Z_350m_dense", "Z_350m_mla", "H_prime_partial_rope"]
)
def test_phase2_rung_builds(rung):
    cfg_path = LADDER_DIR / f"{rung}.yaml"
    # Aggressively shrink so the test fits in seconds.
    cfg = load_config(
        str(cfg_path),
        overrides=[
            "model.hidden=64",
            "model.n_layers=2",
            "model.attn.n_heads=4",
            "model.attn.n_kv_heads=null",
            "model.attn.kv_lora_rank=16",
            "model.attn.q_lora_rank=16",
            "model.attn.qk_nope_head_dim=8",
            "model.attn.qk_rope_head_dim=8",
            "model.attn.v_head_dim=16",
            "model.mlp.intermediate=128",
            "model.mlp.factorize_rank=16",
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
    assert torch.isfinite(out["total_loss"]).item()


def test_O_mla_registers_mla_pairs():
    cfg = load_config(
        str(LADDER_DIR / "O_mla.yaml"),
        overrides=[
            "model.hidden=64",
            "model.n_layers=2",
            "model.attn.n_heads=4",
            "model.attn.kv_lora_rank=16",
            "model.attn.q_lora_rank=16",
            "model.attn.qk_nope_head_dim=8",
            "model.attn.qk_rope_head_dim=8",
            "model.attn.v_head_dim=16",
            "model.mlp.intermediate=128",
            "train.seq_len=64",
            "train.local_batch_size=2",
            "model.moe.enabled=false",  # disable MoE for the simpler param-count check
        ],
    )
    model = build_model(cfg)
    groups = classify_parameters(model, n_heads=4)
    # 2 layers × (UK-DKV) ⇒ 2 MLA-KV pairs (UV routed to plain Muon).
    assert len(groups.coupled_mla_kv) == 2
    # 2 layers × (UQ-DQ) ⇒ 2 MLA-Q pairs
    assert len(groups.coupled_mla_q) == 2


def test_P_factff_registers_factff_pairs():
    cfg = load_config(
        str(LADDER_DIR / "P_factff.yaml"),
        overrides=[
            "model.hidden=64",
            "model.n_layers=2",
            "model.attn.n_heads=4",
            "model.mlp.factorize_rank=16",
            "train.seq_len=64",
            "train.local_batch_size=2",
        ],
    )
    model = build_model(cfg)
    groups = classify_parameters(model, n_heads=4)
    assert len(groups.coupled_factff) == 2
    assert len(groups.coupled_updown) == 0
