"""Stage 3.5: partial RoPE — modded-nanogpt's rotary_dim < head_dim pattern."""
from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf  # noqa: F401

from coupled_muon_nanogpt.model.components import RotaryEmbedding
from coupled_muon_nanogpt.model.transformer import GPT, BlockConfig, GPTConfig
from coupled_muon_nanogpt.optim.factory import build_optimizer


def test_rotary_embedding_partial_passes_through_tail():
    rope = RotaryEmbedding(head_dim=8, rotary_dim=4, max_seq_len=16)
    x = torch.randn(1, 4, 1, 8)  # (B, T, n_heads, head_dim)
    y = rope(x, offset=0)
    # Tail (last 4 channels) is identity (pass-through, exact match).
    torch.testing.assert_close(y[..., 4:], x[..., 4:])
    # Head (first 4 channels) is rotated, so generally not equal.
    assert not torch.allclose(y[..., :4], x[..., :4])


def test_rotary_embedding_full_matches_pre_partial_implementation():
    """rotary_dim=head_dim should reproduce the original full-RoPE behaviour."""
    rope_full = RotaryEmbedding(head_dim=8, max_seq_len=16)
    rope_explicit = RotaryEmbedding(head_dim=8, rotary_dim=8, max_seq_len=16)
    x = torch.randn(1, 4, 1, 8)
    torch.testing.assert_close(rope_full(x), rope_explicit(x))


def _build_partial_rope_model(frac: float) -> GPT:
    block = BlockConfig(
        hidden=64,
        n_heads=4,
        intermediate=128,
        norm_type="rmsnorm",
        mlp_type="swiglu",
        pos_emb_type="rope",
        rope_partial_frac=frac,
        qk_norm=False,
        max_seq_len=128,
    )
    return GPT(GPTConfig(vocab_size=128, n_layers=2, block=block))


@pytest.mark.parametrize("frac", [0.5, 0.75, 1.0])
def test_partial_rope_model_trains_one_step(frac):
    torch.manual_seed(0)
    model = _build_partial_rope_model(frac)
    x = torch.randint(0, 128, (2, 32))
    y = torch.randint(0, 128, (2, 32))
    out = model(x, targets=y)
    assert torch.isfinite(out["total_loss"]).item()
    out["total_loss"].backward()
    grads_finite = all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    )
    assert grads_finite


def _minimal_cfg(*, rope_partial_frac: float, pos_type: str, hidden: int = 64,
                  n_heads: int = 4) -> OmegaConf:
    return OmegaConf.create(
        {
            "model": {
                "hidden": hidden,
                "attn": {
                    "n_heads": n_heads,
                    "n_kv_heads": None,
                    "rope_partial_frac": rope_partial_frac,
                },
                "pos_emb": {"type": pos_type},
            },
            "optimizer": {
                "type": "coupled_muon_v2",
                "lr": 1e-3,
                "wd": 0.0,
                "momentum": 0.95,
                "betas": [0.95, 0.95],
                "eps": 1e-8,
                "ns_steps": 5,
                "coupled_steps": 4,
                "couple_qk": True,
                "couple_vo": True,
                "couple_updown": True,
                "use_multi_head": True,
                "ns_dtype": "bf16",
                "qk_coupling": {
                    "full_rope": "legacy_rope2d",
                    "no_rope_policy": "current_flat2d_fallback",
                    "partial_rope_policy": "current_flat2d_fallback",
                },
            },
        }
    )


def test_factory_forces_vo_flat2d_under_partial_rope():
    """Phase-2.1: V-O routing under rope_status='partial' must always go through
    legacy flat-2D, regardless of use_multi_head=True. The Q-K branch is now
    policy-mediated; V-O is hard-locked to flat-2D off full RoPE (review v3.4).
    """
    torch.manual_seed(0)
    model = _build_partial_rope_model(frac=0.5)
    cfg = _minimal_cfg(rope_partial_frac=0.5, pos_type="rope")
    opt = build_optimizer(model, cfg)
    # User asked for use_multi_head=True; optimizer keeps that for the Q-K
    # dispatch but internally forces V-O to flat-2D.
    assert opt.use_multi_head is True
    assert opt._use_multi_head_vo is False
    assert opt.rope_status == "partial"


def test_factory_forces_vo_flat2d_under_learned_pos_emb():
    """Audit #6 / Phase-2.1: with pos_emb.type=learned, V-O routing goes to
    legacy flat-2D. Q-K dispatch is governed by qk_coupling.no_rope_policy
    (default current_flat2d_fallback preserves legacy numerics)."""
    torch.manual_seed(0)
    block = BlockConfig(
        hidden=64,
        n_heads=4,
        intermediate=128,
        norm_type="rmsnorm",
        mlp_type="swiglu",
        pos_emb_type="learned",
        rope_partial_frac=1.0,
        qk_norm=False,
        max_seq_len=128,
    )
    model = GPT(GPTConfig(vocab_size=128, n_layers=2, block=block))
    cfg = _minimal_cfg(rope_partial_frac=1.0, pos_type="learned")
    opt = build_optimizer(model, cfg)
    assert opt.use_multi_head is True
    assert opt._use_multi_head_vo is False
    assert opt.rope_status == "none"
