"""MLA forward + pair-registration tests (Phase-2 S2).

Validates:
  - MultiLatentAttention produces well-shaped logits.
  - classify_parameters picks up (W_UK, W_DKV) as the K-side coupled pair
    and (W_UQ, W_DQ) as the Q-side pair (when q_lora_rank > 0). W_UV
    routes to plain Muon because CoupledMuon_v2's per-step `processed`
    set rules out two coupled pairs sharing a B-partner.
  - mla_kr_proj falls through to plain Muon (no factored partner).
"""
from __future__ import annotations

import pytest
import torch

from coupled_muon_nanogpt.model.transformer import GPT, BlockConfig, GPTConfig
from coupled_muon_nanogpt.optim.factory import classify_parameters


def _make_mla_model(*, q_lora_rank: int) -> GPT:
    block = BlockConfig(
        hidden=64,
        n_heads=4,
        intermediate=128,
        norm_type="rmsnorm",
        mlp_type="swiglu",
        pos_emb_type="rope",
        qk_norm=False,
        max_seq_len=64,
        attn_type="mla",
        kv_lora_rank=16,
        q_lora_rank=q_lora_rank,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        v_head_dim=16,
    )
    return GPT(GPTConfig(vocab_size=64, n_layers=2, block=block))


def test_mla_forward_shape():
    model = _make_mla_model(q_lora_rank=16)
    x = torch.randint(0, 64, (2, 16))
    out = model(x)
    assert out["logits"].shape == (2, 16, 64)
    assert torch.isfinite(out["logits"]).all()


def test_mla_kv_pair_registration():
    """Per layer: only the (UK, DKV) pair is registered for coupling; UV
    falls through to plain Muon to avoid the `processed`-set conflict in
    CoupledMuon_v2.step (a parameter cannot appear as B-partner in two
    coupled pairs in the same optimizer step). 2 layers ⇒ 2 mla_kv pairs."""
    model = _make_mla_model(q_lora_rank=16)
    groups = classify_parameters(model, n_heads=4)
    assert len(groups.coupled_mla_kv) == 2, (
        f"Expected 2 MLA KV pairs (one (UK, DKV) per layer); got {len(groups.coupled_mla_kv)}"
    )


def test_mla_uv_routed_to_muon():
    """UV is not coupled (its natural partner DKV is already paired with UK);
    it must land in `muon_2d` so it still receives Muon updates."""
    model = _make_mla_model(q_lora_rank=16)
    groups = classify_parameters(model, n_heads=4)
    # UV shape: (n_heads * v_head_dim, kv_lora_rank) = (4*16, 16) = (64, 16)
    muon_param_shapes = {tuple(p.shape) for p in groups.muon_2d}
    assert (64, 16) in muon_param_shapes, (
        f"mla_uv_proj (shape (64, 16)) missing from muon_2d; got {muon_param_shapes}"
    )


def test_mla_q_pair_when_q_lora_enabled():
    model = _make_mla_model(q_lora_rank=16)
    groups = classify_parameters(model, n_heads=4)
    assert len(groups.coupled_mla_q) == 2  # 2 layers
    a, b, _ = groups.coupled_mla_q[0]
    # `a` is UQ (q_lora_rank → n_heads*qk_head_dim); `b` is DQ (hidden → q_lora_rank)
    assert a.shape == (4 * (8 + 8), 16), a.shape
    assert b.shape == (16, 64), b.shape


def test_mla_no_q_pair_when_q_lora_disabled():
    """When q_lora_rank == 0, the full-rank q_proj is used; no Q pair."""
    model = _make_mla_model(q_lora_rank=0)
    groups = classify_parameters(model, n_heads=4)
    assert len(groups.coupled_mla_q) == 0
    # q_proj falls through to plain Muon (regular Q-K pair is also empty in MLA
    # since there's no `k_proj`; the registration code only forms a Q-K pair
    # when both q_proj and k_proj are present in the bucket).
    assert len(groups.coupled_qk) == 0
    # q_proj should be one of the muon_2d entries.
    muon_param_shapes = {tuple(p.shape) for p in groups.muon_2d}
    assert (4 * (8 + 8), 64) in muon_param_shapes, "q_proj missing from muon_2d"


def test_mla_kr_proj_falls_through_to_muon():
    """mla_kr_proj is a flat (qk_rope_head_dim, hidden) matrix with no
    factored partner — must route to plain Muon, never coupled."""
    model = _make_mla_model(q_lora_rank=16)
    groups = classify_parameters(model, n_heads=4)
    # kr_proj shape: (qk_rope_head_dim, hidden) = (8, 64)
    muon_param_shapes = {tuple(p.shape) for p in groups.muon_2d}
    assert (8, 64) in muon_param_shapes, (
        f"mla_kr_proj (shape (8, 64)) missing from muon_2d; got {muon_param_shapes}"
    )
