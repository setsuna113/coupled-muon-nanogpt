"""Pair classification correctness across all MLP variants and toggles."""
from __future__ import annotations

import pytest
import torch

from coupled_muon_nanogpt.model.transformer import GPT, BlockConfig, GPTConfig
from coupled_muon_nanogpt.optim.factory import classify_parameters


def _make_model(mlp_type: str, qk_norm: bool = False, moe_enabled: bool = False) -> GPT:
    block = BlockConfig(
        hidden=64,
        n_heads=4,
        intermediate=128,
        norm_type="rmsnorm",
        mlp_type=mlp_type,
        pos_emb_type="rope",
        qk_norm=qk_norm,
        max_seq_len=128,
        moe_enabled=moe_enabled,
        moe_cfg=dict(num_experts=4, top_k=2, expert_mlp_type=mlp_type) if moe_enabled else {},
    )
    return GPT(GPTConfig(vocab_size=512, n_layers=2, block=block))


@pytest.mark.parametrize("mlp_type", ["swiglu", "gelu_2mat", "relu2"])
def test_dense_pairs(mlp_type):
    model = _make_model(mlp_type)
    groups = classify_parameters(model, n_heads=4)

    # 2 layers ⇒ 2 of each pair.
    assert len(groups.coupled_qk) == 2
    assert len(groups.coupled_vo) == 2
    assert len(groups.coupled_updown) == 2

    # Gate goes to muon_2d for SwiGLU; nothing extra for the others.
    if mlp_type == "swiglu":
        gate_count = sum(1 for p in groups.muon_2d if any(p is gp for gp in [m.gate_proj.weight for m in [model.layers[0].mlp, model.layers[1].mlp]]))
        assert gate_count == 2, "SwiGLU gate must be in muon_2d, not coupled"
    elif mlp_type in ("gelu_2mat", "relu2"):
        # No gate in 2-mat MLPs; muon_2d should be empty under default toggles.
        assert len(groups.muon_2d) == 0, f"unexpected muon_2d: {[p.shape for p in groups.muon_2d]}"

    # AdamW catch-all: embed weight (lm_head is tied) + RMSNorm scales + qk_norm if any.
    # We assert at least the embedding is present.
    assert any(p is model.embed.weight for p in groups.adamw_other)


def test_pair_toggle_demotes_to_muon():
    model = _make_model("swiglu")
    groups = classify_parameters(model, n_heads=4, couple_qk=False, couple_vo=False, couple_updown=False)
    assert len(groups.coupled_qk) == 0
    assert len(groups.coupled_vo) == 0
    assert len(groups.coupled_updown) == 0
    # 2 layers × {q, k, v, o, up, down, gate} = 14 matrices.
    assert len(groups.muon_2d) == 14


def test_router_goes_to_adamw():
    model = _make_model("swiglu", moe_enabled=True)
    groups = classify_parameters(model, n_heads=4)
    # 2 layers, 1 router each → 2 router params.
    assert len(groups.router_params) == 2
    # Router params must NOT appear anywhere else.
    not_in_muon = all(not any(p is r for r in groups.router_params) for p in groups.muon_2d)
    assert not_in_muon


def test_pairs_have_correct_shapes_for_optimizer():
    model = _make_model("swiglu")
    groups = classify_parameters(model, n_heads=4)
    # V-O: A=o_proj (hidden, n_heads*head_dim) -> (64, 64); B=v_proj (64, 64). MHA so HD=HD_B.
    for a, b, _h in groups.coupled_vo:
        assert a.shape == (64, 64)
        assert b.shape == (64, 64)
    # up-down: A=down_proj (hidden, intermediate)=(64,128); B=up_proj (intermediate, hidden)=(128,64).
    for a, b, _h in groups.coupled_updown:
        assert a.shape == (64, 128)
        assert b.shape == (128, 64)
    # Q-K: A=q_proj (n_heads*head_dim, hidden)=(64,64); B=k_proj (n_kv_heads*head_dim, hidden)=(64,64).
    for a, b, _h in groups.coupled_qk:
        assert a.shape == (64, 64)
        assert b.shape == (64, 64)
