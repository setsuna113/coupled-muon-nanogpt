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
    # Under every-other (post-#2 default), MoE lands on odd-indexed blocks.
    # 4 layers ⇒ 2 dense + 2 MoE — enough to assert non-trivial expert/router
    # counts. Dense (non-MoE) tests still see 2 layers' worth of pairs since
    # n_layers=2 is set explicitly below for the dense path.
    n_layers = 4 if moe_enabled else 2
    return GPT(GPTConfig(vocab_size=512, n_layers=n_layers, block=block))


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


def test_router_goes_to_muon_by_default():
    """Post-c.3-rewrite: routers default to Muon (matches Moonlight 2502.16982
    §2.2 and Cerebras nanoMoE). Rung J flips this via couple_router_to_muon=False."""
    model = _make_model("swiglu", moe_enabled=True)
    groups = classify_parameters(model, n_heads=4)
    # 4 layers, every-other ⇒ 2 MoE blocks, 2 routers; default routes them to Muon.
    assert len(groups.router_params) == 0, "default should route routers to Muon, not router_params"
    router_weights = [p for n, p in model.named_parameters() if "router" in n]
    assert len(router_weights) == 2, "expected 2 router weights from 4 layers, every-other MoE"
    for r in router_weights:
        assert any(p is r for p in groups.muon_2d), "router weight should be in muon_2d under default"


def test_router_goes_to_adamw_under_J_ablation():
    """Rung J: couple_router_to_muon=False routes routers into router_params (AdamW)."""
    model = _make_model("swiglu", moe_enabled=True)
    groups = classify_parameters(model, n_heads=4, couple_router_to_muon=False)
    assert len(groups.router_params) == 2
    # Routers must NOT appear anywhere else.
    for r in groups.router_params:
        assert not any(p is r for p in groups.muon_2d)


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


def test_moe_every_expert_lands_in_some_group():
    """Audit-#1 regression: every expert weight in an MoE model must enter
    exactly one optimizer group. Without per-expert bucket keys, all but the
    last expert per layer were silently frozen (252 of 398 weights at I)."""
    model = _make_model("swiglu", moe_enabled=True)
    groups = classify_parameters(model, n_heads=4)
    seen: set[int] = set()
    for a, b, _h in groups.coupled_qk:
        seen.update([id(a), id(b)])
    for a, b, _h in groups.coupled_vo:
        seen.update([id(a), id(b)])
    for a, b, _h in groups.coupled_updown:
        seen.update([id(a), id(b)])
    for p in (*groups.muon_2d, *groups.adamw_other, *groups.router_params):
        seen.add(id(p))

    missing = [n for n, p in model.named_parameters() if p.requires_grad and id(p) not in seen]
    assert not missing, f"params missing from optimizer (would freeze): {missing}"
    # 4 layers, every-other ⇒ 2 dense (1 pair each) + 2 MoE (4 experts × 1 pair) = 10 pairs.
    assert len(groups.coupled_updown) == 10
    # 2 dense gates + 2 MoE × 4 expert gates + 2 MoE routers (default → Muon, c.3) = 12.
    assert len(groups.muon_2d) == 12


def test_moe_shared_expert_lands_in_some_group():
    """Shared-expert (rung L) uses `mlp.shared_expert.{up,down,gate}_proj` naming.
    Verify the per-expert bucket-key logic also covers it."""
    block = BlockConfig(
        hidden=64,
        n_heads=4,
        intermediate=128,
        norm_type="rmsnorm",
        mlp_type="swiglu",
        pos_emb_type="rope",
        qk_norm=False,
        max_seq_len=128,
        moe_enabled=True,
        moe_cfg=dict(
            num_experts=4,
            top_k=1,
            expert_mlp_type="swiglu",
            shared_expert=True,
        ),
    )
    # 4 layers + every-other ⇒ 2 MoE layers, 2 dense layers.
    model = GPT(GPTConfig(vocab_size=128, n_layers=4, block=block))
    groups = classify_parameters(model, n_heads=4)
    seen: set[int] = set()
    for a, b, _h in groups.coupled_qk:
        seen.update([id(a), id(b)])
    for a, b, _h in groups.coupled_vo:
        seen.update([id(a), id(b)])
    for a, b, _h in groups.coupled_updown:
        seen.update([id(a), id(b)])
    for p in (*groups.muon_2d, *groups.adamw_other, *groups.router_params):
        seen.add(id(p))

    missing = [n for n, p in model.named_parameters() if p.requires_grad and id(p) not in seen]
    assert not missing, f"shared-expert params missing: {missing}"
    # 2 dense (1 pair) + 2 MoE × (4 routed + 1 shared) = 12 (down,up) pairs.
    assert len(groups.coupled_updown) == 12
    # 2 dense gates + 2 MoE × (4 routed + 1 shared) gates + 2 MoE routers (default → Muon, c.3) = 14.
    assert len(groups.muon_2d) == 14
