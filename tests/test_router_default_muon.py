"""Pin down the MoE-router default routing decision in the optimizer factory.

experiment.md c.3 (rewritten) defaults the MoE router to Muon, matching
Moonlight (2502.16982 §2.2) and Cerebras nanoMoE. The legacy default sent
routers to AdamW. This test pins that the new default flows routers into
`groups.muon_2d`, and that the `couple_router_to_muon: false` ablation (rung J)
flows them into `groups.router_params` instead.
"""
from __future__ import annotations

import torch

from coupled_muon_nanogpt.model.transformer import GPT, BlockConfig, GPTConfig
from coupled_muon_nanogpt.optim.factory import classify_parameters


def _make_moe_model() -> GPT:
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
        moe_cfg={
            "num_experts": 4,
            "top_k": 2,
            "capacity_factor": 1.25,
            "aux_loss_coef": 0.01,
            "z_loss_coef": 1.0e-3,
            "balancing_type": "aux_loss",
        },
    )
    return GPT(GPTConfig(vocab_size=512, n_layers=4, block=block))


def _router_params(model: GPT) -> list[torch.nn.Parameter]:
    return [p for n, p in model.named_parameters() if "router" in n]


def test_default_routes_routers_to_muon():
    """Default `couple_router_to_muon=True`: routers land in muon_2d, not router_params."""
    model = _make_moe_model()
    routers = _router_params(model)
    assert routers, "expected at least one MoE router parameter on the test model"

    groups = classify_parameters(model, n_heads=4)  # uses default True

    router_set = {id(p) for p in routers}
    muon_set = {id(p) for p in groups.muon_2d}
    adamw_set = {id(p) for p in groups.router_params}

    for r in routers:
        assert id(r) in muon_set, "router weight should be in muon_2d under the default"
        assert id(r) not in adamw_set, "router weight should NOT be in router_params under the default"


def test_explicit_false_routes_routers_to_adamw():
    """`couple_router_to_muon=False` (rung J): routers land in router_params."""
    model = _make_moe_model()
    routers = _router_params(model)

    groups = classify_parameters(model, n_heads=4, couple_router_to_muon=False)

    muon_set = {id(p) for p in groups.muon_2d}
    adamw_set = {id(p) for p in groups.router_params}

    for r in routers:
        assert id(r) in adamw_set, (
            "router weight should be in router_params (AdamW path) under couple_router_to_muon=False"
        )
        assert id(r) not in muon_set, "router weight should NOT be in muon_2d under the J ablation"


def test_dense_model_has_no_router_params():
    """Sanity: a dense model has no router params in either group."""
    block = BlockConfig(
        hidden=64,
        n_heads=4,
        intermediate=128,
        norm_type="rmsnorm",
        mlp_type="swiglu",
        pos_emb_type="rope",
        qk_norm=False,
        max_seq_len=128,
        moe_enabled=False,
    )
    model = GPT(GPTConfig(vocab_size=512, n_layers=2, block=block))
    groups = classify_parameters(model, n_heads=4)
    assert groups.router_params == [], "dense model should have no router params"
