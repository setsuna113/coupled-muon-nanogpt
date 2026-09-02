"""Tests for MoE load probe and shadow-trace ns_internal probe.

These two probes were stubs/silent prior to Stage 1.3/1.4. The tests pin down
that they actually surface metrics now, and that their interaction with the
ProbeManager's pre-step / post-step partitioning is correct.
"""
from __future__ import annotations

import torch
from omegaconf import OmegaConf

from coupled_muon_nanogpt.model.transformer import GPT, BlockConfig, GPTConfig
from coupled_muon_nanogpt.optim.factory import build_optimizer
from coupled_muon_nanogpt.probes.manager import ProbeManager
from coupled_muon_nanogpt.probes.moe_load import moe_load_probe
from coupled_muon_nanogpt.probes.ns_internal import ns_internal_probe


def _moe_model() -> GPT:
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
            top_k=2,
            capacity_factor=1.25,
            aux_loss_coef=0.01,
            z_loss_coef=1e-3,
            expert_mlp_type="swiglu",
        ),
    )
    # Use 4 layers so the every-other default (which matches d.2 rung I)
    # gives us 2 MoE layers worth of probe output to assert against.
    return GPT(GPTConfig(vocab_size=512, n_layers=4, block=block))


def _build_optim(model: GPT) -> torch.optim.Optimizer:
    cfg = OmegaConf.create(
        {
            "model": {
                "hidden": 64,
                "attn": {"n_heads": 4, "n_kv_heads": None, "rope_partial_frac": 1.0},
                "pos_emb": {"type": "rope"},
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
                "use_multi_head": False,
                "ns_dtype": "bf16",
            },
        }
    )
    return build_optimizer(model, cfg)


def test_moe_load_probe_returns_per_layer_metrics():
    torch.manual_seed(0)
    model = _moe_model()
    opt = _build_optim(model)
    x = torch.randint(0, 512, (2, 32))
    y = torch.randint(0, 512, (2, 32))
    out = model(x, targets=y)
    out["total_loss"].backward()

    result = moe_load_probe(model, opt, {})
    # 4-layer model with every-other ⇒ 2 MoEFFN modules ⇒ probe keys "layer.0", "layer.1"
    # (the probe enumerates MoE layers, not transformer layers).
    assert "layer.0" in result and "layer.1" in result
    for k, v in result.items():
        assert "loads" in v and len(v["loads"]) == 4  # 4 experts
        assert "imbalance" in v
        assert "router_entropy" in v
        assert "grad_norms" in v and len(v["grad_norms"]) == 4
        assert "grad_norm_var" in v


def test_ns_internal_probe_emits_shadow_trace():
    torch.manual_seed(1)
    model = _moe_model()
    opt = _build_optim(model)
    x = torch.randint(0, 512, (2, 32))
    y = torch.randint(0, 512, (2, 32))
    out = model(x, targets=y)
    out["total_loss"].backward()

    result = ns_internal_probe(model, opt, {})
    # Should have a Q-K trace and at least one V-O / up-down trace.
    assert "qk0" in result, f"expected qk0 in probe output, got keys={list(result)}"
    assert "ud_or_vo_0" in result
    for key in ("qk0", "ud_or_vo_0"):
        v = result[key]
        # coupled_steps=4 ⇒ 5 norms (initial + 4 iterations).
        assert len(v["norms_A"]) == 5
        assert len(v["norms_B"]) == 5
        assert not v["diverged"]
        # Detect actual numerical explosion (not just early-iteration growth).
        # Truly diverged kernels produce >1e3 quickly; healthy ones stay <100
        # even for ill-conditioned random init.
        assert v["max_norm"] < 1e3, f"norm explosion: {v['max_norm']}"


def test_probe_manager_partitions_needs_grads():
    """ProbeManager.maybe_fire(needs_grads=True/False/None) correctly partitions."""
    pm = ProbeManager()
    fired_pre, fired_post = [], []

    def pre_probe(*_a, **_k):
        fired_pre.append(1)
        return {}

    def post_probe(*_a, **_k):
        fired_post.append(1)
        return {}

    pm.register("pre", pre_probe, interval_tokens=1, needs_grads=True)
    pm.register("post", post_probe, interval_tokens=1, needs_grads=False)

    pm.maybe_fire(None, None, 100, {}, needs_grads=True)
    assert fired_pre == [1]
    assert fired_post == []

    pm.maybe_fire(None, None, 200, {}, needs_grads=False)
    assert fired_pre == [1]
    assert fired_post == [1]
