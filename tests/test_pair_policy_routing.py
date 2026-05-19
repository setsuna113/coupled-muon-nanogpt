"""Phase-2.1: pair_policy enum routing tests.

The 7 tests confirm:
  1. pair_policy=all sets couple_qk/vo/updown all true.
  2. pair_policy=attention_only routes up/down to muon_2d.
  3. pair_policy=ffn_only routes q/k/v/o to muon_2d.
  4. pair_policy=ffn_only in MoE: each expert's up/down is coupled while
     attention pairs go to Muon (load-bearing for Phase C).
  5. pair_policy conflict with explicit booleans raises ValueError unless
     allow_pair_policy_override=true.
  6. pair_policy=attention_only does NOT disable MLA K-side coupling
     (couple_mla is independent, review v3.9).
  7. qk_coupling.no_rope_policy=qk_off does NOT touch MLA K-side coupling.
"""
from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from coupled_muon_nanogpt.model.transformer import GPT, BlockConfig, GPTConfig
from coupled_muon_nanogpt.optim.factory import build_optimizer, classify_parameters


def _build_dense_model() -> GPT:
    block = BlockConfig(
        hidden=64,
        n_heads=4,
        intermediate=128,
        norm_type="rmsnorm",
        mlp_type="swiglu",
        pos_emb_type="rope",
        qk_norm=False,
        max_seq_len=64,
    )
    return GPT(GPTConfig(vocab_size=64, n_layers=2, block=block))


def _build_moe_model() -> GPT:
    block = BlockConfig(
        hidden=64,
        n_heads=4,
        intermediate=128,
        norm_type="rmsnorm",
        mlp_type="swiglu",
        pos_emb_type="rope",
        qk_norm=False,
        max_seq_len=64,
        moe_enabled=True,
        moe_cfg=dict(num_experts=4, top_k=2, expert_mlp_type="swiglu"),
    )
    return GPT(GPTConfig(vocab_size=64, n_layers=2, block=block))


def _build_mla_model(q_lora_rank: int = 16) -> GPT:
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


def _cfg_for(model_kind: str, *, pair_policy=None, allow_override=False,
             couple_qk=True, couple_vo=True, couple_updown=True,
             qk_no_rope_policy: str = "current_flat2d_fallback") -> OmegaConf:
    out = {
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
            "couple_qk": couple_qk,
            "couple_vo": couple_vo,
            "couple_updown": couple_updown,
            "couple_mla": True,
            "couple_factff": True,
            "use_multi_head": True,
            "ns_dtype": "fp32",
            "pair_policy": pair_policy,
            "allow_pair_policy_override": allow_override,
            "qk_coupling": {
                "full_rope": "legacy_rope2d",
                "no_rope_policy": qk_no_rope_policy,
                "partial_rope_policy": "current_flat2d_fallback",
            },
        },
    }
    return OmegaConf.create(out)


def _name_of(param, model) -> str:
    for n, p in model.named_parameters():
        if p is param:
            return n
    return "<unknown>"


def _has_role(params: list, role: str, model: GPT) -> bool:
    return any(role in _name_of(p, model) for p in params)


def _coupled_pair_roles(opt, model) -> dict[str, int]:
    counts: dict[str, int] = {}
    for a, b, _h, _is_qk in opt.coupled_pairs:
        for p in (a, b):
            name = _name_of(p, model)
            for role in ("q_proj", "k_proj", "v_proj", "o_proj",
                         "up_proj", "down_proj",
                         "mla_uk_proj", "mla_dkv_proj",
                         "mla_uq_proj", "mla_dq_proj"):
                if f".{role}." in "." + name:
                    counts[role] = counts.get(role, 0) + 1
    return counts


def test_pair_policy_all_sets_all_booleans_true():
    torch.manual_seed(0)
    model = _build_dense_model()
    cfg = _cfg_for("dense", pair_policy="all", allow_override=True)
    opt = build_optimizer(model, cfg)
    roles = _coupled_pair_roles(opt, model)
    # 2 layers × {q,k,v,o,up,down} all should appear in coupled pairs.
    for role in ("q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"):
        assert roles.get(role, 0) == 2, f"role {role} missing or wrong count: {roles}"


def test_pair_policy_attention_only_disables_updown():
    torch.manual_seed(0)
    model = _build_dense_model()
    cfg = _cfg_for("dense", pair_policy="attention_only", couple_updown=False, allow_override=False)
    opt = build_optimizer(model, cfg)
    roles = _coupled_pair_roles(opt, model)
    for role in ("q_proj", "k_proj", "v_proj", "o_proj"):
        assert roles.get(role, 0) == 2, f"attention role {role}: {roles}"
    # up/down NOT in coupled pairs
    assert roles.get("up_proj", 0) == 0, "attention_only must not couple up_proj"
    assert roles.get("down_proj", 0) == 0, "attention_only must not couple down_proj"
    # up/down ARE in muon_2d
    muon_names = [_name_of(p, model) for p in opt.param_groups[0]["params"]
                   if opt.state[p].get("use_muon", False)]
    assert any("up_proj" in n for n in muon_names), f"up_proj missing from muon_2d: {muon_names}"
    assert any("down_proj" in n for n in muon_names), f"down_proj missing from muon_2d: {muon_names}"


def test_pair_policy_ffn_only_disables_attention():
    torch.manual_seed(0)
    model = _build_dense_model()
    cfg = _cfg_for("dense", pair_policy="ffn_only", couple_qk=False, couple_vo=False, allow_override=False)
    opt = build_optimizer(model, cfg)
    roles = _coupled_pair_roles(opt, model)
    assert roles.get("up_proj", 0) == 2 and roles.get("down_proj", 0) == 2, roles
    for role in ("q_proj", "k_proj", "v_proj", "o_proj"):
        assert roles.get(role, 0) == 0, f"ffn_only must not couple {role}: {roles}"
    muon_names = [_name_of(p, model) for p in opt.param_groups[0]["params"]
                   if opt.state[p].get("use_muon", False)]
    for role in ("q_proj", "k_proj", "v_proj", "o_proj"):
        assert any(role in n for n in muon_names), f"{role} missing from muon_2d: {muon_names}"


def test_pair_policy_routing_with_moe_experts():
    torch.manual_seed(0)
    model = _build_moe_model()
    cfg = _cfg_for("moe", pair_policy="ffn_only", couple_qk=False, couple_vo=False)
    opt = build_optimizer(model, cfg)
    roles = _coupled_pair_roles(opt, model)
    # Each MoE layer contributes per-expert up/down pairs.
    assert roles.get("up_proj", 0) > 0 and roles.get("down_proj", 0) > 0, roles
    for role in ("q_proj", "k_proj", "v_proj", "o_proj"):
        assert roles.get(role, 0) == 0, f"ffn_only at MoE must not couple {role}: {roles}"


def test_pair_policy_conflict_raises():
    """pair_policy=attention_only is incompatible with couple_updown=True
    (base.yaml default) unless allow_pair_policy_override=true.
    """
    torch.manual_seed(0)
    model = _build_dense_model()
    cfg = _cfg_for("dense", pair_policy="attention_only", couple_updown=True, allow_override=False)
    with pytest.raises(ValueError, match="pair_policy"):
        build_optimizer(model, cfg)
    # Override demotes to warning.
    import warnings as _w
    cfg2 = _cfg_for("dense", pair_policy="attention_only", couple_updown=True, allow_override=True)
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        opt = build_optimizer(model, cfg2)
        assert any("pair_policy" in str(w.message) for w in caught)
    # The override took effect: up/down should NOT be coupled.
    roles = _coupled_pair_roles(opt, model)
    assert roles.get("up_proj", 0) == 0 and roles.get("down_proj", 0) == 0, roles


def test_pair_policy_does_not_disable_mla_pair():
    """pair_policy=attention_only must NOT disable MLA K-side coupling.
    MLA factored pairs are governed by `couple_mla` (independent of pair_policy
    which affects only classic q/k/v/o + up/down).
    """
    torch.manual_seed(0)
    model = _build_mla_model()
    cfg = _cfg_for("mla", pair_policy="attention_only", couple_updown=False, allow_override=False)
    opt = build_optimizer(model, cfg)
    roles = _coupled_pair_roles(opt, model)
    assert roles.get("mla_uk_proj", 0) == 2 and roles.get("mla_dkv_proj", 0) == 2, (
        f"MLA K-side pair should still be coupled under pair_policy=attention_only: {roles}"
    )


def test_mla_pair_isolation_from_qk_coupling():
    """qk_coupling.no_rope_policy=qk_off applies only to classic q_proj/k_proj
    via the optimizer's step-time dispatch. MLA K-side pair (W_UK, W_DKV) is
    registered via classify_parameters regardless of qk_coupling.
    """
    torch.manual_seed(0)
    # MLA + full RoPE: qk_coupling.no_rope_policy is inert by rope_status=full.
    # We still want to verify the classify_parameters output is unchanged.
    model = _build_mla_model()
    cfg = _cfg_for("mla", qk_no_rope_policy="qk_off")
    opt = build_optimizer(model, cfg)
    roles = _coupled_pair_roles(opt, model)
    assert roles.get("mla_uk_proj", 0) == 2, (
        f"MLA K-side pair should be coupled regardless of qk_coupling.no_rope_policy: {roles}"
    )
