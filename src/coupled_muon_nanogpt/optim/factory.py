"""Optimizer factory: parameter classification + builder.

Adapts the param-classification logic from
`/home/lyc/dev/QZ/Coupled_muon/__init__.py:136-240` to NanoGPT-style parameter
naming. The model is expected to expose its blocks via a top-level `layers`
ModuleList, with attention projections named `q_proj`, `k_proj`, `v_proj`,
`o_proj`, and MLP projections named `up_proj`, `down_proj`, and optionally
`gate_proj` (SwiGLU only).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from .coupled_muon import CoupledMuon_v2
from .muon import Muon

# Regex extracts integer layer index from "...layers.{i}..." or "...h.{i}..." (NanoGPT style).
_LAYER_RE = re.compile(r"\.(?:layers|h)\.(\d+)\.")
_PROJ_NAMES = ("q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj")


@dataclass
class ParamGroups:
    coupled_qk: list[tuple[nn.Parameter, nn.Parameter, int]] = field(default_factory=list)
    coupled_vo: list[tuple[nn.Parameter, nn.Parameter, int]] = field(default_factory=list)
    coupled_updown: list[tuple[nn.Parameter, nn.Parameter, int]] = field(default_factory=list)
    muon_2d: list[nn.Parameter] = field(default_factory=list)
    adamw_other: list[nn.Parameter] = field(default_factory=list)
    router_params: list[nn.Parameter] = field(default_factory=list)


def _layer_idx(name: str) -> str | None:
    m = _LAYER_RE.search("." + name)
    return m.group(1) if m else None


def _proj_role(name: str) -> str | None:
    for role in _PROJ_NAMES:
        if role in name:
            return role
    return None


def classify_parameters(
    model: nn.Module,
    *,
    couple_qk: bool = True,
    couple_vo: bool = True,
    couple_updown: bool = True,
    n_heads: int = 1,
    n_kv_heads: int | None = None,
) -> ParamGroups:
    """Walk model.named_parameters() and split into Coupled-Muon groups.

    A parameter is a candidate for Muon/coupling iff it is 2-D and is *not* an
    embedding or LM head (those run AdamW).
    """
    groups = ParamGroups()
    n_kv = n_kv_heads or n_heads

    layer_buckets: dict[str, dict[str, nn.Parameter]] = {}
    other_2d: list[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Routers are routed to AdamW by default (per experiment.md c.3).
        if "router" in name or "gate_router" in name:
            groups.router_params.append(param)
            continue
        # Embeddings / LM head / 1D params -> AdamW.
        if param.ndim < 2 or "embed" in name or "lm_head" in name or "wte" in name or "wpe" in name:
            groups.adamw_other.append(param)
            continue

        # 2D matrix params: try to bucket by layer + role.
        layer_idx = _layer_idx(name)
        role = _proj_role(name) if layer_idx is not None else None
        if layer_idx is None or role is None:
            other_2d.append(param)
            continue

        layer_buckets.setdefault(layer_idx, {})[role] = param

    # Now form coupled pairs per the QZ convention.
    # Pair tuples are (param_A, param_B, n_heads_for_pair).
    for _, params in layer_buckets.items():
        # V-O pair
        if couple_vo and "v_proj" in params and "o_proj" in params:
            groups.coupled_vo.append((params["o_proj"], params["v_proj"], n_kv))
        else:
            if "v_proj" in params:
                groups.muon_2d.append(params["v_proj"])
            if "o_proj" in params:
                groups.muon_2d.append(params["o_proj"])

        # up-down pair (gate goes to plain Muon when present, per a.3)
        if couple_updown and "up_proj" in params and "down_proj" in params:
            up_size_0 = params["up_proj"].size(0)
            groups.coupled_updown.append((params["down_proj"], params["up_proj"], up_size_0))
        else:
            if "up_proj" in params:
                groups.muon_2d.append(params["up_proj"])
            if "down_proj" in params:
                groups.muon_2d.append(params["down_proj"])
        if "gate_proj" in params:
            groups.muon_2d.append(params["gate_proj"])

        # Q-K pair
        if couple_qk and "q_proj" in params and "k_proj" in params:
            groups.coupled_qk.append((params["q_proj"], params["k_proj"], n_heads))
        else:
            if "q_proj" in params:
                groups.muon_2d.append(params["q_proj"])
            if "k_proj" in params:
                groups.muon_2d.append(params["k_proj"])

    # Anything else 2D (e.g. MoE expert matrices not under our naming) → plain Muon.
    groups.muon_2d.extend(other_2d)
    return groups


def build_optimizer(
    model: nn.Module,
    cfg: Any,
) -> torch.optim.Optimizer:
    """Construct the optimizer from a resolved config dict.

    Expects `cfg.optimizer` with keys: type, lr, wd, momentum, betas, eps,
    ns_steps, coupled_steps, couple_qk, couple_vo, couple_updown, ns_dtype.
    Also peeks at `cfg.model.attn.n_heads` and optional `attn.n_kv_heads`.
    """
    opt = cfg.optimizer
    n_heads = int(cfg.model.attn.n_heads)
    n_kv_heads = int(cfg.model.attn.get("n_kv_heads", n_heads) or n_heads)

    if opt.type == "adamw":
        return torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=float(opt.lr),
            weight_decay=float(opt.wd),
            betas=tuple(opt.betas),
            eps=float(opt.eps),
        )

    if opt.type == "muon":
        groups = classify_parameters(
            model,
            couple_qk=False,
            couple_vo=False,
            couple_updown=False,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
        )
        # All matrix candidates flow into muon_params.
        muon_params = list(groups.muon_2d)
        adamw_params = list(groups.adamw_other) + list(groups.router_params)
        return Muon(
            muon_params=muon_params,
            adamw_params=adamw_params,
            lr=float(opt.lr),
            wd=float(opt.wd),
            momentum=float(opt.momentum),
            ns_steps=int(opt.ns_steps),
            adamw_betas=tuple(opt.betas),
            adamw_eps=float(opt.eps),
        )

    if opt.type == "coupled_muon_v2":
        groups = classify_parameters(
            model,
            couple_qk=bool(opt.couple_qk),
            couple_vo=bool(opt.couple_vo),
            couple_updown=bool(opt.couple_updown),
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
        )
        # Build the list of (A, B, n_heads_for_pair, is_qk) tuples.
        coupled_pairs: list[tuple[nn.Parameter, nn.Parameter, int, bool]] = []
        for a, b, h in groups.coupled_vo:
            coupled_pairs.append((a, b, h, False))
        for a, b, h in groups.coupled_updown:
            coupled_pairs.append((a, b, h, False))
        for a, b, h in groups.coupled_qk:
            coupled_pairs.append((a, b, h, True))

        return CoupledMuon_v2(
            lr=float(opt.lr),
            wd=float(opt.wd),
            coupled_pairs=coupled_pairs,
            muon_params=list(groups.muon_2d),
            adamw_params=list(groups.adamw_other) + list(groups.router_params),
            momentum=float(opt.momentum),
            nesterov=True,
            ns_steps=int(opt.ns_steps),
            coupled_steps=int(opt.coupled_steps),
            adamw_betas=tuple(opt.betas),
            adamw_eps=float(opt.eps),
            enable_coupled=True,
            use_multi_head=bool(opt.get("use_multi_head", False)),
            n_heads=n_heads,
        )

    raise ValueError(f"Unknown optimizer.type={opt.type!r}")
