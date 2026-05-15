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
import warnings
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from .coupled_muon import CoupledMuon_v2
from .muon import Muon


def _resolve_ns_dtype(name: str) -> torch.dtype:
    name = str(name).lower()
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp32", "float32"):
        return torch.float32
    if name in ("fp16", "float16"):
        return torch.float16
    raise ValueError(f"Unknown ns_dtype={name!r} (expected bf16|fp32|fp16)")

# Regex extracts integer layer index from "...layers.{i}..." or "...h.{i}..." (NanoGPT style).
_LAYER_RE = re.compile(r"\.(?:layers|h)\.(\d+)\.")
# Sub-context regexes: identify per-expert and shared-expert branches under MoEFFN.
# Without these, every `experts.{e}.up_proj.weight` collapsed onto one bucket key
# and only the last expert's weights entered the optimizer (#1 audit).
_EXPERT_RE = re.compile(r"\.experts\.(\d+)\.")
_SHARED_EXPERT_RE = re.compile(r"\.shared_expert\.")
_PROJ_NAMES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "up_proj", "down_proj", "gate_proj",
    # Phase-2 MLA projections (factored attention; suggestion.md S2 / §2.4).
    # Pair convention: (W_UK, W_DKV) and (W_UV, W_DKV) for KV down-up;
    # (W_UQ, W_DQ) for the optional low-rank Q. `mla_kr_proj` (the always-
    # RoPE'd K side) has no factored partner — routed to plain Muon.
    "mla_dkv_proj", "mla_uk_proj", "mla_uv_proj", "mla_kr_proj",
    "mla_dq_proj", "mla_uq_proj",
    # Phase-2 imposed-FFN-factorisation (Tier B3 / §245).
    "factff_up_proj", "factff_down_proj",
)


@dataclass
class ParamGroups:
    coupled_qk: list[tuple[nn.Parameter, nn.Parameter, int]] = field(default_factory=list)
    coupled_vo: list[tuple[nn.Parameter, nn.Parameter, int]] = field(default_factory=list)
    coupled_updown: list[tuple[nn.Parameter, nn.Parameter, int]] = field(default_factory=list)
    # Phase-2 factored-architecture pairs (MLA KV/Q + imposed-factored FFN).
    # The MLA W_DKV param appears as B-partner in BOTH (UK,DKV) and (UV,DKV);
    # CoupledMuon_v2 keys per-pair state by `id(param)`, so DKV accumulates
    # the union of UK-side and UV-side updates each step.
    coupled_mla_kv: list[tuple[nn.Parameter, nn.Parameter, int]] = field(default_factory=list)
    coupled_mla_q: list[tuple[nn.Parameter, nn.Parameter, int]] = field(default_factory=list)
    coupled_factff: list[tuple[nn.Parameter, nn.Parameter, int]] = field(default_factory=list)
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


def _bucket_key(name: str, layer_idx: str) -> str:
    """Compose a unique bucket key per (layer, sub-module-context).

    Without sub-context, every per-expert MoE pair collapses onto a single
    (layer_idx, role) bucket and all but the last expert end up frozen
    (audit point #1). For paths under `experts.{e}.` or `shared_expert.`,
    suffix the layer key with the expert identity so each expert gets its
    own (q,k,v,o,up,down,gate) bucket."""
    s = "." + name
    m_exp = _EXPERT_RE.search(s)
    if m_exp is not None:
        return f"{layer_idx}/expert_{m_exp.group(1)}"
    if _SHARED_EXPERT_RE.search(s) is not None:
        return f"{layer_idx}/shared_expert"
    return layer_idx


def classify_parameters(
    model: nn.Module,
    *,
    couple_qk: bool = True,
    couple_vo: bool = True,
    couple_updown: bool = True,
    couple_mla: bool = True,
    couple_factff: bool = True,
    n_heads: int = 1,
    n_kv_heads: int | None = None,
    couple_router_to_muon: bool = True,
) -> ParamGroups:
    """Walk model.named_parameters() and split into Coupled-Muon groups.

    A parameter is a candidate for Muon/coupling iff it is 2-D and is *not* an
    embedding or LM head (those run AdamW).

    `couple_router_to_muon`: if True (default, per experiment.md c.3 — matches
    Moonlight 2502.16982 §2.2 and Cerebras nanoMoE, both of which put MoE
    router weights under Muon), MoE router weights flow into the plain Muon
    path. If False, routers go to AdamW (rung J ablation; DeepSeek-V2/V3-
    and OLMoE-style).
    """
    groups = ParamGroups()
    n_kv = n_kv_heads or n_heads

    layer_buckets: dict[str, dict[str, nn.Parameter]] = {}
    other_2d: list[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Routers: AdamW by default (c.3); if rung J's flag is set, route to Muon.
        if "router" in name or "gate_router" in name:
            if couple_router_to_muon and param.ndim == 2:
                groups.muon_2d.append(param)
            else:
                groups.router_params.append(param)
            continue
        # Embeddings / LM head / 1D params -> AdamW.
        if param.ndim < 2 or "embed" in name or "lm_head" in name or "wte" in name or "wpe" in name:
            groups.adamw_other.append(param)
            continue

        # 2D matrix params: try to bucket by (layer + sub-context) + role.
        layer_idx = _layer_idx(name)
        role = _proj_role(name) if layer_idx is not None else None
        if layer_idx is None or role is None:
            other_2d.append(param)
            continue

        bucket = _bucket_key(name, layer_idx)
        layer_buckets.setdefault(bucket, {})[role] = param

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

        # MLA KV pairs (Phase-2). DKV is the latent down-projection shared by
        # the K-side (UK) and V-side (UV) up-projections. CoupledMuon_v2's
        # per-step `processed` set skips any tuple whose A or B was already
        # updated, so DKV cannot appear as B-partner in TWO coupled pairs in
        # the same step — the second would silently no-op (and UV would never
        # receive any update). Pick the K-side as the coupled partner of DKV
        # (the (W_UK, W_DKV) pair is the structural analogue of LoRA's (B, A)
        # for the K matrix), and route UV through plain Muon. Phase-2.5 will
        # explore a 3-way (UK, UV, DKV) joint coupling but that requires an
        # algorithmic change to the inner NS loop. `mla_kr_proj` (the always-
        # RoPE'd K side) is a flat x→head_dim matrix with no factored partner;
        # also routed to plain Muon.
        if couple_mla and "mla_dkv_proj" in params and "mla_uk_proj" in params:
            dkv = params["mla_dkv_proj"]
            groups.coupled_mla_kv.append((params["mla_uk_proj"], dkv, n_kv))
            if "mla_uv_proj" in params:
                groups.muon_2d.append(params["mla_uv_proj"])
        else:
            for r in ("mla_dkv_proj", "mla_uk_proj", "mla_uv_proj"):
                if r in params:
                    groups.muon_2d.append(params[r])
        if "mla_kr_proj" in params:
            groups.muon_2d.append(params["mla_kr_proj"])

        # MLA Q pair (only when q_lora_rank > 0; otherwise the regular q_proj
        # branch above handles attention's query side).
        if couple_mla and "mla_uq_proj" in params and "mla_dq_proj" in params:
            groups.coupled_mla_q.append((params["mla_uq_proj"], params["mla_dq_proj"], n_heads))
        else:
            for r in ("mla_uq_proj", "mla_dq_proj"):
                if r in params:
                    groups.muon_2d.append(params[r])

        # Imposed-FFN-factorisation pair (P_factff rung; Tier B3).
        if couple_factff and "factff_up_proj" in params and "factff_down_proj" in params:
            up = params["factff_up_proj"]
            dn = params["factff_down_proj"]
            groups.coupled_factff.append((dn, up, up.size(0)))
        else:
            for r in ("factff_up_proj", "factff_down_proj"):
                if r in params:
                    groups.muon_2d.append(params[r])

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

    # Per experiment.md d.3 + Bergsma 2025 (arXiv 2512.05620): "Combining μP with
    # 1/width independent weight decay, Muon and Shampoo achieve consistent 1.4×
    # and 1.3× speedups". When wd_per_width is true, divide the configured wd
    # by hidden width before passing to the optimizer.
    hidden = int(cfg.model.hidden)
    wd_raw = float(opt.wd)
    wd = wd_raw / hidden if bool(opt.get("wd_per_width", False)) else wd_raw

    if opt.type == "adamw":
        return torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=float(opt.lr),
            weight_decay=wd,
            betas=tuple(opt.betas),
            eps=float(opt.eps),
        )

    if opt.type == "muon":
        groups = classify_parameters(
            model,
            couple_qk=False,
            couple_vo=False,
            couple_updown=False,
            couple_mla=False,
            couple_factff=False,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            couple_router_to_muon=bool(opt.get("couple_router_to_muon", True)),
        )
        # All matrix candidates flow into muon_params.
        muon_params = list(groups.muon_2d)
        adamw_params = list(groups.adamw_other) + list(groups.router_params)
        return Muon(
            muon_params=muon_params,
            adamw_params=adamw_params,
            lr=float(opt.lr),
            wd=wd,
            momentum=float(opt.momentum),
            ns_steps=int(opt.ns_steps),
            adamw_betas=tuple(opt.betas),
            adamw_eps=float(opt.eps),
            ns_dtype=_resolve_ns_dtype(opt.get("ns_dtype", "bf16")),
            ns_coefficients=str(opt.get("ns_coefficients", "bernstein")),
            ns_gram_form=bool(opt.get("ns_gram_form", False)),
            lr_prefactor=str(opt.get("lr_prefactor", "moonlight")),
        )

    if opt.type == "coupled_muon_v2":
        groups = classify_parameters(
            model,
            couple_qk=bool(opt.couple_qk),
            couple_vo=bool(opt.couple_vo),
            couple_updown=bool(opt.couple_updown),
            couple_mla=bool(opt.get("couple_mla", True)),
            couple_factff=bool(opt.get("couple_factff", True)),
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            couple_router_to_muon=bool(opt.get("couple_router_to_muon", True)),
        )
        # Build the list of (A, B, n_heads_for_pair, is_qk) tuples.
        coupled_pairs: list[tuple[nn.Parameter, nn.Parameter, int, bool]] = []
        for a, b, h in groups.coupled_vo:
            coupled_pairs.append((a, b, h, False))
        for a, b, h in groups.coupled_updown:
            coupled_pairs.append((a, b, h, False))
        for a, b, h in groups.coupled_qk:
            coupled_pairs.append((a, b, h, True))
        # Phase-2 factored-architecture pairs. MLA pairs use is_qk=False
        # (the multi-head Q-K reshape path requires the 2-D RoPE rotation
        # block structure, which MLA's split nope/rope head layout doesn't
        # match; flat-2D coupling is the correct path). factff also flat-2D.
        for a, b, h in groups.coupled_mla_kv:
            coupled_pairs.append((a, b, h, False))
        for a, b, h in groups.coupled_mla_q:
            coupled_pairs.append((a, b, h, False))
        for a, b, h in groups.coupled_factff:
            coupled_pairs.append((a, b, h, False))

        # The multi-head Q-K branch in coupled_muon.py reshapes head_dim into
        # 2-D RoPE rotation pairs. That structure only exists with full RoPE.
        # Disable use_multi_head when:
        #   - pos_emb.type == "learned" (no rotation at all), or
        #   - rope_partial_frac < 1 (partial RoPE; tail channels are unrotated).
        # Per experiment.md a.4 / b.2 — rung C ('learned-pos') and modded-style
        # half-RoPE both fall under this guard.
        use_multi_head = bool(opt.get("use_multi_head", False))
        pos_emb_type = str(cfg.model.pos_emb.get("type", "rope"))
        rope_partial_frac = float(cfg.model.attn.get("rope_partial_frac", 1.0))
        rope_unsafe = rope_partial_frac < 1.0 or pos_emb_type != "rope"
        if use_multi_head and rope_unsafe:
            reason = (
                f"pos_emb.type={pos_emb_type!r}"
                if pos_emb_type != "rope"
                else f"rope_partial_frac={rope_partial_frac}"
            )
            warnings.warn(
                f"Disabling use_multi_head: the QK 2-D rotation-pair reshape requires "
                f"full RoPE, but {reason}. V-O multi-head coupling is also disabled "
                f"(global flag). Set pos_emb.type=rope and rope_partial_frac=1.0 to "
                f"re-enable.",
                stacklevel=2,
            )
            use_multi_head = False

        return CoupledMuon_v2(
            lr=float(opt.lr),
            wd=wd,
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
            # Two-stage composition toggle: True (default) preserves the
            # Muon(C_A(g; B)) update — the operating point that produced the
            # LLaMA-60M result. False is the stage-1-only ablation (d.4).
            final_polish=bool(opt.get("final_polish", True)),
            use_multi_head=use_multi_head,
            n_heads=n_heads,
            ns_dtype=_resolve_ns_dtype(opt.get("ns_dtype", "bf16")),
            # Phase-2 NS / LR-prefactor knobs.
            ns_coefficients=str(opt.get("ns_coefficients", "bernstein")),
            ns_gram_form=bool(opt.get("ns_gram_form", False)),
            lr_prefactor=str(opt.get("lr_prefactor", "moonlight")),
        )

    raise ValueError(f"Unknown optimizer.type={opt.type!r}")
