"""Frozen legacy-optimizer reconstruction helpers (review v3.8).

These helpers are the load-bearing reference for the Phase-2.1 invariant
``nonfull_rope_policy=current_flat2d_fallback`` produces bitwise-identical
updates to the pre-refactor factory's "warn-and-disable use_multi_head →
fall through to flat-2D" path. Without a frozen reference, the equivalence
test cannot survive a future refactor of `optim/factory.py`.

The helpers reconstruct the exact CoupledMuon_v2 kwargs the pre-refactor
factory.py emitted under the relevant scenarios:
  - Learned positional embeddings (rope_status == "none")
  - Partial RoPE (rope_status == "partial")
  - Full RoPE                                  (rope_status == "full")
All other knobs (couple_qk/vo/updown, ns_coefficients, lr_prefactor, etc.)
take their pre-refactor defaults from configs/base.yaml.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from coupled_muon_nanogpt.optim.coupled_muon import CoupledMuon_v2
from coupled_muon_nanogpt.optim.factory import classify_parameters


def build_legacy_nonfull_flat2d_optimizer(
    model: nn.Module,
    *,
    lr: float = 1e-3,
    momentum: float = 0.95,
    nesterov: bool = True,
    coupled_steps: int = 4,
    ns_steps: int = 5,
    n_heads: int = 4,
    n_kv_heads: int | None = None,
    ns_dtype=None,
) -> CoupledMuon_v2:
    """Reconstruct the pre-Phase-2.1 factory path for rope_status in {partial, none}.

    Pre-refactor factory.py:367-380 emitted a warning then set
    ``use_multi_head=False``; the rest of the kwargs were the standard
    base.yaml defaults. This helper bypasses the factory and constructs
    CoupledMuon_v2 directly with those exact kwargs PLUS the new Phase-2.1
    knobs set to their bitwise-equivalent defaults
    (``qk_coupling.{full_rope, no_rope_policy, partial_rope_policy}`` =
    legacy_rope2d / current_flat2d_fallback / current_flat2d_fallback;
    ``rope_status`` is whatever the model has; the optimizer ignores the
    ``no/partial_rope_policy`` value when use_multi_head=False).
    """
    groups = classify_parameters(
        model,
        couple_qk=True,
        couple_vo=True,
        couple_updown=True,
        couple_mla=True,
        couple_factff=True,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        couple_router_to_muon=True,
    )
    coupled_pairs: list[tuple] = []
    for a, b, h in groups.coupled_vo:
        coupled_pairs.append((a, b, h, False))
    for a, b, h in groups.coupled_updown:
        coupled_pairs.append((a, b, h, False))
    for a, b, h in groups.coupled_qk:
        coupled_pairs.append((a, b, h, True))
    for a, b, h in groups.coupled_mla_kv:
        coupled_pairs.append((a, b, h, False))
    for a, b, h in groups.coupled_mla_q:
        coupled_pairs.append((a, b, h, False))
    for a, b, h in groups.coupled_factff:
        coupled_pairs.append((a, b, h, False))

    return CoupledMuon_v2(
        lr=lr,
        wd=0.0,
        coupled_pairs=coupled_pairs,
        muon_params=list(groups.muon_2d),
        adamw_params=list(groups.adamw_other) + list(groups.router_params),
        momentum=momentum,
        nesterov=nesterov,
        ns_steps=ns_steps,
        coupled_steps=coupled_steps,
        adamw_betas=(0.95, 0.95),
        adamw_eps=1e-8,
        enable_coupled=True,
        final_polish=True,
        use_multi_head=False,  # ← legacy: factory disabled this under rope_unsafe
        ns_dtype=ns_dtype if ns_dtype is not None else torch.bfloat16,
        n_heads=n_heads,
        ns_coefficients="bernstein",
        ns_gram_form=False,
        lr_prefactor="moonlight",
        # Phase-2.1 knobs at defaults that preserve legacy behavior. Since
        # use_multi_head=False, the qk_coupling dispatch is bypassed entirely
        # (the optimizer's else-branch at coupled_muon.py:line ~ flat-2D path
        # handles is_qk=True with use_multi_head=False the same as any other
        # pair).
        qk_coupling_full_rope="legacy_rope2d",
        qk_coupling_no_rope_policy="current_flat2d_fallback",
        qk_coupling_partial_rope_policy="current_flat2d_fallback",
        rope_status="none",  # value doesn't matter when use_multi_head=False
        rotary_dim=None,
    )
