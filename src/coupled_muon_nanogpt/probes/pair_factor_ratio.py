"""Pair-factor Frobenius ratio probe: ‖W_A‖_F / ‖W_B‖_F per coupled pair.

The LoRA-RITE one-factor-dominates pathology (Yen et al., NeurIPS 2025;
suggestion.md §2.7) predicts that non-invariant optimizers cause one factor to
grow disproportionately during training. Coupled Muon v2 is *not* fully
transformation-invariant under B → BS, A → S⁻¹A (Riemannian Preconditioned
LoRA / LoRA-RITE both have stronger guarantees). The cheap empirical test is
to log ‖W_A‖_F / ‖W_B‖_F per coupled pair over training and check whether the
ratio stays bounded or drifts.

Zero compute cost (two ``.norm()`` calls per pair). Registered in train.py
when ``train.probes.pair_factor_ratio_interval_tokens > 0``. The probe must
mirror the *live* optimizer's coupling flags so it only reports pairs the
optimizer actually couples — train.py threads `couple_qk`, `couple_vo`,
`couple_updown`, `couple_mla`, `couple_factff`, `couple_router_to_muon`
through `ctx`. When the keys are absent (e.g., direct test invocation) the
probe falls back to `classify_parameters`'s defaults.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ..optim.factory import classify_parameters


@torch.no_grad()
def pair_factor_ratio_probe(
    model: nn.Module,
    _optimizer: torch.optim.Optimizer,
    ctx: dict[str, Any],
) -> dict[str, Any]:
    n_heads = int(ctx.get("n_heads", 1))
    n_kv_heads = int(ctx.get("n_kv_heads", n_heads))
    groups = classify_parameters(
        model,
        couple_qk=bool(ctx.get("couple_qk", True)),
        couple_vo=bool(ctx.get("couple_vo", True)),
        couple_updown=bool(ctx.get("couple_updown", True)),
        couple_mla=bool(ctx.get("couple_mla", True)),
        couple_factff=bool(ctx.get("couple_factff", True)),
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        couple_router_to_muon=bool(ctx.get("couple_router_to_muon", True)),
    )
    out: dict[str, Any] = {}

    def report(prefix: str, idx: int, A: torch.Tensor, B: torch.Tensor) -> None:
        a_frob = A.detach().float().norm().item()
        b_frob = B.detach().float().norm().item()
        ratio = a_frob / max(b_frob, 1e-12)
        out[f"{prefix}.{idx}"] = {
            "A_frob": a_frob,
            "B_frob": b_frob,
            "ratio_A_over_B": ratio,
        }

    for i, (a, b, _h) in enumerate(groups.coupled_qk):
        report("qk", i, a, b)
    for i, (a, b, _h) in enumerate(groups.coupled_vo):
        report("vo", i, a, b)
    for i, (a, b, _h) in enumerate(groups.coupled_updown):
        report("updown", i, a, b)
    for i, (a, b, _h) in enumerate(getattr(groups, "coupled_mla_kv", [])):
        report("mla_kv", i, a, b)
    for i, (a, b, _h) in enumerate(getattr(groups, "coupled_mla_q", [])):
        report("mla_q", i, a, b)
    for i, (a, b, _h) in enumerate(getattr(groups, "coupled_factff", [])):
        report("factff", i, a, b)
    return out
