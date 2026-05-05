"""Coupled-pair probe: condition number and σ_max of W_A W_B.

Per experiment.md d.3: "compute condition number κ(W_A W_B) and Frobenius
norm ‖W_A W_B‖_F, plus σ_max(W_A W_B). Coupling should *flatten* κ."
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ..optim.factory import classify_parameters


@torch.no_grad()
def coupled_pair_probe(
    model: nn.Module,
    _optimizer: torch.optim.Optimizer,
    ctx: dict[str, Any],
) -> dict[str, Any]:
    n_heads = int(ctx.get("n_heads", 1))
    n_kv_heads = int(ctx.get("n_kv_heads", n_heads))
    groups = classify_parameters(model, n_heads=n_heads, n_kv_heads=n_kv_heads)
    out: dict[str, Any] = {}

    def report(prefix: str, idx: int, A: torch.Tensor, B: torch.Tensor) -> None:
        # Compose product on the smaller side to limit memory.
        a32 = A.detach().float()
        b32 = B.detach().float()
        # A @ B is the natural product for our naming convention (e.g. o_proj @ v_proj).
        try:
            prod = a32 @ b32
        except RuntimeError:
            return
        sv = torch.linalg.svdvals(prod.cpu() if max(prod.shape) > 1024 else prod)
        kappa = (sv.max() / sv.min().clamp_min(1e-12)).item()
        out[f"{prefix}.{idx}"] = {
            "frob": prod.norm().item(),
            "sigma_max": sv.max().item(),
            "kappa": kappa,
        }

    for i, (a, b, _h) in enumerate(groups.coupled_qk):
        report("qk", i, a, b)
    for i, (a, b, _h) in enumerate(groups.coupled_vo):
        report("vo", i, a, b)
    for i, (a, b, _h) in enumerate(groups.coupled_updown):
        report("updown", i, a, b)
    return out
