"""SVD probe: top-k singular values and SVD entropy per matrix parameter.

Reports per-param: weight Frobenius norm, top-k singular values, SVD entropy
H(σ) = −Σ p_i log p_i with p_i = σ_i² / Σ σ_j² (Moonlight Figure 4 metric).

Computed in fp32 on a copy of the weight; runs on CPU for params > 1024² to
avoid GPU OOM in the smoke regime.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


@torch.no_grad()
def svd_probe(
    model: nn.Module,
    _optimizer: torch.optim.Optimizer,
    _ctx: dict[str, Any],
    top_k: int = 5,
    max_dim_for_gpu: int = 1024,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad or p.ndim != 2:
            continue
        w = p.detach().float()
        moved_to_cpu = max(w.shape) > max_dim_for_gpu
        if moved_to_cpu:
            w = w.cpu()
        try:
            sv = torch.linalg.svdvals(w)
        except Exception:  # noqa: BLE001
            continue
        sv2 = sv.pow(2)
        denom = sv2.sum().clamp_min(1e-12)
        prob = sv2 / denom
        entropy = -(prob * (prob + 1e-12).log()).sum().item()
        out[name] = {
            "frob": p.detach().float().norm().item(),
            "top_sv": sv[: top_k].cpu().tolist(),
            "svd_entropy": entropy,
        }
    return out
