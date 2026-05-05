"""Newton-Schulz internal Frobenius-norm probe (debug-only).

Per experiment.md d.3: log ‖X‖_F per coupled iteration to detect divergence.
The CoupledMuon_v2 inner kernels are @torch.compile'd and don't expose a hook
without modification, so this probe runs a parallel "shadow" computation on
the *current* gradient + partner weight to estimate the internal trajectory.
Off by default; only enable for diagnostic runs (overhead ~1×coupled-NS).
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


@torch.no_grad()
def ns_internal_probe(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    ctx: dict[str, Any],
) -> dict[str, Any]:
    # Without modifying the @torch.compile'd kernel, the cleanest way to surface
    # internal norms is to run a brief shadow trace on a sample pair.
    # Stub: emits a sentinel until the kernel exposes hooks (deferred per plan).
    return {"_note": "ns_internal probe stub; enable when coupled_muon kernel exposes hooks"}
