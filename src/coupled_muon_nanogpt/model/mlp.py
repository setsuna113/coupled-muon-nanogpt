"""MLP variants: SwiGLU | GELU-2mat | ReLU² | FactFF (imposed factorization).

Param naming (required by the optimizer factory):
- 2-mat (gelu_2mat, relu2): `up_proj`, `down_proj`
- 3-mat (swiglu): `gate_proj`, `up_proj`, `down_proj`
- FactFF (Phase-2 P_factff rung): `factff_up_proj`, `factff_down_proj`

For coupled-muon, the up–down pair is coupled; gate (when present) is sent to
plain Muon — gate-Jacobian-ignoring approximation per experiment.md a.3.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class GELU2Mat(nn.Module):
    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.gelu(self.up_proj(x), approximate="tanh"))


class ReLU2(nn.Module):
    """ReLU² activation as in modded-nanogpt."""

    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        self.up_proj = nn.Linear(hidden, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.up_proj(x))
        return self.down_proj(h * h)


class FactorizedMLP(nn.Module):
    """Imposed rank-r factorisation of a hidden→hidden FFN (Phase-2 P_factff).

    The single learnable composition is ``factff_down @ factff_up``. There is
    no gate — the activation is ReLU² between the two mats (matches the
    modded-nanogpt 2-mat MLP convention while keeping the operator-norm-only
    composition clean). The optimizer factory picks up the two named projs and
    couples them as a (down, up) pair, exercising CoupledMuon's distinctive
    geometry on an *imposed* factor structure (suggestion.md B3 / §245).

    Tier-B3 is the cleanest "does CoupledMuon's advantage transfer beyond
    natively-factored architectures (MLA, LoRA, KDA)?" experiment.
    """

    def __init__(self, hidden: int, rank: int):
        super().__init__()
        assert rank > 0, f"FactorizedMLP rank must be positive, got {rank}"
        self.factff_up_proj = nn.Linear(hidden, rank, bias=False)
        self.factff_down_proj = nn.Linear(rank, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.factff_up_proj(x))
        return self.factff_down_proj(h * h)


def make_mlp(hidden: int, intermediate: int, kind: str, *, factorize_rank: int = 0) -> nn.Module:
    """Build an MLP block.

    Args:
        hidden: model hidden size.
        intermediate: inner dimension for SwiGLU/GELU-2mat/ReLU². Ignored when
            ``kind="factff"`` (the rank is the bottleneck).
        kind: ``swiglu`` | ``gelu_2mat`` | ``relu2`` | ``factff``.
        factorize_rank: bottleneck rank for ``factff``; ignored otherwise.
    """
    if kind == "swiglu":
        return SwiGLU(hidden, intermediate)
    if kind == "gelu_2mat":
        return GELU2Mat(hidden, intermediate)
    if kind == "relu2":
        return ReLU2(hidden, intermediate)
    if kind == "factff":
        if factorize_rank <= 0:
            raise ValueError(
                f"mlp.type=factff requires mlp.factorize_rank > 0, got {factorize_rank}"
            )
        return FactorizedMLP(hidden, factorize_rank)
    raise ValueError(f"Unknown mlp.type={kind!r}")
