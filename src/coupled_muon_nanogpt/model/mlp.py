"""MLP variants: SwiGLU | GELU-2mat | ReLU².

Param naming (required by the optimizer factory):
- 2-mat (gelu_2mat, relu2): `up_proj`, `down_proj`
- 3-mat (swiglu): `gate_proj`, `up_proj`, `down_proj`

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


def make_mlp(hidden: int, intermediate: int, kind: str) -> nn.Module:
    if kind == "swiglu":
        return SwiGLU(hidden, intermediate)
    if kind == "gelu_2mat":
        return GELU2Mat(hidden, intermediate)
    if kind == "relu2":
        return ReLU2(hidden, intermediate)
    raise ValueError(f"Unknown mlp.type={kind!r}")
