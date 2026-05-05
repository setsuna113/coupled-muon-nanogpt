"""Norm + positional embedding primitives.

Toggleable to support ladder rungs: RMSNorm vs LayerNorm, RoPE vs learned-pos.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute in fp32 for stability, cast back.
        dt = x.dtype
        x32 = x.float()
        rms = x32.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x32 * rms).to(dt) * self.weight


def make_norm(dim: int, kind: str, eps: float = 1e-6) -> nn.Module:
    if kind == "rmsnorm":
        return RMSNorm(dim, eps=eps)
    if kind == "layernorm":
        return nn.LayerNorm(dim, eps=eps, bias=True)
    raise ValueError(f"Unknown norm.kind={kind!r}")


class RotaryEmbedding(nn.Module):
    """LLaMA-style RoPE applied to the full head dim.

    Frequency convention: theta_i = base^(-2i/head_dim), i in [0, head_dim/2).
    Pair convention: channels are split as `[first_half, second_half]` per head,
    matching the GPT-NeoX / LLaMA layout (NOT the interleaved layout). This is
    the same convention the Coupled-Muon multi-head reshape expects.
    """

    def __init__(self, head_dim: int, base: float = 10000.0, max_seq_len: int = 8192):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE requires even head_dim"
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)  # (T, head_dim/2)
        # Cache cos/sin in fp32; rebroadcast at apply time.
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)
        self.head_dim = head_dim

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        # x: (B, T, n_heads, head_dim).
        T = x.size(1)
        cos = self.cos[offset : offset + T]  # (T, head_dim/2)
        sin = self.sin[offset : offset + T]
        # Split the head dim into two halves and rotate.
        x1, x2 = x.float().chunk(2, dim=-1)  # each (B, T, n_heads, head_dim/2)
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
        rotated = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
        return rotated.to(x.dtype)
