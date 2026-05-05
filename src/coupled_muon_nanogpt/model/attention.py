"""Causal multi-head attention with toggleable RoPE/learned-pos, QK-Norm, GQA.

Param naming convention (required by the optimizer factory):
- `q_proj.weight`, `k_proj.weight`, `v_proj.weight`, `o_proj.weight`

Each `nn.Linear(hidden, n_heads*head_dim, bias=False)`. Weight shape under PyTorch
convention is `(out, in) = (n_heads*head_dim, hidden)` for q/k/v and
`(hidden, n_heads*head_dim)` for o_proj.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .components import RMSNorm, RotaryEmbedding


@dataclass
class AttnConfig:
    hidden_size: int
    n_heads: int
    n_kv_heads: int | None = None  # None ⇒ MHA
    head_dim: int | None = None  # default: hidden_size // n_heads
    qk_norm: bool = False
    pos_emb_type: str = "rope"  # "rope" | "learned"
    rope_base: float = 10000.0
    max_seq_len: int = 4096


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: AttnConfig):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads or cfg.n_heads
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.head_dim = cfg.head_dim or (cfg.hidden_size // cfg.n_heads)
        assert self.head_dim * self.n_heads == cfg.hidden_size or cfg.head_dim is not None
        self.qk_norm = cfg.qk_norm
        self.pos_emb_type = cfg.pos_emb_type

        self.q_proj = nn.Linear(cfg.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, cfg.hidden_size, bias=False)

        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        if cfg.pos_emb_type == "rope":
            self.rope = RotaryEmbedding(self.head_dim, base=cfg.rope_base, max_seq_len=cfg.max_seq_len)
        elif cfg.pos_emb_type == "learned":
            self.rope = None
        else:
            raise ValueError(f"Unknown pos_emb_type={cfg.pos_emb_type!r}")

    def forward(
        self,
        x: torch.Tensor,
        return_max_logit: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)

        # (B, n_heads, T, head_dim) layout for SDPA.
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # GQA repeat-broadcast: each kv-head serves n_heads / n_kv_heads query heads.
        if self.n_kv_heads != self.n_heads:
            n_rep = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        max_logit = None
        if return_max_logit:
            # Compute manually so we can probe the pre-softmax max.
            scale = 1.0 / (self.head_dim**0.5)
            logits = torch.matmul(q, k.transpose(-2, -1)) * scale
            mask = torch.ones(T, T, device=x.device, dtype=torch.bool).tril()
            logits = logits.masked_fill(~mask, float("-inf"))
            max_logit = logits.amax().detach()
            attn = F.softmax(logits.float(), dim=-1).to(v.dtype)
            y = torch.matmul(attn, v)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        return self.o_proj(y), max_logit
