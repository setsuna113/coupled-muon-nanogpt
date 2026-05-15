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
    rope_partial_frac: float = 1.0  # fraction of head_dim that gets RoPE'd (modded-style 0.5)
    max_seq_len: int = 4096
    # Phase-2 MLA fields (DeepSeek-V2/V3 style; rung O_mla). When `attn_type`
    # is "mha", these are ignored. When "mla", the (W_DKV, W_UK), (W_DKV,
    # W_UV), and optional (W_DQ, W_UQ) factored pairs are exposed via named
    # projections that ``optim.factory.classify_parameters`` picks up.
    attn_type: str = "mha"  # "mha" | "mla"
    kv_lora_rank: int = 0
    q_lora_rank: int = 0  # 0 ⇒ full-rank Q proj (no factored Q pair)
    qk_nope_head_dim: int = 0
    qk_rope_head_dim: int = 0
    v_head_dim: int = 0


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
            # Resolve rotary_dim to even fraction of head_dim.
            rotary_dim = int(self.head_dim * cfg.rope_partial_frac) & ~1
            rotary_dim = max(2, min(self.head_dim, rotary_dim))
            self.rotary_dim = rotary_dim
            self.rope = RotaryEmbedding(
                self.head_dim,
                base=cfg.rope_base,
                max_seq_len=cfg.max_seq_len,
                rotary_dim=rotary_dim,
            )
        elif cfg.pos_emb_type == "learned":
            self.rope = None
            self.rotary_dim = 0
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


class MultiLatentAttention(nn.Module):
    """DeepSeek-V2/V3-style Multi-head Latent Attention.

    KV is compressed through a shared latent ``kv_c = x @ W_DKV.T`` of size
    ``kv_lora_rank``, then up-projected to per-head K (the "no-PE" channels)
    and V via ``W_UK`` and ``W_UV``. A separate flat projection ``W_KR`` maps
    ``x → (qk_rope_head_dim)``; this is then broadcast across heads and given
    RoPE — the DeepSeek "decoupled rotary K" pattern that lets KV-compression
    coexist with positional encoding. Q is optionally low-rank-factored via
    ``W_DQ``/``W_UQ`` when ``q_lora_rank > 0``; else the full-rank ``q_proj``
    handles the query side.

    Pair convention (read by ``optim.factory.classify_parameters``):

      - (W_UK,   W_DKV) — coupled_mla_kv  (the K-side factored pair)
      - (W_UQ,   W_DQ)  — coupled_mla_q   (only when q_lora_rank > 0)

    ``W_UV`` is **not** in a coupled pair: ``CoupledMuon_v2.step``'s per-step
    ``processed`` set rules out two coupled pairs sharing a B-partner
    (otherwise the second would silently no-op and UV would never update).
    The K-side is the structural analogue of LoRA's (B, A) for the K matrix,
    so we couple (W_UK, W_DKV) and route ``W_UV`` through plain Muon. A 3-way
    (UK, UV, DKV) joint coupling kernel is Phase-2.5 work.

    All other 2D MLA projections (``W_KR``, ``q_proj`` when q_lora_rank == 0,
    ``W_UV``, ``o_proj``) flow into plain Muon. The total per-head QK channel count is
    ``qk_nope_head_dim + qk_rope_head_dim``; per-head V channel count is
    ``v_head_dim``. The output projection ``o_proj`` maps
    ``(n_heads * v_head_dim) → hidden`` (V head-dim can differ from QK).
    """

    def __init__(self, cfg: AttnConfig):
        super().__init__()
        self.cfg = cfg
        self.n_heads = cfg.n_heads
        self.kv_lora_rank = int(cfg.kv_lora_rank)
        self.q_lora_rank = int(cfg.q_lora_rank)
        self.qk_nope_head_dim = int(cfg.qk_nope_head_dim)
        self.qk_rope_head_dim = int(cfg.qk_rope_head_dim)
        self.v_head_dim = int(cfg.v_head_dim)
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim

        assert self.kv_lora_rank > 0, "MLA requires kv_lora_rank > 0"
        assert self.qk_nope_head_dim > 0 and self.qk_rope_head_dim > 0, (
            "MLA requires positive qk_nope_head_dim and qk_rope_head_dim"
        )
        assert self.v_head_dim > 0, "MLA requires v_head_dim > 0"
        assert self.qk_rope_head_dim % 2 == 0, (
            f"qk_rope_head_dim must be even for RoPE pairing, "
            f"got {self.qk_rope_head_dim}"
        )

        H = cfg.hidden_size
        # KV-side factored pair: x → kv_c (latent) → k_nope / v (per head).
        self.mla_dkv_proj = nn.Linear(H, self.kv_lora_rank, bias=False)
        self.mla_uk_proj = nn.Linear(self.kv_lora_rank, self.n_heads * self.qk_nope_head_dim, bias=False)
        self.mla_uv_proj = nn.Linear(self.kv_lora_rank, self.n_heads * self.v_head_dim, bias=False)
        # Decoupled rotary-K side: flat x → qk_rope_head_dim, broadcast over heads.
        self.mla_kr_proj = nn.Linear(H, self.qk_rope_head_dim, bias=False)

        # Q-side: optional low-rank factored pair vs full-rank Q proj.
        if self.q_lora_rank > 0:
            self.mla_dq_proj = nn.Linear(H, self.q_lora_rank, bias=False)
            self.mla_uq_proj = nn.Linear(self.q_lora_rank, self.n_heads * self.qk_head_dim, bias=False)
            self.q_proj = None
        else:
            self.q_proj = nn.Linear(H, self.n_heads * self.qk_head_dim, bias=False)
            self.mla_dq_proj = None
            self.mla_uq_proj = None

        self.o_proj = nn.Linear(self.n_heads * self.v_head_dim, H, bias=False)

        # RoPE over the qk_rope channels only (DeepSeek "decoupled rotary K").
        self.rope = RotaryEmbedding(
            self.qk_rope_head_dim,
            base=cfg.rope_base,
            max_seq_len=cfg.max_seq_len,
            rotary_dim=self.qk_rope_head_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        return_max_logit: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T, _ = x.shape
        # KV compression and up-projection.
        kv_c = self.mla_dkv_proj(x)  # (B, T, kv_lora_rank)
        k_nope = self.mla_uk_proj(kv_c).view(B, T, self.n_heads, self.qk_nope_head_dim)
        v = self.mla_uv_proj(kv_c).view(B, T, self.n_heads, self.v_head_dim)

        # Decoupled rotary K (broadcast over heads). RotaryEmbedding expects
        # (B, T, H, head_dim); we add a singleton head axis and expand below.
        k_rope = self.mla_kr_proj(x).view(B, T, 1, self.qk_rope_head_dim)
        k_rope = self.rope(k_rope)  # rotated
        k_rope = k_rope.expand(B, T, self.n_heads, self.qk_rope_head_dim).contiguous()

        # Q side.
        if self.mla_uq_proj is not None:
            q_full = self.mla_uq_proj(self.mla_dq_proj(x))
        else:
            q_full = self.q_proj(x)
        q_full = q_full.view(B, T, self.n_heads, self.qk_head_dim)
        q_nope, q_rope = q_full.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_rope = self.rope(q_rope.contiguous())

        # Concatenate nope + rope channels per head.
        q = torch.cat([q_nope, q_rope], dim=-1)  # (B, T, n_heads, qk_head_dim)
        k = torch.cat([k_nope, k_rope], dim=-1)

        # (B, n_heads, T, head_dim) for SDPA.
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        max_logit = None
        if return_max_logit:
            scale = 1.0 / (self.qk_head_dim ** 0.5)
            logits = torch.matmul(q, k.transpose(-2, -1)) * scale
            mask = torch.ones(T, T, device=x.device, dtype=torch.bool).tril()
            logits = logits.masked_fill(~mask, float("-inf"))
            max_logit = logits.amax().detach()
            attn = F.softmax(logits.float(), dim=-1).to(v.dtype)
            y = torch.matmul(attn, v)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.v_head_dim)
        return self.o_proj(y), max_logit
