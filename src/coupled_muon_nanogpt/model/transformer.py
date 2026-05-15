"""Toggleable NanoGPT-style transformer.

Top-level naming (required by the optimizer factory):
  embed.weight                 — token embedding (AdamW)
  pos_emb.weight               — learned positional embedding (AdamW), if pos_emb=learned
  layers.{i}.norm1, norm2      — pre-norms
  layers.{i}.attn.{q,k,v,o}_proj.weight  — attention projections
  layers.{i}.mlp.{up,down,gate}_proj.weight  — dense MLP
  layers.{i}.mlp.experts.{e}.{up,down,gate}_proj.weight  — MoE expert MLPs
  layers.{i}.mlp.gate_router.weight  — MoE router (AdamW)
  norm_out                     — final norm
  lm_head.weight               — logits (tied to embed; AdamW)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import AttnConfig, CausalSelfAttention, MultiLatentAttention
from .components import make_norm
from .mlp import make_mlp
from .moe import MoEConfig, MoEFFN


@dataclass
class BlockConfig:
    hidden: int
    n_heads: int
    n_kv_heads: int | None = None
    head_dim: int | None = None
    intermediate: int = 0  # inferred if 0
    norm_type: str = "rmsnorm"
    norm_eps: float = 1e-6
    mlp_type: str = "swiglu"
    pos_emb_type: str = "rope"
    rope_base: float = 10000.0
    rope_partial_frac: float = 1.0  # 0.5 = modded-nanogpt half-RoPE
    qk_norm: bool = False
    max_seq_len: int = 4096
    moe_enabled: bool = False
    moe_cfg: dict[str, Any] = field(default_factory=dict)
    # Phase-2 attention type + MLA dims (suggestion.md S2 / §2.4). `attn_type`
    # defaults to "mha" → existing CausalSelfAttention path. When "mla", the
    # MLA dims below must all be positive.
    attn_type: str = "mha"
    kv_lora_rank: int = 0
    q_lora_rank: int = 0
    qk_nope_head_dim: int = 0
    qk_rope_head_dim: int = 0
    v_head_dim: int = 0
    # Phase-2 imposed-FFN-factorisation (rung P_factff / Tier B3). When
    # mlp_type == "factff", `mlp_factorize_rank` is the bottleneck rank.
    mlp_factorize_rank: int = 0


@dataclass
class GPTConfig:
    vocab_size: int = 50304  # GPT-2 BPE rounded up to multiple of 64
    n_layers: int = 12
    block: BlockConfig = field(default_factory=BlockConfig)
    tie_embeddings: bool = True
    init_std: float = 0.02
    # Per experiment.md d.2 rung I: "MoE replacing every other dense MLP".
    # When True (default — matches the spec), only odd-indexed blocks become MoE
    # if `block.moe_enabled` is set. When False, every block becomes MoE (the
    # original behaviour, which over-counts per-block MoE params and contradicts
    # d.2 — kept as an opt-in escape hatch).
    moe_every_other: bool = True


class Block(nn.Module):
    def __init__(self, cfg: BlockConfig):
        super().__init__()
        self.cfg = cfg
        self.norm1 = make_norm(cfg.hidden, cfg.norm_type, eps=cfg.norm_eps)

        attn_cfg = AttnConfig(
            hidden_size=cfg.hidden,
            n_heads=cfg.n_heads,
            n_kv_heads=cfg.n_kv_heads,
            head_dim=cfg.head_dim,
            qk_norm=cfg.qk_norm,
            pos_emb_type=cfg.pos_emb_type,
            rope_base=cfg.rope_base,
            rope_partial_frac=cfg.rope_partial_frac,
            max_seq_len=cfg.max_seq_len,
            attn_type=cfg.attn_type,
            kv_lora_rank=cfg.kv_lora_rank,
            q_lora_rank=cfg.q_lora_rank,
            qk_nope_head_dim=cfg.qk_nope_head_dim,
            qk_rope_head_dim=cfg.qk_rope_head_dim,
            v_head_dim=cfg.v_head_dim,
        )
        if cfg.attn_type == "mla":
            self.attn = MultiLatentAttention(attn_cfg)
        elif cfg.attn_type == "mha":
            self.attn = CausalSelfAttention(attn_cfg)
        else:
            raise ValueError(f"Unknown attn_type={cfg.attn_type!r}")

        self.norm2 = make_norm(cfg.hidden, cfg.norm_type, eps=cfg.norm_eps)
        intermediate = cfg.intermediate or _default_intermediate(cfg.hidden, cfg.mlp_type)

        if cfg.moe_enabled:
            moe_kwargs = dict(cfg.moe_cfg)
            moe_kwargs.setdefault("expert_mlp_type", cfg.mlp_type)
            # `intermediate` in moe_cfg is consumed by _block_cfg_for_layer to
            # override the per-expert size; the resolved value is already in
            # `intermediate` (the local). Pop to avoid the kwargs collision.
            moe_kwargs.pop("intermediate", None)
            moe_cfg = MoEConfig(hidden=cfg.hidden, intermediate=intermediate, **moe_kwargs)
            self.mlp = MoEFFN(moe_cfg)
        elif cfg.mlp_type == "factff":
            self.mlp = make_mlp(
                cfg.hidden,
                intermediate,
                cfg.mlp_type,
                factorize_rank=int(cfg.mlp_factorize_rank),
            )
        else:
            self.mlp = make_mlp(cfg.hidden, intermediate, cfg.mlp_type)

    def forward(
        self,
        x: torch.Tensor,
        return_max_logit: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        attn_out, max_logit = self.attn(self.norm1(x), return_max_logit=return_max_logit)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, max_logit


def _default_intermediate(hidden: int, mlp_type: str) -> int:
    if mlp_type == "swiglu":
        # LLaMA convention: round 8/3 hidden to multiple of 128.
        ff = int(8 * hidden / 3)
        return ((ff + 127) // 128) * 128
    # gelu_2mat / relu2 default to 4 * hidden.
    return 4 * hidden


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.block.hidden)
        if cfg.block.pos_emb_type == "learned":
            self.pos_emb = nn.Embedding(cfg.block.max_seq_len, cfg.block.hidden)
        else:
            self.pos_emb = None
        self.layers = nn.ModuleList([Block(self._block_cfg_for_layer(i)) for i in range(cfg.n_layers)])
        self.norm_out = make_norm(cfg.block.hidden, cfg.block.norm_type, eps=cfg.block.norm_eps)
        self.lm_head = nn.Linear(cfg.block.hidden, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        self.apply(self._init_weights)

    def _block_cfg_for_layer(self, layer_idx: int) -> BlockConfig:
        """Per-layer BlockConfig. With `moe_every_other` (default), only the
        odd-indexed blocks get MoE — matching experiment.md d.2 rung I:
        'MoE replacing every other dense MLP'.

        For MoE blocks, `moe.intermediate` (when > 0) overrides the dense
        `mlp.intermediate` to control per-expert size independently of the
        dense MLP. This is what makes M (16 experts × 1024) and N (4 × 4096)
        controlled comparisons against I (8 × 2048) — same dense MLP across
        all three, only per-expert size varies."""
        bc = self.cfg.block
        if not bc.moe_enabled:
            return bc
        if self.cfg.moe_every_other:
            is_moe = (layer_idx % 2) == 1
            if not is_moe:
                # Dense layer: use mlp.intermediate, not moe.intermediate.
                return BlockConfig(
                    **{**bc.__dict__, "moe_enabled": False, "moe_cfg": {}}
                )
        # MoE layer: when moe.intermediate > 0, override BlockConfig.intermediate.
        moe_inter = int(bc.moe_cfg.get("intermediate", 0))
        if moe_inter > 0:
            return BlockConfig(**{**bc.__dict__, "intermediate": moe_inter})
        return bc

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=self.cfg.init_std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=self.cfg.init_std)

    def num_parameters(self, exclude_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if exclude_embedding:
            n -= self.embed.weight.numel()
            if not self.cfg.tie_embeddings:
                n -= self.lm_head.weight.numel()
        return n

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_max_logit: bool = False,
    ) -> dict[str, torch.Tensor]:
        B, T = idx.shape
        x = self.embed(idx)
        if self.pos_emb is not None:
            pos = torch.arange(T, device=idx.device)
            x = x + self.pos_emb(pos)[None, :, :]

        per_layer_max_logits: list[torch.Tensor] = []
        for blk in self.layers:
            x, max_logit = blk(x, return_max_logit=return_max_logit)
            if max_logit is not None:
                per_layer_max_logits.append(max_logit)

        x = self.norm_out(x)
        logits = self.lm_head(x)

        out: dict[str, torch.Tensor] = {"logits": logits}

        # Aggregate MoE losses if any.
        aux_total = torch.zeros((), device=idx.device)
        z_total = torch.zeros((), device=idx.device)
        for blk in self.layers:
            if isinstance(blk.mlp, MoEFFN):
                a, z = blk.mlp.collect_losses()
                aux_total = aux_total + a
                z_total = z_total + z
        out["aux_loss"] = aux_total
        out["z_loss"] = z_total

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            out["loss"] = loss
            out["total_loss"] = loss + aux_total + z_total

        if per_layer_max_logits:
            out["max_attn_logits"] = torch.stack(per_layer_max_logits)
        return out
