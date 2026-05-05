"""Forward + backward smoke across (norm × mlp × pos_emb × qk_norm × moe) cells."""
from __future__ import annotations

import itertools

import pytest
import torch

from coupled_muon_nanogpt.model.transformer import GPT, BlockConfig, GPTConfig

NORMS = ["rmsnorm", "layernorm"]
MLPS = ["swiglu", "gelu_2mat", "relu2"]
POSEMBS = ["rope", "learned"]
QKNS = [False, True]
MOES = [False, True]


def _build(norm_type, mlp_type, pos_emb_type, qk_norm, moe_enabled):
    block = BlockConfig(
        hidden=64,
        n_heads=4,
        intermediate=128,
        norm_type=norm_type,
        mlp_type=mlp_type,
        pos_emb_type=pos_emb_type,
        qk_norm=qk_norm,
        max_seq_len=64,
        moe_enabled=moe_enabled,
        moe_cfg=dict(num_experts=4, top_k=2, expert_mlp_type=mlp_type) if moe_enabled else {},
    )
    return GPT(GPTConfig(vocab_size=128, n_layers=2, block=block))


@pytest.mark.parametrize(
    "cell",
    list(itertools.product(NORMS, MLPS, POSEMBS, QKNS, MOES)),
)
def test_forward_backward(cell):
    norm, mlp, pos, qkn, moe = cell
    torch.manual_seed(0)
    model = _build(norm, mlp, pos, qkn, moe)
    model.train()
    B, T = 2, 32
    x = torch.randint(0, 128, (B, T))
    y = torch.randint(0, 128, (B, T))
    out = model(x, targets=y, return_max_logit=True)
    assert torch.isfinite(out["loss"]), f"non-finite loss in cell {cell}"
    out["total_loss"].backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads, "no gradients flowed"
    for g in grads:
        assert torch.isfinite(g).all(), f"non-finite grad in cell {cell}"
    assert "max_attn_logits" in out
    assert out["max_attn_logits"].numel() == 2  # n_layers
