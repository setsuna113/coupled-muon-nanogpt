"""FactorizedMLP forward + pair-registration test (Phase-2 B3)."""
from __future__ import annotations

import torch

from coupled_muon_nanogpt.model.mlp import FactorizedMLP, make_mlp
from coupled_muon_nanogpt.model.transformer import GPT, BlockConfig, GPTConfig
from coupled_muon_nanogpt.optim.factory import classify_parameters


def test_factorized_mlp_forward_smoke():
    mlp = FactorizedMLP(hidden=64, rank=16)
    x = torch.randn(2, 8, 64)
    y = mlp(x)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_make_mlp_factff_dispatch():
    m = make_mlp(hidden=64, intermediate=0, kind="factff", factorize_rank=16)
    assert isinstance(m, FactorizedMLP)


def test_make_mlp_factff_requires_positive_rank():
    import pytest
    with pytest.raises(ValueError, match="factorize_rank"):
        make_mlp(hidden=64, intermediate=0, kind="factff", factorize_rank=0)


def _make_factff_model(n_layers: int = 2) -> GPT:
    block = BlockConfig(
        hidden=64,
        n_heads=4,
        intermediate=0,
        norm_type="rmsnorm",
        mlp_type="factff",
        pos_emb_type="rope",
        qk_norm=False,
        max_seq_len=64,
        mlp_factorize_rank=16,
    )
    return GPT(GPTConfig(vocab_size=64, n_layers=n_layers, block=block))


def test_factff_pairs_registered():
    model = _make_factff_model(n_layers=2)
    groups = classify_parameters(model, n_heads=4)
    assert len(groups.coupled_factff) == 2, (
        f"Expected 2 factff pairs (2 layers); got {len(groups.coupled_factff)}"
    )
    # No SwiGLU updown pair when factff replaces FFN.
    assert len(groups.coupled_updown) == 0


def test_factff_forward_through_full_model():
    model = _make_factff_model(n_layers=2)
    x = torch.randint(0, 64, (2, 16))
    out = model(x)
    assert out["logits"].shape == (2, 16, 64)
    assert torch.isfinite(out["logits"]).all()
