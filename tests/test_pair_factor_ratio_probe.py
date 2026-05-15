"""pair_factor_ratio probe end-to-end smoke (Phase-2 §2.7 LoRA-RITE probe)."""
from __future__ import annotations

import torch

from coupled_muon_nanogpt.model.transformer import GPT, BlockConfig, GPTConfig
from coupled_muon_nanogpt.probes.pair_factor_ratio import pair_factor_ratio_probe


def _swiglu_model() -> GPT:
    block = BlockConfig(
        hidden=64,
        n_heads=4,
        intermediate=128,
        norm_type="rmsnorm",
        mlp_type="swiglu",
        pos_emb_type="rope",
        max_seq_len=64,
    )
    return GPT(GPTConfig(vocab_size=64, n_layers=2, block=block))


def test_swiglu_model_emits_finite_ratios():
    model = _swiglu_model()
    out = pair_factor_ratio_probe(model, None, {"n_heads": 4})
    # SwiGLU on dense yields Q-K, V-O, up-down pairs per layer (3 × 2 = 6).
    assert len(out) >= 6
    for key, entry in out.items():
        assert "ratio_A_over_B" in entry
        assert torch.tensor(entry["ratio_A_over_B"]).isfinite().item(), (
            f"non-finite ratio for pair {key}: {entry}"
        )
        assert entry["A_frob"] > 0
        assert entry["B_frob"] > 0


def _mla_model() -> GPT:
    block = BlockConfig(
        hidden=64,
        n_heads=4,
        intermediate=128,
        norm_type="rmsnorm",
        mlp_type="swiglu",
        pos_emb_type="rope",
        max_seq_len=64,
        attn_type="mla",
        kv_lora_rank=16,
        q_lora_rank=16,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        v_head_dim=16,
    )
    return GPT(GPTConfig(vocab_size=64, n_layers=2, block=block))


def test_mla_model_emits_mla_pair_entries():
    model = _mla_model()
    out = pair_factor_ratio_probe(model, None, {"n_heads": 4})
    # Only (UK, DKV) is coupled per layer; UV routes to plain Muon.
    # 2 layers ⇒ 2 mla_kv entries + 2 mla_q entries.
    mla_kv_keys = [k for k in out if k.startswith("mla_kv.")]
    mla_q_keys = [k for k in out if k.startswith("mla_q.")]
    assert len(mla_kv_keys) == 2, mla_kv_keys
    assert len(mla_q_keys) == 2, mla_q_keys


def _factff_model() -> GPT:
    block = BlockConfig(
        hidden=64,
        n_heads=4,
        intermediate=0,
        norm_type="rmsnorm",
        mlp_type="factff",
        pos_emb_type="rope",
        max_seq_len=64,
        mlp_factorize_rank=16,
    )
    return GPT(GPTConfig(vocab_size=64, n_layers=2, block=block))


def test_factff_model_emits_factff_entries():
    model = _factff_model()
    out = pair_factor_ratio_probe(model, None, {"n_heads": 4})
    factff_keys = [k for k in out if k.startswith("factff.")]
    assert len(factff_keys) == 2, factff_keys
