"""Optimizer-stamp checkpoint guard: ckpt refuses to load under different optimizer."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from coupled_muon_nanogpt.model.transformer import GPT, BlockConfig, GPTConfig
from coupled_muon_nanogpt.optim.factory import build_optimizer
from coupled_muon_nanogpt.utils import load_ckpt, save_ckpt


def _cfg(opt_type: str, coupled_steps: int = 4) -> any:
    return OmegaConf.create(
        {
            "name": "test",
            "model": {
                "attn": {"n_heads": 4, "n_kv_heads": None},
            },
            "optimizer": {
                "type": opt_type,
                "lr": 1e-3,
                "wd": 0.1,
                "momentum": 0.95,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "ns_steps": 5,
                "coupled_steps": coupled_steps,
                "couple_qk": True,
                "couple_vo": True,
                "couple_updown": True,
                "use_multi_head": False,
            },
        }
    )


def _build_model() -> GPT:
    block = BlockConfig(
        hidden=64,
        n_heads=4,
        intermediate=128,
        mlp_type="swiglu",
        pos_emb_type="rope",
        max_seq_len=64,
    )
    return GPT(GPTConfig(vocab_size=128, n_layers=2, block=block))


def test_ckpt_rejects_optimizer_change():
    model = _build_model()
    cfg_a = _cfg("coupled_muon_v2", coupled_steps=4)
    opt_a = build_optimizer(model, cfg_a)
    with tempfile.TemporaryDirectory() as tmp:
        path = save_ckpt(Path(tmp), step=10, model=model, optimizer=opt_a, cfg=cfg_a, seed=0)

        # Same config: load should succeed.
        model2 = _build_model()
        opt_a2 = build_optimizer(model2, cfg_a)
        load_ckpt(path, model2, opt_a2, cfg_a)

        # Different optimizer type: load must reject.
        model3 = _build_model()
        cfg_b = _cfg("muon")
        opt_b = build_optimizer(model3, cfg_b)
        with pytest.raises(RuntimeError, match="Optimizer-stamp mismatch"):
            load_ckpt(path, model3, opt_b, cfg_b)

        # Same type, different coupled_steps: also rejected.
        model4 = _build_model()
        cfg_c = _cfg("coupled_muon_v2", coupled_steps=2)
        opt_c = build_optimizer(model4, cfg_c)
        with pytest.raises(RuntimeError, match="Optimizer-stamp mismatch"):
            load_ckpt(path, model4, opt_c, cfg_c)
