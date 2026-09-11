"""TangentMuon port: pair-update algebra, factory routing, and a tiny training step.

  1. The low-rank (2k x 2k core) and dense (out x in) routes give the same
     update for both variants — the derivation's "low-rank acceleration is
     exact" claim.
  2. The damped Sylvester solver satisfies its equation.
  3. The joint normalization puts one Muon-sized step on each factor.
  4. The factory builds `tangent_muon` from the same pair classification as
     `coupled_muon_v2` (same pairs, kinds vo/flat/qk in the emitted order).
  5. A few steps on a tiny GPT lower the loss, move every coupled matrix and
     stay finite — MHA and GQA, v3 and v7, per-head and flat.
"""
from __future__ import annotations

import copy

import pytest
import torch
from omegaconf import OmegaConf

from coupled_muon_nanogpt.model.transformer import GPT, BlockConfig, GPTConfig
from coupled_muon_nanogpt.optim.factory import build_optimizer
from coupled_muon_nanogpt.optim.tangent_muon import (
    TangentMuon,
    damping_lambda,
    solve_damped_sylvester,
    tangent_pair_update,
)


def _rand_pair(out: int, k: int, inn: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    P = torch.randn(out, k, generator=g) / k**0.5
    Q = torch.randn(k, inn, generator=g) / inn**0.5
    M_P = torch.randn(out, k, generator=g) * 0.1
    M_Q = torch.randn(k, inn, generator=g) * 0.1
    return P, Q, M_P, M_Q


@pytest.mark.parametrize("variant", ["v3", "v7"])
@pytest.mark.parametrize("shape", [(64, 8, 64), (96, 16, 48), (40, 4, 120)])
def test_lowrank_and_dense_routes_agree(variant, shape):
    out, k, inn = shape
    P, Q, M_P, M_Q = _rand_pair(out, k, inn, seed=hash((variant, shape)) % 1000)
    U_P_lr, U_Q_lr = tangent_pair_update(P, Q, M_P, M_Q, variant=variant, route="lowrank")
    U_P_d, U_Q_d = tangent_pair_update(P, Q, M_P, M_Q, variant=variant, route="dense")
    for a, b in ((U_P_lr, U_P_d), (U_Q_lr, U_Q_d)):
        rel = (a - b).norm() / (b.norm() + 1e-12)
        assert rel < 1e-4, f"{variant} {shape}: routes disagree, rel diff {rel:.2e}"


def test_sylvester_solves_its_equation():
    g = torch.Generator().manual_seed(3)
    A = torch.randn(12, 5, generator=g)
    B = torch.randn(12, 7, generator=g)
    G_L = A @ A.T  # rank 5 PSD
    G_R = B @ B.T  # rank 7 PSD
    rhs = torch.randn(12, 12, generator=g)
    lam = torch.tensor(0.3)
    X = solve_damped_sylvester(G_L, G_R, rhs, lam)
    resid = (G_L @ X + X @ G_R + lam * X - rhs).norm() / rhs.norm()
    assert resid < 1e-5, f"Sylvester residual {resid:.2e}"


def test_joint_normalization_is_muon_scale():
    P, Q, M_P, M_Q = _rand_pair(64, 8, 64, seed=5)
    U_P, U_Q = tangent_pair_update(P, Q, M_P, M_Q)
    total = (U_P * U_P).sum() + (U_Q * U_Q).sum()
    target = float(min(U_P.shape) + min(U_Q.shape))
    assert abs(total.item() - target) / target < 1e-4
    lam = damping_lambda(P, Q, 8, 0.1)
    assert lam.item() > 0


def test_bad_inputs_raise():
    P, Q, M_P, M_Q = _rand_pair(16, 4, 16)
    with pytest.raises(ValueError):
        tangent_pair_update(P, Q, M_P, M_Q, variant="v9")
    with pytest.raises(ValueError):
        tangent_pair_update(P, Q[:3], M_P, M_Q[:3])
    with pytest.raises(ValueError):
        TangentMuon(
            coupled_pairs=[],
            muon_params=[torch.nn.Parameter(torch.zeros(4, 4))],
            tangent_variant="v1",
        )


# ---------------------------------------------------------------- factory --
def _make_model(n_kv_heads: int | None = None, mlp_type: str = "swiglu") -> GPT:
    block = BlockConfig(
        hidden=64,
        n_heads=4,
        n_kv_heads=n_kv_heads,
        intermediate=128,
        norm_type="rmsnorm",
        mlp_type=mlp_type,
        pos_emb_type="rope",
        qk_norm=False,
        max_seq_len=64,
    )
    return GPT(GPTConfig(vocab_size=64, n_layers=2, block=block))


def _cfg(opt_type: str, *, n_kv_heads=None, use_multi_head=True, variant="v3") -> OmegaConf:
    return OmegaConf.create(
        {
            "model": {
                "hidden": 64,
                "attn": {"n_heads": 4, "n_kv_heads": n_kv_heads, "rope_partial_frac": 1.0},
                "pos_emb": {"type": "rope"},
            },
            "optimizer": {
                "type": opt_type,
                "lr": 1e-2,
                "wd": 0.0,
                "momentum": 0.95,
                "betas": [0.95, 0.95],
                "eps": 1e-8,
                "ns_steps": 5,
                "coupled_steps": 4,
                "couple_qk": True,
                "couple_vo": True,
                "couple_updown": True,
                "couple_mla": True,
                "couple_factff": True,
                "use_multi_head": use_multi_head,
                "ns_dtype": "fp32",
                "tangent_variant": variant,
                "damping": 0.1,
            },
        }
    )


def test_factory_builds_same_pairs_as_coupled_muon():
    torch.manual_seed(0)
    model = _make_model()
    opt_t = build_optimizer(model, _cfg("tangent_muon"))
    opt_c = build_optimizer(copy.deepcopy(model), _cfg("coupled_muon_v2"))
    assert isinstance(opt_t, TangentMuon)
    assert len(opt_t.coupled_pairs) == len(opt_c.coupled_pairs) == 6  # 2 layers x (vo, updown, qk)
    assert opt_t._pair_kinds == ["vo", "vo", "flat", "flat", "qk", "qk"]
    n_muon = sum(1 for p in opt_t.param_groups[0]["params"] if opt_t.state[p].get("use_muon"))
    assert n_muon == 2  # the SwiGLU gate of each layer


def _loss(model: GPT, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = model(x, targets=y)
    if torch.is_tensor(out):
        return out
    if isinstance(out, dict):
        return out["loss"]
    if hasattr(out, "loss"):
        return out.loss
    losses = [t for t in out if torch.is_tensor(t) and t.ndim == 0]
    assert losses, "model(x, targets=y) returned no scalar loss"
    return losses[0]


def _run_steps(opt_type: str, n_steps: int = 6, **cfg_kw):
    torch.manual_seed(0)
    model = _make_model(n_kv_heads=cfg_kw.get("n_kv_heads"))
    init = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt = build_optimizer(model, _cfg(opt_type, **cfg_kw))
    g = torch.Generator().manual_seed(1)
    x = torch.randint(0, 64, (4, 32), generator=g)
    y = torch.roll(x, -1, dims=1)
    losses = []
    for _ in range(n_steps):
        opt.zero_grad(set_to_none=True)
        loss = _loss(model, x, y)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    return losses, model, opt, init


def _assert_trained(losses, model, opt, init):
    assert all(torch.isfinite(torch.tensor(losses))), losses
    assert losses[-1] < losses[0], losses
    names = {id(p): n for n, p in model.named_parameters()}
    for p in model.parameters():
        assert torch.isfinite(p).all(), names[id(p)]
    for a, b, _h, _is_qk in opt.coupled_pairs:
        for p in (a, b):
            assert not torch.equal(p.detach(), init[names[id(p)]]), f"{names[id(p)]} did not move"


@pytest.mark.parametrize("variant", ["v3", "v7"])
@pytest.mark.parametrize("use_multi_head", [True, False])
def test_tiny_gpt_steps_decrease_loss(variant, use_multi_head):
    losses, model, opt, init = _run_steps("tangent_muon", variant=variant, use_multi_head=use_multi_head)
    _assert_trained(losses, model, opt, init)


def test_gqa_per_head_path_runs():
    losses, model, opt, init = _run_steps("tangent_muon", n_kv_heads=2, use_multi_head=True)
    _assert_trained(losses, model, opt, init)
    # k_proj / v_proj have 2 kv heads, q_proj / o_proj have 4: the GQA branch was exercised.
    qk = [pair for pair, kd in zip(opt.coupled_pairs, opt._pair_kinds, strict=True) if kd == "qk"]
    assert qk and qk[0][0].shape[0] == 2 * qk[0][1].shape[0]


def test_coupled_muon_path_unchanged_by_port():
    # Sanity that the factory refactor did not alter CoupledMuon_v2's pair list.
    torch.manual_seed(0)
    model = _make_model()
    opt = build_optimizer(model, _cfg("coupled_muon_v2"))
    kinds = [is_qk for _a, _b, _h, is_qk in opt.coupled_pairs]
    assert kinds == [False, False, False, False, True, True]
