"""Pin down the `final_polish` semantics in CoupledMuon_v2.

Per experiment.md a.2 / d.4, the optimizer's update is a two-stage composition:

  Stage 1 (coupled NS, `C_A`):  X_1 = C_A(g; B)    so X_1·B → polar(g·B)
  Stage 2 (plain NS):           U   = NS(X_1)      so ‖U‖_spec ≈ 1

Stage 2 is gated by `final_polish` (default True; False is the d.4 ablation).
Two invariants this test pins down:

  1. With `final_polish=False`, the applied update direction equals (the
     sign-flipped) coupled iterate `u_A_c` *up to dtype/scale conversion* — no
     trailing zeropower NS pass is run.
  2. With `final_polish=True`, the applied update direction equals
     `zeropower_via_newtonschulz5(u_A_c)` — the historical two-stage path.

The test runs one optimizer step on a tiny coupled pair and inspects the
parameter delta against a hand-computed reference.
"""
from __future__ import annotations

import math

import torch

from coupled_muon_nanogpt.optim.coupled_muon import (
    CoupledMuon_v2,
    coupled_newtonschulz5_A,
    coupled_newtonschulz5_B,
    zeropower_via_newtonschulz5,
)


def _make_pair(seed: int = 0) -> tuple[torch.nn.Parameter, torch.nn.Parameter]:
    g = torch.Generator().manual_seed(seed)
    A = torch.nn.Parameter(torch.randn(32, 16, generator=g) * 0.1)
    B = torch.nn.Parameter(torch.randn(16, 24, generator=g) * 0.1)
    return A, B


def _populate_grads(A: torch.nn.Parameter, B: torch.nn.Parameter, seed: int = 1) -> None:
    g = torch.Generator().manual_seed(seed)
    A.grad = torch.randn(A.shape, generator=g) * 0.1
    B.grad = torch.randn(B.shape, generator=g) * 0.1


def _build_opt(A: torch.nn.Parameter, B: torch.nn.Parameter, *, final_polish: bool) -> CoupledMuon_v2:
    return CoupledMuon_v2(
        lr=1e-2,
        wd=0.0,
        coupled_pairs=[(A, B, 1, False)],   # is_qk=False, single head
        muon_params=[],
        adamw_params=[],
        momentum=0.0,
        nesterov=False,
        ns_steps=5,
        coupled_steps=4,
        enable_coupled=True,
        final_polish=final_polish,
        use_multi_head=False,
        n_heads=1,
        ns_dtype=torch.float32,            # avoid bf16 noise in the comparison
    )


def test_final_polish_false_skips_stage2():
    A, B = _make_pair(seed=0)
    A0 = A.detach().clone()
    B0 = B.detach().clone()
    _populate_grads(A, B, seed=42)
    g_A_in = A.grad.detach().clone()
    g_B_in = B.grad.detach().clone()

    opt = _build_opt(A, B, final_polish=False)
    opt.step()

    # Hand-compute stage 1 only.
    u_A_c = coupled_newtonschulz5_A(g_A_in.clone(), B0.clone(), steps=4, dtype=torch.float32)
    u_B_c = coupled_newtonschulz5_B(g_B_in.clone(), A0.clone(), steps=4, dtype=torch.float32)

    adj_lr_A = 1e-2 * 0.2 * math.sqrt(max(A.shape))
    adj_lr_B = 1e-2 * 0.2 * math.sqrt(max(B.shape))
    expected_A = A0 - adj_lr_A * u_A_c
    expected_B = B0 - adj_lr_B * u_B_c

    assert torch.allclose(A.detach(), expected_A, atol=1e-5, rtol=1e-4), (
        f"final_polish=False should apply stage-1 directly. "
        f"Max diff {(A.detach() - expected_A).abs().max().item():.2e}"
    )
    assert torch.allclose(B.detach(), expected_B, atol=1e-5, rtol=1e-4), (
        f"final_polish=False should apply stage-1 directly. "
        f"Max diff {(B.detach() - expected_B).abs().max().item():.2e}"
    )


def test_final_polish_true_applies_stage2():
    A, B = _make_pair(seed=0)
    A0 = A.detach().clone()
    B0 = B.detach().clone()
    _populate_grads(A, B, seed=42)
    g_A_in = A.grad.detach().clone()
    g_B_in = B.grad.detach().clone()

    opt = _build_opt(A, B, final_polish=True)
    opt.step()

    # Hand-compute stage 1 then stage 2.
    u_A_c = coupled_newtonschulz5_A(g_A_in.clone(), B0.clone(), steps=4, dtype=torch.float32)
    u_B_c = coupled_newtonschulz5_B(g_B_in.clone(), A0.clone(), steps=4, dtype=torch.float32)
    u_A = zeropower_via_newtonschulz5(u_A_c.clone(), steps=5, dtype=torch.float32)
    u_B = zeropower_via_newtonschulz5(u_B_c.clone(), steps=5, dtype=torch.float32)

    adj_lr_A = 1e-2 * 0.2 * math.sqrt(max(A.shape))
    adj_lr_B = 1e-2 * 0.2 * math.sqrt(max(B.shape))
    expected_A = A0 - adj_lr_A * u_A
    expected_B = B0 - adj_lr_B * u_B

    assert torch.allclose(A.detach(), expected_A, atol=1e-5, rtol=1e-4), (
        f"final_polish=True should apply two-stage update. "
        f"Max diff {(A.detach() - expected_A).abs().max().item():.2e}"
    )
    assert torch.allclose(B.detach(), expected_B, atol=1e-5, rtol=1e-4), (
        f"final_polish=True should apply two-stage update. "
        f"Max diff {(B.detach() - expected_B).abs().max().item():.2e}"
    )


def test_final_polish_changes_update():
    """Sanity: the two variants should produce *different* updates on the same
    starting state. If they're identical, stage 2 is a numerical no-op and the
    `final_polish` toggle is meaningless on this pair (informative for d.4)."""
    A1, B1 = _make_pair(seed=0)
    A2 = torch.nn.Parameter(A1.detach().clone())
    B2 = torch.nn.Parameter(B1.detach().clone())
    _populate_grads(A1, B1, seed=42)
    _populate_grads(A2, B2, seed=42)

    opt1 = _build_opt(A1, B1, final_polish=True)
    opt2 = _build_opt(A2, B2, final_polish=False)
    opt1.step()
    opt2.step()

    diff_A = (A1.detach() - A2.detach()).abs().max().item()
    diff_B = (B1.detach() - B2.detach()).abs().max().item()
    # Expect non-trivial difference on a random pair (both updates are unit-spectral
    # ish but stage 2 changes the geometry); 1e-3 is a generous floor.
    assert max(diff_A, diff_B) > 1e-3, (
        f"final_polish toggle had ~no effect: diff_A={diff_A:.2e}, diff_B={diff_B:.2e}. "
        "Either stage 1 already produces near-orthogonal output here (informative "
        "for d.4) or the toggle is broken."
    )


def test_final_polish_records_stage1_norm():
    """The stage-1 Frobenius norm is captured in optimizer state for the probe."""
    A, B = _make_pair(seed=0)
    _populate_grads(A, B, seed=42)
    opt = _build_opt(A, B, final_polish=True)
    opt.step()
    assert "stage1_frob_norm" in opt.state[A]
    assert "stage1_frob_norm" in opt.state[B]
    # Norm must be finite and positive — coupled NS does not zero out signal.
    assert opt.state[A]["stage1_frob_norm"] > 0.0
    assert opt.state[B]["stage1_frob_norm"] > 0.0
