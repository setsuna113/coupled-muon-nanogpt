"""Algorithmic convergence test for the partner-aware Newton-Schulz kernel.

Per `experiment.md` a.2, the inner step

    X ← G/‖GB‖_F; repeat: W = X B; T = b·W Wᵀ + c·(W Wᵀ)²; X ← a·X + T·X

drives the *product* `X B` toward the Newton-Schulz quintic image of `G B`. The
quintic does not produce exact `U V^T` (per the kernel docstring at
`coupled_muon.py:11-19`); it produces `U S' V^T` with `S' ∈ [≈0.5, ≈1.5]`. So:

  1. The orthogonal factor of `X B` (its polar factor) should match the
     orthogonal factor of `G B` — i.e., the same singular subspaces.
  2. The singular values of `X B` should be bounded in roughly [0.5, 1.5].
  3. Coupled-NS on (G_B, A) should produce `A·X = ns(A·G_B)` numerically.

Together these are the convergence guarantee that grounds every claim about the
optimizer. Inputs use full-rank shapes — rank-deficient inputs cause NS to
amplify floating-point noise on near-zero singular values, which would mask the
algorithm's actual contract.
"""
from __future__ import annotations

import torch

from coupled_muon_nanogpt.optim.coupled_muon import (
    coupled_newtonschulz5_A,
    coupled_newtonschulz5_B,
    zeropower_via_newtonschulz5,
)


def _polar_factor(M: torch.Tensor) -> torch.Tensor:
    """U V^T from the SVD M = U S V^T (the orthogonal factor of M's polar decomposition)."""
    U, _, Vt = torch.linalg.svd(M.float(), full_matrices=False)
    return U @ Vt


def test_kernel_B_singular_subspaces_match_partner_product():
    torch.manual_seed(0)
    # Square A (full rank) ⇒ A @ G is full row-rank. Otherwise NS amplifies
    # floating-point noise on the rank-deficient null space and the test fails
    # for the wrong reason.
    A = torch.randn(32, 32)
    G = torch.randn(32, 48)

    target_polar = _polar_factor(A @ G)

    out = coupled_newtonschulz5_B(G.clone(), A.clone(), steps=20, dtype=torch.float32).float()
    iterate_polar = _polar_factor(A.float() @ out)

    rel_err = (iterate_polar - target_polar).norm() / target_polar.norm()
    assert rel_err < 1e-4, (
        f"coupled_newtonschulz5_B singular subspaces did not match polar(A G): "
        f"rel_err={rel_err:.3e}"
    )


def test_kernel_A_singular_subspaces_match_partner_product():
    torch.manual_seed(1)
    # Square B (full rank) ⇒ G @ B is full column-rank.
    B = torch.randn(48, 48)
    G = torch.randn(32, 48)

    target_polar = _polar_factor(G @ B)

    out = coupled_newtonschulz5_A(G.clone(), B.clone(), steps=20, dtype=torch.float32).float()
    iterate_polar = _polar_factor(out @ B.float())

    rel_err = (iterate_polar - target_polar).norm() / target_polar.norm()
    assert rel_err < 1e-4, (
        f"coupled_newtonschulz5_A singular subspaces did not match polar(G B): "
        f"rel_err={rel_err:.3e}"
    )


def test_kernel_singular_values_in_quintic_band():
    """Quintic NS contract: singular values of `X B` should be in roughly [0.5, 1.5]
    (KellerJordan/Muon docstring, repeated at coupled_muon.py:14-18)."""
    torch.manual_seed(2)
    A = torch.randn(32, 32)
    G = torch.randn(32, 48)

    out = coupled_newtonschulz5_B(G.clone(), A.clone(), steps=20, dtype=torch.float32).float()
    sv = torch.linalg.svdvals(A.float() @ out)

    # Loose band — the quintic targets [0.5, 1.5] but real noise pushes a touch outside.
    assert sv.min() > 0.4, f"singular values dipped below 0.4: min={sv.min():.3f}"
    assert sv.max() < 1.6, f"singular values rose above 1.6: max={sv.max():.3f}"


def test_kernel_B_matches_standalone_NS_on_partner_product():
    """The whole point of the partner-aware kernel: A·X (output) ≈ ns(A·G_B). If
    these don't match, the partner-weight novelty isn't doing what experiment.md
    a.2 claims."""
    torch.manual_seed(3)
    A = torch.randn(32, 32)
    G = torch.randn(32, 48)

    ns_target = zeropower_via_newtonschulz5(
        (A @ G).clone(), steps=20, dtype=torch.float32
    ).float()
    out = coupled_newtonschulz5_B(G.clone(), A.clone(), steps=20, dtype=torch.float32).float()
    prod = A.float() @ out

    rel_err = (prod - ns_target).norm() / ns_target.norm()
    assert rel_err < 1e-3, (
        f"A·X (coupled-B output) does not match ns(A·G_B): rel_err={rel_err:.3e}"
    )


def test_kernel_zero_steps_is_normalized_input():
    """Sanity check: with steps=0 the kernel returns the pre-loop normalisation
    `G / ‖A G‖_F`, NOT plain Muon. (The optimizer's `step()` shortcuts around
    this case — see test_coupled_zero_equals_muon — but the bare kernel still
    has the documented behaviour, which downstream code must not rely on.)"""
    torch.manual_seed(5)
    A = torch.randn(32, 32)
    G = torch.randn(32, 48)
    out = coupled_newtonschulz5_B(G.clone(), A.clone(), steps=0, dtype=torch.float32).float()
    expected = G.float() / ((A.float() @ G.float()).norm() + 1e-7)
    rel_err = (out - expected).norm() / expected.norm()
    assert rel_err < 1e-5, f"steps=0 should return G/‖AG‖: rel_err={rel_err:.3e}"
