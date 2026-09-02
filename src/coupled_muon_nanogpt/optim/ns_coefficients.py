"""Newton-Schulz coefficient policies for Phase-2 NS-policy sweep.

Three policies are supported. All numeric tables below are taken **verbatim**
from canonical published / reference-implementation sources (see the
PROVENANCE comments next to each table). When the user-requested K is at or
below a table's published length, we take the first K entries; for K greater
than the published length we extend by repeating the final (near-converged)
step. ``coeffs=None`` paths in the optimizer kernels fall through to the
hardcoded Bernstein triple, preserving Phase-1 numerics bitwise.

- ``bernstein``  — K copies of the canonical quintic ``(3.4445, -4.7750, 2.0315)``.
  Verified from Keller Jordan's Muon blog and from NVIDIA's
  ``emerging-optimizers`` library's ``"simple"`` entry. The coefficients are
  deliberately *non-convergent* — optimised for steep slope at the origin
  (inflating small singular values fast) at the cost of error plateauing
  at ≈ 0.3 (Modula docs; Polar Express §2).

- ``polar_express`` — Amsel, Persson, Musco & Gower, "The Polar Express:
  Optimal Matrix Sign Methods and Their Application to the Muon Algorithm"
  (arXiv 2505.16932v3, ICLR 2026). Per-step degree-5 polynomial whose
  coefficients are computed by Algorithm 2 of the paper (Remez-based
  minimax). The canonical 8-step table is shipped by NVIDIA's
  ``emerging-optimizers/muon_utils.py`` (``_COEFFICIENT_SETS["polar_express"]``).
  For K ≤ 8 we use the first K rows; for K > 8 we repeat the last (near-1)
  triple to extend.

- ``cesista`` — Cesista, YouJiacheng & Jordan, "Squeezing 1-2% efficiency
  gains in Muon via per-step Newton-Schulz coefficient gradient descent"
  (https://leloykun.github.io/ponder/muon-opt-coeffs/). The published artifact
  of this work is the per-step "quintic" 5-step table shipped by NVIDIA's
  ``emerging-optimizers/muon_utils.py`` (``_COEFFICIENT_SETS["quintic"]``).
  For K ≤ 5 we use the first K rows; for K > 5 we extend by repeating the
  last (near-converged) triple. This extension is approximate — the proper
  Cesista approach is to gradient-descent fresh coefficients per K — but
  repeating the converged step keeps σ near 1 and avoids divergence, which
  is the property the S1 sweep actually relies on.

References:
  - Amsel, Persson, Musco, Gower. "The Polar Express." arXiv 2505.16932v3
    (ICLR 2026). https://arxiv.org/abs/2505.16932
  - Cesista, Jordan. "Squeezing 1-2% Efficiency Gains Out of Muon by
    Optimizing the Newton-Schulz Coefficients."
    https://leloykun.github.io/ponder/muon-opt-coeffs/
  - NVIDIA NeMo Emerging-Optimizers, ``muon_utils._COEFFICIENT_SETS``.
    https://github.com/NVIDIA-NeMo/Emerging-Optimizers (canonical numeric source)
  - Keller Jordan. "Muon: An optimizer for hidden layers in neural networks."
    https://kellerjordan.github.io/posts/muon/ (Bernstein triple verification)
  - Bernstein & Newhouse. "Old Optimizer, New Norm" arXiv 2409.20325 (2024).
"""
from __future__ import annotations

import torch

_BERNSTEIN_TRIPLE: tuple[float, float, float] = (3.4445, -4.7750, 2.0315)


# Canonical Polar Express 8-step table — verbatim from
# NVIDIA-NeMo/Emerging-Optimizers, muon_utils.py, _COEFFICIENT_SETS["polar_express"].
# Step 1 has the largest slope at the origin (drives σ_min → 1); subsequent
# steps narrow until the final step settles at ≈ (15/8, -10/8, 3/8) — the
# polynomial that maps σ ≈ 1 → 1.
_POLAR_EXPRESS_CANONICAL: list[tuple[float, float, float]] = [
    (8.2051, -22.9019, 16.4607),
    (4.0664,  -2.8612,  0.5184),
    (3.9096,  -2.8234,  0.5250),
    (3.2856,  -2.4153,  0.4853),
    (2.2779,  -1.6198,  0.3985),
    (1.8726,  -1.2307,  0.3585),
    (1.8564,  -1.2132,  0.3568),
    (1.8750,  -1.2500,  0.3750),
]


# Canonical Cesista 5-step table — verbatim from NVIDIA-NeMo/Emerging-Optimizers
# muon_utils.py _COEFFICIENT_SETS["quintic"]. Derived by gradient-descent on
# the per-step coefficient tensor (YouJiacheng / Cesista / Jordan, 2024-25)
# to maximise slope at zero subject to a tolerance margin around σ = 1.
_CESISTA_CANONICAL: list[tuple[float, float, float]] = [
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),
]


def _truncate_or_extend(
    table: list[tuple[float, float, float]],
    steps: int,
) -> list[tuple[float, float, float]]:
    """Return `steps` coefficients from a canonical table.

    K ≤ len(table): first K rows.
    K > len(table):  full table + repeat the last (near-converged) row.
    """
    if steps <= len(table):
        return list(table[:steps])
    pad = [table[-1]] * (steps - len(table))
    return list(table) + pad


# Pre-materialise the K-indexed view used by the existing get_coefficients() API.
# Keys cover the S1 sweep's K ∈ {3, 5, 8} plus the K-curve sweep's
# {1, 2, 3, 5, 8, 12} (suggestion.md S3 / configs/sweeps/k_curve_at_winner.yaml).
_K_VALUES_OF_INTEREST: tuple[int, ...] = (1, 2, 3, 5, 8, 12)

_POLAR_EXPRESS: dict[int, list[tuple[float, float, float]]] = {
    K: _truncate_or_extend(_POLAR_EXPRESS_CANONICAL, K)
    for K in _K_VALUES_OF_INTEREST
}

_CESISTA: dict[int, list[tuple[float, float, float]]] = {
    K: _truncate_or_extend(_CESISTA_CANONICAL, K) for K in _K_VALUES_OF_INTEREST
}


_SUPPORTED_POLICIES = ("bernstein", "polar_express", "cesista")


def get_coefficients(policy: str, steps: int) -> list[tuple[float, float, float]]:
    """Return the per-step (a, b, c) coefficient triples for the given policy.

    Args:
        policy: One of ``bernstein``, ``polar_express``, ``cesista``.
        steps: Number of NS iterations (K). Must be ≥ 1.

    Returns:
        A list of length ``steps``; element ``k`` is the (a, b, c) for the
        ``k``-th iteration.

    Raises:
        ValueError: when policy is unknown or (policy, steps) is not in the
        published coefficient table. ``bernstein`` accepts any positive steps.
    """
    if steps < 1:
        raise ValueError(f"steps must be ≥ 1, got {steps}")
    if policy not in _SUPPORTED_POLICIES:
        raise ValueError(
            f"Unknown ns_coefficients={policy!r}. Supported: {_SUPPORTED_POLICIES}."
        )

    if policy == "bernstein":
        return [_BERNSTEIN_TRIPLE] * steps

    table = _POLAR_EXPRESS if policy == "polar_express" else _CESISTA
    if steps not in table:
        raise ValueError(
            f"No published per-step table for policy={policy!r} at K={steps}. "
            f"Available K values: {sorted(table.keys())}."
        )
    return list(table[steps])


def coefficients_to_tensor(
    coeffs: list[tuple[float, float, float]],
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Pack a coefficient list into a ``(K, 3)`` tensor for the NS kernels."""
    return torch.tensor(coeffs, dtype=dtype, device=device)
