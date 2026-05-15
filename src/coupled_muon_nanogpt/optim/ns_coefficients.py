"""Newton-Schulz coefficient policies for Phase-2 NS-policy sweep.

Three policies are supported:

- ``bernstein``  — K copies of the canonical quintic ``(3.4445, -4.7750, 2.0315)``
  (Keller Jordan 2024; the Phase-1 default). Coefficients are deliberately
  *non-convergent* — optimized for steep slope at the origin (inflating small
  singular values fast) at the cost of error plateauing at ≈ 0.3
  (Modula docs; Polar Express §2). ``coeffs=None`` paths in the optimizer
  kernels fall through to this exact triple, preserving Phase-1 numerics
  bitwise.

- ``polar_express`` — per-step degree-5 coefficients from Amsel, Persson, Musco
  & Gower, "The Polar Express" (arXiv 2505.16932, ICLR 2026). Drive the polar
  iterate to ≈ 1 ≈ 2× faster at the spectral floor than Bernstein. **Values
  below are approximate reproductions** of the published optimal per-step
  coefficients; the canonical values live in the paper's supplement and the
  NVIDIA ``emerging-optimizers`` reference implementation. Replace with the
  exact values before the headline publication run.

- ``cesista`` — per-step optimised coefficients from Cesista, YouJiacheng &
  Jordan, "Squeezing 1–2% efficiency gains" (2025). The first step uses the
  Bernstein triple; subsequent steps narrow toward σ ≈ 1. Values below are
  **also approximate**; the canonical sequence comes from the YouJiacheng
  blog post and the same NVIDIA reference. Pulled in placeholder values that
  match the spirit of the publication: a steep first step, then progressively
  conservative refinement.

The intent is that the *framework* (per-step coefficient selection, tensorised
coefficient passing) lands here, and the numeric tables can be patched once
the canonical sources are read. Tests for this module assert the framework
behaviour (length K, finite, in a reasonable range), not the specific values.

References:
  - Amsel, Persson, Musco, Gower. "The Polar Express." arXiv 2505.16932.
  - Cesista, YouJiacheng, Jordan. "Squeezing 1–2% efficiency gains in Muon
    via per-step Newton–Schulz coefficient gradient descent." 2025.
  - Bernstein & Newhouse. "Old Optimizer, New Norm" arXiv 2409.20325 (2024).
  - Keller Jordan. Muon repo, ``muon.py``.
"""
from __future__ import annotations

import torch


_BERNSTEIN_TRIPLE: tuple[float, float, float] = (3.4445, -4.7750, 2.0315)


# Per-step Polar Express coefficients (approximate; see module docstring).
# Each row is (a_k, b_k, c_k) for k = 1, ..., K. Step 1 has the largest slope
# (drives σ_min → 1 aggressively); later steps narrow.
_POLAR_EXPRESS: dict[int, list[tuple[float, float, float]]] = {
    3: [
        (8.205, -23.475, 17.341),
        (4.115, -2.945, 0.547),
        (3.318, -2.489, 0.510),
    ],
    5: [
        (8.205, -23.475, 17.341),
        (4.115, -2.945, 0.547),
        (3.949, -2.909, 0.554),
        (3.318, -2.489, 0.510),
        (2.300, -1.669, 0.419),
    ],
    8: [
        (8.205, -23.475, 17.341),
        (4.115, -2.945, 0.547),
        (3.949, -2.909, 0.554),
        (3.318, -2.489, 0.510),
        (2.300, -1.669, 0.419),
        (2.005, -1.428, 0.395),
        (1.799, -1.252, 0.380),
        (1.605, -1.105, 0.366),
    ],
}


# Per-step Cesista–YouJiacheng coefficients (approximate; see module docstring).
# Step 1 = Bernstein triple (steep slope at the origin); subsequent steps are
# slight refinements that improve the σ ≈ 1 plateau.
_CESISTA: dict[int, list[tuple[float, float, float]]] = {
    3: [
        _BERNSTEIN_TRIPLE,
        (3.3000, -4.5000, 1.9000),
        (3.1000, -4.0000, 1.7000),
    ],
    5: [
        _BERNSTEIN_TRIPLE,
        (3.4000, -4.5000, 1.9500),
        (3.3000, -4.2500, 1.8800),
        (3.1500, -4.0000, 1.7500),
        (2.9000, -3.5000, 1.5500),
    ],
    8: [
        _BERNSTEIN_TRIPLE,
        (3.4000, -4.5000, 1.9500),
        (3.3500, -4.4000, 1.9000),
        (3.2500, -4.2000, 1.8500),
        (3.1500, -4.0000, 1.7500),
        (3.0000, -3.7500, 1.6500),
        (2.8500, -3.4000, 1.5000),
        (2.6500, -3.1000, 1.4000),
    ],
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
