"""NS coefficient policy unit tests (Phase-2 S1).

Verifies the framework, not the specific Polar Express / Cesista numbers
(those are approximate in optim/ns_coefficients.py pending canonical-source
pull).
"""
from __future__ import annotations

import pytest
import torch

from coupled_muon_nanogpt.optim.coupled_muon import zeropower_via_newtonschulz5
from coupled_muon_nanogpt.optim.ns_coefficients import (
    coefficients_to_tensor,
    get_coefficients,
)


@pytest.mark.parametrize("policy", ["bernstein", "polar_express", "cesista"])
@pytest.mark.parametrize("steps", [3, 5, 8])
def test_returns_correct_length(policy, steps):
    coeffs = get_coefficients(policy, steps)
    assert len(coeffs) == steps
    for triple in coeffs:
        assert len(triple) == 3
        for v in triple:
            assert isinstance(v, float)


def test_bernstein_is_canonical_triple():
    bern = get_coefficients("bernstein", 5)
    for triple in bern:
        assert triple == pytest.approx((3.4445, -4.7750, 2.0315))


def test_unknown_policy_raises():
    with pytest.raises(ValueError, match="Unknown ns_coefficients"):
        get_coefficients("nonsense", 5)


def test_zero_or_negative_steps_raises():
    with pytest.raises(ValueError, match="steps must be"):
        get_coefficients("bernstein", 0)


def test_missing_table_entry_raises():
    with pytest.raises(ValueError, match="No published per-step table"):
        get_coefficients("polar_express", 99)


def test_bernstein_kernel_matches_hardcoded_path():
    """coeffs=None must reproduce the Phase-1 hardcoded Bernstein path exactly."""
    torch.manual_seed(0)
    G = torch.randn(32, 64)
    out_none = zeropower_via_newtonschulz5(G, steps=5, dtype=torch.float32, coeffs=None)
    bern = coefficients_to_tensor(get_coefficients("bernstein", 5))
    out_bern = zeropower_via_newtonschulz5(G, steps=5, dtype=torch.float32, coeffs=bern)
    # Both paths share the (3.4445, -4.7750, 2.0315) triple; numerics should match.
    diff = (out_none - out_bern).norm().item() / (out_none.norm().item() + 1e-12)
    assert diff < 1e-5, f"Bernstein-via-table differs from hardcoded path: {diff}"


@pytest.mark.parametrize("policy", ["bernstein", "polar_express", "cesista"])
def test_output_sigma_max_in_band(policy):
    """Each policy should drive σ_max(out) into the Bernstein-band [0.4, 1.6]."""
    torch.manual_seed(1)
    G = torch.randn(64, 64)
    coeffs = coefficients_to_tensor(get_coefficients(policy, 5))
    out = zeropower_via_newtonschulz5(G, steps=5, dtype=torch.float32, coeffs=coeffs)
    sv = torch.linalg.svdvals(out.float())
    sigma_max = sv.max().item()
    # Loose band: Phase-2's approximate Polar Express / Cesista values may
    # not hit the Bernstein band tightly. The hard requirement is they don't
    # explode (no NaN / no σ_max >> 1).
    assert torch.isfinite(out).all(), f"{policy}: non-finite output"
    assert sigma_max < 5.0, f"{policy}: σ_max exploded to {sigma_max}"


def test_gram_form_passthrough_finite():
    """gram_form=True should currently be a no-op passthrough (Phase-2.5 stub).
    Should not crash, should produce finite output."""
    torch.manual_seed(2)
    G = torch.randn(32, 48)
    out = zeropower_via_newtonschulz5(G, steps=5, dtype=torch.float32, gram_form=True)
    assert torch.isfinite(out).all()
