"""LR-prefactor policy unit tests (Phase-2 A2).

Verifies that `CoupledMuon_v2.adjust_lr_for_muon` and `Muon.adjust_lr_for_muon`
implement each of the three published prefactor formulas correctly.
"""
from __future__ import annotations

import math

import pytest
import torch

from coupled_muon_nanogpt.optim.coupled_muon import CoupledMuon_v2
from coupled_muon_nanogpt.optim.muon import Muon


def _bare_coupled(prefactor: str, coupled_steps: int = 4):
    p = torch.nn.Parameter(torch.zeros(8, 16))
    return CoupledMuon_v2(
        lr=1.0,
        coupled_pairs=[],
        muon_params=[p],
        adamw_params=[],
        coupled_steps=coupled_steps,
        ns_steps=5,
        lr_prefactor=prefactor,
    )


def _bare_plain(prefactor: str, ns_steps: int = 5):
    p = torch.nn.Parameter(torch.zeros(8, 16))
    return Muon(
        muon_params=[p],
        adamw_params=[],
        lr=1.0,
        ns_steps=ns_steps,
        lr_prefactor=prefactor,
    )


@pytest.mark.parametrize("Opt", [_bare_coupled, _bare_plain])
def test_moonlight_matches_phase1_formula(Opt):
    opt = Opt("moonlight")
    # `0.2 * sqrt(max(A, B))` for shape (128, 64) ⇒ 0.2 * sqrt(128).
    val = opt.adjust_lr_for_muon(1.0, (128, 64))
    assert val == pytest.approx(0.2 * math.sqrt(128))


@pytest.mark.parametrize("Opt", [_bare_coupled, _bare_plain])
def test_bernstein_ratio_formula(Opt):
    opt = Opt("bernstein_ratio")
    # `0.2 * sqrt(d_out / d_in)` for shape (128, 64) ⇒ 0.2 * sqrt(2.0).
    val = opt.adjust_lr_for_muon(1.0, (128, 64))
    assert val == pytest.approx(0.2 * math.sqrt(128 / 64))


def test_cesista_formula_coupled():
    opt = _bare_coupled("cesista", coupled_steps=4)
    K = 4
    expected = 0.2 * math.sqrt(128) / (1.0 + math.log(K + 1))
    val = opt.adjust_lr_for_muon(1.0, (128, 64))
    assert val == pytest.approx(expected)


def test_cesista_formula_plain():
    opt = _bare_plain("cesista", ns_steps=5)
    K = 5
    expected = 0.2 * math.sqrt(128) / (1.0 + math.log(K + 1))
    val = opt.adjust_lr_for_muon(1.0, (128, 64))
    assert val == pytest.approx(expected)


@pytest.mark.parametrize("Opt", [_bare_coupled, _bare_plain])
def test_unknown_prefactor_raises(Opt):
    opt = Opt("moonlight")
    opt.lr_prefactor = "unsupported_policy"
    with pytest.raises(ValueError, match="Unknown lr_prefactor"):
        opt.adjust_lr_for_muon(1.0, (128, 64))
