"""Stage 1.1 regression: CoupledMuon_v2(coupled_steps=0) must equal plain Muon.

`base.yaml:52` declares the contract `coupled_steps: 0 ⇒ plain Muon path`. Before
the Stage 1.1 fix, the kernel pre-divided by ‖XB‖_F even with steps=0, producing
neither plain Muon nor coupled Muon. This test pins down that the gate now works.
"""
from __future__ import annotations

import copy

import torch

from coupled_muon_nanogpt.model.transformer import GPT, BlockConfig, GPTConfig
from coupled_muon_nanogpt.optim.coupled_muon import CoupledMuon_v2
from coupled_muon_nanogpt.optim.factory import classify_parameters
from coupled_muon_nanogpt.optim.muon import Muon


def _make_model() -> GPT:
    block = BlockConfig(
        hidden=64,
        n_heads=4,
        intermediate=128,
        norm_type="rmsnorm",
        mlp_type="swiglu",
        pos_emb_type="rope",
        qk_norm=False,
        max_seq_len=128,
    )
    return GPT(GPTConfig(vocab_size=512, n_layers=2, block=block))


def _seed(s: int = 0) -> None:
    torch.manual_seed(s)


def _populate_grads(model: torch.nn.Module, scale: float = 0.1, seed: int = 1) -> None:
    g = torch.Generator().manual_seed(seed)
    for p in model.parameters():
        if p.requires_grad:
            p.grad = torch.randn(p.shape, generator=g) * scale


def test_coupled_steps_zero_matches_plain_muon():
    _seed(0)
    m_coupled = _make_model()
    m_muon = copy.deepcopy(m_coupled)

    # Build coupled groups for the coupled optimizer; build flat-muon groups for plain Muon.
    g_c = classify_parameters(m_coupled, n_heads=4)
    coupled_pairs = (
        [(a, b, h, False) for a, b, h in g_c.coupled_vo]
        + [(a, b, h, False) for a, b, h in g_c.coupled_updown]
        + [(a, b, h, True) for a, b, h in g_c.coupled_qk]
    )
    opt_c = CoupledMuon_v2(
        lr=1e-2,
        wd=0.0,
        coupled_pairs=coupled_pairs,
        muon_params=list(g_c.muon_2d),
        adamw_params=list(g_c.adamw_other) + list(g_c.router_params),
        momentum=0.0,  # disable momentum so the test is deterministic at first step
        nesterov=False,
        ns_steps=5,
        coupled_steps=0,
        use_multi_head=False,
        n_heads=4,
    )

    g_m = classify_parameters(
        m_muon, n_heads=4, couple_qk=False, couple_vo=False, couple_updown=False
    )
    opt_m = Muon(
        muon_params=list(g_m.muon_2d),
        adamw_params=list(g_m.adamw_other) + list(g_m.router_params),
        lr=1e-2,
        wd=0.0,
        momentum=0.0,
        nesterov=False,
        ns_steps=5,
    )

    _populate_grads(m_coupled, seed=42)
    # Mirror the same gradients on the second model param-by-param.
    state_dict_grads = {n: p.grad.clone() for n, p in m_coupled.named_parameters() if p.grad is not None}
    for n, p in m_muon.named_parameters():
        if n in state_dict_grads:
            p.grad = state_dict_grads[n].clone()

    opt_c.step()
    opt_m.step()

    # Compare every matrix-shape (2D) parameter the muon paths actually update.
    diffs = []
    for (n_c, p_c), (n_m, p_m) in zip(m_coupled.named_parameters(), m_muon.named_parameters()):
        assert n_c == n_m
        if p_c.ndim != 2:
            continue
        if "router" in n_c or "embed" in n_c or "lm_head" in n_c:
            continue
        d = (p_c.detach() - p_m.detach()).abs().max().item()
        diffs.append((n_c, d))

    max_diff = max(d for _, d in diffs)
    assert max_diff < 1e-4, f"coupled_steps=0 should equal plain Muon; max abs diff {max_diff:.2e} across:\n" + "\n".join(
        f"  {n}: {d:.2e}" for n, d in diffs if d > 1e-5
    )
