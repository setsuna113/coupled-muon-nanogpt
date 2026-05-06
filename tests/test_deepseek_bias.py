"""DeepSeek-V2 §3.2 aux-loss-free bias-update correctness (audit #7)."""
from __future__ import annotations

import torch

from coupled_muon_nanogpt.model.moe import MoEConfig, MoEFFN


def _make_moe(num_experts: int, top_k: int) -> MoEFFN:
    cfg = MoEConfig(
        hidden=16,
        intermediate=32,
        num_experts=num_experts,
        top_k=top_k,
        balancing_type="deepseek_bias",
        bias_update_lr=0.1,
    )
    return MoEFFN(cfg)


def test_balanced_load_produces_zero_or_small_bias_update():
    """In a perfectly balanced state, sign(f_i − 1/E) = 0; biases must NOT drift."""
    m = _make_moe(num_experts=4, top_k=2)
    # Manually set a uniform load.
    m._last_expert_load = torch.tensor([8.0, 8.0, 8.0, 8.0])
    before = m.router_bias.clone()
    m.update_router_bias()
    after = m.router_bias
    # f_i = 1/4 for all i; target = 1/4. sign(0) = 0. No drift.
    torch.testing.assert_close(after, before)


def test_imbalanced_load_pushes_bias_in_correct_direction():
    """Over-loaded expert should get its bias DECREASED (pushing tokens away);
    under-loaded expert should get its bias INCREASED."""
    m = _make_moe(num_experts=4, top_k=2)
    # Expert 0 over-loaded; expert 3 under-loaded; experts 1,2 balanced.
    m._last_expert_load = torch.tensor([16.0, 8.0, 8.0, 0.0])
    m.update_router_bias()
    bias = m.router_bias
    assert bias[0] < 0, f"over-loaded expert bias should decrease, got {bias[0]}"
    assert bias[3] > 0, f"under-loaded expert bias should increase, got {bias[3]}"
    # Magnitudes should match the bias_update_lr.
    assert abs(abs(bias[0].item()) - 0.1) < 1e-6
    assert abs(abs(bias[3].item()) - 0.1) < 1e-6


def test_balanced_state_is_a_fixed_point_for_top_k_2():
    """Repro for the audit-#7 bug: with target = top_k / num_experts (the old,
    wrong formula), under f_i-sums-to-1 normalisation EVERY f_i is below the
    target for K>1, so all biases drift in the same direction. With the fix
    (target = 1/E), a balanced state stays put."""
    m = _make_moe(num_experts=8, top_k=2)
    m._last_expert_load = torch.tensor([8.0] * 8)
    before = m.router_bias.clone()
    for _ in range(10):
        m.update_router_bias()
    torch.testing.assert_close(m.router_bias, before)
