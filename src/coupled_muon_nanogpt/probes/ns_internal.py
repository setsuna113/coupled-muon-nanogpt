"""Newton-Schulz internal Frobenius-norm probe (diagnostic).

Per experiment.md d.3: log ‖X‖_F per coupled iteration to detect divergence
(d.6.3 NaN-watch signal). The CoupledMuon_v2 inner kernels are
`@torch.compile`'d, so direct hooks would force recompiles. Instead this probe
runs a parallel "shadow trace" in eager fp32 on the *current* gradient and
partner weight for one Q-K pair and one up-down pair.

Per experiment.md d.4 (`final_polish` ablation row): the probe also records
the post-stage-1 spectral norm σ_max(X) — computed via cheap power iteration
on the final shadow iterate. This is the diagnostic that tells you whether
stage 2 (the trailing zeropower NS pass, gated by `final_polish`) is doing
real work or is approximately a no-op because stage-1 already produces a
near-orthogonal output. If σ_max(post-stage-1) ≈ 1 across coupled pairs at
the operating point, `final_polish=False` should match training behavior.

Cost: ~2× a full stage-1 pass (both sides) plus two short power iterations
per probe fire — negligible at the d.3 cadence. Healthy band per Bernstein-Newhouse / Polar Express: ‖X‖_F in
roughly (0.5, 5.0); values outside flag divergence.

Must be registered with `needs_grads=True` so it fires before optimizer.step()
zeros the .grad tensors. No-ops on plain Muon (no coupled_pairs).
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from ..optim.coupled_muon import CoupledMuon_v2


@torch.no_grad()
def ns_internal_probe(
    _model: nn.Module,
    optimizer: torch.optim.Optimizer,
    _ctx: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(optimizer, CoupledMuon_v2):
        return {"_note": "ns_internal probe no-op: optimizer is not CoupledMuon_v2"}
    coupled_pairs = optimizer.coupled_pairs
    coupled_steps = optimizer.coupled_steps
    if not coupled_pairs or coupled_steps <= 0:
        return {"_note": "ns_internal probe no-op: no coupled pairs or coupled_steps=0"}

    out: dict[str, Any] = {}
    qk_done = False
    other_done = False
    for param_A, param_B, _h, is_qk in coupled_pairs:
        if is_qk and qk_done:
            continue
        if (not is_qk) and other_done:
            continue
        g_A = param_A.grad
        g_B = param_B.grad
        if g_A is None or g_B is None:
            continue
        key = "qk0" if is_qk else "ud_or_vo_0"
        out[key] = _shadow_trace_pair(g_A, g_B, param_A.data, param_B.data, coupled_steps, is_qk)
        if is_qk:
            qk_done = True
        else:
            other_done = True
        if qk_done and other_done:
            break
    return out


def _shadow_trace_pair(
    g_A: torch.Tensor,
    g_B: torch.Tensor,
    param_A: torch.Tensor,
    param_B: torch.Tensor,
    coupled_steps: int,
    is_qk: bool,
) -> dict[str, Any]:
    """Eager-fp32 re-run of the 2-D coupled-NS inner loop with per-iteration
    ‖X‖_F logged. Re-runs the same recurrence with the canonical Bernstein
    triple hardcoded; it does NOT read `optimizer._coeffs_stage1`, so under
    ns_coefficients=polar_express|cesista it would trace the Bernstein iteration
    rather than the one the optimizer ran (every sweep that enabled this probe
    used bernstein, so the committed series are consistent). Operates on the
    flat 2-D shape (no multi-head reshape) — sufficient for divergence
    detection."""
    a, b, c = (3.4445, -4.7750, 2.0315)

    A_data = param_A.float()
    B_data = param_B.float()
    G_A = g_A.detach().float()
    G_B = g_B.detach().float()

    # _A side: iterate X = G_A; partner is B_data; W = X @ B_data.
    X_a = G_A / ((G_A @ B_data).norm() + 1e-7)
    norms_a = [X_a.norm().item()]
    for _ in range(coupled_steps):
        W = X_a @ B_data
        WWT = W @ W.T
        T = b * WWT + c * (WWT @ WWT)
        X_a = a * X_a + T @ X_a
        norms_a.append(X_a.norm().item())

    # _B side: iterate X = G_B; partner is A_data; W = A_data @ X.
    X_b = G_B / ((A_data @ G_B).norm() + 1e-7)
    norms_b = [X_b.norm().item()]
    for _ in range(coupled_steps):
        W = A_data @ X_b
        WTW = W.T @ W
        T = b * WTW + c * (WTW @ WTW)
        X_b = a * X_b + X_b @ T
        norms_b.append(X_b.norm().item())

    # Post-stage-1 spectral norm via power iteration. ‖X‖_spec ≈ 1 across
    # pairs ⇒ stage 2 (the trailing zeropower NS pass) is approximately a
    # no-op and `final_polish=False` should match training behavior; values
    # far from 1 mean stage 2 is doing real work (experiment.md d.4).
    stage1_sigma_max_a = _power_iter_sigma_max(X_a)
    stage1_sigma_max_b = _power_iter_sigma_max(X_b)

    return {
        "is_qk": is_qk,
        "norms_A": norms_a,
        "norms_B": norms_b,
        "max_norm": max(max(norms_a), max(norms_b)),
        "min_norm": min(min(norms_a), min(norms_b)),
        "diverged": any(not _finite(n) for n in (*norms_a, *norms_b)),
        # Stage-1 σ_max diagnostics (post-coupled-NS, pre-final-polish).
        # Healthy two-stage operating point: σ_max far from 1 (stage 2 needed).
        # Stage-2-redundant operating point: σ_max ≈ 1 (stage 2 ≈ no-op).
        "stage1_sigma_max_A": stage1_sigma_max_a,
        "stage1_sigma_max_B": stage1_sigma_max_b,
    }


@torch.no_grad()
def _power_iter_sigma_max(X: torch.Tensor, n_iter: int = 5) -> float:
    """Approximate ‖X‖_spec via a few rounds of power iteration on X^T X.

    Cheap (O(n_iter × m·n) flops). With n_iter=5 the relative error vs the true
    spectral norm is < 1% on well-conditioned matrices and < 5% on poorly
    conditioned ones, which is plenty for the diagnostic "is X near unit
    spectral norm".
    """
    if X.numel() == 0:
        return 0.0
    m, n = X.shape
    # Pick the smaller side to iterate against (cheaper).
    if m >= n:
        v = torch.randn(n, device=X.device, dtype=X.dtype)
        v = v / (v.norm() + 1e-12)
        for _ in range(n_iter):
            u = X @ v
            u = u / (u.norm() + 1e-12)
            v = X.t() @ u
            sigma = v.norm()
            v = v / (sigma + 1e-12)
        return float(sigma.item())
    else:
        u = torch.randn(m, device=X.device, dtype=X.dtype)
        u = u / (u.norm() + 1e-12)
        for _ in range(n_iter):
            v = X.t() @ u
            v = v / (v.norm() + 1e-12)
            u = X @ v
            sigma = u.norm()
            u = u / (sigma + 1e-12)
        return float(sigma.item())


def _finite(x: float) -> bool:
    import math

    return math.isfinite(x)
