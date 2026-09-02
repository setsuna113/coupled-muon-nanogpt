"""Telescoping LR sweep helper (Essential AI 2505.02222 protocol).

Given the (lr, val_loss) results at width N, emit the centered halved-spacing
grid for width 2N. Trivial but worth codifying so the sweep config can call it.

Unused at project close — kept as a reference implementation of the Essential
AI telescoping protocol; no config or script calls it.
"""
from __future__ import annotations

import math


def next_grid(prev_lrs: list[float], prev_losses: list[float], n_points: int = 5) -> list[float]:
    """Center on the LR with lowest loss; halve the (log) spacing; emit n_points."""
    if len(prev_lrs) != len(prev_losses) or not prev_lrs:
        raise ValueError("Need matched non-empty (lrs, losses)")
    best_idx = min(range(len(prev_losses)), key=lambda i: prev_losses[i])
    center_log = math.log10(prev_lrs[best_idx])
    # Estimate the prior spacing by neighbours of the best.
    neighbours = [prev_lrs[i] for i in range(len(prev_lrs)) if i != best_idx]
    if not neighbours:
        prev_spacing = 0.5
    else:
        prev_spacing = min(abs(math.log10(x) - center_log) for x in neighbours)
    new_spacing = prev_spacing / 2
    half = (n_points - 1) // 2
    return [10 ** (center_log + (i - half) * new_spacing) for i in range(n_points)]
