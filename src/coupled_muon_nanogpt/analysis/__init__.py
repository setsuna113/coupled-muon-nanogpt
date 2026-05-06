"""Sweep aggregation and bootstrap-CI analysis.

Run after a multi-seed sweep to compute median final loss with 95 % bootstrap
CIs across seeds, grouped by (config, optimizer, lr). The output CSV is the
input to the d.3 telescoping decision: pick the per-rung best LR by lowest
median, narrow the grid, repeat.
"""
from .bootstrap import aggregate_runs, bootstrap_median_ci

__all__ = ["aggregate_runs", "bootstrap_median_ci"]
