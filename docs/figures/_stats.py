"""Bootstrap helpers for the report.

Two flavours:

- `bootstrap_median_ci(losses)` — percentile bootstrap of the median of a
  single sample. Mirrors the helper in
  `src/coupled_muon_nanogpt/analysis/bootstrap.py`.

- `paired_bootstrap(df_a, df_b, on=['rung', 'seed'])` — paired bootstrap on the
  *median of seed-wise differences*. The two halves must share the pairing
  keys (e.g., rung+seed), and runs come in at each arm's own best LR per
  `reference_wandb_query.md` idiom — Muon-family and AdamW sweet spots differ
  by ~10×, so pairing at a common LR understates the gap.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CI:
    median: float
    lo: float
    hi: float
    n: int

    def brackets_zero(self) -> bool:
        if math.isnan(self.lo) or math.isnan(self.hi):
            return True
        return self.lo <= 0.0 <= self.hi


def bootstrap_median_ci(
    losses,
    *,
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    rng_seed: int = 0,
) -> CI:
    arr = np.asarray([x for x in losses if x is not None and math.isfinite(x)], dtype=float)
    if arr.size == 0:
        return CI(float("nan"), float("nan"), float("nan"), 0)
    if arr.size == 1:
        return CI(float(arr[0]), float(arr[0]), float(arr[0]), 1)
    rng = np.random.default_rng(rng_seed)
    medians = np.empty(n_resamples)
    n = arr.size
    for i in range(n_resamples):
        sample = arr[rng.integers(0, n, size=n)]
        medians[i] = np.median(sample)
    lo = float(np.quantile(medians, alpha / 2))
    hi = float(np.quantile(medians, 1 - alpha / 2))
    return CI(float(np.median(arr)), lo, hi, int(arr.size))


def best_lr_per_arm(
    df: pd.DataFrame,
    *,
    group_keys: list[str],
    lr_col: str = "lr",
    loss_col: str = "final_val_loss",
    drop_diverged: bool = True,
) -> pd.DataFrame:
    """For each group (e.g., (rung, optimizer)), find the LR with the smallest
    median final-val-loss across seeds (converged only). Returns a frame with
    one row per group and columns: *group_keys, lr, median_loss, n_seeds.
    """
    work = df.copy()
    if drop_diverged and "diverged" in work.columns:
        work = work[~work["diverged"].astype(bool)]
    work = work.dropna(subset=[loss_col, lr_col])
    by = work.groupby([*group_keys, lr_col], dropna=False)[loss_col].agg(["median", "size"]).reset_index()
    by = by.rename(columns={"median": "median_loss", "size": "n_seeds"})
    idx = by.groupby(group_keys, dropna=False)["median_loss"].idxmin()
    return by.loc[idx].reset_index(drop=True)


def paired_bootstrap(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    *,
    on: list[str] = ("rung", "seed"),
    loss_col: str = "final_val_loss",
    n_resamples: int = 10_000,
    alpha: float = 0.05,
    rng_seed: int = 0,
) -> CI:
    """Paired bootstrap on the median seed-wise (loss_a - loss_b) difference.

    Inner join on `on`; runs must have the *same* pairing-keys in both arms
    (typically rung+seed at each arm's best LR). Returns CI of (median(a) -
    median(b)) computed pair-wise.
    """
    on = list(on)
    a = df_a[[*on, loss_col]].copy().rename(columns={loss_col: "loss_a"})
    b = df_b[[*on, loss_col]].copy().rename(columns={loss_col: "loss_b"})
    merged = a.merge(b, on=on, how="inner").dropna()
    if len(merged) == 0:
        return CI(float("nan"), float("nan"), float("nan"), 0)
    diffs = (merged["loss_a"] - merged["loss_b"]).to_numpy()
    return bootstrap_median_ci(diffs, n_resamples=n_resamples, alpha=alpha, rng_seed=rng_seed)


def summary_with_ci(
    df: pd.DataFrame,
    *,
    group_keys: list[str],
    loss_col: str = "final_val_loss",
    drop_diverged: bool = True,
    n_resamples: int = 10_000,
) -> pd.DataFrame:
    work = df.copy()
    if drop_diverged and "diverged" in work.columns:
        work = work[~work["diverged"].astype(bool)]
    work = work.dropna(subset=[loss_col])
    rows = []
    for keys, sub in work.groupby(group_keys, dropna=False):
        ci = bootstrap_median_ci(sub[loss_col].tolist(), n_resamples=n_resamples)
        row = dict(zip(group_keys, keys if isinstance(keys, tuple) else (keys,)))
        row.update({
            "n_seeds": ci.n,
            "median": ci.median,
            "ci_lo": ci.lo,
            "ci_hi": ci.hi,
        })
        rows.append(row)
    return pd.DataFrame(rows)


__all__ = ["CI", "bootstrap_median_ci", "paired_bootstrap", "best_lr_per_arm", "summary_with_ci"]
