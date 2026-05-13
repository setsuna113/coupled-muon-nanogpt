"""Fig 1 — A0 val-loss curves at each optimizer's best LR (median ± IQR across seeds)."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import DATA_DIR, OPT_COLOR, OPT_LABEL, OPTIMIZERS, save_fig, setup_mpl
from _stats import best_lr_per_arm


def _curve_band(df_runs: pd.DataFrame, metric: str, n_bins: int = 40):
    """Bin by `tokens`, return (token_center, q25, q50, q75)."""
    df = df_runs[df_runs["metric"] == metric].dropna(subset=["value"])
    if len(df) == 0:
        return None
    tok = df["tokens"].to_numpy()
    if tok.max() == tok.min():
        return None
    edges = np.linspace(tok.min(), tok.max(), n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    q25, q50, q75 = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (tok >= lo) & (tok < hi)
        if not m.any():
            q25.append(np.nan); q50.append(np.nan); q75.append(np.nan)
            continue
        v = df.loc[m, "value"].to_numpy()
        q25.append(np.quantile(v, 0.25))
        q50.append(np.median(v))
        q75.append(np.quantile(v, 0.75))
    return centers, np.array(q25), np.array(q50), np.array(q75)


def main() -> None:
    setup_mpl()
    s1 = pd.read_parquet(DATA_DIR / "coupled_muon_A0_repro_summary.parquet")
    h1 = pd.read_parquet(DATA_DIR / "coupled_muon_A0_repro_history_bestlr_curves.parquet")
    bests = best_lr_per_arm(s1, group_keys=["rung", "optimizer"])

    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    for opt in OPTIMIZERS:
        row = bests[bests["optimizer"] == opt]
        if not len(row):
            continue
        lr = row.iloc[0]["lr"]
        opt_runs = s1[(s1["optimizer"] == opt) & (s1["lr"] == lr) & (~s1["diverged"].astype(bool))]
        rids = opt_runs["run_id"].tolist()
        hsub = h1[h1["run_id"].isin(rids)]
        out = _curve_band(hsub, "val_loss")
        if out is None:
            continue
        centers, q25, q50, q75 = out
        ax.fill_between(centers / 1e9, q25, q75, color=OPT_COLOR[opt], alpha=0.18)
        ax.plot(centers / 1e9, q50, color=OPT_COLOR[opt], lw=1.4,
                label=f"{OPT_LABEL[opt]} (lr={lr:.0e}, n={len(rids)})")
    ax.set_xlabel("Tokens (billions)")
    ax.set_ylabel("Validation loss")
    ax.set_title("Stage 1 (A0 LLaMA-60M): val-loss curves at each optimizer's best LR")
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    ax.set_ylim(3.3, 5.5)
    save_fig(fig, "fig01_stage1_loss_curves")


if __name__ == "__main__":
    main()
