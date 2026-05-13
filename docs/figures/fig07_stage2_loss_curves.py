"""Fig 7 — Stage 2 per-rung val-loss curves at each opt's best LR (6 panels)."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import DATA_DIR, OPT_COLOR, OPT_LABEL, OPTIMIZERS, RUNG_ORDER, save_fig, setup_mpl
from _stats import best_lr_per_arm


def _curve_band(df_runs: pd.DataFrame, metric: str, n_bins: int = 30):
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
    s2 = pd.read_parquet(DATA_DIR / "coupled_muon_ladder_summary.parquet")
    h2 = pd.read_parquet(DATA_DIR / "coupled_muon_ladder_history_bestlr_curves.parquet")
    bests = best_lr_per_arm(s2, group_keys=["rung", "optimizer"])

    rungs = [r for r in RUNG_ORDER if r != "A0"]
    fig, axes = plt.subplots(2, 3, figsize=(8.0, 4.5), sharex=True, sharey=True)
    for ax, rung in zip(axes.flat, rungs):
        for opt in OPTIMIZERS:
            row = bests[(bests["optimizer"] == opt) & (bests["rung"] == rung)]
            if not len(row):
                continue
            lr = row.iloc[0]["lr"]
            opt_runs = s2[(s2["rung"] == rung) & (s2["optimizer"] == opt) & (s2["lr"] == lr) & (~s2["diverged"].astype(bool))]
            rids = opt_runs["run_id"].tolist()
            hsub = h2[h2["run_id"].isin(rids)]
            out = _curve_band(hsub, "val_loss")
            if out is None:
                continue
            centers, q25, q50, q75 = out
            ax.fill_between(centers / 1e9, q25, q75, color=OPT_COLOR[opt], alpha=0.18)
            ax.plot(centers / 1e9, q50, color=OPT_COLOR[opt], lw=1.1,
                    label=f"{OPT_LABEL[opt]} (lr={lr:.0e})")
        ax.set_title(f"Rung {rung}", fontsize=9)
        ax.set_ylim(3.2, 5.5)
        ax.legend(frameon=False, fontsize=6.5, loc="upper right")
    for ax in axes[1, :]:
        ax.set_xlabel("Tokens (B)")
    for ax in axes[:, 0]:
        ax.set_ylabel("val loss")
    fig.suptitle("Stage 2: val-loss curves at each (rung, opt) best LR (median ± IQR over 3 seeds)",
                 y=0.995, fontsize=9)
    fig.tight_layout()
    save_fig(fig, "fig07_stage2_loss_curves")


if __name__ == "__main__":
    main()
