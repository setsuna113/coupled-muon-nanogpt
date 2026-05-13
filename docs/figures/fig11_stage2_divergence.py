"""Fig 11 — Stage 2 divergence heatmap: rung × LR, 3 panels by optimizer."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import DATA_DIR, OPT_LABEL, OPTIMIZERS, RUNG_ORDER, save_fig, setup_mpl


def main() -> None:
    setup_mpl()
    s2 = pd.read_parquet(DATA_DIR / "coupled_muon_ladder_summary.parquet")

    rungs = [r for r in RUNG_ORDER if r != "A0"]
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.5), constrained_layout=True)
    for ax, opt in zip(axes, OPTIMIZERS):
        sub = s2[s2["optimizer"] == opt]
        lrs = sorted(sub["lr"].dropna().unique())
        mat = np.zeros((len(rungs), len(lrs)))
        for i, rg in enumerate(rungs):
            for j, l in enumerate(lrs):
                cell = sub[(sub["rung"] == rg) & (sub["lr"] == l)]
                if len(cell):
                    mat[i, j] = cell["diverged"].astype(bool).mean()
                else:
                    mat[i, j] = np.nan
        im = ax.imshow(mat, aspect="auto", cmap="Reds", vmin=0, vmax=1)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if np.isnan(v):
                    txt = "·"
                else:
                    txt = f"{int(round(v * 100))}%"
                ax.text(j, i, txt, ha="center", va="center", fontsize=6.5,
                        color="white" if (not np.isnan(v) and v > 0.5) else "black")
        ax.set_xticks(range(len(lrs)))
        ax.set_xticklabels([f"{l:.0e}" for l in lrs], fontsize=6.5, rotation=45)
        ax.set_yticks(range(len(rungs)))
        ax.set_yticklabels(rungs)
        ax.set_title(OPT_LABEL[opt])
        ax.set_xlabel("learning rate")
        if opt == OPTIMIZERS[0]:
            ax.set_ylabel("ladder rung")
    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02, label="fraction of seeds diverged")
    cbar.ax.tick_params(labelsize=7)
    fig.suptitle("Stage 2: divergence rate per (rung, LR, opt); 88/270 cells overall", y=1.02, fontsize=9)
    save_fig(fig, "fig11_stage2_divergence")


if __name__ == "__main__":
    main()
