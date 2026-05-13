"""Fig 8 — Stage 2 rung × LR heatmap of median final val loss, faceted by optimizer."""
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

    conv = s2[~s2["diverged"].astype(bool)]
    pivot = (
        conv.groupby(["optimizer", "rung", "lr"])["final_val_loss"]
        .median()
        .reset_index()
    )

    # Use the same LR grid per optimizer (each opt has only its own 5 LRs)
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.8), constrained_layout=True)
    rungs = [r for r in RUNG_ORDER if r != "A0"]
    vmin, vmax = 3.2, 4.5
    for ax, opt in zip(axes, OPTIMIZERS):
        sub = pivot[pivot["optimizer"] == opt]
        lrs = sorted(sub["lr"].unique())
        mat = np.full((len(rungs), len(lrs)), np.nan)
        for i, rg in enumerate(rungs):
            for j, l in enumerate(lrs):
                v = sub[(sub["rung"] == rg) & (sub["lr"] == l)]["final_val_loss"]
                if len(v):
                    mat[i, j] = v.iloc[0]
        im = ax.imshow(mat, aspect="auto", cmap="viridis_r", vmin=vmin, vmax=vmax)
        # Annotate cells
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if np.isnan(v):
                    ax.text(j, i, "div", ha="center", va="center", fontsize=6, color="white")
                else:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                            color="white" if v > (vmin + vmax) / 2 else "black")
        ax.set_xticks(range(len(lrs)))
        ax.set_xticklabels([f"{l:.0e}" for l in lrs], fontsize=6.5, rotation=45)
        ax.set_yticks(range(len(rungs)))
        ax.set_yticklabels(rungs, fontsize=8)
        ax.set_title(OPT_LABEL[opt])
        ax.set_xlabel("learning rate")
        if opt == OPTIMIZERS[0]:
            ax.set_ylabel("ladder rung")
    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02, label="median final val loss")
    cbar.ax.tick_params(labelsize=7)
    fig.suptitle("Stage 2: median final val loss per (rung, LR), faceted by optimizer", y=1.02, fontsize=9)
    save_fig(fig, "fig08_stage2_lr_rung_heatmap")


if __name__ == "__main__":
    main()
