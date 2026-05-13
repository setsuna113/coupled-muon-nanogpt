"""Fig B.4 — 2D heatmap of (coupled_steps × final_polish) interaction at A0."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import DATA_DIR, save_fig, setup_mpl


def main() -> None:
    setup_mpl()
    a = pd.read_parquet(DATA_DIR / "coupled_muon_ablation_summary.parquet")
    base = a[
        (a["optimizer"] == "coupled_muon_v2")
        & (a["couple_qk"]) & (a["couple_vo"]) & (a["couple_updown"])
        & (a["lr"] == 3e-3)
        & (~a["diverged"].astype(bool))
    ]
    steps = sorted(base["coupled_steps"].unique())
    polish = [True, False]
    mat = np.full((len(polish), len(steps)), np.nan)
    counts = np.zeros((len(polish), len(steps)), dtype=int)
    for i, fp in enumerate(polish):
        for j, st in enumerate(steps):
            sub = base[(base["final_polish"] == fp) & (base["coupled_steps"] == st)]
            if len(sub):
                mat[i, j] = float(sub["final_val_loss"].median())
                counts[i, j] = len(sub)

    fig, ax = plt.subplots(figsize=(5.2, 2.4))
    im = ax.imshow(mat, aspect="auto", cmap="viridis_r")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            n = counts[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center", fontsize=8)
            else:
                ax.text(j, i, f"{v:.3f}\n(n={n})", ha="center", va="center", fontsize=7,
                        color="white" if v > 3.5 else "black")
    ax.set_xticks(range(len(steps))); ax.set_xticklabels([str(s) for s in steps])
    ax.set_yticks(range(len(polish))); ax.set_yticklabels([f"polish={p}" for p in polish])
    ax.set_xlabel("coupled_steps")
    ax.set_title("App. B.4 — coupled_steps × final_polish at A0 (median final val loss)")
    fig.colorbar(im, ax=ax, label="median final val loss")
    save_fig(fig, "figB4_steps_x_polish")


if __name__ == "__main__":
    main()
