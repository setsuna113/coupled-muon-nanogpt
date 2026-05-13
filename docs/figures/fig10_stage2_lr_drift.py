"""Fig 10 — Best LR per optimizer per rung (log y); the 3e-3 → 3e-2 step at G/H."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import DATA_DIR, OPT_COLOR, OPT_LABEL, OPT_MARKER, OPTIMIZERS, RUNG_ORDER, save_fig, setup_mpl
from _stats import best_lr_per_arm


def main() -> None:
    setup_mpl()
    s1 = pd.read_parquet(DATA_DIR / "coupled_muon_A0_repro_summary.parquet")
    s2 = pd.read_parquet(DATA_DIR / "coupled_muon_ladder_summary.parquet")
    s = pd.concat([s1, s2], ignore_index=True)
    bests = best_lr_per_arm(s, group_keys=["rung", "optimizer"])

    fig, ax = plt.subplots(figsize=(5.6, 2.6))
    for opt in OPTIMIZERS:
        sub = bests[bests["optimizer"] == opt].set_index("rung").reindex(RUNG_ORDER).reset_index()
        sub = sub.dropna(subset=["lr"])
        ax.plot(sub["rung"], sub["lr"], marker=OPT_MARKER[opt], color=OPT_COLOR[opt],
                lw=1.2, markersize=5, label=OPT_LABEL[opt])
    ax.set_yscale("log")
    ax.set_ylabel("best LR (median final val loss)")
    ax.set_xlabel("ladder rung")
    ax.set_title("Best LR per (rung, optimizer): operating point drift between non-QK-Norm and QK-Norm rungs")
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    ax.axvspan(4.5, 6.5, color="0.92", zorder=0)  # G, H region
    ax.text(5.5, ax.get_ylim()[1] * 0.7, "qk-norm on", ha="center", va="top", fontsize=7, color="0.4", style="italic")
    save_fig(fig, "fig10_stage2_lr_drift")


if __name__ == "__main__":
    main()
