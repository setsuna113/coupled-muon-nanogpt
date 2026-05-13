"""Fig 2 — A0 LR sensitivity per optimizer + bottom divergence strip."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import DATA_DIR, OPT_COLOR, OPT_LABEL, OPT_MARKER, OPTIMIZERS, save_fig, setup_mpl


def main() -> None:
    setup_mpl()
    s1 = pd.read_parquet(DATA_DIR / "coupled_muon_A0_repro_summary.parquet")

    fig, axes = plt.subplots(2, 3, figsize=(7.5, 3.3), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    for col, opt in enumerate(OPTIMIZERS):
        sub = s1[s1["optimizer"] == opt]
        ax_top = axes[0, col]
        ax_bot = axes[1, col]
        # Top: median + scatter of converged seeds
        converged = sub[~sub["diverged"].astype(bool)]
        diverged = sub[sub["diverged"].astype(bool)]
        if len(converged):
            agg = converged.groupby("lr")["final_val_loss"].agg(["median", "min", "max", "size"]).reset_index()
            ax_top.errorbar(
                agg["lr"], agg["median"],
                yerr=[agg["median"] - agg["min"], agg["max"] - agg["median"]],
                fmt=OPT_MARKER[opt], color=OPT_COLOR[opt], capsize=2, lw=0.8,
                markersize=4, label=f"converged (n=1–{int(agg['size'].max())})",
            )
        if len(diverged):
            ax_top.scatter(diverged["lr"], diverged["final_val_loss"],
                           marker="x", color=OPT_COLOR[opt], alpha=0.45, s=20, label="diverged")
        ax_top.set_xscale("log")
        ax_top.set_title(OPT_LABEL[opt])
        if col == 0:
            ax_top.set_ylabel("Final val loss")
        ax_top.set_ylim(3.2, 7.0)
        ax_top.legend(loc="upper left", fontsize=7, frameon=False)
        # Bottom: stacked bar of converged / diverged seed counts per LR
        all_lrs = sorted(sub["lr"].dropna().unique())
        conv_counts = [(sub[(sub["lr"] == l) & (~sub["diverged"].astype(bool))]).shape[0] for l in all_lrs]
        div_counts = [(sub[(sub["lr"] == l) & (sub["diverged"].astype(bool))]).shape[0] for l in all_lrs]
        x = np.arange(len(all_lrs))
        ax_bot.bar(x, conv_counts, color=OPT_COLOR[opt], alpha=0.85, label="conv")
        ax_bot.bar(x, div_counts, bottom=conv_counts, color="lightgrey", alpha=0.95, label="div")
        ax_bot.set_xticks(x)
        ax_bot.set_xticklabels([f"{l:.0e}" for l in all_lrs], fontsize=6, rotation=45)
        ax_bot.set_ylim(0, 5.5)
        if col == 0:
            ax_bot.set_ylabel("n seeds")
        ax_bot.set_xlabel("learning rate")
    fig.suptitle("Stage 1 (A0 LLaMA-60M): LR sensitivity and divergence count per optimizer", y=1.02, fontsize=9)
    save_fig(fig, "fig02_stage1_lr_sensitivity")


if __name__ == "__main__":
    main()
