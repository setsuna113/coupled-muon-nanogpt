"""Part II Stage-3 paired-delta forest plot."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import MOE_RUNG_ORDER, save_fig, setup_mpl
from part2_analysis import dedupe_runs, load_part2_summary, stage3_forest_rows


def main() -> None:
    setup_mpl()
    summary = load_part2_summary()
    kept, _ = dedupe_runs(summary)
    rows = stage3_forest_rows(kept)
    comparisons = ["Coupled - Muon", "Muon - AdamW", "Coupled - AdamW"]
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.2), sharey=True)
    order = {r: i for i, r in enumerate(MOE_RUNG_ORDER)}

    for ax, comp in zip(axes, comparisons, strict=True):
        sub = rows[rows["comparison"].eq(comp)].copy()
        ax.axvline(0.0, color="0.35", lw=0.8)
        ax.set_title(comp)
        ax.set_xlabel("delta final val loss (nats)")
        if sub.empty:
            ax.text(0.5, 0.5, "no valid pairs", transform=ax.transAxes, ha="center", va="center")
            ax.set_yticks(range(len(MOE_RUNG_ORDER)))
            ax.set_yticklabels(MOE_RUNG_ORDER)
            continue
        sub["_ord"] = sub["rung"].map(order)
        sub = sub.sort_values("_ord", ascending=False)
        y = [order.get(r, 999) for r in sub["rung"]]
        xerr = [sub["median"] - sub["lo"], sub["hi"] - sub["median"]]
        ax.errorbar(sub["median"], y, xerr=xerr, fmt="o", color="#d7301f", ecolor="0.25", capsize=2)
        for _, row in sub.iterrows():
            ax.text(row["hi"] + 0.002, order.get(row["rung"], 999), f"n={int(row['n'])}", va="center", fontsize=7)
        ax.set_yticks(range(len(MOE_RUNG_ORDER)))
        ax.set_yticklabels(MOE_RUNG_ORDER)
        ax.set_ylim(-0.5, len(MOE_RUNG_ORDER) - 0.5)
    axes[0].set_ylabel("Sparse-MoE rung")
    fig.suptitle("Part II: Stage-3 paired-bootstrap deltas at each arm's own best LR", y=1.02)
    fig.tight_layout()
    save_fig(fig, "part2_fig_stage3_forest")


if __name__ == "__main__":
    main()
