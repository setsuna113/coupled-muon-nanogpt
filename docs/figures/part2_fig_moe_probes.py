"""Part II MoE probe summaries at best LR."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import MOE_RUNG_ORDER, OPT_COLOR, OPT_LABEL, save_fig, setup_mpl
from part2_analysis import best_lr_run_rows, dedupe_runs, load_part2_summary, stage3_rows, valid_final_rows

METRICS = [
    ("probe_router_entropy_mean", "router entropy"),
    ("probe_load_imbalance_mean", "load imbalance"),
    ("probe_grad_norm_var_mean", "grad-norm var"),
    ("probe_attn_logit_global_max", "attn-logit max"),
    ("runtime_s", "runtime (s)"),
]


def main() -> None:
    setup_mpl()
    summary = load_part2_summary()
    kept, _ = dedupe_runs(summary)
    stage3 = stage3_rows(valid_final_rows(kept))
    best_rows, _ = best_lr_run_rows(stage3, group_keys=["rung", "optimizer"])
    fig, axes = plt.subplots(1, len(METRICS), figsize=(11.2, 2.6), sharex=True)
    order = {r: i for i, r in enumerate(MOE_RUNG_ORDER)}
    opts = ["adamw", "muon", "coupled_muon_v2"]
    offsets = {"adamw": -0.22, "muon": 0.0, "coupled_muon_v2": 0.22}

    if best_rows.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "no best-LR rows", transform=ax.transAxes, ha="center", va="center")
        save_fig(fig, "part2_fig_moe_probes")
        return

    for ax, (metric, title) in zip(axes, METRICS, strict=True):
        for opt in opts:
            sub = best_rows[best_rows["optimizer"].eq(opt)]
            xs = []
            ys = []
            for rung, rung_sub in sub.groupby("rung"):
                vals = rung_sub[metric].dropna() if metric in rung_sub else []
                if len(vals) == 0:
                    continue
                xs.append(order.get(rung, 999) + offsets[opt])
                ys.append(float(np.median(vals)))
            if xs:
                ax.scatter(xs, ys, s=18, label=OPT_LABEL.get(opt, opt), color=OPT_COLOR.get(opt, "0.3"))
        ax.set_title(title)
        ax.set_xticks(range(len(MOE_RUNG_ORDER)))
        ax.set_xticklabels(MOE_RUNG_ORDER, rotation=0)
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].set_ylabel("median over best-LR seeds")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    save_fig(fig, "part2_fig_moe_probes")


if __name__ == "__main__":
    main()
