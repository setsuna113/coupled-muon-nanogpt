"""Fig B.2 — final_polish on/off effect on final val loss (left) + the optional
post-stage-1 σ_max diagnostic trace (right) if available.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import DATA_DIR, OPT_COLOR, save_fig, setup_mpl
from _stats import bootstrap_median_ci


def main() -> None:
    setup_mpl()
    a = pd.read_parquet(DATA_DIR / "coupled_muon_ablation_summary.parquet")
    base = a[
        (a["optimizer"] == "coupled_muon_v2")
        & (a["coupled_steps"] == 4)
        & (a["couple_qk"]) & (a["couple_vo"]) & (a["couple_updown"])
        & (a["lr"] == 3e-3)
    ]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 2.8))

    ax = axes[0]
    rows = []
    for fp in [True, False]:
        sub = base[(base["final_polish"] == fp) & (~base["diverged"].astype(bool))]
        ci = bootstrap_median_ci(sub["final_val_loss"].tolist())
        n_div = ((base["final_polish"] == fp) & (base["diverged"].astype(bool))).sum()
        rows.append((str(fp), ci.median, ci.lo, ci.hi, ci.n, int(n_div)))
    xs = np.arange(len(rows))
    meds = [r[1] for r in rows]
    los = [r[1] - r[2] for r in rows]
    his = [r[3] - r[1] for r in rows]
    ax.bar(xs, meds, color=OPT_COLOR["coupled_muon_v2"], alpha=0.85)
    ax.errorbar(xs, meds, yerr=[los, his], fmt="none", color="0.2", capsize=3, lw=0.8)
    for i, r in enumerate(rows):
        ax.text(i, r[1] + 0.005, f"n={r[4]}, div={r[5]}", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(xs); ax.set_xticklabels([f"final_polish={r[0]}" for r in rows], fontsize=8)
    ax.set_ylabel("final val loss")
    ax.set_title("Effect of stage-2 polishing NS pass")
    ax.set_ylim(3.30, 4.0)

    # Right panel: σ_max trace if probe data available.
    ax2 = axes[1]
    sigma_path = DATA_DIR / "coupled_muon_ablation_history_final_polish_sigma.parquet"
    if sigma_path.exists():
        h = pd.read_parquet(sigma_path)
        if len(h) > 0:
            for metric in ["probe/ns_internal/coupled_pre_polish/sigma_max", "probe/ns_internal/coupled_post_polish/sigma_max"]:
                sub = h[h["metric"] == metric].sort_values("tokens")
                if not len(sub):
                    continue
                ax2.plot(sub["tokens"] / 1e9, sub["value"], lw=1.1,
                         label=metric.split("/")[-2])
            ax2.set_yscale("log")
            ax2.set_xlabel("Tokens (B)")
            ax2.set_ylabel(r"$\sigma_\mathrm{max}(X)$")
            ax2.set_title("Pre- vs post-polish operator-norm trace")
            ax2.axhline(1.0, color="grey", lw=0.5, ls=":")
            ax2.legend(frameon=False, fontsize=7)
        else:
            ax2.text(0.5, 0.5, "(σ_max probe history not available\nin current sync)",
                     ha="center", va="center", transform=ax2.transAxes, fontsize=8, color="0.4")
            ax2.set_axis_off()
    else:
        ax2.text(0.5, 0.5, "(σ_max probe history not available\nin current sync)",
                 ha="center", va="center", transform=ax2.transAxes, fontsize=8, color="0.4")
        ax2.set_axis_off()
    fig.suptitle("App. B.2 — final_polish at A0 (lr=3e-3, coupled_steps=4, full pair coupling)", y=1.03, fontsize=9)
    fig.tight_layout()
    save_fig(fig, "figB2_final_polish")


if __name__ == "__main__":
    main()
