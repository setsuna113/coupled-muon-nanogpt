"""Part II NS-policy paired-delta heatmap."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import PROJECT_NS_POLICY, save_fig, setup_mpl
from _stats import paired_bootstrap
from part2_analysis import dedupe_runs, load_part2_summary, valid_final_rows


def main() -> None:
    setup_mpl()
    summary = load_part2_summary()
    kept, _ = dedupe_runs(summary)
    clean = valid_final_rows(kept[kept["project"].eq(PROJECT_NS_POLICY)] if not kept.empty else kept)
    policies = ["bernstein", "cesista", "polar_express"]
    ks = [3, 5, 8]
    vals = np.full((len(policies), len(ks)), np.nan)
    labels = [["" for _ in ks] for _ in policies]
    for i, policy in enumerate(policies):
        for j, k in enumerate(ks):
            muon = clean[
                clean["optimizer"].eq("muon")
                & clean["ns_coefficients"].eq(policy)
                & clean["coupled_steps"].eq(k)
            ]
            coup = clean[
                clean["optimizer"].eq("coupled_muon_v2")
                & clean["ns_coefficients"].eq(policy)
                & clean["coupled_steps"].eq(k)
            ]
            ci = paired_bootstrap(coup, muon, on=["seed"])
            if math.isfinite(ci.median):
                vals[i, j] = ci.median
                labels[i][j] = f"{ci.median:+.4f}\nn={ci.n}"
            else:
                labels[i][j] = "missing"

    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    vmax = np.nanmax(np.abs(vals)) if np.isfinite(vals).any() else 0.02
    vmax = max(vmax, 0.005)
    im = ax.imshow(vals, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(ks)))
    ax.set_xticklabels([str(k) for k in ks])
    ax.set_yticks(range(len(policies)))
    ax.set_yticklabels([p.replace("_", " ") for p in policies])
    ax.set_xlabel("NS steps K")
    ax.set_title("NS-policy sweep: Coupled - Muon final val loss")
    for i in range(len(policies)):
        for j in range(len(ks)):
            ax.text(j, i, labels[i][j], ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label("nats (negative favors Coupled)")
    fig.tight_layout()
    save_fig(fig, "part2_fig_ns_policy_heatmap")


if __name__ == "__main__":
    main()
