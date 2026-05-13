"""Fig B.5 — Muon NS-budget control: does Coupled's extra NS budget *alone*
explain the gain over plain Muon?

At lr=3e-3, plot final val loss for:
- Muon @ ns_steps ∈ {5, 7, 9, 12, ...}     (the ablation `muon_ns_budget` sweep)
- Coupled @ default settings (a reference line)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import DATA_DIR, OPT_COLOR, save_fig, setup_mpl
from _stats import bootstrap_median_ci


def main() -> None:
    setup_mpl()
    a = pd.read_parquet(DATA_DIR / "coupled_muon_ablation_summary.parquet")
    muon = a[(a["optimizer"] == "muon") & (a["lr"] == 3e-3) & (~a["diverged"].astype(bool))]
    if "ns_steps" not in muon.columns or muon["ns_steps"].nunique() <= 1:
        # Fallback: this ablation may not have ns_steps variation yet.
        fig, ax = plt.subplots(figsize=(5.4, 2.4))
        ax.text(0.5, 0.5, "(Muon NS-budget sweep not yet synced\nor only single ns_steps value present)",
                ha="center", va="center", transform=ax.transAxes, fontsize=8, color="0.4")
        ax.set_axis_off()
        save_fig(fig, "figB5_ns_budget")
        return

    grouped = muon.groupby("ns_steps")["final_val_loss"].apply(list).reset_index()
    fig, ax = plt.subplots(figsize=(5.4, 2.6))
    xs, meds, los, his = [], [], [], []
    for _, row in grouped.iterrows():
        ci = bootstrap_median_ci(row["final_val_loss"])
        xs.append(int(row["ns_steps"]))
        meds.append(ci.median); los.append(ci.lo); his.append(ci.hi)
    ax.errorbar(xs, meds, yerr=[[m - lo for m, lo in zip(meds, los)],
                                [hi - m for m, hi in zip(meds, his)]],
                fmt="o-", color=OPT_COLOR["muon"], capsize=3, lw=1.0, markersize=5, label="Muon")
    coup = a[(a["optimizer"] == "coupled_muon_v2") & (a["lr"] == 3e-3) &
             (a["coupled_steps"] == 4) & (a["final_polish"]) &
             (a["couple_qk"]) & (a["couple_vo"]) & (a["couple_updown"]) &
             (~a["diverged"].astype(bool))]["final_val_loss"].tolist()
    if coup:
        ci_c = bootstrap_median_ci(coup)
        ax.axhline(ci_c.median, color=OPT_COLOR["coupled_muon_v2"], lw=0.9, ls="--",
                   label=f"Coupled v2 default (n={ci_c.n})")
    ax.set_xlabel("ns_steps (Muon)")
    ax.set_ylabel("final val loss")
    ax.set_title("App. B.5 — Muon NS budget control (does extra NS alone match coupling?)")
    ax.legend(frameon=False, fontsize=7.5)
    save_fig(fig, "figB5_ns_budget")


if __name__ == "__main__":
    main()
