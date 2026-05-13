"""Fig B.1 — Effect of `coupled_steps ∈ {0,1,2,4,8}` on final val loss at A0."""
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
    # Restrict to: coupled_muon_v2, qk+vo+updown all True, final_polish True, lr=3e-3.
    base = a[
        (a["optimizer"] == "coupled_muon_v2")
        & (a["couple_qk"]) & (a["couple_vo"]) & (a["couple_updown"])
        & (a["final_polish"])
        & (a["lr"] == 3e-3)
        & (~a["diverged"].astype(bool))
    ]
    print(f"  fig B.1 base n={len(base)}; coupled_steps={sorted(base['coupled_steps'].unique())}")
    grouped = base.groupby("coupled_steps")["final_val_loss"].apply(list).reset_index()

    xs, meds, los, his, ns = [], [], [], [], []
    for _, row in grouped.iterrows():
        ci = bootstrap_median_ci(row["final_val_loss"])
        xs.append(int(row["coupled_steps"]))
        meds.append(ci.median); los.append(ci.lo); his.append(ci.hi); ns.append(ci.n)

    # Also include Muon (coupled_steps=0 surrogate) at the same LR for reference.
    muon = a[
        (a["optimizer"] == "muon") & (a["lr"] == 3e-3) & (~a["diverged"].astype(bool))
    ]["final_val_loss"].tolist()
    ci_m = bootstrap_median_ci(muon) if muon else None

    order = sorted(range(len(xs)), key=lambda i: xs[i])
    xs, meds, los, his, ns = [xs[i] for i in order], [meds[i] for i in order], [los[i] for i in order], [his[i] for i in order], [ns[i] for i in order]

    fig, ax = plt.subplots(figsize=(5.4, 2.8))
    ax.errorbar(xs, meds, yerr=[[m - lo for m, lo in zip(meds, los)],
                                [hi - m for m, hi in zip(meds, his)]],
                fmt="o-", color=OPT_COLOR["coupled_muon_v2"], capsize=3, lw=1.0, markersize=5,
                label="Coupled Muon v2")
    for x, m, n in zip(xs, meds, ns):
        ax.text(x, m + 0.005, f"n={n}", ha="center", va="bottom", fontsize=6)
    if ci_m and ci_m.n > 0:
        ax.axhline(ci_m.median, color=OPT_COLOR["muon"], lw=0.9, ls="--",
                   label=f"Muon @ lr=3e-3 (median, n={ci_m.n})")
        ax.fill_between([min(xs) - 0.5, max(xs) + 0.5], ci_m.lo, ci_m.hi,
                        color=OPT_COLOR["muon"], alpha=0.10)
    ax.set_xlabel(r"$coupled\_steps$ (inner NS iterations in stage 1)")
    ax.set_ylabel("Final val loss")
    ax.set_title("App. B.1 — Effect of coupled_steps at A0 (lr=3e-3, full Q-K+V-O+up-down)")
    ax.set_xticks([0, 1, 2, 4, 8])
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    save_fig(fig, "figB1_coupled_steps")


if __name__ == "__main__":
    main()
