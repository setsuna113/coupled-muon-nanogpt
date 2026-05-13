"""Fig B.3 — Per-pair coupling ablation: which of {qk, vo, updown} carries the dense gain?"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import DATA_DIR, OPT_COLOR, save_fig, setup_mpl
from _stats import bootstrap_median_ci

CONFIGS = [
    ("ON / ON / ON", True, True, True),
    ("OFF / ON / ON", False, True, True),
    ("ON / OFF / ON", True, False, True),
    ("ON / ON / OFF", True, True, False),
]


def main() -> None:
    setup_mpl()
    a = pd.read_parquet(DATA_DIR / "coupled_muon_ablation_summary.parquet")
    base = a[
        (a["optimizer"] == "coupled_muon_v2")
        & (a["coupled_steps"] == 4)
        & (a["final_polish"])
        & (a["lr"] == 3e-3)
        & (~a["diverged"].astype(bool))
    ]

    rows = []
    for label, qk, vo, ud in CONFIGS:
        sub = base[(base["couple_qk"] == qk) & (base["couple_vo"] == vo) & (base["couple_updown"] == ud)]
        ci = bootstrap_median_ci(sub["final_val_loss"].tolist())
        rows.append((label, ci.median, ci.lo, ci.hi, ci.n))
    # Also Muon for reference
    muon = a[(a["optimizer"] == "muon") & (a["lr"] == 3e-3) & (~a["diverged"].astype(bool))]
    ci_m = bootstrap_median_ci(muon["final_val_loss"].tolist()) if len(muon) else None

    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    xs = np.arange(len(rows))
    meds = [r[1] for r in rows]
    los = [r[1] - r[2] for r in rows]
    his = [r[3] - r[1] for r in rows]
    ax.bar(xs, meds, color=OPT_COLOR["coupled_muon_v2"], alpha=0.85)
    ax.errorbar(xs, meds, yerr=[los, his], fmt="none", color="0.2", capsize=3, lw=0.8)
    for i, (label, m, lo, hi, n) in enumerate(rows):
        ax.text(i, m + 0.005, f"n={n}", ha="center", va="bottom", fontsize=6.5)
    if ci_m and ci_m.n > 0:
        ax.axhline(ci_m.median, color=OPT_COLOR["muon"], lw=0.9, ls="--",
                   label=f"Muon (median, n={ci_m.n})")
    ax.set_xticks(xs)
    ax.set_xticklabels([r[0] for r in rows], fontsize=7.5)
    ax.set_xlabel(r"coupling state (qk / vo / updown)")
    ax.set_ylabel("final val loss (median ± 95% CI)")
    ax.set_title("App. B.3 — Per-pair coupling ablation at A0 (lr=3e-3, coupled_steps=4)")
    ax.set_ylim(3.30, 3.55)
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    save_fig(fig, "figB3_per_pair")


if __name__ == "__main__":
    main()
