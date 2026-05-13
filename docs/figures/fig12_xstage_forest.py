"""Fig 12 — THE HEADLINE: cross-stage forest of (Coupled − Muon, Coupled − AdamW)
95% CIs for A0 (n=5) and each Stage 2 rung (n=3).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import DATA_DIR, OPT_COLOR, RUNG_ORDER, save_fig, setup_mpl
from _stats import best_lr_per_arm, paired_bootstrap


def _arm_at_best_lr(s: pd.DataFrame, opt: str, rung: str) -> pd.DataFrame:
    bests = best_lr_per_arm(s[s["rung"] == rung], group_keys=["rung", "optimizer"])
    row = bests[bests["optimizer"] == opt]
    if len(row) == 0:
        return s.iloc[0:0]
    lr = row.iloc[0]["lr"]
    return s[(s["optimizer"] == opt) & (s["rung"] == rung) & (s["lr"] == lr) & (~s["diverged"].astype(bool))]


def main() -> None:
    setup_mpl()
    s1 = pd.read_parquet(DATA_DIR / "coupled_muon_A0_repro_summary.parquet")
    s2 = pd.read_parquet(DATA_DIR / "coupled_muon_ladder_summary.parquet")
    s = pd.concat([s1, s2], ignore_index=True)

    rows = []
    for rg in RUNG_ORDER:
        coup = _arm_at_best_lr(s, "coupled_muon_v2", rg)
        muon = _arm_at_best_lr(s, "muon", rg)
        adamw = _arm_at_best_lr(s, "adamw", rg)
        ci_m = paired_bootstrap(coup, muon, on=["rung", "seed"])
        ci_a = paired_bootstrap(coup, adamw, on=["rung", "seed"])
        rows.append({
            "rung": rg,
            "n_seed_repro": ci_m.n,
            "ci_m": ci_m,
            "ci_a": ci_a,
        })

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.6), sharey=True, gridspec_kw={"width_ratios": [1.0, 1.8]})
    for col, (key, color, title, xlim) in enumerate([
        ("ci_m", OPT_COLOR["coupled_muon_v2"], "Coupled − Muon", (-0.05, 0.05)),
        ("ci_a", OPT_COLOR["adamw"],           "Coupled − AdamW", (-0.40, 0.05)),
    ]):
        ax = axes[col]
        for i, row in enumerate(rows):
            ci = row[key]
            if ci.n == 0:
                continue
            ax.errorbar([ci.median], [i], xerr=[[ci.median - ci.lo], [ci.hi - ci.median]],
                        fmt="o", color=color, capsize=3, lw=1.0, markersize=5)
            bracket = " *" if ci.lo <= 0 <= ci.hi else ""
            ax.text(xlim[1] - 0.002, i, f"{ci.median:+.3f}{bracket}",
                    ha="right", va="center", fontsize=7.5)
        ax.axvline(0, color="0.4", lw=0.7, ls="--")
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([f"{r['rung']} (n={r['n_seed_repro']})" for r in rows])
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel("paired Δ final val loss (nats)")
        ax.set_xlim(*xlim)
        if col == 0:
            ax.set_ylabel("ladder rung")
    fig.suptitle("Cross-stage headline: paired-bootstrap 95% CIs at each arm's best LR\n"
                 "(* = CI brackets zero; A0 is Stage 1, B–H are Stage 2)",
                 y=1.05, fontsize=9)
    fig.tight_layout()
    save_fig(fig, "fig12_xstage_forest")


if __name__ == "__main__":
    main()
