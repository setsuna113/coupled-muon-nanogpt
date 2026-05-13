"""Fig 9 — Stage 2 paired-bootstrap forest: (Coupled − Muon) and (Coupled − AdamW) per rung."""
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
    s2 = pd.read_parquet(DATA_DIR / "coupled_muon_ladder_summary.parquet")

    rungs = [r for r in RUNG_ORDER if r != "A0"]
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.0), sharey=True)
    for col, (other, color, title) in enumerate([
        ("muon",  OPT_COLOR["coupled_muon_v2"], "Coupled − Muon"),
        ("adamw", OPT_COLOR["adamw"],           "Coupled − AdamW"),
    ]):
        ax = axes[col]
        for i, rg in enumerate(rungs):
            coup = _arm_at_best_lr(s2, "coupled_muon_v2", rg)
            ot   = _arm_at_best_lr(s2, other, rg)
            ci = paired_bootstrap(coup, ot, on=["rung", "seed"])
            if ci.n == 0:
                continue
            ax.errorbar([ci.median], [i], xerr=[[ci.median - ci.lo], [ci.hi - ci.median]],
                        fmt="o", color=color, capsize=3, lw=1.0, markersize=4)
            bracket = "*" if ci.lo <= 0 <= ci.hi else ""
            ax.text(0.99, i, f"{ci.median:+.3f}{bracket}",
                    transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=7)
        ax.axvline(0, color="0.4", lw=0.7, ls="--")
        ax.set_yticks(range(len(rungs)))
        ax.set_yticklabels(rungs)
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel("paired Δ final val loss (nats)")
        if col == 0:
            ax.set_ylabel("ladder rung")
        ax.set_xlim(-0.4 if col == 1 else -0.06, 0.06 if col == 0 else 0.04)
    fig.suptitle("Stage 2: paired-bootstrap 95% CIs at each arm's best LR (* = CI brackets 0)",
                 y=1.02, fontsize=9)
    fig.tight_layout()
    save_fig(fig, "fig09_stage2_forest")


if __name__ == "__main__":
    main()
