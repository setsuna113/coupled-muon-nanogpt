"""Fig 3 — A0 paired-bootstrap forest of pairwise gaps at best LR."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import DATA_DIR, OPT_COLOR, save_fig, setup_mpl
from _stats import best_lr_per_arm, paired_bootstrap


def _arm_at_best_lr(s: pd.DataFrame, opt: str) -> pd.DataFrame:
    """Return the converged-seed rows of `opt` at its best LR (single-rung use only)."""
    bests = best_lr_per_arm(s, group_keys=["rung", "optimizer"])
    row = bests[bests["optimizer"] == opt].iloc[0]
    return s[(s["optimizer"] == opt) & (s["lr"] == row["lr"]) & (~s["diverged"].astype(bool))]


def main() -> None:
    setup_mpl()
    s1 = pd.read_parquet(DATA_DIR / "coupled_muon_A0_repro_summary.parquet")

    adamw = _arm_at_best_lr(s1, "adamw")
    muon = _arm_at_best_lr(s1, "muon")
    coup = _arm_at_best_lr(s1, "coupled_muon_v2")

    pairs = [
        ("Coupled − Muon",  coup, muon,  OPT_COLOR["coupled_muon_v2"]),
        ("Coupled − AdamW", coup, adamw, OPT_COLOR["coupled_muon_v2"]),
        ("Muon − AdamW",    muon, adamw, OPT_COLOR["muon"]),
    ]
    rows = []
    for label, a, b, color in pairs:
        ci = paired_bootstrap(a, b, on=["rung", "seed"])
        rows.append((label, ci.median, ci.lo, ci.hi, ci.n, color))

    fig, ax = plt.subplots(figsize=(5.6, 2.4))
    for i, (label, m, lo, hi, n, color) in enumerate(rows):
        ax.errorbar([m], [i], xerr=[[m - lo], [hi - m]], fmt="o", color=color,
                    capsize=3, lw=1.0, markersize=5)
        bracket = " (CI brackets 0)" if lo <= 0 <= hi else ""
        ax.text(hi + 0.005, i, f"Δ={m:+.3f}, n={n}{bracket}", va="center", fontsize=7.5)
    ax.axvline(0, color="0.4", lw=0.7, ls="--")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Paired (best-LR) final val-loss difference (nats)")
    ax.set_title("Stage 1 (A0): paired-bootstrap 95% CIs at each arm's best LR")
    ax.set_xlim(-0.35, 0.10)
    save_fig(fig, "fig03_stage1_forest")


if __name__ == "__main__":
    main()
