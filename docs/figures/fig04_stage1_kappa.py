"""Fig 4 — A0 coupled-pair κ traces for layer 0 (Q-K, V-O, up-down), Coupled vs Muon vs AdamW."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import DATA_DIR, OPT_COLOR, OPT_LABEL, OPTIMIZERS, save_fig, setup_mpl


PAIRS = [("qk.0", "Q–K (layer 0)"), ("vo.0", "V–O (layer 0)"), ("updown.0", "up–down (layer 0)")]


def main() -> None:
    setup_mpl()
    h = pd.read_parquet(DATA_DIR / "coupled_muon_A0_repro_history_bestlr_probes.parquet")
    if len(h) == 0:
        print("  empty probe parquet — skipping fig 4")
        return
    s1 = pd.read_parquet(DATA_DIR / "coupled_muon_A0_repro_summary.parquet")

    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.8), sharey=True)
    for ax, (pair_key, title) in zip(axes, PAIRS):
        metric = f"probe/coupled_pair/{pair_key}/kappa"
        for opt in OPTIMIZERS:
            run_ids = s1[s1["optimizer"] == opt]["run_id"].tolist()
            sub = h[(h["metric"] == metric) & (h["run_id"].isin(run_ids))]
            if len(sub) == 0:
                continue
            sub = sub.sort_values("tokens")
            ax.plot(sub["tokens"] / 1e9, sub["value"], color=OPT_COLOR[opt],
                    lw=1.1, label=OPT_LABEL[opt])
        ax.set_yscale("log")
        ax.set_xlabel("Tokens (B)")
        ax.set_title(title)
        if pair_key == PAIRS[0][0]:
            ax.set_ylabel(r"$\kappa(W_A W_B)$")
        ax.legend(frameon=False, fontsize=7, loc="upper right")
    fig.suptitle("Stage 1 (A0): condition number of bilinear products under each optimizer", y=1.02, fontsize=9)
    fig.tight_layout()
    save_fig(fig, "fig04_stage1_kappa")


if __name__ == "__main__":
    main()
