"""Fig 5 — A0 max attention logit (`probe/attn_logit/global_max`) vs tokens.

Tests the MuonClip-stability signal: does coupling tame Q-K logit growth
relative to vanilla Muon? (See experiment.md a.5, c.7.)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import DATA_DIR, OPT_COLOR, OPT_LABEL, OPTIMIZERS, save_fig, setup_mpl


def main() -> None:
    setup_mpl()
    h = pd.read_parquet(DATA_DIR / "coupled_muon_A0_repro_history_bestlr_probes.parquet")
    s1 = pd.read_parquet(DATA_DIR / "coupled_muon_A0_repro_summary.parquet")
    metric = "probe/attn_logit/global_max"
    if len(h) == 0 or metric not in set(h["metric"]):
        print("  no attn_logit data — skipping fig 5")
        return

    fig, ax = plt.subplots(figsize=(5.4, 2.6))
    for opt in OPTIMIZERS:
        run_ids = s1[s1["optimizer"] == opt]["run_id"].tolist()
        sub = h[(h["metric"] == metric) & (h["run_id"].isin(run_ids))]
        if len(sub) == 0:
            continue
        sub = sub.sort_values("tokens")
        ax.plot(sub["tokens"] / 1e9, sub["value"], color=OPT_COLOR[opt], lw=1.2, label=OPT_LABEL[opt])
    ax.axhline(50.0, color="grey", lw=0.6, ls=":", label="MuonClip τ=50 reference")
    ax.set_yscale("log")
    ax.set_xlabel("Tokens (B)")
    ax.set_ylabel(r"max pre-softmax attn logit")
    ax.set_title("Stage 1 (A0): global-max attention logit trace (MuonClip stability)")
    ax.legend(frameon=False, fontsize=7.5, loc="best")
    save_fig(fig, "fig05_stage1_attn_logit")


if __name__ == "__main__":
    main()
