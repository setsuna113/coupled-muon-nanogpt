"""Tab 3 — Stage 1 numerical summary (optimizer × best LR → median, CI, n)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import DATA_DIR, OPT_LABEL, OPTIMIZERS, write_table
from _stats import best_lr_per_arm, bootstrap_median_ci


def main() -> None:
    s1 = pd.read_parquet(DATA_DIR / "coupled_muon_A0_repro_summary.parquet")
    bests = best_lr_per_arm(s1, group_keys=["rung", "optimizer"])

    rows = []
    for opt in OPTIMIZERS:
        row = bests[bests["optimizer"] == opt]
        if not len(row):
            continue
        lr = row.iloc[0]["lr"]
        sub_all = s1[(s1["optimizer"] == opt) & (s1["lr"] == lr)]
        sub_conv = sub_all[~sub_all["diverged"].astype(bool)]
        ci = bootstrap_median_ci(sub_conv["final_val_loss"].tolist())
        rows.append({
            "opt": OPT_LABEL[opt],
            "lr": f"{lr:.0e}",
            "median": f"{ci.median:.4f}",
            "ci": f"[{ci.lo:.4f}, {ci.hi:.4f}]",
            "n_conv": ci.n,
            "n_total": len(sub_all),
        })

    cols = ["Optimizer", "Best LR", "Median val loss", "95\\% CI", "$n_\\mathrm{conv}/n_\\mathrm{total}$"]
    lines = [
        r"\begin{tabular}{lrrcr}",
        r"\toprule",
        " & ".join(cols) + r" \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r['opt']} & {r['lr']} & {r['median']} & {r['ci']} & {r['n_conv']}/{r['n_total']} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table("tab03_stage1_summary", "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
