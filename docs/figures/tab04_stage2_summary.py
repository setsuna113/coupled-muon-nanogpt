"""Tab 4 — Stage 2 numerical summary: rung × optimizer → best LR, median, CI, n."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import DATA_DIR, OPT_LABEL, OPTIMIZERS, RUNG_ORDER, write_table
from _stats import best_lr_per_arm, bootstrap_median_ci


def main() -> None:
    s2 = pd.read_parquet(DATA_DIR / "coupled_muon_ladder_summary.parquet")
    bests = best_lr_per_arm(s2, group_keys=["rung", "optimizer"])

    rungs = [r for r in RUNG_ORDER if r != "A0"]
    rows = []
    for rg in rungs:
        for opt in OPTIMIZERS:
            row = bests[(bests["optimizer"] == opt) & (bests["rung"] == rg)]
            if not len(row):
                continue
            lr = row.iloc[0]["lr"]
            sub_all = s2[(s2["rung"] == rg) & (s2["optimizer"] == opt) & (s2["lr"] == lr)]
            sub_conv = sub_all[~sub_all["diverged"].astype(bool)]
            ci = bootstrap_median_ci(sub_conv["final_val_loss"].tolist())
            rows.append({
                "rung": rg,
                "opt": OPT_LABEL[opt],
                "lr": f"{lr:.0e}",
                "median": f"{ci.median:.4f}",
                "ci": f"[{ci.lo:.4f}, {ci.hi:.4f}]",
                "n_conv": ci.n,
                "n_total": len(sub_all),
            })

    cols = ["Rung", "Optimizer", "Best LR", "Median val loss", "95\\% CI", "$n_\\mathrm{conv}/n_\\mathrm{total}$"]
    lines = [
        r"\begin{tabular}{llrrcr}",
        r"\toprule",
        " & ".join(cols) + r" \\",
        r"\midrule",
    ]
    prev_rung = None
    for r in rows:
        rung = r["rung"] if r["rung"] != prev_rung else ""
        prev_rung = r["rung"]
        if rung != "" and prev_rung is not None and rung != rows[0]["rung"]:
            lines.append(r"\addlinespace")
        lines.append(
            f"{rung} & {r['opt']} & {r['lr']} & {r['median']} & {r['ci']} & {r['n_conv']}/{r['n_total']} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table("tab04_stage2_summary", "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
