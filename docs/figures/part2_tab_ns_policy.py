"""Part II NS-policy table."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import PROJECT_NS_POLICY, write_table
from _stats import paired_bootstrap
from part2_analysis import dedupe_runs, fmt_num, load_part2_summary, valid_final_rows


def _esc(s: str) -> str:
    return str(s).replace("_", r"\_")


def _median(sub: pd.DataFrame) -> str:
    if len(sub) == 0:
        return "--"
    return fmt_num(float(sub["final_val_loss"].median()), 4)


def _seeds(sub: pd.DataFrame) -> str:
    if len(sub) == 0:
        return "--"
    return ",".join(str(int(s)) for s in sorted(sub["seed"].dropna().unique()))


def main() -> None:
    summary = load_part2_summary()
    kept, _ = dedupe_runs(summary)
    clean = valid_final_rows(kept[kept["project"].eq(PROJECT_NS_POLICY)] if not kept.empty else kept)
    policies = ["bernstein", "cesista", "polar_express"]
    ks = [3, 5, 8]
    lines = [
        r"\begin{tabular}{llrrrl}",
        r"\toprule",
        r"Policy & K & Muon loss & Coupled loss & Coupled--Muon & Paired seeds \\",
        r"\midrule",
    ]
    for policy in policies:
        for k in ks:
            muon = clean[
                clean["optimizer"].eq("muon")
                & clean["ns_coefficients"].eq(policy)
                & clean["coupled_steps"].eq(k)
            ]
            coup = clean[
                clean["optimizer"].eq("coupled_muon_v2")
                & clean["ns_coefficients"].eq(policy)
                & clean["coupled_steps"].eq(k)
            ]
            ci = paired_bootstrap(coup, muon, on=["seed"])
            delta = fmt_num(ci.median, 4) if math.isfinite(ci.median) else "--"
            lines.append(
                rf"{_esc(policy)} & {k} & {_median(muon)} ({_seeds(muon)}) "
                rf"& {_median(coup)} ({_seeds(coup)}) & {delta} & {ci.n} \\"
            )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    write_table("tab22_part2_ns_policy", "\n".join(lines))


if __name__ == "__main__":
    main()
