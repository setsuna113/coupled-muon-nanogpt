"""Part II best-LR table for sparse-MoE rungs."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import OPT_LABEL, write_table
from part2_analysis import dedupe_runs, fmt_lr, fmt_num, load_part2_summary, stage3_best_lr_table


def _esc(s: str) -> str:
    return str(s).replace("_", r"\_")


def main() -> None:
    summary = load_part2_summary()
    kept, _ = dedupe_runs(summary)
    tab = stage3_best_lr_table(kept)
    lines = [
        r"\begin{tabular}{lllrl}",
        r"\toprule",
        r"Rung & Optimizer & Best LR & Median val loss & Seeds \\",
        r"\midrule",
    ]
    for _, row in tab.iterrows():
        opt = OPT_LABEL.get(row["optimizer"], row["optimizer"])
        lines.append(
            rf"{_esc(row['rung'])} & {_esc(opt)} & {fmt_lr(row['best_lr'])} "
            rf"& {fmt_num(row['median_loss'], 4)} & {_esc(row['seeds'])} \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    write_table("tab21_part2_best_lr", "\n".join(lines))


if __name__ == "__main__":
    main()
