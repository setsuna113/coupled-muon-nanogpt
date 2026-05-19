"""Part II completion matrix."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import write_table
from part2_analysis import completion_table, load_part2_summary


def _esc(s: str) -> str:
    return str(s).replace("_", r"\_")


def main() -> None:
    summary = load_part2_summary()
    tab = completion_table(summary)
    lines = [
        r"\begin{tabular}{lrrrrl}",
        r"\toprule",
        r"Project & Planned & W\&B runs & Final cells & Missing final & States \\",
        r"\midrule",
    ]
    for _, row in tab.iterrows():
        lines.append(
            rf"\code{{{_esc(row['project'])}}} & {int(row['expected'])} & {int(row['wandb_runs'])} "
            rf"& {int(row['final_cells'])} & {int(row['missing_final'])} & {_esc(row['states'])} \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    write_table("tab20_part2_completion", "\n".join(lines))


if __name__ == "__main__":
    main()
