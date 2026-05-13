"""Tab 2 — Compute budget per rung (tokens, est. hours, hardware)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import write_table

ROWS = [
    # stage, scope, n_cells, tokens/cell, est. hrs/cell, hardware
    ("Stage 1", "A0 (LLaMA-60M)",                           "75",  "1.2\\,B", "$\\approx$1.0", "2$\\times$H200"),
    ("Stage 2", "B/C/D/E/G/H (dense ladder)",               "270", "2.5\\,B", "$\\approx$2.0", "8$\\times$H100 (Rig A)"),
    ("App. B",  "\\S d.4 ablations (coupled\\_steps, polish, per-pair)", "105", "1.2\\,B", "$\\approx$0.9", "2$\\times$H200"),
]


def main() -> None:
    cols = ["Stage", "Scope", "Cells", "Tokens/cell", "GPU-h/cell", "Hardware"]
    lines = [
        r"\begin{tabular}{lllrrl}",
        r"\toprule",
        " & ".join(cols) + r" \\",
        r"\midrule",
    ]
    for r in ROWS:
        lines.append(" & ".join(str(c) for c in r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table("tab02_compute_budget", "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
