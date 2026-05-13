"""Tab 1 — Dense ladder rung definitions (rung × architectural knobs)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import REPO_ROOT, RUNG_ORDER, write_table

# Hand-curated knob table per `configs/ladder/*.yaml` and experiment.md §d.2.
# Sourced once at write-time; not re-derived from YAML to avoid coupling the
# report to optional config-file fields.
ROWS = [
    # rung, MLP, intermediate, norm, pos_emb, qk_norm, tokens, notes
    ("A0", "SwiGLU",       "1408",  "RMSNorm",   "RoPE",       "off", "1.2B",  "LLaMA-60M baseline (Stage 1 origin)"),
    ("B",  "GELU 2-mat",   "2048",  "RMSNorm",   "RoPE",       "off", "2.5B",  "drop SwiGLU gate"),
    ("C",  "GELU 2-mat",   "2048",  "RMSNorm",   "learned",    "off", "2.5B",  "swap RoPE for learned pos."),
    ("D",  "GELU 2-mat",   "2048",  "LayerNorm", "learned",    "off", "2.5B",  "swap RMSNorm for LayerNorm + biases"),
    ("E",  "GELU 2-mat",   "2048",  "LayerNorm", "learned",    "off", "2.5B",  "Karpathy nanoGPT reference"),
    ("G",  "SwiGLU",       "1408",  "RMSNorm",   "RoPE",       "on",  "2.5B",  "A0 + qk\\_norm"),
    ("H",  "ReLU\\textsuperscript{2} 2-mat", "2048", "RMSNorm", "RoPE", "on", "2.5B", "modded-nanogpt-style dense"),
]


def main() -> None:
    cols = ["Rung", "MLP", "Inter.", "Norm", "Pos.", "QK-Norm", "Tokens", "What changes vs the previous rung"]
    lines = [
        r"\begin{tabular}{llrlllrl}",
        r"\toprule",
        " & ".join(cols) + r" \\",
        r"\midrule",
    ]
    for r in ROWS:
        lines.append(" & ".join(str(c) for c in r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table("tab01_ladder_defs", "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
