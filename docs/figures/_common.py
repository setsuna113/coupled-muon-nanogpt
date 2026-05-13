"""Shared plot style, paths, and constants for the docs/ figure pipeline.

Single source of truth for rung order, optimizer order/colors, and matplotlib
rcParams so every figure looks consistent. Imported by every fig*.py and
tab*.py script.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = REPO_ROOT / "docs"
DATA_DIR = DOCS_ROOT / "data"
FIGURES_DIR = DOCS_ROOT / "figures"
TABLES_DIR = DOCS_ROOT / "tables"

OPTIMIZERS = ["adamw", "muon", "coupled_muon_v2"]
OPT_LABEL = {
    "adamw": "AdamW",
    "muon": "Muon",
    "coupled_muon_v2": "Coupled Muon v2",
}
OPT_COLOR = {
    "adamw": "#f1a340",      # warm orange
    "muon": "#1f78b4",       # blue
    "coupled_muon_v2": "#d7301f",  # red
}
OPT_MARKER = {
    "adamw": "s",
    "muon": "o",
    "coupled_muon_v2": "D",
}

# Dense ladder rung short labels and their wandb config.name values.
RUNG_ORDER = ["A0", "B", "C", "D", "E", "G", "H"]
RUNG_LONG = {
    "A0": "ladder_A0_llama60m",
    "B": "ladder_B_gelu2mat",
    "C": "ladder_C_learnedpos",
    "D": "ladder_D_layernorm",
    "E": "ladder_E_karpathy",
    "G": "ladder_G_qknorm",
    "H": "ladder_H_modded_dense",
}
LONG_TO_SHORT = {v: k for k, v in RUNG_LONG.items()}

# Project IDs (entity = liuyc1025-university-of-cambridge, omitted from the
# names; see _fetch.PROJECT_PREFIX).
WANDB_ENTITY = "liuyc1025-university-of-cambridge"
PROJECT_STAGE1 = "coupled-muon-A0-repro"
PROJECT_STAGE2 = "coupled-muon-ladder"
PROJECT_ABLATION = "coupled-muon-ablation"


def setup_mpl() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,        # editable text in PDF (no Type-3 raster)
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.4,
    })


def save_fig(fig: plt.Figure, name: str) -> Path:
    out = FIGURES_DIR / f"{name}.pdf"
    fig.savefig(out)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")
    return out


def write_table(name: str, body: str) -> Path:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out = TABLES_DIR / f"{name}.tex"
    out.write_text(body)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")
    return out
