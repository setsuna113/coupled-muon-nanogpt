"""Fig 6 — Dense ladder schematic A0 → B/C/D/E and A0 → G/H, one-knob-per-edge."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import save_fig, setup_mpl

# Per-rung short description of architecture knobs.
# Color encodes MLP family: SwiGLU (gate-bearing) vs gate-free 2-mat.
NODES = {
    "A0": dict(mlp="SwiGLU", norm="RMSNorm", pos="RoPE", qk="qk_norm=off", family="swiglu"),
    "B":  dict(mlp="GELU 2-mat", norm="RMSNorm", pos="RoPE", qk="qk_norm=off", family="2mat"),
    "C":  dict(mlp="GELU 2-mat", norm="RMSNorm", pos="learned", qk="qk_norm=off", family="2mat"),
    "D":  dict(mlp="GELU 2-mat", norm="LayerNorm", pos="learned", qk="qk_norm=off", family="2mat"),
    "E":  dict(mlp="GELU 2-mat", norm="LayerNorm", pos="learned", qk="qk_norm=off", family="2mat"),
    "G":  dict(mlp="SwiGLU", norm="RMSNorm", pos="RoPE", qk="qk_norm=on", family="swiglu"),
    "H":  dict(mlp="ReLU² 2-mat", norm="RMSNorm", pos="RoPE", qk="qk_norm=on", family="2mat"),
}
EDGES_TOP = [("A0", "B", "SwiGLU → GELU 2-mat"),
             ("B", "C", "RoPE → learned-pos"),
             ("C", "D", "RMSNorm → LayerNorm"),
             ("D", "E", "+ Karpathy nits")]
EDGES_BOT = [("A0", "G", "qk_norm: off → on"),
             ("G", "H", "SwiGLU → ReLU² 2-mat")]

FAMILY_COLOR = {"swiglu": "#cbc9e2", "2mat": "#fdbe85"}


def _draw_node(ax, x, y, key, w=1.4, h=0.95):
    info = NODES[key]
    box = patches.FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                 boxstyle="round,pad=0.02,rounding_size=0.08",
                                 linewidth=0.6, edgecolor="0.3",
                                 facecolor=FAMILY_COLOR[info["family"]])
    ax.add_patch(box)
    ax.text(x, y + h / 2 - 0.18, key, ha="center", va="top", fontsize=11, fontweight="bold")
    ax.text(x, y + 0.05, info["mlp"], ha="center", va="center", fontsize=7)
    ax.text(x, y - 0.10, info["norm"] + " / " + info["pos"], ha="center", va="center", fontsize=6.5)
    ax.text(x, y - 0.28, info["qk"], ha="center", va="center", fontsize=6.5, color="0.3")


def _draw_edge(ax, src, dst, label, y_label_offset=0.18):
    sx, sy = src
    dx, dy = dst
    ax.annotate("",
                xy=(dx - 0.7, dy), xytext=(sx + 0.7, sy),
                arrowprops=dict(arrowstyle="-|>", color="0.4", lw=0.8))
    mx, my = (sx + dx) / 2, (sy + dy) / 2 + y_label_offset
    ax.text(mx, my, label, ha="center", va="bottom", fontsize=6.8, color="0.25", style="italic")


def main() -> None:
    setup_mpl()
    fig, ax = plt.subplots(figsize=(8.2, 3.0))

    # Layout: A0 at origin; top row B-E to the right at y=1; bottom row G,H at y=-1.
    pos = {"A0": (0.0, 0.0)}
    for i, k in enumerate(["B", "C", "D", "E"]):
        pos[k] = (2.0 + 2.0 * i, 1.0)
    for i, k in enumerate(["G", "H"]):
        pos[k] = (2.0 + 2.0 * i, -1.0)

    for key, (x, y) in pos.items():
        _draw_node(ax, x, y, key)
    for a, b, lbl in EDGES_TOP:
        _draw_edge(ax, pos[a], pos[b], lbl)
    for a, b, lbl in EDGES_BOT:
        _draw_edge(ax, pos[a], pos[b], lbl, y_label_offset=-0.32)

    legend_handles = [
        patches.Patch(facecolor=FAMILY_COLOR["swiglu"], edgecolor="0.3", label="SwiGLU (gate-bearing)"),
        patches.Patch(facecolor=FAMILY_COLOR["2mat"], edgecolor="0.3", label="2-mat MLP (gate-free)"),
    ]
    ax.legend(handles=legend_handles, loc="lower left", fontsize=7, frameon=False, ncol=2)
    ax.text(0.0, 1.7, "Stage 2: dense ladder rungs (60M Common-Size: hidden=512, n_layers=8, n_heads=8)",
            fontsize=8, style="italic", color="0.3")

    ax.set_xlim(-1.2, 11.2)
    ax.set_ylim(-2.0, 2.0)
    ax.set_aspect("equal")
    ax.axis("off")
    save_fig(fig, "fig06_ladder_schematic")


if __name__ == "__main__":
    main()
