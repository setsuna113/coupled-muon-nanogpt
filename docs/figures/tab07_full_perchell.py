"""Tab 7 (Appendix D) — Full per-cell summary as a longtable.

One row per (project, rung, optimizer, lr, seed). Sorted for human scan.
Divergent runs are kept (flagged) so the long tail is honest.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import DATA_DIR, write_table


def main() -> None:
    frames = []
    for name, project in [
        ("Stage 1", "coupled-muon-A0-repro"),
        ("Stage 2", "coupled-muon-ladder"),
        ("App. B",  "coupled-muon-ablation"),
    ]:
        f = DATA_DIR / f"{project.replace('-', '_')}_summary.parquet"
        if not f.exists():
            continue
        df = pd.read_parquet(f)[["rung", "optimizer", "lr", "seed", "final_val_loss", "diverged"]].copy()
        df.insert(0, "stage", name)
        frames.append(df)
    if not frames:
        return
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["stage", "rung", "optimizer", "lr", "seed"], na_position="last")

    cols = ["Stage", "Rung", "Opt.", "LR", "Seed", "Final val loss", "Div."]
    lines = [
        r"\begin{longtable}{lllrrrl}",
        r"\caption{All cells in Stage 1, Stage 2, and \S d.4 (Appendix B). Diverged cells excluded from CI computation but listed here for accounting.}\label{tab:perchell} \\",
        r"\toprule",
        " & ".join(cols) + r" \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        " & ".join(cols) + r" \\",
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{7}{r}{\textit{(continued)}} \\",
        r"\endfoot",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for _, r in df.iterrows():
        loss = "—" if pd.isna(r["final_val_loss"]) else f"{r['final_val_loss']:.4f}"
        div = r"\textbf{Y}" if r["diverged"] else ""
        lr = f"{r['lr']:.0e}" if not pd.isna(r["lr"]) else "—"
        seed = "—" if pd.isna(r["seed"]) else int(r["seed"])
        opt = str(r["optimizer"]).replace("_", r"\_")
        lines.append(f"{r['stage']} & {r['rung']} & {opt} & {lr} & {seed} & {loss} & {div} \\\\")
    lines += [r"\end{longtable}"]
    write_table("tab07_full_perchell", "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
