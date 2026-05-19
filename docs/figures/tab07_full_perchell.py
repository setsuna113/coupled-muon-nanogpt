"""Tab 7 (Appendix D) — Paired endpoint evidence.

Historical filename retained because the main report already includes this
script's output. The table intentionally replaces the older full per-cell dump:
one row per shared seed after each Muon-family arm is restricted to its own
best LR. This is the compact evidence needed to audit the headline
Coupled-vs-Muon medians without opening W&B.
"""
from __future__ import annotations

import sys
import math
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import DATA_DIR, RUNG_ORDER, write_table
from _stats import best_lr_per_arm
from part2_analysis import dedupe_runs, load_part2_summary, stage3_rows, valid_final_rows

SPARSE_RUNG_ORDER = ["I", "I'", "J", "K"]


def _finite(x: object) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _valid_dense_rows(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    mask = work["seed"].notna() & work["final_val_loss"].map(_finite)
    if "state" in work.columns:
        mask &= work["state"].eq("finished")
    if "diverged" in work.columns:
        mask &= ~work["diverged"].fillna(False).astype(bool)
    return work[mask].copy()


def _paired_rows_for(df: pd.DataFrame, rung: str) -> list[dict[str, object]]:
    sub = df[df["rung"].eq(rung)].copy()
    if sub.empty:
        return []
    bests = best_lr_per_arm(sub, group_keys=["rung", "optimizer"])
    muon_best = bests[bests["optimizer"].eq("muon")]
    coupled_best = bests[bests["optimizer"].eq("coupled_muon_v2")]
    if muon_best.empty or coupled_best.empty:
        return []

    muon_lr = float(muon_best.iloc[0]["lr"])
    coupled_lr = float(coupled_best.iloc[0]["lr"])
    muon = sub[
        sub["optimizer"].eq("muon") & sub["lr"].eq(muon_lr)
    ][["seed", "final_val_loss"]].rename(columns={"final_val_loss": "muon_loss"})
    coupled = sub[
        sub["optimizer"].eq("coupled_muon_v2") & sub["lr"].eq(coupled_lr)
    ][["seed", "final_val_loss"]].rename(columns={"final_val_loss": "coupled_loss"})
    paired = muon.merge(coupled, on="seed", how="inner").sort_values("seed")

    rows = []
    for _, row in paired.iterrows():
        muon_loss = float(row["muon_loss"])
        coupled_loss = float(row["coupled_loss"])
        rows.append({
            "rung": rung,
            "seed": int(row["seed"]),
            "muon_lr": muon_lr,
            "coupled_lr": coupled_lr,
            "muon_loss": muon_loss,
            "coupled_loss": coupled_loss,
            "delta": coupled_loss - muon_loss,
        })
    return rows


def _fmt_lr(x: float) -> str:
    return f"{x:.0e}"


def _fmt_loss(x: float) -> str:
    return f"{x:.4f}"


def _fmt_delta(x: float) -> str:
    return f"{x:+.4f}"


def main() -> None:
    dense = pd.concat(
        [
            pd.read_parquet(DATA_DIR / "coupled_muon_A0_repro_summary.parquet"),
            pd.read_parquet(DATA_DIR / "coupled_muon_ladder_summary.parquet"),
        ],
        ignore_index=True,
        sort=False,
    )
    dense = _valid_dense_rows(dense)

    part2_summary = load_part2_summary()
    kept, _ = dedupe_runs(part2_summary)
    sparse = stage3_rows(valid_final_rows(kept))

    rows = []
    for rung in RUNG_ORDER:
        rows.extend(_paired_rows_for(dense, rung))
    for rung in SPARSE_RUNG_ORDER:
        rows.extend(_paired_rows_for(sparse, rung))

    cols = ["Rung", "Seed", "Muon LR", "Coupled LR", "Muon final", "Coupled final", "$\\Delta$"]
    lines = [
        r"\begin{longtable}{llrrrrr}",
        (
            r"\caption{Paired endpoint evidence for the Coupled-vs-Muon headline "
            r"claims. Each row is a shared seed after restricting each arm to its "
            r"own best LR and excluding divergent or non-final rows. "
            r"$\Delta$ is Coupled $-$ Muon final validation loss, so negative "
            r"values favour \optCoupled{}. A0 is an in-repo LLaMA-like 60M "
            r"harness reproduction, not proof of full external LLaMA equivalence.}"
            r"\label{tab:paired-endpoints} \\"
        ),
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
    last_rung = None
    for row in rows:
        if last_rung is None:
            lines.append(r"\multicolumn{7}{l}{\emph{Dense 60M-CS rungs}} \\")
        if last_rung == "H" and row["rung"] == "I":
            lines.append(r"\addlinespace")
            lines.append(r"\multicolumn{7}{l}{\emph{Sparse-MoE Stage 3 rungs}} \\")
        rung = row["rung"] if row["rung"] != last_rung else ""
        lines.append(
            f"{rung} & {row['seed']} & {_fmt_lr(row['muon_lr'])} & {_fmt_lr(row['coupled_lr'])} "
            f"& {_fmt_loss(row['muon_loss'])} & {_fmt_loss(row['coupled_loss'])} "
            f"& {_fmt_delta(row['delta'])} \\\\"
        )
        last_rung = row["rung"]
    lines += [r"\end{longtable}"]
    write_table("tab07_full_perchell", "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
