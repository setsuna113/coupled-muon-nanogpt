"""Unified dense+sparse Coupled-vs-Muon comparison table."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import DATA_DIR, RUNG_ORDER, write_table
from _stats import best_lr_per_arm, paired_bootstrap
from part2_analysis import dedupe_runs, load_part2_summary, stage3_rows, valid_final_rows


SPARSE_RUNG_ORDER = ["I", "I'", "J", "K"]


def _valid_dense_rows(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    mask = work["seed"].notna() & work["final_val_loss"].map(lambda x: math.isfinite(float(x)) if pd.notna(x) else False)
    mask &= work["state"].eq("finished")
    mask &= ~work["diverged"].fillna(False).astype(bool)
    return work[mask].copy()


def _arm_at_best_lr(df: pd.DataFrame, rung: str, optimizer: str) -> tuple[pd.DataFrame, float | None]:
    sub = df[df["rung"].eq(rung)]
    if sub.empty:
        return sub, None
    bests = best_lr_per_arm(sub, group_keys=["rung", "optimizer"])
    row = bests[bests["optimizer"].eq(optimizer)]
    if row.empty:
        return sub.iloc[0:0].copy(), None
    lr = float(row.iloc[0]["lr"])
    return sub[sub["optimizer"].eq(optimizer) & sub["lr"].eq(lr)].copy(), lr


def _fmt_lr(lr: float | None) -> str:
    if lr is None or not math.isfinite(lr):
        return "--"
    return f"{lr:.0e}"


def _fmt_delta(x: float) -> str:
    if not math.isfinite(x):
        return "--"
    return f"{x:+.4f}"


def _fmt_ratio(coupled: pd.DataFrame, muon: pd.DataFrame) -> str:
    if coupled.empty or muon.empty or "runtime_s" not in coupled or "runtime_s" not in muon:
        return "--"
    c_rt = coupled["runtime_s"].dropna()
    m_rt = muon["runtime_s"].dropna()
    if c_rt.empty or m_rt.empty:
        return "--"
    ratio = float(c_rt.median()) / float(m_rt.median())
    if not math.isfinite(ratio):
        return "--"
    return f"{ratio:.2f}x"


def _rows_for(df: pd.DataFrame, rungs: list[str], regime: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for rung in rungs:
        coupled, c_lr = _arm_at_best_lr(df, rung, "coupled_muon_v2")
        muon, m_lr = _arm_at_best_lr(df, rung, "muon")
        if coupled.empty or muon.empty:
            continue
        ci = paired_bootstrap(coupled, muon, on=["rung", "seed"])
        rows.append({
            "regime": regime,
            "rung": rung,
            "n": ci.n,
            "delta": _fmt_delta(ci.median),
            "muon_lr": _fmt_lr(m_lr),
            "coupled_lr": _fmt_lr(c_lr),
            "runtime_ratio": _fmt_ratio(coupled, muon),
        })
    return rows


def main() -> None:
    dense = pd.concat(
        [
            pd.read_parquet(DATA_DIR / "coupled_muon_A0_repro_summary.parquet"),
            pd.read_parquet(DATA_DIR / "coupled_muon_ladder_summary.parquet"),
        ],
        ignore_index=True,
    )
    dense = _valid_dense_rows(dense)

    part2_summary = load_part2_summary()
    kept, _ = dedupe_runs(part2_summary)
    sparse = stage3_rows(valid_final_rows(kept))

    rows = [
        *_rows_for(dense, RUNG_ORDER, "Dense"),
        *_rows_for(sparse, SPARSE_RUNG_ORDER, "Sparse"),
    ]

    lines = [
        r"\begin{tabular}{llrllll}",
        r"\toprule",
        r"Regime & Rung & Paired seeds & Coupled $-$ Muon & Muon LR & Coupled LR & C/M runtime \\",
        r"\midrule",
    ]
    last_regime = None
    for row in rows:
        regime = row["regime"] if row["regime"] != last_regime else ""
        lines.append(
            rf"{regime} & {row['rung']} & {row['n']} & {row['delta']} "
            rf"& {row['muon_lr']} & {row['coupled_lr']} & {row['runtime_ratio']} \\"
        )
        last_regime = row["regime"]
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    write_table("tab23_dense_sparse_muon", "\n".join(lines))


if __name__ == "__main__":
    main()
