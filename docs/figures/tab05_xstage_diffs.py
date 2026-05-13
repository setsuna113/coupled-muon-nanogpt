"""Tab 5 — Cross-stage paired-bootstrap differences (Coupled − Muon, Coupled − AdamW)."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import DATA_DIR, RUNG_ORDER, write_table
from _stats import best_lr_per_arm, paired_bootstrap


def _arm_at_best_lr(s: pd.DataFrame, opt: str, rung: str) -> pd.DataFrame:
    bests = best_lr_per_arm(s[s["rung"] == rung], group_keys=["rung", "optimizer"])
    row = bests[bests["optimizer"] == opt]
    if len(row) == 0:
        return s.iloc[0:0]
    lr = row.iloc[0]["lr"]
    return s[(s["optimizer"] == opt) & (s["rung"] == rung) & (s["lr"] == lr) & (~s["diverged"].astype(bool))]


def _fmt_ci(med: float, lo: float, hi: float, n: int) -> str:
    marker = r"\,\textbf{*}" if (lo <= 0 <= hi) else ""
    return f"{med:+.4f} [{lo:+.4f}, {hi:+.4f}]{marker}"


def main() -> None:
    s1 = pd.read_parquet(DATA_DIR / "coupled_muon_A0_repro_summary.parquet")
    s2 = pd.read_parquet(DATA_DIR / "coupled_muon_ladder_summary.parquet")
    s = pd.concat([s1, s2], ignore_index=True)

    rows = []
    for rg in RUNG_ORDER:
        coup = _arm_at_best_lr(s, "coupled_muon_v2", rg)
        muon = _arm_at_best_lr(s, "muon", rg)
        adamw = _arm_at_best_lr(s, "adamw", rg)
        ci_m = paired_bootstrap(coup, muon, on=["rung", "seed"])
        ci_a = paired_bootstrap(coup, adamw, on=["rung", "seed"])
        rows.append({
            "rung": rg,
            "n_m": ci_m.n,
            "m": _fmt_ci(ci_m.median, ci_m.lo, ci_m.hi, ci_m.n),
            "n_a": ci_a.n,
            "a": _fmt_ci(ci_a.median, ci_a.lo, ci_a.hi, ci_a.n),
        })

    cols = ["Rung", "$n$", "Coupled $-$ Muon (95\\% CI)", "$n$", "Coupled $-$ AdamW (95\\% CI)"]
    lines = [
        r"\begin{tabular}{lrlrl}",
        r"\toprule",
        " & ".join(cols) + r" \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(f"{r['rung']} & {r['n_m']} & {r['m']} & {r['n_a']} & {r['a']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write_table("tab05_xstage_diffs", "\n".join(lines) + "\n")
    # Save a free-text summary string for the prose to reference
    print(f"  Coupled-vs-Muon CI brackets zero on {sum(1 for r in rows if '*' in r['m'])}/{len(rows)} rungs.")


if __name__ == "__main__":
    main()
