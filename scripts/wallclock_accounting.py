#!/usr/bin/env python3
"""Convert fixed-token Coupled-vs-Muon loss deltas into a wall-clock reading.

For each rung: take Muon's best-LR validation curve (from the cached
``docs/data/*_history_bestlr_curves.parquet`` / ``*_probes.parquet`` files),
fit the tail slope ``dL/d ln(tokens)`` over the last 40% of tokens per seed
(median over seeds), convert the Coupled − Muon final-loss delta into the
token multiple Muon would need to reach Coupled's loss, and net it against
the measured runtime ratio at each arm's own best LR. Reproduces
``docs/FINAL_STATUS.md`` §4.7.

    python scripts/wallclock_accounting.py docs/data/wandb_all_runs_2026-09-02.parquet
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wandb_status_snapshot import best_lr, load  # noqa: E402

DATA = Path(__file__).resolve().parent.parent / "docs" / "data"
CURVE_FILES = [
    "coupled_muon_A0_repro_history_bestlr_curves.parquet",
    "coupled_muon_ladder_history_bestlr_curves.parquet",
    "coupled_muon_moe_pt1_prod_history_part2_bestlr_probes.parquet",
]
CANONICAL = [
    "coupled-muon-A0-repro",
    "coupled-muon-ladder",
    "coupled-muon-moe-pt1-prod",
    "coupled-muon-moe-pt2-screen",
]
NAME_RE = re.compile(r"ladder_(\w+?)·(\w+)·lr([0-9e.\-]+)·s(\d+)")


def _letter(r: str) -> str:
    if r.startswith("H_prime"):
        return "H'"
    if r.startswith("I_prime"):
        return "I'"
    if r.startswith("A0"):
        return "A0"
    return r.split("_")[0]


def load_curves() -> pd.DataFrame:
    frames = [pd.read_parquet(DATA / f) for f in CURVE_FILES]
    c = pd.concat(frames)
    c = c[c["metric"] == "val_loss"].copy()
    parsed = c["name"].map(lambda n: NAME_RE.match(n).groups() if NAME_RE.match(n) else (None,) * 4)
    c["rung"] = parsed.map(lambda t: t[0])
    c["opt"] = parsed.map(lambda t: t[1])
    c["lr"] = parsed.map(lambda t: float(t[2]) if t[2] else np.nan)
    c["seed"] = parsed.map(lambda t: int(t[3]) if t[3] else -1)
    c["L"] = c["rung"].astype(str).map(_letter)
    return c


def main(snapshot: str) -> None:
    _, valid = load(snapshot)
    valid = valid[valid["project"].isin(CANONICAL)]
    curves = load_curves()
    rows = []
    for L in ["A0", "B", "C", "D", "E", "G", "H", "I", "I'"]:
        v = valid[valid["L"] == L]
        bm = best_lr(v[v["opt"] == "muon"])
        bc = best_lr(v[v["opt"] == "coupled_muon_v2"])
        if not bm or not bc:
            continue
        cm = curves[(curves["L"] == L) & (curves["opt"] == "muon") & np.isclose(curves["lr"], bm[0])]
        slopes = []
        for _, g in cm.groupby("seed"):
            g = g.sort_values("tokens")
            tail = g[g["tokens"] >= 0.6 * g["tokens"].max()]
            if len(tail) >= 3:
                slopes.append(np.polyfit(np.log(tail["tokens"]), tail["value"], 1)[0])
        if not slopes:
            continue
        slope = float(np.median(slopes))
        dl = bc[1] - bm[1]
        token_multiple = float(np.exp(dl / slope))
        rt_c = v[(v["opt"] == "coupled_muon_v2") & (v["lr"] == bc[0])]["runtime"].median()
        rt_m = v[(v["opt"] == "muon") & (v["lr"] == bm[0])]["runtime"].median()
        tax = float(rt_c / rt_m)
        rows.append(
            dict(
                rung=L,
                coupled_minus_muon=round(dl, 4),
                muon_tail_slope=round(slope, 3),
                muon_tokens_to_match=round(token_multiple, 3),
                runtime_tax=round(tax, 3),
                net_wallclock=round(token_multiple / tax, 3),
                n_seeds=len(slopes),
            )
        )
    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).to_string(index=False))
    print("net_wallclock > 1 means Coupled reaches Muon's loss in less wall-clock at its own best LR.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(DATA / "wandb_all_runs_2026-09-02.parquet"))
