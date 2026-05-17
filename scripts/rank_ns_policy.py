"""Rank Phase-2 S1 NS-policy winner from on-disk metrics.jsonl + config.yaml.

Aggregates val_loss across all coupled_steps for `coupled_muon_v2` cells, prints
both the full breakdown and the policy ranking. Top row of the second table is
the winner to plug into S3/A2/B3/S2/A1/B1/stretch YAMLs.

    uv run python scripts/rank_ns_policy.py [--root <results_dir>]
"""

import argparse
import glob
import json
import math
import statistics
from collections import defaultdict

import yaml


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--root",
        default="/inspire/hdd/global_user/yanjunchi-24040/yancheng/phase2-rigC-ns-policy",
    )
    args = p.parse_args()

    d: dict[tuple, list[float]] = defaultdict(list)
    for cfg_p in sorted(glob.glob(f"{args.root}/*/config.yaml")):
        cfg = yaml.safe_load(open(cfg_p))
        metrics_p = cfg_p.replace("config.yaml", "metrics.jsonl")
        vl = None
        for line in reversed(open(metrics_p).readlines()):
            rec = json.loads(line)
            if "val_loss" in rec:
                vl = rec["val_loss"]
                break
        if vl is None or (isinstance(vl, float) and math.isnan(vl)):
            continue
        key = (
            cfg["optimizer"]["ns_coefficients"],
            cfg["optimizer"]["type"],
            cfg["optimizer"].get("coupled_steps"),
        )
        d[key].append(vl)

    print("=== full breakdown (policy, opt, K) -> median val_loss (diverged count) ===")
    # Count diverged cells per key (those filtered out above).
    diverged: dict[tuple, int] = defaultdict(int)
    for cfg_p in sorted(glob.glob(f"{args.root}/*/config.yaml")):
        cfg = yaml.safe_load(open(cfg_p))
        metrics_p = cfg_p.replace("config.yaml", "metrics.jsonl")
        vl = None
        for line in reversed(open(metrics_p).readlines()):
            rec = json.loads(line)
            if "val_loss" in rec:
                vl = rec["val_loss"]
                break
        if vl is not None and isinstance(vl, float) and math.isnan(vl):
            key = (
                cfg["optimizer"]["ns_coefficients"],
                cfg["optimizer"]["type"],
                cfg["optimizer"].get("coupled_steps"),
            )
            diverged[key] += 1

    for k in sorted(set(d) | set(diverged)):
        v = d.get(k, [])
        nd = diverged.get(k, 0)
        if v:
            print(f"  {k} -> median {statistics.median(v):.4f} (n={len(v)}, diverged={nd})")
        else:
            print(f"  {k} -> ALL DIVERGED (diverged={nd})")

    print()
    print("=== aggregated over K, coupled_muon_v2 only (stable cells only) ===")
    agg: dict[str, list[float]] = defaultdict(list)
    agg_div: dict[str, int] = defaultdict(int)
    for (pol, opt, k), v in d.items():
        if opt == "coupled_muon_v2":
            agg[pol].extend(v)
    for (pol, opt, k), n in diverged.items():
        if opt == "coupled_muon_v2":
            agg_div[pol] += n
    pol_universe = sorted(set(agg) | set(agg_div))
    if not pol_universe:
        print("  (no coupled_muon_v2 cells found)")
        return
    # Rank policies that have ZERO divergences first (by median val-loss),
    # then policies with divergences (also by median, but they're flagged).
    ranked = sorted(
        pol_universe,
        key=lambda p: (
            agg_div[p] > 0,  # stable policies first
            statistics.median(agg[p]) if agg.get(p) else float("inf"),
        ),
    )
    for pol in ranked:
        v = agg.get(pol, [])
        nd = agg_div.get(pol, 0)
        med = f"{statistics.median(v):.4f}" if v else "n/a"
        flag = " ⚠ UNSTABLE" if nd > 0 else ""
        print(f"  {pol:>15s} -> median {med} (n_stable={len(v)}, diverged={nd}){flag}")

    winner = ranked[0]
    print()
    print(f"WINNER: {winner}  (rank: stable-first, then lowest median val_loss)")


if __name__ == "__main__":
    main()
