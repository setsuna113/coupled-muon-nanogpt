"""Rank Phase-2 S1 NS-policy winner from on-disk metrics.jsonl + config.yaml.

Aggregates val_loss across all coupled_steps for `coupled_muon_v2` cells, prints
both the full breakdown and the policy ranking. Top row of the second table is
the winner to plug into S3/A2/B3/S2/A1/B1/stretch YAMLs.

    uv run python scripts/rank_ns_policy.py [--root <results_dir>]
"""

import argparse
import glob
import json
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
        if vl is None:
            continue
        key = (
            cfg["optimizer"]["ns_coefficients"],
            cfg["optimizer"]["type"],
            cfg["optimizer"].get("coupled_steps"),
        )
        d[key].append(vl)

    print("=== full breakdown (policy, opt, K) -> median val_loss ===")
    for k, v in sorted(d.items()):
        print(f"  {k} -> median {statistics.median(v):.4f} (n={len(v)})")

    print()
    print("=== aggregated over K, coupled_muon_v2 only ===")
    agg: dict[str, list[float]] = defaultdict(list)
    for (pol, opt, k), v in d.items():
        if opt == "coupled_muon_v2":
            agg[pol].extend(v)
    if not agg:
        print("  (no coupled_muon_v2 cells found)")
        return
    for pol, v in sorted(agg.items(), key=lambda x: statistics.median(x[1])):
        print(f"  {pol:>15s} -> median {statistics.median(v):.4f} (n={len(v)})")

    winner = min(agg.items(), key=lambda x: statistics.median(x[1]))[0]
    print()
    print(f"WINNER: {winner}")


if __name__ == "__main__":
    main()
