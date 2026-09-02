#!/usr/bin/env python3
"""Materialize Phase-2.1 winner-controlled jobs from on-disk results.

This script intentionally uses the shared filesystem as the source of truth:
completed cells are directories whose ``stdout.log`` contains ``[saved_ckpt]``;
the final score is the last finite ``val_loss`` in ``metrics.jsonl``; and all
arms are identified from the resolved ``config.yaml`` written by train.py.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import statistics
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

GLOBAL_DEFAULT = Path("/inspire/hdd/global_user/yanjunchi-24040/yancheng")
CONTROL_NAME = "phase2.1-control"

RESULT_DIRS = {
    "no_rope_C": ("phase2.1-rigB-b-qk-no-rope-C", 27),
    "no_rope_D": ("phase2.1-rigB-b-qk-no-rope-D", 27),
    "no_rope_E": ("phase2.1-rigB-b-qk-no-rope-E", 27),
    "partial_Hprime": ("phase2.1-rigB-b-qk-partial-rope-Hprime", 48),
    "muon_Hprime": ("phase2.1-rigB-b-muon-baseline-Hprime", 15),
    "pair_policy": ("phase2.1-rigB-c-pair-policy", 45),
    "d1_mla": ("phase2.1-rigA-d1-mla", 45),
}

BASE_C = "configs/ladder/C_learnedpos.yaml"
BASE_D = "configs/ladder/D_layernorm.yaml"
BASE_HPRIME = "configs/ladder/H_prime_partial_rope.yaml"
BASE_I = "configs/ladder/I_moe.yaml"
BASE_O = "configs/ladder/O_mla.yaml"
BASE_A0 = "configs/ladder/A0_llama60m.yaml"
BASE_P = "configs/ladder/P_factff.yaml"
BASE_Z350_MLA = "configs/ladder/Z_350m_mla.yaml"

RUNG_BASES = {
    "ladder_C_learnedpos": BASE_C,
    "ladder_D_layernorm": BASE_D,
}

NO_ROPE_POLICIES = ("current_flat2d_fallback", "qk_off", "headwise_no_rope")
PARTIAL_ROPE_POLICIES = (
    "current_flat2d_fallback",
    "qk_off",
    "headwise_no_rope",
    "partial_rope_split",
)
TIEBREAK = {
    "qk_off": 0,
    "current_flat2d_fallback": 1,
    "headwise_no_rope": 2,
    "partial_rope_split": 3,
}
PAIR_POLICIES = ("attention_only", "ffn_only", "all")


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    run_dir: Path
    name: str
    optimizer_type: str
    lr: float
    seed: int
    val_loss: float
    no_rope_policy: str | None = None
    partial_rope_policy: str | None = None
    pair_policy: str | None = None


@dataclass(frozen=True)
class BestGroup:
    key: str
    lr: float
    median: float
    values: tuple[float, ...]
    seeds: tuple[int, ...]

    @property
    def seed_std(self) -> float:
        return _sample_std(self.values)


def has_saved_ckpt(stdout_log: Path) -> bool:
    if not stdout_log.exists():
        return False
    try:
        with stdout_log.open(errors="replace") as f:
            return any("[saved_ckpt]" in line for line in f)
    except OSError:
        return False


def final_finite_val_loss(metrics_path: Path) -> float:
    last: float | None = None
    with metrics_path.open(errors="replace") as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "val_loss" not in row:
                continue
            try:
                value = float(row["val_loss"])
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                last = value
    if last is None:
        raise ValueError(f"no finite val_loss in {metrics_path}")
    return last


def _seed_from_run_id(run_id: str) -> int:
    match = re.search(r"-s(\d+)-[0-9a-f]{12}$", run_id)
    if match is None:
        raise ValueError(f"cannot parse seed from run_id={run_id!r}")
    return int(match.group(1))


def _cfg_select(cfg: Any, key: str, default: Any = None) -> Any:
    value = OmegaConf.select(cfg, key, default=default)
    return default if value is None else value


def read_completed_runs(result_dir: Path) -> list[RunRecord]:
    runs: list[RunRecord] = []
    if not result_dir.exists():
        return runs
    for run_dir in sorted(result_dir.iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith("_"):
            continue
        if not has_saved_ckpt(run_dir / "stdout.log"):
            continue
        cfg_path = run_dir / "config.yaml"
        metrics_path = run_dir / "metrics.jsonl"
        if not cfg_path.exists() or not metrics_path.exists():
            continue
        cfg = OmegaConf.load(cfg_path)
        runs.append(
            RunRecord(
                run_id=run_dir.name,
                run_dir=run_dir,
                name=str(_cfg_select(cfg, "name", "")),
                optimizer_type=str(_cfg_select(cfg, "optimizer.type", "")),
                lr=float(_cfg_select(cfg, "optimizer.lr", "nan")),
                seed=_seed_from_run_id(run_dir.name),
                val_loss=final_finite_val_loss(metrics_path),
                no_rope_policy=str(
                    _cfg_select(cfg, "optimizer.qk_coupling.no_rope_policy", "")
                )
                or None,
                partial_rope_policy=str(
                    _cfg_select(cfg, "optimizer.qk_coupling.partial_rope_policy", "")
                )
                or None,
                pair_policy=str(_cfg_select(cfg, "optimizer.pair_policy", "")) or None,
            )
        )
    return runs


def _sample_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(statistics.stdev(values))


def pooled_seed_std(groups: Iterable[Sequence[float]]) -> float:
    numerator = 0.0
    denominator = 0
    for values in groups:
        if len(values) < 2:
            continue
        std = statistics.stdev(values)
        numerator += (len(values) - 1) * std * std
        denominator += len(values) - 1
    if denominator <= 0:
        return 0.0
    return math.sqrt(numerator / denominator)


def _median(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot take median of an empty sequence")
    return float(statistics.median(values))


def _best_groups(
    records: Sequence[RunRecord],
    *,
    key_attr: str,
    candidates: Sequence[str],
) -> dict[str, BestGroup]:
    by_key_lr: dict[tuple[str, float], list[RunRecord]] = {}
    for record in records:
        key = getattr(record, key_attr)
        if key not in candidates:
            continue
        by_key_lr.setdefault((str(key), float(record.lr)), []).append(record)

    out: dict[str, BestGroup] = {}
    for key in candidates:
        groups: list[BestGroup] = []
        for (group_key, lr), group_records in by_key_lr.items():
            if group_key != key:
                continue
            values = tuple(float(r.val_loss) for r in sorted(group_records, key=lambda r: r.seed))
            seeds = tuple(int(r.seed) for r in sorted(group_records, key=lambda r: r.seed))
            groups.append(BestGroup(key=key, lr=lr, median=_median(values), values=values, seeds=seeds))
        if groups:
            out[key] = min(groups, key=lambda g: (g.median, g.lr))
    return out


def _within(candidate: float, reference: float, threshold: float) -> bool:
    return candidate <= reference + max(0.0, threshold)


def decide_no_rope_winner(
    runs_by_rung: dict[str, Sequence[RunRecord]],
) -> dict[str, Any]:
    required = ("ladder_C_learnedpos", "ladder_D_layernorm")
    best_by_rung = {
        rung: _best_groups(records, key_attr="no_rope_policy", candidates=NO_ROPE_POLICIES)
        for rung, records in runs_by_rung.items()
    }
    missing = [
        f"{rung}/{policy}"
        for rung in required
        for policy in NO_ROPE_POLICIES
        if policy not in best_by_rung.get(rung, {})
    ]
    if missing:
        raise ValueError("incomplete no-RoPE readout: missing " + ", ".join(missing))

    scores: dict[str, float] = {}
    values_for_threshold: list[tuple[float, ...]] = []
    for policy in NO_ROPE_POLICIES:
        groups = [best_by_rung[rung][policy] for rung in required]
        scores[policy] = sum(g.median for g in groups) / len(groups)
        values_for_threshold.extend(g.values for g in groups)

    e_notes: list[str] = []
    e_ok = set(NO_ROPE_POLICIES)
    if "ladder_E_karpathy" in best_by_rung and "current_flat2d_fallback" in best_by_rung["ladder_E_karpathy"]:
        e_groups = best_by_rung["ladder_E_karpathy"]
        fallback = e_groups["current_flat2d_fallback"]
        e_ok = set()
        for policy, group in e_groups.items():
            threshold = pooled_seed_std((group.values, fallback.values))
            if _within(group.median, fallback.median, threshold):
                e_ok.add(policy)
            else:
                e_notes.append(
                    f"{policy} excluded on E: median={group.median:.8g} "
                    f"> fallback+std={fallback.median + threshold:.8g}"
                )

    best_score = min(scores.values())
    match_threshold = pooled_seed_std(values_for_threshold)
    matched = {
        policy
        for policy, score in scores.items()
        if _within(score, best_score, match_threshold) and policy in e_ok
    }
    if not matched:
        matched = {policy for policy, score in scores.items() if _within(score, best_score, match_threshold)}
        e_notes.append("no E-safe no-RoPE candidate in score tie set; ignoring E constraint")

    winner = min(matched, key=lambda p: TIEBREAK[p])
    best_lrs_by_base = {
        RUNG_BASES[rung]: best_by_rung[rung][winner].lr
        for rung in required
    }
    return {
        "winner": winner,
        "best_lrs_by_base": best_lrs_by_base,
        "score": scores[winner],
        "match_threshold": match_threshold,
        "matched_candidates": sorted(matched, key=lambda p: TIEBREAK[p]),
        "scores": scores,
        "best_by_rung": _serialise_best_by_rung(best_by_rung),
        "notes": e_notes,
    }


def decide_partial_rope_winner(
    coupled_runs: Sequence[RunRecord],
    muon_runs: Sequence[RunRecord],
) -> dict[str, Any]:
    best = _best_groups(coupled_runs, key_attr="partial_rope_policy", candidates=PARTIAL_ROPE_POLICIES)
    missing = [policy for policy in PARTIAL_ROPE_POLICIES if policy not in best]
    if missing:
        raise ValueError("incomplete partial-RoPE readout: missing " + ", ".join(missing))

    muon_best_by_lr: dict[float, list[RunRecord]] = {}
    for record in muon_runs:
        if record.optimizer_type == "muon":
            muon_best_by_lr.setdefault(float(record.lr), []).append(record)
    if not muon_best_by_lr:
        raise ValueError("missing H' Muon baseline runs")
    muon_groups = [
        BestGroup(
            key="muon",
            lr=lr,
            median=_median([r.val_loss for r in sorted(group, key=lambda r: r.seed)]),
            values=tuple(r.val_loss for r in sorted(group, key=lambda r: r.seed)),
            seeds=tuple(r.seed for r in sorted(group, key=lambda r: r.seed)),
        )
        for lr, group in muon_best_by_lr.items()
    ]
    muon_best = min(muon_groups, key=lambda g: (g.median, g.lr))

    eligible: set[str] = set()
    for policy, group in best.items():
        threshold = pooled_seed_std((group.values, muon_best.values))
        if _within(group.median, muon_best.median, threshold):
            eligible.add(policy)
    if not eligible:
        eligible = set(best)

    best_score = min(best[p].median for p in eligible)
    match_threshold = pooled_seed_std(best[p].values for p in eligible)
    matched = {
        policy
        for policy in eligible
        if _within(best[policy].median, best_score, match_threshold)
    }
    winner = min(matched, key=lambda p: TIEBREAK[p])
    return {
        "winner": winner,
        "best_lr": best[winner].lr,
        "median": best[winner].median,
        "match_threshold": match_threshold,
        "matched_candidates": sorted(matched, key=lambda p: TIEBREAK[p]),
        "muon_baseline": _serialise_best_group(muon_best),
        "best_by_policy": {k: _serialise_best_group(v) for k, v in best.items()},
    }


def decide_pair_policy(runs: Sequence[RunRecord]) -> dict[str, Any]:
    best = _best_groups(runs, key_attr="pair_policy", candidates=PAIR_POLICIES)
    missing = [policy for policy in PAIR_POLICIES if policy not in best]
    if missing:
        raise ValueError("incomplete pair_policy readout: missing " + ", ".join(missing))

    best_score = min(group.median for group in best.values())
    threshold = pooled_seed_std(group.values for group in best.values())
    equivalent = {policy for policy, group in best.items() if _within(group.median, best_score, threshold)}
    worse = {policy for policy, group in best.items() if group.median > best_score + threshold}

    if equivalent == {"all"}:
        winner = "all"
        reason = "all is uniquely best"
    elif equivalent == {"attention_only"}:
        winner = "attention_only"
        reason = "attention_only is uniquely best"
    elif equivalent == {"ffn_only"}:
        winner = "ffn_only"
        reason = "ffn_only is uniquely best"
    elif {"attention_only", "all"}.issubset(equivalent) and "ffn_only" in worse:
        winner = "attention_only"
        reason = "attention_only tied with all; ffn_only worse"
    elif {"ffn_only", "all"}.issubset(equivalent) and "attention_only" in worse:
        winner = "ffn_only"
        reason = "ffn_only tied with all; attention_only worse"
    else:
        winner = "all"
        reason = "inconclusive; defaulting to all"

    return {
        "winner": winner,
        "best_lr": best[winner].lr,
        "median": best[winner].median,
        "equivalent": sorted(equivalent),
        "threshold": threshold,
        "reason": reason,
        "best_by_policy": {k: _serialise_best_group(v) for k, v in best.items()},
    }


def decide_phase21_winners(
    *,
    no_rope_runs_by_rung: dict[str, Sequence[RunRecord]],
    partial_runs: Sequence[RunRecord],
    hprime_muon_runs: Sequence[RunRecord],
    pair_runs: Sequence[RunRecord],
) -> dict[str, Any]:
    no_rope = decide_no_rope_winner(no_rope_runs_by_rung)
    partial = decide_partial_rope_winner(partial_runs, hprime_muon_runs)
    pair = decide_pair_policy(pair_runs)
    return {
        "no_rope_winner": no_rope["winner"],
        "no_rope_best_lrs_by_base": no_rope["best_lrs_by_base"],
        "partial_rope_winner": partial["winner"],
        "partial_rope_best_lr": partial["best_lr"],
        "pair_policy": pair["winner"],
        "pair_policy_best_lr": pair["best_lr"],
        "decisions": {
            "no_rope": no_rope,
            "partial_rope": partial,
            "pair_policy": pair,
        },
    }


def _serialise_best_group(group: BestGroup) -> dict[str, Any]:
    return {
        "lr": group.lr,
        "median": group.median,
        "seed_std": group.seed_std,
        "seeds": list(group.seeds),
        "values": list(group.values),
    }


def _serialise_best_by_rung(
    best_by_rung: dict[str, dict[str, BestGroup]],
) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        rung: {policy: _serialise_best_group(group) for policy, group in groups.items()}
        for rung, groups in best_by_rung.items()
    }


def _job(base: str, seed: int, overrides: dict[str, Any]) -> dict[str, Any]:
    return {"base": base, "overrides": dict(overrides), "seed": int(seed)}


def _winner_overrides(winners: dict[str, Any], *, include_pair_policy: bool = True) -> dict[str, Any]:
    overrides: dict[str, Any] = {
        "optimizer.qk_coupling.full_rope": "legacy_rope2d",
        "optimizer.qk_coupling.no_rope_policy": winners["no_rope_winner"],
        "optimizer.qk_coupling.partial_rope_policy": winners["partial_rope_winner"],
    }
    if include_pair_policy:
        overrides["optimizer.pair_policy"] = winners["pair_policy"]
        overrides["optimizer.allow_pair_policy_override"] = True
    return overrides


def build_a1_jobs() -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for lr in (3.0e-5, 1.0e-4, 3.0e-4, 1.0e-3, 3.0e-3):
        for seed in (3, 4):
            jobs.append(
                _job(
                    BASE_I,
                    seed,
                    {
                        "optimizer.type": "adamw",
                        "optimizer.lr": lr,
                        "run.wandb_project": "coupled-muon-moe-anchor-adamw",
                    },
                )
            )
    return jobs


def build_a3_jobs() -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for coupled_steps in (1, 2, 4):
        for seed in (0, 1, 2):
            jobs.append(
                _job(
                    BASE_I,
                    seed,
                    {
                        "optimizer.type": "coupled_muon_v2",
                        "optimizer.lr": 1.0e-2,
                        "optimizer.coupled_steps": coupled_steps,
                        "optimizer.ns_steps": 5,
                        "optimizer.ns_coefficients": "bernstein",
                        "run.wandb_project": "coupled-muon-k-curve-I",
                        "run.wandb_group": "${name}/phase2.1_a3_I_K${optimizer.coupled_steps}",
                    },
                )
            )
    return jobs


def build_b2_no_rope_jobs(winners: dict[str, Any]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    lr_by_base = winners["no_rope_best_lrs_by_base"]
    for base in (BASE_C, BASE_D):
        if base not in lr_by_base:
            raise ValueError(f"missing no-RoPE best LR for {base}")
        for seed in range(5):
            jobs.append(
                _job(
                    base,
                    seed,
                    {
                        "optimizer.type": "coupled_muon_v2",
                        "optimizer.coupled_steps": 4,
                        "optimizer.ns_steps": 5,
                        "optimizer.ns_coefficients": "bernstein",
                        "optimizer.qk_coupling.full_rope": "legacy_rope2d",
                        "optimizer.qk_coupling.no_rope_policy": winners["no_rope_winner"],
                        "optimizer.qk_coupling.partial_rope_policy": "current_flat2d_fallback",
                        "optimizer.lr": lr_by_base[base],
                        "optimizer.use_multi_head": True,
                        "optimizer.couple_qk": True,
                        "optimizer.couple_vo": True,
                        "optimizer.couple_updown": True,
                        "run.wandb_project": "coupled-muon-qk-policy-confirm",
                        "run.wandb_group": "${name}/phase2.1_b2_no_rope_confirm",
                    },
                )
            )
    return jobs


def build_b2_partial_rope_jobs(winners: dict[str, Any]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for seed in range(5):
        jobs.append(
            _job(
                BASE_HPRIME,
                seed,
                {
                    "optimizer.type": "coupled_muon_v2",
                    "optimizer.coupled_steps": 4,
                    "optimizer.ns_steps": 5,
                    "optimizer.ns_coefficients": "bernstein",
                    "optimizer.qk_coupling.full_rope": "legacy_rope2d",
                    "optimizer.qk_coupling.no_rope_policy": winners["no_rope_winner"],
                    "optimizer.qk_coupling.partial_rope_policy": winners["partial_rope_winner"],
                    "optimizer.lr": winners["partial_rope_best_lr"],
                    "optimizer.use_multi_head": True,
                    "optimizer.couple_qk": True,
                    "optimizer.couple_vo": True,
                    "optimizer.couple_updown": True,
                    "run.wandb_project": "coupled-muon-qk-policy-confirm",
                    "run.wandb_group": "${name}/phase2.1_b2_partial_rope_confirm",
                },
            )
        )
    return jobs


def build_d1_jobs(winners: dict[str, Any]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    fixed = {
        "optimizer.ns_coefficients": "bernstein",
        "run.wandb_project": "coupled-muon-mla",
        **_winner_overrides(winners),
    }
    for optimizer_type in ("muon", "coupled_muon_v2", "adamw"):
        for lr in (3.0e-3, 1.0e-2, 3.0e-2):
            for seed in range(5):
                jobs.append(_job(BASE_O, seed, {**fixed, "optimizer.type": optimizer_type, "optimizer.lr": lr}))
    return jobs


def build_d3_jobs(winners: dict[str, Any]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    fixed = {
        "optimizer.lr": 1.0e-2,
        "optimizer.coupled_steps": 4,
        "optimizer.ns_steps": 5,
        "optimizer.ns_coefficients": "bernstein",
        "optimizer.couple_mla": True,
        "model.attn.q_lora_rank": 64,
        "run.wandb_project": "coupled-muon-lora-rank-paired",
        "run.wandb_group": "${name}/phase2.1_d3_kv${model.attn.kv_lora_rank}_${optimizer.type}",
        **_winner_overrides(winners),
    }
    for optimizer_type in ("muon", "coupled_muon_v2"):
        for kv_rank in (16, 32, 64, 128, 256):
            for seed in range(3):
                jobs.append(
                    _job(
                        BASE_O,
                        seed,
                        {**fixed, "optimizer.type": optimizer_type, "model.attn.kv_lora_rank": kv_rank},
                    )
                )
    return jobs


def build_d4_jobs(winners: dict[str, Any]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    fixed = {
        "optimizer.lr": 3.0e-3,
        "optimizer.ns_coefficients": "bernstein",
        "run.wandb_project": "coupled-muon-imposed-factff",
        **_winner_overrides(winners),
    }
    for base in (BASE_A0, BASE_P):
        for optimizer_type in ("muon", "coupled_muon_v2"):
            for seed in range(3):
                jobs.append(_job(base, seed, {**fixed, "optimizer.type": optimizer_type}))
    return jobs


def build_e1_jobs(winners: dict[str, Any]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    fixed = {
        "optimizer.ns_coefficients": "bernstein",
        "run.wandb_project": "coupled-muon-mla-350m",
        **_winner_overrides(winners),
    }
    for optimizer_type in ("muon", "coupled_muon_v2"):
        for lr in (3.0e-3, 1.0e-2, 3.0e-2):
            for seed in range(5):
                jobs.append(_job(BASE_Z350_MLA, seed, {**fixed, "optimizer.type": optimizer_type, "optimizer.lr": lr}))
    return jobs


def build_dynamic_jobs(winners: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    return {
        "p21_b2_no_rope.jsonl": build_b2_no_rope_jobs(winners),
        "p21_b2_partial_rope.jsonl": build_b2_partial_rope_jobs(winners),
        "p21_d1_mla.jsonl": build_d1_jobs(winners),
        "p21_d3_lora_rank.jsonl": build_d3_jobs(winners),
        "p21_d4_factff.jsonl": build_d4_jobs(winners),
        "p21_e1_mla_350m.jsonl": build_e1_jobs(winners),
    }


def build_static_jobs() -> dict[str, list[dict[str, Any]]]:
    return {
        "p21_a1_adamw_I.jsonl": build_a1_jobs(),
        "p21_a3_kcurve_I.jsonl": build_a3_jobs(),
    }


def shard_jobs(jobs: Sequence[dict[str, Any]], residues: Sequence[int], *, modulus: int = 7) -> list[dict[str, Any]]:
    residue_set = {int(r) for r in residues}
    return [job for i, job in enumerate(jobs) if i % modulus in residue_set]


def write_jsonl(path: Path, jobs: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for job in jobs:
            f.write(json.dumps(job, sort_keys=True) + "\n")


def _write_env(path: Path, winners: dict[str, Any]) -> None:
    lr_by_base = winners["no_rope_best_lrs_by_base"]
    lines = [
        "# Generated by scripts/phase21_materialize.py",
        f"NO_ROPE_WINNER={shlex.quote(str(winners['no_rope_winner']))}",
        f"NO_ROPE_LR_C={shlex.quote(str(lr_by_base[BASE_C]))}",
        f"NO_ROPE_LR_D={shlex.quote(str(lr_by_base[BASE_D]))}",
        f"PARTIAL_ROPE_WINNER={shlex.quote(str(winners['partial_rope_winner']))}",
        f"PARTIAL_ROPE_LR_HPRIME={shlex.quote(str(winners['partial_rope_best_lr']))}",
        f"PAIR_POLICY={shlex.quote(str(winners['pair_policy']))}",
        f"PAIR_POLICY_BEST_LR={shlex.quote(str(winners['pair_policy_best_lr']))}",
        "",
    ]
    path.write_text("\n".join(lines))


def _load_expected(global_root: Path, key: str) -> list[RunRecord]:
    dirname, expected = RESULT_DIRS[key]
    result_dir = global_root / dirname
    runs = read_completed_runs(result_dir)
    if len(runs) != expected:
        raise SystemExit(f"{key}: expected {expected} completed runs under {result_dir}, found {len(runs)}")
    return runs


def materialize(global_root: Path, control_root: Path) -> dict[str, Any]:
    no_c = _load_expected(global_root, "no_rope_C")
    no_d = _load_expected(global_root, "no_rope_D")
    no_e = _load_expected(global_root, "no_rope_E")
    partial = _load_expected(global_root, "partial_Hprime")
    hprime_muon = _load_expected(global_root, "muon_Hprime")
    pair = _load_expected(global_root, "pair_policy")

    winners = decide_phase21_winners(
        no_rope_runs_by_rung={
            "ladder_C_learnedpos": no_c,
            "ladder_D_layernorm": no_d,
            "ladder_E_karpathy": no_e,
        },
        partial_runs=partial,
        hprime_muon_runs=hprime_muon,
        pair_runs=pair,
    )

    control_root.mkdir(parents=True, exist_ok=True)
    jobs_dir = control_root / "jobs"
    generated = {**build_static_jobs(), **build_dynamic_jobs(winners)}
    for filename, jobs in generated.items():
        write_jsonl(jobs_dir / filename, jobs)

    counts = {filename: len(jobs) for filename, jobs in generated.items()}
    payload = {
        "generated_at_unix": time.time(),
        "global_root": str(global_root),
        "control_root": str(control_root),
        **winners,
        "job_counts": counts,
    }
    (control_root / "p21_winners.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    _write_env(control_root / "p21_winners.env", winners)
    return payload


def materialize_static(control_root: Path) -> dict[str, int]:
    jobs_dir = control_root / "jobs"
    counts: dict[str, int] = {}
    for filename, jobs in build_static_jobs().items():
        write_jsonl(jobs_dir / filename, jobs)
        counts[filename] = len(jobs)
    return counts


def _best_optimizer_group(records: Sequence[RunRecord], optimizer_type: str) -> BestGroup:
    by_lr: dict[float, list[RunRecord]] = {}
    for record in records:
        if record.optimizer_type == optimizer_type:
            by_lr.setdefault(record.lr, []).append(record)
    if not by_lr:
        raise ValueError(f"missing optimizer.type={optimizer_type}")
    groups = [
        BestGroup(
            key=optimizer_type,
            lr=lr,
            median=_median([r.val_loss for r in sorted(group, key=lambda r: r.seed)]),
            values=tuple(r.val_loss for r in sorted(group, key=lambda r: r.seed)),
            seeds=tuple(r.seed for r in sorted(group, key=lambda r: r.seed)),
        )
        for lr, group in by_lr.items()
    ]
    return min(groups, key=lambda g: (g.median, g.lr))


def _bootstrap_paired_delta(
    muon_by_seed: dict[int, float],
    coupled_by_seed: dict[int, float],
    *,
    samples: int = 2000,
) -> dict[str, float]:
    import random

    seeds = sorted(set(muon_by_seed) & set(coupled_by_seed))
    if not seeds:
        raise ValueError("no paired seeds for D-gate bootstrap")
    diffs = [muon_by_seed[s] - coupled_by_seed[s] for s in seeds]
    rng = random.Random(20260525)
    medians: list[float] = []
    for _ in range(samples):
        draw = [diffs[rng.randrange(len(diffs))] for _ in diffs]
        medians.append(float(statistics.median(draw)))
    medians.sort()

    def pct(q: float) -> float:
        idx = min(len(medians) - 1, max(0, int(round(q * (len(medians) - 1)))))
        return medians[idx]

    return {"p05": pct(0.05), "p50": pct(0.50), "p95": pct(0.95)}


def decide_d_gate(d1_runs: Sequence[RunRecord]) -> dict[str, Any]:
    o_runs = [r for r in d1_runs if r.name == "ladder_O_mla" and r.optimizer_type in {"muon", "coupled_muon_v2"}]
    muon = _best_optimizer_group(o_runs, "muon")
    coupled = _best_optimizer_group(o_runs, "coupled_muon_v2")
    muon_by_seed = {r.seed: r.val_loss for r in o_runs if r.optimizer_type == "muon" and math.isclose(r.lr, muon.lr)}
    coupled_by_seed = {
        r.seed: r.val_loss
        for r in o_runs
        if r.optimizer_type == "coupled_muon_v2" and math.isclose(r.lr, coupled.lr)
    }
    delta_p50 = muon.median - coupled.median
    pooled = pooled_seed_std((muon.values, coupled.values))
    bootstrap = _bootstrap_paired_delta(muon_by_seed, coupled_by_seed)
    direction_stable = bootstrap["p05"] > 0 and bootstrap["p50"] > 0 and bootstrap["p95"] > 0
    positive = delta_p50 > 1.5 * pooled and direction_stable
    return {
        "positive": positive,
        "delta_p50": delta_p50,
        "pooled_seed_std": pooled,
        "threshold": 1.5 * pooled,
        "direction_stable": direction_stable,
        "bootstrap": bootstrap,
        "muon": _serialise_best_group(muon),
        "coupled_muon_v2": _serialise_best_group(coupled),
    }


def run_d_gate(global_root: Path, control_root: Path, d1_dir: Path | None = None) -> dict[str, Any]:
    result_dir = d1_dir or (global_root / RESULT_DIRS["d1_mla"][0])
    runs = read_completed_runs(result_dir)
    expected = RESULT_DIRS["d1_mla"][1]
    if len(runs) != expected:
        raise SystemExit(f"D1 gate needs {expected} completed runs under {result_dir}, found {len(runs)}")
    gate = decide_d_gate(runs)
    payload = {
        "generated_at_unix": time.time(),
        "d1_dir": str(result_dir),
        **gate,
    }
    control_root.mkdir(parents=True, exist_ok=True)
    (control_root / "p21_d_gate.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("full", "static", "d-gate"), default="full")
    parser.add_argument("--global-root", type=Path, default=GLOBAL_DEFAULT)
    parser.add_argument("--control-root", type=Path, default=None)
    parser.add_argument("--d1-dir", type=Path, default=None)
    args = parser.parse_args()

    control_root = args.control_root or (args.global_root / CONTROL_NAME)
    if args.mode == "static":
        counts = materialize_static(control_root)
        print(json.dumps({"control_root": str(control_root), "job_counts": counts}, sort_keys=True))
    elif args.mode == "d-gate":
        gate = run_d_gate(args.global_root, control_root, args.d1_dir)
        print(json.dumps(gate, sort_keys=True))
    else:
        payload = materialize(args.global_root, control_root)
        print(json.dumps({"control_root": str(control_root), "job_counts": payload["job_counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
