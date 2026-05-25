from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import phase21_materialize as p21  # noqa: E402


_COUNTER = 0


def _run(
    tmp_path: Path,
    *,
    name: str,
    optimizer_type: str,
    lr: float,
    seed: int,
    val_loss: float,
    no_rope_policy: str | None = None,
    partial_rope_policy: str | None = None,
    pair_policy: str | None = None,
) -> p21.RunRecord:
    global _COUNTER
    run_id = f"{name}-s{seed}-{_COUNTER:012x}"
    _COUNTER += 1
    run_dir = tmp_path / run_id
    run_dir.mkdir(parents=True)
    optimizer_lines = [
        "optimizer:",
        f"  type: {optimizer_type}",
        f"  lr: {lr}",
    ]
    if no_rope_policy or partial_rope_policy:
        optimizer_lines.extend(
            [
                "  qk_coupling:",
                f"    no_rope_policy: {no_rope_policy or 'current_flat2d_fallback'}",
                f"    partial_rope_policy: {partial_rope_policy or 'current_flat2d_fallback'}",
            ]
        )
    if pair_policy is not None:
        optimizer_lines.append(f"  pair_policy: {pair_policy}")
    (run_dir / "config.yaml").write_text("\n".join([f"name: {name}", *optimizer_lines, ""]))
    (run_dir / "metrics.jsonl").write_text(
        json.dumps({"event": "val", "val_loss": val_loss + 1.0}) + "\n"
        + json.dumps({"event": "val", "val_loss": val_loss}) + "\n"
    )
    (run_dir / "stdout.log").write_text(f"finished [saved_ckpt] {run_id}\n")
    return next(record for record in p21.read_completed_runs(tmp_path) if record.run_id == run_id)


def _records_for_policy_grid(
    tmp_path: Path,
    *,
    name: str,
    attr: str,
    values: dict[str, dict[float, list[float]]],
) -> list[p21.RunRecord]:
    out: list[p21.RunRecord] = []
    for policy, lr_map in values.items():
        for lr, losses in lr_map.items():
            for seed, loss in enumerate(losses):
                kwargs = {
                    "name": name,
                    "optimizer_type": "coupled_muon_v2",
                    "lr": lr,
                    "seed": seed,
                    "val_loss": loss,
                }
                if attr == "no_rope_policy":
                    kwargs["no_rope_policy"] = policy
                elif attr == "partial_rope_policy":
                    kwargs["partial_rope_policy"] = policy
                elif attr == "pair_policy":
                    kwargs["pair_policy"] = policy
                else:
                    raise AssertionError(attr)
                out.append(_run(tmp_path, **kwargs))
    return out


def test_completion_detection_reads_only_saved_runs(tmp_path: Path) -> None:
    _run(
        tmp_path,
        name="ladder_C_learnedpos",
        optimizer_type="coupled_muon_v2",
        lr=0.003,
        seed=0,
        val_loss=1.23,
        no_rope_policy="qk_off",
    )
    unfinished = tmp_path / "ladder_C_learnedpos-s1-aaaaaaaaaaaa"
    unfinished.mkdir()
    (unfinished / "stdout.log").write_text("no checkpoint yet\n")
    (unfinished / "config.yaml").write_text("name: ladder_C_learnedpos\noptimizer:\n  type: coupled_muon_v2\n  lr: 0.003\n")
    (unfinished / "metrics.jsonl").write_text(json.dumps({"event": "val", "val_loss": 9.0}) + "\n")

    runs = p21.read_completed_runs(tmp_path)
    assert len(runs) == 1
    assert runs[0].val_loss == 1.23


def test_winners_and_b2_no_rope_use_per_base_lr(tmp_path: Path) -> None:
    no_rope_runs = {
        "ladder_C_learnedpos": _records_for_policy_grid(
            tmp_path / "c",
            name="ladder_C_learnedpos",
            attr="no_rope_policy",
            values={
                "qk_off": {0.003: [1.00, 1.01, 0.99]},
                "current_flat2d_fallback": {0.001: [1.08, 1.09, 1.07]},
                "headwise_no_rope": {0.01: [1.10, 1.11, 1.09]},
            },
        ),
        "ladder_D_layernorm": _records_for_policy_grid(
            tmp_path / "d",
            name="ladder_D_layernorm",
            attr="no_rope_policy",
            values={
                "qk_off": {0.01: [1.20, 1.21, 1.19]},
                "current_flat2d_fallback": {0.003: [1.28, 1.29, 1.27]},
                "headwise_no_rope": {0.001: [1.30, 1.31, 1.29]},
            },
        ),
        "ladder_E_karpathy": _records_for_policy_grid(
            tmp_path / "e",
            name="ladder_E_karpathy",
            attr="no_rope_policy",
            values={
                "qk_off": {0.003: [1.40, 1.40, 1.40]},
                "current_flat2d_fallback": {0.003: [1.40, 1.40, 1.40]},
                "headwise_no_rope": {0.003: [1.41, 1.41, 1.41]},
            },
        ),
    }
    partial = _records_for_policy_grid(
        tmp_path / "partial",
        name="ladder_H_prime_partial_rope",
        attr="partial_rope_policy",
        values={
            "current_flat2d_fallback": {0.003: [1.20, 1.20, 1.20]},
            "qk_off": {0.003: [1.15, 1.15, 1.15]},
            "headwise_no_rope": {0.003: [1.10, 1.10, 1.10]},
            "partial_rope_split": {0.01: [1.00, 1.00, 1.00]},
        },
    )
    hprime_muon = [
        _run(
            tmp_path / "hmuon",
            name="ladder_H_prime_partial_rope",
            optimizer_type="muon",
            lr=0.01,
            seed=seed,
            val_loss=1.05,
        )
        for seed in range(5)
    ]
    pair = _records_for_policy_grid(
        tmp_path / "pair",
        name="ladder_I_moe",
        attr="pair_policy",
        values={
            "attention_only": {0.003: [1.0, 1.0, 1.0, 1.0, 1.0]},
            "ffn_only": {0.003: [1.0, 1.0, 1.0, 1.0, 1.0]},
            "all": {0.003: [1.0, 1.0, 1.0, 1.0, 1.0]},
        },
    )

    winners = p21.decide_phase21_winners(
        no_rope_runs_by_rung=no_rope_runs,
        partial_runs=partial,
        hprime_muon_runs=hprime_muon,
        pair_runs=pair,
    )

    assert winners["no_rope_winner"] == "qk_off"
    assert winners["partial_rope_winner"] == "partial_rope_split"
    assert winners["pair_policy"] == "all"

    b2 = p21.build_b2_no_rope_jobs(winners)
    c_lrs = {job["overrides"]["optimizer.lr"] for job in b2 if job["base"] == p21.BASE_C}
    d_lrs = {job["overrides"]["optimizer.lr"] for job in b2 if job["base"] == p21.BASE_D}
    assert c_lrs == {0.003}
    assert d_lrs == {0.01}


def test_pair_policy_inconclusive_defaults_to_all(tmp_path: Path) -> None:
    pair = _records_for_policy_grid(
        tmp_path,
        name="ladder_I_moe",
        attr="pair_policy",
        values={
            "attention_only": {0.003: [1.0, 1.0, 1.0, 1.0, 1.0]},
            "ffn_only": {0.003: [1.0, 1.0, 1.0, 1.0, 1.0]},
            "all": {0.003: [1.0, 1.0, 1.0, 1.0, 1.0]},
        },
    )
    assert p21.decide_pair_policy(pair)["winner"] == "all"


def _d1_records(tmp_path: Path, muon: list[float], coupled: list[float]) -> list[p21.RunRecord]:
    records: list[p21.RunRecord] = []
    for seed, loss in enumerate(muon):
        records.append(
            _run(
                tmp_path / "d1",
                name="ladder_O_mla",
                optimizer_type="muon",
                lr=0.01,
                seed=seed,
                val_loss=loss,
            )
        )
    for seed, loss in enumerate(coupled):
        records.append(
            _run(
                tmp_path / "d1",
                name="ladder_O_mla",
                optimizer_type="coupled_muon_v2",
                lr=0.01,
                seed=seed,
                val_loss=loss,
            )
        )
    return records


def test_d_gate_positive_and_negative(tmp_path: Path) -> None:
    positive = p21.decide_d_gate(
        _d1_records(tmp_path / "positive", [1.10, 1.11, 1.09, 1.10, 1.12], [1.00, 1.01, 0.99, 1.00, 1.02])
    )
    assert positive["positive"] is True

    negative = p21.decide_d_gate(
        _d1_records(tmp_path / "negative", [1.00, 1.01, 0.99, 1.00, 1.02], [1.00, 1.00, 1.00, 1.00, 1.00])
    )
    assert negative["positive"] is False


def test_d1_shards_cover_all_jobs_without_overlap() -> None:
    winners = {
        "no_rope_winner": "qk_off",
        "partial_rope_winner": "partial_rope_split",
        "pair_policy": "all",
    }
    jobs = p21.build_d1_jobs(winners)
    assert len(jobs) == 45

    seen: set[int] = set()
    for residues in ((0, 1, 2, 3), (4,), (5, 6)):
        shard_indices = {i for i, _ in enumerate(jobs) if i % 7 in residues}
        assert seen.isdisjoint(shard_indices)
        seen.update(shard_indices)
        assert p21.shard_jobs(jobs, residues) == [jobs[i] for i in sorted(shard_indices)]
    assert seen == set(range(45))
