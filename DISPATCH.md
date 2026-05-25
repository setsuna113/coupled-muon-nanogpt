# Coupled-Muon Stage 3 Dispatch Manifest

Operational record of which experiment runs on which rig and where the outputs
land. See `experiment.md` for the design rationale; this file is the lookup
table for "where do I find result X?".

**Headline goal**: deliver the cross-optimizer triple `(AdamW, Muon, Coupled
Muon)` at MoE rung I + its gate-omission control I' within ~7.5 days of the
T=0 Rig B launch. Stage 3 cell budget cut from the full 350 (in
`configs/sweeps/lr_grid_moe.yaml`) down to **180 cells** across three rigs:
60 production + 90 screening + 15+15 cross-optimizer anchors.

---

## Active dispatch (Stage 3)

| Rig | Job | Sweep YAML | Cells | Results dir (`$GLOBAL = /inspire/hdd/global_user/yanjunchi-24040/yancheng`) | Wandb project | ETA |
|---|---|---|---|---|---|---|
| **Rig A — 8×H100** | pt1A (AdamW anchor @ I) + pt2 (J/K/L/M/N screening), single 105-cell queue | `lr_grid_moe_adamw_I.yaml` (15) **then** `lr_grid_moe_pt2_screening.yaml` (90), `cat`d in that order into `rigA.jsonl` | 105 | `$GLOBAL/coupled-muon-stage3-rigA-anchor-screen/` | `coupled-muon-moe-anchor-adamw` (first 15), `coupled-muon-moe-pt2-screen` (next 90) | T≈2.5d → T≈7.48d |
| **Rig B — 4×H200** | pt1B (production tier I, I') ordered I-first | `lr_grid_moe_pt1_production.yaml` | 60 | `$GLOBAL/coupled-muon-stage3-rigB-pt1/` | `coupled-muon-moe-pt1-prod` | T=0 → T≈5.7d |
| **Rig B — 4×H200** | pt3 (adaptive: telescope edges or bonus seeds on production tier) | `lr_grid_moe_pt3_telescope.yaml` *(generated at T≈5.7d from pt1B results)* | ≤19 | `$GLOBAL/coupled-muon-stage3-rigB-pt3/` | `coupled-muon-moe-pt3-telescope` | T≈5.7d → T≈7.5d |
| **2×H200 (new)** | AdamW anchor @ rung I' (cross-optimizer symmetry for the I-vs-I' gate-omission test) | `lr_grid_moe_adamw_I_prime.yaml` | 15 | `$GLOBAL/coupled-muon-stage3-rig2H200-adamw-Iprime/` | `coupled-muon-moe-anchor-adamw-Iprime` | T=new-rig-launch → +2.84d |
| **2×H200 (existing)** | §d.4 ablations (coupled-steps × final_polish × pair × NS-budget) — **unchanged scope** | `configs/sweeps/{coupled_steps_ablation, final_polish_ablation, pair_ablation, coupled_x_polish, muon_ns_budget}.yaml` | 105 (28 done) | `$GLOBAL/coupled-muon-d4ablations/` | per-ablation projects | ~3.2d remaining |

## Cross-optimizer triple delivery timeline (relative to Rig B launch at T=0)

| Milestone | T (days) | Source |
|---|---|---|
| I-Muon (5 seeds × 3 LRs) ready | 1.42 | Rig B pt1B, first 15 cells |
| I-CoupledMuon (5 seeds × 3 LRs) ready | 2.84 | Rig B pt1B, cells 16–30 |
| I-AdamW (3 seeds × 5 LRs) ready | 3.26 | Rig A pt1A, all 15 cells |
| **Full I-triple analysable** | **3.26** | **Headline production claim** |
| I'-Muon ready | 4.26 | Rig B pt1B, cells 31–45 |
| I'-AdamW ready | new-rig-launch + 2.84 | 2×H200 (new) |
| I'-CoupledMuon ready | 5.68 | Rig B pt1B, cells 46–60 |
| **Full I'-triple analysable** | max(5.68, new-rig-launch+2.84) | **Gate-omission control** |
| Screening tier (J/K/L/M/N) ready | 7.48 | Rig A pt2 |
| Telescope/bonus seeds ready | 7.5 | Rig B pt3 |

---

## Finished / archived

| Stage | Rig | Results dir | Note |
|---|---|---|---|
| Stage 1 (A0 LLaMA-60M, 75 cells) | 2×H200 (original) | `$PROJECT_QUANTUM/ladder_A0_llama60m-*` (project quota — check_progress.sh default) | reproduction step |
| Stage 2 (dense ladder B/C/D/E/G/H, 270 cells) | 8×H100 (Rig A, pre-Stage-3) | `$GLOBAL/coupled-muon-stage2-dense-archive/` (offloaded via `scripts/archive_finished.sh` from `$PROJECT_AML/coupled-muon-nanogpt/results/` — symlinks remain at the source) | dense MLP-arity tests |

`$PROJECT_QUANTUM = /inspire/hdd/project/quantum-artificial-intelligence/yanjunchi-24040/yancheng/coupled-muon-nanogpt/results`
`$PROJECT_AML = /inspire/hdd/project/advanced-machine-learning/yanjunchi-24040/yancheng`

---

## Per-rig pre-launch hygiene (run once before each launch)

1. `git pull` to pick up `da04283` (NCCL probe-all-ranks fix) and `e9f9882` (loader skip-undersized-shard).
2. Make sure `--results-dir` is set to the `$GLOBAL` path in the table above. Leaving it unset defaults to the project quota path (Stage 2 was caught by this; see `scripts/archive_finished.sh`).
3. Use `WANDB_MODE=offline` if the rig has no outbound network — base.yaml line 13 honors the env var (null in cfg ⇒ env wins ⇒ default "online" if env unset).

## Resume after interrupt

`scripts/filter_unfinished_jobs.py` recomputes `run_id = hash(resolved_cfg + seed)`; it MUST be called with the same `--results-dir` and `--extra-override` that the original dispatch used or every cell will look "missing". Wrap the launch in a tiny shell script per rig if doing this repeatedly.

## Monitoring

`scripts/check_progress.sh` (one-line per rig) or `scripts/check_progress.sh -v` (also shows the 2 most-recent cells per rig). `RIGS` array is committed; edit there to add new dispatch dirs.

## Where results live (single-source-of-truth, for analysis)

- Per-cell `metrics.jsonl` — canonical numeric record (loss, val_loss, probe outputs). Read this for analysis; `wandb sync` is a nice-to-have, not the source of truth.
- Per-cell `stdout.log` — `[saved_ckpt]` substring marks completion (used by `archive_finished.sh` and `check_progress.sh`).
- Per-cell `wandb/offline-run-*/` — for upload to wandb cloud via `wandb sync <run_dir>` (requires the run to have started in offline mode; an online-mode failed init won't produce an offline-run dir).

---

## Active dispatch (Phase 2.1)

Phase 2.1 replaces the flat Phase-2 schedule with a gated tree (A/B/C/D/E).
Phase B (qk_coupling repair at non-RoPE / partial-RoPE rungs) is highest
priority — it fixes a known algorithm failure (factory's silent flat-2D
fallback under non-full RoPE) and produces the v2.1 (`no_rope_winner`,
`partial_rope_winner`) tuple required for Phase D. Phase C re-runs MoE
pair_policy localization with own-best-LR per policy. Phase D launches O_mla
only after both winners are declared; Phase E is gated on Phase D positive.

See `experiment.md §d.7` for the design rationale, decision rules, and
semantic table. Items removed from active plan (LR-prefactor, unconditional
FactFF/350M/LoRA-rank, full K-curve, L MoE rung, A1' AdamW@I' top-up) stay
on disk with `# status: ...` comment-headers; do NOT relaunch them.

### Phase 2.1 - live progress

Operational state log. Current launch mode is a three-machine chained dispatch
coordinated by `$GLOBAL/phase2.1-control/`. The physical machines run separate
roles (`h100`, `h200x2`, `h200x4`) but share one control directory and one set
of result directories. The run_id hash is cfg+seed, so D1 can be split across
machines as long as every shard uses the same materialized JSONL and
`--results-dir`.

Control artifacts:
- `$GLOBAL/phase2.1-control/p21_winners.{json,env}`: on-disk B/C readout.
- `$GLOBAL/phase2.1-control/jobs/p21_*.jsonl`: generated launch JSONLs.
- `$GLOBAL/phase2.1-control/done/*.done`: cross-machine gates.
- `$GLOBAL/phase2.1-control/p21_d_gate.json`: D1 readout and conditional gate.

Generate the three pasteable one-line commands from the repo root with:
`uv run python scripts/phase21_make_launchers.py --out /tmp/phase21_launch_commands.txt`

| # | Sweep | Phase | Cells | Physical rig | Results dir (`$GLOBAL/…`) | State | Wall-clock | Wandb |
|---|---|---|---|---|---|---|---|---|
| 1 | `phaseA3_k_curve_minimal_A0` | A3 | 9 | 2×H200 | `phase2.1-rigC-a3-kcurve-a0/` | ✅ done 9/9, synced | 4.17 h | `coupled-muon-k-curve` (synced) |
| 2 | `phaseB_qk_no_rope_C` | B1 | 27 | 2×H200 | `phase2.1-rigB-b-qk-no-rope-C/` | ✅ done 27/27 | 20.20 h | `coupled-muon-qk-policy` (offline; sync optional) |
| 3 | `phaseB_qk_no_rope_D` | B1 | 27 | 8×H100 | `phase2.1-rigB-b-qk-no-rope-D/` | ✅ done 27/27 | ~5.3 h | `coupled-muon-qk-policy` (offline; sync optional) |
| 4 | `phaseB_qk_no_rope_E` | B1 | 27 | 8×H100 | `phase2.1-rigB-b-qk-no-rope-E/` | ✅ done 27/27 | 5.29 h | `coupled-muon-qk-policy` (offline; sync optional) |
| 5 | `phaseC_pair_policy_lr` | C1 | 45 | 8×H100 | `phase2.1-rigB-c-pair-policy/` | ✅ done 45/45, on-disk readout for B2/D1 | ~24 h | `coupled-muon-pair-policy-lr` (offline; sync optional) |
| 6 | `phaseB_qk_partial_rope_Hprime` | B1 | 48 | 2×H200 | `phase2.1-rigB-b-qk-partial-rope-Hprime/` | ✅ done 48/48, on-disk readout for B2 | ~36 h | `coupled-muon-qk-policy` (offline; sync optional) |
| 7 | `phaseB_muon_baseline_Hprime` + top-up | B1 | 15 | 2×H200 | `phase2.1-rigB-b-muon-baseline-Hprime/` | ✅ done 15/15, on-disk readout for B2 | >10 h | `coupled-muon-qk-policy` (offline; sync optional) |
| 8 | `phaseB2_confirm_no_rope` | B2 | 10 | 8×H100 | `phase2.1-rigB-b2-no-rope-confirm/` | launch via `h100` role after materializing winners | ~3.3 h | `coupled-muon-qk-policy-confirm` |
| 9 | `phaseB2_confirm_partial_rope` | B2 | 5 | 2×H200 | `phase2.1-rigB-b2-partial-rope-confirm/` | launch via `h200x2` role after materializing winners | ~6.5 h | `coupled-muon-qk-policy-confirm` |
| 10 | `adamw_equalization_I` | A1 | 10 | 4×H200 | `phase2.1-rigA-a1-adamw-I-topup/` | launch via `h200x4` role, then A3-I | ~1.25 rig-days est. | `coupled-muon-moe-anchor-adamw` (offline) |
| 11 | `phaseA3_k_curve_minimal_I` | A3 | 9 | 4×H200 | `phase2.1-rigA-a3-kcurve-I/` | chained after #10 in `h200x4` role | ~1.46 rig-days est. | `coupled-muon-k-curve-I` (offline) |
| 12 | `mla_lr_grid` | D1 | 45 | all three | `phase2.1-rigA-d1-mla/` | gated on B2 + A3-I; modulo-7 shards | ~6.6 rig-days est. | `coupled-muon-mla` |
| 13 | `phaseD3_lora_rank_paired_conditional` | D3 | 30 | 8×H100 | `phase2.1-rigB-d3-lora-rank/` | conditional on D-gate positive | ~1.22 rig-days est. | `coupled-muon-lora-rank-paired` |
| 14 | `imposed_factff` | D4 | 12 | 2×H200 | `phase2.1-rigC-d4-factff/` | conditional on D-gate positive | ~0.65 rig-days est. | `coupled-muon-imposed-factff` |
| 15 | `mla_scaling_350m` | E1 | 30 | 4×H200 | `phase2.1-rigA-e1-mla-350m/` | conditional on D-gate positive | ~16.9 rig-days est. | `coupled-muon-mla-350m` |

Measured anchors (supersede the conservative plan estimates):
- A0 dense 60M-CS, 1.2B tokens: **0.46 h/cell** on a 2-GPU slice (4.17 h / 9).
- C/D/E/H' dense 60M-CS, 2.5B tokens: **~0.75 h/cell** on a 2-GPU slice
  (C: 20.20 h / 27 = 0.748). A 2-GPU slice of the 8×H100 ≈ a dedicated
  2×H200 for this workload (E: 5.29 h, 27 cells, 4-way ⇒ 0.78 h/cell-slice).
- I MoE 60M-CS backbone, 5B tokens: **~2 h/cell** est. on a 2-GPU slice
  (no direct measurement yet; token-scaled from C × MoE overhead).

**8×H100 grad_accum note**: 4-way-parallel mode keeps each cell at
world_size=2 (2 GPU/cell), so the ladder YAML's `grad_accum_steps` is
correct as-is — do NOT quarter it. Quartering only applies to a single cell
spanning all 8 GPUs (world_size=8), which this dispatch does not use.

#### Chain / dependency state

```
Phase B1 no_rope sweep (C/D/E)  ✅ COMPLETE (#2/#3/#4, 81 cells)
  -> no_rope_winner is now materializable from on-disk metrics.
Phase B1 H' partial_rope sweep  ✅ COMPLETE (#6, 48/48; on-disk readout)
Phase B1 H' Muon baseline  ✅ COMPLETE (#7, 15 cells = 5 seeds × 3 LRs; on-disk readout)
  -> partial_rope_winner is now materializable from on-disk metrics.
Phase C  pair_policy localization  ✅ COMPLETE (#5, 45/45; on-disk readout)
  -> MoE pair_policy default is now materializable from on-disk metrics.

Next:
  h100   -> materialize winners if needed -> B2 no-RoPE -> wait B2/A3 -> D1 residues 0,1,2,3 -> D3 if D-positive.
  h200x2 -> materialize winners if needed -> B2 partial-RoPE -> wait B2/A3 -> D1 residue 4 -> D4 if D-positive.
  h200x4 -> A1 AdamW@I -> A3 K-curve-I -> wait B2 -> D1 residues 5,6 -> E1 if D-positive.

Blocked:
  D1 O_mla (45) - blocked on both B2 confirmations and A3-I.
  D3 / D4 / E1 - blocked on D1 + D-gate.
```

**Work split**: B1/C1 readout is done by `scripts/phase21_materialize.py`.
Every launch first runs `scripts/filter_unfinished_jobs.py` against the exact
target `--results-dir`; a restart should not duplicate completed cells.

**H'-Muon (#7) must be read from disk before Phase-B partial-RoPE decision**:
it is the plain-Muon baseline the H' `partial_rope_winner` is judged against
(decision rule d.7.4 §2 — "matches or beats the H' Muon baseline"). The run
is 5-seed after the 6-cell top-up.

`phaseA2_*` NS top-up (4 cells, ~3 h) remains a deferred filler — blocked on
the canonical-coefficient hash check (pre-launch step 4).

### Lane queue per rig

| Physical rig / role | Queue (in order) | Cells | Parallelism | Results dirs |
|---|---|---|---|---|
| **8×H100 / `h100`** | B2 no-RoPE (10) -> D1 shard residues `0,1,2,3` (27) -> D3 LoRA-rank (30, conditional) | 10 + 27 (+30) | `--nproc-per-node 2 --num-workers 4 --gpus-per-worker 2` | `$GLOBAL/phase2.1-rigB-b2-no-rope-confirm/`, `$GLOBAL/phase2.1-rigA-d1-mla/`, `$GLOBAL/phase2.1-rigB-d3-lora-rank/` |
| **2×H200 / `h200x2`** | B2 partial-RoPE (5) -> D1 shard residue `4` (6) -> D4 FactFF (12, conditional) | 5 + 6 (+12) | `--nproc-per-node 2 --num-workers 1 --gpus-per-worker 2` | `$GLOBAL/phase2.1-rigB-b2-partial-rope-confirm/`, `$GLOBAL/phase2.1-rigA-d1-mla/`, `$GLOBAL/phase2.1-rigC-d4-factff/` |
| **4×H200 / `h200x4`** | A1 AdamW@I (10) -> A3 K-curve-I (9) -> D1 shard residues `5,6` (12) -> E1 350M MLA (30, conditional) | 19 + 12 (+30) | A/D1: `2×2-GPU`; E1: `--nproc-per-node 4 --num-workers 1 --gpus-per-worker 4` | `$GLOBAL/phase2.1-rigA-a1-adamw-I-topup/`, `$GLOBAL/phase2.1-rigA-a3-kcurve-I/`, `$GLOBAL/phase2.1-rigA-d1-mla/`, `$GLOBAL/phase2.1-rigA-e1-mla-350m/` |

D1 shard rule is fixed modulo 7 over `p21_d1_mla.jsonl`: `h100={0,1,2,3}`,
`h200x2={4}`, `h200x4={5,6}`. This covers all 45 D1 cells exactly once.

Expected result counts: B2 no-RoPE 10, B2 partial 5, A1 10, A3-I 9, D1 45,
D3 30, D4 12, E1 30.

### Phase-2.1 timeline (T=0 = Phase 2.1 launch)

| T (rig-days) | Event |
|---|---|
| 0.0    | R_screen → B1 C (27). R_prod → A1 AdamW@I (10). R_abl → A3 K-curve A0 (9). |
| 0.5    | R_abl finishes A3 A0. Run **canonical-coefficient hash check** before A2 (see pre-launch step 4). If pass → A2 (4 cells, 0.22 d). If approximate → rerun affected NS-policy cells before top-up. |
| 0.55   | R_screen finishes B1 C → starts B1 D. |
| 0.72   | R_abl finishes A2 (top-up case). Idle. |
| 1.10   | R_screen finishes B1 D → starts B1 E. |
| 1.25   | R_prod finishes A1 → starts A3 K-curve I (9 cells, 1.46 d). |
| 1.65   | R_screen finishes B1 E → starts B1 H' (48 cells, 0.65 d). |
| 2.30   | 2×H200 finishes B1 H' → starts B1 H' Muon baseline + top-up (15 cells, >10 h). |
| 2.39+  | 2×H200 finishes H' Muon; 8×H100 C1 is already complete in parallel. **Read B1/C1 from on-disk `metrics.jsonl`; write `p21_winners.env`.** |
| 2.39+  | C1 readout declares the MoE pair_policy default for D1. |
| 2.39+  | 8×H100 → B2-no-RoPE confirm (10 cells from winner overrides). |
| 2.39+  | 2×H200 → B2-partial-RoPE confirm (5 cells from winner overrides). |
| 2.60+  | Both B2 confirms finish. **MILESTONE: Phase B closed; v2.1 winner tuple declared.** |
| 2.71   | 4×H200 finishes A3 K-curve I; all roles wait until B2 and A3 gates are satisfied. |
| 2.71+  | All three roles launch D1 modulo-7 shards into one shared result dir. |
| 9.31+  | D1 finishes. **MILESTONE: D-gate decision.** delta = Muon - Coupled at own-best LR per optimizer. Commission iff delta_p50 > 1.5 × pooled_seed_std AND bootstrap CI direction stable. |
| 9.31+  | If D-positive: 4×H200 -> E1 350M MLA; 8×H100 -> D3 paired LoRA-rank; 2×H200 -> D4 P_factff. |
| 26.21+ | E1 finishes (if commissioned). Phase 2.1 complete. |

### Phase-2.1 dependencies

- **A independent**: A1/A3-I run on the 4×H200 role before D1.
- **B → D1**: D1 launches with `qk_coupling.no_rope_policy = <B no-RoPE winner>`,
  `qk_coupling.partial_rope_policy = <B partial-RoPE winner>`. Cannot launch
  before T = 2.60.
- **C → D1**: D1 launches with `pair_policy = <C winner>`. Cannot launch
  before the on-disk C1 readout exists.
- **D1 launch gate**: all roles wait for `b2_no_rope.done`,
  `b2_partial_rope.done`, and `a3.done` under `$GLOBAL/phase2.1-control/done/`.
- **D-gate**: commission E1 + D3 + D4 iff `delta_p50 > 1.5 × pooled_seed_std`
  with stable bootstrap CI sign. Otherwise stop; headline reframes per
  `suggestion.md §0`.

### Pre-launch hygiene (Phase 2.1 specific)

1. Offline launch is the default. The one-line payload writes
   `phase21_materialize.py` and `phase21_orchestrate.py` into
   `$GLOBAL/phase2.1-control/bin` on the remote node, then runs from the
   current repo root. Use `--no-skip-git-pull` only on a node that can safely
   refresh the repo.
2. Run **all** new and affected tests on the cluster:
   ```
   uv run pytest tests/test_qk_coupling_dispatch.py \
                 tests/test_pair_policy_routing.py \
                 tests/test_partial_rope.py \
                 tests/test_phase2_ladder_configs_load.py \
                 tests/test_optimizer_pairs.py \
                 tests/test_pair_factor_ratio_probe.py \
                 tests/test_coupled_zero_equals_muon.py \
                 tests/test_ns_coefficients.py \
                 tests/test_mla_forward_and_pairs.py \
                 tests/test_factorized_mlp_pair.py -x
   ```
   The three load-bearing correctness gates:
   - `test_full_rope_qk_policy_invariance` (preserves A0/B/G/H anchors)
   - `test_current_flat2d_fallback_matches_legacy_learned_pos` (preserves C/D
     negative control)
   - `test_nonfull_qk_policy_does_not_change_vo_routing` (Phase B is
     Q-K-only)
3. Update `scripts/check_progress.sh` `RIGS` array with the new
   `phase2.1-rig*-*` paths.
4. **NS-policy canonical-coefficient hash check (before A2)**: confirm via
   W&B configs / artifact hashes that the existing 33 Phase-2 S1 cells used
   canonical Polar-Express and Cesista coefficient tables (NVIDIA
   `emerging-optimizers` reference). If approximate, rerun the affected
   cells before the A2 top-up — do NOT mix coefficient generations within
   a single grid.
5. **Sign-convention sanity (before D-gate)**: scripts / notebooks computing
   the D-gate must use `delta = Muon − Coupled` (positive ⇒ Coupled wins).
   Part II tables use `Coupled − Muon`; numbers must be negated when imported.
6. **Smoke configs** (`configs/smoke/*.yaml`) override
   `pair_factor_ratio_interval_tokens: 0` so the new base.yaml default does
   not add cost to smoke runs. Verify these overrides still load post-`git pull`.
7. **Phase-B B1 -> B2 launch coupling**: do not use W&B as a gate. Read
   completed cell directories from disk (`config.yaml`, `metrics.jsonl`,
   `stdout.log`) and generate JSONL with explicit overrides. B2 no-RoPE must
   carry separate C/D best LRs; D1/D3/D4/E1 carry both qk winners plus the C1
   `pair_policy`.
