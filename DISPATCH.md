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

### Phase 2.1 — live progress

Operational state log. Update on every launch / completion. The "Lane queue
per rig" table below is the *plan*; this table is what has actually run.

**Note on rig usage**: two rigs now active in parallel — the **2×H200** and
the **8×H100**. The plan's results-dir label `rigB` is the *logical*
Phase-2.1 lane (R_screen in the plan), not the physical rig — Phase B rungs
C/D/E are split across both physical rigs. The run_id hash is cfg+seed, so
the physical rig does not matter for resume / dedupe.

| # | Sweep | Phase | Cells | Physical rig | Results dir (`$GLOBAL/…`) | State | Wall-clock | Wandb |
|---|---|---|---|---|---|---|---|---|
| 1 | `phaseA3_k_curve_minimal_A0` | A3 | 9 | 2×H200 | `phase2.1-rigC-a3-kcurve-a0/` | ✅ done 9/9, synced | 4.17 h | `coupled-muon-k-curve` (synced) |
| 2 | `phaseB_qk_no_rope_C` | B1 | 27 | 2×H200 | `phase2.1-rigB-b-qk-no-rope-C/` | ▶ running | ~26 h est. | `coupled-muon-qk-policy` (offline) |
| 3 | `phaseB_qk_no_rope_D` | B1 | 27 | 8×H100 | `phase2.1-rigB-b-qk-no-rope-D/` | ▶ running (4-way parallel) | ~9.4 h est. | `coupled-muon-qk-policy` (offline) |
| 4 | `phaseB_qk_no_rope_E` | B1 | 27 | 8×H100 | `phase2.1-rigB-b-qk-no-rope-E/` | ⏳ queued (chained after #3) | ~9.4 h est. | `coupled-muon-qk-policy` (offline) |

Measured anchors (supersede the conservative plan estimates): A0 dense
60M-CS at 1.2B tokens ran **0.46 h/cell** on the 2×H200 (4.17 h / 9). C/D/E
are 2.5B tokens (2.08×) ⇒ ≈ **1.0 h/cell on 2×H200**, ≈ **1.34 h/cell on
2×H100** (H200 ≈ 1.4× H100). On the 8×H100 in 4-way-parallel mode
(`--nproc-per-node 2 --num-workers 4`, 2 GPU/cell) a 27-cell rung is
≈ 7 waves × 1.34 h ≈ **9.4 h**; D + E chained ≈ **19 h**.

**8×H100 grad_accum note**: 4-way-parallel mode keeps each cell at
world_size=2 (2 GPU/cell), so the ladder YAML's `grad_accum_steps: 8` (tuned
for the 2-GPU baseline) is correct as-is — do NOT quarter it. Quartering
only applies to a single cell spanning all 8 GPUs (world_size=8), which this
dispatch does not use.

**Work split for the Phase B no_rope sweep (C/D/E)**: 2×H200 → C (#2);
8×H100 → D + E chained (#3, #4). All three finish roughly in parallel
(~26 h on 2×H200 vs ~19 h on 8×H100).

**Next after the above**:
- 2×H200 after #2 (C): `phaseB_qk_partial_rope_Hprime` (48) →
  `phaseB_muon_baseline_Hprime` (9).
- 8×H100 after #4 (E): B2 confirms once the B1 winners are read off W&B,
  or pick up H' / H'-Muon if the 2×H200 is still busy.
- `phaseA2_*` NS top-up (4 cells, ~2 h) is a deferred filler — blocked on
  the canonical-coefficient hash check (pre-launch step 4 below).

### Lane queue per rig

| Rig | Queue (in order) | Cells | Wandb projects | Results dirs |
|---|---|---|---|---|
| **R_abl — 2×H200** | `phaseA3_k_curve_minimal_A0` (9) → `phaseA2_polar_express_K8_coupled_topup` + `phaseA2_cesista_K8_paired_topup` (4) → idle / `imposed_factff` (D4, 12, conditional) | 13 (+12 cond.) | `coupled-muon-k-curve`, `coupled-muon-ns-policy`, `coupled-muon-imposed-factff` | `$GLOBAL/phase2.1-rigC-{a3-kcurve-a0,a2-ns-topup,d4-factff}/` |
| **R_prod — 4×H200** | `adamw_equalization_I` (A1, 10) → `phaseA3_k_curve_minimal_I` (9) → `mla_lr_grid` (D1, 45) → `mla_scaling_350m` (E1, 30, conditional) | 64 (+30 cond.) | `coupled-muon-moe-anchor-adamw`, `coupled-muon-k-curve-I`, `coupled-muon-mla`, `coupled-muon-mla-350m` | `$GLOBAL/phase2.1-rigA-{a1-adamw-I-topup,a3-kcurve-I,d1-mla,e1-mla-350m}/` |
| **R_screen — 8×H100** | `phaseB_qk_no_rope_C` (27) → `phaseB_qk_no_rope_D` (27) → `phaseB_qk_no_rope_E` (27) → `phaseB_qk_partial_rope_Hprime` (48) → `phaseB_muon_baseline_Hprime` (9) → `phaseB2_confirm_no_rope` (10) → `phaseB2_confirm_partial_rope` (5) → `phaseC_pair_policy_lr` (45) → idle / `phaseD3_lora_rank_paired_conditional` (D3, 30, conditional) | 198 (+30 cond.) | `coupled-muon-qk-policy`, `coupled-muon-qk-policy-confirm`, `coupled-muon-pair-policy-lr`, `coupled-muon-lora-rank-paired` | `$GLOBAL/phase2.1-rigB-{b-qk-{no-rope,partial-rope,muon-baseline}-*, b2-{no,partial}-rope-confirm, c-pair-policy, d3-lora-rank}/` |

R_screen is the global critical path at ~4.12 rig-days through Phase B + C.

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
| 2.30   | R_screen finishes B1 H' → starts B1 H' Muon baseline (9 cells, 0.09 d). |
| 2.39   | R_screen finishes H' Muon. **Read B1 W&B; declare (no_rope_winner, partial_rope_winner).** |
| 2.39   | R_screen → B2-no-RoPE confirm (10 cells, override no_rope_policy + best-LR per rung). |
| 2.53   | R_screen → B2-partial-RoPE confirm (5 cells, override partial_rope_policy + H' best LR). |
| 2.60   | R_screen finishes B2. **MILESTONE: Phase B closed; v2.1 winner tuple declared.** |
| 2.60   | R_screen → C1 pair_policy (45 cells, 1.83 d). |
| 2.71   | R_prod finishes A3 K-curve I. R_prod **IDLE** (waiting on C). |
| 4.43   | R_screen finishes C1. **MILESTONE: Phase C closed; MoE pair_policy default declared.** |
| 4.43   | R_prod → D1 O_mla (45 cells, with B-winners + C-winner baked in via `--override`; 6.6 d). |
| 11.03  | R_prod finishes D1. **MILESTONE: D-gate decision.** delta = Muon − Coupled at own-best LR per optimizer. Commission iff delta_p50 > 1.5 × pooled_seed_std AND bootstrap CI direction stable. |
| 11.03+ | If D-positive: R_prod → E1 350M MLA (16.9 d); R_screen → D3 paired LoRA-rank (1.22 d); R_abl → D4 P_factff (0.65 d). |
| 27.93  | E1 finishes (if commissioned). Phase 2.1 complete. |

### Phase-2.1 dependencies

- **A independent**: A1 (R_prod), A2 / A3 (R_abl) run on their own rigs.
- **B → D1**: D1 launches with `qk_coupling.no_rope_policy = <B no-RoPE winner>`,
  `qk_coupling.partial_rope_policy = <B partial-RoPE winner>`. Cannot launch
  before T = 2.60.
- **C → D1**: D1 launches with `pair_policy = <C winner>`. Cannot launch
  before T = 4.43.
- **D1 launch gate**: `T_D1_launch = max(R_prod free, Phase B done, Phase C done) = max(2.71, 2.60, 4.43) = 4.43`.
- **D-gate**: at T = 11.03, commission E1 + D3 + D4 iff `delta_p50 > 1.5 × pooled_seed_std`
  with stable bootstrap CI sign. Otherwise stop; headline reframes per
  `suggestion.md §0`.

### Pre-launch hygiene (Phase 2.1 specific)

1. `git pull` for the Phase-2.1 code: `optim/coupled_muon.py` `qk_coupling`
   dispatch (3-key dict), `optim/factory.py` `pair_policy` enum,
   `configs/base.yaml` defaults (Q-K policy preserves Phase-1 bitwise),
   `configs/ladder/H_prime_partial_rope.yaml` (new rung), new tests.
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
7. **Phase-B B1 → B2 launch coupling**: at T = 2.39, query W&B for the
   no_rope_winner and partial_rope_winner. Use `--override` at B2 launch to
   bake them in. The same overrides go into the D1 launch at T = 4.43 plus
   the C1 pair_policy winner.
