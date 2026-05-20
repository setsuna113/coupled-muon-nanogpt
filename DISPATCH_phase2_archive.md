# ARCHIVE — Phase 2 (flat) dispatch

> **Status: ARCHIVED / SUPERSEDED.** The active plan is the Phase-2.1 gated
> tree in `DISPATCH.md` (committed in `9213840`). This file preserves the
> *old flat Phase-2 dispatch* (last committed state: `10e4487`, including the
> A3.I + A1 → R_screen reschedule) because the three rigs were launched
> against it and that in-flight work is being allowed to finish before any
> switch to Phase-2.1.
>
> **Do not start new work from this file.** Use it only to (a) interpret
> results from cells already running under the flat plan, and (b) decide the
> next step for a rig that is mid-flat-lane. Once a rig drains its flat lane,
> its *next* job comes from `DISPATCH.md` Phase-2.1, not from here.
>
> Provenance: `git show 10e4487:DISPATCH.md` (Phase-2 section).

---

## Old Phase-2 dispatch (verbatim, as of `10e4487`)

Phase 2 adds three new lanes of cells across the three idle rigs. The headline goal is the **MLA cross-optimizer triple at O_mla** plus an NS-policy defence (S1) and an AdamW-tuning equalizer (A3). See `experiment.md` §d.7 for the design rationale. Total expansion ≈ 207 cells, ~18 rig-days work against the +22 rig-day generous budget.

| Rig | Sweeps (run in order) | Cells | Wandb project(s) | Results dir |
|---|---|---|---|---|
| **R_abl — 2×H200** | `ns_coeff_K_sweep` (S1, 36) → `k_curve_at_winner` (S3, 18) → `lr_prefactor_ablation` (A2, 16) → `imposed_factff` (B3, 12) | 82 | `coupled-muon-ns-policy`, `coupled-muon-k-curve`, `coupled-muon-lr-prefactor`, `coupled-muon-imposed-factff` | `$GLOBAL/phase2-rigC-{ns-policy,k-curve,lr-prefactor,imposed-factff}/` |
| **R_prod — 4×H200** | `mla_lr_grid` (S2, 45) → `mla_scaling_350m` (stretch, 30 — conditional) | 75 (or 45 if stretch NO-GO) | `coupled-muon-mla`, `coupled-muon-mla-350m` | `$GLOBAL/phase2-rigA-{mla,mla-350m}/` |
| **R_screen — 8×H100** | `adamw_equalization_I_prime` (A3.I', 10) → `lora_rank_sweep` (B1, 15) → `adamw_equalization_I` (A3.I, 10) → `pair_policy_{attn_only,ffn_only,all}` (A1, 15) | 50 | `coupled-muon-moe-anchor-adamw-Iprime`, `coupled-muon-lora-rank`, `coupled-muon-moe-anchor-adamw`, `coupled-muon-pair-policy` | `$GLOBAL/phase2-rigB-{adamw-Iprime,lora-rank,adamw-I,pair-policy}/` |

### Phase-2 timeline (T=0 = Phase-2 launch)

| T (rig-days) | Event |
|---|---|
| 0     | R_abl starts S1; R_prod starts S2 (Bernstein default — winner backfilled at T+1.5d); R_screen starts A3.I'. |
| 0.25  | R_screen finishes A3.I'. Starts B1. |
| 1.5   | R_abl finishes S1. **MILESTONE: NS-policy winner declared.** Re-emit S3/S2/A1/B1/stretch JSONLs with the winner's `optimizer.ns_coefficients` value. |
| 2.15  | R_screen finishes B1. Starts A3.I (reassigned from R_prod). |
| 2.25  | R_abl finishes S3. Starts A2. |
| 2.4   | R_screen finishes A3.I. Starts A1 (reassigned from R_prod). |
| 2.92  | R_abl finishes A2. Starts B3. |
| 3.4   | R_screen finishes A1. R_screen idle. |
| 3.42  | R_abl finishes B3. R_abl idle. |
| 5.6   | R_prod finishes S2. **MILESTONE: O_mla cross-optimizer triple done. 350M-stretch GO/NO-GO decision.** |
| ~15.3 | R_prod finishes 350M stretch (if commissioned; sequential 4-GPU DDP). All Phase 2 complete. |

### Phase-2 dependencies

- **S1 → {S3, S2, A1, B1, stretch}** — the winning NS policy backfills the
  `optimizer.ns_coefficients` fixed value for these sweeps. In-flight
  Bernstein cells from S2 remain valid as a "Bernstein baseline" half-grid;
  remaining cells are re-launched with the winner.
- **O_mla S2 → 350M stretch** — gating rule: at T+5.6 d, read out the
  bootstrap CI on the median val-loss gap at the best LR per optimizer;
  commission stretch only if `gap_p50 > 1.5 × pooled_seed_std`. If gap is
  unresolved, drop the 30 stretch cells (saves ~3.75–7.5 R_prod rig-days).
- **A3** is independent — runs in parallel with everything else.

### Pre-launch hygiene (Phase 2 specific)

1. `git pull` to pick up the Phase-2 code: `optim/ns_coefficients.py`, the
   MLA layer in `model/attention.py`, the `FactorizedMLP` in `model/mlp.py`,
   the `pair_factor_ratio` probe in `probes/`, the new ladder/sweep YAMLs.
2. Update `scripts/check_progress.sh` `RIGS` array to add the new Phase-2
   results dirs.
3. **Tests on the cluster** before launch: `uv run pytest tests/test_ns_coefficients.py tests/test_lr_prefactor.py tests/test_ns_gram_form.py tests/test_mla_forward_and_pairs.py tests/test_factorized_mlp_pair.py tests/test_phase2_ladder_configs_load.py tests/test_pair_factor_ratio_probe.py` — local dev box lacks cudnn so these were syntax-checked only.
4. Replace `optimizer.ns_coefficients: bernstein` in S3, S2, A1, B1, and the
   350M-stretch sweep YAMLs with the S1 winner before re-launch at T+1.5d.
   Convention: keep Bernstein as the committed default so the YAML survives
   resume semantics if S1 names Bernstein the winner.
5. **Polar Express / Cesista coefficient tables are approximate** in
   `optim/ns_coefficients.py` (see module docstring). Before the headline
   publication run, patch in the canonical values from the published
   supplements / NVIDIA `emerging-optimizers` reference.

---

## Drain plan — what each rig finishes under the flat plan

This is the "let the old work finish" tracking section. S1's NS-policy winner
is **bernstein** (already the committed YAML default), so no `ns_coefficients`
backfill is required for the remaining flat-lane jobs.

### R_abl — 2×H200

- **Done**: S1 `ns_coeff_K_sweep` — 36 cells launched, 33 completed + 3 diverged
  (`polar_express` K=8 ×2 seeds, `cesista` K=8 ×1 seed). Winner: bernstein.
- **Next (flat)**: S3 `k_curve_at_winner` (18 cells).
  ```
  source ../my_env.sh && uv run python -m coupled_muon_nanogpt.sweep.expand configs/sweeps/k_curve_at_winner.yaml --out /tmp/phase2_s3.jsonl ; WANDB_MODE=offline nohup uv run python scripts/run_sweep.py /tmp/phase2_s3.jsonl --results-dir /inspire/hdd/global_user/yanjunchi-24040/yancheng/phase2-rigC-k-curve --nproc-per-node 2 --num-workers 1 > /tmp/phase2_rig_abl_s3.log 2>&1 & disown ; sleep 2 ; tail -f /tmp/phase2_rig_abl_s3.log
  ```
- **Then (flat)**: A2 `lr_prefactor_ablation` (16) → B3 `imposed_factff` (12).
- **After the flat lane drains**: switch R_abl to Phase-2.1
  (`phaseA3_k_curve_minimal_A0` etc., per `DISPATCH.md`).

### R_prod — 4×H200

- **Running / next (flat)**: S2 `mla_lr_grid` (45 cells). If not yet launched:
  ```
  source my_env.sh && uv run python -m coupled_muon_nanogpt.sweep.expand configs/sweeps/mla_lr_grid.yaml --out /tmp/phase2_s2.jsonl ; WANDB_MODE=offline nohup uv run python scripts/run_sweep.py /tmp/phase2_s2.jsonl --results-dir /inspire/hdd/global_user/yanjunchi-24040/yancheng/phase2-rigA-mla --nproc-per-node 4 --num-workers 1 --extra-override training.grad_accum_steps=8 > /tmp/phase2_rig_prod_s2.log 2>&1 & disown ; sleep 2 ; tail -f /tmp/phase2_rig_prod_s2.log
  ```
- **Then (flat)**: 350M-stretch `mla_scaling_350m` (30, conditional on the
  T+5.6d gate `gap_p50 > 1.5 × pooled_seed_std`).
- **After the flat lane drains**: Phase-2.1 reuses `mla_lr_grid` as D1 but
  gated on Phase-B/C winners — see `DISPATCH.md`. S2 cells run here under the
  flat plan are **Bernstein-baseline only** and may need a re-run for D1.

### R_screen — 8×H100

- **Running**: A3.I' `adamw_equalization_I_prime` (10) + B1 `lora_rank_sweep`
  (15), chained.
- **Next (flat, rescheduled)**: A3.I `adamw_equalization_I` (10) → A1
  `pair_policy_{attn_only,ffn_only,all}` (15). Chained launch:
  ```
  source ../my_env.sh && nohup bash -c 'set -e ; uv run python -m coupled_muon_nanogpt.sweep.expand configs/sweeps/adamw_equalization_I.yaml --out /tmp/phase2_a3i.jsonl && WANDB_MODE=offline uv run python scripts/run_sweep.py /tmp/phase2_a3i.jsonl --results-dir /inspire/hdd/global_user/yanjunchi-24040/yancheng/phase2-rigB-adamw-I --nproc-per-node 4 --num-workers 2 --extra-override training.grad_accum_steps=8 && for s in attn_only ffn_only all ; do uv run python -m coupled_muon_nanogpt.sweep.expand configs/sweeps/pair_policy_$s.yaml --out /tmp/phase2_a1_$s.jsonl ; done && cat /tmp/phase2_a1_*.jsonl > /tmp/phase2_a1.jsonl && WANDB_MODE=offline uv run python scripts/run_sweep.py /tmp/phase2_a1.jsonl --results-dir /inspire/hdd/global_user/yanjunchi-24040/yancheng/phase2-rigB-pair-policy --nproc-per-node 4 --num-workers 2 --extra-override training.grad_accum_steps=8' > /tmp/phase2_rig_screen_a3i_a1.log 2>&1 & disown ; sleep 2 ; tail -f /tmp/phase2_rig_screen_a3i_a1.log
  ```
- **Caveat**: under Phase-2.1, `adamw_equalization_I_prime` and the
  unconditional `lora_rank_sweep` are *removed from the active plan*; A3.I'/B1
  results stay valid as standalone equalization / LoRA-rank data but are not
  on the Phase-2.1 critical path. After the flat lane drains, switch R_screen
  to Phase-2.1 Phase-B (`phaseB_qk_no_rope_C` first) — it is the global
  critical path and gates R_prod's D1.

---

## Crosswalk: flat Phase-2 → Phase-2.1

| Flat-Phase-2 job | Phase-2.1 fate |
|---|---|
| S1 `ns_coeff_K_sweep` | Kept — feeds both; Phase-2.1 A2 is an S1 top-up. |
| S3 `k_curve_at_winner` (18, K∈{1,2,3,5,8,12}) | Superseded by `phaseA3_k_curve_minimal_A0` (9, K∈{1,2,4}). |
| A2 `lr_prefactor_ablation` | Removed (phase3-conditional). |
| B3 `imposed_factff` | Now D4 — conditional on the Phase-2.1 D-gate. |
| S2 `mla_lr_grid` | Now D1 — gated on Phase-B/C winners; flat S2 cells are Bernstein-baseline only. |
| A1 `pair_policy_*` | Superseded by `phaseC_pair_policy_lr` (own-best-LR per policy). |
| A3.I `adamw_equalization_I` | Kept as Phase-2.1 A1 (R_prod lane). |
| A3.I' `adamw_equalization_I_prime` | Removed ("A1' AdamW@I' top-up"). |
| `lora_rank_sweep` (B1) | Superseded by `phaseD3_lora_rank_paired_conditional` (D-gated). |
| `mla_scaling_350m` (stretch) | Now E1 — gated on Phase-2.1 D-gate positive. |
