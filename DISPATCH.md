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

## Active dispatch (Phase 2)

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
