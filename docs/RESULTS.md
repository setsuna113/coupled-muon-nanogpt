# Results index: where they live, how to query, what maps to which plan step

Snapshot: **2026-07-16**. Companion docs: `experiment.md` (the plan), `DISPATCH.md`
(live dispatch state), `docs/README.md` (report build).

---

## 1. Three places results live

| Layer | What | Authority |
|---|---|---|
| **Per-cell on disk** | `<results-dir>/<run_id>/` → `metrics.jsonl`, `stdout.log`, `wandb/offline-run-*/`, `*.pt` | **Canonical numeric record.** `metrics.jsonl` is the source of truth |
| **wandb cloud** | one project per sweep/arm | How results are *explored*; populated post-hoc by `wandb sync` |
| **`docs/data/*.parquet`** | frozen summary snapshots the reports build from | Report inputs; **stale — dated 2026-05-19** |

Training runs offline (`WANDB_MODE=offline` in `my_env.sh`), so wandb is always
*behind* disk until someone syncs. `wandb_utils.py:74` passes `dir=out_dir` to
`wandb.init`, which is why offline runs land **per-cell**, not in the global
`WANDB_DIR`.

A cell is finished iff its `stdout.log` contains `[saved_ckpt]` (that is what
`scripts/check_progress.sh` counts).

## 2. Where results have historically been written

`$GLOBAL = /inspire/hdd/global_user/yanjunchi-24040/yancheng`
`$PROJECT = /inspire/hdd/project/quantum-artificial-intelligence/yanjunchi-24040/yancheng/coupled-muon-nanogpt/results`

| Work | Directory |
|---|---|
| Stage 1 (A0 repro) | `$PROJECT/ladder_A0_llama60m-*` |
| Stage 2 dense ladder | `$GLOBAL/coupled-muon-stage2-dense-archive/` |
| Stage 3 MoE | `$GLOBAL/coupled-muon-stage3-moe/`, `-rigB-pt1/`, `-rigA-anchor-screen/`, `-rig2H200-adamw-Iprime/`, `-rigB-pt3/` |
| §d.4 ablations | `$GLOBAL/coupled-muon-d4ablations/` |
| Phase 2 — MLA | `$GLOBAL/phase2-rigA-mla/` ← note `phase2-`, **not** `phase2.1-` |
| Phase 2.1 — everything else | `$GLOBAL/phase2.1-<rig>-<arm>/` |
| **Not this project** | `$GLOBAL/muon_outputs/` — the separate Riemannion / Muon-finetuning paper line (`muon-bench-*`, `riemannion*` on wandb). Never mix it into this project's analysis |

Big sweeps go to `$GLOBAL` via `--results-dir` to spare the project quota.

## 3. How to query wandb

Auth is in `~/.netrc`; the `wandb` CLI is not on `PATH` — go through `uv run`.
Entity: **`liuyc1025-university-of-cambridge`**.

```python
# uv run python -c "..." from the repo root
import wandb
api = wandb.Api()
ENT = "liuyc1025-university-of-cambridge"

# inventory
for p in sorted(x.name for x in api.projects(ENT)):
    print(p)

# one project
runs = list(api.runs(f"{ENT}/coupled-muon-mla", per_page=500))
for r in runs:
    cfg  = r.config
    rung = cfg.get("name")                          # e.g. ladder_O_mla
    opt  = cfg.get("optimizer", {}).get("type")     # adamw | muon | coupled_muon_v2
    lr   = cfg.get("optimizer", {}).get("lr")
    fvl  = r.summary.get("final_val_loss")          # the outcome
    div  = r.summary.get("diverged")                # bool
```

Gotchas that cost real time:

- **Don't pull `r.history()`** across a whole project — it will time out. Summary
  fields only, unless you're comparing curves.
- `state == "finished"` even for diverged cells. Filter on `summary["diverged"]`.
- `seed` is often absent from config — parse it from the run name (`·s(\d+)`).
- **Run counts > cell counts.** `init_wandb` does *not* pass `id=`, so every
  relaunch of a cell creates a *new* wandb run with the same display name.
  Dedup on full config, not on name (that is what `part2_summary_deduped.parquet`
  and `part2_discarded_duplicates.csv` exist for).
- **Display names are not unique per cell.** The swept axis (K, kv-rank,
  qk-policy) lives in config/tags, not in the name — so several distinct cells
  legitimately share one seed-based name.
- For cross-optimizer comparison pair by `(rung, seed)` at **each arm's own best
  LR** — Muon-family and AdamW sweet spots are ~10× apart.

Useful summary fields: `final_val_loss`, `diverged`, `final_step`, `final_tokens`,
`_runtime`. Probes (`probe/coupled_pair/*`, `probe/attn_logit/*`, `probe/moe_load/*`)
are high-cardinality — ignore unless investigating a specific arm.

## 4. wandb project → plan step

Counts as of 2026-07-16. "Expected" = cell count from `scripts/check_progress.sh`.

### Already written up

| Project | Runs | Plan step | In report? |
|---|---|---|---|
| `coupled-muon-A0-repro` | 75 | Stage 1 — A0 LLaMA-60M reproduction | **Part I §03** |
| `coupled-muon-ladder` | 270 | Stage 2 — dense ladder B/C/D/E/G/H (6×3×5×3) | **Part I §04** |
| `coupled-muon-ablation` | 106 | §d.4 mechanical ablations | **Part I App B** |
| `coupled-muon-moe-pt1-prod` | 60 | Stage 3 MoE rungs I/I′/J/K | **Part II** |
| `coupled-muon-moe-pt2-screen` | 60 | Stage 3 MoE screen (**grew from 34 → 60** after the report snapshot) | **Part II — stale** |
| `coupled-muon-moe-anchor-adamw-Iprime` | 25 | Stage 3 AdamW@I′ anchor | **Part II** |
| `coupled-muon-ns-policy` | 36 | Phase 2.1 **A2** — NS-coefficient policy sweep | **Part II** (defensive partial) |

### Phase 2.1 — synced but NOT in any report yet

| Project | Runs | Expected | Plan step (`experiment.md` §d.7) | Status |
|---|---|---|---|---|
| `coupled-muon-qk-policy` | **144** | 27+27+27+48+15 = **144** | **B1** — qk_coupling repair: no-rope C/D/E, partial-rope H′, Muon baseline H′ | ✅ complete |
| `coupled-muon-qk-policy-confirm` | **15** | 10+5 = **15** | **B2** — no-rope + partial-rope winner confirmation | ✅ complete |
| `coupled-muon-mla` | **51** | **45** (+reruns) | **D1** — MLA cross-optimizer LR grid (AdamW/Muon/Coupled) | ✅ complete |
| `coupled-muon-k-curve-I` | **9** | **9** | **A3** — coupled-steps K∈{1,2,4} at rung I | ✅ complete |
| `coupled-muon-k-curve` | **28** | 9 (+reruns) | **A3** — K-curve at A0 | ✅ complete |
| `coupled-muon-moe-anchor-adamw` | **45** | 25 + 10 topup (+reruns) | **A1** — AdamW@I anchor equalization | ✅ complete |
| `coupled-muon-pair-policy` | **15** | — | **C1** — MoE pair_policy {attention_only, ffn_only, all} at rung I | ⏳ partial |
| `coupled-muon-pair-policy-lr` | **6** | — | **C1** — pair_policy × LR | ⏳ partial |
| `coupled-muon-lora-rank` | 15 | — | **repurposed** → MLA KV-compression rank sweep (`ladder_O_mla`, kv∈{16..256}). The 2 old LoRA rows Part II cites are stale | ⚠️ misleading name |

Why B1/B2 matter: `factory.py` silently disabled multi-head Q–K coupling whenever
RoPE wasn't full, falling back to flat-2D. **That bug is the leading explanation
for Part I's C/D regression** (+0.055/+0.066 nats) — B repairs it and re-decides
the claim. D1/MLA is the central new claim from `suggestion.md`: CoupledMuon ≡
plain Muon on unfactored weights, so MLA is the one production family that
actually exercises the mechanism.

### Should exist but NOT yet on wandb

| Plan step | Expected cells | Where | Why absent |
|---|---|---|---|
| **C1** `pair_policy` remainder | ~24 of 45 | `$GLOBAL/phase2.1-rigB-c-pair-policy/` | **sync still in flight** — resume with the command below |
| **A2** NS-policy top-up | 4 | `$GLOBAL/phase2.1-rigC-a2-ns-topup/` | conditional on a coefficient-hash check |
| **A2** LR-prefactor sweep | 18 | — | never launched |
| **D3** paired LoRA-rank | 30 | `$GLOBAL/phase2.1-rigB-d3-lora-rank/` | gated on a positive **D-gate** |
| **D4** imposed FFN factorization | 12 | `$GLOBAL/phase2.1-rigC-d4-factff/` | gated on D-gate |
| **E1** MLA @ 350M | 30 | `$GLOBAL/phase2.1-rigA-e1-mla-350m/` | gated on D-gate |

D-gate (per `DISPATCH.md`): commission D3/D4/E1 only if
`delta(Muon − Coupled) > 1.5 × pooled-seed-std` with a stable bootstrap CI.
Now that D1 is fully synced, **the D-gate is evaluable.**

## 5. Syncing offline runs → wandb

Offline cells only reach wandb via an explicit post-hoc sync from an
internet-connected node. The driver script lives on persistent storage:

```bash
# resume / continue a sync (idempotent — already-synced runs are skipped)
tmux kill-session -t wsync 2>/dev/null
tmux new -d -s wsync "bash $GLOBAL/sync_phase2.sh 2>&1 | tee -a $GLOBAL/wsync_p2.log"
```

It loops `wandb sync --no-include-synced <glob>` until a pass uploads nothing new,
so transient `TransientError` retry stalls self-heal and an interrupted run just
resumes.

Hard-won rules:

- **Scope the glob to `phase2*`** — `$GLOBAL/phase2*/*/wandb/offline-run-*`. It
  catches both `phase2-rigA-mla` and `phase2.1-*`. A `phase2.1-*` glob **misses
  MLA**; a `$GLOBAL/*/*` glob drags in `muon_outputs/` (the other paper) and
  re-uploads ~1000 old archived runs first, because the glob expands
  alphabetically (`coupled-muon-*` < `muon_outputs` < `phase2*`).
- **`WANDB_MODE=online` must be set explicitly** — `my_env.sh` forces `offline`.
- **Re-syncing cannot create duplicates.** The run id is baked into
  `run-<ID>.wandb`, so a re-upload overwrites the same run. Old archives lack a
  `.synced` marker and therefore *do* re-upload — harmless, just wasted time,
  which is exactly why the glob must be scoped.
- Put scripts/logs on `$GLOBAL`, not `~` — only project/personal/global paths
  survive a pod restart.

## 6. Refreshing the reports after a sync

`docs/data/*.parquet` is frozen at 2026-05-19 and does **not** contain any Phase
2.1 result, nor `moe-pt2-screen`'s +26 top-up runs.

```bash
uv run python docs/figures/fetch_all.py   --refresh                  # Part I
uv run python docs/figures/fetch_part2.py --refresh --summary-only   # Part II
uv run python docs/figures/make_all.py && uv run python docs/figures/make_part2.py
```

`fetch_part2.py` records missing future projects in `part2_manifest.json` rather
than failing, so newly-created Phase 2.1 projects need adding there before they
appear.
