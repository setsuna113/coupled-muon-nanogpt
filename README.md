# coupled-muon-nanogpt

Research scaffold for the **Coupled Muon v2** bridging-experiment ladder defined in
`experiment.md`. Migrates a partner-weight-aware Newton–Schulz optimizer from a
LLaMA-60M baseline onto a toggleable, NanoGPT-style harness so the source of its gain
can be localized via architectural ablations.

## Status: closed (2026-09-02)

The study ran 960 W&B runs (≈3.4k GPU-hours, ≈2.75T tokens) between May and July
2026 and closed when cluster access ended. **Read `docs/FINAL_STATUS.md` first** —
it is the single up-to-date readout, regenerated from a committed W&B snapshot.

Headline results (final val loss, own-best LR per optimizer, paired seeds, 95% CI):

| Setting | Coupled − Muon | Coupled − AdamW |
|---|---|---|
| Dense, full RoPE (A0 LLaMA-60M, B, G, H) | −0.008 … −0.024 nats, all CIs exclude 0 | −0.12 … −0.29 |
| Dense, learned position (C, D, E) | +0.055 / +0.066 / tie with the legacy flat-2D Q–K path (mostly an LR-window artefact of the divergence-flag rule); −0.011 … −0.024 at own-best LR with per-head Q–K coupling, probe-flagged runs admitted, n=3 (FINAL_STATUS §4.5) | −0.12 |
| Sparse MoE (I, I′, J, K, L, M; 5B tokens) | −0.007 … −0.016, no cell diverged | −0.09 … −0.10 |
| MLA rung O (natively factored KV) | **+0.003 [+0.003, +0.005]** — D-gate negative; factored-KV hypothesis falsified | −0.10 |

Coupled costs 1.04–1.12× Muon's wall-clock per cell (a fixed cost, not
proportional to the inner step count), which nets the fixed-token gain out to
0.94–1.02× on every rung: it is a mechanism-level result, not a speedup
(FINAL_STATUS §4.7). Where to look:

| Question | Read |
|---|---|
| What is the algorithm, why might it work, what would falsify it | `experiment.md` (§a, Appendix X, §d.7) |
| What ran, what it found, what is left | `docs/FINAL_STATUS.md` |
| Project summary / contributions | `docs/PROJECT_SUMMARY.md` |
| Could this still become a paper | `docs/PAPER_FEASIBILITY.md` |
| Formal write-ups (LaTeX) of Stage 1–2 and Stage 3 at the 2026-05-19 snapshot | `docs/report.tex`, `docs/report_part2.tex` (build: `docs/README.md`) |
| Historical dispatch state | `DISPATCH.md`, `docs/RESULTS.md` (frozen) |

### The update rule in one paragraph

Muon replaces a matrix gradient `G` by its polar factor `NS(G) ≈ UVᵀ` (Newton–Schulz
quintic). Coupled Muon v2 treats weights that only ever appear as a *product* in the
forward pass — `W_Q W_Kᵀ` per head (in RoPE's 2-D rotation blocks), `W_V W_O` per
head, `W_up W_down` — as pairs `(A, B)` and computes a two-stage update
`U_A = NS( C_A(G_A; B) )`, where stage 1 runs the same quintic on the iterate `X B`
so that `X B → polar(G_A B)` (the *product* moves in its polar direction, using the
partner's current value), and stage 2 re-orthogonalises `X` so the usual
`lr·0.2·√max(rows, cols)` scaling applies. `coupled_steps=0` recovers plain Muon
(tested to 1e-4 in `tests/test_coupled_zero_equals_muon.py`). Implementation:
`src/coupled_muon_nanogpt/optim/coupled_muon.py`; pair classification:
`optim/factory.py`.

## Quickstart

```bash
# 1. Install (uv recommended)
uv sync --all-extras

# 2. Tokenize FineWeb-Edu into uint16 shards (~30 min, one-time)
bash scripts/prepare_data.sh

# 3. Smoke test on 1×4090 (5 min)
bash scripts/smoke_4090.sh configs/smoke/tiny_dev.yaml

# 4. Production run on 2×H200
bash scripts/train_2xh200.sh configs/ladder/A0_llama60m.yaml
```

## Configs are the experiment unit

There are no hyperparameter CLI flags. Every experiment is a YAML file that documents
itself. The training entrypoint takes only `--config <path>` plus an optional `--seed`
and `--override key=value` escape hatch:

```bash
python -m coupled_muon_nanogpt.train --config configs/ladder/A0_llama60m.yaml --seed 0
```

`configs/base.yaml` holds defaults; ladder/smoke configs override.

## Toggleable architecture (per ladder)

The transformer block is parameterised so that bridging-ladder rungs A0 → N from
`experiment.md` are reachable by editing config alone:

| Knob | Values | Ladder rungs it controls |
|---|---|---|
| `norm.type` | `rmsnorm`, `layernorm` | D, E |
| `mlp.type` | `swiglu`, `gelu_2mat`, `relu2` | A0/A1, B, H |
| `pos_emb.type` | `rope`, `learned` | C |
| `attn.qk_norm` | bool | G, H |
| `attn.gqa_groups` | int (0 = MHA) | future |
| `moe.enabled` + `moe.*` | bool + experts/top_k/balancing | I–N |

Optimizer side:

| Knob | Values | Notes |
|---|---|---|
| `optimizer.type` | `adamw`, `muon`, `coupled_muon_v2` | three optimizers compared |
| `optimizer.coupled_steps` | int (0 = plain Muon) | d.4 ablation |
| `optimizer.couple_qk` / `couple_vo` / `couple_updown` | bool | individual coupling toggles |
| `optimizer.ns_dtype` | `bf16`, `fp32` | d.6.3 stability |

## Compute targets

| Config | Tokens | Wall-clock | Hardware |
|---|---|---|---|
| `smoke/tiny_dev` | ~10M | ~5 min | 1×4090 |
| `smoke/A0_llama60m` | ~100M | ~30 min | 1×4090 |
| `ladder/A0_llama60m` (60M-CS, all dense rungs collapse to this shape) | 1.2B | ~1 h/seed | 2×H200 |
| `ladder/B_gelu2mat` (representative dense ablation rung; LR-sweep base) | 2.5B | ~2 h/seed | 2×H200 |
| `ladder/I_moe` (~112M total / ~60M active) | 5B | ~3.5 h/seed | 2×H200 |

## Repo layout

```
configs/        base.yaml + smoke/ + ladder/
src/coupled_muon_nanogpt/
  model/        transformer, mlp, attention, moe, components
  optim/        coupled_muon (lifted from QZ), muon, factory
  data/         prepare_fineweb, loader
  probes/       svd, coupled_pair, attn_logit, ns_internal
  sweep/        expand, telescoping
  train.py · eval.py · utils.py
scripts/        smoke_4090.sh, train_2xh200.sh, prepare_data.sh, prepare_synthetic.sh, run_sweep.py
tests/          pair classification, toggle combinations, ckpt stamp
```

## Wandb

Every ladder/sweep config logs to wandb by default (`configs/base.yaml` sets
`run.wandb: true`); smoke configs override to `false`. Each run carries four
wandb dimensions, all derived from the resolved config:

| dimension | source | purpose |
|---|---|---|
| `project` | `cfg.run.wandb_project` (sweep YAMLs override per study) | Top-level study container. |
| `group` | `cfg.run.wandb_group` if set, else `${name}/${optimizer.type}/lr{lr}` | Auto-aggregates seeds-of-same-cell. |
| `job_type` | `cfg.optimizer.type` | UI optimizer-family separator. |
| `tags` | rung name, optimizer, seed, MoE flags, coupling flags, `smoke` | Filterable axes. |

Cross-config-comparable metrics (`loss`, `val_loss`, slow probes) plot against
**tokens** as the x-axis (`wandb.define_metric(..., step_metric="tokens")`).
Within-run-only metrics (`lr`, `opt_step_s`, `tok_per_s`, `ns_internal/*`) keep
the default step axis.

```bash
# Online (training nodes with internet):
torchrun --nproc_per_node=2 -m coupled_muon_nanogpt.train --config configs/ladder/A0_llama60m.yaml --seed 0

# Offline (training nodes without internet — recommended workflow):
WANDB_MODE=offline torchrun --nproc_per_node=2 -m coupled_muon_nanogpt.train --config configs/ladder/A0_llama60m.yaml --seed 0

# Sync from a machine with internet (after offline training completes):
wandb sync results/<run_id>/wandb/
```

Sweep YAMLs that vary axes the fallback group formula doesn't capture (e.g.
`pair_ablation` varies `couple_qk/vo/updown`) override `run.wandb_group` with
OmegaConf interpolation; see `configs/sweeps/pair_ablation.yaml` for the
pattern.

## References

- Design doc: `experiment.md` (top of this repo).
- Optimizer source: `/home/lyc/dev/QZ/Coupled_muon/coupled_muon_v2.py` (lifted verbatim).
- Bernstein & Newhouse, *Old Optimizer, New Norm* (arXiv 2409.20325).
- Liu et al., *Moonlight* (arXiv 2502.16982).
