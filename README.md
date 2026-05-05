# coupled-muon-nanogpt

Research scaffold for the **Coupled Muon v2** bridging-experiment ladder defined in
`experiment.md`. Migrates a partner-weight-aware Newton–Schulz optimizer from a
LLaMA-60M baseline onto a toggleable, NanoGPT-style harness so the source of its gain
can be localized via architectural ablations.

## Quickstart

```bash
# 1. Install (uv recommended)
uv sync --all-extras

# 2. Tokenize FineWeb-Edu into uint16 shards (~30 min, one-time)
bash scripts/prepare_data.sh

# 3. Smoke test on 1×4090 (5 min)
bash scripts/smoke_4090.sh configs/smoke/tiny_dev.yaml

# 4. Production run on 4×H200
bash scripts/train_4xh200.sh configs/ladder/A0_llama60m.yaml
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
| `ladder/A0_llama60m` | 1.2B | ~30 min/seed | 4×H200 |
| `ladder/A1_llama125m` | 2.5B | ~1.5 h/seed | 4×H200 |
| `ladder/B_gelu2mat_125m` | 2.5B | ~1.5 h/seed | 4×H200 |
| `ladder/I_moe_500m` | 5B | ~8 h/seed | 4×H200 |

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
scripts/        smoke_4090.sh, train_4xh200.sh, prepare_data.sh
tests/          pair classification, toggle combinations, ckpt stamp
```

## References

- Design doc: `experiment.md` (top of this repo).
- Optimizer source: `/home/lyc/dev/QZ/Coupled_muon/coupled_muon_v2.py` (lifted verbatim).
- Bernstein & Newhouse, *Old Optimizer, New Norm* (arXiv 2409.20325).
- Liu et al., *Moonlight* (arXiv 2502.16982).
