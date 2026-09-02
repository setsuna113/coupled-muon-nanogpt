# Coupled Muon v2 — project summary and CV material

Companion to `FINAL_STATUS.md` (numbers) and `experiment.md` (design). This
document is the narrative: what the idea was, what was built, what was found,
and how to present it in a PhD application. Everything quantitative here is
traceable to `FINAL_STATUS.md`.

---

## 1. The idea in one paragraph

Muon orthogonalises each weight matrix's gradient independently (`NS(G) ≈ UVᵀ`).
But a transformer never *uses* `W_Q` or `W_K` alone — the forward pass only
ever forms the bilinear products `W_Q W_Kᵀ` (per head, inside RoPE's 2-D
rotation blocks), `W_V W_O` (per head) and `W_up W_down`. Coupled Muon v2 asks:
what if the preconditioner respected that? For each such pair `(A, B)` it runs
the Newton–Schulz quintic on the *product iterate* `X·B` using the partner's
current value, so the product `A·B` moves in its polar direction, then
re-orthogonalises the result so Muon's `lr·√max(rows, cols)` scaling still
applies: `U_A = NS(C_A(G_A; B))`. No published optimiser (Manifold Muon,
Gram-Space Muon, CASPR, MuonEq, PolarGrad, Polar Express) uses the partner
weight's current value this way. The honest theoretical status, worked out in
`experiment.md` Appendix X, is that this is a Stiefel-limit approximation to
steepest descent on the product — exact when `B` has orthonormal rows, which
Muon-trained weights approximately satisfy — not a proven steepest-descent
rule. The project was designed to find out *where* that approximation pays.

## 2. What the project set out to do

An earlier result (LLaMA-60M, one architecture, one seed) suggested the coupled
update helped. Rather than scale it up blindly, the project asked a
localisation question: **which architectural component makes coupling pay,
and does it survive the transition from dense LLaMA to NanoGPT-style dense,
sparse-MoE, and factored (MLA) attention?** The design was a pre-registered
"bridging ladder" that changes one axis at a time:

```
A0 LLaMA-60M (SwiGLU, RoPE, RMSNorm)
 ├─ B  GELU 2-mat → C  + learned pos-emb → D  + LayerNorm → E  Karpathy vanilla GPT-2
 └─ G  + QK-Norm  → H  + ReLU² (modded-nanogpt-like) → H′ + 50% RoPE
        └─ I′/I  sparse MoE (8 experts, top-2) → J router→AdamW, K aux-loss-free, L shared expert, M 16 experts
                └─ O  DeepSeek-V2-style MLA on I's backbone (natively factored KV)
```

Every rung ran `{AdamW, Muon, Coupled}` × 5 LRs × 3–5 seeds at a fixed 60M
"common size" shape, with decision rules written down before the data
existed: a 3σ gate to proceed past Stage 1, own-best-LR paired comparison,
and a D-gate (`Muon − Coupled > 1.5 × pooled seed std` with a stable
bootstrap CI) to commission the expensive factored-architecture follow-ups.

## 3. What was built (all by the author, May–July 2026)

* **A config-driven, toggleable transformer harness** (`src/coupled_muon_nanogpt/`,
  ~11k lines): every ladder rung is a YAML diff over `configs/base.yaml`;
  knobs for norm type, MLP family (SwiGLU / GELU-2mat / ReLU² / imposed
  low-rank FactFF), RoPE / partial RoPE / learned positions, QK-Norm, GQA,
  sparse MoE (Switch aux-loss, DeepSeek-V2 bias balancing, shared expert,
  every-other-layer placement) and DeepSeek-V2/V3-style MLA with the
  factored `(W_DKV, W_UK, W_UV, W_DQ, W_UQ, W_KR)` projections exposed by name.
* **Three optimisers behind one factory.** `optim/factory.py` walks
  `named_parameters()`, classifies every 2-D weight into coupled pairs
  (Q–K, V–O, up–down, MLA K-side, FactFF), plain-Muon or AdamW buckets,
  handles per-expert buckets in MoE, GQA head broadcasting, router-optimizer
  choice, and a `pair_policy` enum. `coupled_steps=0` reduces bitwise to
  plain Muon (`test_coupled_zero_equals_muon.py`).
* **The Phase-2.1 Q–K dispatcher.** Part I found the multi-head Q–K branch
  silently fell back to flat-2D coupling whenever RoPE was absent. The fix
  introduced a three-flag `qk_coupling` policy (`full_rope`,
  `no_rope_policy`, `partial_rope_policy`) with four implementations
  (legacy RoPE-2D-block, flat-2D, per-head 3-D, RoPE/non-RoPE channel split)
  and *bitwise-invariance tests* guaranteeing that the default path
  reproduced Phase-1 numerics exactly, so 400+ existing cells stayed valid.
* **Diagnostics as first-class probes**: SVD entropy, condition number and
  spectral norm of the coupled products `W_A W_B`, a MuonClip-style global
  max-attention-logit trace, Newton–Schulz internal norms, post-stage-1
  spectral norm (to test whether stage 2 is a no-op), MoE per-expert load /
  router entropy / per-expert gradient-norm variance, per-expert gradient SNR,
  and a LoRA-RITE-style pair-factor-ratio probe.
* **Sweep and orchestration tooling for three heterogeneous rigs**
  (2×H200, 4×H200, 8×H100): grid expansion with tied axes, deterministic
  `run_id = hash(resolved config + seed)` so any cell can be re-dispatched
  or resumed on any machine, a filesystem-gated chained orchestrator
  (`scripts/phase21_orchestrate.py`) that reads winners off `metrics.jsonl`
  and materialises the next phase's jobs (`phase21_materialize.py`), offline
  W&B with idempotent post-hoc sync, and progress/archival scripts.
* **Statistics and reporting**: paired percentile bootstrap on the median
  seed-wise difference, own-best-LR selection, dedupe of relaunched W&B
  rows on the full config axis, a parquet-snapshot → figures/tables → LaTeX
  pipeline (`docs/figures/*.py`, two reports), and a close-out status script.
* **201 unit tests** covering pair classification, toggle combinations
  (48 architecture × optimizer combos), NS convergence, coefficient tables,
  MLA forward/pair routing, partial-RoPE split, MoE bias updates, probes,
  sweep expansion, W&B init, and the Phase-2.1 materialiser.

## 4. What was found

1. **The dense gain is real, small, and lives in attention.** On every
   full-RoPE dense rung Coupled beats Muon by 0.008–0.024 nats at 60M with
   CIs excluding zero; the per-pair factorial puts ~0.010 each on Q–K and
   V–O and ~0.003 on up–down; extra plain-NS depth does not reproduce it;
   the gain is invariant to the NS coefficient policy (Bernstein / Cesista /
   Polar-Express) at K ≥ 5; one coupled inner step recovers ~75% of it.
2. **It survives sparse MoE.** Against the pre-registered expectation that
   per-expert gradient noise and gate omission would erase it, the gain
   persists on I, I′, J, K, L, M (−0.007 … −0.016 nats, no divergence) and
   is indifferent to router optimizer and to aux-loss-free balancing.
3. **Geometry matters more than "coupling".** Treating the stacked multi-head
   Q/K matrices as one flat product is actively harmful (worse than not
   coupling Q–K at all), which is what produced the Part-I "regression" on
   learned-position rungs. Coupling per head, without the RoPE pair split,
   flips those rungs to the largest Coupled-over-Muon margins in the study
   (−0.02 … −0.04 nats at matched LR), with the caveat that those runs sit in
   the attention-logit > 1000 regime and need a QK-Norm/QK-Clip control.
4. **The factored-KV hypothesis was falsified.** On the MLA rung — the one
   architecture whose weights are *natively* factored and therefore the
   place where a product-aware preconditioner "should" shine — Coupled is
   slightly *worse* than Muon (+0.003 nats, CI [+0.003, +0.005]). The
   pre-registered D-gate said stop, and the 350M / FactFF / LoRA-rank
   follow-ups were not run. In hindsight the MLA pair `(W_UK, W_DKV)` is a
   low-rank factorisation, not a forward-pass bilinear form against another
   weight, and it was coupled through exactly the flat-2D path that finding 3
   shows to be wrong.
5. **Calibration.** Muon-family beats a tuned AdamW by 0.09–0.29 nats
   everywhere (≈1.3–1.5× token efficiency, in line with the literature); the
   Coupled-over-Muon increment is ≈1.02–1.04× tokens and costs 1.04–1.12×
   wall-clock, so it is a fixed-token improvement, not a wall-clock win.

## 5. What the author learned (for the SOP / interviews)

* Designing the experiment so that a *negative* result is informative — the
  ladder localised the effect, and the D-gate stopped a 300-GPU-hour branch —
  was worth more than any single positive number.
* A silent code path (`use_multi_head` auto-disabling under non-RoPE) was
  the biggest scientific confound in the project. Fixing it as a policy
  dispatcher with bitwise regression tests, rather than a patch, is what
  made the repair auditable.
* Exclusion rules interact with hyper-parameter grids: the probe-based
  "diverged" flag, applied uniformly, excluded whole LR columns for one arm
  and not the other and manufactured a 0.06-nat regression that a matched-LR
  reading shrinks to 0.01–0.02 (`FINAL_STATUS.md` §6).
* Literature calibration first: knowing that well-tuned AdamW-vs-Muon is
  1.1–1.5× (not 2×) set the effect-size expectations and the seed budget.

## 6. CV material

### 6.1 One-paragraph description (SOP style, ~150 words)

> I led an independent study of *Coupled Muon*, an optimiser I designed that
> extends Muon's Newton–Schulz orthogonalisation from single matrices to the
> bilinear products a transformer actually computes (`W_Q W_Kᵀ`, `W_V W_O`,
> `W_up W_down`), preconditioning each factor with its partner's current
> value. To find where the idea pays, I built a config-driven NanoGPT-style
> harness with toggleable architecture (RoPE/learned positions, QK-Norm,
> SwiGLU/GELU/ReLU², sparse MoE, DeepSeek-style MLA) and ran a pre-registered
> "bridging ladder" of 960 runs (~3.4k GPU-hours, 2.75T tokens) with
> own-best-LR, paired-seed bootstrap comparisons. The coupled update beats
> Muon by 0.01–0.02 nats at 60M on RoPE-equipped dense and sparse-MoE models,
> the gain localises to per-head attention pairs, and a pre-registered gate
> falsified my hypothesis that natively factored MLA attention would benefit
> most. I also found and fixed a silent flat-2D fallback whose repair turns
> the one negative dense rung into the largest gain in the study.

### 6.2 CV entry — 3 bullets

* Designed **Coupled Muon**, a partner-aware Newton–Schulz optimiser that
  orthogonalises transformer weight *products* (`W_Q W_Kᵀ`, `W_V W_O`,
  `W_up W_down`) instead of single matrices; beats Muon by 0.01–0.02 nats on
  10 of 15 dense / sparse-MoE rungs at 60M (every RoPE-equipped dense rung and
  every MoE rung) and tuned AdamW by 0.1–0.3 nats on all of them.
* Built the evaluation stack solo: toggleable NanoGPT harness (MoE, MLA,
  partial RoPE, QK-Norm), unified optimiser factory with pair classification,
  10 training-dynamics probes, resumable multi-rig sweep orchestration,
  paired-bootstrap statistics, LaTeX report pipeline; 11k LOC, 201 tests.
* Ran a pre-registered 960-run bridging ladder (~3.4k GPU-h) that localised
  the gain to per-head attention coupling and *falsified* the factored-KV
  (MLA) hypothesis via a pre-declared decision gate; wrote two technical
  reports.

### 6.3 CV entry — 6 bullets

* Proposed and implemented Coupled Muon v2: two-stage update
  `U_A = NS(C_A(G_A; B))` that runs the Newton–Schulz quintic on the product
  iterate `X·B` so `A·B` moves in its polar direction, then re-orthogonalises;
  derived (Appendix X) that it is the Stiefel-limit approximation of steepest
  descent under `‖ΔA·B‖_op ≤ η`.
* Built a config-driven transformer research harness in PyTorch: every
  architecture axis (norm, MLP family, positional encoding, QK-Norm, GQA,
  Switch/DeepSeek MoE, DeepSeek-V2 MLA, imposed low-rank FFN) is a YAML knob;
  DDP training loop, uint16 FineWeb-Edu sharding, offline W&B with idempotent sync.
* Wrote an optimiser factory that classifies `named_parameters()` into coupled
  pairs / Muon / AdamW buckets (per-expert MoE buckets, GQA broadcasting,
  MLA factored pairs), plus a `qk_coupling` policy dispatcher with four
  Q–K geometries and bitwise-invariance tests protecting 400+ earlier cells.
* Designed and ran a pre-registered 15-rung bridging ladder
  (LLaMA-60M → Karpathy GPT-2 → modded-nanogpt → sparse MoE → MLA),
  3 optimisers × 5 LRs × 3–5 seeds, 960 runs / ~3.4k GPU-h / 2.75T tokens
  across three rigs with a filesystem-gated chained orchestrator.
* Showed the coupled gain is real but attention-side (Q–K + V–O ≈ 0.02 nats,
  MLP ≈ 0.003), invariant to NS coefficient policy, saturating at one inner
  step, and survives sparse MoE (I/I′/J/K/L/M); showed flat-2D multi-head
  coupling is harmful and per-head coupling repairs the learned-position regression.
* Applied a pre-declared D-gate to the MLA rung, obtained a clean negative
  (Coupled − Muon = +0.003 nats, CI excludes 0) and stopped the 350M/FactFF/
  LoRA-rank branch; wrote Part I/II LaTeX reports and a reproducible
  close-out analysis (`scripts/wandb_status_snapshot.py`).

### 6.4 Interview talking points

* *The idea*: precondition in the geometry the loss is computed in — the
  product, not the factor. Cheap (two extra matmuls per inner step), no new
  state, reduces bitwise to Muon.
* *The best evidence*: seven dense rungs and six MoE rungs, own-best-LR
  paired seeds, all CIs on RoPE rungs exclude zero; ablations rule out
  "more NS depth" and "better NS coefficients" as explanations.
* *The most interesting number*: flat-2D Q–K coupling is worse than no Q–K
  coupling, per-head is better than Muon by 0.02–0.04 on learned-position
  rungs. Geometry is the whole story.
* *The negative result and why I trust it*: MLA D-gate, pre-registered rule,
  5 seeds, and a mechanistic reason it failed (the coupled MLA pair is not a
  forward-pass bilinear form and went through the flat-2D path).
* *What I would do next with 100 GPU-hours*: C/D/E × {Muon, per-head coupled,
  Q–K off} × 3 LRs × 5 seeds with QK-Norm on, plus attention-logit traces, to
  turn §4.5 into a claim; then one 350M rung.
* *What I got wrong*: I let a probe-based divergence flag act as an exclusion
  rule across an LR grid; and I coupled MLA through the one code path my own
  later ablation showed to be harmful.
* *Why it matters for LLM research*: Muon-family optimisers are now in
  production (Kimi K2 / Moonlight); MuonClip exists because Q–K growth is the
  dominant instability. A Q–K-aware preconditioner is a principled sibling of
  that fix, and this study measured where such structure helps and where it
  does not.

### 6.5 Skills demonstrated

PyTorch DDP training from scratch; optimiser design and numerics
(Newton–Schulz, bf16 stability, μP-style LR scaling, weight-decay transfer);
transformer architecture internals (RoPE block structure, QK-Norm, GQA,
Switch/DeepSeek MoE routing and balancing, DeepSeek-V2 MLA); experimental
design with pre-registered decision rules; paired-bootstrap statistics;
multi-machine job orchestration and fault-tolerant sweeps; W&B at scale
(16 projects, 960 runs, offline sync); LaTeX reporting from data snapshots;
test-driven research code (201 tests, bitwise-invariance regression tests).

## 7. Attribution

Sole author of everything in this repository (50 commits, 2026-05-06 →
2026-07-16). The optimiser kernel was lifted from the author's earlier
`QZ/Coupled_muon` repository and extended here (coefficient policies,
LR-prefactor policies, `final_polish`, multi-head / partial-RoPE /
per-head Q–K dispatch, MLA and FactFF pairs). Design document
(`experiment.md`) and both reports written by the author.
