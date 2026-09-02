# Coupled Muon v2 — project summary and CV material

Companion to `FINAL_STATUS.md` (numbers) and `experiment.md` (design). This
document is the narrative: what the idea was, what was built, what was found,
and how to present it in a PhD application. Every number here traces to
`FINAL_STATUS.md`; conventions are its (final validation loss in nats on
FineWeb-Edu, each optimizer at its own best LR, seed-paired arms, 95%
percentile-bootstrap CI on the median seed-wise difference, negative
Coupled − Muon means Coupled wins).

---

## 1. The idea in one paragraph

Muon orthogonalises each gradient matrix on its own, but attention never uses
`W_Q` or `W_K` alone — the logits are `x (W_Q W_Kᵀ) xᵀ`, a head's value path
is `W_V W_O`, and a two-matrix MLP is `W_up W_down`. Coupled Muon v2
preconditions the *product*: for each such pair `(A, B)` it runs the
Newton–Schulz quintic on the iterate `X·B`, using the partner's current value,
so that the product moves in its polar direction, then re-orthogonalises `X`
so Muon's `lr·√max(rows, cols)` scaling still applies. The update is
`U_A = NS(C_A(G_A; B))`; with `coupled_steps = 0` it reproduces plain Muon
(to 1e-4 by test). No published optimiser surveyed (Manifold Muon, Gram-Space
Muon, CASPR, MuonEq, PolarGrad, Polar Express) uses the partner weight's
current value this way. The honest theoretical status, derived in
`experiment.md` Appendix X before any run: the exact solution of
"minimise ⟨G_A, Δ_A⟩ subject to ‖Δ_A B‖_op ≤ η" is `polar(G_A · P_B)` with the
row-space projector `P_B = B⁺B`, not `polar(G_A · B)`; the two coincide only
when `B` has orthonormal rows, which Muon-trained weights approximately are.
So the method is a Stiefel-limit approximation to steepest descent on the
product, chosen because the rigorous version needs a `(BBᵀ)⁻¹` solve that
would break the matmul-only design. The project was built to find out *where*
that approximation pays, not just *whether*.

## 2. What the project set out to do

The prior evidence was one dense LLaMA-60M run. Rather than scale it up, the
project asked a localisation question — **which architectural component does
the gain need, and does it survive the move to NanoGPT-style dense,
sparse-MoE and factored (MLA) attention?** — with a pre-registered bridging
ladder that changes one axis per rung:

```
A0 LLaMA-60M (SwiGLU, RoPE, RMSNorm)
 ├─ B  GELU 2-mat → C  + learned pos-emb → D  + LayerNorm → E  Karpathy vanilla GPT-2
 └─ G  + QK-Norm  → H  + ReLU² (modded-nanogpt-like) → H′ + 50% RoPE
        └─ I′/I  sparse MoE (8 experts, top-2) → J router→AdamW, K aux-loss-free, L shared expert, M 16 experts
                └─ O  DeepSeek-V2-style MLA on I's backbone (natively factored KV)
```

Every rung ran `{AdamW, Muon, Coupled}` over a 5-point LR grid with 3–5 seeds
at one 60M "common size" shape. The decision rules were written before the
data existed: a 3σ gate to proceed past Stage 1, own-best-LR paired
comparison, and a D-gate (`Muon − Coupled > 1.5 × pooled seed std` with a
sign-stable bootstrap CI) to commission the factored-architecture follow-ups.

## 3. What was built (sole author, 2026-05-06 → 07-16)

* **Config-driven, toggleable transformer harness** (`src/coupled_muon_nanogpt/`,
  ~11k lines of Python): every rung is a YAML diff over `configs/base.yaml`;
  knobs for norm type, MLP family (SwiGLU / GELU-2mat / ReLU² / imposed
  low-rank FactFF), RoPE / partial RoPE / learned positions, QK-Norm, GQA,
  sparse MoE (Switch aux-loss, DeepSeek-V2 bias balancing, shared expert,
  every-other-layer placement) and DeepSeek-V2/V3-style MLA with the factored
  projections exposed by name.
* **Three optimisers behind one factory.** `optim/factory.py` walks
  `named_parameters()` and classifies every 2-D weight into coupled pairs
  (Q–K, V–O, up–down, MLA K-side, FactFF), plain-Muon or AdamW buckets,
  handling per-expert MoE buckets, GQA head broadcasting, router-optimizer
  choice and a `pair_policy` enum.
* **The Phase-2.1 Q–K dispatcher.** Part I found that the multi-head Q–K
  branch silently fell back to flat-2D coupling whenever RoPE was absent. The
  repair replaced warn-and-disable with a three-flag `qk_coupling` policy
  (`full_rope`, `no_rope_policy`, `partial_rope_policy`) and four Q–K
  geometries (legacy RoPE-2D-block, flat-2D, per-head 3-D, RoPE/non-RoPE
  channel split), guarded by regression tests that hold the full-RoPE update
  exactly equal and the legacy fallback equal to 1e-6 — so the 400+ earlier
  cells stayed valid.
* **Diagnostics as probes**: weight SVD entropy, condition number and
  spectral norm of the coupled products, a MuonClip-style global
  max-attention-logit trace, Newton–Schulz internal norms, post-stage-1
  spectral norm, MoE expert load / router entropy / per-expert grad-norm
  variance, per-expert gradient SNR, and a LoRA-RITE-style pair-factor ratio.
* **Sweep and orchestration tooling for three heterogeneous rigs** (2×H200,
  4×H200, 8×H100): grid expansion with tied axes, deterministic
  `run_id = hash(resolved config + seed)` so any cell can be re-dispatched or
  resumed on any machine, a filesystem-gated chained orchestrator that reads
  phase winners off `metrics.jsonl` and materialises the next phase's jobs,
  offline W&B with idempotent post-hoc sync, progress and archival scripts.
* **Statistics and reporting**: paired percentile bootstrap on the median
  seed-wise difference, own-best-LR selection, dedupe of relaunched W&B rows
  on the full config axis (960 → 916), a parquet-snapshot → figures/tables →
  LaTeX pipeline (two reports), and a close-out script that regenerates every
  number in `FINAL_STATUS.md` from a committed snapshot.
* **201 unit tests**: pair classification, 48 architecture × optimizer toggle
  combinations, NS convergence, coefficient tables, MLA forward/pair routing,
  partial-RoPE split, MoE bias updates, probes, sweep expansion, W&B init,
  the Phase-2.1 materialiser, and the invariance guards above.

## 4. What was found

1. **The dense gain is real, small, and lives in attention.** On every
   full-RoPE dense rung (A0, B, G, H) Coupled beats Muon by 0.008–0.024 nats
   with CIs excluding zero. The per-pair factorial at A0 puts ≈0.010 on Q–K
   alone, ≈0.010 on V–O alone, ≈0.003 on up–down alone, 0.015 on Q–K+V–O and
   0.018 on all three; giving Muon more NS depth (`ns_steps=9`) does not move
   it; the gap is −0.016 to −0.019 under all three NS coefficient policies
   (Bernstein / Cesista / Polar-Express) at K ≥ 5; and one coupled inner step
   recovers 75–80% of it, with the K-curve flat for K ≥ 3.
2. **It survives sparse MoE.** Against the pre-registered expectation that
   per-expert gradient noise and gate omission would erase it, the gain
   persists on I, I′, J, K, L (−0.009 to −0.016 nats, CIs exclude zero; five
   seeds on I/I′) and on the two-seed M rung (−0.007, n=1 pair), with no
   divergence. It is indifferent to the router optimizer (J) and to
   aux-loss-free balancing (K). The MoE pair-policy sweep at I mirrors the
   dense factorial: all pairs > attention-only > FFN-only, each step with a
   CI excluding zero.
3. **Geometry decides the sign.** Part I reported a +0.055/+0.066 regression
   on the learned-position rungs C and D. The Phase-2.1 sweep showed that
   headline to be mostly an LR-window artefact of the divergence-flag
   exclusion rule (§6 of `FINAL_STATUS.md`); the residual, matched-LR gap of
   the legacy flat-2D Q–K path is 0.01–0.06 nats, and flat-2D coupling is
   worse than not coupling Q–K at all (D at lr 1e-2: 3.435 vs 3.408). Coupling
   per head, without the RoPE pair split, flips all three learned-position
   rungs to a Coupled-over-Muon win of the same order as the full-RoPE rungs
   (own-best-LR paired medians −0.011 to −0.024 nats) — *provided*
   probe-flagged runs are admitted; n=3 and no QK-Norm, so this is the top
   open lead, not a claim. The upshot is that Part I's "the gain needs RoPE"
   reading was wrong: the gain needs the per-head geometry, and the legacy
   code lost it without RoPE. On the partial-RoPE rung H′ (no flagged runs),
   splitting RoPE'd from non-RoPE channels is the right policy but the margin
   is at the edge of resolution (−0.003, CI [−0.004, +0.001], n=5).
4. **The factored-KV hypothesis was falsified.** On the MLA rung — the one
   architecture whose weights are *natively* factored and the motivation for
   the whole Phase-2 tree — Coupled is slightly *worse* than Muon (+0.0034
   nats, CI [+0.0028, +0.0047], n=4 paired seeds, one seed crashed). The
   pre-registered D-gate said stop, and the 72 gated cells (paired
   LoRA-rank, imposed-FactFF, 350M MLA) were never run. The post-hoc reading
   is coherent, not a rescue: the coupled MLA pair `(W_UK, W_DKV)` is a
   low-rank factorisation, not a forward-pass bilinear form against another
   weight; `W_UV` shares `W_DKV` but cannot be coupled in the same step; and
   the pair was routed through the flat-2D path that finding 3 shows to be
   harmful. The (unpaired) kv-rank sweep is monotone in rank (3.359 → 3.222
   from rank 16 to 256), which says MLA at 60M wants more KV rank and says
   nothing about coupling.
5. **Calibration, and what the mechanism probably is.** Coupled beats a
   tuned AdamW by 0.09–0.29 nats on every rung with an AdamW arm (Muon by
   0.08–0.25), i.e. the usual 1.3–1.5× token-efficiency reference for the
   Muon family. The Coupled-over-Muon increment converts to 1.03–1.08× tokens
   and costs 1.04–1.12× wall-clock — a fixed cost, not proportional to the
   inner step count — so netted out it is 0.94–1.02× on every rung: a
   mechanism-level fixed-token result, not a speedup, and nothing was
   validated above 60M. The probes say what the mechanism likely is: at A0
   Coupled's maximum attention logit runs 12–24% below Muon's (and below
   AdamW's) all through training, which with the K=1 saturation, the
   NS-policy invariance and the flat-2D harm points at Q–K coupling acting as
   an implicit attention-logit stabiliser — the same instability MuonClip
   fixes post hoc. The decisive control (Muon + QK-Clip vs Coupled) was never
   run.

## 5. What the author learned (for the SOP / interviews)

* Building the ladder before believing the result was the highest-leverage
  decision: it converted "does this optimizer work?" into "which component
  does it need?", which is answerable at 60M.
* Writing the decision rules down in advance is what made a negative result
  cheap — nothing was spent on the 72 cells the D-gate would have authorised.
* A silent code path (`use_multi_head` auto-disabling under non-RoPE) was
  the biggest confound in the project; fixing it as a policy dispatcher with
  exact-equality regression tests, rather than a patch, made the repair auditable.
* An exclusion rule can be simultaneously conservative and misleading: the
  probe-based `diverged` flag (max attention logit > 1000, not a loss NaN;
  158/158 flagged dense runs finished with finite loss) removed whole LR
  columns for one arm and not the other. Both readouts have to be reported.
* Literature calibration first: knowing that well-tuned AdamW-vs-Muon is
  1.1–1.5× (not 2×) set the effect-size expectations and the seed budget.

## 6. CV material

Plain-text versions (no code font, no Unicode operators) are given for the
CV blocks so they survive application portals.

### 6.1 One-paragraph description (SOP style, ~150 words)

> I designed Coupled Muon, an optimiser that extends Muon's Newton–Schulz
> orthogonalisation from single weight matrices to the bilinear products a
> transformer actually computes (Q–K, V–O, up–down), preconditioning each
> factor with its partner's current value. To learn *where* the idea pays
> rather than just *whether*, I built a config-driven NanoGPT-style harness
> in which architecture is a toggle (RoPE or learned positions, QK-Norm,
> SwiGLU/GELU/ReLU², sparse MoE, DeepSeek-style MLA) and ran a pre-registered
> bridging ladder of 960 runs (≈3.4k GPU-hours, 2.75T tokens) with
> own-best-LR, seed-paired bootstrap comparisons. The gain is real, small and
> conditional: 0.008–0.024 nats over Muon on full-RoPE dense rungs, surviving
> sparse MoE, and localised to per-head attention pairs. My central
> hypothesis — that natively factored MLA attention would benefit most —
> failed its pre-registered gate, and I closed that branch rather than keep
> it alive. The most promising lead I left is a per-head Q–K coupling that
> repairs the method on non-RoPE attention, reported as unconfirmed.

### 6.2 CV entry — 3 bullets (plain text)

* Designed Coupled Muon, a Newton–Schulz optimiser that orthogonalises paired
  weight products (Q–K, V–O, up–down) instead of single matrices; beats Muon
  by 0.008–0.024 nats at 60M on full-RoPE dense and sparse-MoE transformers.
* Built the evaluation stack solo (11k lines, 201 tests): toggleable NanoGPT
  harness with MoE and MLA, three optimisers behind one pair-classifying
  factory, training-dynamics probes, resumable three-rig sweep orchestration.
* Ran a pre-registered 960-run ladder (3.4k GPU-h) that localised the gain to
  per-head attention coupling and falsified the factored-KV (MLA) hypothesis
  under a pre-declared gate; wrote two technical reports.

### 6.3 CV entry — 6 bullets (plain text)

* Proposed a two-stage partner-aware update: Newton–Schulz on the product
  iterate so the pair moves in its polar direction, then re-orthogonalisation;
  derived it as the Stiefel-limit approximation of steepest descent under a
  product spectral-norm constraint.
* Built a config-driven transformer research harness in PyTorch: norm, MLP
  family, positional encoding, QK-Norm, GQA, Switch/DeepSeek MoE and
  DeepSeek-V2 MLA as YAML knobs; DDP training loop, FineWeb-Edu sharding,
  offline W&B with idempotent sync.
* Wrote an optimiser factory that classifies parameters into coupled pairs,
  Muon and AdamW buckets (per-expert MoE, GQA broadcasting, MLA pairs), plus a
  Q–K policy dispatcher with four coupling geometries and exact-equality
  regression tests protecting 400+ earlier cells.
* Ran a pre-registered 15-rung bridging ladder (LLaMA-60M to GPT-2 to
  modded-nanogpt to sparse MoE to MLA), 3 optimisers × 5 LRs × 3–5 seeds:
  960 runs, 3.4k GPU-hours, 2.75T tokens across three rigs with a
  filesystem-gated chained orchestrator.
* Showed the gain is attention-side (Q–K and V–O ≈0.010 each, MLP ≈0.003),
  invariant to NS coefficient policy, 75–80% recovered by one inner step,
  and preserved on six sparse-MoE rungs; showed flat-2D multi-head Q–K
  coupling is harmful and per-head coupling repairs it.
* Applied a pre-declared decision gate to the MLA rung, obtained a clean
  negative (Coupled − Muon = +0.003 nats, CI excludes zero) and stopped the
  gated 350M / FactFF / paired-LoRA follow-ups; wrote Part I/II reports and a
  reproducible close-out analysis.

### 6.4 Interview talking points

* *The idea.* Precondition in the geometry the loss is computed in — the
  product, not the factor. Two stages: stage 1 picks the geometry, stage 2
  enforces the scale. No new optimiser state; reduces to Muon at K = 0.
* *Where the theory is soft, and that I wrote it down first.* The exact
  constrained solution is `polar(G_A · P_B)` with the row-space projector,
  not `polar(G_A · B)`; the two coincide when the partner has orthonormal
  rows. The algorithm is a Stiefel-limit approximation that avoids a
  `(BBᵀ)⁻¹` solve. Appendix X predates the first run.
* *The best evidence.* Four full-RoPE dense rungs and five MoE rungs with
  CIs excluding zero at own-best LR; ablations rule out "more NS depth" and
  "better NS coefficients"; the MoE pair-policy sweep reproduces the dense
  factorial ordering.
* *The most interesting number.* Flat-2D Q–K coupling is worse than no Q–K
  coupling; per-head coupling beats Muon by 0.01–0.02 nats at own-best LR on
  the learned-position rungs, which undoes Part I's "needs RoPE" story.
  Geometry is the whole story — and I have not claimed it, because those runs
  sit above the attention-logit threshold and need a QK-Norm or QK-Clip
  control (≈120–160 GPU-hours at 3 seeds).
* *The cheapest decisive experiment.* Muon + QK-Clip vs Coupled at A0, ≈40
  GPU-hours: if they match, coupling is an implicit MuonClip and the paper
  says so; if Coupled still wins, the geometric claim survives its hardest
  control. Worth running whichever way it comes out.
* *The negative result and why I trust it.* MLA D-gate, pre-registered rule,
  four paired seeds at 5B tokens, and a mechanistic reason it failed: the
  coupled MLA pair is a low-rank factorisation, not a forward-pass bilinear
  form, and it went through the flat-2D path. The hypothesis was falsifiable,
  and it got falsified.
* *Cost, stated against my own interest.* 1.04–1.12× Muon's wall-clock for a
  1.03–1.08× token-budget gain, i.e. wall-clock neutral; the overhead is a
  fixed cost, so the K=1 result simplifies the algorithm but does not make it
  cheaper.
* *Scale.* Everything is 60M. The literature says matrix-optimiser speedups
  shrink with scale; one 350M rung at 7B tokens (≈430 GPU-hours, no AdamW
  anchor) is the first thing a reviewer would ask for, and I would run it
  only after the mechanism control.
* *What I got wrong.* I let a probe-based divergence flag act as an
  exclusion rule across an LR grid, which manufactured most of a 0.06-nat
  regression; and I coupled MLA through the one code path my own later
  ablation showed to be harmful.
* *Why it matters.* Muon-family optimisers are in production (Moonlight,
  Kimi K2), and MuonClip exists because Q–K growth is the dominant
  instability. A Q–K-aware preconditioner is a principled sibling of that
  fix; this study measured where such structure helps and where it does not.

### 6.5 Skills demonstrated

PyTorch DDP training from scratch; optimiser design and numerics
(Newton–Schulz polar iteration, bf16 stability, Moonlight spectral LR
scaling, 1/width weight-decay transfer); transformer internals (RoPE block
structure, QK-Norm, GQA, Switch/DeepSeek MoE routing and balancing,
DeepSeek-V2 MLA); experimental design with pre-registered decision rules;
paired-bootstrap statistics; multi-machine job orchestration and
fault-tolerant sweeps; W&B at scale (16 projects, 960 runs, offline sync);
LaTeX reporting from data snapshots; test-driven research code (201 tests,
exact-equality regression guards).

## 7. Attribution

Sole author of everything in this repository (50 commits, 2026-05-06 →
2026-07-16, plus the 2026-09-02 close-out). The optimiser kernel was lifted
from the author's earlier `QZ/Coupled_muon` repository and extended here
(coefficient and LR-prefactor policies, `final_polish`, multi-head /
partial-RoPE / per-head Q–K dispatch, MLA and FactFF pairs). Design document
and both reports written by the author.
