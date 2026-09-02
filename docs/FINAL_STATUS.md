# Coupled Muon v2 — project status at close (2026-09-02)

This is the close-out record for the project. It supersedes the Phase-2.1
"live" tables in `DISPATCH.md` and the 2026-07-16 snapshot in `docs/RESULTS.md`.
Every number below is regenerated from the W&B summary snapshot committed at
`docs/data/wandb_all_runs_2026-09-02.parquet` by

```
python scripts/wandb_status_snapshot.py --analyze docs/data/wandb_all_runs_2026-09-02.parquet
```

Conventions are the report's (Part I §2): final validation loss in nats on
FineWeb-Edu; each optimizer read at its **own best LR** per rung; arms paired
by seed; 95% percentile-bootstrap CI on the **median seed-wise difference**;
negative Coupled − Muon means Coupled wins. Cells flagged `diverged` are
excluded from the protocol readout (see §6 for what that flag actually is).

---

## 1. TL;DR

* **Compute footprint.** 960 W&B runs across 16 projects, ≈1.7k 2-GPU-cell
  hours (≈3.4k GPU-hours on H100/H200), ≈2.75T tokens trained, 2026-05-10 →
  2026-07-13. 50 commits, ~11k lines of Python, 201 passing unit tests.
* **Dense result (Stage 1+2) stands.** Coupled Muon beats per-matrix Muon by
  0.008–0.024 nats at 60M on every full-RoPE rung (A0/B/G/H, CIs exclude 0)
  and beats a tuned AdamW by 0.12–0.29 nats everywhere.
* **Sparse-MoE result (Stage 3) stands and is now complete.** The gain
  survives on every MoE rung that ran (I, I′, J, K, L, M: −0.007 to −0.016
  nats), and the AdamW@I anchor is now 5-seed complete (Coupled − AdamW
  −0.091 nats).
* **Phase 2.1 mostly closed the gaps the reports left open.** K-curve, NS
  policy, pair-policy, qk-policy repair and the MLA rung all ran.
* **The central Phase-2 hypothesis failed the D-gate.** On the natively
  factored MLA rung (O_mla), Coupled Muon is *not* better than Muon
  (+0.0034 nats, CI [+0.0028, +0.0047], n=4). D3/D4/E1 were therefore never
  commissioned, correctly per the pre-registered decision rule.
* **The Part-I "RoPE-dependence" story does not survive Phase 2.1.** The C/D
  regression was mostly an LR-window artefact of the divergence-flag
  exclusion rule plus a genuinely harmful flat-2D Q–K geometry; a per-head
  Q–K coupling (`headwise_no_rope`) removes it and gives the usual small win
  on the learned-position rungs too (own-best-LR paired medians −0.011 to
  −0.024 nats, n=3, probe-flagged runs admitted; §4.5, §6). It needs a
  QK-Norm or QK-Clip control before it can be claimed.
* **It is not a wall-clock win.** Converting each rung's loss gain into the
  extra tokens Muon would need, and netting against Coupled's measured
  runtime tax, gives 0.94–1.02× on every rung (§4.7). The overhead is a fixed
  cost of forming the products and the second NS pass, not proportional to K.
* **The mechanism evidence points at attention-logit control.** At A0's best
  LR Coupled's global max pre-softmax logit runs 12–24% below Muon's (and
  below AdamW's) throughout training (§4.8), consistent with the K=1 result,
  the NS-policy invariance and the flag rates on C/D/E.

## 2. Evidence inventory (W&B, entity `liuyc1025-university-of-cambridge`)

| Project | Runs | Plan step | Complete? | Written up? |
|---|---|---|---|---|
| `coupled-muon-A0-repro` | 75 | Stage 1 A0 | ✅ 75/75 | Part I §3 |
| `coupled-muon-ladder` | 270 | Stage 2 B/C/D/E/G/H | ✅ 270/270 | Part I §4 |
| `coupled-muon-ablation` | 106 | §d.4 ablations | ✅ | Part I App. B |
| `coupled-muon-moe-pt1-prod` | 60 | Stage 3 I, I′ (5 seeds) | ✅ | Part II |
| `coupled-muon-moe-pt2-screen` | 60 | Stage 3 J/K/L/M (3 seeds) | ✅ (N never run) | Part II (stale: had 34) |
| `coupled-muon-moe-anchor-adamw` | 45 | A1 AdamW@I anchor | ✅ 5 seeds × 5 LRs | Part II (stale: had 15 cells) |
| `coupled-muon-moe-anchor-adamw-Iprime` | 25 | AdamW@I′ anchor | ✅ | Part II |
| `coupled-muon-ns-policy` | 36 | A2 NS-coefficient policy | 33/36 final | Part II (partial) |
| `coupled-muon-k-curve` | 28 | A3 K-curve @A0, K∈{1,2,3,4,5,8,12} | ✅ | **not written up** |
| `coupled-muon-k-curve-I` | 9 | A3 K-curve @I, K∈{1,2,4} | ✅ | **not written up** |
| `coupled-muon-qk-policy` | 144 | B1 qk-policy sweep C/D/E/H′ + H′ Muon | ✅ 144/144 | **not written up** |
| `coupled-muon-qk-policy-confirm` | 15 | B2 5-seed confirmation | ✅ 15/15 | **not written up** |
| `coupled-muon-pair-policy` | 15 | C1 pair_policy @I, single LR, 5 seeds | ✅ | **not written up** |
| `coupled-muon-pair-policy-lr` | 6 | C1 pair_policy × LR | 5/6 (of 45 on disk) | **not written up** |
| `coupled-muon-mla` | 51 | D1 O_mla triple, 3 opts × 3 LRs × 5 seeds | 44/45 usable (+6 half-batch reruns) | **not written up** |
| `coupled-muon-lora-rank` | 15 | MLA kv-rank sweep (Coupled only) | ✅ | **not written up** |

Never launched (gated on a positive D-gate, which was negative): D3 paired
LoRA-rank (30), D4 imposed-FactFF (12), E1 350M MLA (30), Z_350m_dense.
Never launched (deprioritised): N (4-expert MoE), A2 LR-prefactor sweep, A2
NS top-up. On disk but not synced: the remaining ~39 cells of the 45-cell
C1 pair-policy × LR grid (`$GLOBAL/phase2.1-rigB-c-pair-policy/`).

Deduplication: 960 → 916 rows (44 relaunch duplicates, mostly in `ablation`
and `moe-anchor-adamw`). W&B `state=="crashed"`: 6.

## 3. Dense ladder (Stage 1 + 2), protocol readout

| Rung (arch) | Coupled best | Muon best | AdamW best | Coupled − Muon [95% CI] (n) | Coupled − AdamW |
|---|---|---|---|---|---|
| A0 LLaMA-60M, SwiGLU+RoPE (1.2B tok, 5 seeds) | 3.4065 @1e-2 | 3.4308 @1e-2 | 3.7016 @3e-3 | **−0.024** [−0.026, −0.023] (3) | −0.293 |
| B  GELU 2-mat (2.5B tok, 3 seeds) | 3.3048 @3e-3 | 3.3124 @3e-3 | 3.4244 @3e-3 | **−0.008** [−0.009, −0.006] (3) | −0.118 |
| C  + learned pos | 3.4259 @1e-3 | 3.3618 @1e-2 | 3.5485 @1e-3 | **+0.066** [+0.058, +0.074] (2) | −0.123 |
| D  + LayerNorm | 3.4260 @1e-3 | 3.3729 @1e-2 | 3.5494 @1e-3 | **+0.055** [+0.048, +0.062] (2) | −0.120 |
| E  Karpathy vanilla | 3.4260 @1e-3 | 3.4281 @1e-3 | 3.5492 @1e-3 | −0.002 [−0.003, +0.001] (3) | −0.120 |
| G  A0 + QK-Norm | 3.2580 @3e-2 | 3.2686 @1e-2 | 3.4113 @3e-3 | **−0.011** [−0.012, −0.010] (3) | −0.153 |
| H  G + ReLU² (modded-nanogpt-like) | 3.2464 @3e-2 | 3.2616 @1e-2 | 3.3899 @3e-3 | **−0.014** [−0.016, −0.013] (3) | −0.141 |

Unchanged from Part I. Coupled needs ~1.04–1.07× Muon's wall-clock per cell.

§d.4 ablations at A0 (lr 3e-3, `coupled-muon-ablation`; Part I App. B):
`coupled_steps=0` reproduces Muon (3.432); K=1 already gives 3.417; stage-2
polish is load-bearing (`final_polish=False` → ≈3.55 with 2/3 seeds flagged);
per-pair factorial: all-off ≈ Muon 3.432, Q–K alone and V–O alone each
recover ≈0.010, up–down alone ≈0.003, Q–K+V–O 3.417, all three 3.414; Muon
with `ns_steps=9` stays at 3.432 (extra NS depth alone does not close the gap).

## 4. Everything that ran after the reports

### 4.1 Sparse MoE (Stage 3) — now complete at 5 seeds for I/I′ and AdamW@I

| Rung | Coupled best (n) | Muon best (n) | AdamW best (n) | Coupled − Muon [CI] | Coupled − AdamW [CI] |
|---|---|---|---|---|---|
| I  SwiGLU MoE 8×top-2 (5B tok) | 3.0565 @1e-2 (5) | 3.0680 @1e-2 (5) | 3.1477 @3e-3 (5) | **−0.0105** [−0.0121, −0.0080] | **−0.091** [−0.094, −0.085] |
| I′ ReLU² MoE | 3.0496 @3e-2 (5) | 3.0648 @1e-2 (5) | 3.1539 @1e-3 (5) | **−0.0155** [−0.0168, −0.0104] | −0.103 [−0.273, −0.100] |
| J  router→AdamW (3 seeds) | 3.0559 @1e-2 | 3.0679 @1e-2 | — | **−0.0104** [−0.0120, −0.0090] | — |
| K  aux-loss-free (DeepSeek bias) | 3.0630 @1e-2 | 3.0730 @3e-3 | — | **−0.0103** [−0.0106, −0.0097] | — |
| L  shared expert | 3.1631 @1e-2 | 3.1718 @1e-2 | — | **−0.0086** [−0.0097, −0.0069] | — |
| M  16 experts (2 seeds) | 3.0701 @3e-2 (1) | 3.0782 @3e-2 (2) | — | −0.0073 (n=1) | — |

No MoE cell diverged. J ≈ I says the router optimizer does not matter for
the coupled gain; K keeps the gain but has worse absolute loss and a
different routing regime (Part II); L is worse in absolute terms for both
optimizers. The Coupled/Muon runtime ratio on MoE is 1.03–1.12×.

### 4.2 A3 — coupled-steps K-curve (`coupled-muon-k-curve`, `-I`)

A0, lr 3e-3, vs plain Muon (n=5 reference seeds at 3.432):

| K | 1 | 2 | 3 | 4 | 5 | 8 | 12 |
|---|---|---|---|---|---|---|---|
| Coupled median | 3.4174 | 3.4175 | 3.4143 | 3.4146 | 3.4131 | 3.4142 | 3.4143 |
| Δ vs Muon | −0.014 | −0.015 | −0.019 | −0.017 | −0.020 | −0.018 | −0.017 |

I (MoE), lr 1e-2, vs Muon 3.0680: K=1 −0.0086, K=2 −0.0097, K=4 −0.0107.
**One coupled inner step recovers ~75–80% of the gain; the curve is flat for
K ≥ 3.** The default K=4 is not special. Runtime does *not* follow K: at A0
Muon takes 1586 s per cell, Coupled with K=0 1592 s, K=1 1642–1656 s (+4%),
K=4 1665–1673 s, K=12 1717 s (+8%). The overhead is a fixed cost (forming
the products, the stage-2 NS pass), so K=1 does not buy the runtime back.

### 4.3 A2 — Newton–Schulz coefficient policy (`coupled-muon-ns-policy`, A0, lr 3e-3, 2 seeds)

| Policy | K=3 Coupled/Muon | K=5 Coupled/Muon | K=8 Coupled/Muon |
|---|---|---|---|
| Bernstein | 3.414 / 3.467 (−0.053) | 3.413 / 3.432 (−0.019) | 3.417 / 3.434 (−0.016) |
| Cesista | 3.417 / 3.457 (−0.040) | 3.412 / 3.431 (−0.018) | 3.413 / 3.432 (−0.019, n=1) |
| Polar-Express | 3.424 / 3.449 (−0.025) | 3.413 / 3.431 (−0.019) | — / 3.432 |

At K ≥ 5 the Coupled − Muon gap is −0.016 to −0.019 nats regardless of the
coefficient policy; better NS coefficients do not close it. (K=3 gaps are
inflated by Muon being under-iterated.)

### 4.4 C1 — MoE pair-policy localisation at I

5 seeds, lr 1e-2, 5B tokens (`coupled-muon-pair-policy`; note this project's
absolute losses are ~0.11 higher than `pt1-prod`, consistent with a different
data/eval shard — compare within the project only):

| pair_policy | median | paired vs `all` |
|---|---|---|
| all (Q–K, V–O, up–down) | 3.1735 | — |
| attention_only | 3.1778 | +0.0052 [+0.0042, +0.0064] |
| ffn_only | 3.1821 | +0.0090 [+0.0073, +0.0107] |

The single-seed LR grid (`-lr` project) agrees: all 3.0559 < ffn_only 3.0628 <
attention_only 3.0651 < Muon 3.0680. **On MoE the composite is best and the
attention pairs carry more than the FFN pair**, mirroring the dense A0
factorial. 39 of the 45 planned cells exist on disk but were never synced.

### 4.5 B1/B2 — the qk-coupling repair on non-RoPE rungs

Part I attributed the C/D regression to `factory.py` silently demoting the
multi-head Q–K branch to flat-2D coupling when RoPE is absent. Phase 2.1
added a three-flag `qk_coupling` dispatcher and swept the non-RoPE policies.

**Protocol readout (probe-flagged runs excluded):** on C/D/E every Coupled
policy except `headwise_no_rope` is flagged 3/3 at lr ≥ 3e-3 and so is read at
lr 1e-3 (≈3.426, identical to the legacy fallback — `current_flat2d_fallback`
reproduces the old path to 1e-6 by test, and `qk_off` is indistinguishable
from it in outcome).
`headwise_no_rope` survives at 3e-3–1e-2 in 1–3 of 3 seeds and reads
C 3.353 (n=1), D 3.356 (n=1), E 3.354 (n=2), i.e. **below the ladder Muon best
(C 3.362, D 3.373, E 3.428)**. B2 5-seed confirmation at the B1 winner
(`headwise_no_rope`; C @3e-3, D @1e-2): C 3.356 (2/5 unflagged), D 3.355
(3/5 unflagged).

**Readout including probe-flagged rows with finite loss** (all n=3, medians):

| Rung | lr | Muon (ladder) | legacy flat-2D | `qk_off` | `headwise_no_rope` |
|---|---|---|---|---|---|
| C | 3e-3 | 3.392 | 3.400 | 3.399 | **3.358** |
| C | 1e-2 | 3.371 | 3.396 | 3.391 | **3.369** |
| D | 3e-3 | 3.400 | 3.404 | 3.406 | **3.368** |
| D | 1e-2 | 3.376 | 3.435 | 3.408 | **3.357** |
| E | 3e-3 | 3.400 | 3.408 | 3.417 | **3.366** |
| E | 1e-2 | 3.393 | 3.407 | 3.406 | **3.355** |

Reading the same rows at each arm's **own best LR** (flag ignored, finite
loss only, paired by seed, n=3):

| Rung | Muon best | legacy flat-2D | `qk_off` | `headwise_no_rope` | B2 5-seed headwise |
|---|---|---|---|---|---|
| C | 3.3713 @1e-2 | +0.012 (3.3945) | +0.021 (3.3909) | **−0.011** (3.3579 @3e-3) | −0.012 (3.3553) |
| D | 3.3762 @1e-2 | +0.032 (3.4019) | +0.032 (3.4060) | **−0.013** (3.3565 @1e-2) | −0.015 (3.3561) |
| E | 3.3926 @1e-2 | +0.014 (3.4049) | −0.001 (3.4060) | **−0.024** (3.3546 @1e-2) | — |

(paired median differences; medians in parentheses). Under the protocol
readout, unflagged rows only, B2 gives C −0.006 (n=2 vs 2) and D −0.018
(n=3 vs 2).

Three things follow. (i) The Part-I "+0.055/+0.066 regression" is mostly an
LR-window artefact of the flag-exclusion rule: at own-best LR with the flag
ignored, the legacy flat-2D path trails Muon by 0.012–0.032 and `qk_off` by
0.00–0.03. (ii) Flat-2D Q–K coupling is harmful relative to not coupling Q–K
at all on D (3.435 vs 3.408 @1e-2) and no better on C/E — the coupling must
respect the per-head bilinear structure. (iii) Per-head coupling without the
RoPE pair-split beats Muon by 0.011–0.024 nats (paired medians) on all three
learned-position rungs, the same order as the full-RoPE rungs; the −0.03 to
−0.04 numbers in the matched-LR table above compare against Muon at a
non-best LR and overstate it. So the correct statement is not "the gain is
RoPE-dependent" but "the gain is per-head-geometry-dependent, and the
legacy code got the geometry wrong without RoPE". Caveats: 60M, no QK-Norm,
max attention logit above 1000 on most of these runs (§6), n=3; on D the
flagged seeds are worse than the unflagged ones (3.372 vs 3.355), so the
unflagged readout there is partly selection on outcome.

**H′ (H with 50% RoPE, modded-nanogpt-faithful).** No cell flagged. Muon
best 3.2519 @1e-2 (n=5). Coupled policies @1e-2: flat-2D 3.2542, `qk_off`
3.2534, `headwise_no_rope` 3.2524, **`partial_rope_split` 3.2499**. B2
5-seed at `partial_rope_split`: 3.2500, Coupled − Muon **−0.0031 [−0.0039,
+0.0006]** (n=5). Splitting the RoPE'd channels (legacy 2-D pair coupling)
from the non-RoPE channels (plain Muon) is the right policy, but the margin
on partial-RoPE attention is at the edge of resolution.

### 4.6 D1 — MLA rung and the D-gate (`coupled-muon-mla`)

O_mla = rung I's MoE backbone with DeepSeek-V2-style MLA (kv rank 64, q rank
64, 32 nope + 32 rope dims), coupled pairs (W_UK, W_DKV) and (W_UQ, W_DQ),
W_UV on plain Muon. 3 optimizers × {3e-3, 1e-2, 3e-2} × 5 seeds, 5B tokens.
Six seeds were re-run at half global batch (9536 steps; loss ≈3.14–3.21) and
are excluded from pairing.

| Optimizer | best LR | median (n) | at lr 3e-3 | at lr 3e-2 |
|---|---|---|---|---|
| Coupled | 1e-2 | 3.2594 (4) | 3.2683 (5) | 6.83 (all 5 blew up, unflagged) |
| Muon | 1e-2 | 3.2560 (5) | 3.2668 (4) | 6.91 (all 5) |
| AdamW | 1e-2 | 3.3605 (4) | 3.365 (4) | diverged 5/5 |

Coupled − Muon at own-best: **+0.0034 [+0.0028, +0.0047]** (n=4); at 3e-3
+0.0015 [−0.0002, +0.0031]. `delta = Muon − Coupled = −0.0034 < 1.5 ×
pooled_std = 0.0023` ⇒ **D-gate NEGATIVE**. Coupled − AdamW −0.101.
The kv-rank sweep (Coupled only, `coupled-muon-lora-rank`) is monotone:
rank 16/32/64/128/256 → 3.359/3.303/3.258/3.237/3.222 — it says MLA at 60M
wants more KV rank, nothing about coupling.

Interpretation: on MHA the coupled pairs are (Q,K), (V,O) — products the
forward pass actually forms. On MLA the coupled pair (W_UK, W_DKV) is a
low-rank factorisation whose product is *not* a forward-pass bilinear form
against another weight, W_UV shares W_DKV but cannot be coupled in the same
step, and the `use_multi_head=false` flat-2D path is exactly the geometry
§4.5 shows to be harmful. The "natural optimizer for factored KV" framing
was falsified at this scale; the experiment did its job.

### 4.7 Wall-clock accounting

Fixed-token loss deltas are not speedups. For each rung, fit the Muon
best-LR validation curve's tail slope `dL/d ln(tokens)` (last 40% of tokens,
median over seeds, from `docs/data/*_history_bestlr_curves.parquet`),
convert the Coupled − Muon delta into the token multiple Muon would need to
reach Coupled's loss, and net it against the measured runtime ratio at each
arm's best LR (`scripts/wallclock_accounting.py`):

| Rung | Coupled − Muon | Muon tail slope | Muon tokens to match | runtime tax | **net wall-clock** |
|---|---|---|---|---|---|
| A0 | −0.024 | −0.32 | 1.078× | 1.057× | **1.02×** |
| B | −0.008 | −0.26 | 1.029× | 1.058× | 0.97× |
| C (legacy) | +0.064 | −0.26 | 0.781× | 1.036× | 0.75× |
| D (legacy) | +0.053 | −0.22 | 0.789× | 1.045× | 0.76× |
| E | −0.002 | −0.27 | 1.008× | 1.045× | 0.96× |
| G | −0.011 | −0.24 | 1.044× | 1.043× | 1.00× |
| H | −0.015 | −0.25 | 1.064× | 1.067× | 1.00× |
| I | −0.012 | −0.28 | 1.042× | 1.044× | 1.00× |
| I′ | −0.015 | −0.28 | 1.056× | 1.118× | 0.94× |

Cosine decay steepens the tail slope, so the token multiples are
conservative (an upper bound on Coupled's token advantage is roughly 2× these
excess fractions, which would still leave every rung except A0 within ±6%).
**At its own operating point Coupled Muon is wall-clock neutral against
Muon.** The honest framing of the result is therefore a fixed-token,
mechanism-level finding, not an efficiency claim.

### 4.8 Attention-logit evidence for the mechanism

The probe `probe/attn_logit/global_max` (max pre-softmax logit over all
layers/heads) at A0's best LR, median over seeds
(`docs/data/coupled_muon_A0_repro_history_bestlr_probes.parquet`):

| tokens | AdamW @3e-3 | Muon @1e-2 | Coupled @1e-2 |
|---|---|---|---|
| 0.23B | 362 | 370 | 210 |
| 0.45B | 372 | 366 | 296 |
| 0.66B | 276 | 336 | 290 |
| 0.88B | 244 | 288 | 229 |
| 1.10B | 220 | 248 | 202 |

Coupled's maximum attention logit sits 12–24% below Muon's for the whole run
and below AdamW's for most of it — the direction MuonClip enforces by
post-hoc rescaling. On the QK-Norm MoE rungs (I, I′) the max logit is 14–24
for every optimizer, i.e. the instability is already removed there. Taken
with the K=1 result (§4.2), the NS-policy invariance (§4.3), the harm of
flat-2D Q–K geometry (§4.5) and the flag rates on C/D/E (§6), the
most economical explanation is that Q–K coupling acts as an implicit
attention-logit stabiliser. The decisive control — Muon + QK-Clip vs
Coupled at A0 — was never run (§7).

## 5. Timeline

| Window | Work |
|---|---|
| 2026-05-06 → 05-14 | Harness built; Stage 1 A0 (75) + Stage 2 ladder (270) + §d.4 ablations (106) on 2×H200 + 8×H100 |
| 05-17 → 05-20 | NS-policy (36); Stage 3 MoE pt1 (60), pt2 screen, AdamW anchors; Part I + Part II reports (2026-05-19 snapshot) |
| 05-21 → 05-25 | Phase 2.1 design (gated tree), qk_coupling dispatcher, three-machine chained orchestrator; B1 C/D/E start |
| 07-04 → 07-13 | Everything else: B1 H′, B2, A1 top-up, A3 K-curves, C1, D1 MLA, kv-rank; last run 2026-07-13 |
| 07-16 | `docs/RESULTS.md` written; compute lost thereafter |

## 6. Caveat: what `diverged` means in this codebase

`wandb_utils.probe_indicates_divergence` sets `diverged=True` when the
attention-logit probe's global max exceeds 1000 (the MuonClip threshold) or
the NS-internal probe sees a non-finite norm. It is **not** a loss-NaN flag:
of 158 flagged dense runs, 94 finished with final loss < 4.0 and 158/158 have
a finite final loss. On the three learned-position, no-QK-Norm rungs
(C/D/E) essentially every Muon-family run at lr ≥ 3e-3 trips the logit
threshold — Muon included — while still converging. The reports' protocol
excludes those cells, which is conservative but interacts badly with the
LR grid on C/D (§4.5). A re-analysis that (a) treats the flag as a
diagnostic and (b) re-runs C/D/E with QK-Norm or QK-Clip is the first thing
to do if the project restarts.

## 7. Open leads, ranked by value per GPU-hour

1. **Why coupling helps: stability vs geometry** (§4.8). A0 × {Muon,
   Muon + QK-Clip(τ≈50–1000), Coupled, Coupled + QK-Clip} × 2–3 LRs × 5 seeds
   with logit traces: ~40 cells × 0.46 h × 2 GPUs ≈ 40 GPU-hours. QK-Clip is
   not implemented (only the probe threshold exists) — ~50 lines. If
   Coupled ≈ Muon + QK-Clip, the mechanism is settled; if Coupled still wins,
   the geometric claim survives its hardest control. Publishable either way.
2. **Per-head Q–K coupling on non-RoPE attention** (§4.5). C/D/E × {Muon,
   `headwise_no_rope`, `qk_off`} × 3 LRs with QK-Norm on: 81 cells at 3 seeds
   ≈ 120–160 GPU-hours, 135 cells at 5 seeds ≈ 200–270. Pre-register the
   readout as "all finite-loss runs, flag rate reported separately".
3. **Scale.** Everything is 60M-CS. One 350M dense rung at 7B tokens
   (`Z_350m_dense.yaml`, ~9 h/cell on 4×H200 ⇒ ~36 GPU-h/cell): Muon vs
   Coupled × 2 LRs × 3 seeds ≈ 430 GPU-hours, with no AdamW anchor. The
   literature prior is that the gap shrinks; a null there ends the paper.
4. **MLA done right**: 3-way (UK, UV, DKV) coupling or `headwise` coupling
   of the per-head up-projections — an algorithmic change, not a rerun.
5. Sync the 39 on-disk C1 cells; run the N rung (4 experts).

## 8. Reproducing this document

```
pip install wandb pandas pyarrow numpy
WANDB_API_KEY=… python scripts/wandb_status_snapshot.py --fetch docs/data/wandb_all_runs_<date>.parquet
python scripts/wandb_status_snapshot.py --analyze docs/data/wandb_all_runs_<date>.parquet
```

`docs/figures/fetch_all.py` / `fetch_part2.py` + `make_all.py` /
`make_part2.py` still rebuild the LaTeX reports from their own (2026-05-19)
parquet snapshots; they have not been refreshed with the Phase-2.1 projects.
