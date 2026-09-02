# Could this still become a paper? (assessment at close, 2026-09-02)

Inputs: `FINAL_STATUS.md` (what exists, including §4.7 wall-clock accounting
and §4.8 attention-logit evidence), `experiment.md` §a.2.3 / §c.7 (the
literature calibration), the measured cost anchors (A0 1.2B-token cell
0.46 h; dense 2.5B-token rung 0.75–1.3 h; MoE 5B-token rung ≈5 h; MLA
≈2.5 h; all per 2-GPU slice, so ×2 for GPU-hours), and a four-way
advocate/skeptic review of the close-out data. Assumed restart budget:
≈500 GPU-hours over about a month.

## 1. Verdict

* **Not a main-track paper (ICLR / NeurIPS / ICML), at any spend of this
  budget.** The effect is 0.008–0.024 nats at one size (60M), which in the
  field's currency is a 1–6% token saving (§2); it is wall-clock neutral
  (0.94–1.02× after netting the 4–12% runtime tax, `FINAL_STATUS.md` §4.7);
  the study sits below the smallest model in every paper it calibrates
  against; the central Phase-2 hypothesis was falsified; and there is no
  theorem (Appendix X concedes the implementation computes `polar(G_A·B)`
  where the exact answer is `polar(G_A·P_B)`). A 350M point cannot rescue
  this — at compute-optimal tokens one 12-cell rung costs ≈380–560
  GPU-hours, would land at n=3, and the literature prior is that the gap
  shrinks.
* **A mechanism paper is achievable and is the right paper.** TMLR (which
  judges support-for-claims, not novelty or significance) is the target;
  an OPT / HiLD / ES-FoMo workshop version and an immediate arXiv report are
  the milestones. Roughly 70% of it exists already; ≈250–300 GPU-hours buys
  the missing controls. Reframe from "a better optimiser" to **"what should
  a Newton–Schulz preconditioner orthogonalise in attention, and why"**.
* **Three methodology repairs are pass/fail for TMLR and cost zero
  GPU-hours** (§3, X0): (i) the n=3 "95% CIs" are arithmetically the
  min–max of three numbers — report raw seed differences plus a paired
  t / Wilcoxon; (ii) the `diverged` flag is a post-treatment attention-logit
  probe used as an exclusion rule and applied inconsistently between the two
  §4.5 readouts — make the primary analysis "all finite-loss runs" and report
  the flag rate; (iii) the §4.5 lead has no contemporaneous Muon arm on
  C/D/E (the Muon baseline is the May ladder, the coupled cells are
  May–July) and its −0.02…−0.04 figure breaks the own-best-LR lock (the
  correct paired readout is −0.011…−0.024).

## 2. The effect in the field's units

Converting each rung's Coupled − Muon delta into tokens with the Muon tail
slope (`FINAL_STATUS.md` §4.7): Muon needs 1.03–1.08× the tokens to match
Coupled on the rungs where Coupled wins; Coupled costs 1.04–1.12× the
wall-clock; net 0.94–1.02×. Expressed as a fraction of the Muon-over-AdamW
gap on the same rung (7–17%) and mapped through the literature's 1.12–1.4×
Muon-over-AdamW speedups, coupling buys 0.8–5.8% fewer tokens. Either way,
an optimiser paper whose headline is in nats invites a reviewer to do this
arithmetic in public. The one cheap engineering lead here: Newton–Schulz is
≈0.1–0.2% of model FLOPs at d=512–1024, so the measured 4–7% tax is
kernel-launch / Python-loop overhead (plain Muon itself pays +4% over
AdamW in this harness); a batched per-head NS kernel could make the
overhead <1% (X5).

## 3. The headline claim if restarted

> Per-matrix orthogonalisation is the wrong invariance for attention: what a
> Newton–Schulz preconditioner should orthogonalise is the forward-pass
> bilinear map at head granularity. Coupling that object gives a small,
> seed-robust, NS-policy-invariant gain on dense and sparse-MoE transformers
> at 60M that saturates at one inner iteration; coupling the same two
> matrices with whole-matrix geometry is worse than not coupling; coupling a
> low-rank factorisation the forward pass never forms as a product (MLA)
> does nothing, as a pre-registered gate confirmed. The gain co-occurs with
> a 12–24% lower maximum attention logit, and the decisive control — is
> coupling redundant with QK-Clip / QK-Norm / per-head blocking? — is the
> experiment this restart runs first.

Every clause is measured except the last, and the paper's value does not
depend on how that control comes out: "coupling is an implicit, cheaper
MuonClip" and "coupling has geometric content beyond logit control" are both
publishable at TMLR / a workshop.

## 4. Ranked experiments (GPU-hours = 2 × 2-GPU-cell hours)

| # | Experiment | Cells | GPU-h | What it decides |
|---|---|---|---|---|
| X0 | Zero compute: archive W&B histories (`scripts/fetch_wandb_histories.py`; retention risk), sync the 39 on-disk C1 cells, replace bootstrap CIs with raw seed diffs + paired t, iso-token / iso-wall-clock table (§4.7 done), logit-probe re-analysis (§4.8 done), erratum to `02_setup.tex`'s divergence definition | 0 (+39) | 0 | Fixes the three red cards; ≈70% of the paper |
| X1 | **Mechanism gate at A0**: {Muon + QK-Norm, Muon + QK-Clip(τ), Muon per-head-blocked NS (no coupling), Coupled + QK-Norm} × 2 LRs × 5 seeds, plus ≈10 drift cells re-running existing arms | ≈50 | ≈48 | Is coupling redundant with logit control, or with per-head blocking alone? QK-Clip is ≈50 lines (only the probe threshold exists today) |
| X2 | qk-repair under a stability control: D × {Muon, `headwise_no_rope`, `qk_off`} × 3 LRs × 5 seeds, plus an E replication × 2 LRs × 3 seeds; QK-Norm on, contemporaneous Muon, no exclusions | ≈63 | ≈115 | Whether the §4.5 lead is real or a flag / LR-window / cross-project artefact |
| X3 | Token-budget axis at A0: {0.6, 1.2, 2.4, 4.8}B × {Muon, Coupled} × 3 seeds, schedules resized per budget | 24 | ≈41 | Does the gain decay with training length (the only affordable answer to "vanishes with scale")? |
| X4 | AdamW grid extended past its boundary (A0's optimum sits at the grid edge 3e-3) + batch {0.25M, 1M} × {Muon, Coupled} | ≈22 | ≈25 | Removes the two tuning confounds a referee will use first |
| X5 | Batch/fuse the per-head NS kernels, then benchmark A0 and H | ≈12 | ≈15 | Whether an efficiency claim can exist at all |
| — | Contingency: flagged reruns, seed top-ups (H′, A0 to n=8) | — | ≈55 | Stage 2 lost a third of its cells to flags at the top two LRs |
| X6 | 350M dense rung at 7B tokens (`Z_350m_dense.yaml`, ≈36 GPU-h/cell) | 12 | 380–560 | **Do not run** on this budget: a null is fatal, a positive at 350M is inside the regime where the literature already reports ≈1.3× for every matrix optimiser |

Committed: ≈245 GPU-hours plus ≈55 contingency ≈ 300; ≈200 left unspent.

## 5. Go / no-go rules

* **G0 (day 3, no compute).** If net iso-wall-clock is ≥ +5% on at least
  two rungs under both slope conversions, keep an efficiency claim;
  otherwise delete every efficiency claim and retitle to the geometry
  question. Expect the latter (§4.7 says 0.94–1.02×).
* **G1 (after X1).** Coupled minus the best of {Muon + QK-Norm, Muon +
  QK-Clip, Muon per-head} ≤ −0.010 nats at own-best LR, n=5, paired-t CI
  excluding zero → coupling has content beyond logit control; proceed to
  X2. Margin in (−0.010, 0) → publish "coupling is an implicit QK-Clip" and
  skip X2's second rung. Margin ≥ 0 → the algorithmic contribution is dead;
  write the geometry / negative-result paper on existing data and stop at
  ≈90 GPU-hours.
* **G2 (after X2).** Require D *and* E, QK-Norm on, no exclusions,
  in-project Muon, n=5, own-best LR, ≤ −0.010 with CI excluding zero and no
  divergence-rate penalty → the §4.5 lead becomes the headline. Prior ≈50%:
  H′ is the only rung with an in-project Muon arm *and* a stability control,
  and it returned −0.003 [−0.004, +0.001].
* **G3 (after X3).** Monotone decay from 0.6B to 4.8B tokens at fixed 60M →
  buy no scale rung; report the decay as the result.
* **Hard refusal.** No 350M rung and no reopening of the MLA gate on this
  budget; both are one-sided bets that would consume ≥40% of it.

## 6. If not restarted

Ship anyway; the paper is written, not run.

1. Do X0 regardless — the histories are the one asset that can disappear.
2. Merge Part I, Part II and `FINAL_STATUS.md` into one arXiv report
   (≈3 weeks of writing) with the geometry framing, the honest 1–6% effect
   size, the wall-clock-neutral accounting, and the five projects never
   written up (K-curve, NS policy, qk-policy, pair-policy, MLA).
3. Submit the same document to TMLR (rolling) and to the nearest
   non-archival optimisation workshop.
4. Correct Part I in public: the C/D "regression" was mostly a
   flag-exclusion artefact plus a flat-2D Q–K geometry error, not a
   RoPE-dependence.
5. Release the harness, the 916-row snapshot, the histories and the 201
   tests. The pre-registered gate that fired negative and was obeyed is the
   package's strongest asset.

For a PhD application, that report plus this repository is worth more than
a speculative workshop submission: it shows the design, the rigour, and the
willingness to publish a negative result.
