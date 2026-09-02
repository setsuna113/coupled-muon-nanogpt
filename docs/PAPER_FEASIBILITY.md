# Could this still become a paper? (assessment at close, 2026-09-02)

Inputs: `FINAL_STATUS.md` (what exists), `experiment.md` §a.2.3 / §c.7 (the
literature calibration), and the measured cost anchors (A0 1.2B-token cell
0.46 h; dense 2.5B-token rung 0.75–1.3 h; MoE 5B-token rung ≈5 h; MLA ≈2.5 h;
all per 2-GPU slice, so ×2 for GPU-hours).

## 1. Verdict

* **As it stands, no main-track paper.** The effect size (0.01–0.02 nats,
  ≈1.02–1.04× tokens) is inside the range reviewers now discount; there is a
  single model size (60M); there is no wall-clock win (1.04–1.12× slower);
  and the one architecture where the method "should" shine (MLA) came out
  negative. Any of these alone is survivable; together they read as "a
  careful negative-ish result".
* **A workshop / TMLR-style paper is realistic with ≈100–150 GPU-hours**, if
  the headline is reframed from "a better optimiser" to **"which weight
  products should a Newton–Schulz preconditioner respect, and why"** — a
  localisation study whose most striking finding is that *geometry* decides
  the sign: flat-2D Q–K coupling hurts, per-head coupling helps
  (`FINAL_STATUS.md` §4.5), and factored-KV coupling does nothing (§4.6).
* **With ≈500 GPU-hours** a main-track submission becomes *arguable* but not
  safe: it would need one ≥350M point with μP-transferred hyper-parameters
  and a mechanistic story tying the gain to attention-logit control. The
  "speedup vanishes with scale" prior (Wen et al. 2025; Qiu et al. 2025)
  means the 350M point is a coin flip, and a null there ends the paper.

## 2. The headline claim if restarted

> Preconditioning the *products* an attention head computes — not the
> factors — is what a partner-aware Newton–Schulz step buys. Per-head Q–K
> and V–O coupling delivers a small, seed-robust gain over Muon on dense
> and sparse-MoE transformers at 60M and closes the learned-position gap
> that flat-2D coupling opens; coupling a low-rank factorisation (MLA KV)
> instead of a forward-pass bilinear form does not help. The gain arrives
> with a single inner iteration and is invariant to Newton–Schulz
> coefficient policy, consistent with a stability (attention-logit)
> mechanism rather than a curvature one.

Everything in that paragraph is already measured except the two clauses
that carry the paper: the "closes the gap" clause needs the probe-flagged
runs re-run under a stability control, and the "stability mechanism" clause
needs a direct comparison against QK-Clip / QK-Norm.

## 3. Minimal experiment set

| # | Experiment | Cells | GPU-h | What it decides |
|---|---|---|---|---|
| 1 | C/D/E × {Muon, Coupled `headwise_no_rope`, Coupled `qk_off`} × 3 LRs × 5 seeds, **QK-Norm on**, attention-logit trace on | 135 | ≈200 (or ≈70 at 3 seeds) | Turns §4.5 into a claim; removes the probe-flag confound |
| 2 | A0 × {Muon + QK-Clip(τ=50), Coupled, Coupled + QK-Clip} × 2 LRs × 5 seeds | 30 | ≈30 | Whether coupling is a prophylactic for the same instability QK-Clip fixes (mechanism) |
| 3 | Sync the 39 on-disk C1 pair-policy cells (no compute) | 0 | 0 | Completes the MoE localisation table |
| 4 | One 350M dense rung (A0 shape scaled, μP wd) × {Muon, Coupled} × 2 LRs × 3 seeds | 12 | ≈220 | The scale point a main-track reviewer will demand |
| 5 | MLA with per-head coupling of the up-projections (algorithmic change: couple `(W_UK_h, W_DKV)` per head or 3-way UK/UV/DKV) × 5 seeds | 10 | ≈50 | Whether the MLA null is geometry (fixable) or fundamental |

Workshop/TMLR path: 1 (at 3 seeds) + 2 + 3 ≈ 100 GPU-h. Main-track path:
1 + 2 + 3 + 4 (+ 5) ≈ 450–500 GPU-h.

## 4. Go / no-go rule

Run experiment 1 first (≈70 GPU-h at 3 seeds). **Go** if, with QK-Norm on,
per-head Coupled beats Muon at own-best LR on at least two of C/D/E with
CIs excluding zero and no divergence-rate penalty; **no-go** otherwise —
in which case the honest paper is a negative/localisation tech report and
the remaining budget is better spent elsewhere.

## 5. If not restarted

The existing material is already a coherent, citable artefact: Part I
(dense ladder), Part II (sparse MoE) and `FINAL_STATUS.md` (Phase 2.1). The
cheapest high-value action is a single arXiv tech report that merges the
three, states the MLA null and the flat-2D/per-head finding plainly, and
releases the harness. For a PhD application that report plus the
repository is worth more than a speculative workshop submission: it shows
the design, the rigour, and the willingness to publish a negative result.
