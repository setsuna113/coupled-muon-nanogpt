# Coupled Muon v2 → NanoGPT Migration: Experiment Design Report

**TL;DR**
- Use **a stripped, single-file fork of `KellerJordan/modded-nanogpt`** taken at an early-Oct-2024 commit (RoPE + RMSNorm + ReLU² + QK-Norm, before the value-embedding/U-net-skip stack accumulated) as the dense baseline; for MoE, fork the Cerebras `train_gpt_moe.py` extension. Karpathy's vanilla `nanoGPT` is a useful *intermediate* rung, but is too far from your LLaMA baseline to be the primary harness.
- Coupled Muon v2's "preprocess gradient with the *partner weight*" is genuinely novel — no public optimizer applies a Newton–Schulz iterate `C(G_A; B)` using the *current value* of the partner W_B. The closest analogues — Manifold Muon (Bernstein, Thinking Machines, 2025), Gram-Space Manifold Muon (Tilde, 2025), CASPR-without-accumulation (which Cesista shows equals Muon), MuonEq's two-sided RC equilibration (arXiv 2603.28254), and PolarGrad (Lau et al., 2025) — all operate per-matrix. Treat the LLaMA-60M result as a real but small-scale, single-architecture signal until it survives the bridging ladder below.
- The MoE failure most likely lives in **two places at once**: (i) Moonlight Lemma 1 — Muon's update RMS is √(1/max(A,B)), so per-expert matrices and the dense MLP get *qualitatively different* effective updates; coupling further amplifies noise on small expert matrices fed by a 25%-of-batch effective sample size; and (ii) the SwiGLU `(down, up)` coupling **ignores the gate** that multiplies up in the forward pass, an approximation that worsens when routing makes `silu(x·gate)` token-skewed and sparse. Run the ladder in section (d) before drawing any conclusion about coupling itself.

---

## (a) Mathematical understanding of coupled-Muon-style updates

### a.1 What standard Muon computes

Muon's Newton–Schulz iteration with quintic coefficients `(a, b, c) = (3.4445, −4.7750, 2.0315)` approximates the **orthogonal factor of the polar decomposition** of a momentum-buffered gradient `G = U Σ Vᵀ`, returning approximately `U Vᵀ` — equivalently, the matrix sign function applied through repeated `X ← a·X + b·(X Xᵀ) X + c·(X Xᵀ)² X`. Bernstein & Newhouse ("Old Optimizer, New Norm", arXiv 2409.20325, 2024; "Modular Duality", arXiv 2410.21265, 2024) prove that the **exact polar limit `U Vᵀ` is the steepest-descent solution** to `argmin_{Δ} <G, Δ> + (λ/2) ‖Δ‖²_op = -(1/λ) trace(Σ) · U Vᵀ` under the spectral (RMS→RMS) operator norm, hence the `lr × √max(rows, cols)` scaling, which yields width-stable optimisation under μP.

The **practical iterate is an approximation of the limit**, not the limit itself. The (3.4445, −4.7750, 2.0315) quintic was deliberately tuned by Keller Jordan (2024) to converge into a singular-value band of roughly [0.7, 1.3] rather than to exactly 1 — trading asymptotic exactness at σ≈1 for a steeper slope at σ≈0 and faster convergence (because for SGD, getting the small singular directions right matters far more than nailing σ=1 exactly). So the few-step iterate is `U S' Vᵀ` with `S'_ii ∈ [≈0.5, ≈1.5]`, not `U Vᵀ`. The proven-steepest-descent claim is about the limit object; the practical algorithm is its few-step approximation.

Two consequences matter for what *coupling* does on top:
1. **Spectral whitening.** Every singular value of the update is forced to ≈ 1 (within the [0.5, 1.5] band), so updates are rotationally near-uniform and approximately condition-number-1. This is why Muon raises SVD entropy of weights vs AdamW — Moonlight (Liu et al., arXiv 2502.16982, Figure 4) measured directly that "for over 90% of the weight matrices, the SVD entropy when optimized by Muon is higher than that of AdamW".
2. **Norm balance.** Since `‖U Vᵀ‖_spec ≈ 1`, the *direction* and *magnitude* are decoupled — magnitude is approximately determined by `lr × √max(A,B)`; direction is the singular-subspace pair of G alone.

### a.2 What "couple with the *partner weight*" computes

The Coupled Muon v2 inner step,
`X ← G/‖GB‖_F; repeat: W = X B; T = b·W Wᵀ + c·(W Wᵀ)²; X ← a·X + T·X`,
is the Newton–Schulz quintic *applied to the product W = X B*, with the iterate written in terms of `X ≡ ∇A`. After convergence, `X B → polar(G_A B)` — i.e. the iterate returns a search direction for A that, when *multiplied through B*, orthogonalises the composed transformation `A B`. The symmetric branch `C_B(G_B; A)` does the dual on the right factor.

Geometrically, the forward pass uses *products* (`O V_h` per head, `K Qᵀ` per head with RoPE, `down · up` in MLP). Standard Muon orthogonalises each factor in isolation. Coupled Muon's stage-1 inner step targets the **product** under the spectral norm of the composed operator — a partner-aware preconditioner motivated by the bilinear structure of attention and the linear-product structure of 2-mat MLPs.

#### a.2.1 The actual two-stage update

The implementation in `src/coupled_muon_nanogpt/optim/coupled_muon.py` does **not** stop after the coupled inner step. The actual Coupled Muon v2 update is:

> **`U_A = NS(C_A(G_A; B))`**

— a **two-stage preconditioner**:

1. **Stage 1 (coupled NS, `C_A`)** finds an iterate X with `X B → polar(G_A B)`, i.e. it picks a search direction *informed by the partner weight* such that the product `A · B` moves in its polar direction. The output X has no spectral-norm guarantee on itself — `‖X‖_spec` is whatever it needs to be to make `X B` orthogonal, and depends on the conditioning of B.
2. **Stage 2 (plain NS)** orthogonalises X itself, so `‖U_A‖_spec ≈ 1`. This is what makes the matrix-size LR scaling `lr × 0.2 × √max(rows, cols)` meaningful — that scaling assumes a unit-spectral-norm update. Without stage 2, update magnitude per parameter would depend on the partner weight's conditioning, which is uncontrolled and drifts during training.

The two stages are not the same operation; dropping either one breaks something. **Stage 1 picks the geometry; stage 2 enforces the scale.** Whether stage 2 is doing meaningful work or is approximately a no-op (because stage-1's output happens to be near-orthogonal at the operating point) is an *empirical* question we resolve via the `final_polish` ablation (d.4) and the post-stage-1 `‖X‖_spec` probe.

#### a.2.2 Caveat — this is not a proven steepest descent

The earlier framing "steepest descent for A on the variety {AB : A ∈ ℝ^{m×k}, B fixed}" is **heuristic, not derived** by anyone in the 2024–2026 literature we surveyed (OONN, Modular Duality, Modular Manifolds, Manifold Muon ADMM, Stiefel Muon, Polar Express, Moonlight, MuonClip). The proper constrained problem `argmin <G_A, Δ_A> s.t. ‖Δ_A B‖_op ≤ η` resolves (Appendix X) to `Δ_A B = -η · polar(G_A · P_B)`, where `P_B = B^+ B` is the row-space projector — *not* `polar(G_A · B)`. The two coincide only when B has orthonormal rows. The architectural argument in a.3 (the bilinear products are what the forward pass uses) remains a strong motivation for orthogonalising the product, but it is not the same thing as a steepest-descent derivation, and the experimental ladder is what tests whether the heuristic delivers in practice.

#### a.2.3 Relationship to other partner-aware / paired optimizers

There is no published 2024–2026 optimizer that does exactly the two-stage `Muon ∘ C_A(·; B)` with B = current partner weight:

- **Manifold Muon / Modular Manifolds** (Bernstein, Thinking Machines blog, 2025) — steepest descent under spectral norm restricted to the Stiefel manifold of *one* matrix. Bernstein explicitly poses the open question: *"What manifolds should attention heads live on? … We can mix-and-match constraints in different parts of the network."* Coupled Muon v2 is a concrete, non-Stiefel, **paired** answer to that question.
- **Gram-Space Manifold Muon** (Tilde Research, 2025) — uses the Gram matrix `Wᵀ W` of *the same* W to relax the Stiefel constraint.
- **CASPR / Shampoo two-sided preconditioning** — uses gradient-statistic preconditioners `L_t G_t R_t` from gradient outer products, *not* a partner weight; Cesista (2025) shows CASPR-without-accumulation = Muon.
- **MuonEq** (arXiv 2603.28254) — "two-sided RC normalisation" on row/column norms of the **same** gradient; "two-sided" in the literature does *not* mean paired matrices.
- **PolarGrad** (Lau et al., May 2025; arXiv 2505.21799) — operates on each matrix's gradient via polar decomposition; single-matrix.

### a.3 Why Q–K, V–O, up–down coupling can be principled

In the forward pass, attention pre-softmax logits per head are `(x W_Q)(x W_K)ᵀ = x (W_Q W_Kᵀ) xᵀ`; what matters operationally is the bilinear product **W_Q W_Kᵀ**. Likewise, the value-output composition for one head is `softmax(·) · x W_V W_O^h^T` — only **W_V W_O** matters at the head level. The architectural argument is therefore: orthogonalising these *products* under the spectral norm captures the geometry that actually appears in the forward pass, which standard Muon misses by orthogonalising each factor independently. Stage 1 of Coupled Muon v2 targets exactly these products. **This is the strongest theoretical motivation for the algorithm**, even though, per a.2.2, it is not a proven steepest descent.

For 2-mat MLPs (GELU 2-mat or ReLU² 2-mat), `(down, up)` is the *only* learnable composition (no gate); coupling captures the linear-product geometry, but the activation function still inserts an input-dependent diagonal Jacobian — coupling is "linear-exact", not "Jacobian-exact". So even on Karpathy's GELU 2-mat or modded-nanogpt's ReLU² 2-mat, coupling is still an approximation, just a tighter one than on SwiGLU.

For SwiGLU, additionally, the gate enters multiplicatively as a *learnable second matrix*: `(silu(x W_gate) ⊙ (x W_up)) W_down`, where the gate-modulated up matrix is what `down` actually composes with. The user's code couples only `(down, up)` and applies Muon-without-coupling to gate. In the Jacobian of the MLP w.r.t. W_up, the relevant operator is `W_down ⊙ silu(x W_gate)` — the gate enters multiplicatively. Coupling `(down, up)` while ignoring gate is therefore a **first-order approximation that drops the gating diagonal**, valid only when `silu(x W_gate)` has roughly uniform variance across channels. This is plausible at small dense scale; in MoE, where each expert is rarely activated and gate statistics are skewed, this approximation is likely much worse.

### a.4 RoPE and the Q–K coupling

RoPE injects a position-dependent block-diagonal rotation between W_Q and W_K. Per token-pair (m, n) the effective matrix is `W_Qᵀ R_{m−n} W_K`. The user's block reshape (`head_dim/2` two-channel blocks) suggests the algorithm orthogonalises within RoPE's 2-D rotation blocks, which is the right move — RoPE acts trivially on each 2-D block's plane direction but rotates across positions. **However:** if any layer uses partial RoPE (modded-nanogpt applies RoPE to 50% of head dims), the block reshape must match. Verify in code before any cross-architecture comparison.

### a.5 Connections to cite

- Muon = steepest descent under RMS→RMS spectral norm (Bernstein–Newhouse 2024, "Old Optimizer, New Norm", arXiv 2409.20325).
- Newton–Schulz quintic ≈ matrix sign / polar factor (classical, Higham; refined by Su 2024; analysed in Polar Express, arXiv 2505.16932).
- Shampoo / SOAP: Kronecker-factored second-order; Muon = "instantaneous Shampoo" (Anil et al.; Cesista's "CASPR-without-accumulation is Muon" note).
- **MuonClip / QK-Clip** (Kimi K2, Moonshot, arXiv 2507.20534, July 2025): demonstrates that Q–K **coupled growth** is the dominant Muon instability mode at scale — they fix it by post-step rescaling of W_Q and W_K once max pre-softmax logits exceed τ. Coupled Muon v2's Q–K branch is in some sense a *prophylactic* analogue.
- AdaMuon (arXiv 2507.11005), NorMuon (arXiv 2510.05491), Muon-NSR / Muon-VS, MuonEq, Polar Express, Turbo-Muon (arXiv 2512.04632): all post- or pre-orthogonalisation rescaling — orthogonal to coupling.
- "Practical Efficiency of Muon for Pretraining" (Essential AI, Shah et al., arXiv 2505.02222, May 2025): provides "the first empirical demonstration of μP used to calibrate hyperparameters for a large language model using Muon"; reports "Muon requires 10–15% fewer tokens than AdamW to reach an identical loss".
- "Hyperparameter Transfer Enables Consistent Gains of Matrix-Preconditioned Optimizers Across Scales" (Bergsma et al., arXiv 2512.05620, NeurIPS 2025): "Combining μP with 1/width independent weight decay, Muon and Shampoo achieve consistent 1.4× and 1.3× speedups, respectively, over well-tuned AdamW in training 190M to 1.4B-parameter models on FineWeb"; speedup "vanishes rapidly with scale under incorrect scaling".

### a.6 Best guess at why coupling helped on dense LLaMA-60M

(a) The bilinear products `W_Q W_Kᵀ`, `W_V W_O`, `W_up W_down` are the operationally meaningful objects; orthogonalising them rather than each factor separately is a sharper inductive bias, equivalent to cheap second-order curvature *along the product variety*. (b) At 60M params, the dense MLP has a single 4d×d composition where coupling matches the forward pass exactly; the gate term is small enough that ignoring it doesn't materially hurt. (c) Coupling raises update SVD entropy on attention-pair products, plausibly reducing "rich-get-richer" alignment across heads — the same effect Moonlight observed for vanilla Muon vs AdamW, but stronger on the products that actually drive attention.

---

## (b) Why dense LLaMA-60M differs from dense NanoGPT

### b.1 Architectural differences

| Component | LLaMA (your baseline) | Karpathy `nanoGPT` | Modded `nanoGPT` (Jordan, current head) |
|---|---|---|---|
| Norm | RMSNorm | LayerNorm + bias | RMSNorm + QK-Norm |
| MLP | SwiGLU (gate, up, down; intermediate ≈ 8/3·d) | GELU 2-mat (c_fc, c_proj; intermediate = 4·d) | ReLU² 2-mat |
| Attention | RoPE, optional GQA, no bias | Learned pos. emb., bias=True | Partial RoPE on 50 % head dims, sliding-window (SSSL pattern), QK-Norm, value embeddings, U-net skip, smear gate |
| Optimizer (default) | AdamW or Muon | AdamW | Muon (hidden) + AdamW (embed/head) — fundamental |
| Other | none | dropout 0.0–0.2 | logit soft-cap, YaRN window warmup, scalar gates per layer, attention residuals |

### b.2 Where this bites coupled Muon

1. **MLP arity.** Coupled Muon couples `(down, up)`. In Karpathy's GELU 2-mat MLP this is the *only learnable* composition — coupling is "linear-exact" (modulo the activation Jacobian; see a.3). In LLaMA's SwiGLU it is an additional approximation that drops the gate Jacobian. In modded-nanogpt with ReLU², MLP is again 2-mat — coupling captures the linear product cleanly, again modulo activation Jacobian. So if you migrate naively to dense NanoGPT (either flavour), you're coupling a **different** approximation than the one that worked on LLaMA. Insist on running rung B (LLaMA + GELU 2-mat) before blaming MoE.
2. **QK-Norm.** Modded-nanogpt applies RMSNorm to Q and K **before** attention — this changes what `W_Q W_Kᵀ` does in the forward pass (effective bilinear becomes `(norm(W_Q x))ᵀ (norm(W_K x))`). The Q–K coupling assumption that the gradient should orthogonalise `W_Q W_Kᵀ` is partially invalidated, because that product is followed by row-wise normalisation. Strong reason to *expect* less coupled-Muon benefit on modded-nanogpt-class architectures.
3. **MuonClip evidence.** Moonshot's Kimi K2 (arXiv 2507.20534) showed that even *vanilla* Muon causes attention-logit explosion via correlated singular-direction growth in W_Q and W_K — a mid-scale 9B-activated/53B-total MoE run with vanilla Muon saw "maximum attention logits quickly exceed a magnitude of 1000". Coupled Q–K may be partially fixing the same problem; if you then run on QK-Norm-equipped modded-nanogpt, the upstream cure is already applied and the marginal coupled gain disappears.
4. **Value embeddings, U-net skip, smear gate.** Modded-nanogpt's "extra embeddings mixed into the values" make `W_V` an effectively non-2D object (a learned mix of multiple sources). The V–O coupling assumption that the value path is `V W_V W_O` cleanly is broken.
5. **Bias parameters in Karpathy's variant.** Coupled Muon falls back to AdamW for non-2D parameters; biases are 1D so they go to AdamW — but their LR is typically tuned together with the matrix LR. AdamW LR is *not* transferable across optimizers, so a Muon-tuned LR on biases will be wrong.
6. **GQA & Q/K/V head asymmetry.** In GQA, K and V share heads while Q does not. Per-head V–O coupling is well-defined; per-head Q–K needs Q-group broadcasting of K. If LLaMA-60M is MHA (no GQA) and NanoGPT migration enables GQA, the head-broadcasting code path is exercised for the first time.

### b.3 Norm choice matters less

RMSNorm vs LayerNorm shouldn't materially affect coupled Muon's gradient geometry — both are scalar rescalings per token. Likewise learned-pos vs RoPE matters for Q–K only at the block-reshape level. These are secondary axes; spend time on MLP arity and QK-Norm first.

---

## (c) Why sparse MoE NanoGPT may fail to show improvements

### c.1 The Moonlight Lemma 1 lens

Moonlight (Liu et al., arXiv 2502.16982, Feb 2025) proved that Muon's update RMS for an (A,B)-shaped matrix is √(1/max(A,B)). They found two failure modes empirically:

> *"When max(A,B) is too small, e.g. treating each KV head in GQA or MLA as a separate parameter, the updates become too large, thus causing training instabilities."*
>
> *"When max(A,B) is too large, e.g. the dense MLP matrix, the updates become too small, thus limiting the model's representational capacity."*

Their fix: scale each Muon update by √max(A,B) (the same scaling Coupled Muon v2 already uses).

But MoE introduces a third regime: **per-expert intermediate dim is typically smaller than the dense MLP**. There are two distinct matchings to be careful about:

- **Equal-active-parameter** matching (most production MoE): `d_expert_ff ≈ d_ff_dense / top_k`, so a token sees roughly the same FFN compute as in the dense baseline. For (d_model=768, d_ff_dense=3072, top_k=2), each expert is `(768, 1536)`.
- **Equal-total-parameter** matching: `d_expert_ff ≈ d_ff_dense / E`, so the total parameter count matches the dense baseline. Same setup with E=8 gives expert shape `(768, 384)`.

The configs in this study (rungs I/M/N) hold *total expert capacity* `E × d_expert_ff` constant at the dense-FF target `8/3 · hidden = 11264` (matching the same dense-to-per-expert ratio the original 768-hidden ladder used). I has 8 × 1408 = 11264, M has 16 × 704 = 11264, N has 4 × 2816 = 11264 — per-expert width varies as 704 → 1408 → 2816 across M → I → N. **The 60M Common-Size adoption pushes M's per-expert FF down to 704, which is a more aggressive probe of Moonlight's "max(A,B) too small → instability" regime than the prior ladder's 1024 was at hidden=768 (max(512,704)=704 vs max(768,1024)=1024).**

For comparison, Kimi K2 (arXiv 2507.20534) is "1.04 trillion total parameters with 32 billion activated parameters per token, 384 experts activating exactly 8 per forward pass (sparsity 48), MoE hidden dimension 2048" — production training therefore knowingly puts Muon in the small-matrix regime. Even with √max(A,B) scaling, the *effective gradient signal* in the user's smaller setup is qualitatively different:

- Each token visits ~top_k/E of experts. Per-expert effective batch is `top_k/E` of global batch (e.g. 25% at top-2, E=8).
- Per-expert gradient SNR is ~`√(E/top_k)`× worse than the dense MLP.
- Newton–Schulz on a *noisier* G yields singular directions that are noisier estimates of the population polar factor. With 4 coupled steps, this noise is **amplified through the partner weight**: `‖G B‖_F` in the denominator becomes ill-conditioned when G is dominated by stochastic noise rather than signal.

### c.2 Coupling-ignores-gate is worse with experts

In dense SwiGLU, `silu(x W_gate)` is averaged over all training tokens and is reasonably smooth. In a *routed* expert that sees ~25 % of tokens — and only a token-skewed slice — `silu(x W_gate)` can be sparse and spiky. The coupled preprocessing's implicit assumption of approximately uniform gating becomes violently wrong, and the coupled inner step now subtracts the wrong quantity from the gradient.

### c.3 Routers — Muon by default, AdamW as the rung-J ablation

Router weights are (d_model, num_experts), e.g. (768, 8). max(A,B) = 768; with the standard Muon scaling, the effective LR is lr × 0.2 × √768 ≈ 5.5 × lr. The router gradient flows from the LM loss through the routing-prob multiplier on expert outputs, plus a load-balancing auxiliary term. Newton–Schulz on a router gradient is computing the polar factor of a near-degenerate matrix (rank ≤ min(d_model, num_experts) = 8) — Moonlight's "max(A,B) too small" regime, where one might naively expect instability.

The MoE-Muon literature, however, is **not unanimous** on this and the empirical evidence cuts the other way:

- **Moonlight** (Liu et al., arXiv 2502.16982, 2025) §2.2: "AdamW is used in couple with Muon to handle non-matrix based parameters, like RMSNorm, LM head, and embedding parameters." Their reference `examples/toy_train.py` splits parameters by `ndim ≥ 2 and not embed and not lm_head` — **routers go to Muon**. Section 3.4 explicitly compares router-Muon SVD entropy to router-AdamW: *"the SVD entropy of Muon is higher than that of AdamW … this discrepancy is more significant in the router weights for expert selection."*
- **Cerebras nanoMoE** (Cerebras blog, 2025) reports that Muon-routed runs *outperform* AdamW-routed runs: *"with Adam, learned routing … cannot outperform the dense GPT-2 baseline, unlike with Muon it actually does"*.
- **Kimi K2 / MuonClip** (arXiv 2507.20534) inherits Moonlight's optimizer split.
- **DeepSeek-V2/V3** (arXiv 2412.19437) and **OLMoE** (arXiv 2409.02060) use AdamW for *all* parameters and don't use Muon at all — so they don't speak to the router-optimizer choice within a Muon recipe.

**Default in this study is therefore router → Muon**, matching the published precedent that does use Muon. Rung J becomes the AdamW-router ablation: if J shows a coupled-vs-Muon gap *larger* than I, the Muon-router default is hurting and we should switch; if J degrades or NaNs, the default stands. The earlier framing "send router to AdamW, never Muon" was wrong; the partial overlap with Moonlight Lemma 1 (max(A,B) too small → instability) is real but is mitigated by the `√max(A,B)` scaling, which Coupled Muon v2 already applies.

### c.4 Load-balancing aux loss interferes with coupling

**Direct contamination is router-side, not expert-side.** The standard Switch-Transformer aux loss is `aux = E · Σᵢ Pᵢ · stop_grad(fᵢ)`, where `Pᵢ` is the average router probability for expert `i` and `fᵢ` is the no-grad token-fraction routed to it. Backprop therefore flows **only** through `Pᵢ` to the router weights — it does **not** directly contaminate `W_up/W_down/W_gate` of any expert. (An earlier version of this section claimed otherwise on the basis that the LB loss depends on activation magnitudes; that claim conflates Switch-style aux with an activation-magnitude regulariser, and is incorrect for the formulation actually used in this codebase, OLMoE, DeepSeek-V2, etc.)

**Indirect expert contamination via the routing-prob multiplier.** Expert outputs are scaled by router probabilities in the forward pass (`expert_out_e * topk_vals_e`). Aux-induced changes to router probabilities therefore *indirectly* perturb the LM-loss gradient on expert weights through this multiplier — a real but second-order effect.

**Implication for rung K.** Aux-loss-free routing (DeepSeek-V2 §3.2 bias-update style; Trinity SMEBU) removes the **router-side** contamination of the LM-gradient signal — that's specifically what K tests, not direct expert-weight contamination. If the dominant MoE failure mode for coupling is router-side aux contamination *or* c.5 (inter-expert competition / per-expert noise, which K also affects via cleaner routing dynamics), K should help. If the failure mode is a hypothetical activation-magnitude aux that contaminates expert weights directly, K is the wrong rung and that variant must be implemented and ablated separately (Phase 2). Reading "K helps" therefore tells you "router-side aux contamination matters", not "all aux contamination matters".

### c.5 Per-expert "competing over a shared input" effect

Experts within an MoE layer compete for tokens; their up/down matrices are coupled *across experts* through the router. Coupled Muon v2 only couples within an expert — `(down_i, up_i)` for each i — and so misses the inter-expert geometry. Two experts may end up specialising in the same direction and then fight each other every step, with coupled Muon faithfully orthogonalising each in its own little subspace.

### c.6 What public MoE training actually does

- **Moonlight (Muon, 2025):** uses Muon for *all* matrix-shape parameters including expert MLPs and routers, with √max(A,B) scaling; AdamW for embeddings, RMSNorm scales, LM head. Paper's headline: "Muon only requires about 52% training FLOPs to match the performance of AdamW under compute-optimal setting" (Figure 1a), demonstrated through scaling-law experiments culminating in the 3B-activated/16B-total MoE trained on 5.7T tokens. They report MoE benefits *more* from Muon than dense: "*This discrepancy is more significant in the router weights for expert selection, which indicates that mixture-of-expert models can benefit more from Muon.*"
- **Kimi K2 / MuonClip:** 1T-param MoE, 384 fine-grained experts, expert intermediate 2048 — production-scale, requires QK-Clip.
- **OLMoE (arXiv 2409.02060) / DeepSeek-V2:** AdamW-only, β₂ = 0.95.
- **Cerebras nanoMoE-on-modded-nanogpt:** "with Muon … expert networks learn with less router supervision" — observational, not ablated.

So vanilla Muon *does* work on MoE at scale. The fact that **coupled** Muon does not improve over Muon on MoE is therefore most likely *not* about MoE breaking Muon — it's about the **coupling assumptions** (gate-free SwiGLU, single-product geometry, intra-expert only) becoming noisier and less predictive at the per-expert scale.

### c.7 The Muon-vs-AdamW reference gap

Calibrate expectations against the published spread, not against the high-end:

- Essential AI (Shah et al., arXiv 2505.02222, May 2025, 100M–4B): "Muon requires 10–15 % fewer tokens than AdamW to reach an identical loss" — i.e. ~1.10–1.18×.
- "Fantastic Pretraining Optimizers" (Wen, Hall, Ma & Liang, arXiv 2509.02046, 2025): "matrix-based optimizers all deliver approximately a 1.3× speedup over AdamW for models under 520M parameters", and "Muon and Soap's speedup decays with model size to only 1.1×" for 1.2B-parameter models at 8× Chinchilla ratio.
- Keller Jordan blog: 1.5B GPT-2-XL parity in 10 vs 13.3 H100-hours = **1.33×**; speedrun improvement = **1.35×**.
- "Hyperparameter Transfer …" (Bergsma et al., arXiv 2512.05620, NeurIPS 2025): "Combining μP with 1/width independent weight decay, Muon and Shampoo achieve consistent **1.4× and 1.3× speedups**, respectively, over well-tuned AdamW in training 190M to 1.4B-parameter models on FineWeb"; the 1.4× for Muon "vanishes rapidly with scale under incorrect scaling".
- Moonlight (arXiv 2502.16982, 2025): "Muon only requires about 52 % training FLOPs to match the performance of AdamW under compute-optimal setting" (≈ **2×**), but this number is from scaling-law sweeps culminating in 3B/16B MoE — and Wen et al. flag it as partly a tuning-quality artefact ("their most extensively tuned experiments on 130M models use a batch size of only 0.1M and 0.02M tokens, whereas our experiments operate with tuned batch sizes that are not smaller than 0.4M tokens").
- Yuchen Jin (Oct 2024): 1.5B GPT-2 reaches val 2.90 in 4.2B vs 10B AdamW tokens — **2.4×**, but at relatively under-tuned AdamW.

The "≈ 50 % wall-clock" Muon-vs-AdamW figure you recalled is the **upper end of the literature**; under a properly tuned AdamW baseline at 60M–500M, the realistic reference gap is **1.2–1.5×**, not 2×. If your LLaMA-60M coupled-Muon-vs-Muon gap is in the 1.05–1.15× range, MoE non-improvement may simply mean you are inside noise on a smaller increment built atop a smaller base.

---

## (d) **Concrete experiment design — main deliverable**

### d.1 Recommendation: which NanoGPT base?

**Recommended: a stripped fork of `KellerJordan/modded-nanogpt`**, taken at commit `~ba3e54f` (early Oct 2024 architecture: RoPE + RMSNorm + ReLU² + QK-Norm, *before* the value-embedding / U-net-skip / smear-gate / sliding-window / scalar-gate stack accumulated).

| Repo | Pros | Cons | Verdict |
|---|---|---|---|
| Karpathy `nanoGPT` | Minimal, clean, single-file. No Muon assumptions. AdamW baseline obvious. | GELU/LayerNorm/learned-pos: too far from LLaMA in 3 axes. No FlashAttention-3, slow. | **Use as a single ladder rung, not the harness.** |
| Modded-`nanogpt` (current head, 2026) | Best-tuned baseline; Muon already wired up; FlashAttention 3 ready. | Many speedrun-only tricks (value embeddings, U-net skip, smear gate, partial RoPE, soft-cap, scalar gates) that confound optimizer comparison and break V/O coupling assumptions. | **Avoid current head.** |
| Modded-`nanogpt` @ ~Oct 2024 commit | Modern but minimal: RoPE + RMSNorm + ReLU² + QK-Norm + Muon. Closest to LLaMA but with a 2-mat MLP. | Still has QK-Norm which interacts with Q–K coupling. | **Recommended.** Disable QK-Norm in one ladder rung to test interaction. |
| Build from scratch | Maximum control. | Slow, error-prone. | **No.** |

For MoE, fork **Cerebras's `train_gpt_moe.py`** (a modded-nanogpt MoE port; see https://www.cerebras.ai/blog/moe-guide-debug). Already has Muon × MoE wired. Cameron Wolfe's `nanoMoE` on top of Karpathy nanoGPT is the AdamW-only equivalent.

**Concrete first action:** copy the train script, delete `value_embeds`, `u_net_skip`, `smear_gate`, `attention_residuals`, `logit_softcap`, `window_warmup`. Keep RoPE, RMSNorm, QK-Norm (toggle-able), ReLU² (toggle-able to GELU/SwiGLU). Verify your reduced architecture trains to a sensible loss in single-optimizer mode before moving on.

### d.2 Bridging experiment ladder

Under the 2×H200 compute budget the ladder collapses to a **fixed-architecture-only** comparison: every dense rung A0–H and the dense backbone of every MoE rung I/I'/J/K/L/M/N use the same 60M Common-Size shape (hidden=512, n_layers=8, n_heads=8, head_dim=64). MoE per-expert MLP capacity is held constant via E · d_expert_ff = 11264 (SwiGLU rungs) or 16384 (ReLU², rung I' inheriting the dense `4·hidden` formula). The earlier "scale to 125M" axis (A1) is removed — A1 was shape-identical to A0 once the collapse landed, and the ablations it would have supported now route through A0 directly. Scale validation is deferred to a separate ~350M dense rung run only after the 60M-CS ladder closes. The ladder still exercises the same architectural axes (MLP arity, pos-emb, norm, QK-Norm, MoE topology) — only the size axis is muted.

The ladder transforms LLaMA-60M into MoE-NanoGPT one axis per step. For *every* configuration, run `{AdamW, Muon, Coupled Muon v2}` with ≥ 3 seeds and a small LR sweep (Section d.3).

```
A0  LLaMA-60M-CS baseline (your existing config)           [reproduces the 60M result; canonical 60M-CS shape (hidden=512, n_layers=8, n_heads=8)]
B   A0 with SwiGLU → GELU 2-matrix MLP (delete W_gate)     [does coupling still help when (down,up) is the EXACT composition?]
C   B with RoPE → learned positional embeddings            [does Q–K coupling effect survive without RoPE block reshape?]
D   C with RMSNorm → LayerNorm + biases                    [Karpathy regime; AdamW LR more transferable]
E   D ≈ Karpathy nanoGPT (vanilla GPT-2-style)             [reference: dense NanoGPT, no LLaMA-isms]

--- now go in the OTHER direction from A0 ---

G   A0 with QK-Norm ON                                     [isolate QK-Norm × Q–K coupling interaction]
H   G with ReLU² activation                                [modded-nanogpt-style dense: 2-mat MLP + RMSNorm + QK-Norm + RoPE]
H'  H with rope_partial_frac=0.5                           [optional follow-up: faithful modded-nanogpt 50% RoPE; forces flat-2D Q-K coupling fallback — see note]

--- now sparse MoE; all share H's 60M-CS dense backbone ---

I'  H with MoE (every-other-layer, 8 experts top-2, capacity 1.25, aux 0.01,
    z 1e-3); router under Muon (default, per c.3). Experts inherit ReLU² 2-mat
    from H at I_expert = 4·hidden = 2048 (matches dense FF) — no SwiGLU gate complication.
    [first MoE rung: tests MoE-introduction alone, single-axis change vs H]
I   I' with experts → SwiGLU (and dense MLP → SwiGLU); per-expert SwiGLU at
    I_expert = 1408 (= 8/3·512), matching dense FF and yielding E·I_expert = 11264.
    [tests gate-omission-under-routing on MoE; closer to production recipe]
J   I with router under AdamW                              [router-optimizer ablation]
K   I with auxiliary-loss-free balancing (DeepSeek bias-update style; Trinity SMEBU)
L   I with shared expert (1 always-on + 8 routed top-1)    [DeepSeekMoE-style; reduces per-expert variance. 9 expert MLPs/layer total — kept symmetric with I/J/K/M/N (which all have 8 routed) so the comparison is across a single axis (presence of shared expert)]
M   I with 16 experts top-2 (per-expert I_expert = 704)    [stress test small-matrix regime; max(512,704)=704 is more aggressive than the prior 1024 at hidden=768]
N   I with 4 experts top-2 (per-expert I_expert = 2816)    [stress test large-matrix regime; per-expert FF 2× the dense FF]
```

Expected localisation rules:
- If the coupled gap drops between **B and A0** (MLP-arity axis only — same 60M-CS shape) → the gate-ignoring approximation is the issue. Fix by adding gate to a triple coupling, e.g. `(down, up, gate)` joint NS.
- If it drops between **C and B** → the RoPE block reshape was crucial; investigate whether the partial-RoPE / no-RoPE Q–K coupled formula is even well-defined. Note: the implementation auto-falls-back to flat-2D Q–K coupling whenever `pos_emb.type=learned` *or* `rope_partial_frac<1.0` (no per-head 2-D rotation pair exists in those cases), so C/D/E and any partial-RoPE rung silently lose the multi-head Q-K branch.
- If it drops between **G and A0** (QK-Norm-on vs LLaMA) → Q–K coupling and QK-Norm are doing the same job; one is redundant, prefer QK-Norm.
- If H matches G and H' (partial RoPE) underperforms H → the multi-head Q-K branch is doing real work, and partial RoPE on modded-style architectures defeats it.
- If gap survives all the way to **H** but dies at **I'** → MoE-introduction itself is the issue (per-expert SNR, inter-expert competition; Section c.5). Pursue Section d.4 ablations.
- If gap survives at **I'** but dies at **I** → the SwiGLU gate-omission-under-routing hypothesis (Section c.2) is confirmed. The triple `(down, up, gate)` coupling becomes the highest-priority Phase-2 algorithmic change.

### d.3 Optimizer comparison protocol

**Compute-matched, on two axes simultaneously.** Every (config × optimizer × seed × LR) point must report **both**:

1. **Loss-vs-tokens** (sample efficiency): how much data each optimizer needs to reach a given loss.
2. **Loss-vs-wall-clock** (compute efficiency): how long each optimizer takes to reach a given loss, including its own per-step overhead.

Coupled Muon's extra matmul-heavy optimizer step (~9 NS iterates per coupled parameter per step — see d.6.9) means the two axes diverge: the wall-clock crossover may favour Muon even where the sample-efficiency crossover favours Coupled Muon. Both numbers are real results; the design is fair only if both are reported. Aggregate the LR sweep by reporting the *envelope* on each axis (the tuned-best loss curve at each token / wall-clock budget), per Essential AI (arXiv 2505.02222) protocol.

Use FineWeb-Edu val for dense (target loss in the 3.4–3.6 band at 60M-CS, slightly higher than the 3.20–3.30 a 125M dense recipe would reach at the same token budget) and a held-out FineWeb subset for MoE (same loss target as the dense model with equal *active* params).

**Per-optimizer LR sweep.** LRs are NOT transferable across optimizers (Lakernewhouse, "Understanding Muon": "Muon works well with learning rate around 0.02"; Liu & Hong 2025: optimal LR shifts with NS precision). Use the **telescoping sweep** from Essential AI (arXiv 2505.02222):
- 6-point log-spaced LR grid at the smallest model size (60M).
- At each width-doubling, halve the grid spacing, keeping it centred on the previous optimum.
- For Muon variants, sweep `{1e-3, 3e-3, 1e-2, 3e-2, 1e-1}`; for AdamW, `{3e-5, 1e-4, 3e-4, 1e-3, 3e-3}`.
- Hold weight decay = 0.1 / width (Bergsma et al., arXiv 2512.05620 finding); β₁ = 0.95, β₂ = 0.95 for AdamW; β = 0.95, ns_steps = 5, coupled_steps = 4 for Coupled Muon.

**Seeds.** ≥ 3 seeds per (config × optimizer × LR) point. Report median final loss with bootstrap 95 % CI. The Muon-vs-AdamW gap is small enough that single-seed reports are not trustworthy — at the 1.4× speedup level, within-seed std is often > 50 % of the between-optimizer mean gap.

**Metrics to log per run:**
- Train and val loss curves (downsampled every 100 steps).
- Per-parameter group: `‖W‖_F`, `‖∇W‖_F`, top-5 singular values of W (every 1000 steps).
- For coupled pairs (Q,K), (V,O), (down,up): compute condition number `κ(W_A W_B)` and Frobenius norm `‖W_A W_B‖_F`, plus `σ_max(W_A W_B)` every 1000 steps. Coupling should *flatten* κ.
- Effective rank / SVD entropy: `H(σ) = −Σ p_i log p_i`, `p_i = σ_i² / Σ σ_j²` (Moonlight Figure 4 metric).
- For Q–K specifically: max pre-softmax logit per layer per step (the MuonClip stability signal).
- Newton–Schulz internal: `‖X‖_F` per coupled iteration to detect divergence.
- For MoE: per-expert load (token count), per-expert grad norm, router entropy.

**Stopping rule.** Run each config to a fixed token budget (Section d.5) regardless of convergence — the only fair compute-matched comparison.

### d.4 Diagnostic experiments / ablations

| Ablation | What it tells you |
|---|---|
| `final_polish ∈ {True, False}` (`True` = current code's two-stage `Muon ∘ C_A`; `False` = stage-1 only — the coupled iterate `u_A_c` is applied directly without the trailing zeropower NS pass) | Localizes whether stage 2 (the trailing zeropower NS) is doing real work, or is approximately a no-op because stage-1 output is already near-orthogonal at the operating point. Run on rung A0 with 3 seeds × 3 LRs each, alongside the post-stage-1 `‖X‖_spec` probe (extended `ns_internal`). Three outcomes: `True` wins → two-stage interpretation is correct, report it as the operating point; `False` wins → simpler variant is better, re-tune LR for it and consider dropping stage 2 from the algorithm; tie → simpler is better. |
| Post-stage-1 `‖X‖_spec` probe (passive; runs alongside any coupled-Muon rung) | Single diagnostic that tells you directly whether stage 2 is needed. If stage-1 X is near-spectral-norm-1 across coupled pairs (`σ_max ≈ 1`), the two stages are nearly redundant and `final_polish=False` should match training behavior; if `σ_max` varies far from 1, stage 2 is doing real work. Reported per (Q-K, V-O, up-down) representative pair every `probes.ns_internal_interval_tokens`. |
| `coupled_steps ∈ {0, 1, 2, 4, 8}` | 0 = plain Muon; monotone improvement up to 4 in dense → coupling does real work; flat or non-monotone in MoE → noise amplification. |
| Couple **only Q–K** (turn off V–O and down–up) | Is the gain attention-side or MLP-side? |
| Couple **only V–O** | Cleanest geometry; if this alone gives the LLaMA gain, it's the simplest story. |
| Couple **only down–up** | The MLP-side claim; expect this to *break* on SwiGLU, *work* on GELU. |
| Couple `(down, up, gate)` triple (define a 3-matrix coupled NS variant) | Tests the gate-omission hypothesis directly. |
| `multi-head couple` ON vs OFF | Does the per-head batched branch matter? |
| Coupling **across experts** (concatenate all expert up's into one big up; orthogonalise with concatenated down) | Tests whether inter-expert geometry was the missing piece in MoE. |
| **Per-expert SNR probe** | Compute, on a held-out batch, the cosine between mini-batch G and the full-data G estimated by averaging 32 mini-batches. If per-expert cosine < 0.5, NS is iterating on noise. |
| **bf16 vs fp32 NS** | NS at bf16 with the standard quintic is generally fine on H200, but coupled NS uses ~2× the matmuls and can accumulate error. Run one config at fp32 NS to bound the error contribution. |
| **Router optimizer**: Muon (default) vs AdamW (rung J) vs Coupled-Muon-on-router | Default is Muon (Section c.3). Rung J ablates AdamW. Coupled-Muon-on-router (i.e. pairing the router with one of the experts) is a Phase-2 variant — out of scope for the immediate ladder. |
| **MoE z-loss on/off + aux-loss-free balancing** | Aux-loss-free balancing (DeepSeek bias updates) removes LB-grad contamination of expert weights; should make coupled Muon "more itself". If coupled Muon improves under aux-loss-free routing relative to with-aux-loss, that's strong evidence that LB-grad contamination is the killer. |
| **Couple within active subset only** | Mask the gradient G_A by tokens that actually routed to the expert before NS; reduces noise floor. |
| **Per-expert max(A,B) regime check** | At 60M-CS, M has max(512, 704)=704 (small-matrix Moonlight regime); N has max(512, 2816)=2816 (large-matrix). Confirm Muon's `√max(A,B)` scaling is applied per-expert in the optimizer factory before treating M's "small-matrix stress" as a real signal vs an unscaled-LR artefact. |

### d.5 Specific 2×H200 configs

H200 has 141 GB HBM3e at 4.8 TB/s, ~1.4× H100 throughput on bf16 GEMM. With FlashAttention-3, plan ~150 K tok/s/H200 for 60M-CS dense. 2×H200 ≈ 300 K tok/s aggregate at 60M-CS dense, ~150 K tok/s aggregate at the deferred 350M dense validation rung, ~150 K tok/s for the ~112M-total / ~60M-active MoE class.

**Note on actual model sizes.** The size labels below are the *class* the rung tests, not exact param counts. Under the 60M Common-Size collapse (hidden=512, n_layers=8, n_heads=8) all dense rungs share a single shape and all MoE rungs share a single dense backbone with proportionally scaled experts. Measured sizes from the implementation:

| Rung class | Spec target | Measured (8 layers, hidden=512, every-other) | Notes |
|---|---|---|---|
| A0 (SwiGLU) | 60M | 51.5M | tied embed + SwiGLU 8/3-rounded ff = 60M class, sub-60M actual |
| B, H (2-mat MLP) | 60M | 50.9M | GELU/ReLU² intermediate = 4·hidden = 2048; tied embed |
| C, D, E (2-mat + learned-pos / LN / Karpathy) | 60M | 52.0M | learned pos-emb adds ~1.05M (2048×512); otherwise same as B |
| G (SwiGLU + QK-Norm) | 60M | 51.5M | QK-Norm scales are tiny |
| I, J, K, M, N (SwiGLU MoE) | 60M-CS backbone, E·I_expert = 11264 | 112.0M total / ~60M active | 4 dense + 4 MoE layers (every-other); per-expert SwiGLU at 1408 (I/J/K), 704 (M), 2816 (N) |
| I' (ReLU² MoE) | 60M-CS backbone, E·I_expert = 16384 | 109.7M total / ~58M active | per-expert ReLU² 2-mat at I_expert = 4·hidden = 2048 (matches dense); E=8 |
| L (shared expert) | 60M-CS backbone | 120.7M total | 8 routed top-1 + 1 shared SwiGLU expert at 1408 each |
| 350M dense validation | 350M | not specified — pick recipe at the time | deferred until 60M-CS ladder closes; original "hidden=1024, n_layers=18" recipe still ~350M |

**Compute estimates revised to 60M-CS sizes on 2×H200:**

| Config | Params | Tokens | Wall-clock on 2×H200 | Comment |
|---|---|---|---|---|
| 60M-CS LLaMA (A0) | 51M | 1.2B (≈20× Chinchilla) | ~1 h/seed | Reproduce existing result; 5 seeds; ~5 GPU-h. Adds `final_polish ∈ {True,False}` × 3 LRs × 3 seeds (~9 runs ≈ ~9 GPU-h). |
| 60M-CS dense ladder rung (B–H, optional H') | ~51M | 2.5B | ~2 h/seed | 7 rungs × 3 optimizers × 3 seeds × 5 LRs ≈ ~315 runs; total ~630 GPU-h on 2×H200. |
| 350M dense (validation rung; deferred) | 350M | 7B | ~18 h/seed | Run once on the **best** (config × optimizer) from the 60M-CS ladder. Out of immediate 2×H200 budget. |
| MoE rung I' (ReLU² experts) | ~110M / ~58M active | 5B | ~3.5 h/seed | First MoE rung between H and I (d.2). 3 optimizers × 3 seeds × 5 LRs ≈ ~150 GPU-h. |
| MoE class I, J, K, M, N | ~112M / ~60M active | 5B | ~3.5–4 h/seed | 5 rungs × 3 optimizers × 3 seeds × 5 LRs ≈ ~750 GPU-h. |
| MoE class L (with shared expert) | ~121M | 5B | ~4 h/seed | Slightly bigger: extra always-on expert per MoE block. |
| MoE diagnostic (d.4 ablations) | ~112M / ~60M active | 1B | ~50 min/seed | Short-run ablations at fixed token count. |

If the 350M+ scale-up rung is commissioned (e.g. to reach the regime where Wen et al.'s 1.3× Muon gap is empirically demonstrated rather than extrapolated), commit to one of these recipes for the dense or MoE upper bound:
- `n_layers: 20, hidden: 768` → ~510M total / ~110M active
- `n_layers: 12, hidden: 896, intermediate: 2389 (8/3 of hidden)` → ~480M total
- `n_layers: 12, hidden: 768, num_experts: 14` → ~510M total

Each adds 30–50 % to wall-clock vs the deferred 350M dense recipe; rebudget on 2×H200 accordingly.

**Sequence length:** 2048 throughout. **Global batch:** 0.5M tokens (gradient accumulation as needed). Essential AI's analysis says Muon's relative advantage *grows* at large batch — but at 0.5M you're already in the regime where the literature's 1.3–1.5× gap is observed, and you avoid the under-tuned-AdamW critique that plagues the 0.02M-batch Moonlight numbers (Wen et al. flag this directly).

**Total compute budget estimate:** ~**1300 GPU-hours** across the full 60M-CS ladder + ablations — roughly half the original 4×H200 figure because the dense ladder shrinks from 124M to ~51M params (compute is roughly param-linear at this scale) and the MoE ladder shrinks from ~322M total to ~112M total. On 2×H200 = ~27 wall-clock days at full sequential utilisation. Cut to A0-only + spot validation (~25 GPU-h: AdamW 25 cells + Muon-family 50 cells × ~1 h/seed at 1.2B tokens, plus the 9-run final_polish ablation) if budget is tighter. The deferred 350M dense rung adds ~50–100 GPU-h on top whenever it runs.

### d.6 Common pitfalls and confounders

1. **Optimizer-state mixing in checkpointing.** If you ever resume a Muon run with AdamW or vice-versa — even by accident through `--resume` — results are silently wrong. Moonlight: *"a model that is both Muon-pretrained and Muon-finetuned outperforms others … when the SFT optimizer differs from the pretraining optimizer, SFT with Muon does not show a significant advantage over AdamW."* Stamp the optimizer name into the checkpoint filename and assert on resume.
2. **LR warmup interaction.** Muon and Coupled Muon tolerate shorter warmup (~100 steps) than AdamW (~1000 steps). A warmup tuned for AdamW will under-train Muon early. Use cosine + linear warmup; sweep warmup as a hyperparameter at one rung.
3. **bf16 vs fp32 for Newton–Schulz.** Standard Muon NS is bf16-stable. Coupled NS does ~2× the matmuls per step and the iterate involves products with the partner weight, which can have arbitrary scale. Either (a) divide G by `‖G B‖_F` in fp32 (verify your code does), or (b) cast the inner NS to fp32 if you see NaN/Inf or non-monotone loss. Polar Express analysis (arXiv 2505.16932) shows NS error correlates with `σ_min(G)`, which is ≈ 0 on the first iterations.
4. **AdamW baseline tuning quality.** Wen et al. ("Fantastic Pretraining Optimizers", arXiv 2509.02046) warn explicitly: a poorly-tuned AdamW makes any Muon variant look 2×; a well-tuned AdamW shrinks the gap to 1.3–1.5×. Tune AdamW with the same LR-grid effort as Muon. If your LLaMA-60M result is 1.5× over AdamW, that's plausible; if it's 2.5×, look at your AdamW LR.
5. **Weight decay matched.** Use 1/width-scaled weight decay (Bergsma et al. 2025) for both optimizers. WD on Muon and WD on AdamW are *not* the same regulariser (Muon WD acts on already-orthogonal updates), so don't directly transfer numbers.
6. **`grad_norm_clip` matched.** If LLaMA pipeline does `torch.nn.utils.clip_grad_norm_` and modded-nanogpt does not (it relies on Muon's spectral bound), this is a hidden confounder.
7. **Reproducibility on H200.** H200 tensor-core paths can produce slightly different bf16 results than H100; lock `torch.use_deterministic_algorithms(True)` only for diagnostic runs (it's slow).
8. **Routing replay across seeds in MoE.** Different seeds → different router init → different expert load → different per-expert effective batch → different Muon dynamics. Increase seed count to ≥ 5 for MoE.
9. **Optimizer step time.** Coupled Muon v2 has 4 inner coupled steps + 5 outer NS = 9 matmul-heavy iterations per coupled parameter per step. Profile this. If overhead is ≥ 30 % of forward+backward, you may be paying for coupling in wall-clock not bought back in samples — important for the Muon-vs-Coupled-Muon comparison to be fair on the wall-clock axis.
10. **Don't over-anchor on the 50 % wall-clock figure.** Section c.7's literature spread shows ~1.3–1.5× is the realistic well-tuned reference at your scale. If coupled Muon adds another 10–20 % on top, that is a real result; don't expect it to add another 50 %.

---

## Recommendations (staged, with thresholds)

**Stage 1 (~500 GPU-h) — Reproduce.** Run A0 (LLaMA-60M with all three optimizers) under your bridging harness with ≥ 5 seeds and a 5-point LR sweep. Confirm coupled-Muon-vs-Muon gap is real and ≥ 3× the seed-to-seed std. **Threshold to proceed:** coupled Muon beats Muon on dense LLaMA-60M with p < 0.05 across seeds.

**Stage 2 (~1000 GPU-h) — MLP arity isolation.** Run rungs A1, B, E (LLaMA-125M; LLaMA + GELU 2-mat MLP; Karpathy nanoGPT). **Decision rule:** if B (GELU 2-mat) loses the coupled gap, the (down, up, gate) triple coupling is the highest-priority next algorithmic change. If E loses it but B doesn't, the issue is at the Karpathy-style level (LayerNorm/biases/learned-pos) — investigate via D and C.

**Stage 3 (~800 GPU-h) — Modded and MoE.** Run G/H (modded-nanogpt-style dense rungs) and the MoE rung I, then the d.4 ablations. **Decision rule:** if (i) coupling helps on dense modded-nanogpt H but not on MoE I → the failure is MoE-specific; pursue per-expert SNR probe (d.4) and aux-loss-free routing (rung K). If (ii) coupling already disappears on H → it is a QK-Norm × Q–K coupling interaction; selectively disable QK-Norm or modify the Q–K coupled formula to account for the post-norm geometry.

**Stage 4 (~700 GPU-h) — Best-config validation.** 350M dense and 500M-MoE, tighter seed count, final result.

**Algorithmic changes worth attempting in parallel:**
- `(down, up, gate)` triple coupling for SwiGLU. Define the inner step with a 3-factor product `down · diag(silu') · up` linearised at current activation statistics.
- "Gate-aware" coupling: precondition (up, down) using `B = down · diag(σ̂(gate-act))` where σ̂ is an EMA estimate of mean activation. Cheap and isolates the missing gate Jacobian.
- Couple **across experts** in MoE: define `up_concat = [up_1; …; up_E]`, `down_concat = [down_1, …, down_E]`; run coupled NS on the concatenated pair every step. Cost ≈ 1 large coupled step instead of E small ones.

## Caveats

- **The "≈ 50 % wall-clock" Muon-vs-AdamW figure is the literature upper bound.** Realistic, well-tuned reference at 60M–500M dense is 1.10–1.50× (Essential AI 2505.02222: 10–15 %; Wen et al. 2509.02046: ~1.3× under 520M; Bergsma et al. 2512.05620: 1.4× across 190M–1.4B). Calibrate before claiming a coupled-Muon-vs-Muon delta on top.
- **Coupled Muon v2 is, to the best of public knowledge, a novel algorithm.** No 2024–2026 paper performs the `C(G_A; B)` update using the partner's *current value*. The closest published ideas (Manifold Muon, Gram-Space Muon, CASPR-without-accumulation, MuonEq, PolarGrad) all operate per-matrix. This is a real contribution if it survives the bridging ladder; the LLaMA-60M result alone is suggestive but not yet evidence at scale.
- **The MoE non-result may itself be the most informative signal.** Coupling's failure mode in MoE pinpoints where its inductive bias mismatches reality (gate ignored, per-expert SNR low, inter-expert competition unmodelled). Treat the MoE experiments as positive science about the algorithm, not as a debugging chore.
- **MuonClip / QK-Clip is the production fix for the same Q–K instability that motivates coupling.** If your scale-up shows attention-logit explosion with coupled Muon (it shouldn't, but verify the max-logit metric), do not invent a new fix — adopt QK-Clip directly.
- **Routers default to Muon (Section c.3 rewritten).** Matches Moonlight 2502.16982 and Cerebras nanoMoE; rung J is the AdamW-router ablation. The earlier "send routers to AdamW, never Muon" wording was wrong.
- **Coupled Muon's update is two-stage** (Section a.2): `U = NS(C_A(G; B))`. Stage 1 is partner-aware NS; stage 2 is plain NS, gated by `final_polish` (default True, the operating point that produced the LLaMA-60M result). Whether stage 2 is doing real work or is a numerical no-op is an open ablation (rung A0 in d.4) — not a settled algorithmic claim.
- **Architectural confounders dominate at 60M–125M.** Single-seed reports at this scale routinely flip on re-runs. Three seeds is a floor, not a target; five for MoE.
- **Some 2026-dated arXiv IDs cited above (e.g. 2602.xxxx, 2603.xxxx, 2512.xxxx) are recent preprints; verify their final published versions before relying on the exact numbers.**
- **Aux-loss formulation matters.** Section c.4 was rewritten after the implementation audit: the standard Switch-Transformer aux loss `aux = E · Σ Pᵢ · stop_grad(fᵢ)` only contaminates router weights directly; expert MLP gradients are perturbed *indirectly* through the routing-prob multiplier in the forward pass. If you want to test the activation-magnitude aux variant (which would directly contaminate expert grads), implement and ablate it as a separate Phase 2 rung — rung K alone does not isolate that effect.
- **Multi-head Q-K coupling is RoPE-only.** The 2-D rotation-pair reshape (a.4) is meaningful only with full RoPE; the implementation auto-disables `use_multi_head` whenever `pos_emb.type=learned` or `rope_partial_frac<1.0`. Rungs C, D, E, and any partial-RoPE rung (e.g. H') therefore silently lose the multi-head Q-K branch and run the flat-2D coupling instead. Factor this into the "where did the gap go?" diagnosis: the gap can drop at C, D, or E *because* multi-head fell off, not because of pos-emb / norm choice per se.
- **K and L are paper implementations, not algorithmic inventions.** Aux-loss-free balancing (rung K) is from DeepSeek-V2 §3.2; the shared-expert pattern (rung L) is from DeepSeekMoE. Treat the experiments as testing whether those published mechanisms restore coupled-Muon's gain on MoE — not as proposing new mechanisms. Genuine algorithmic inventions — `(down, up, gate)` triple coupling and cross-expert coupling — stay in Phase 2 and should only be commissioned if A0–N data shows the predicted failure pattern (gap survives all dense rungs but dies at MoE I and J/M/N rule out the simple explanations).

---

## Appendix X — Honest derivation of the constrained product problem

Section a.2 made the heuristic claim: "Coupled Muon's stage 1 is steepest descent for A on the variety {AB : A ∈ ℝ^{m×k}, B fixed}." This appendix walks through what the *rigorous* answer to that constrained problem is, and shows where stage 1's `polar(G_A B)` target deviates from it.

### X.1 The constrained problem

Fix B ∈ ℝ^{k×n}. For a gradient G_A = ∇_A L ∈ ℝ^{m×k}, we want the search direction Δ_A ∈ ℝ^{m×k} minimising the linearised loss subject to a spectral-norm bound on the *product*:

> minimise   ⟨G_A, Δ_A⟩_F
> subject to ‖Δ_A · B‖_op ≤ η

The constraint is a *seminorm* on Δ_A (a norm only when B has full row rank); it can be zero on the kernel of the linear map T : Δ_A ↦ Δ_A B.

### X.2 The dual / closed form

By standard convex analysis (the dual of the spectral norm is the nuclear norm), the optimum Δ_A* satisfies

> Δ_A* B = -η · polar(G_A · P_B),

where **P_B = B^+ B = B^T (B B^T)^{-1} B** (when B has full row rank) is the orthogonal projector onto the row space of B. The minimum-Frobenius preimage in {Δ_A : Δ_A B = Y*} is Δ_A* = Y* · B^+; alternative preimages differ by elements of ker(T) and do not change the linearised objective.

Sketch of derivation: substitute Y = Δ_A B; rewrite ⟨G_A, Δ_A⟩_F = ⟨G_A B^{+T}, Y⟩_F (using Δ_A = Y B^+); the problem becomes argmin_{Y : ‖Y‖_op ≤ η} ⟨H, Y⟩ with H = G_A B^{+T} = G_A · B^T (B B^T)^{-1}. The minimiser is Y* = -η · polar(H). Substituting back: H · B = G_A B^T (B B^T)^{-1} B = G_A · P_B, and polar(H) · B simplifies (under full row rank) to polar(G_A · P_B). Hence Δ_A* B = -η · polar(G_A · P_B). □

### X.3 Where stage 1 deviates

Stage 1 of Coupled Muon v2 targets

> X · B → polar(G_A · B)

via the Newton-Schulz quintic on the iterate W = X B. This is **not** the rigorous answer above — the rigorous answer involves polar(G_A · **P_B**), with P_B = B^+ B, not polar(G_A · B). The two coincide *only* when B has orthonormal rows (P_B becomes B^T B = I_k after rescaling). Under that condition,

> G_A · P_B = G_A · B^T B   (B with orthonormal rows ⇒ BB^T = I)

so polar(G_A · P_B) and polar(G_A · B) differ only by the post-multiplication by B itself (a partial isometry under orthonormality), which doesn't change the polar factor's left-singular structure.

In practice, B is *approximately* orthonormal because Muon-style updates push every weight toward unit-spectral-norm SVD bands (Section a.1 — the [0.5, 1.5] band; Bernstein–Newhouse's Manifold Muon makes this Stiefel constraint explicit). So polar(G_A · B) is an approximation to polar(G_A · P_B) that is correct in the limit of orthonormal partners, and degrades as B drifts off-Stiefel during training. This is the honest motivation for the algorithm: it is not a proven steepest descent, but a Stiefel-limit approximation to one.

### X.4 Why the architectural argument still stands

The bilinear products `W_Q W_K^T`, `W_V W_O`, `W_up W_down` are what the forward pass uses (a.3); orthogonalising those products under the spectral norm has a clean architectural justification independent of any constrained-optimisation derivation. Stage 1's `polar(G_A · B)` target is the version of "orthogonalise the product" that is computable in one Newton-Schulz quintic without a B^+ inverse — the rigorous P_B-projection version would need an extra (B B^T)^{-1} solve per step, defeating the matmul-only design. The empirical question is whether the Stiefel-limit approximation is good enough to deliver the LLaMA-60M sample-efficiency gain at scale, which the bridging ladder (d.2) is designed to test.

### X.5 Implication for stage 2

Section a.2.1 frames stage 2 as enforcing `‖U_A‖_spec ≈ 1`. Under the rigorous derivation, we'd want Δ_A B = -η · polar(G_A P_B), which has spectral norm η — not Δ_A itself with spectral norm η. Stage 2 corrects the `‖Δ_A‖_spec` rather than the `‖Δ_A B‖_spec`. The two coincide when B has orthonormal rows; the gap between them is what `final_polish` controls in the ablation. If post-stage-1 σ_max(X) ≈ σ_min(B), stage 2 does meaningful work; if σ_max(X) ≈ 1 already, stage 2 ≈ no-op, and the Stiefel-limit approximation is tight. The post-stage-1 `‖X‖_spec` probe (d.4) measures this directly.

---

## (d.7) Phase 2 — factored-architecture rungs (new rungs and ablations)

Phase 2 expands the bridging ladder along three axes: a **factored-architecture rung (MLA + low-rank Q)** to test CoupledMuon's distinctive claim where it actually bites; a **Newton–Schulz coefficient policy axis** + a **K-curve** to defend every CoupledMuon-vs-Muon delta against the "did a better NS policy close the gap?" attack; and a set of **methodology equalizers** (AdamW seeds × LR matched to Muon-family, pair-policy split at production, LR-prefactor formulas). Plus a **350M MLA scaling rung** to produce a 2-point scaling-trend claim, and an **imposed-FFN-factorization** rung to test whether the gain transfers beyond natively factored architectures.

Scope follows `suggestion.md` Tier S + A + selected B (B1 LoRA-rank, B3 imposed-FFN). KDA / linear-attention (B2), GPT-2 baseline, FFN-nonlinearity sweep, and sliding-window attention are out of scope.

### d.7.1 New ladder rungs

```
O_mla        H with attention → DeepSeek-V2/V3-style MLA:
             - KV down-up factored pair (W_DKV, W_UK) — natively factored
             - V up-projection (W_UV) shares W_DKV but is routed to plain Muon
               (CoupledMuon_v2.step's `processed` set rules out two coupled
                pairs sharing a B-partner; a 3-way (UK, UV, DKV) joint
                coupling is deferred to Phase 2.5)
             - Optional low-rank Q (W_DQ, W_UQ) when q_lora_rank > 0
             [headline novel architecture for Phase 2; the single rung where
              CoupledMuon couples *natively factored* weights — suggestion.md
              §1.2 + §2.4]

P_factff     A0 with each SwiGLU FFN replaced by an imposed
             (d_in, r) × (r, d_out) 2-mat factorisation at r=128, no gate.
             Couples (factff_down, factff_up) — the only learnable
             composition.
             [tests whether CoupledMuon's gain transfers from natively-
              factored (MLA) to imposed-factored (GaLore-style) FFN;
              suggestion.md §B3]

Z_350m_dense Scaled-up A0: hidden=1024, n_layers=18 ⇒ ~350M dense. SwiGLU,
             RMSNorm, RoPE, no QK-Norm. 7B tokens ≈ 20× Chinchilla.
             [size-axis validation; existing experiment.md d.5 §329
              commissioned recipe, now formally part of Phase 2]

Z_350m_mla   Z_350m_dense + MLA (kv_lora_rank=128, q_lora_rank=128) + MoE
             (8 experts top-2, every-other). ~480M total / ~110M active.
             [stretch rung for the MLA scaling-trend claim;
              suggestion.md §5 generous version, rescoped to 350M]
```

### d.7.2 Localisation rules added to d.2

- If the coupled gap survives at H but dies at **O_mla** → factor-aware
  coupling is *also* approximate on MLA (likely the K-channel partial-RoPE ×
  DKV interaction; examine `pair_factor_ratio` of (W_UK, W_DKV)).
- If the gap is **larger at O_mla than at I** → the factored-update reframing
  (suggestion.md §1.2) is empirically confirmed; the paper's central claim
  becomes "CoupledMuon is the natural optimizer for MLA's factored KV".
- If the gap survives at both **O_mla and P_factff** → factor-aware coupling
  transfers to *imposed* factorisation; CoupledMuon becomes a GaLore-class
  tool.
- If the gap survives at **O_mla but dies at P_factff** → CoupledMuon is
  strictly an MLA/LoRA tool; the paper scope narrows accordingly.
- If the gap at **Z_350m_mla < 0.5× the gap at O_mla** → scaling-trend claim
  weakens; the post-Wen-2025 "speedup vanishes with scale" critique applies
  and the headline must be reframed as small-scale.

### d.7.3 d.3 protocol additions

**Newton–Schulz coefficient policy.** Phase 1 fixed the Bernstein
(3.4445, −4.7750, 2.0315) quintic. Phase 2 adds two alternatives via the
`optimizer.ns_coefficients` knob:

- `polar_express` — degree-5 per-step coefficients from Amsel, Persson, Musco
  & Gower, "The Polar Express" (arXiv 2505.16932, ICLR 2026). Drives the
  polar iterate to ≈ 1 ≈ 2× faster at the spectral floor (`suggestion.md`
  §1.3).
- `cesista` — per-step optimised coefficients from Cesista, YouJiacheng &
  Jordan, "Squeezing 1–2% efficiency gains" (2025).

Coefficient tables are static per (policy, K) — see
`src/coupled_muon_nanogpt/optim/ns_coefficients.py`. **Phase-2 Polar-Express
and Cesista values are approximate reproductions**; canonical values must be
patched in from the published supplements before the headline paper run.

A Gram-NS form flag (`optimizer.ns_gram_form`) is wired through but the
actual Zhang–Amsel–Chen–Dao 2026 kernel is deferred to Phase 2.5 (the flag is
currently a passthrough to the standard form). Sweep YAMLs may set the flag
without surprises; numerics will match the standard NS form until the kernel
lands.

**LR-prefactor policy.** Phase 1 used `0.2·√max(d_out, d_in)`
(Moonlight Lemma 1). Phase 2 adds `optimizer.lr_prefactor ∈ {moonlight,
bernstein_ratio, cesista}` (default `moonlight` — Phase-1 bitwise identical):

- `moonlight`: `0.2 · √max(d_out, d_in)`.
- `bernstein_ratio`: `0.2 · √(d_out / d_in)` (Bernstein–Newhouse 2024 /
  Cesista per-row normalisation argument).
- `cesista`: `0.2 · √max(d_out, d_in) / (1 + log(K + 1))` — per-step
  normalised.

**Coupled-K curve.** Phase 1 fixed K=4. Phase 2 sweeps
K ∈ {1, 2, 3, 5, 8, 12} at A0 (with the winning S1 NS policy) to bound the
per-step cost story and test that the LLaMA-60M result is not K=4-specific.

**Pair-factor-ratio probe.** A new logging-only probe `pair_factor_ratio`
logs `||W_A||_F / ||W_B||_F` per coupled pair every
`pair_factor_ratio_interval_tokens`. Tests the LoRA-RITE one-factor-
dominates pathology (suggestion.md §2.7). Zero compute cost (two `.norm()`
calls per pair).

### d.7.4 d.4 ablation table additions

| Ablation | What it tells you |
|---|---|
| NS coefficient policy ∈ {bernstein, polar_express, cesista} × K ∈ {3, 5, 8} (S1) | Defends every CoupledMuon-vs-Muon delta against "Polar-Express closes the gap". 36 cells at A0 covering both plain Muon and CoupledMuon. |
| Gram-NS flag on plain Muon's stage 2 (S1 sub-axis) | Tests whether NS-on-G·Gᵀ improves vanilla Muon enough to close half the gap before MLA is even introduced. Currently a passthrough; the flag wires through for Phase 2.5 readiness. |
| Coupled-K curve at A0 (S3) | 18 cells = 6 K values × 3 seeds. Localises the K-curve to detect whether K=2 is sufficient or K=12 is required. |
| Pair-selection at production rung I (A1) | 15 cells × 3 pair policies (attn-only / FFN-only / all). Localises whether the gain is attention-side, FFN-side, or composite at production scale. |
| LR-prefactor policy ∈ {moonlight, bernstein_ratio, cesista} (A2) | 16 cells = 3 prefactors × 3 LRs × ~2 seeds at A0. |
| AdamW @ 5 LRs × 5 seeds at I and I' (A3) | +10 cells at I, +10 at I'. Closes the asymmetric-tuning attack against the Phase-1 5-vs-3 schedule. |
| LoRA-rank ∈ {16, 32, 64, 128, 256} on (W_UK, W_DKV) at O_mla (B1) | 15 cells = 5 ranks × 3 seeds. Bounds the rank-knee of CoupledMuon's gain. |
| Imposed FFN factorisation transfer (B3) | 12 cells = 2 factorisation states × 2 optimizers × 3 seeds. Tests whether the factor-aware advantage transfers from native (MLA) to imposed (FactFF) factor structure. |
| MLA at 350M scaling (Z_350m_mla) | 30 cells × {Muon, CoupledMuon} × 3 LRs × 5 seeds. Conditional dispatch (gating rule in DISPATCH.md). |

### d.7.5 d.5 compute-estimate rows

| Config | Params | Tokens | Wall-clock on R_prod (4×H200) | Comment |
|---|---|---|---|---|
| O_mla (60M-CS MLA MoE) | ~70M total / ~30M active | 5B | ~3.0 h/seed | Same dense backbone as I; MLA adds DKV/UK/UV/DQ/UQ params. |
| P_factff (A0 with imposed-factored FFN, r=128) | ~45M | 1.2B | ~1.0 h/seed on R_abl | A0 dense backbone; smaller because no gate. |
| Z_350m_dense | 350M | 7B | ~9 h/seed | hidden=1024, n_layers=18. |
| Z_350m_mla | ~480M total / ~110M active | 7B | ~12 h/seed | 350M + MLA + MoE 8 experts top-2. |

### d.7.6 Phase-2 scope, dependencies, termination criteria

**Scope** (suggestion.md §5 generous envelope; user-confirmed). Tier S
(S1 NS-policy, S2 MLA single rung, S3 K-curve) + Tier A (A1 pair-selection
at production, A2 LR-prefactor, A3 AdamW equalization) + Tier B subset (B1
LoRA-rank sweep, B3 imposed-FFN). Plus the 350M MLA stretch. **Out of
scope**: KDA / linear-attention (deferred to Phase 3), GPT-2 baseline, FFN
nonlinearity sweep, sliding-window attention, momentum/Nesterov sweeps,
multi-rung scale ladder (L/M sub-rungs).

**Dependency graph**:

- `S1 NS-policy winner` → propagates to S3, S2, A1, B1, and the 350M stretch
  as the default `optimizer.ns_coefficients` value. In-flight Bernstein
  cells from S2 remain valid as a "Bernstein baseline" half-grid; remaining
  cells re-launch with the winner.
- `O_mla S2 readout` → gates the 350M-stretch dispatch. Decision rule: at
  T+5.6 d, read out the bootstrap CI on median val-loss gap at the best LR
  per optimizer; commission stretch only if
  `gap_p50 > 1.5 × pooled_seed_std`.
- `A3` (AdamW equalization) is independent — runs in parallel with
  everything else.

**Termination criteria**. If O_mla shows no resolved CoupledMuon-vs-Muon gap
at matched LR/seeds, Phase 2 reframes the paper toward "Muon NS-policy is
the dominant axis; CoupledMuon is a modest factor-aware refinement when
factors are present" (matches the suggestion.md §0 reframing). Stop the
350M-stretch cells in that branch. S1 still produces a publishable
NS-policy result regardless of MLA outcome; the pair-factor-ratio probe
results stand as the LoRA-RITE empirical complement to the optimizer
comparison. **MLA UV is routed to plain Muon** (not coupled) because
`CoupledMuon_v2.step`'s `processed` set rules out two coupled pairs sharing
a B-partner; this is an honest scoping note — the K-side coupling is the
testable factored claim, and a 3-way (UK, UV, DKV) joint coupling kernel is
the Phase 2.5 algorithmic improvement if Phase 2 lands a positive result.