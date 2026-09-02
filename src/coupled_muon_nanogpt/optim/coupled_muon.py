import math

import torch


# This code snippet is a modified version adapted from the following GitHub repository:
# https://github.com/KellerJordan/Muon/blob/master/muon.py
#
# Phase-2 extension: accepts an optional per-step coefficient tensor `coeffs`
# of shape (K, 3) for the NS-policy sweep (see optim/ns_coefficients.py).
# When `coeffs is None`, falls through to the canonical Bernstein triple —
# preserves Phase-1 numerics bitwise. The `gram_form` flag is reserved for the
# Zhang–Amsel–Chen–Dao 2026 Gram-NS variant; the real kernel is deferred to
# Phase 2.5 (the standard form is already X·X^T-based internally, so the
# expected speedup is marginal at the user's matrix sizes — implementation
# left to the actual Gram-NS coefficient table being read). The flag is wired
# through so sweep YAMLs can set it without a code change later.
@torch.compile
def zeropower_via_newtonschulz5(G, steps, dtype=torch.bfloat16, coeffs=None, gram_form: bool = False):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    X = G.to(dtype)
    if G.size(0) > G.size(1):
        X = X.T
    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    if coeffs is None:
        a, b, c = (3.4445, -4.7750, 2.0315)
        for _ in range(steps):
            A = X @ X.T
            B = (
                b * A + c * A @ A
            )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
            X = a * X + B @ X
    else:
        coeffs = coeffs.to(device=X.device)
        for k in range(steps):
            a = coeffs[k, 0]
            b = coeffs[k, 1]
            c = coeffs[k, 2]
            A = X @ X.T
            B = b * A + c * A @ A
            X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X



@torch.compile
def coupled_newtonschulz5_B(M_B, A, steps, dtype=torch.bfloat16, coeffs=None):
    """
    Coupled Newton-Schulz for matrix B with coupling to matrix A.
    M_{B,t+1} = 3.4445 M_{B,t} - 4.7750 (M_{B,t} M_{B,t}^T A^T A) M_{B,t} + 2.0315 (M_{B,t} M_{B,t}^T A^T A)^2 M_{B,t}

    Phase-2 extension: optional per-step coefficient tensor `coeffs` of shape
    (K, 3); when None, falls back to the Bernstein triple (Phase-1 numerics).
    """
    assert len(M_B.shape) == len(A.shape)
    #assert len(M_B.shape) in (2, 3, 4)
    X = M_B.to(dtype)
    A = A.to(dtype)

    if len(M_B.shape) == 2:
        # Normalize
        #X = X / (X.norm()*A_bf16.norm() + 1e-7)
        X = X / ((A @ X).norm() + 1e-7)

        if coeffs is None:
            a, b, c = (3.4445, -4.7750, 2.0315)
            for step_idx in range(steps):
                # Compute M_{B,t} M_{B,t}^T A^T A
                W = A @ X
                WTW = W.T @ W
                # B = b * MMATA + c * MMATA^2
                T = b * WTW + c * (WTW @ WTW)
                X = a * X + X @ T
        else:
            coeffs = coeffs.to(device=X.device)
            for step_idx in range(steps):
                a = coeffs[step_idx, 0]
                b = coeffs[step_idx, 1]
                c = coeffs[step_idx, 2]
                W = A @ X
                WTW = W.T @ W
                T = b * WTW + c * (WTW @ WTW)
                X = a * X + X @ T
    else:
        W = A @ X
        X = X / (W.norm(dim=(-2,-1)).unsqueeze(-1).unsqueeze(-1) + 1e-7)

        ATA = A.transpose(-1, -2) @ A
        if coeffs is None:
            a, b, c = (3.4445, -4.7750, 2.0315)
            for step_idx in range(steps):
                XXT = X @ X.transpose(-1, -2)
                MMATA = XXT @ ATA
                B = b * MMATA + c * (MMATA @ MMATA)
                X = a * X + B @ X
        else:
            coeffs = coeffs.to(device=X.device)
            for step_idx in range(steps):
                a = coeffs[step_idx, 0]
                b = coeffs[step_idx, 1]
                c = coeffs[step_idx, 2]
                XXT = X @ X.transpose(-1, -2)
                MMATA = XXT @ ATA
                B = b * MMATA + c * (MMATA @ MMATA)
                X = a * X + B @ X
    # Returned dtype matches the kernel dtype the caller chose; the consuming code
    # casts back when it writes into the (typically bf16) parameter buffer.
    return X


@torch.compile
def coupled_newtonschulz5_A(M_A, B, steps, dtype=torch.bfloat16, coeffs=None):
    """
    Coupled Newton-Schulz for matrix A with coupling to matrix B.
    M_{A,t+1} = 3.4445 M_{A,t} - 4.7750 M_{A,t} (B B^T M_{A,t}^T M_{A,t}) + 2.0315 M_{A,t} (B B^T M_{A,t}^T M_{A,t})^2

    Phase-2 extension: optional per-step coefficient tensor `coeffs` of shape
    (K, 3); when None, falls back to the Bernstein triple (Phase-1 numerics).
    """
    assert len(M_A.shape) == len(B.shape)
    assert len(M_A.shape) in (2, 3, 4)
    X = M_A.to(dtype)
    B = B.to(dtype)

    if len(M_A.shape) == 2:
        # Normalize
        #X = X / (X.norm()*B_bf16.norm() + 1e-7)
        X = X / ((X @ B).norm() + 1e-7)
        if coeffs is None:
            a, b, c = (3.4445, -4.7750, 2.0315)
            for step_idx in range(steps):
                W = X @ B
                WWT = W @ W.T
                T = b * WWT + c * (WWT @ WWT)
                X = a * X + T @ X
        else:
            coeffs = coeffs.to(device=X.device)
            for step_idx in range(steps):
                a = coeffs[step_idx, 0]
                b = coeffs[step_idx, 1]
                c = coeffs[step_idx, 2]
                W = X @ B
                WWT = W @ W.T
                T = b * WWT + c * (WWT @ WWT)
                X = a * X + T @ X
    else:
        W = X @ B
        X = X / (W.norm(dim=(-2, -1)).unsqueeze(-1).unsqueeze(-1) + 1e-7)

        BBT = B @ B.transpose(-1, -2)
        if coeffs is None:
            a, b, c = (3.4445, -4.7750, 2.0315)
            for step_idx in range(steps):
                XTX = X.transpose(-1, -2) @ X
                BBTMM = BBT @ XTX
                B_term = b * BBTMM + c * BBTMM @ BBTMM
                X = a * X + X @ B_term
        else:
            coeffs = coeffs.to(device=X.device)
            for step_idx in range(steps):
                a = coeffs[step_idx, 0]
                b = coeffs[step_idx, 1]
                c = coeffs[step_idx, 2]
                XTX = X.transpose(-1, -2) @ X
                BBTMM = BBT @ XTX
                B_term = b * BBTMM + c * BBTMM @ BBTMM
                X = a * X + X @ B_term
    return X




class CoupledMuon_v2(torch.optim.Optimizer):
    """
    Coupled Muon optimizer for paired matrix updates (e.g., V and O in transformers).

    The update is a **two-stage preconditioner**: U_A = NS(C_A(G_A; B)).

      1. Stage 1 (coupled NS, `C_A`/`C_B`): finds an iterate X such that X·B
         converges to polar(G_A·B). The search direction is informed by the
         partner weight; the *product* A·B moves in its polar direction.
         Stage-1 X has no spectral-norm guarantee on itself — its scale is
         whatever it needs to be to make X·B orthogonal, and depends on the
         conditioning of B.
      2. Stage 2 (plain NS, controlled by `final_polish`): orthogonalises X
         itself so ‖U_A‖_spec ≈ 1. This is what makes the matrix-size LR
         scaling `lr × 0.2 × √max(rows, cols)` meaningful — that scaling
         assumes a unit-spectral-norm update. Without stage 2, update
         magnitude per parameter would depend on the partner weight's
         conditioning, which is uncontrolled and drifts during training.

    Stage 1 picks the geometry; stage 2 enforces the scale. Whether stage 2
    is doing meaningful work or is approximately a no-op (because stage-1's
    output happens to be near-orthogonal at the operating point) is an
    empirical question — the `final_polish` toggle exists to ablate it.
    Note: "steepest descent on the variety {AB}" is heuristic, not derived;
    see experiment.md a.2 caveat and Appendix X for the rigorous picture.

    Stage-1 recurrences (mirrored across A, B):
      M_{B,t+1} = a M_{B,t} + (b (M_{B,t} M_{B,t}^T A^T A) + c (...)^2) M_{B,t}
      M_{A,t+1} = a M_{A,t} + M_{A,t} (b (B B^T M_{A,t}^T M_{A,t}) + c (...)^2)

    Arguments:
        lr: Learning rate
        wd: Weight decay
        coupled_pairs: List of tuples (param_A, param_B, n_heads, is_qk).
        muon_params: The parameters to be optimized by plain Muon.
        momentum: Momentum coefficient
        nesterov: Use Nesterov momentum
        ns_steps: Newton-Schulz iteration steps for stage 2 (and for plain Muon path)
        coupled_steps: Newton-Schulz iteration steps for stage 1 (coupled kernel).
        adamw_params: Parameters optimized by AdamW fallback
        adamw_betas: AdamW beta parameters
        adamw_eps: AdamW epsilon
        enable_coupled: If False, treat coupled pairs with standard Muon updates (no coupling). If True, use coupled updates.
        final_polish: If True (default), apply plain NS after the coupled kernel
            (the two-stage composition described above — current code's
            historical behavior, the operating point that produced the
            LLaMA-60M result). If False, skip stage 2 and use the coupled
            iterate directly. See experiment.md d.4 for the planned ablation.
        use_multi_head: Reshape Q/K (RoPE 2-D pairs) and V/O (per head) before
            stage 1 so the coupling sees the per-head bilinear structure.
        n_heads: Number of attention heads (for reshape branching).
        ns_dtype: dtype for Newton-Schulz iterates.
    """

    def __init__(
        self,
        lr=1e-3,
        wd=0.1,
        coupled_pairs=None,
        muon_params=None,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
        coupled_steps=5,
        adamw_params=None,
        adamw_betas=(0.95, 0.95),
        adamw_eps=1e-8,
        # master switch to enable/disable coupled behavior
        enable_coupled: bool = True,
        # Two-stage composition toggle (see class docstring; experiment.md a.2/d.4).
        # True (default) preserves the Muon(C_A(g; B)) update — the operating
        # point that produced the LLaMA-60M result. False uses the coupled
        # iterate u_A_c directly without the trailing zeropower NS pass.
        final_polish: bool = True,
        use_multi_head=False,
        n_heads=1,
        # dtype for Newton-Schulz iterates (per experiment.md d.6.3 fp32 diagnostic).
        ns_dtype: torch.dtype = torch.bfloat16,
        # Phase-2 NS-policy / LR-prefactor axes. Defaults preserve Phase-1
        # numerics exactly (`bernstein` ⇒ `coeffs=None` is passed to the NS
        # kernels; `moonlight` ⇒ existing `0.2·√max(A,B)` LR scaling;
        # `ns_gram_form=False` ⇒ standard form). See optim/ns_coefficients.py
        # and experiment.md d.3 for the policy menu.
        ns_coefficients: str = "bernstein",
        ns_gram_form: bool = False,
        lr_prefactor: str = "moonlight",
        # Phase-2.1 Q-K policy interface (experiment.md d.7.3). Three keys:
        #   full_rope: legacy_rope2d | off
        #     Controls Q-K branch when rope_status=="full". Default
        #     ``legacy_rope2d`` dispatches to the unmodified Phase-1 path
        #     (preserves A0/B/G/H bitwise numerics). ``off`` demotes Q-K to
        #     plain Muon (explicit ablation; review v3 §"explicit-off").
        #   no_rope_policy: current_flat2d_fallback | qk_off |
        #                   headwise_no_rope    (partial_rope_split: INVALID)
        #     Controls Q-K branch when rope_status=="none" (learned-pos).
        #   partial_rope_policy: current_flat2d_fallback | qk_off |
        #                       headwise_no_rope | partial_rope_split
        #     Controls Q-K branch when rope_status=="partial".
        # ``rope_status`` and ``rotary_dim`` are computed by the factory and
        # passed in; ``rope_status`` is one of {"full", "partial", "none"}.
        # V-O routing under rope_status in {partial, none} ALWAYS goes to the
        # legacy flat-2D path regardless of qk_coupling choice (hard rule;
        # review v3.4) — Phase B is Q-K-only.
        qk_coupling_full_rope: str = "legacy_rope2d",
        qk_coupling_no_rope_policy: str = "current_flat2d_fallback",
        qk_coupling_partial_rope_policy: str = "current_flat2d_fallback",
        rope_status: str = "full",
        rotary_dim: int | None = None,
    ):
        coupled_pairs = coupled_pairs or []
        muon_params = muon_params or []
        adamw_params = adamw_params or []
        
        defaults = dict(
            lr=lr,
            wd=wd,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
        )

        all_params = []
        for param_A, param_B, _pair_heads, _pair_is_qk in coupled_pairs:
            all_params.extend([param_A, param_B])
        all_params.extend(muon_params)
        all_params.extend(adamw_params)
        
        super().__init__(all_params, defaults)
        
        # Store coupling information
        self.coupled_pairs = coupled_pairs
        self.coupled_steps = coupled_steps
        # master switch to control coupled behavior
        self.enable_coupled = enable_coupled
        # Stage-2 (final polish) toggle. See class docstring; ablated per d.4.
        self.final_polish = bool(final_polish)
        self.use_multi_head = use_multi_head
        # Constructor kwarg. (The pair loop above used to shadow this name, so this
        # attribute previously held the LAST coupled pair's per-pair head count;
        # nothing reads it either way — per-pair counts live in self.state.)
        self.n_heads = n_heads
        self.ns_dtype = ns_dtype
        # crossfade configuration
        self.iter_num = 0

        # Phase-2 NS-policy bookkeeping. Precompute the coefficient tensors
        # once at __init__; passed into the NS kernels every step (broadcast
        # to the right device on first use). When the policy is `bernstein`,
        # we deliberately keep coeffs=None so the kernel falls back to the
        # hardcoded Phase-1 path — bitwise-identical numerics.
        from .ns_coefficients import coefficients_to_tensor, get_coefficients

        self.ns_coefficients_policy = str(ns_coefficients).lower()
        self.ns_gram_form = bool(ns_gram_form)
        self.lr_prefactor = str(lr_prefactor).lower()

        # Phase-2.1 Q-K policy bookkeeping.
        self.rope_status = str(rope_status).lower()
        if self.rope_status not in ("full", "partial", "none"):
            raise ValueError(
                f"Unknown rope_status={self.rope_status!r}. "
                f"Expected one of: full | partial | none."
            )
        self.qk_coupling_full_rope = str(qk_coupling_full_rope).lower()
        if self.qk_coupling_full_rope not in ("legacy_rope2d", "off"):
            raise ValueError(
                f"Unknown qk_coupling.full_rope={self.qk_coupling_full_rope!r}. "
                f"Expected one of: legacy_rope2d | off."
            )
        self.qk_coupling_no_rope_policy = str(qk_coupling_no_rope_policy).lower()
        _no_rope_valid = ("current_flat2d_fallback", "qk_off", "headwise_no_rope")
        if self.qk_coupling_no_rope_policy not in _no_rope_valid:
            raise ValueError(
                f"Unknown qk_coupling.no_rope_policy={self.qk_coupling_no_rope_policy!r}. "
                f"Expected one of: {' | '.join(_no_rope_valid)}. "
                f"(partial_rope_split is INVALID for rope_status=none.)"
            )
        self.qk_coupling_partial_rope_policy = str(qk_coupling_partial_rope_policy).lower()
        _partial_rope_valid = (
            "current_flat2d_fallback", "qk_off", "headwise_no_rope", "partial_rope_split",
        )
        if self.qk_coupling_partial_rope_policy not in _partial_rope_valid:
            raise ValueError(
                f"Unknown qk_coupling.partial_rope_policy={self.qk_coupling_partial_rope_policy!r}. "
                f"Expected one of: {' | '.join(_partial_rope_valid)}."
            )
        # Hard rule (review v3.4): V-O routing under rope_status in {partial,
        # none} ALWAYS uses the legacy flat-2D path. Internally derive a V-O
        # specific multi-head flag so Phase B sweeps that set use_multi_head=true
        # do not accidentally promote V-O to per-head reshape under non-full-RoPE.
        self._use_multi_head_vo = bool(use_multi_head) and self.rope_status == "full"
        self.rotary_dim = rotary_dim
        # partial_rope_policy=partial_rope_split is INERT under rope_status
        # in {full, none} — the dispatch never reaches it. Only validate
        # rotary_dim when it will actually be used (rope_status=partial).
        if (
            self.rope_status == "partial"
            and self.qk_coupling_partial_rope_policy == "partial_rope_split"
        ):
            if self.rotary_dim is None or self.rotary_dim <= 0:
                raise ValueError(
                    "partial_rope_split requires rotary_dim > 0 (factory must "
                    "thread it from rope_partial_frac × head_dim)."
                )

        if self.ns_coefficients_policy == "bernstein":
            self._coeffs_stage1 = None
            self._coeffs_stage2 = None
        else:
            self._coeffs_stage1 = coefficients_to_tensor(
                get_coefficients(self.ns_coefficients_policy, int(coupled_steps))
            )
            self._coeffs_stage2 = coefficients_to_tensor(
                get_coefficients(self.ns_coefficients_policy, int(ns_steps))
            )

        if self.ns_gram_form:
            # Reserved for the Zhang–Amsel–Chen–Dao 2026 Gram-NS variant.
            # Phase-2 wires the flag through but the kernel is currently a
            # passthrough to the standard form. Tracking this in `experiment.md`
            # so sweep YAMLs can light up the knob without surprises.
            import warnings
            warnings.warn(
                "optimizer.ns_gram_form=true requested but the Phase-2 Gram-NS "
                "kernel is currently a passthrough to the standard form. The "
                "flag is forwarded for compatibility; numerics match standard "
                "NS. See optim/ns_coefficients.py docstring.",
                stacklevel=2,
            )
        for param_A, param_B, n_heads, is_qk in coupled_pairs:
            assert param_A.ndim == 2 and param_B.ndim == 2
            self.state[param_A]["use_coupled"] = True
            self.state[param_A]["coupled_with"] = param_B
            self.state[param_A]["is_A"] = True
            self.state[param_A]["is_qk"] = is_qk
            self.state[param_A]["num_heads"] = n_heads
            self.state[param_B]["use_coupled"] = True
            self.state[param_B]["coupled_with"] = param_A
            self.state[param_B]["is_A"] = False
            self.state[param_B]["num_heads"] = n_heads
            self.state[param_B]["is_qk"] = is_qk
        
        # Mark standard Muon params
        for p in muon_params:
            assert p.ndim == 2
            self.state[p]["use_muon"] = True
            self.state[p]["use_coupled"] = False
        
        # Mark AdamW params
        for p in adamw_params:
            self.state[p]["use_coupled"] = False
            self.state[p]["use_muon"] = False

    def set_coupled_enabled(self, enabled: bool):
        """Enable or disable the coupled update logic at runtime.

        When disabled (False), parameters in `coupled_pairs` will be updated using
        standard Muon orthogonalized updates, identical to those in `src.optim.muon`.
        """
        self.enable_coupled = enabled

    def adjust_lr_for_muon(self, lr, param_shape):
        """Per-parameter LR scaling. The default (`moonlight`) matches Phase-1's
        ``0.2·√max(A,B)`` (Moonlight 2502.16982 §2.2 Eq. 4 —
        ``W_t = W_{t-1} − η_t (0.2·O_t·√max(A,B) + λW_{t-1})``). Phase-2 adds
        two alternatives:

        - ``bernstein_ratio``: ``0.2·√(d_out/d_in)`` — Jeremy Bernstein,
          "Deriving Muon" (https://jeremybernste.in/writing/deriving-muon)
          defines ``W ← W − η·√(fan-out/fan-in)·NewtonSchulz(∇)``; we keep
          Moonlight's ``0.2`` RMS-matching factor on top of Bernstein's
          dimensional ratio.
        - ``cesista``: ``0.2·√max(A,B) / (1 + log(K+1))`` — **project-local
          heuristic, not from any Cesista publication**. Cesista's work
          optimises NS *coefficients* per step (see ``ns_coefficients.py``),
          not the outer LR prefactor. Name retained for sweep compatibility.
        """
        A, B = param_shape[:2]
        policy = self.lr_prefactor
        if policy == "moonlight":
            adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        elif policy == "bernstein_ratio":
            # Avoid divide-by-zero on degenerate shapes.
            denom = max(B, 1)
            adjusted_ratio = 0.2 * math.sqrt(A / denom)
        elif policy == "cesista":
            # K is the stage-1 (coupled) step count and is used for EVERY parameter
            # this optimizer updates — including the plain-Muon block and the
            # coupled_steps==0 path, where it clamps to 1. optim/muon.py uses
            # ns_steps for K instead, so `cesista` prefactors would differ between
            # muon and coupled_muon_v2 runs when coupled_steps != ns_steps. Never
            # exercised: the lr_prefactor sweep was not launched (FINAL_STATUS.md).
            K = max(int(self.coupled_steps), 1)
            adjusted_ratio = 0.2 * math.sqrt(max(A, B)) / (1.0 + math.log(K + 1))
        else:
            raise ValueError(
                f"Unknown lr_prefactor={policy!r}. "
                f"Supported: moonlight | bernstein_ratio | cesista."
            )
        return lr * adjusted_ratio

    # ------------------------------------------------------------------
    # Phase-2.1 Q-K coupling helpers.
    # `is_qk=True` pairs dispatch through `_qk_stage1_dispatch`, which returns
    # the post-stage-1 iterates (u_A_c, u_B_c). For `current_flat2d_fallback`,
    # `qk_off`, `legacy_rope2d`, and `off` the implementations dispatch to the
    # EXACT existing kernels so the legacy code paths are reused bit-for-bit
    # (review v2.5 / v3.8). `headwise_no_rope` and `partial_rope_split` are
    # new code paths gated on the corresponding policy values.
    # ------------------------------------------------------------------
    def _qk_legacy_rope2d_stage1(self, g_A_eff, g_B_eff, param_A, param_B, state_A, state_B, n_heads):
        """Legacy Phase-1 RoPE-2D-block multi-head coupled stage-1. Reproduces
        the pre-refactor branch at coupled_muon.py:470-509 unchanged.
        """
        HD, I = param_A.shape
        HD_B, I_B = param_B.shape
        head_dim = HD_B // n_heads
        if HD == HD_B:
            g_A_reshaped = g_A_eff.view(state_A["num_heads"], 2, head_dim//2, I).permute(0, 2, 3, 1).contiguous()
            g_B_reshaped = g_B_eff.view(state_B["num_heads"], 2, head_dim//2, I).permute(0, 2, 1, 3).contiguous()
            p_A_reshaped = param_A.data.view(state_A["num_heads"], 2, head_dim//2, I).permute(0, 2, 3, 1).contiguous()
            p_B_reshaped = param_B.data.view(state_B["num_heads"], 2, head_dim//2, I).permute(0, 2, 1, 3).contiguous()
            u_A_c = coupled_newtonschulz5_A(g_A_reshaped, p_B_reshaped, steps=self.coupled_steps, dtype=self.ns_dtype, coeffs=self._coeffs_stage1)
            u_B_c = coupled_newtonschulz5_B(g_B_reshaped, p_A_reshaped, steps=self.coupled_steps, dtype=self.ns_dtype, coeffs=self._coeffs_stage1)
            u_A_c = u_A_c.permute(0, 3, 1, 2).contiguous().reshape(HD, I)
            u_B_c = u_B_c.permute(0, 2, 1, 3).contiguous().reshape(HD_B, I_B)
            return u_A_c, u_B_c
        # GQA broadcast cases (preserved verbatim from legacy code).
        if HD > HD_B:
            n_heads_group = HD // HD_B
            g_A_reshaped = g_A_eff.view(n_heads_group, state_B["num_heads"], 2, head_dim//2, I).permute(0, 1, 3, 4, 2).contiguous()
            g_B_reshaped = g_B_eff.view(1, state_B["num_heads"], 2, head_dim//2, I).permute(0, 1, 3, 2, 4).contiguous()
            p_A_reshaped = param_A.data.view(n_heads_group, state_B["num_heads"], 2, head_dim//2, I).permute(0, 1, 3, 4, 2).contiguous()
            p_B_reshaped = param_B.data.view(1, state_B["num_heads"], 2, head_dim//2, I).permute(0, 1, 3, 2, 4).contiguous()
            u_A_c = coupled_newtonschulz5_A(g_A_reshaped, p_B_reshaped, steps=self.coupled_steps, dtype=self.ns_dtype, coeffs=self._coeffs_stage1)
            u_B_c = coupled_newtonschulz5_B(g_B_reshaped, p_A_reshaped, steps=self.coupled_steps, dtype=self.ns_dtype, coeffs=self._coeffs_stage1)
            u_A_c = u_A_c.permute(0, 1, 4, 2, 3).contiguous().reshape(HD, I)
            u_B_c = u_B_c.mean(dim=0).permute(0, 2, 1, 3).contiguous().reshape(HD_B, I_B)
            return u_A_c, u_B_c
        # HD < HD_B
        n_heads_group = HD_B // HD
        g_A_reshaped = g_A_eff.view(1, state_B["num_heads"] // n_heads_group, 2, head_dim // 2, I).permute(0, 1, 3, 4, 2).contiguous()
        g_B_reshaped = g_B_eff.view(n_heads_group, state_B["num_heads"] // n_heads_group, 2, head_dim // 2, I).permute(0, 1, 3, 2, 4).contiguous()
        p_A_reshaped = param_A.data.view(1, state_B["num_heads"] // n_heads_group, 2, head_dim // 2, I).permute(0, 1, 3, 4, 2).contiguous()
        p_B_reshaped = param_B.data.view(n_heads_group, state_B["num_heads"] // n_heads_group, 2, head_dim // 2, I).permute(0, 1, 3, 2, 4).contiguous()
        u_A_c = coupled_newtonschulz5_A(g_A_reshaped, p_B_reshaped, steps=self.coupled_steps, dtype=self.ns_dtype, coeffs=self._coeffs_stage1)
        u_B_c = coupled_newtonschulz5_B(g_B_reshaped, p_A_reshaped, steps=self.coupled_steps, dtype=self.ns_dtype, coeffs=self._coeffs_stage1)
        u_A_c = u_A_c.mean(dim=0).permute(0, 3, 1, 2).contiguous().reshape(HD, I)
        u_B_c = u_B_c.permute(0, 1, 3, 2, 4).contiguous().reshape(HD_B, I_B)
        return u_A_c, u_B_c

    def _qk_flat2d_stage1(self, g_A_eff, g_B_eff, param_A, param_B):
        """Legacy flat-2D coupling (no head reshape). Reproduces the pre-refactor
        fallback at coupled_muon.py:527-528 unchanged — load-bearing for the
        invariant `current_flat2d_fallback` bitwise-equals legacy on C/D/E.
        """
        u_A_c = coupled_newtonschulz5_A(g_A_eff, param_B.data, steps=self.coupled_steps, dtype=self.ns_dtype, coeffs=self._coeffs_stage1)
        u_B_c = coupled_newtonschulz5_B(g_B_eff, param_A.data, steps=self.coupled_steps, dtype=self.ns_dtype, coeffs=self._coeffs_stage1)
        return u_A_c, u_B_c

    def _qk_qk_off_stage1(self, g_A_eff, g_B_eff, ns_steps):
        """Demote Q-K to plain Muon. To make the final write to Q,K
        BITWISE-IDENTICAL to a plain-Muon-step on the same gradients, we
        return the raw gradients here when ``final_polish=True`` and let the
        outer NS5 pass do all the orthogonalisation work. When
        ``final_polish=False``, we polish here directly (the outer code
        would otherwise apply no polish). Total NS work = ``ns_steps`` in
        both cases (one pass), matching the plain-Muon path."""
        if self.final_polish:
            return g_A_eff, g_B_eff
        u_A_c = zeropower_via_newtonschulz5(
            g_A_eff, steps=ns_steps, dtype=self.ns_dtype,
            coeffs=self._coeffs_stage2, gram_form=self.ns_gram_form,
        )
        u_B_c = zeropower_via_newtonschulz5(
            g_B_eff, steps=ns_steps, dtype=self.ns_dtype,
            coeffs=self._coeffs_stage2, gram_form=self.ns_gram_form,
        )
        return u_A_c, u_B_c

    def _qk_headwise_no_rope_stage1(self, g_A_eff, g_B_eff, param_A, param_B, state_A, state_B):
        """Per-head 3-D coupling without RoPE 2-D rotation-pair split.

        Reshape Q, K as (n_heads, head_dim, in_features); the partner-aware
        kernel sees the per-head Q·K^T product geometry. Only the MHA case
        (n_heads_A == n_heads_B) is implemented; GQA falls back to
        ``coupled_newtonschulz5_A``/``_B`` flat-2D path for now.
        """
        HD, I = param_A.shape
        HD_B, I_B = param_B.shape
        n_heads_A = state_A["num_heads"]
        n_heads_B = state_B["num_heads"]
        if n_heads_A != n_heads_B or HD != HD_B:
            # GQA / asymmetric — fall back to flat-2D coupling (safe default).
            return self._qk_flat2d_stage1(g_A_eff, g_B_eff, param_A, param_B)
        head_dim = HD // n_heads_A
        g_A_3d = g_A_eff.view(n_heads_A, head_dim, I).contiguous()
        g_B_3d = g_B_eff.view(n_heads_B, head_dim, I).contiguous()
        p_A_3d = param_A.data.view(n_heads_A, head_dim, I).contiguous()
        p_B_3d = param_B.data.view(n_heads_B, head_dim, I).contiguous()
        # Partners are transposed so X @ B yields the (head_dim, head_dim) per-head product.
        p_A_3d_T = p_A_3d.transpose(-1, -2).contiguous()
        p_B_3d_T = p_B_3d.transpose(-1, -2).contiguous()
        u_A_3d = coupled_newtonschulz5_A(g_A_3d, p_B_3d_T, steps=self.coupled_steps, dtype=self.ns_dtype, coeffs=self._coeffs_stage1)
        u_B_3d = coupled_newtonschulz5_A(g_B_3d, p_A_3d_T, steps=self.coupled_steps, dtype=self.ns_dtype, coeffs=self._coeffs_stage1)
        u_A_c = u_A_3d.reshape(HD, I)
        u_B_c = u_B_3d.reshape(HD_B, I_B)
        return u_A_c, u_B_c

    def _qk_partial_rope_split_stage1(self, g_A_eff, g_B_eff, param_A, param_B, state_A, state_B, n_heads, ns_steps):
        """Split per-head head_dim into RoPE'd [0:rotary_dim] and non-RoPE
        [rotary_dim:] slices. RoPE'd slice → legacy 2-D rotation-pair coupled
        NS. Non-RoPE slice → uncoupled NS (no partner coupling; the safe choice per
 review v3 §"partial_rope_split semantics"). NOTE the NS budget: with
 final_polish=True this branch pre-applies coupled_steps + ns_steps passes
 AND the outer stage-2 pass in step() polishes the concatenated iterate
 again, so the non-RoPE channels receive coupled_steps + 2*ns_steps NS
 passes — unlike _qk_qk_off_stage1, which returns raw grads under
 final_polish. Retained as-is because the H' partial_rope_split results
 in docs/FINAL_STATUS.md §4.5 were produced with it; changing it is a
 new experiment, not a cleanup.
        """
        HD, I = param_A.shape
        HD_B, I_B = param_B.shape
        n_heads_A = state_A["num_heads"]
        n_heads_B = state_B["num_heads"]
        if n_heads_A != n_heads_B or HD != HD_B:
            # GQA not supported in partial_rope_split; fall back to current_flat2d.
            return self._qk_flat2d_stage1(g_A_eff, g_B_eff, param_A, param_B)
        head_dim = HD // n_heads_A
        rope_dim = int(self.rotary_dim) if self.rotary_dim is not None else head_dim
        rope_dim = max(2, min(head_dim, rope_dim & ~1))  # even, in [2, head_dim]
        non_rope_dim = head_dim - rope_dim
        # View per-head head_dim.
        g_A_view = g_A_eff.view(n_heads_A, head_dim, I)
        g_B_view = g_B_eff.view(n_heads_B, head_dim, I)
        p_A_view = param_A.data.view(n_heads_A, head_dim, I)
        p_B_view = param_B.data.view(n_heads_B, head_dim, I)
        # RoPE slice [0:rope_dim] — legacy 2-D rotation-pair block coupling.
        g_A_rope = g_A_view[:, :rope_dim, :].reshape(n_heads_A, 2, rope_dim // 2, I).permute(0, 2, 3, 1).contiguous()
        g_B_rope = g_B_view[:, :rope_dim, :].reshape(n_heads_B, 2, rope_dim // 2, I).permute(0, 2, 1, 3).contiguous()
        p_A_rope = p_A_view[:, :rope_dim, :].reshape(n_heads_A, 2, rope_dim // 2, I).permute(0, 2, 3, 1).contiguous()
        p_B_rope = p_B_view[:, :rope_dim, :].reshape(n_heads_B, 2, rope_dim // 2, I).permute(0, 2, 1, 3).contiguous()
        u_A_rope_4d = coupled_newtonschulz5_A(g_A_rope, p_B_rope, steps=self.coupled_steps, dtype=self.ns_dtype, coeffs=self._coeffs_stage1)
        u_B_rope_4d = coupled_newtonschulz5_B(g_B_rope, p_A_rope, steps=self.coupled_steps, dtype=self.ns_dtype, coeffs=self._coeffs_stage1)
        u_A_rope = u_A_rope_4d.permute(0, 3, 1, 2).contiguous().reshape(n_heads_A, rope_dim, I)
        u_B_rope = u_B_rope_4d.permute(0, 2, 1, 3).contiguous().reshape(n_heads_B, rope_dim, I)
        if non_rope_dim == 0:
            u_A_3d = u_A_rope
            u_B_3d = u_B_rope
        else:
            # Non-RoPE slice [rope_dim:] — plain Muon (no coupling).
            g_A_norope = g_A_view[:, rope_dim:, :].reshape(n_heads_A * non_rope_dim, I).contiguous()
            g_B_norope = g_B_view[:, rope_dim:, :].reshape(n_heads_B * non_rope_dim, I).contiguous()
            steps = self.coupled_steps + ns_steps if self.final_polish else self.coupled_steps
            u_A_norope_flat = zeropower_via_newtonschulz5(g_A_norope, steps=steps, dtype=self.ns_dtype, coeffs=self._coeffs_stage2, gram_form=self.ns_gram_form)
            u_B_norope_flat = zeropower_via_newtonschulz5(g_B_norope, steps=steps, dtype=self.ns_dtype, coeffs=self._coeffs_stage2, gram_form=self.ns_gram_form)
            u_A_norope = u_A_norope_flat.view(n_heads_A, non_rope_dim, I)
            u_B_norope = u_B_norope_flat.view(n_heads_B, non_rope_dim, I)
            u_A_3d = torch.cat([u_A_rope, u_A_norope], dim=1)
            u_B_3d = torch.cat([u_B_rope, u_B_norope], dim=1)
        u_A_c = u_A_3d.reshape(HD, I)
        u_B_c = u_B_3d.reshape(HD_B, I_B)
        return u_A_c, u_B_c

    def _qk_stage1_dispatch(self, g_A_eff, g_B_eff, param_A, param_B, state_A, state_B, n_heads, ns_steps):
        """Dispatch for is_qk=True pairs based on (rope_status, qk_coupling)."""
        if self.rope_status == "full":
            if self.qk_coupling_full_rope == "legacy_rope2d":
                return self._qk_legacy_rope2d_stage1(g_A_eff, g_B_eff, param_A, param_B, state_A, state_B, n_heads)
            elif self.qk_coupling_full_rope == "off":
                return self._qk_qk_off_stage1(g_A_eff, g_B_eff, ns_steps)
            else:
                raise ValueError(f"Unreachable: qk_coupling.full_rope={self.qk_coupling_full_rope!r}")
        # rope_status in {"partial", "none"}
        if self.rope_status == "partial":
            policy = self.qk_coupling_partial_rope_policy
        else:
            policy = self.qk_coupling_no_rope_policy
        if policy == "current_flat2d_fallback":
            return self._qk_flat2d_stage1(g_A_eff, g_B_eff, param_A, param_B)
        if policy == "qk_off":
            return self._qk_qk_off_stage1(g_A_eff, g_B_eff, ns_steps)
        if policy == "headwise_no_rope":
            return self._qk_headwise_no_rope_stage1(g_A_eff, g_B_eff, param_A, param_B, state_A, state_B)
        if policy == "partial_rope_split":
            return self._qk_partial_rope_split_stage1(g_A_eff, g_B_eff, param_A, param_B, state_A, state_B, n_heads, ns_steps)
        raise ValueError(f"Unreachable: qk policy={policy!r} for rope_status={self.rope_status!r}")

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            ############################
            #     Coupled Muon         #
            ############################

            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]
            ns_steps = group["ns_steps"]
            
            # Process coupled pairs
            processed = set()
            for param_A, param_B, n_heads, is_qk in self.coupled_pairs:
                if param_A in processed or param_B in processed:
                    continue
                    
                g_A = param_A.grad
                g_B = param_B.grad
                
                if g_A is None or g_B is None:
                    continue
                
                # Momentum for both
                state_A = self.state[param_A]
                state_B = self.state[param_B]
                
                if "momentum_buffer" not in state_A:
                    state_A["momentum_buffer"] = torch.zeros_like(g_A)
                if "momentum_buffer" not in state_B:
                    state_B["momentum_buffer"] = torch.zeros_like(g_B)
                
                buf_A = state_A["momentum_buffer"]
                buf_B = state_B["momentum_buffer"]
                
                buf_A.mul_(momentum).add_(g_A)
                buf_B.mul_(momentum).add_(g_B)
                
                if group["nesterov"]:
                    g_A_eff = g_A.add(buf_A, alpha=momentum)
                    g_B_eff = g_B.add(buf_B, alpha=momentum)
                else:
                    g_A_eff = buf_A
                    g_B_eff = buf_B


                # When coupled_steps == 0 or coupling globally disabled, the contract
                # (base.yaml: "0 ⇒ plain Muon path inside CoupledMuon_v2") is plain Muon.
                # Skip the partner-aware kernels entirely — calling them with steps=0
                # still pre-divides by the partner-product norm inside coupled_newtonschulz5_A/_B, producing a degraded
                # update that is *neither* coupled-NS nor plain Muon.
                if self.coupled_steps > 0 and self.enable_coupled:
                    if is_qk and self.use_multi_head:
                        # Phase-2.1: dispatch on (rope_status, qk_coupling).
                        # When rope_status=="full" + full_rope=="legacy_rope2d",
                        # this dispatches bit-for-bit to the legacy Phase-1
                        # 2-D rotation-pair multi-head path.
                        u_A_c, u_B_c = self._qk_stage1_dispatch(
                            g_A_eff, g_B_eff, param_A, param_B, state_A, state_B, n_heads, ns_steps,
                        )
                    elif (not is_qk) and self._use_multi_head_vo and param_A.shape[1] % state_A["num_heads"] == 0 and param_B.shape[0] % state_B["num_heads"] == 0:
                        # V-O / FFN per-head reshape — only active under
                        # rope_status=="full" (hard rule v3.4: V-O routing under
                        # non-full-RoPE always falls through to flat-2D).
                        O, HD = param_A.shape
                        HD_B, I = param_B.shape
                        assert HD == HD_B
                        head_dim = HD // state_A["num_heads"]

                        g_A_reshaped = g_A_eff.view(O, state_A["num_heads"], head_dim).permute(1, 0, 2)
                        g_B_reshaped = g_B_eff.view(state_B["num_heads"], head_dim, I)
                        p_A_reshaped = param_A.data.view(O, state_A["num_heads"], head_dim).permute(1, 0, 2)
                        p_B_reshaped = param_B.data.view(state_B["num_heads"], head_dim, I)

                        u_A_c = coupled_newtonschulz5_A(g_A_reshaped, p_B_reshaped, steps=self.coupled_steps, dtype=self.ns_dtype, coeffs=self._coeffs_stage1)
                        u_B_c = coupled_newtonschulz5_B(g_B_reshaped, p_A_reshaped, steps=self.coupled_steps, dtype=self.ns_dtype, coeffs=self._coeffs_stage1)

                        u_A_c = u_A_c.permute(1, 0, 2).reshape(O, HD)
                        u_B_c = u_B_c.reshape(HD_B, I)
                    else:
                        u_A_c = coupled_newtonschulz5_A(g_A_eff, param_B.data, steps=self.coupled_steps, dtype=self.ns_dtype, coeffs=self._coeffs_stage1)
                        u_B_c = coupled_newtonschulz5_B(g_B_eff, param_A.data, steps=self.coupled_steps, dtype=self.ns_dtype, coeffs=self._coeffs_stage1)

                    # Stage-1 norm capture, kept because
                    # tests/test_final_polish_toggle.py asserts it. No probe reads it —
                    # the `stage1_norm_interval_tokens` key once planned here was never
                    # implemented. Under qk_off + final_polish=True the value is the
                    # momentum-buffered gradient norm (that path returns g_eff as-is).
                    state_A["stage1_frob_norm"] = u_A_c.detach().float().norm().item()
                    state_B["stage1_frob_norm"] = u_B_c.detach().float().norm().item()

                    if self.final_polish:
                        # Stage 2 (default): plain NS on stage-1 output, so
                        # ‖U_A‖_spec ≈ 1 and the matrix-size LR scaling is
                        # well-defined. See class docstring; experiment.md a.2.
                        u_A = zeropower_via_newtonschulz5(u_A_c, steps=ns_steps, dtype=self.ns_dtype, coeffs=self._coeffs_stage2, gram_form=self.ns_gram_form)
                        u_B = zeropower_via_newtonschulz5(u_B_c, steps=ns_steps, dtype=self.ns_dtype, coeffs=self._coeffs_stage2, gram_form=self.ns_gram_form)
                    else:
                        # Stage-1-only ablation (experiment.md d.4 final_polish row):
                        # use the coupled iterate directly. ‖u‖_spec is uncontrolled;
                        # the LR scaling lr × 0.2 × √max(A,B) becomes a partner-
                        # weight-conditioned magnitude rather than a unit-norm step.
                        u_A = u_A_c.to(self.ns_dtype)
                        u_B = u_B_c.to(self.ns_dtype)
                else:
                    # coupled_steps == 0 or enable_coupled == False: plain Muon path
                    # on the same momentum-buffered gradients the standard-Muon block uses.
                    u_A = zeropower_via_newtonschulz5(g_A_eff, steps=ns_steps, dtype=self.ns_dtype, coeffs=self._coeffs_stage2, gram_form=self.ns_gram_form)
                    u_B = zeropower_via_newtonschulz5(g_B_eff, steps=ns_steps, dtype=self.ns_dtype, coeffs=self._coeffs_stage2, gram_form=self.ns_gram_form)


                # Apply updates
                adjusted_lr_A = self.adjust_lr_for_muon(lr, param_A.shape)
                adjusted_lr_B = self.adjust_lr_for_muon(lr, param_B.shape)
                
                param_A.data.mul_(1 - lr * wd).add_(u_A, alpha=-adjusted_lr_A) #param_A × (1 - lr × wd) - adjusted_lr_A × u_A
                param_B.data.mul_(1 - lr * wd).add_(u_B, alpha=-adjusted_lr_B)
                
                processed.add(param_A)
                processed.add(param_B)
            ############################
            #      Standard Muon       #
            ############################

            params = [p for p in group["params"] if self.state[p].get("use_muon", False)]
            
            for p in params:
                g = p.grad
                if g is None:
                    continue
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)
                assert g is not None
                
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                
                if group["nesterov"]:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf

                u = zeropower_via_newtonschulz5(g, steps=ns_steps, dtype=self.ns_dtype, coeffs=self._coeffs_stage2, gram_form=self.ns_gram_form)

                adjusted_lr = self.adjust_lr_for_muon(lr, p.shape)
                p.data.mul_(1 - lr * wd).add_(u, alpha=-adjusted_lr)

            ############################
            #       AdamW backup       #
            ############################

            params = [p for p in group["params"] if not self.state[p].get("use_coupled", False) and not self.state[p].get("use_muon", False)]
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]

            for p in params:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * wd).add_(g, alpha=-lr / scale)

        # advance internal iteration counter for crossfade bookkeeping
        self.iter_num += 1
        return loss


