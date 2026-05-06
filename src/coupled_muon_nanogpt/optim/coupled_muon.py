import math
import torch
import os
from datetime import datetime


# This code snippet is a modified version adapted from the following GitHub repository:
# https://github.com/KellerJordan/Muon/blob/master/muon.py
@torch.compile
def zeropower_via_newtonschulz5(G, steps, dtype=torch.bfloat16):
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
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.to(dtype)
    if G.size(0) > G.size(1):
        X = X.T
    # Ensure spectral norm is at most 1
    X = X / (X.norm() + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.T
        B = (
            b * A + c * A @ A
        )  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X



@torch.compile
def coupled_newtonschulz5_B(M_B, A, steps, dtype=torch.bfloat16):
    """
    Coupled Newton-Schulz for matrix B with coupling to matrix A.
    M_{B,t+1} = 3.4445 M_{B,t} - 4.7750 (M_{B,t} M_{B,t}^T A^T A) M_{B,t} + 2.0315 (M_{B,t} M_{B,t}^T A^T A)^2 M_{B,t}
    """
    assert len(M_B.shape) == len(A.shape)
    #assert len(M_B.shape) in (2, 3, 4)
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = M_B.to(dtype)
    A = A.to(dtype)

    if len(M_B.shape) == 2:
        # Normalize
        #X = X / (X.norm()*A_bf16.norm() + 1e-7)
        X = X / ((A @ X).norm() + 1e-7)

        for step_idx in range(steps):
            # Compute M_{B,t} M_{B,t}^T A^T A
            W = A @ X
            WTW = W.T @ W
            # B = b * MMATA + c * MMATA^2
            T = b * WTW + c * (WTW @ WTW)
            X = a * X + X @ T
    else:
        W = A @ X
        X = X / (W.norm(dim=(-2,-1)).unsqueeze(-1).unsqueeze(-1) + 1e-7)

        ATA = A.transpose(-1, -2) @ A
        for step_idx in range(steps):
            XXT = X @ X.transpose(-1, -2)
            MMATA = XXT @ ATA
            B = b * MMATA + c * (MMATA @ MMATA)
            X = a * X + B @ X
    # Returned dtype matches the kernel dtype the caller chose; the consuming code
    # casts back when it writes into the (typically bf16) parameter buffer.
    return X


@torch.compile
def coupled_newtonschulz5_A(M_A, B, steps, dtype=torch.bfloat16):
    """
    Coupled Newton-Schulz for matrix A with coupling to matrix B.
    M_{A,t+1} = 3.4445 M_{A,t} - 4.7750 M_{A,t} (B B^T M_{A,t}^T M_{A,t}) + 2.0315 M_{A,t} (B B^T M_{A,t}^T M_{A,t})^2
    """
    assert len(M_A.shape) == len(B.shape)
    assert len(M_A.shape) in (2, 3, 4)
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = M_A.to(dtype)
    B = B.to(dtype)

    if len(M_A.shape) == 2:
        # Normalize
        #X = X / (X.norm()*B_bf16.norm() + 1e-7)
        X = X / ((X @ B).norm() + 1e-7)
        for step_idx in range(steps):
            W = X @ B
            WWT = W @ W.T
            T = b * WWT + c * (WWT @ WWT)
            X = a * X + T @ X
    else:
        W = X @ B
        X = X / (W.norm(dim=(-2, -1)).unsqueeze(-1).unsqueeze(-1) + 1e-7)

        BBT = B @ B.transpose(-1, -2)
        for step_idx in range(steps):
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
        for param_A, param_B, n_heads, is_qk in coupled_pairs:
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
        self.n_heads = n_heads
        self.ns_dtype = ns_dtype
        # crossfade configuration
        self.iter_num = 0
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
        A, B = param_shape[:2]
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        return lr * adjusted_ratio

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
                # still pre-divides by ‖XB‖_F (lines 56, 93), producing a degraded
                # update that is *neither* coupled-NS nor plain Muon.
                if self.coupled_steps > 0 and self.enable_coupled:
                    if self.use_multi_head and param_A.shape[1] % state_A["num_heads"] == 0 and param_B.shape[0] % state_B["num_heads"] == 0:
                        if is_qk:
                            HD, I = param_A.shape
                            HD_B, I_B = param_B.shape
                            head_dim = HD_B // n_heads
                            if HD == HD_B:
                                g_A_reshaped = g_A_eff.view(state_A["num_heads"], 2, head_dim//2, I).permute(0, 2, 3, 1).contiguous()
                                g_B_reshaped = g_B_eff.view(state_B["num_heads"], 2, head_dim//2, I).permute(0, 2, 1, 3).contiguous()
                                p_A_reshaped = param_A.data.view(state_A["num_heads"], 2, head_dim//2, I).permute(0, 2, 3, 1).contiguous()
                                p_B_reshaped = param_B.data.view(state_B["num_heads"], 2, head_dim//2, I).permute(0, 2, 1, 3).contiguous()

                                u_A_c = coupled_newtonschulz5_A(g_A_reshaped, p_B_reshaped, steps=self.coupled_steps, dtype=self.ns_dtype)
                                u_B_c = coupled_newtonschulz5_B(g_B_reshaped, p_A_reshaped, steps=self.coupled_steps, dtype=self.ns_dtype)

                                u_A_c = u_A_c.permute(0, 3, 1, 2).contiguous().reshape(HD, I)
                                u_B_c = u_B_c.permute(0, 2, 1, 3).contiguous().reshape(HD_B, I_B)
                            else:
                                if HD > HD_B:
                                    n_heads_group = HD // HD_B
                                    g_A_reshaped = g_A_eff.view(n_heads_group, state_B["num_heads"], 2, head_dim//2, I).permute(0, 1, 3, 4, 2).contiguous()
                                    g_B_reshaped = g_B_eff.view(1, state_B["num_heads"], 2, head_dim//2, I).permute(0, 1, 3, 2, 4).contiguous()
                                    p_A_reshaped = param_A.data.view(n_heads_group, state_B["num_heads"], 2, head_dim//2, I).permute(0, 1, 3, 4, 2).contiguous()
                                    p_B_reshaped = param_B.data.view(1, state_B["num_heads"], 2, head_dim//2, I).permute(0, 1, 3, 2, 4).contiguous()

                                    u_A_c = coupled_newtonschulz5_A(g_A_reshaped, p_B_reshaped, steps=self.coupled_steps, dtype=self.ns_dtype)
                                    u_B_c = coupled_newtonschulz5_B(g_B_reshaped, p_A_reshaped, steps=self.coupled_steps, dtype=self.ns_dtype)

                                    u_A_c = u_A_c.permute(0, 1, 4, 2, 3).contiguous().reshape(HD, I)
                                    u_B_c = u_B_c.mean(dim=0).permute(0, 2, 1, 3).contiguous().reshape(HD_B, I_B)
                                else:
                                    n_heads_group = HD_B // HD
                                    g_A_reshaped = g_A_eff.view(1, state_B["num_heads"] // n_heads_group, 2, head_dim // 2, I).permute(0, 1, 3, 4, 2).contiguous()
                                    g_B_reshaped = g_B_eff.view(n_heads_group, state_B["num_heads"] // n_heads_group, 2, head_dim // 2, I).permute(0, 1, 3, 2, 4).contiguous()
                                    p_A_reshaped = param_A.data.view(1, state_B["num_heads"] // n_heads_group, 2, head_dim // 2, I).permute(0, 1, 3, 4, 2).contiguous()
                                    p_B_reshaped = param_B.data.view(n_heads_group, state_B["num_heads"] // n_heads_group, 2, head_dim // 2, I).permute(0, 1, 3, 2, 4).contiguous()

                                    u_A_c = coupled_newtonschulz5_A(g_A_reshaped, p_B_reshaped, steps=self.coupled_steps, dtype=self.ns_dtype)
                                    u_B_c = coupled_newtonschulz5_B(g_B_reshaped, p_A_reshaped, steps=self.coupled_steps, dtype=self.ns_dtype)

                                    u_A_c = u_A_c.mean(dim=0).permute(0, 3, 1, 2).contiguous().reshape(HD, I)
                                    u_B_c = u_B_c.permute(0, 1, 3, 2, 4).contiguous().reshape(HD_B, I_B)
                        else:
                            O, HD = param_A.shape
                            HD_B, I = param_B.shape
                            assert HD == HD_B
                            head_dim = HD // state_A["num_heads"]

                            g_A_reshaped = g_A_eff.view(O, state_A["num_heads"], head_dim).permute(1, 0, 2)
                            g_B_reshaped = g_B_eff.view(state_B["num_heads"], head_dim, I)
                            p_A_reshaped = param_A.data.view(O, state_A["num_heads"], head_dim).permute(1, 0, 2)
                            p_B_reshaped = param_B.data.view(state_B["num_heads"], head_dim, I)

                            u_A_c = coupled_newtonschulz5_A(g_A_reshaped, p_B_reshaped, steps=self.coupled_steps, dtype=self.ns_dtype)
                            u_B_c = coupled_newtonschulz5_B(g_B_reshaped, p_A_reshaped, steps=self.coupled_steps, dtype=self.ns_dtype)

                            u_A_c = u_A_c.permute(1, 0, 2).reshape(O, HD)
                            u_B_c = u_B_c.reshape(HD_B, I)
                    else:
                        u_A_c = coupled_newtonschulz5_A(g_A_eff, param_B.data, steps=self.coupled_steps, dtype=self.ns_dtype)
                        u_B_c = coupled_newtonschulz5_B(g_B_eff, param_A.data, steps=self.coupled_steps, dtype=self.ns_dtype)

                    # Stage-1 norm capture: per-pair Frobenius norm of the
                    # post-coupled-NS iterate, cheap (~one .norm() call). Probe
                    # reads from state every `stage1_norm_interval_tokens`.
                    state_A["stage1_frob_norm"] = u_A_c.detach().float().norm().item()
                    state_B["stage1_frob_norm"] = u_B_c.detach().float().norm().item()

                    if self.final_polish:
                        # Stage 2 (default): plain NS on stage-1 output, so
                        # ‖U_A‖_spec ≈ 1 and the matrix-size LR scaling is
                        # well-defined. See class docstring; experiment.md a.2.
                        u_A = zeropower_via_newtonschulz5(u_A_c, steps=ns_steps, dtype=self.ns_dtype)
                        u_B = zeropower_via_newtonschulz5(u_B_c, steps=ns_steps, dtype=self.ns_dtype)
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
                    u_A = zeropower_via_newtonschulz5(g_A_eff, steps=ns_steps, dtype=self.ns_dtype)
                    u_B = zeropower_via_newtonschulz5(g_B_eff, steps=ns_steps, dtype=self.ns_dtype)


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

                u = zeropower_via_newtonschulz5(g, steps=ns_steps, dtype=self.ns_dtype)

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


