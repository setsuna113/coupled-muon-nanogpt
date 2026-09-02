"""Plain Muon optimizer (no coupling).

Extracted as a thin wrapper around the same `zeropower_via_newtonschulz5` kernel
used by CoupledMuon_v2, so that runs with `optimizer.type=muon` share numerics
with `coupled_muon_v2` minus the coupling step.
"""
from __future__ import annotations

import math

import torch

from .coupled_muon import zeropower_via_newtonschulz5


class Muon(torch.optim.Optimizer):
    """Standard Muon: Newton-Schulz orthogonalisation of momentum-buffered gradients
    for matrix params, with an AdamW backup for everything else (1D, embeddings, head).

    Faithful to KellerJordan/Muon: lr is multiplied by `0.2 * sqrt(max(A, B))` for
    each matrix-shaped parameter (Moonlight Lemma 1 scaling).
    """

    def __init__(
        self,
        muon_params: list[torch.nn.Parameter],
        adamw_params: list[torch.nn.Parameter] | None = None,
        lr: float = 1e-3,
        wd: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        adamw_betas: tuple[float, float] = (0.95, 0.95),
        adamw_eps: float = 1e-8,
        ns_dtype: torch.dtype = torch.bfloat16,
        # Phase-2 NS policy / Gram-form / LR-prefactor knobs (cf. CoupledMuon_v2).
        # Defaults preserve Phase-1 numerics exactly (`bernstein` ⇒ `coeffs=None`
        # passed to the kernel ⇒ hardcoded Bernstein triple).
        ns_coefficients: str = "bernstein",
        ns_gram_form: bool = False,
        lr_prefactor: str = "moonlight",
    ):
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
        all_params = list(muon_params) + list(adamw_params)
        super().__init__(all_params, defaults)
        self.ns_dtype = ns_dtype
        for p in muon_params:
            assert p.ndim == 2, f"Muon expects 2D matrices, got {tuple(p.shape)}"
            self.state[p]["use_muon"] = True
        for p in adamw_params:
            self.state[p]["use_muon"] = False

        from .ns_coefficients import coefficients_to_tensor, get_coefficients

        self.ns_coefficients_policy = str(ns_coefficients).lower()
        self.ns_gram_form = bool(ns_gram_form)
        self.lr_prefactor = str(lr_prefactor).lower()
        self._ns_steps = int(ns_steps)
        if self.ns_coefficients_policy == "bernstein":
            self._coeffs = None
        else:
            self._coeffs = coefficients_to_tensor(
                get_coefficients(self.ns_coefficients_policy, self._ns_steps)
            )
        if self.ns_gram_form:
            import warnings
            warnings.warn(
                "optimizer.ns_gram_form=true requested but the Phase-2 Gram-NS "
                "kernel is a passthrough to the standard form. The flag is "
                "forwarded for compatibility; numerics match standard NS.",
                stacklevel=2,
            )

    def adjust_lr_for_muon(self, lr: float, shape: tuple[int, ...]) -> float:
        """Per-parameter LR scaling; matches the CoupledMuon_v2 policy menu.

        Default (`moonlight`) ⇒ ``0.2·√max(A,B)`` (Moonlight 2502.16982 §2.2
        Eq. 4 — verified: ``W_t = W_{t-1} − η_t (0.2·O_t·√max(A,B) + λ W_{t-1})``).

        `bernstein_ratio` ⇒ ``0.2·√(d_out/d_in)`` (Jeremy Bernstein,
        "Deriving Muon", https://jeremybernste.in/writing/deriving-muon —
        verified: blog defines ``W ← W - η·√(fan-out/fan-in)·NewtonSchulz(∇)``;
        the ``0.2`` factor adopts Moonlight's RMS-matching constant on top of
        Bernstein's dimensional ratio).

        `cesista` ⇒ ``0.2·√max(A,B) / (1 + log(ns_steps + 1))`` —
        **NOTE: this is a project-local heuristic, not from any Cesista
        publication.** The Cesista work optimises NS *coefficients* per step
        (see ``ns_coefficients.py``), not the outer LR prefactor. We retain
        the name for sweep-config compatibility, but the formula is a
        log-damped Moonlight scaling and should not be cited as Cesista's.
        """
        a, b = shape[:2]
        policy = getattr(self, "lr_prefactor", "moonlight")
        if policy == "moonlight":
            return lr * 0.2 * math.sqrt(max(a, b))
        if policy == "bernstein_ratio":
            return lr * 0.2 * math.sqrt(a / max(b, 1))
        if policy == "cesista":
            K = max(int(getattr(self, "_ns_steps", 5)), 1)
            return lr * 0.2 * math.sqrt(max(a, b)) / (1.0 + math.log(K + 1))
        raise ValueError(
            f"Unknown lr_prefactor={policy!r}. "
            f"Supported: moonlight | bernstein_ratio | cesista."
        )

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["wd"]
            momentum = group["momentum"]
            ns_steps = group["ns_steps"]

            # Muon (matrix) params
            for p in (q for q in group["params"] if self.state[q].get("use_muon", False)):
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g_eff = g.add(buf, alpha=momentum) if group["nesterov"] else buf
                u = zeropower_via_newtonschulz5(g_eff, steps=ns_steps, dtype=self.ns_dtype, coeffs=self._coeffs, gram_form=self.ns_gram_form)
                adj_lr = self.adjust_lr_for_muon(lr, p.shape)
                p.data.mul_(1 - lr * wd).add_(u, alpha=-adj_lr)

            # AdamW backup
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            for p in (q for q in group["params"] if not self.state[q].get("use_muon", False)):
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
                m1 = state["moment1"]
                m2 = state["moment2"]
                m1.lerp_(g, 1 - beta1)
                m2.lerp_(g.square(), 1 - beta2)
                g_hat = m1 / (eps + m2.sqrt())
                bc1 = 1 - beta1**step
                bc2 = 1 - beta2**step
                scale = bc1 / bc2**0.5
                p.data.mul_(1 - lr * wd).add_(g_hat, alpha=-lr / scale)

        return loss
