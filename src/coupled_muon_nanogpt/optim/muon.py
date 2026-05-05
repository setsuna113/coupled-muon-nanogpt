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
        for p in muon_params:
            assert p.ndim == 2, f"Muon expects 2D matrices, got {tuple(p.shape)}"
            self.state[p]["use_muon"] = True
        for p in adamw_params:
            self.state[p]["use_muon"] = False

    @staticmethod
    def adjust_lr_for_muon(lr: float, shape: tuple[int, ...]) -> float:
        a, b = shape[:2]
        return lr * 0.2 * math.sqrt(max(a, b))

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
                u = zeropower_via_newtonschulz5(g_eff, steps=ns_steps)
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
