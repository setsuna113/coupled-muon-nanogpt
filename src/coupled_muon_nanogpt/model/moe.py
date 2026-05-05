"""Sparse MoE FFN with top-k routing, capacity-bounded dispatch, aux + z loss.

This is a clean, didactic MoE implementation — not a fused-kernel speed-runner.
Per experiment.md d.4 the router goes to AdamW (parameter naming `gate_router`
keeps it out of Muon path via the factory's "router" filter). The expert MLP
matrices follow the same `up_proj` / `down_proj` (and optional `gate_proj`)
naming so the optimizer factory's pair detection works on them too.

Balancing modes (`balancing_type`):
- `aux_loss`     — Switch-Transformer style auxiliary load-balance loss.
- `deepseek_bias`— Aux-loss-free, bias-update routing (DeepSeek-V3 style).
                   *Stub for now*: routes via aux_loss but exposes the bias parameter
                   that future work will update offline (per d.4 of experiment.md).

Returns aux losses through a side-channel; the training loop adds them to the LM
loss with the configured coefficients.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mlp import make_mlp


@dataclass
class MoEConfig:
    hidden: int
    intermediate: int
    num_experts: int
    top_k: int = 2
    capacity_factor: float = 1.25
    aux_loss_coef: float = 0.01
    z_loss_coef: float = 1e-3
    balancing_type: str = "aux_loss"  # aux_loss | deepseek_bias | smebu (smebu deferred)
    expert_mlp_type: str = "swiglu"


class MoEFFN(nn.Module):
    def __init__(self, cfg: MoEConfig):
        super().__init__()
        if cfg.balancing_type not in ("aux_loss", "deepseek_bias"):
            raise NotImplementedError(f"balancing_type={cfg.balancing_type!r} not in first-cut scope")
        self.cfg = cfg
        self.experts = nn.ModuleList(
            [make_mlp(cfg.hidden, cfg.intermediate, cfg.expert_mlp_type) for _ in range(cfg.num_experts)]
        )
        # Router weight: gate_router so the factory's "router" filter sends it to AdamW.
        self.gate_router = nn.Linear(cfg.hidden, cfg.num_experts, bias=False)
        if cfg.balancing_type == "deepseek_bias":
            # Bias term updated offline by the training loop (deferred).
            self.register_buffer("router_bias", torch.zeros(cfg.num_experts))
        self._last_aux_loss: torch.Tensor | None = None
        self._last_z_loss: torch.Tensor | None = None
        self._last_router_entropy: torch.Tensor | None = None
        self._last_expert_load: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        N = B * T
        x_flat = x.reshape(N, C)

        # Routing logits (fp32 for stability).
        logits = self.gate_router(x_flat).float()  # (N, E)
        if self.cfg.balancing_type == "deepseek_bias":
            logits = logits + self.router_bias  # type: ignore[operator]

        # z-loss: LSE^2 of router logits, encourages bounded router scale.
        z = torch.logsumexp(logits, dim=-1)
        self._last_z_loss = (z**2).mean() * self.cfg.z_loss_coef

        probs = F.softmax(logits, dim=-1)  # (N, E)
        topk_vals, topk_idx = probs.topk(self.cfg.top_k, dim=-1)  # (N, K)
        # Renormalise the top-k weights.
        topk_vals = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + 1e-9)

        # aux-loss: encourages uniform expert utilisation.
        # Pi = mean prob assigned to expert i; fi = fraction of tokens routed to i.
        E = self.cfg.num_experts
        with torch.no_grad():
            one_hot = F.one_hot(topk_idx, num_classes=E).float()  # (N, K, E)
            tokens_per_expert = one_hot.sum(dim=(0, 1))  # (E,)
        Pi = probs.mean(dim=0)  # (E,)
        fi = tokens_per_expert / (tokens_per_expert.sum() + 1e-9)
        aux = (Pi * fi).sum() * float(E)
        self._last_aux_loss = aux * self.cfg.aux_loss_coef
        self._last_expert_load = tokens_per_expert.detach()
        self._last_router_entropy = -(probs * (probs + 1e-9).log()).sum(dim=-1).mean().detach()

        # Capacity-bounded dispatch.
        capacity = max(1, int(self.cfg.capacity_factor * N * self.cfg.top_k / E))

        out = torch.zeros_like(x_flat)
        # Iterate per slot (k) then per expert. K is small (1–2), so the outer loop is cheap.
        for k_slot in range(self.cfg.top_k):
            slot_idx = topk_idx[:, k_slot]  # (N,)
            slot_w = topk_vals[:, k_slot]  # (N,)
            for e in range(E):
                token_mask = slot_idx == e
                if not token_mask.any():
                    continue
                token_pos = token_mask.nonzero(as_tuple=False).squeeze(-1)
                # Drop overflow beyond capacity.
                if token_pos.numel() > capacity:
                    token_pos = token_pos[:capacity]
                expert_in = x_flat[token_pos]
                expert_out = self.experts[e](expert_in)
                weighted = expert_out * slot_w[token_pos].unsqueeze(-1).to(expert_out.dtype)
                out.index_add_(0, token_pos, weighted)

        return out.view(B, T, C)

    def collect_losses(self) -> tuple[torch.Tensor, torch.Tensor]:
        aux = self._last_aux_loss if self._last_aux_loss is not None else torch.zeros((), device=self.gate_router.weight.device)
        z = self._last_z_loss if self._last_z_loss is not None else torch.zeros((), device=self.gate_router.weight.device)
        return aux, z
