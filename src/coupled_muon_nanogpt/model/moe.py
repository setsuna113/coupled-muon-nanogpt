"""Sparse MoE FFN with top-k routing, capacity-bounded dispatch, aux + z loss.

This is a clean, didactic MoE implementation — not a fused-kernel speed-runner.
The router parameter is named `gate_router` so the optimizer factory's
"router" filter (`optim/factory.py`) routes it to Muon by default
(matching Moonlight 2502.16982 §3.4 "Dynamics of Singular Spectrum" —
which observes routers benefit more from Muon than other matrices — and
Cerebras nanoMoE). Setting `optimizer.couple_router_to_muon: false` flips
it to AdamW (rung J's DeepSeek-V2/V3 / OLMoE ablation). The expert MLP matrices follow the same
`up_proj` / `down_proj` (and optional `gate_proj`) naming so the factory's
pair detection works on them too.

Balancing modes (`balancing_type`):
- `aux_loss`     — Switch-Transformer style auxiliary load-balance loss:
                   aux = E · Σ_i Pi · stop_grad(fi). The gradient flows ONLY
                   through Pi (the router probabilities), so aux directly
                   contaminates **router** weights — not the expert MLPs.
                   Rung K therefore tests router-side contamination only.
                   experiment.md c.4 conjectures contamination of expert MLP
                   weights too; that would require a different aux formulation
                   (e.g. an activation-magnitude term) and is out of scope here.
- `deepseek_bias`— Aux-loss-free, sign-rule bias-update routing
                   (DeepSeek-V2 §3.2 / DeepSeek-V3). The training loop calls
                   `update_router_bias()` after each `optimizer.step()` to nudge
                   the per-expert bias toward uniform load. The aux-loss coefficient
                   is forced to 0 in this mode.

Shared-expert mode (`shared_expert`):
- When `shared_expert: True`, an additional always-on expert processes every
  token at every step (DeepSeekMoE-style); top-k routing then selects from the
  remaining `num_experts` experts. The shared-expert path uses the same
  `up_proj`/`down_proj` naming and is automatically pair-detected by the
  optimizer factory.

Returns aux losses through a side-channel; the training loop adds them to the LM
loss with the configured coefficients.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
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
    balancing_type: str = "aux_loss"  # aux_loss | deepseek_bias
    expert_mlp_type: str = "swiglu"
    bias_update_lr: float = 1.0e-3      # DeepSeek-V2 §3.2 sign-rule step size
    shared_expert: bool = False          # DeepSeekMoE-style always-on expert
    shared_expert_intermediate: int = 0  # 0 ⇒ inherit from `intermediate`


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
            # Per-expert bias added to routing logits; updated offline by
            # update_router_bias() per DeepSeek-V2 §3.2 sign rule.
            self.register_buffer("router_bias", torch.zeros(cfg.num_experts))
        if cfg.shared_expert:
            shared_inter = (
                cfg.shared_expert_intermediate if cfg.shared_expert_intermediate > 0 else cfg.intermediate
            )
            self.shared_expert = make_mlp(cfg.hidden, shared_inter, cfg.expert_mlp_type)
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
        # Per c.4: in deepseek_bias mode, aux loss contaminates the LM gradient
        # with load-balancing bias. Force it to 0 regardless of aux_loss_coef.
        if self.cfg.balancing_type == "deepseek_bias":
            self._last_aux_loss = torch.zeros((), device=logits.device)
        else:
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
                # Under autocast, expert_out can be bf16 while `out` (zeros_like(x_flat))
                # is fp32 because reshape doesn't trigger autocast. index_add_ requires
                # matching dtypes, so cast weighted into `out`'s dtype.
                out.index_add_(0, token_pos, weighted.to(out.dtype))

        # Always-on shared expert (DeepSeekMoE).
        if self.cfg.shared_expert:
            shared_out = self.shared_expert(x_flat)
            out = out + shared_out.to(out.dtype)

        return out.view(B, T, C)

    def collect_losses(self) -> tuple[torch.Tensor, torch.Tensor]:
        aux = self._last_aux_loss if self._last_aux_loss is not None else torch.zeros((), device=self.gate_router.weight.device)
        z = self._last_z_loss if self._last_z_loss is not None else torch.zeros((), device=self.gate_router.weight.device)
        return aux, z

    @torch.no_grad()
    def grad_norms(self) -> torch.Tensor:
        """Per-expert gradient Frobenius norm: ‖∇W_up_i ⊕ ∇W_down_i (⊕ ∇W_gate_i)‖_F.

        Must be called between `loss.backward()` and `optimizer.zero_grad()` —
        consumes the `.grad` tensors directly. Returns a 1-D tensor of length
        `num_experts`. Used by `probes.moe_load` to log per-expert gradient
        signal magnitudes (experiment.md d.3 "per-expert grad norm")."""
        # `sq` must be a CUDA scalar even when the expert has no grads this
        # step (top-k routing can leave an expert untouched in a microbatch).
        # Initialising with `0.0` and relying on `torch.as_tensor` made the
        # untouched-expert path produce a CPU scalar; stacking it with
        # touched-expert CUDA scalars raised a device-mismatch RuntimeError.
        device = self.gate_router.weight.device
        norms = []
        for expert in self.experts:
            sq = torch.zeros((), device=device, dtype=torch.float32)
            for p in expert.parameters():
                if p.grad is None:
                    continue
                sq = sq + p.grad.detach().float().pow(2).sum()
            norms.append(sq.sqrt())
        return torch.stack(norms) if norms else torch.zeros(0, device=device)

    @torch.no_grad()
    def update_router_bias(self) -> None:
        """DeepSeek-V2 §3.2 aux-loss-free balancing: nudge `router_bias` toward
        the value that equalises per-expert load.

        Algorithm:
            load_i  = number of (token, slot) pairs routed to expert i
            f_i     = load_i / Σ load_j         # sums to 1
            f_mean  = 1 / num_experts           # target under f_i sums-to-1
            router_bias_i ← router_bias_i − γ · sign(f_i − f_mean)

        Note on the target: with K = top_k > 1, an alternative normalisation
        gives f_i' = load_i / N (number of tokens) which sums to K and has
        target K / E. We use the f_i-sums-to-1 normalisation because that's
        what `load.sum()` produces directly; the `K / E` formula was a bug
        — under our normalisation every f_i was below K/E for K>1 and biases
        drifted uniformly upward (a no-op due to softmax shift-invariance,
        but also nondiagnostic).

        Under DDP the per-rank `_last_expert_load` is summed across ranks
        before the update so all ranks stay in sync.
        """
        if self.cfg.balancing_type != "deepseek_bias":
            return
        load = self._last_expert_load
        if load is None:
            return
        load = load.detach().float().clone()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(load, op=dist.ReduceOp.SUM)
        total = load.sum().clamp_min(1e-9)
        f_i = load / total                        # sums to 1
        f_mean = 1.0 / float(self.cfg.num_experts)  # target under that normalisation
        gamma = float(self.cfg.bias_update_lr)
        # `router_bias` is a buffer registered in __init__ when balancing_type == deepseek_bias.
        self.router_bias.add_(-gamma * torch.sign(f_i - f_mean).to(self.router_bias.dtype))
