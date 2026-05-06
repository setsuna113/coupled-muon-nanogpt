"""MoE per-expert metrics probe.

Surfaces the per-layer per-expert load (token count), router entropy, and
gradient Frobenius norm — the missing d.3 "per-expert load (token count),
per-expert grad norm, router entropy" signals plus a d.4-row-7 imbalance ratio.

Wired in train.py as a *needs-grads* probe so it fires between `loss.backward()`
and `optimizer.zero_grad()`. Under DDP it all-reduces `_last_expert_load` so the
reported counts are global across ranks. The probe gracefully no-ops on dense
runs (no MoEFFN modules in the model).
"""
from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

from ..model.moe import MoEFFN


@torch.no_grad()
def moe_load_probe(
    model: nn.Module,
    _optimizer: torch.optim.Optimizer,
    _ctx: dict[str, Any],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    use_dist = dist.is_available() and dist.is_initialized()

    for layer_idx, m in enumerate(_iter_moe_layers(model)):
        load = m._last_expert_load
        entropy = m._last_router_entropy
        if load is None or entropy is None:
            # MoEFFN forward hasn't run this step (rare; e.g. dropped under capacity).
            continue
        load = load.detach().float().clone()
        if use_dist:
            dist.all_reduce(load, op=dist.ReduceOp.SUM)
        # Per-expert grad norms (zeroed if probe fires post-zero_grad).
        grad_norms = m.grad_norms().detach().float()
        if use_dist and grad_norms.numel() > 0:
            dist.all_reduce(grad_norms, op=dist.ReduceOp.SUM)
            grad_norms = grad_norms / float(dist.get_world_size())

        load_list = load.tolist()
        load_max = max(load_list) if load_list else 0.0
        load_mean = sum(load_list) / len(load_list) if load_list else 0.0
        imbalance = (load_max / load_mean) if load_mean > 0 else float("nan")

        gn_list = grad_norms.tolist()
        gn_mean = sum(gn_list) / len(gn_list) if gn_list else 0.0
        gn_var = (
            sum((x - gn_mean) ** 2 for x in gn_list) / len(gn_list) if gn_list else 0.0
        )
        gn_norm_var = (gn_var ** 0.5) / gn_mean if gn_mean > 0 else 0.0

        out[f"layer.{layer_idx}"] = {
            "loads": load_list,
            "imbalance": imbalance,
            "router_entropy": float(entropy.detach().item()),
            "grad_norms": gn_list,
            "grad_norm_var": gn_norm_var,
        }
    return out


def _iter_moe_layers(model: nn.Module):
    for module in model.modules():
        if isinstance(module, MoEFFN):
            yield module
