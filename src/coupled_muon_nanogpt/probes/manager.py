"""Token-relative probe scheduler.

Probes are callables `(model, optimizer, ctx) -> dict[str, float | list[float]]`.
The manager fires each probe when the cumulative token count crosses the next
multiple of its `interval_tokens`. This decouples logging cadence from step
count, so smoke (small batch) and prod (large batch) configs log at comparable
token-rate.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

ProbeFn = Callable[[nn.Module, torch.optim.Optimizer, dict[str, Any]], dict[str, Any]]


@dataclass
class ProbeSpec:
    name: str
    fn: ProbeFn
    interval_tokens: int
    # If True, the probe must fire BEFORE optimizer.zero_grad() — it consumes
    # `.grad` tensors directly. Per-expert grad-norm probes need this.
    needs_grads: bool = False
    last_fire_tokens: int = 0


class ProbeManager:
    def __init__(self):
        self._probes: list[ProbeSpec] = []

    def register(
        self,
        name: str,
        fn: ProbeFn,
        interval_tokens: int,
        needs_grads: bool = False,
    ) -> None:
        self._probes.append(
            ProbeSpec(
                name=name, fn=fn, interval_tokens=interval_tokens, needs_grads=needs_grads
            )
        )

    def maybe_fire(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        cumulative_tokens: int,
        ctx: dict[str, Any] | None = None,
        *,
        needs_grads: bool | None = None,
    ) -> dict[str, Any]:
        """Fire all probes whose interval has elapsed. If `needs_grads` is None
        (default), all probes are eligible. If True/False, restrict to that
        partition — used by train.py to fire grad-needing probes between
        backward and optimizer.step()."""
        ctx = ctx or {}
        out: dict[str, Any] = {}
        for spec in self._probes:
            if needs_grads is not None and spec.needs_grads != needs_grads:
                continue
            if cumulative_tokens - spec.last_fire_tokens >= spec.interval_tokens:
                out[spec.name] = spec.fn(model, optimizer, ctx)
                spec.last_fire_tokens = cumulative_tokens
        return out
