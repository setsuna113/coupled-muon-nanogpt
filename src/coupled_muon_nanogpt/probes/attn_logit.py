"""Max pre-softmax attention logit probe (MuonClip stability signal).

Per experiment.md c.4 + d.3: the runaway-attention failure mode of vanilla Muon
shows up as max attention logit > ~1000. This probe is a wrapper around the
forward pass — it requires `return_max_logit=True` on the model call, which
the training loop sets when this probe is scheduled.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def attn_logit_probe(
    _model: nn.Module,
    _optimizer: torch.optim.Optimizer,
    ctx: dict[str, Any],
) -> dict[str, Any]:
    max_logits: torch.Tensor | None = ctx.get("max_attn_logits")
    if max_logits is None:
        return {"per_layer": [], "global_max": None}
    return {
        "per_layer": max_logits.cpu().tolist(),
        "global_max": float(max_logits.amax().item()),
    }
