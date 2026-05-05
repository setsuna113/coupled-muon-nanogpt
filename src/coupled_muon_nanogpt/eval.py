"""Validation pass: averaged loss over a fixed token budget."""
from __future__ import annotations

import torch
import torch.nn as nn

from .data.loader import ShardDataLoader


@torch.no_grad()
def evaluate(model: nn.Module, val_loader: ShardDataLoader, num_tokens: int, device: torch.device) -> dict[str, float]:
    was_training = model.training
    model.eval()
    seq_len = val_loader.seq_len
    bs = val_loader.local_batch_size
    tokens_per_batch = bs * seq_len
    n_batches = max(1, num_tokens // tokens_per_batch)
    total_loss = 0.0
    total_tok = 0
    for _ in range(n_batches):
        x, y = val_loader.next_batch()
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        out = model(x, targets=y)
        total_loss += out["loss"].item() * tokens_per_batch
        total_tok += tokens_per_batch
    if was_training:
        model.train()
    avg = total_loss / max(1, total_tok)
    return {"val_loss": avg, "val_tokens": total_tok}
