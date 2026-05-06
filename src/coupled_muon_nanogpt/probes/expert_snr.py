"""Per-expert gradient SNR probe (manual, post-training CLI).

Per experiment.md d.4 row 9 / c.1: if per-expert mini-batch gradients have low
cosine alignment with their average (cos < 0.5), Newton-Schulz is iterating on
noise rather than signal. This probe runs K micro-batches through a checkpointed
model, accumulates per-expert gradients per layer, and reports:

  * `cos_to_mean`: average cosine of mini-batch g_i^(k) to mean_k g_i^(k);
                   noise floor is 1/sqrt(rank) ≈ 0.1 for typical d×d matrices.
  * `snr`: ‖mean_g‖_F / std_k(‖g_i^(k) − mean_g‖_F).

Heavyweight (K backward passes), so it is a manual one-shot CLI rather than an
auto-fired training-loop probe.

Usage:
    python -m coupled_muon_nanogpt.probes.expert_snr \\
        --config configs/ladder/I_moe.yaml \\
        --ckpt results/<run_id>/checkpoints/step_<n>_*.pt \\
        --num_micro 32 \\
        --out snr_report.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ..data.loader import ShardDataLoader
from ..model.moe import MoEFFN


def _flat_grad(module: nn.Module) -> torch.Tensor:
    parts = [p.grad.detach().float().reshape(-1) for p in module.parameters() if p.grad is not None]
    return torch.cat(parts) if parts else torch.zeros(0)


@torch.enable_grad()
def estimate_per_expert_snr(
    model: nn.Module,
    loader: ShardDataLoader,
    *,
    num_micro: int = 32,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Run `num_micro` independent forward+backward passes; for each MoE layer
    and expert, compute the cosine of each mini-batch gradient to the mean and
    the SNR of the gradient norm distribution."""
    device = device or next(model.parameters()).device
    model.eval()  # disable dropout (unaffected by autograd)

    moe_layers = [m for m in model.modules() if isinstance(m, MoEFFN)]
    if not moe_layers:
        return {"_note": "no MoEFFN modules found in model — dense run"}

    # Storage: per-layer per-expert list of flat grad tensors.
    histories: list[list[list[torch.Tensor]]] = [
        [[] for _ in range(len(layer.experts))] for layer in moe_layers
    ]

    for k in range(num_micro):
        x, y = loader.next_batch()
        x = x.to(device)
        y = y.to(device)
        for p in model.parameters():
            p.grad = None
        out = model(x, targets=y)
        loss = out["total_loss"]
        loss.backward()

        for li, layer in enumerate(moe_layers):
            for ei, expert in enumerate(layer.experts):
                g = _flat_grad(expert).cpu()
                histories[li][ei].append(g)

    out: dict[str, Any] = {}
    for li, per_expert_grads in enumerate(histories):
        layer_out = {}
        for ei, grads in enumerate(per_expert_grads):
            if not grads:
                continue
            stacked = torch.stack(grads, dim=0)  # (K, P)
            mean_g = stacked.mean(dim=0)  # (P,)
            mean_norm = mean_g.norm().item()
            # Cosine of each mini-batch gradient to the mean.
            cos = torch.nn.functional.cosine_similarity(
                stacked, mean_g.unsqueeze(0).expand_as(stacked), dim=-1
            )
            # SNR: ‖mean‖ / std of ‖per-mini-batch − mean‖.
            residuals = stacked - mean_g.unsqueeze(0)
            res_norms = residuals.norm(dim=-1)
            snr = mean_norm / (res_norms.std().item() + 1e-9)
            layer_out[f"expert.{ei}"] = {
                "cos_to_mean_avg": float(cos.mean().item()),
                "cos_to_mean_min": float(cos.min().item()),
                "snr": float(snr),
                "mean_grad_norm": float(mean_norm),
            }
        out[f"layer.{li}"] = layer_out
    return out


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--num_micro", type=int, default=32)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from ..train import build_model, load_config
    from ..utils import load_model_only

    cfg = load_config(args.config, overrides=[])
    model = build_model(cfg).to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    # We never call optimizer.step() — only forward+backward over micro-batches
    # — so use the model-only loader and skip optimizer state entirely.
    load_model_only(Path(args.ckpt), model)

    loader = ShardDataLoader(
        shard_dir=Path(cfg.data.shard_dir) / "val",
        seq_len=int(cfg.train.seq_len),
        local_batch_size=int(cfg.train.local_batch_size),
        rank=0,
        world_size=1,
        seed=args.seed,
        infinite=True,
    )
    report = estimate_per_expert_snr(model, loader, num_micro=args.num_micro)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    _main()
