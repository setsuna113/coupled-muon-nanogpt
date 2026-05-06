"""Seeding, run-id hashing, optimizer-stamped checkpoint I/O."""
from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_id(cfg: Any, seed: int) -> str:
    """Deterministic run-id hash from the resolved config + seed."""
    payload = json.dumps(_to_jsonable(cfg), sort_keys=True) + f"|seed={seed}"
    h = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    name = getattr(cfg, "name", "run")
    return f"{name}-s{seed}-{h}"


def _to_jsonable(x: Any) -> Any:
    """Best-effort coercion from OmegaConf / dataclass to a plain dict."""
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(x):
            return OmegaConf.to_container(x, resolve=True)
    except ImportError:
        pass
    if hasattr(x, "__dict__"):
        return {k: _to_jsonable(v) for k, v in vars(x).items()}
    if isinstance(x, dict):
        return {k: _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    return x


def optimizer_meta(cfg: Any) -> dict[str, Any]:
    return {
        "type": str(cfg.optimizer.type),
        "lr": float(cfg.optimizer.lr),
        "wd": float(cfg.optimizer.wd),
        "ns_steps": int(cfg.optimizer.get("ns_steps", 5)),
        "coupled_steps": int(cfg.optimizer.get("coupled_steps", 0)),
        "couple_qk": bool(cfg.optimizer.get("couple_qk", True)),
        "couple_vo": bool(cfg.optimizer.get("couple_vo", True)),
        "couple_updown": bool(cfg.optimizer.get("couple_updown", True)),
        "use_multi_head": bool(cfg.optimizer.get("use_multi_head", False)),
        "ns_dtype": str(cfg.optimizer.get("ns_dtype", "bf16")),
    }


def save_ckpt(
    out_dir: Path,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: Any,
    seed: int,
    extra: dict[str, Any] | None = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = optimizer_meta(cfg)
    stamp = f"opt={meta['type']}_ns={meta['ns_steps']}_cs={meta['coupled_steps']}"
    path = out_dir / f"step_{step:08d}_{stamp}.pt"
    payload = {
        "step": step,
        "model": _strip_compile_prefix(model.state_dict()),
        "optimizer": optimizer.state_dict(),
        "optimizer_meta": meta,
        "seed": seed,
        "run_id": run_id(cfg, seed),
        "extra": extra or {},
    }
    torch.save(payload, path)
    return path


def load_model_only(path: Path, model: torch.nn.Module) -> dict[str, Any]:
    """Load model weights without touching the optimizer.

    Used by analysis tools (e.g. expert-SNR CLI) that need to re-run forward+
    backward on a checkpointed model but never call optimizer.step()."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"], strict=True)
    return payload


def _strip_compile_prefix(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {(k[len("_orig_mod.") :] if k.startswith("_orig_mod.") else k): v for k, v in sd.items()}


def setup_ddp() -> tuple[int, int, int]:
    """Returns (rank, world_size, local_rank)."""
    if "RANK" not in os.environ:
        return 0, 1, 0
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def cosine_lr(step: int, total_steps: int, warmup_steps: int, base_lr: float, min_lr_ratio: float = 0.1) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    import math

    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    cos = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cos)
