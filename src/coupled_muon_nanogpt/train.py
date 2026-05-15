"""Config-driven DDP training loop.

Single entry point. Usage:

    torchrun --nproc_per_node=$N -m coupled_muon_nanogpt.train --config <path>

CLI accepts:
    --config <path>       (required)  resolved against configs/base.yaml
    --seed <int>          (default 0)
    --override key=value  (repeatable) escape hatch, e.g. --override optimizer.lr=3e-3
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

from . import wandb_utils
from .data.loader import ShardDataLoader
from .eval import evaluate
from .model.moe import MoEFFN
from .model.transformer import GPT, BlockConfig, GPTConfig
from .optim.factory import build_optimizer
from .probes.attn_logit import attn_logit_probe
from .probes.coupled_pair import coupled_pair_probe
from .probes.manager import ProbeManager
from .probes.moe_load import moe_load_probe
from .probes.ns_internal import ns_internal_probe
from .probes.pair_factor_ratio import pair_factor_ratio_probe
from .probes.svd import svd_probe
from .utils import cosine_lr, run_id, save_ckpt, seed_all, setup_ddp


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--override", action="append", default=[], help="dot.path=value")
    return p.parse_args(argv)


def load_config(path: str, overrides: list[str]) -> Any:
    from omegaconf import OmegaConf

    base_path = Path(__file__).resolve().parent.parent.parent / "configs" / "base.yaml"
    cfg = OmegaConf.load(base_path)
    cfg = OmegaConf.merge(cfg, OmegaConf.load(path))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    OmegaConf.resolve(cfg)
    return cfg


def build_model(cfg: Any) -> GPT:
    block = BlockConfig(
        hidden=int(cfg.model.hidden),
        n_heads=int(cfg.model.attn.n_heads),
        n_kv_heads=cfg.model.attn.get("n_kv_heads") or None,
        head_dim=cfg.model.attn.get("head_dim") or None,
        intermediate=int(cfg.model.mlp.get("intermediate", 0)),
        norm_type=str(cfg.model.norm.type),
        norm_eps=float(cfg.model.norm.eps),
        mlp_type=str(cfg.model.mlp.type),
        pos_emb_type=str(cfg.model.pos_emb.type),
        rope_base=float(cfg.model.pos_emb.get("rope_base", 10000.0)),
        rope_partial_frac=float(cfg.model.attn.get("rope_partial_frac", 1.0)),
        qk_norm=bool(cfg.model.attn.qk_norm),
        max_seq_len=int(cfg.train.seq_len),
        # Phase-2 MLA + factff knobs (defaults preserve MHA / non-factored FFN).
        attn_type=str(cfg.model.attn.get("attn_type", "mha")),
        kv_lora_rank=int(cfg.model.attn.get("kv_lora_rank", 0)),
        q_lora_rank=int(cfg.model.attn.get("q_lora_rank", 0)),
        qk_nope_head_dim=int(cfg.model.attn.get("qk_nope_head_dim", 0)),
        qk_rope_head_dim=int(cfg.model.attn.get("qk_rope_head_dim", 0)),
        v_head_dim=int(cfg.model.attn.get("v_head_dim", 0)),
        mlp_factorize_rank=int(cfg.model.mlp.get("factorize_rank", 0)),
        moe_enabled=bool(cfg.model.moe.enabled),
        moe_cfg=dict(
            num_experts=int(cfg.model.moe.get("num_experts", 8)),
            top_k=int(cfg.model.moe.get("top_k", 2)),
            capacity_factor=float(cfg.model.moe.get("capacity_factor", 1.25)),
            aux_loss_coef=float(cfg.model.moe.get("aux_loss_coef", 0.01)),
            z_loss_coef=float(cfg.model.moe.get("z_loss_coef", 1e-3)),
            balancing_type=str(cfg.model.moe.get("balancing_type", "aux_loss")),
            bias_update_lr=float(cfg.model.moe.get("bias_update_lr", 1.0e-3)),
            shared_expert=bool(cfg.model.moe.get("shared_expert", False)),
            shared_expert_intermediate=int(cfg.model.moe.get("shared_expert_intermediate", 0)),
            # 0 ⇒ inherit from mlp.intermediate; >0 overrides for MoE layers only.
            intermediate=int(cfg.model.moe.get("intermediate", 0)),
        )
        if cfg.model.moe.enabled
        else {},
    )
    gpt_cfg = GPTConfig(
        vocab_size=int(cfg.model.vocab_size),
        n_layers=int(cfg.model.n_layers),
        block=block,
        tie_embeddings=bool(cfg.model.get("tie_embeddings", True)),
        init_std=float(cfg.model.get("init_std", 0.02)),
        moe_every_other=bool(cfg.model.moe.get("every_other", True)),
    )
    return GPT(gpt_cfg)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv or sys.argv[1:])
    cfg = load_config(args.config, args.override)

    rank, world_size, local_rank = setup_ddp()
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    seed_all(args.seed + rank)
    rid = run_id(cfg, args.seed)
    out_dir = Path(cfg.run.output_dir) / rid
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        # Persist resolved config alongside results.
        from omegaconf import OmegaConf

        with (out_dir / "config.yaml").open("w") as f:
            f.write(OmegaConf.to_yaml(cfg))
        print(f"[run_id] {rid}")
        print(f"[out_dir] {out_dir}")

    # Build model on the right device, with bf16 weights left in fp32 master.
    model = build_model(cfg).to(device)
    if rank == 0:
        n_params = model.num_parameters(exclude_embedding=False)
        n_active = model.num_parameters(exclude_embedding=True)
        print(f"[model] params={n_params/1e6:.2f}M  (excl. embed: {n_active/1e6:.2f}M)")

    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
        unwrapped = model.module
    else:
        unwrapped = model

    optimizer = build_optimizer(unwrapped, cfg)

    # Data
    tr_loader = ShardDataLoader(
        shard_dir=Path(cfg.data.shard_dir) / "train",
        seq_len=int(cfg.train.seq_len),
        local_batch_size=int(cfg.train.local_batch_size),
        rank=rank,
        world_size=world_size,
        seed=args.seed,
        infinite=True,
    )
    val_loader = ShardDataLoader(
        shard_dir=Path(cfg.data.shard_dir) / "val",
        seq_len=int(cfg.train.seq_len),
        local_batch_size=int(cfg.train.local_batch_size),
        rank=rank,
        world_size=world_size,
        seed=args.seed + 12345,
        infinite=True,
    )

    # Probes
    probes = ProbeManager()
    probe_cfg = cfg.train.get("probes", {})
    if probe_cfg.get("svd_interval_tokens", 0):
        probes.register("svd", svd_probe, int(probe_cfg.svd_interval_tokens))
    if probe_cfg.get("coupled_pair_interval_tokens", 0):
        probes.register("coupled_pair", coupled_pair_probe, int(probe_cfg.coupled_pair_interval_tokens))
    if probe_cfg.get("attn_logit_interval_tokens", 0):
        probes.register("attn_logit", attn_logit_probe, int(probe_cfg.attn_logit_interval_tokens))
    # MoE load + per-expert grad-norm probe needs to fire BEFORE optimizer.zero_grad().
    if probe_cfg.get("moe_load_interval_tokens", 0):
        probes.register(
            "moe_load",
            moe_load_probe,
            int(probe_cfg.moe_load_interval_tokens),
            needs_grads=True,
        )
    if probe_cfg.get("ns_internal_interval_tokens", 0):
        probes.register(
            "ns_internal",
            ns_internal_probe,
            int(probe_cfg.ns_internal_interval_tokens),
            needs_grads=True,
        )
    if probe_cfg.get("pair_factor_ratio_interval_tokens", 0):
        probes.register(
            "pair_factor_ratio",
            pair_factor_ratio_probe,
            int(probe_cfg.pair_factor_ratio_interval_tokens),
        )

    # Wandb (rank 0 only). All policy lives in wandb_utils; this module just
    # decides whether to call it.
    wandb_handle: wandb_utils.WandbHandle | None = None
    if rank == 0 and bool(cfg.run.get("wandb", False)):
        wandb_handle = wandb_utils.init_wandb(cfg, rid, args.seed, out_dir)

    metrics_path = out_dir / "metrics.jsonl"
    metrics_f = metrics_path.open("a") if rank == 0 else None

    # Training loop
    total_tokens_target = int(cfg.train.total_tokens)
    grad_accum = int(cfg.train.grad_accum_steps)
    global_batch_tokens = int(cfg.train.local_batch_size) * int(cfg.train.seq_len) * world_size * grad_accum
    total_steps = total_tokens_target // global_batch_tokens
    warmup_steps = int(cfg.train.warmup_steps)
    base_lr = float(cfg.optimizer.lr)
    eval_every_steps = int(cfg.train.get("eval_every_steps", max(100, total_steps // 20)))
    log_every_steps = int(cfg.train.get("log_every_steps", 10))
    val_tokens = int(cfg.train.get("val_tokens", 1_000_000))
    grad_clip = float(cfg.train.get("grad_clip", 1.0))
    bf16 = bool(cfg.train.get("bf16", True))

    if rank == 0:
        print(
            f"[schedule] total_steps={total_steps}  warmup={warmup_steps}  "
            f"global_batch_tokens={global_batch_tokens}  total_tokens={total_tokens_target}"
        )

    model.train()
    cumulative_tokens = 0
    t0 = time.time()
    # Track divergence flag (ns_internal.diverged or attn_logit > 1000) for the
    # final wandb summary.
    diverged_seen = False
    last_val_loss: float | None = None
    last_train_loss: float | None = None
    for step in range(total_steps):
        # Set LR
        lr_t = cosine_lr(step, total_steps, warmup_steps, base_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_t

        ctx_for_probes: dict[str, Any] = {
            "step": step,
            "n_heads": int(cfg.model.attn.n_heads),
            "n_kv_heads": int(cfg.model.attn.get("n_kv_heads") or cfg.model.attn.n_heads),
            # Forward the live optimizer-coupling flags so pair-aware probes
            # (e.g. pair_factor_ratio) report only pairs the optimizer
            # actually couples, not pairs `classify_parameters` would
            # default-on. Plain-Muon and AdamW paths set the relevant flags
            # to False in their builders, so we mirror those here.
            "couple_qk": bool(cfg.optimizer.get("couple_qk", True)) and cfg.optimizer.type == "coupled_muon_v2",
            "couple_vo": bool(cfg.optimizer.get("couple_vo", True)) and cfg.optimizer.type == "coupled_muon_v2",
            "couple_updown": bool(cfg.optimizer.get("couple_updown", True)) and cfg.optimizer.type == "coupled_muon_v2",
            "couple_mla": bool(cfg.optimizer.get("couple_mla", True)) and cfg.optimizer.type == "coupled_muon_v2",
            "couple_factff": bool(cfg.optimizer.get("couple_factff", True)) and cfg.optimizer.type == "coupled_muon_v2",
            "couple_router_to_muon": bool(cfg.optimizer.get("couple_router_to_muon", True)),
        }

        loss_accum = 0.0
        max_logits_collect: torch.Tensor | None = None

        # Whether to ask the model to return per-layer max logits this step.
        want_max_logit = (
            probe_cfg.get("attn_logit_interval_tokens", 0)
            and (cumulative_tokens % int(probe_cfg.attn_logit_interval_tokens)) < global_batch_tokens
        )

        for micro in range(grad_accum):
            x, y = tr_loader.next_batch()
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bf16 and device.type == "cuda"):
                out = unwrapped(x, targets=y, return_max_logit=bool(want_max_logit))
                loss = out["total_loss"] / grad_accum
            loss.backward()
            loss_accum += loss.item() * grad_accum
            if "max_attn_logits" in out:
                max_logits_collect = out["max_attn_logits"].detach()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(unwrapped.parameters(), grad_clip)

        # Pre-step probes (those that consume `.grad` directly: per-expert grad
        # norms, etc.). Fire on `cumulative_tokens + global_batch_tokens` so
        # tokens align with the current step's grads. Must run on every rank
        # because probes like `moe_load_probe` issue `dist.all_reduce`; firing
        # only on rank 0 leaves rank 1 unmatched and watchdog kills the job.
        pre_step_probe_tokens = cumulative_tokens + global_batch_tokens
        pre_step_probe_out = probes.maybe_fire(
            unwrapped, optimizer, pre_step_probe_tokens, ctx_for_probes, needs_grads=True
        )

        # Optimizer step (timed, per d.6.9).
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_opt0 = time.time()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        opt_step_seconds = time.time() - t_opt0
        optimizer.zero_grad(set_to_none=True)

        # DeepSeek-V2 §3.2 aux-loss-free balancing: bias-update happens AFTER
        # optimizer.step() and is a no-op for non-deepseek_bias modes.
        for module in unwrapped.modules():
            if isinstance(module, MoEFFN):
                module.update_router_bias()

        cumulative_tokens += global_batch_tokens

        if rank == 0 and (step % log_every_steps == 0 or step == total_steps - 1):
            elapsed = time.time() - t0
            tps = cumulative_tokens / max(1e-6, elapsed)
            row = {
                "step": step,
                "tokens": cumulative_tokens,
                "wall_clock_s": elapsed,
                "lr": lr_t,
                "loss": loss_accum / grad_accum,
                "opt_step_s": opt_step_seconds,
                "tok_per_s": tps,
            }
            print(json.dumps(row))
            metrics_f.write(json.dumps({"event": "step", **row}) + "\n")
            metrics_f.flush()
            wandb_utils.log_step(wandb_handle, row)
            last_train_loss = float(row["loss"])

        # Probes (post-step: those that don't need `.grad`). Fire on every rank
        # for the same reason as the pre-step probe; only rank 0 logs the result.
        ctx_for_probes["max_attn_logits"] = max_logits_collect
        probe_out = probes.maybe_fire(
            unwrapped, optimizer, cumulative_tokens, ctx_for_probes, needs_grads=False
        )
        if rank == 0:
            # Merge any pre-step (grad-needing) probe output collected before optimizer.step.
            if pre_step_probe_out:
                probe_out = {**probe_out, **pre_step_probe_out}
            if probe_out:
                metrics_f.write(json.dumps({"event": "probe", "step": step, "tokens": cumulative_tokens, **probe_out}) + "\n")
                metrics_f.flush()
                if wandb_utils.probe_indicates_divergence(probe_out):
                    diverged_seen = True
                wandb_utils.log_probe(
                    wandb_handle, probe_out, tokens=cumulative_tokens,
                    wall_clock_s=time.time() - t0,
                )

        # Eval
        if (step + 1) % eval_every_steps == 0 or step == total_steps - 1:
            val_metrics = evaluate(unwrapped, val_loader, val_tokens, device)
            if world_size > 1 and dist.is_initialized():
                t = torch.tensor([val_metrics["val_loss"]], device=device)
                dist.all_reduce(t, op=dist.ReduceOp.AVG)
                val_metrics["val_loss"] = float(t.item())
            if rank == 0:
                row = {
                    "step": step,
                    "tokens": cumulative_tokens,
                    "wall_clock_s": time.time() - t0,
                    **val_metrics,
                }
                print(json.dumps({"event": "val", **row}))
                metrics_f.write(json.dumps({"event": "val", **row}) + "\n")
                metrics_f.flush()
                wandb_utils.log_eval(wandb_handle, row)
                last_val_loss = float(val_metrics.get("val_loss", float("nan")))

    # Final checkpoint
    if rank == 0:
        ckpt_path = save_ckpt(out_dir / "checkpoints", total_steps, unwrapped, optimizer, cfg, args.seed)
        print(f"[saved_ckpt] {ckpt_path}")
        metrics_f.close()
        wandb_utils.finalize(
            wandb_handle,
            summary={
                "final_val_loss": last_val_loss if last_val_loss is not None else float("nan"),
                "final_train_loss": last_train_loss if last_train_loss is not None else float("nan"),
                "final_step": total_steps,
                "final_tokens": cumulative_tokens,
                "wall_clock_s": time.time() - t0,
                "params_total": int(unwrapped.num_parameters(exclude_embedding=False)),
                "params_active": int(unwrapped.num_parameters(exclude_embedding=True)),
                "global_batch_tokens": global_batch_tokens,
                "total_steps": total_steps,
                "world_size": world_size,
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                "diverged": diverged_seen,
            },
        )
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
