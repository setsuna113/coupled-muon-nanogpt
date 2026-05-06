"""Wandb integration policy for the coupled-muon-nanogpt training harness.

All wandb logic lives here so train.py stays focused and the policy is
unit-testable. Rank-0 gating is the caller's responsibility (train.py); this
module assumes it is invoked only on rank 0 with `cfg.run.wandb` truthy.

The four wandb dimensions:

    project   = cfg.run.wandb_project                         (study container)
    group     = cfg.run.wandb_group  or  fallback formula     (auto-aggregates seeds)
    job_type  = cfg.optimizer.type                            (UI optimizer family)
    tags      = derived from cfg + extra cfg.run.wandb_tags   (filterable axes)

X-axis policy: tokens for cross-run-comparable metrics (loss, val_loss, slow
probes), default step for within-run-only metrics (lr, opt_step_s, tok_per_s,
ns_internal diagnostics). Critical: `wandb.log` is called WITHOUT `step=` kwarg
for any payload that includes `tokens` — passing both silently overrides
`step_metric="tokens"` and reverts to the default step axis.

Offline workflow: training nodes run with `WANDB_MODE=offline`; runs are
synced post-hoc with `wandb sync results/<rid>/wandb/`. We honor
`cfg.run.wandb_mode` (null ⇒ env var ⇒ "online") and never mutate os.environ.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Per-key cap on per-index expansion of list-valued probe outputs. Keeps the
# wandb metric namespace bounded even at 32 layers / 32 experts.
_MAX_INDEX_KEYS = 32


@dataclass
class WandbHandle:
    """Opaque handle returned by init_wandb. Holds the live `wandb` module and
    the run object so callers don't have to import wandb themselves."""

    wandb: Any
    run: Any
    # Mutable summary block accumulated during training; flushed by finalize.
    summary: dict[str, Any] = field(default_factory=dict)


def init_wandb(cfg: Any, rid: str, seed: int, out_dir: Path) -> WandbHandle | None:
    """Initialize a wandb run for the given config, or return None on failure.

    Caller (train.py) must already have checked rank == 0 and cfg.run.wandb.
    Returns None if the wandb package is missing — training continues silently.
    """
    try:
        import wandb
    except ImportError:
        return None

    project = str(cfg.run.get("wandb_project", "coupled-muon-nanogpt"))
    group = _resolve_group(cfg)
    job_type = str(cfg.optimizer.type)
    name = _resolve_name(cfg, seed)
    tags = _resolve_tags(cfg, seed)
    mode = _resolve_mode(cfg)

    from omegaconf import OmegaConf

    init_kwargs: dict[str, Any] = dict(
        project=project,
        group=group,
        job_type=job_type,
        name=name,
        tags=tags,
        config=OmegaConf.to_container(cfg, resolve=True),
        dir=str(out_dir),
        # `console="off"` keeps wandb from hijacking stdout; preserves the
        # `[run_id] …` line that scripts/run_sweep.py parses to find result dirs.
        # `start_method="thread"` avoids the fork+CUDA hang under torchrun.
        settings=wandb.Settings(start_method="thread", console="off"),
    )
    if mode is not None:
        init_kwargs["mode"] = mode

    run = wandb.init(**init_kwargs)
    handle = WandbHandle(wandb=wandb, run=run)
    define_metrics(handle)
    handle.summary["rid"] = rid
    return handle


def define_metrics(handle: WandbHandle) -> None:
    """Register tokens (and wall_clock_s) as x-axes for cross-run-comparable
    metrics. Within-run-only metrics keep wandb's default step axis."""
    w = handle.wandb
    w.define_metric("tokens")
    w.define_metric("wall_clock_s")
    # Cross-run comparable — plot vs tokens.
    w.define_metric("loss", step_metric="tokens")
    w.define_metric("val_loss", step_metric="tokens")
    w.define_metric("probe/svd/*", step_metric="tokens")
    w.define_metric("probe/coupled_pair/*", step_metric="tokens")
    w.define_metric("probe/attn_logit/*", step_metric="tokens")
    w.define_metric("probe/moe_load/*", step_metric="tokens")
    # Within-run-only — default step axis is fine.
    w.define_metric("lr")
    w.define_metric("opt_step_s")
    w.define_metric("tok_per_s")
    w.define_metric("probe/ns_internal/*")


def log_step(handle: WandbHandle | None, row: dict[str, Any]) -> None:
    """Log a per-step train row. Row must include `tokens` so the tokens-axis
    metrics resolve correctly. Do NOT pass `step=` to wandb.log."""
    if handle is None:
        return
    handle.wandb.log(row)


def log_eval(handle: WandbHandle | None, row: dict[str, Any]) -> None:
    """Log an eval row. Row should include `tokens` and `val_loss`."""
    if handle is None:
        return
    handle.wandb.log(row)


def log_probe(
    handle: WandbHandle | None,
    probe_out: dict[str, Any],
    tokens: int,
    wall_clock_s: float | None = None,
) -> None:
    """Log probe outputs, namespaced under `probe/{probe_name}/...`. Lists are
    expanded to per-index scalars (capped at _MAX_INDEX_KEYS) plus a
    wandb.Histogram for distribution-shaped data (per-expert loads / grad
    norms). The flat dict includes `tokens` so all probe metrics share the
    tokens x-axis."""
    if handle is None or not probe_out:
        return
    flat = flatten_probe(handle.wandb, probe_out)
    if not flat:
        return
    flat["tokens"] = tokens
    if wall_clock_s is not None:
        flat["wall_clock_s"] = wall_clock_s
    handle.wandb.log(flat)


def finalize(handle: WandbHandle | None, summary: dict[str, Any]) -> None:
    """Flush the accumulated summary dict to wandb.run.summary and call finish."""
    if handle is None:
        return
    merged = {**handle.summary, **summary}
    for k, v in merged.items():
        handle.run.summary[k] = v
    handle.wandb.finish()


# ---------------------------------------------------------------------------
# Identity helpers
# ---------------------------------------------------------------------------

def _resolve_group(cfg: Any) -> str:
    """Sweep YAMLs may set `cfg.run.wandb_group` (with OmegaConf ${...}
    interpolation) to override the fallback. Fallback varies on rung,
    optimizer, and lr — sufficient for the LR-grid sweeps but not for
    pair_ablation / coupled_steps_ablation, which must override."""
    explicit = cfg.run.get("wandb_group")
    if explicit:
        return str(explicit)
    return f"{cfg.name}/{cfg.optimizer.type}/lr{float(cfg.optimizer.lr):.0e}"


def _resolve_name(cfg: Any, seed: int) -> str:
    return (
        f"{cfg.name}·{cfg.optimizer.type}"
        f"·lr{float(cfg.optimizer.lr):.0e}·s{seed}"
    )


def _resolve_tags(cfg: Any, seed: int) -> list[str]:
    tags: list[str] = [
        str(cfg.name),
        str(cfg.optimizer.type),
        f"seed{seed}",
    ]
    moe = cfg.model.get("moe", {})
    if bool(moe.get("enabled", False)):
        tags.append("moe")
        tags.append(f"experts{int(moe.get('num_experts', 0))}")
        tags.append(f"topk{int(moe.get('top_k', 0))}")
        balancing = str(moe.get("balancing_type", "aux_loss"))
        if balancing != "aux_loss":
            tags.append(f"balance:{balancing}")
        if bool(moe.get("shared_expert", False)):
            tags.append("shared_expert")
    if str(cfg.optimizer.type) == "coupled_muon_v2":
        tags.append(f"cs{int(cfg.optimizer.get('coupled_steps', 0))}")
        if bool(cfg.optimizer.get("couple_qk", False)):
            tags.append("couple_qk")
        if bool(cfg.optimizer.get("couple_vo", False)):
            tags.append("couple_vo")
        if bool(cfg.optimizer.get("couple_updown", False)):
            tags.append("couple_updown")
        if bool(cfg.optimizer.get("final_polish", True)):
            tags.append("final_polish")
        if bool(cfg.optimizer.get("use_multi_head", False)):
            tags.append("multi_head")
    if str(cfg.name).startswith("smoke_"):
        tags.append("smoke")
    extra = cfg.run.get("wandb_tags") or []
    for t in extra:
        if t and str(t) not in tags:
            tags.append(str(t))
    return tags


def _resolve_mode(cfg: Any) -> str | None:
    explicit = cfg.run.get("wandb_mode")
    if explicit:
        return str(explicit)
    env = os.environ.get("WANDB_MODE")
    if env:
        return env
    return None  # let wandb default to online


# ---------------------------------------------------------------------------
# Probe flattener
# ---------------------------------------------------------------------------

def flatten_probe(wandb: Any, probe_out: dict[str, Any]) -> dict[str, Any]:
    """Convert nested probe output into wandb-loggable scalars / histograms.

    Each top-level key is a probe name (svd, coupled_pair, attn_logit,
    moe_load, ns_internal); the inner structure varies. The flattener:
      - prefixes everything with `probe/{probe_name}/`
      - expands lists of scalars into per-index keys (capped)
      - emits a wandb.Histogram for known distribution-shaped lists
    """
    flat: dict[str, Any] = {}
    for probe_name, probe_data in probe_out.items():
        if probe_name.startswith("_"):
            continue
        prefix = f"probe/{probe_name}/"
        if isinstance(probe_data, dict):
            _flatten_recursive(wandb, probe_data, prefix, flat, probe_name=probe_name)
        elif isinstance(probe_data, (int, float)):
            flat[prefix.rstrip("/")] = float(probe_data)
    return flat


def _flatten_recursive(
    wandb: Any,
    obj: dict[str, Any],
    prefix: str,
    out: dict[str, Any],
    probe_name: str,
) -> None:
    for k, v in obj.items():
        if isinstance(k, str) and k.startswith("_"):
            continue
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            _flatten_recursive(wandb, v, key + "/", out, probe_name=probe_name)
        elif isinstance(v, bool):
            # Treat booleans as 0/1 scalars (wandb panels render fine).
            out[key] = float(v)
        elif isinstance(v, (int, float)):
            out[key] = float(v)
        elif isinstance(v, list) and v and isinstance(v[0], (int, float, bool)):
            _emit_list(wandb, v, key, out, probe_name=probe_name)
        # Strings, None, and tensors are silently skipped.


def _emit_list(
    wandb: Any,
    values: list,
    key: str,
    out: dict[str, Any],
    probe_name: str,
) -> None:
    """Per-index scalars for line-plotting + histogram for distribution-shaped
    data. moe_load.loads / grad_norms are genuine per-expert distributions —
    add a histogram. attn_logit.per_layer / svd.top_sv / ns_internal.norms_*
    are vectors (small N, ordered) — per-index scalars only."""
    floats = [float(x) for x in values]
    n = min(len(floats), _MAX_INDEX_KEYS)
    # Per-index scalars (line-plottable).
    if key.endswith("/per_layer"):
        # attn_logit.per_layer → probe/attn_logit/L0, L1, …
        base = key[: -len("/per_layer")]
        for i in range(n):
            out[f"{base}/L{i}"] = floats[i]
    elif key.endswith("/top_sv"):
        # svd top singular values → /sv0, /sv1, …
        base = key[: -len("/top_sv")]
        for i in range(n):
            out[f"{base}/sv{i}"] = floats[i]
    elif key.endswith("/norms_A") or key.endswith("/norms_B"):
        # ns_internal NS-iterate norms → /i0, /i1, …
        for i in range(n):
            out[f"{key}/i{i}"] = floats[i]
    elif key.endswith("/loads") or key.endswith("/grad_norms"):
        # MoE per-expert distribution → per-expert scalars + histogram
        for i in range(n):
            out[f"{key}/E{i}"] = floats[i]
        try:
            out[f"{key}_hist"] = wandb.Histogram(floats)
        except Exception:  # noqa: BLE001
            # If Histogram construction fails (e.g. mocked wandb in tests),
            # silently skip — per-index scalars still carry the data.
            pass
    else:
        # Unknown list shape: fall back to per-index scalars.
        for i in range(n):
            out[f"{key}/i{i}"] = floats[i]


# ---------------------------------------------------------------------------
# Divergence detection (called inline from train.py)
# ---------------------------------------------------------------------------

ATTN_LOGIT_DIVERGE_THRESHOLD = 1000.0  # MuonClip threshold per experiment.md d.6.4


def probe_indicates_divergence(probe_out: dict[str, Any]) -> bool:
    """Scan a probe-out dict for divergence signals: ns_internal.diverged
    flags or attn_logit.global_max above the MuonClip threshold."""
    if not probe_out:
        return False
    ns = probe_out.get("ns_internal", {})
    if isinstance(ns, dict):
        for _pair_key, pair_data in ns.items():
            if isinstance(pair_data, dict) and bool(pair_data.get("diverged", False)):
                return True
    al = probe_out.get("attn_logit", {})
    if isinstance(al, dict):
        gm = al.get("global_max")
        if isinstance(gm, (int, float)) and float(gm) > ATTN_LOGIT_DIVERGE_THRESHOLD:
            return True
    return False
