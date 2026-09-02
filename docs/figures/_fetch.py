"""wandb.Api() wrapper with on-disk parquet caching.

`summary_df(project)` pulls `r.summary` + `r.config` + `r.name` for every run
in one API call (`per_page=500`). `history_df(project, run_ids, keys, samples)`
pulls downsampled per-step history for the named runs only — slow, so we only
ever call it on best-LR runs.

Both write Parquet snapshots under docs/data/; pass `refresh=True` to bypass
the cache. The CLI lives in `fetch_all.py`.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import wandb
from _common import (
    DATA_DIR,
    LONG_TO_SHORT,
    PROJECT_ABLATION,
    PROJECT_STAGE1,
    PROJECT_STAGE2,
    WANDB_ENTITY,
)

SEED_RE = re.compile(r"(?:^|[^\w])s(\d+)(?:$|[^\d])")


def _path(project: str) -> str:
    return f"{WANDB_ENTITY}/{project}"


def _parse_seed(name: str, cfg_seed) -> int | None:
    if cfg_seed is not None:
        try:
            return int(cfg_seed)
        except (TypeError, ValueError):
            pass
    m = SEED_RE.search(str(name))
    return int(m.group(1)) if m else None


def _rung_short(rung_long: str) -> str | None:
    return LONG_TO_SHORT.get(rung_long, rung_long)


def _float_or_none(x) -> float | None:
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _mean_summary(summary, prefix: str, suffix: str) -> float | None:
    vals = []
    for key, value in summary.items():
        if key.startswith(prefix) and key.endswith(suffix):
            v = _float_or_none(value)
            if v is not None:
                vals.append(v)
    return float(np.mean(vals)) if vals else None


def _max_summary(summary, prefix: str, suffix: str) -> float | None:
    vals = []
    for key, value in summary.items():
        if key.startswith(prefix) and key.endswith(suffix):
            v = _float_or_none(value)
            if v is not None:
                vals.append(v)
    return float(np.max(vals)) if vals else None


def summary_df(project: str, *, refresh: bool = False) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cache = DATA_DIR / f"{project.replace('-', '_')}_summary.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)

    api = wandb.Api()
    runs = list(api.runs(_path(project), per_page=500))
    rows = []
    for r in runs:
        cfg = r.config
        opt_cfg = cfg.get("optimizer") or {}
        model_cfg = cfg.get("model") or {}
        attn_cfg = model_cfg.get("attn") or {}
        mlp_cfg = model_cfg.get("mlp") or {}
        moe_cfg = model_cfg.get("moe") or {}
        rung_long = cfg.get("name", "")
        rows.append({
            "run_id": r.id,
            "name": r.name,
            "group": getattr(r, "group", None),
            "job_type": getattr(r, "job_type", None),
            "tags": ",".join(getattr(r, "tags", []) or []),
            "rung_long": rung_long,
            "rung": _rung_short(rung_long),
            "optimizer": opt_cfg.get("type"),
            "lr": _float_or_none(opt_cfg.get("lr")),
            "coupled_steps": opt_cfg.get("coupled_steps"),
            "final_polish": opt_cfg.get("final_polish"),
            "couple_qk": opt_cfg.get("couple_qk"),
            "couple_vo": opt_cfg.get("couple_vo"),
            "couple_updown": opt_cfg.get("couple_updown"),
            "couple_router_to_muon": opt_cfg.get("couple_router_to_muon"),
            "couple_mla": opt_cfg.get("couple_mla"),
            "ns_steps": opt_cfg.get("ns_steps"),
            "ns_coefficients": opt_cfg.get("ns_coefficients"),
            "ns_gram_form": opt_cfg.get("ns_gram_form"),
            "lr_prefactor": opt_cfg.get("lr_prefactor"),
            "use_multi_head": opt_cfg.get("use_multi_head"),
            "attn_type": attn_cfg.get("attn_type", "mha"),
            "kv_lora_rank": attn_cfg.get("kv_lora_rank"),
            "q_lora_rank": attn_cfg.get("q_lora_rank"),
            "mlp_type": mlp_cfg.get("type"),
            "moe_enabled": moe_cfg.get("enabled"),
            "moe_num_experts": moe_cfg.get("num_experts"),
            "moe_top_k": moe_cfg.get("top_k"),
            "moe_balancing_type": moe_cfg.get("balancing_type"),
            "moe_expert_intermediate": moe_cfg.get("intermediate"),
            "has_mla": attn_cfg.get("attn_type") == "mla",
            "has_factff": mlp_cfg.get("type") == "factff",
            "seed": _parse_seed(r.name, cfg.get("seed")),
            "final_val_loss": _float_or_none(r.summary.get("final_val_loss")),
            "final_train_loss": _float_or_none(r.summary.get("final_train_loss")),
            "diverged": bool(r.summary.get("diverged")) if r.summary.get("diverged") is not None else False,
            "final_step": r.summary.get("final_step"),
            "final_tokens": r.summary.get("final_tokens"),
            "runtime_s": _float_or_none(r.summary.get("_runtime")),
            "probe_router_entropy_mean": _mean_summary(r.summary, "probe/moe_load/", "/router_entropy"),
            "probe_load_imbalance_mean": _mean_summary(r.summary, "probe/moe_load/", "/imbalance"),
            "probe_load_imbalance_max": _max_summary(r.summary, "probe/moe_load/", "/imbalance"),
            "probe_grad_norm_var_mean": _mean_summary(r.summary, "probe/moe_load/", "/grad_norm_var"),
            "probe_attn_logit_global_max": _float_or_none(r.summary.get("probe/attn_logit/global_max")),
            "project": project,
            "state": r.state,
            "created_at": getattr(r, "created_at", None),
            "updated_at": getattr(r, "updated_at", None),
            "url": getattr(r, "url", None),
        })
    df = pd.DataFrame(rows)
    df.to_parquet(cache, index=False)
    print(f"  cached {cache.relative_to(DATA_DIR.parent.parent)} ({len(df)} runs)")
    return df


def _history_cache_key(project: str, run_ids: Iterable[str], keys: Iterable[str], samples: int) -> str:
    h = hashlib.sha256()
    h.update(project.encode())
    h.update(b"|")
    h.update(",".join(sorted(run_ids)).encode())
    h.update(b"|")
    h.update(",".join(sorted(keys)).encode())
    h.update(b"|")
    h.update(str(samples).encode())
    return h.hexdigest()[:12]


def history_df(
    project: str,
    run_ids: list[str],
    keys: list[str],
    *,
    samples: int = 500,
    refresh: bool = False,
    cache_tag: str | None = None,
) -> pd.DataFrame:
    """Pull per-step history per run_id, one event-type at a time.

    wandb's `r.history(keys=...)` returns the intersection of rows containing
    *all* keys — useless when val_loss and training loss live in different
    event rows. We instead call `scan_history(keys=['tokens', K])` once per K
    to keep each key's rows separately, then concat with one row per
    (run_id, tokens, key, value).

    Returns a long-form dataframe with columns
    `[run_id, name, tokens, metric, value]`. Callers can pivot to wide form
    if needed.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tag = cache_tag or _history_cache_key(project, run_ids, keys, samples)
    cache = DATA_DIR / f"{project.replace('-', '_')}_history_{tag}.parquet"
    if cache.exists() and not refresh:
        return pd.read_parquet(cache)

    api = wandb.Api()
    frames: list[pd.DataFrame] = []
    for run_id in run_ids:
        try:
            r = api.run(f"{_path(project)}/{run_id}")
        except Exception as e:
            print(f"  history: skip {run_id} ({e})")
            continue
        for metric in keys:
            rows = []
            try:
                for row in r.scan_history(keys=["tokens", metric], page_size=max(samples, 500)):
                    if metric in row and "tokens" in row:
                        rows.append({"tokens": row["tokens"], "value": row[metric]})
            except Exception as e:
                print(f"  history: error on {run_id}/{metric}: {e}")
                continue
            if not rows:
                continue
            df_metric = pd.DataFrame(rows)
            # Downsample to ~samples points (uniform stride).
            if len(df_metric) > samples:
                idx = np.linspace(0, len(df_metric) - 1, samples).round().astype(int)
                df_metric = df_metric.iloc[idx].reset_index(drop=True)
            df_metric["metric"] = metric
            df_metric["run_id"] = run_id
            df_metric["name"] = r.name
            frames.append(df_metric)
    if not frames:
        df = pd.DataFrame(columns=["run_id", "name", "tokens", "metric", "value"])
    else:
        df = pd.concat(frames, ignore_index=True)
    df.to_parquet(cache, index=False)
    n_runs = df["run_id"].nunique() if len(df) else 0
    print(f"  cached {cache.relative_to(DATA_DIR.parent.parent)} ({len(df)} rows across {n_runs} runs, {len(keys)} metrics)")
    return df


def manifest_update(record: dict) -> None:
    """Append a manifest entry to docs/data/cache_manifest.json."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / "cache_manifest.json"
    if path.exists():
        data = json.loads(path.read_text())
    else:
        data = {"entries": []}
    record["fetched_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    data["entries"].append(record)
    data["latest_fetched_at"] = record["fetched_at"]
    path.write_text(json.dumps(data, indent=2))


__all__ = [
    "summary_df",
    "history_df",
    "manifest_update",
    "PROJECT_STAGE1",
    "PROJECT_STAGE2",
    "PROJECT_ABLATION",
]
