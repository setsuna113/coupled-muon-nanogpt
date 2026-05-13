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
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

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

SEED_RE = re.compile(r"s(\d+)$")


def _path(project: str) -> str:
    return f"{WANDB_ENTITY}/{project}"


def _parse_seed(name: str, cfg_seed) -> int | None:
    if cfg_seed is not None:
        try:
            return int(cfg_seed)
        except (TypeError, ValueError):
            pass
    m = SEED_RE.search(name)
    return int(m.group(1)) if m else None


def _rung_short(rung_long: str) -> str | None:
    return LONG_TO_SHORT.get(rung_long, rung_long)


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
        rung_long = cfg.get("name", "")
        rows.append({
            "run_id": r.id,
            "name": r.name,
            "rung_long": rung_long,
            "rung": _rung_short(rung_long),
            "optimizer": opt_cfg.get("type"),
            "lr": float(opt_cfg.get("lr")) if opt_cfg.get("lr") is not None else None,
            "coupled_steps": opt_cfg.get("coupled_steps"),
            "final_polish": opt_cfg.get("final_polish"),
            "couple_qk": opt_cfg.get("couple_qk"),
            "couple_vo": opt_cfg.get("couple_vo"),
            "couple_updown": opt_cfg.get("couple_updown"),
            "ns_steps": opt_cfg.get("ns_steps"),
            "use_multi_head": opt_cfg.get("use_multi_head"),
            "seed": _parse_seed(r.name, cfg.get("seed")),
            "final_val_loss": r.summary.get("final_val_loss"),
            "diverged": bool(r.summary.get("diverged")) if r.summary.get("diverged") is not None else False,
            "final_step": r.summary.get("final_step"),
            "final_tokens": r.summary.get("final_tokens"),
            "runtime_s": r.summary.get("_runtime"),
            "project": project,
            "state": r.state,
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
    record["fetched_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
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
