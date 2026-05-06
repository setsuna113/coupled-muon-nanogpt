"""Generate uint16 random-token shards for smoke / tiny_dev configs.

Smoke configs (configs/smoke/tiny_dev.yaml) point at `data/synthetic_smoke/`
and exist to verify the training pipeline end-to-end without paying the
~30-min FineWeb tokenization cost. This script writes random uint16 shards
in the layout the ShardDataLoader expects:

    data/synthetic_smoke/
    ├── train/shard_000000.bin
    └── val/shard_000000.bin

Tokens are uniform over [0, vocab_size) and the resulting "loss" is whatever
the model lands on — the goal is pipeline coverage, not training quality.

Usage:
    python -m coupled_muon_nanogpt.data.prepare_synthetic \\
        --out-dir data/synthetic_smoke \\
        --train-tokens 10_000_000 --val-tokens 500_000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=str, default="data/synthetic_smoke")
    p.add_argument("--vocab-size", type=int, default=50304)
    p.add_argument("--train-tokens", type=int, default=10_000_000)
    p.add_argument("--val-tokens", type=int, default=500_000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out = Path(args.out_dir)
    train_dir = out / "train"
    val_dir = out / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    train_path = train_dir / "shard_000000.bin"
    val_path = val_dir / "shard_000000.bin"

    if train_path.exists() and val_path.exists():
        print(f"shards already exist at {out}; nothing to do")
        return

    rng.integers(0, args.vocab_size, size=args.train_tokens, dtype=np.uint16).tofile(train_path)
    rng.integers(0, args.vocab_size, size=args.val_tokens, dtype=np.uint16).tofile(val_path)
    print(f"wrote {train_path} ({args.train_tokens} tokens)")
    print(f"wrote {val_path} ({args.val_tokens} tokens)")


if __name__ == "__main__":
    main()
