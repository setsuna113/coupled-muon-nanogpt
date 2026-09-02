"""One-shot tokenizer for FineWeb-Edu → uint16 .bin shards.

Produces files compatible with the memory-mapped loader. Idempotent: skips
shards that already exist on disk.

Usage:
    python -m coupled_muon_nanogpt.data.prepare_fineweb \
        --out-dir data/fineweb_edu_2048 \
        --total-tokens 5_000_000_000 \
        --val-tokens 50_000_000 \
        --shard-size 100_000_000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="data/fineweb_edu_2048")
    parser.add_argument("--dataset-name", type=str, default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--dataset-config", type=str, default="sample-10BT")
    parser.add_argument("--total-tokens", type=int, default=5_000_000_000)
    parser.add_argument("--val-tokens", type=int, default=50_000_000)
    parser.add_argument("--shard-size", type=int, default=100_000_000)
    parser.add_argument("--num-proc", type=int, default=8)
    args = parser.parse_args()

    try:
        import tiktoken
        from datasets import load_dataset
    except ImportError as e:
        print(f"Missing dependency: {e}", file=sys.stderr)
        print("Install with: uv add tiktoken datasets", file=sys.stderr)
        sys.exit(1)

    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token  # 50256 for gpt2

    out = Path(args.out_dir)
    train_dir = out / "train"
    val_dir = out / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    print(f"Streaming dataset {args.dataset_name} ({args.dataset_config})...")
    ds = load_dataset(args.dataset_name, name=args.dataset_config, split="train", streaming=True)

    def tokenize(doc: dict) -> np.ndarray:
        toks = [eot] + enc.encode_ordinary(doc["text"])
        return np.asarray(toks, dtype=np.uint32)

    def write_shard(target_dir: Path, shard_idx: int, buf: np.ndarray):
        path = target_dir / f"shard_{shard_idx:06d}.bin"
        if path.exists():
            print(f"  shard exists, skipping: {path}")
            return
        # Truncate any token > 2**16-1 (shouldn't happen for gpt2, vocab=50257).
        if buf.max() >= 2**16:
            raise ValueError("Token id exceeds uint16 range; widen dtype")
        buf.astype(np.uint16).tofile(path)
        print(f"  wrote {path} ({buf.size} tokens)")

    total_target = args.total_tokens
    val_target = args.val_tokens
    shard_size = args.shard_size

    val_buf: list[np.ndarray] = []
    val_count = 0
    train_buf: list[np.ndarray] = []
    train_count = 0
    train_shard_idx = 0
    train_total = 0

    pbar = tqdm(total=total_target + val_target, unit="tok")
    for doc in ds:
        toks = tokenize(doc)
        if val_count < val_target:
            val_buf.append(toks)
            val_count += toks.size
            pbar.update(toks.size)
            continue

        train_buf.append(toks)
        train_count += toks.size
        train_total += toks.size
        pbar.update(toks.size)

        while train_count >= shard_size:
            cat = np.concatenate(train_buf)
            chunk = cat[:shard_size]
            remainder = cat[shard_size:]
            write_shard(train_dir, train_shard_idx, chunk)
            train_shard_idx += 1
            train_buf = [remainder] if remainder.size else []
            train_count = remainder.size

        if train_total >= total_target:
            break

    if train_buf:
        cat = np.concatenate(train_buf)
        write_shard(train_dir, train_shard_idx, cat)
    if val_buf:
        cat = np.concatenate(val_buf)
        write_shard(val_dir, 0, cat)
    pbar.close()
    print(f"Done. Train shards: {train_shard_idx + 1}, Val tokens: {val_count}")


if __name__ == "__main__":
    main()
