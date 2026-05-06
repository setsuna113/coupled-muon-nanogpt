"""Memory-mapped uint16 shard loader.

Deterministic per (seed, dp_rank): the data sequence depends on (seed, rank,
world_size) only, so different optimizer runs at the same seed see the *same*
token stream — required by the d.3 protocol's compute-matched comparisons.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class ShardDataLoader:
    def __init__(
        self,
        shard_dir: str | Path,
        seq_len: int,
        local_batch_size: int,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
        infinite: bool = True,
    ):
        self.shard_dir = Path(shard_dir)
        self.seq_len = seq_len
        self.local_batch_size = local_batch_size
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.infinite = infinite

        self.shards = sorted(self.shard_dir.glob("shard_*.bin"))
        if not self.shards:
            raise FileNotFoundError(f"No shards found under {self.shard_dir}")
        self._mmap_current: np.memmap | None = None
        self._shard_cursor = 0
        self._token_cursor = 0
        self._reset_to_first_shard()

    def _open_shard(self, idx: int) -> np.memmap:
        path = self.shards[idx]
        # uint16 mmap; sized automatically.
        return np.memmap(path, dtype=np.uint16, mode="r")

    def _reset_to_first_shard(self):
        self._shard_cursor = 0
        self._mmap_current = self._open_shard(0)
        # Stagger starting offset by rank to keep ranks decorrelated.
        stride = self.local_batch_size * self.seq_len * self.world_size
        rank_offset = self.rank * self.local_batch_size * self.seq_len
        self._token_cursor = rank_offset % max(1, self._mmap_current.size - stride)

    def _advance_shard(self):
        self._shard_cursor += 1
        if self._shard_cursor >= len(self.shards):
            if self.infinite:
                self._shard_cursor = 0
            else:
                raise StopIteration
        self._mmap_current = self._open_shard(self._shard_cursor)
        self._token_cursor = self.rank * self.local_batch_size * self.seq_len

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Need (local_batch_size * (seq_len + 1)) tokens for input + targets.
        need = self.local_batch_size * (self.seq_len + 1)
        if self._mmap_current is None:
            self._reset_to_first_shard()
        assert self._mmap_current is not None
        end = self._token_cursor + need
        if end > self._mmap_current.size:
            self._advance_shard()
            end = self._token_cursor + need
            assert self._mmap_current is not None
            assert end <= self._mmap_current.size, "shard too small for one batch"

        buf = np.asarray(self._mmap_current[self._token_cursor : end], dtype=np.int64)
        # Stride to next rank's window.
        self._token_cursor += self.local_batch_size * self.seq_len * self.world_size
        toks = torch.from_numpy(buf)
        toks = toks.view(self.local_batch_size, self.seq_len + 1)
        x = toks[:, :-1].contiguous()
        y = toks[:, 1:].contiguous()
        return x, y

    def __iter__(self):
        return self

    def __next__(self):
        return self.next_batch()
