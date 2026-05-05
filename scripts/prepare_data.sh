#!/usr/bin/env bash
# One-shot tokenizer for FineWeb-Edu → uint16 .bin shards.
# Override with env vars: TOTAL_TOKENS, VAL_TOKENS, SHARD_SIZE, OUT_DIR.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT_DIR=${OUT_DIR:-data/fineweb_edu_2048}
TOTAL_TOKENS=${TOTAL_TOKENS:-5000000000}
VAL_TOKENS=${VAL_TOKENS:-50000000}
SHARD_SIZE=${SHARD_SIZE:-100000000}

exec python -m coupled_muon_nanogpt.data.prepare_fineweb \
  --out-dir "$OUT_DIR" \
  --total-tokens "$TOTAL_TOKENS" \
  --val-tokens "$VAL_TOKENS" \
  --shard-size "$SHARD_SIZE"
