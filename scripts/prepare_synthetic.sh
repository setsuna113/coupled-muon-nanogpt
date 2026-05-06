#!/usr/bin/env bash
# Generate random-token shards for smoke / tiny_dev configs (no FineWeb needed).
# Override with env vars: OUT_DIR, TRAIN_TOKENS, VAL_TOKENS, VOCAB_SIZE.
set -euo pipefail
cd "$(dirname "$0")/.."

OUT_DIR=${OUT_DIR:-data/synthetic_smoke}
TRAIN_TOKENS=${TRAIN_TOKENS:-10000000}
VAL_TOKENS=${VAL_TOKENS:-500000}
VOCAB_SIZE=${VOCAB_SIZE:-50304}

exec python -m coupled_muon_nanogpt.data.prepare_synthetic \
  --out-dir "$OUT_DIR" \
  --train-tokens "$TRAIN_TOKENS" \
  --val-tokens "$VAL_TOKENS" \
  --vocab-size "$VOCAB_SIZE"
