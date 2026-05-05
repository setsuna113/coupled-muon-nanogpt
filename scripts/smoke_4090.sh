#!/usr/bin/env bash
# Smoke training on a single 4090 (or any single GPU).
# Usage: bash scripts/smoke_4090.sh configs/smoke/tiny_dev.yaml [--seed 0]
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <config.yaml> [extra args]" >&2
  exit 1
fi
CFG="$1"
shift

cd "$(dirname "$0")/.."

exec torchrun --standalone --nproc_per_node=1 \
  -m coupled_muon_nanogpt.train --config "$CFG" "$@"
