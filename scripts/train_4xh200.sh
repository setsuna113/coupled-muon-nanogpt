#!/usr/bin/env bash
# Production training on 4×H200.
# Usage: bash scripts/train_4xh200.sh configs/ladder/A0_llama60m.yaml [--seed 1]
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <config.yaml> [extra args]" >&2
  exit 1
fi
CFG="$1"
shift

cd "$(dirname "$0")/.."

# OMP_NUM_THREADS keeps NCCL stable; NCCL_DEBUG can be flipped on for issues.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

exec torchrun --standalone --nproc_per_node=4 \
  -m coupled_muon_nanogpt.train --config "$CFG" "$@"
