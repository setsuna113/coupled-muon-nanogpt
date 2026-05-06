#!/usr/bin/env bash
# Production training on 2×H200.
# Usage: bash scripts/train_2xh200.sh configs/ladder/A0_llama60m.yaml [--seed 1]
#
# Ladder yamls' grad_accum_steps is sized for this 2-GPU rig — global batch
# stays at ~0.5M tokens/step (the d.3 protocol target). For a different GPU
# count, override grad_accum_steps via --override (e.g. `--override
# train.grad_accum_steps=4` on 4 GPUs to halve it back to 0.5M).
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

exec torchrun --standalone --nproc_per_node=2 \
  -m coupled_muon_nanogpt.train --config "$CFG" "$@"
