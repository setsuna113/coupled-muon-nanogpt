#!/usr/bin/env bash
# Inspire Studio environment setup for coupled-muon-nanogpt.
#
# Usage:
#     source my_env.sh
#
# All caches and large artifacts are redirected to persistent paths so they
# survive Notebook restarts and don't bloat the saved image. Per Inspire's
# directory rules, only the project / personal / global paths persist;
# everything else is wiped on stop.

# --- HuggingFace cache (FineWeb-Edu download lands here, ~10 GB) -------------
export HF_HOME=/inspire/hdd/global_user/yanjunchi-24040/huggingface
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME/hub

# --- uv / pip cache (wheel downloads — torch+cuda are several GB) -----------
# Uncommenting (per the cluster admin tip) so wheels don't re-download on
# every fresh container.
export UV_CACHE_DIR=/inspire/hdd/global_user/yanjunchi-24040/uv-cache
export PIP_CACHE_DIR=/inspire/hdd/global_user/yanjunchi-24040/pip-cache

# --- Wandb: offline on the H200 box, sync post-hoc from an internet machine -
# The wandb library reads `WANDB_MODE`, not a MUON_-prefixed alias — our
# `wandb_utils.init_wandb` honors `cfg.run.wandb_mode` first, then falls back
# to this env var.
export WANDB_MODE=offline
# Redirect wandb's local scratch out of $HOME so offline run dirs and the
# wandb config / cache don't accumulate on transient paths.
export WANDB_DIR=/inspire/hdd/global_user/yanjunchi-24040/wandb
export WANDB_CACHE_DIR=$WANDB_DIR/cache
export WANDB_CONFIG_DIR=$WANDB_DIR/config
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

# --- Training output dir (checkpoints + metrics.jsonl + offline wandb runs) -
# A 210-job sweep produces hundreds of GB of checkpoints; keep that on
# project storage, not the project root.  Two ways to use it:
#   (a) `--override run.output_dir=$CMNG_RESULTS_DIR` on every torchrun call
#   (b) one-time symlink:
#         ln -sfn "$CMNG_RESULTS_DIR" "$(dirname "${BASH_SOURCE[0]}")/results"
# After (b), every run writes through `./results -> CMNG_RESULTS_DIR` with
# zero per-command override.
export CMNG_RESULTS_DIR=/inspire/hdd/project/quantum-artificial-intelligence/yanjunchi-24040/coupled-muon-nanogpt/results
mkdir -p "$CMNG_RESULTS_DIR"

# --- Legacy compatibility: kept so other Muon-flavored scripts the user has
#     elsewhere still work.  Not read by this project.
export MUON_OUTPUT_DIR=/inspire/hdd/project/quantum-artificial-intelligence/yanjunchi-24040/yancheng/muon_outputs

# --- Local bin on PATH (uv installs there) ----------------------------------
export PATH="$HOME/.local/bin:$PATH"

# --- .venv recommendation (NOT exported; informational only) ----------------
# .venv on the project root will survive Notebook restarts (project paths
# persist) but bloats the saved image since torch+cuda wheels are several GB.
# Recommended one-time setup (run once after `git clone`):
#
#     uv venv /inspire/hdd/global_user/yanjunchi-24040/venvs/coupled-muon
#     ln -sfn /inspire/hdd/global_user/yanjunchi-24040/venvs/coupled-muon .venv
#     uv sync --all-extras
#
# After this, `uv run …` and `.venv/bin/python …` continue to work
# transparently while the actual files live on the persistent global dir.

echo "[my_env] HF_HOME=$HF_HOME"
echo "[my_env] WANDB_MODE=$WANDB_MODE  WANDB_DIR=$WANDB_DIR"
echo "[my_env] CMNG_RESULTS_DIR=$CMNG_RESULTS_DIR"
