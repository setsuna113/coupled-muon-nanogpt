#!/usr/bin/env bash
# Wait for Rig B pt1B, generate adaptive pt3 JSONL, then launch pt3.
#
# Typical use:
#   nohup bash scripts/chain_rigB_pt3.sh > /path/to/pt3/chained.log 2>&1 & disown
#
# Override defaults with env vars if needed:
#   PT1_DIR=... PT3_DIR=... ENV_SH=../my_env.sh bash scripts/chain_rigB_pt3.sh
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT=${PROJECT:-$(cd "$SCRIPT_DIR/.." && pwd)}
GLOBAL=${GLOBAL:-/inspire/hdd/global_user/yanjunchi-24040/yancheng}
PT1_DIR=${PT1_DIR:-$GLOBAL/coupled-muon-stage3-rigB-pt1}
PT3_DIR=${PT3_DIR:-$GLOBAL/coupled-muon-stage3-rigB-pt3}
PT3_JSONL=${PT3_JSONL:-$PT3_DIR/pt3_adaptive.jsonl}
PT3_MANIFEST=${PT3_MANIFEST:-$PT3_DIR/pt3_adaptive.manifest}
EXPECTED_PT1=${EXPECTED_PT1:-60}
MAX_PT3_JOBS=${MAX_PT3_JOBS:-19}
SLEEP_SECONDS=${SLEEP_SECONDS:-300}
NPROC_PER_NODE=${NPROC_PER_NODE:-2}
NUM_WORKERS=${NUM_WORKERS:-2}
GPUS_PER_WORKER=${GPUS_PER_WORKER:-2}
SKIP_GIT_PULL=${SKIP_GIT_PULL:-1}
PT1_PROCESS_PATTERN=${PT1_PROCESS_PATTERN:-run_sweep.py.*(pt1B|coupled-muon-stage3-rigB-pt1)}

timestamp() {
  date '+%Y-%m-%d %H:%M:%S %z'
}

log() {
  printf '[%s] %s\n' "$(timestamp)" "$*"
}

source_env() {
  if [[ -n "${ENV_SH:-}" ]]; then
    # shellcheck disable=SC1090
    source "$ENV_SH"
    return
  fi
  if [[ -f ../my_env.sh ]]; then
    # shellcheck disable=SC1091
    source ../my_env.sh
    return
  fi
  if [[ -f my_env.sh ]]; then
    # shellcheck disable=SC1091
    source my_env.sh
    return
  fi
  log "No env file found. Set ENV_SH=/path/to/my_env.sh if this node needs one."
  return 1
}

count_done() {
  grep -l '\[saved_ckpt\]' "$PT1_DIR"/*/stdout.log 2>/dev/null | wc -l
}

mkdir -p "$PT3_DIR"
cd "$PROJECT"

log "Project: $PROJECT"
log "pt1 dir: $PT1_DIR"
log "pt3 dir: $PT3_DIR"
log "pt3 jsonl: $PT3_JSONL"
log "Waiting for pt1B dispatcher pattern: $PT1_PROCESS_PATTERN"

while pgrep -f "$PT1_PROCESS_PATTERN" >/dev/null; do
  done_count=$(count_done)
  log "pt1B still running: $done_count/$EXPECTED_PT1 completed; sleeping ${SLEEP_SECONDS}s"
  sleep "$SLEEP_SECONDS"
done

sleep 10
done_count=$(count_done)
if [[ "$done_count" != "$EXPECTED_PT1" ]]; then
  log "pt1B dispatcher ended with $done_count/$EXPECTED_PT1 completed; pt3 NOT launched"
  exit 1
fi
log "pt1B complete: $done_count/$EXPECTED_PT1"

source_env

if [[ "$SKIP_GIT_PULL" == "1" ]]; then
  log "Skipping git pull because SKIP_GIT_PULL=1/offline"
else
  log "Refreshing repo before pt3 launch"
  git pull
fi

log "Generating adaptive pt3 JSONL"
uv run python scripts/make_rigB_pt3_jsonl.py \
  --pt1-dir "$PT1_DIR" \
  --out "$PT3_JSONL" \
  --manifest "$PT3_MANIFEST" \
  --expected-completed "$EXPECTED_PT1" \
  --max-jobs "$MAX_PT3_JOBS"

if ! grep -q '"optimizer.lr"' "$PT3_JSONL"; then
  log "Generated JSONL has no optimizer.lr override; pt3 NOT launched"
  exit 1
fi

log "Launching pt3"
WANDB_MODE=offline uv run python scripts/run_sweep.py "$PT3_JSONL" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --num-workers "$NUM_WORKERS" \
  --gpus-per-worker "$GPUS_PER_WORKER" \
  --results-dir "$PT3_DIR"

log "pt3 launcher exited"
