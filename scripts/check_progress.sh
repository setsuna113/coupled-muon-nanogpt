#!/usr/bin/env bash
# Snapshot of "how many cells finished" across all running sweeps.
# A cell is "finished" iff its stdout.log contains [saved_ckpt].
#
# Usage:
#   scripts/check_progress.sh           one-line summary per rig
#   scripts/check_progress.sh -v        also show 2 most-recent cells per rig
#
# Edit the RIGS list to match your --results-dir choices. Each entry is
# "name|cell-glob|expected-total". The glob must expand to per-cell dirs.
set -u

GLOBAL=/inspire/hdd/global_user/yanjunchi-24040/yancheng
PROJECT=/inspire/hdd/project/quantum-artificial-intelligence/yanjunchi-24040/yancheng/coupled-muon-nanogpt/results

RIGS=(
  "stage1-A0|${CMNG_RESULTS_DIR:-$PROJECT}/ladder_A0_llama60m-*|75"
  "stage2-dense|$GLOBAL/coupled-muon-stage2-dense-archive/*|270"
  "stage3-moe|$GLOBAL/coupled-muon-stage3-moe/*|350"
  "stage3-rigB-pt1|$GLOBAL/coupled-muon-stage3-rigB-pt1/*|60"
  "stage3-rigA-anchor-screen|$GLOBAL/coupled-muon-stage3-rigA-anchor-screen/*|105"
  "stage3-rig2H200-adamwIp|$GLOBAL/coupled-muon-stage3-rig2H200-adamw-Iprime/*|15"
  "stage3-rigB-pt3|$GLOBAL/coupled-muon-stage3-rigB-pt3/*|19"
  "d4ablations|$GLOBAL/coupled-muon-d4ablations/*|105"
)

verbose=0
[[ "${1:-}" == "-v" || "${1:-}" == "--verbose" ]] && verbose=1

now=$(date +%s)

summarise() {
  local name=$1 glob=$2 total=$3
  if ! compgen -G "$glob" >/dev/null 2>&1; then
    printf "%-16s  (no cell dirs matching: %s)\n" "$name" "$glob"
    return
  fi
  local done_count latest mtime age status
  done_count=$(grep -l '\[saved_ckpt\]' $glob/stdout.log 2>/dev/null | wc -l)
  latest=$(ls -td $glob/metrics.jsonl 2>/dev/null | head -1)
  if [[ -n "$latest" ]]; then
    mtime=$(stat -c %Y "$latest")
    age=$((now - mtime))
    if   (( age < 300  )); then status=$(printf 'ALIVE  (last %3ds ago)' "$age")
    elif (( age < 3600 )); then status=$(printf 'quiet  (last %3dm ago)' "$((age/60))")
    else                        status=$(printf 'STALE  (last %3dh ago)' "$((age/3600))")
    fi
  else
    status="no metrics yet"
  fi
  printf "%-16s  %3d / %3d  %s\n" "$name" "$done_count" "$total" "$status"
}

detail() {
  local name=$1 glob=$2
  compgen -G "$glob" >/dev/null 2>&1 || return
  echo "--- $name : 2 most-recent cells ---"
  ls -td $glob/metrics.jsonl 2>/dev/null | head -2 | while read f; do
    local cell mtime last
    cell=$(basename "$(dirname "$f")")
    mtime=$(stat -c '%y' "$f" | cut -d. -f1)
    last=$(tail -1 "$f" 2>/dev/null | head -c 240)
    echo "  [$mtime] $cell"
    [[ -n "$last" ]] && echo "    $last"
  done
}

for rig in "${RIGS[@]}"; do
  IFS='|' read -r name glob total <<< "$rig"
  summarise "$name" "$glob" "$total"
done

if (( verbose )); then
  echo
  for rig in "${RIGS[@]}"; do
    IFS='|' read -r name glob total <<< "$rig"
    detail "$name" "$glob"
  done
fi
