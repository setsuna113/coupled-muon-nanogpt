#!/usr/bin/env bash
# Move sweep cells that have completed ([saved_ckpt] in stdout.log) from
# the project-quota results dir to the global_user path, replacing each
# moved cell with a symlink at the original location so analysis tools
# and filter_unfinished_jobs.py keep finding it.
#
# Safe-by-construction: only cells whose stdout.log contains
# `[saved_ckpt]` AND have no open file descriptors get moved. Cells
# still being written (active wandb flush, etc.) are skipped.
#
# Usage:
#   scripts/archive_finished.sh                 dry-run (default)
#   scripts/archive_finished.sh --execute       actually move
#
# Override SRC/DST via env. Defaults are wired for stage 2 on the
# advanced-machine-learning project's 8xH100 box. Examples:
#   SRC=/.../coupled-muon-nanogpt/results \
#   DST=/inspire/hdd/global_user/.../coupled-muon-stage2-dense-archive \
#       scripts/archive_finished.sh --execute
set -u

: "${SRC:=/inspire/hdd/project/advanced-machine-learning/yanjunchi-24040/yancheng/coupled-muon-nanogpt/results}"
: "${DST:=/inspire/hdd/global_user/yanjunchi-24040/yancheng/coupled-muon-stage2-dense-archive}"

execute=0
[[ "${1:-}" == "--execute" || "${1:-}" == "-x" ]] && execute=1

if [[ ! -d "$SRC" ]]; then
    echo "SRC does not exist: $SRC" >&2
    exit 1
fi
mkdir -p "$DST"

echo "SRC=$SRC"
echo "DST=$DST"
echo "mode=$([[ $execute -eq 1 ]] && echo execute || echo dry-run)"
echo

moved=0
unfinished=0
open_fds=0
already=0
failed=0

for d in "$SRC"/*/; do
    # `for d in $SRC/*/` iterates only directories *and* dir-symlinks.
    # Strip trailing slash so test on the path itself works.
    p="${d%/}"
    cell=$(basename "$p")

    if [[ -L "$p" ]]; then
        already=$((already + 1))
        continue
    fi

    if ! grep -q '\[saved_ckpt\]' "$p/stdout.log" 2>/dev/null; then
        unfinished=$((unfinished + 1))
        continue
    fi

    if command -v lsof >/dev/null 2>&1; then
        if lsof +D "$p" 2>/dev/null | tail -n +2 | grep -q .; then
            echo "[skip-open] $cell"
            open_fds=$((open_fds + 1))
            continue
        fi
    fi

    if (( execute )); then
        echo "[move] $cell"
        if [[ -e "$DST/$cell" ]]; then
            echo "  destination already exists, skipping: $DST/$cell" >&2
            failed=$((failed + 1))
            continue
        fi
        if mv "$p" "$DST/$cell"; then
            ln -s "$DST/$cell" "$p"
            moved=$((moved + 1))
        else
            echo "  mv failed for $cell" >&2
            failed=$((failed + 1))
        fi
    else
        echo "[would-move] $cell"
        moved=$((moved + 1))
    fi
done

echo
echo "Summary:"
printf "  moved%s:         %d\n" "$([[ $execute -eq 0 ]] && echo ' (dry-run)')" "$moved"
printf "  unfinished:      %d\n" "$unfinished"
printf "  open file descr: %d\n" "$open_fds"
printf "  already-moved:   %d\n" "$already"
[[ $failed -gt 0 ]] && printf "  failed:          %d\n" "$failed"
[[ $execute -eq 0 ]] && echo "(re-run with --execute to actually move)"
