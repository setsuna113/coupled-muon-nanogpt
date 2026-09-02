#!/usr/bin/env bash
# Fail-closed launcher for subagent-model-guard.py (rule: sub-agents run on
# opus or sonnet only, never fable; see CLAUDE.md).
#
# Exit 0 = allow, exit 2 = deny. Anything else (no python, a crash) is turned
# into a deny so the guard can never silently fail open. Builtins only, so it
# behaves the same even with a broken PATH.
set -u
case "${BASH_SOURCE[0]}" in
  */*) here="${BASH_SOURCE[0]%/*}" ;;
  *) here="." ;;
esac
py="$(command -v python3 || command -v python || true)"
if [ -z "$py" ]; then
  echo "[subagent-model-guard] python3 not found; blocking sub-agent launch (sub-agents must run on opus or sonnet, never fable)" >&2
  exit 2
fi
"$py" "$here/subagent-model-guard.py"
rc=$?
case "$rc" in
  0|2) exit "$rc" ;;
  *)
    echo "[subagent-model-guard] hook crashed (exit $rc); blocking sub-agent launch (sub-agents must run on opus or sonnet, never fable)" >&2
    exit 2
    ;;
esac
