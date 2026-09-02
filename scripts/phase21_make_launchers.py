#!/usr/bin/env python3
"""Print Phase-2.1 one-line launch commands for the three physical nodes.

The encoded payload writes the current Phase-2.1 orchestrator/materializer into
``$GLOBAL/phase2.1-control/bin`` on the remote node, then executes it from the
node's current working directory. Run the printed commands from the remote repo
root; no local Codex path is embedded.
"""
from __future__ import annotations

import argparse
import base64
import shlex
from pathlib import Path

GLOBAL_DEFAULT = "/inspire/hdd/global_user/yanjunchi-24040/yancheng"
ROLES = ("h100", "h200x2", "h200x4")


def _script_text(name: str) -> str:
    path = Path(__file__).resolve().with_name(name)
    return path.read_text()


def build_payload(global_root: str) -> str:
    materialize = _script_text("phase21_materialize.py")
    orchestrate = _script_text("phase21_orchestrate.py")
    return f"""#!/usr/bin/env bash
set -euo pipefail
ROLE="${{1:?usage: bash -s -- <h100|h200x2|h200x4> [orchestrator args...]}}"
shift
GLOBAL_ROOT={shlex.quote(global_root)}
args=("$@")
i=0
while (( i < ${{#args[@]}} )); do
  case "${{args[$i]}}" in
    --global-root)
      i=$((i + 1))
      GLOBAL_ROOT="${{args[$i]}}"
      ;;
    --global-root=*)
      GLOBAL_ROOT="${{args[$i]#--global-root=}}"
      ;;
  esac
  i=$((i + 1))
done
CONTROL="$GLOBAL_ROOT/phase2.1-control"
BIN="$CONTROL/bin"
mkdir -p "$BIN"
cat > "$BIN/phase21_materialize.py" <<'__P21_MATERIALIZE_PY__'
{materialize}
__P21_MATERIALIZE_PY__
cat > "$BIN/phase21_orchestrate.py" <<'__P21_ORCHESTRATE_PY__'
{orchestrate}
__P21_ORCHESTRATE_PY__
chmod +x "$BIN/phase21_materialize.py" "$BIN/phase21_orchestrate.py"
PROJECT="${{PROJECT:-$PWD}}"
cd "$PROJECT"
exec uv run python "$BIN/phase21_orchestrate.py" --role "$ROLE" "$@"
"""


def build_command(role: str, *, global_root: str, env_sh: str | None = None) -> str:
    payload = build_payload(global_root)
    encoded = base64.b64encode(payload.encode()).decode()
    args = [role, "--global-root", global_root, "--skip-git-pull"]
    if env_sh:
        args.extend(["--env-sh", env_sh])
    quoted_args = " ".join(shlex.quote(arg) for arg in args)
    out = f"phase21_{role}.out"
    return (
        f"printf %s '{encoded}' | tr -d '[:space:]' | base64 -d | "
        f"WANDB_MODE=offline nohup bash -s -- {quoted_args} > {out} 2>&1 & "
        f"disown; sleep 5 && tail -f {out}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--global-root", default=GLOBAL_DEFAULT)
    parser.add_argument("--env-sh", default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    lines: list[str] = []
    for role in ROLES:
        lines.append(f"# {role}")
        lines.append(build_command(role, global_root=args.global_root, env_sh=args.env_sh))
        lines.append("")
    text = "\n".join(lines)
    if args.out:
        args.out.write_text(text)
        print(f"Wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
