"""Run every fig*.py and tab*.py under docs/figures/.

Each script is a self-contained module that reads docs/data/*.parquet and
writes docs/figures/*.pdf or docs/tables/*.tex. We import each module and call
its `main()` so import errors surface here rather than as silent build gaps.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def _run(script: Path) -> bool:
    name = script.stem
    spec = importlib.util.spec_from_file_location(name, script)
    if spec is None or spec.loader is None:
        print(f"  could not load {script.name}")
        return False
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        if hasattr(mod, "main"):
            mod.main()
        return True
    except Exception as e:
        print(f"  ERROR in {script.name}: {type(e).__name__}: {e}")
        return False


def main() -> int:
    scripts = sorted(p for p in HERE.glob("fig*.py")) + sorted(p for p in HERE.glob("tab*.py"))
    ok = fail = 0
    for s in scripts:
        print(f"[run] {s.name}")
        if _run(s):
            ok += 1
        else:
            fail += 1
    print(f"\nSummary: {ok} ok, {fail} failed")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
