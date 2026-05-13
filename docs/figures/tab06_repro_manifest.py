"""Tab 6 — Reproducibility manifest: project IDs, fetched_at, n_runs, SHA-256."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import DATA_DIR, write_table


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()[:12]


def main() -> None:
    manifest_path = DATA_DIR / "cache_manifest.json"
    fetched_at = "n/a"
    by_project: dict[str, dict] = {}
    if manifest_path.exists():
        m = json.loads(manifest_path.read_text())
        fetched_at = m.get("latest_fetched_at", "n/a")
        for entry in m.get("entries", []):
            by_project[entry["project"]] = entry

    rows = []
    for p in sorted(DATA_DIR.glob("*.parquet")):
        proj = p.name.split("_history_")[0].replace("_", "-") if "_history_" in p.name else p.stem.replace("_summary", "").replace("_", "-")
        kind = "history" if "_history_" in p.name else "summary"
        n_runs = by_project.get(proj, {}).get("n_runs", "—") if kind == "summary" else "—"
        rows.append({
            "file": p.name,
            "kind": kind,
            "n_runs": n_runs,
            "sha": _sha256(p),
            "size_kb": f"{p.stat().st_size / 1024:.1f}",
        })

    cols = ["Parquet file", "Kind", "$n_\\mathrm{runs}$", "SHA-256 (12)", "Size (KB)"]
    lines = [
        r"\begin{tabular}{lllll}",
        r"\toprule",
        " & ".join(cols) + r" \\",
        r"\midrule",
    ]
    for r in rows:
        f = r["file"].replace("_", r"\_")
        lines.append(f"\\texttt{{{f}}} & {r['kind']} & {r['n_runs']} & \\texttt{{{r['sha']}}} & {r['size_kb']} \\\\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        "",
        r"\smallskip",
        f"Latest fetch: \\texttt{{{fetched_at}}}",
    ]
    write_table("tab06_repro_manifest", "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
