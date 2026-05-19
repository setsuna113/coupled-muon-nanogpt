"""Part II reproducibility snapshot and parquet hash tables."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import DATA_DIR, PART2_PROJECTS, write_table


def _esc(s: object) -> str:
    text = str(s)
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("#", r"\#")
        .replace("_", r"\_")
    )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:12]


def _csv_rows(path: Path) -> int:
    if not path.exists():
        return 0
    return max(0, sum(1 for _ in path.open()) - 1)


def _part2_parquets() -> list[Path]:
    paths: set[Path] = set(DATA_DIR.glob("part2_*.parquet"))
    for project in PART2_PROJECTS:
        stem = project.replace("-", "_")
        paths.update(DATA_DIR.glob(f"{stem}*.parquet"))
    return sorted(paths, key=lambda p: p.name)


def _write_snapshot_table(manifest: dict) -> None:
    completion_csv = DATA_DIR / "part2_completion.csv"
    missing_csv = DATA_DIR / "part2_missing_cells.csv"
    duplicates_csv = DATA_DIR / "part2_discarded_duplicates.csv"
    completion_rows = _csv_rows(completion_csv)
    missing_rows = _csv_rows(missing_csv)
    duplicate_rows = _csv_rows(duplicates_csv)

    rows = [
        (
            "Snapshot UTC",
            manifest.get("snapshot_utc", "n/a"),
            "",
        ),
        (
            "Project completion counts",
            "docs/data/part2_completion.csv",
            f"{completion_rows} projects",
        ),
        (
            "Missing-cell CSV",
            manifest.get("missing_cells_csv", "docs/data/part2_missing_cells.csv"),
            f"{missing_rows} rows",
        ),
        (
            "Discarded-duplicate CSV",
            manifest.get("discarded_duplicates_csv", "docs/data/part2_discarded_duplicates.csv"),
            f"{duplicate_rows} rows",
        ),
        (
            "Dedupe rule",
            manifest.get("dedupe_rule", "n/a"),
            "",
        ),
    ]
    lines = [
        r"\begin{tabular}{p{0.24\linewidth}p{0.48\linewidth}p{0.12\linewidth}}",
        r"\toprule",
        r"Item & Value & Count \\",
        r"\midrule",
    ]
    for item, value, count in rows:
        lines.append(rf"{_esc(item)} & {_esc(value)} & {_esc(count)} \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    write_table("tab24_part2_repro_snapshot", "\n".join(lines))


def _write_hash_table() -> None:
    rows = []
    for path in _part2_parquets():
        try:
            n_rows = len(pd.read_parquet(path))
        except Exception:
            n_rows = "n/a"
        rows.append({
            "file": path.name,
            "rows": n_rows,
            "size_kb": f"{path.stat().st_size / 1024:.1f}",
            "sha": _sha256(path),
        })

    lines = [
        r"\begin{tabular}{lrrl}",
        r"\toprule",
        r"Parquet snapshot & Rows & Size (KB) & SHA-256 (12) \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            rf"\code{{{_esc(row['file'])}}} & {row['rows']} & {row['size_kb']} & \code{{{row['sha']}}} \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    write_table("tab25_part2_repro_hashes", "\n".join(lines))


def main() -> None:
    manifest_path = DATA_DIR / "part2_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    _write_snapshot_table(manifest)
    _write_hash_table()


if __name__ == "__main__":
    main()
