# Stage 1 + Stage 2 report, plus Part II sparse-MoE report (LaTeX)

Hand-built scientific report covering the Coupled Muon v2 dense-architecture
ladder results. Audience: pre-training specialists; no Muon recap.

## Three-command rebuild

```
uv sync --extra docs
uv run python docs/figures/fetch_all.py --refresh    # only if wandb sync state changed
uv run python docs/figures/make_all.py               # regenerates every fig + tab
latexmk -pdf -outdir=docs/build docs/report.tex
```

The first command installs `matplotlib`, `pyarrow`, `pandas`, `scipy`; the
second pulls run summaries and downsampled per-step history from the three
wandb projects (`coupled-muon-A0-repro`, `coupled-muon-ladder`,
`coupled-muon-ablation`) and writes them to `docs/data/*.parquet` (~50KB
total). The third runs every `fig*.py` and `tab*.py` script in
`docs/figures/`, writing PDFs to `docs/figures/` and TeX snippets to
`docs/tables/`.

## Layout

```
docs/
  report.tex         entry point for latexmk
  preamble.tex       packages + macros
  refs.bib
  README.md          this file
  sections/          per-section .tex (00_abstract..07_concl, A_stats..D_full_run_tables)
  figures/
    _common.py       palette, rcParams, paths
    _fetch.py        wandb.Api() + scan_history + parquet caching
    _stats.py        bootstrap helpers (mirrors src/coupled_muon_nanogpt/analysis/bootstrap.py)
    fetch_all.py     CLI; refreshes docs/data/*.parquet
    fig*.py          one figure per file -> docs/figures/*.pdf
    figB*.py         appendix B figures (§d.4 ablations)
    tab*.py          one table per file  -> docs/tables/*.tex
    make_all.py      runs every fig*.py + tab*.py
  tables/            (gitignored regenerated tex snippets)
  data/              parquet snapshots, committed
  build/             latexmk output (gitignored)
```

## How to refresh numbers after a sync

When new Stage 1 or §d.4 cells sync to wandb:

```
uv run python docs/figures/fetch_all.py --refresh
uv run python docs/figures/make_all.py
latexmk -pdf -outdir=docs/build docs/report.tex
```

Inspect `docs/tables/tab05_xstage_diffs.tex` — if any rung's CI changes
direction (brackets-zero status flips), update the corresponding sentence in
`sections/04_stage2.tex` or `sections/05_synthesis.tex`. The build script
prints a summary at the end of `tab05_xstage_diffs.py` ("Coupled-vs-Muon CI
brackets zero on N/7 rungs") to make this easy to spot.

## Part II sparse-MoE rebuild

```
uv run python docs/figures/fetch_part2.py --refresh --summary-only
uv run python docs/figures/make_part2.py
latexmk -pdf -outdir=docs/build docs/report_part2.tex
```

`fetch_part2.py` writes `docs/data/part2_manifest.json`,
`docs/data/part2_summary*.parquet`, `docs/data/part2_completion.csv`,
`docs/data/part2_missing_cells.csv`, and
`docs/data/part2_discarded_duplicates.csv`. Missing future Phase-2 projects
are recorded in the manifest and are not treated as CLI failures.

History/probe curves are optional because W\&B `scan_history` is slow:

```
uv run python docs/figures/fetch_part2.py --with-history
```
