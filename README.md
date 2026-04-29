# Dendritic Quantization Benchmark

This repo now contains a `uv`-managed benchmark scaffold for the 10-model / 13-condition experiment described in `Dendritic Quantization Benchmark Plan.md`.

## Setup

```bash
uv venv .venv
```

The benchmark downloads public datasets on first use and caches them under `data/` by default. Set `DQB_DATA_ROOT=/path/to/cache` if you want the datasets stored somewhere else.

## Run

```bash
uv run dqb run
```

Results are written to:
- `results/<model>/<condition>/`
- `comparison/`
- `data/` for downloaded datasets, unless `DQB_DATA_ROOT` is set

## Compare Existing Runs

```bash
uv run dqb compare
```
