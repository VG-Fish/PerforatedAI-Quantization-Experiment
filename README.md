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

## Documentation

The repository includes extended documentation under the `information/` directory. Below are short summaries with links to the full markdown files.

- `information/DOCUMENTATION.md` — Comprehensive project documentation (recommended start):
	- Experiment plan for 10 models across 13 conditions (per-model and cross-model graphs).
	- Execution strategy targeting Apple M3 Pro (MPS) and PyTorch integration notes.
	- Detailed PerforatedAI (PAI) integration steps, quantization (`torchao`) and pruning examples, and training loop hooks.
	- Round-2 expansion with 15 additional models and research findings from a preliminary run.

- `information/CLI_DIAGRAMS.md` — CLI reference and diagrams:
	- Command summaries and Mermaid flowcharts for `uv run dqb run`, `uv run dqb download_data`, `uv run dqb compare`, and `uv run dqb generate_graphs`.
	- Global CLI flags and the recommended output directory layout.

Read the full documents for architecture details, hypotheses, and example commands:

[DOCUMENTATION.md](information/DOCUMENTATION.md)

[CLI_DIAGRAMS.md](information/CLI_DIAGRAMS.md)

