# Script Architecture Guide

This guide explains how the benchmark code is organized and how data moves from the CLI to training, saved results, and plots.

## Entry Point

### `src/dendritic_benchmark/cli.py`

The package exposes the `dqb` entry point via `pyproject.toml`.

- `uv run dqb run` constructs a `BenchmarkRunner` and executes the selected models and conditions.
- `uv run dqb compare` loads saved `record.json` files and rebuilds per-model and cross-model comparison outputs.
- `uv run dqb generate_graphs` regenerates training plots from existing `history.csv` files.
- `--results-root` controls where per-model results are read from or written to.
- `--comparison-root` controls where comparison outputs are written.
- `.env` credentials are loaded before running so PerforatedAI aliases and API variables are mirrored into the environment.

## Experiment Configuration

### `src/dendritic_benchmark/specs.py`

This file declares benchmark model and condition metadata.

- `MODEL_SPECS` defines the 10 benchmark models and their evaluation metrics.
- `CONDITION_SPECS` defines the 13 experimental conditions, including source checkpoint dependencies, quantization settings, dendrite usage, pruning, and QAT.
- `model_by_key()` and `condition_by_key()` look up valid keys and raise on unknown values.

## Pipeline Orchestration

### `src/dendritic_benchmark/pipeline.py`

`BenchmarkRunner` orchestrates the full benchmark run.

- It creates the results and comparison directories.
- It expands requested condition keys into a dependency-resolved order.
- It builds dataset bundles via `data.py` and models via `models.py`.
- It loads source checkpoints when a condition depends on a previous result.
- It perforates models with PerforatedAI for dendritic conditions and configures them as needed.
- It calls `train_and_evaluate()` for every model-condition pair.
- It writes per-condition training records and regenerates comparison outputs during the run.

The run method also computes `max_epochs` from `_base_epoch_budget()` so baseline and dendritic FP32 conditions receive larger budgets than post-training quantized conditions.

## Real Data

### `src/dendritic_benchmark/data.py`

Builds task bundles for each benchmark task and caches datasets under `data/` by default. Set `DQB_DATA_ROOT` to move the cache location.

## Models

### `src/dendritic_benchmark/models.py`

Defines the compact PyTorch model implementations used by the benchmark and exposes `build_model()` to construct each architecture by key.

## Compatibility Helpers

### `src/dendritic_benchmark/compat.py`

Isolates optional dependencies and PerforatedAI integration.

- Safely imports PyTorch and optional `dotenv`.
- Detects MPS, CUDA, or CPU at runtime.
- Mirrors PerforatedAI token and email aliases from environment variables.
- Wraps and configures models with PerforatedAI when available.
- Provides simple symmetric, ternary, and binary quantization helpers.

## Training And Evaluation

### `src/dendritic_benchmark/training.py`

Runs each individual condition.

- Moves the model to the selected device.
- Applies pruning for pruned conditions.
- Applies quantization for low-bit conditions and optionally QAT-style weight projection.
- Compiles non-dendritic MPS models with `torch.compile(..., backend='aot_eager')` when available.
- Trains for the condition-specific epoch budget, or skips training for post-training quantization conditions.
- Evaluates validation and test metrics.
- Saves `model.pt`, and for dendritic runs also writes `best_model` and `final_clean_pai`.
- Writes `metrics.json`, `history.csv`, and returns a normalized `TrainingRecord`.

For dendritic runs, the module also writes `best_arch_scores.csv` and `paramCounts.csv`.

## Results And Reports

### `src/dendritic_benchmark/results.py`

Reads and writes benchmark records and generates final visualizations.

- `save_training_record()` writes `record.json` and `record.csv` for each condition.
- `load_training_records()` scans saved records under `results/*/*/record.json`.
- `write_manifest()` writes `results/manifest.csv`.
- `write_model_reports()` generates per-model metric, parameter, and size comparison charts.
- `write_comparison_reports()` generates cross-model heatmaps, a tradeoff scatter plot, a dendrite delta chart, and `summary.csv`.
- `generate_training_graphs()` regenerates training plots from `history.csv` and architecture evolution plots from `best_arch_scores.csv`.

## Plotting

### `src/dendritic_benchmark/plots.py`

Creates charts using Matplotlib's non-interactive `Agg` backend.

- Bar charts are drawn with adaptive label sizing and optional hatch patterns for PTQ conditions.
- Line charts plot metric and loss series over epochs.
- Heatmaps and scatter plots are used for cross-model comparison outputs.
- Plot label overlap detection uses rendered text extents to avoid collisions.

## Typical Flows

### Full benchmark

```bash
uv run dqb run
```

Flow:

1. `cli.py` parses the command and global options.
2. `pipeline.py` loops over selected models and conditions.
3. `data.py` downloads missing datasets and builds task bundles.
4. `models.py` builds the requested model architecture.
5. `training.py` trains, evaluates, and saves artifacts.
6. `results.py` writes records and reports.
7. `plots.py` writes SVG charts.

### Regenerate plots after training

```bash
uv run dqb compare --manifest
```

Flow:

1. `cli.py` loads saved record files.
2. `results.py` optionally rewrites `manifest.csv`.
3. `results.py` rewrites per-model charts from saved records.
4. `results.py` rewrites comparison charts in the comparison directory.

### Generate training graphs

```bash
uv run dqb generate_graphs --results-root results
```

Flow:

1. `cli.py` loads the results root.
2. `results.py` scans existing `history.csv` files.
3. `results.py` recreates `plots/` directories and writes metric and loss curves.
4. For dendritic conditions, it also generates architecture evolution plots from `best_arch_scores.csv`.
