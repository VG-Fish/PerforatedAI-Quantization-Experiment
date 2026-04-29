# Script Architecture Guide

This guide explains how the benchmark code is organized and how data moves from the CLI to training, saved records, and plots.

## Entry Point

### `src/dendritic_benchmark/cli.py`

Defines the `dqb` command exposed by `pyproject.toml`.

- `uv run dqb run` creates a `BenchmarkRunner` and runs selected models and conditions.
- `uv run dqb compare` loads existing `record.json` files and rebuilds per-model and comparison outputs.
- `--results-root` controls where per-model artifacts are read or written.
- `--comparison-root` controls where cross-model plots and `summary.csv` are written.
- `.env` credentials are loaded before running so PerforatedAI can use local beta credentials when available.

## Experiment Configuration

### `src/dendritic_benchmark/specs.py`

Contains the declarative model and condition lists.

- `MODEL_SPECS` lists the 10 benchmark models, their display names, domains, datasets, metric names, and whether each metric should be maximized or minimized.
- `CONDITION_SPECS` lists the 13 experimental conditions, including quantization bit width, source checkpoint, dendrite usage, pruning, QAT, and fine-tuning settings.
- `model_by_key()` and `condition_by_key()` validate CLI keys and return the matching spec.

Update this file first when adding a new model or condition.

## Pipeline Orchestration

### `src/dendritic_benchmark/pipeline.py`

Coordinates complete benchmark runs.

- Creates result and comparison directories.
- Expands condition dependencies so requested quantized conditions have their source checkpoints available.
- Downloads/caches real datasets and builds model instances.
- Loads checkpoints from previous conditions when a condition depends on another condition.
- Applies PerforatedAI wrapping when dendritic conditions are requested.
- Calls `train_and_evaluate()` for each model-condition pair.
- Saves records and regenerates per-model and cross-model reports.

The important class is `BenchmarkRunner`.

## Real Data

### `src/dendritic_benchmark/data.py`

Builds task bundles from the public datasets named in the benchmark plan and stores downloaded assets under `data/` by default. Set `DQB_DATA_ROOT` to move the cache.

- TorchVision/Torchaudio download MNIST and SpeechCommands.
- Hugging Face Datasets caches AG News and SST-2.
- Direct URL downloaders cache ETTh1, Adult Income, Cora, and ESOL.
- Gymnasium creates CartPole-v1 rollouts and caches the resulting observations.
- WFDB downloads MIT-BIH ECG records from PhysioNet.

Each bundle returns train, validation, and test data loaders plus metric metadata.

## Models

### `src/dendritic_benchmark/models.py`

Defines compact PyTorch implementations for every benchmark slot.

- `LeNet5` for image classification.
- `M5` for 1D audio classification.
- `LSTMForecaster` for sequence forecasting.
- `TextCNN` for text classification.
- `GCN` for graph classification.
- `TabNetLite` for tabular classification.
- `MPNN` for graph regression.
- `ActorCritic` for the reinforcement-learning proxy task.
- `LSTMAutoencoder` for anomaly detection.
- `DistilBertFallback` as a lightweight DistilBERT-style sequence classifier stand-in.

`build_model()` selects the factory by model key.

## Compatibility Helpers

### `src/dendritic_benchmark/compat.py`

Isolates optional or environment-sensitive dependencies.

- Safely imports PyTorch and dotenv.
- Chooses `mps`, CUDA, or CPU at runtime.
- Loads `.env` credentials and mirrors PerforatedAI token/email aliases.
- Wraps models with PerforatedAI when installed, otherwise returns the original model.
- Provides simple symmetric, ternary, and binary weight quantization helpers.

Keeping this logic centralized lets the rest of the package stay readable and import safely before optional ML dependencies are available.

## Training And Evaluation

### `src/dendritic_benchmark/training.py`

Runs each individual condition.

- Moves the model to the selected device.
- Applies pruning for pruned conditions.
- Applies fake quantization or QAT-style weight projection for low-bit conditions.
- Trains for the condition-specific number of epochs.
- Evaluates validation and test metrics.
- Saves `model.pt`, and dendritic sidecars such as `best_model`, `final_clean_pai`, `best_arch_scores.csv`, and `paramCounts.csv`.
- Writes `metrics.json`, `history.csv`, and returns a `TrainingRecord`.

`TrainingRecord` is the normalized record format used by plotting and manifests.

## Results And Reports

### `src/dendritic_benchmark/results.py`

Reads and writes benchmark records and dispatches plot generation.

- `save_training_record()` writes `record.json` and `record.csv`.
- `load_training_records()` scans `results/*/*/record.json`.
- `write_manifest()` writes the combined `results/manifest.csv`.
- `write_model_reports()` creates each model's metric, parameter, and file-size charts.
- `write_comparison_reports()` creates cross-model heatmaps, scatter plots, dendrite delta bars, and `summary.csv`.

This module computes normalized score retention and file-size reduction before handing data to the plotting layer.

## Plotting

### `src/dendritic_benchmark/plots.py`

Creates all charts using Matplotlib's non-interactive `Agg` backend.

- Bar charts use wrapped tick labels, adaptive rotation, and renderer-based overlap detection.
- Value labels above bars are only kept when their measured bounding boxes do not collide.
- Heatmaps rotate or shrink labels when needed and use dynamic figure sizes.
- Scatter labels are placed using multiple candidate offsets; labels that still collide are hidden and counted in a small note.
- Existing output filenames remain `.svg`, so `uv run dqb compare` can regenerate plots from completed training records without retraining.

The overlap detection uses Matplotlib's actual rendered text extents instead of guessing from string length.

## Typical Flows

### Full benchmark

```bash
uv run dqb run
```

Flow:

1. `cli.py` parses the command.
2. `pipeline.py` loops over models and conditions.
3. `data.py` downloads missing datasets and builds real-data loaders.
4. `models.py` builds each PyTorch model.
5. `training.py` trains, evaluates, and saves artifacts.
6. `results.py` writes records and reports.
7. `plots.py` writes Matplotlib SVG charts.

### Regenerate plots after training

```bash
uv run dqb compare --manifest
```

Flow:

1. `cli.py` loads saved records.
2. `results.py` optionally rewrites `manifest.csv`.
3. `results.py` rewrites per-model charts from saved records.
4. `results.py` recomputes normalized comparison data.
5. `plots.py` rewrites the comparison SVG files.

This is the fastest way to refresh charts after changing plot logic.
