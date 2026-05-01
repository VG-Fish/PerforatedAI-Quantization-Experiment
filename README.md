# Dendritic Quantization Benchmark

This repo now contains a `uv`-managed benchmark scaffold for the 10-model / 13-condition experiment described in `Dendritic Quantization Benchmark Plan.md`.

## How It Works

The benchmark automates training neural networks under different quantization and pruning conditions to measure the impact of [PerforatedAI](https://github.com/PerforatedAI/PerforatedAI) techniques. Here's the workflow:

1. **Setup**: Initialize a Python environment with all dependencies (`uv sync`)
2. **Download Data** (optional): Pre-download datasets for specific models with `uv run dqb download_data`
3. **Train**: Run `uv run dqb run` to train models across 12 conditions that isolate baseline vs dendritic models at each quantization level
4. **Results**: Training metrics are saved to `results/<model>/<condition>/` with per-epoch histories and final performance records
5. **Compare**: Generate comparison charts and summary reports using `uv run dqb compare`
6. **Visualize**: Render training curves and analysis plots with `uv run dqb generate_graphs`
7. **Benchmark**: Measure inference latency on your hardware with `uv run dqb benchmark_models`

Each condition applies only two experimental factors to the same models: quantization level and whether the model is dendritic, allowing cleaner side-by-side comparison of efficiency vs. accuracy tradeoffs.

Dendritic FP32 training uses PerforatedAI's dynamic tracker hooks (`set_optimizer`, `setup_optimizer`, and `add_validation_score`) during validation and keeps training until PerforatedAI returns `training_complete=True`. The model's configured `max_epochs` is treated as the canonical base-model budget, not as a hard stop for dendrite growth. Any dendritic epochs beyond that canonical budget are written under `results/<model>/<condition>/continued_until_complete/` so the over-budget PAI completion phase is explicit in the artifacts.

For Apple Silicon runs, the training path selects MPS automatically, disables CUDA-only pinned memory, keeps DataLoader workers persistent, uses larger per-model batch sizes, sets high float32 matmul precision where supported, and compiles non-dendritic MPS models with `torch.compile(..., backend="aot_eager")` when available.

## Setup

```bash
git clone https://github.com/VG-Fish/PerforatedAI-Quantization-Experiment.git
uv venv .venv
uv sync
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
- `logs/` for command logs

To scope outputs for a specific experiment, use `--results-directory`:

```bash
uv run dqb --results-directory experiment_a run
uv run dqb --results-directory experiment_a compare
uv run dqb --results-directory experiment_a generate_graphs
uv run dqb --results-directory experiment_a benchmark_models
```

When set, results are read/written under `results/<results-directory>/...`.

Dendritic runs pass PerforatedAI save names under `PAI/`, so library-created
checkpoints and sidecars stay in the `PAI/` tree. They also snapshot the
active PerforatedAI config to
`results/<model>/<condition>/PAI_config.json` and
`PAI/<model>_<condition>_PAI_config.json`, so each model/condition keeps its
own reproducibility config instead of relying only on the latest global
`PAI/PAI_config.json`.

When `--allow-PQAT` is supplied, PQAT is applied to all quantized conditions
after their source checkpoint has been trained. Each quantized run saves a PTQ
evaluation under `before_pqat/`, fine-tunes for the model-aware PQAT budget, and
saves the post-PQAT artifacts under `after_pqat/`.

## Compare Existing Runs

```bash
uv run dqb compare
```

## Shell Commands

```bash
# Show top-level help
uv run dqb --help

# Show help for each subcommand
uv run dqb run --help
uv run dqb download_data --help
uv run dqb compare --help
uv run dqb generate_graphs --help
uv run dqb benchmark_models --help

# Download datasets
uv run dqb download_data
uv run dqb download_data --models lenet5 mpnn
uv run dqb download_data --strict

# Train runs
uv run dqb run
uv run dqb run --models lenet5 textcnn
uv run dqb run --conditions base_fp32 dendrites_fp32
uv run dqb run --allow-PQAT
uv run dqb run --ignore-saved-models

# Compare outputs (includes benchmark timing plots when benchmarks/manifest.csv exists)
uv run dqb compare
uv run dqb compare --manifest
uv run dqb compare --benchmark-root benchmarks

# Generate training graphs
uv run dqb generate_graphs
uv run dqb generate_graphs --regenerate-graphs

# Run latency benchmarks
uv run dqb benchmark_models
uv run dqb benchmark_models --models lenet5 resnet18_cifar10
uv run dqb benchmark_models --batch-sizes 1 32 --num-runs 20
uv run dqb benchmark_models --benchmark-root my_benchmarks

# Use an experiment namespace under results/
uv run dqb --results-directory experiment_a run
uv run dqb --results-directory experiment_a compare
uv run dqb --results-directory experiment_a generate_graphs
uv run dqb --results-directory experiment_a benchmark_models
```

## Shell Completion (Tab)

Tab completion is installed automatically with the project, so no manual shell
setup is required. After installation, `Tab` completion works for
`uv run dqb` subcommands and flags.

The completion bridge only returns completions when the command starts with
`uv run dqb`; other `uv` commands fall back to your shell's normal behavior. If
an existing terminal does not pick up completion immediately, open a new shell.

## Documentation

The repository includes extended documentation under the `information/` directory. Below are short summaries with links to the full markdown files.

- `information/DOCUMENTATION.md` — Comprehensive project documentation (recommended start):
	- Experiment plan for 10 models across the benchmark condition grid (per-model and cross-model graphs).
	- Execution strategy targeting Apple M3 Pro (MPS) and PyTorch integration notes.
	- Detailed PerforatedAI (PAI) integration steps, quantization (`torchao`) and pruning examples, and training loop hooks.
	- Round-2 expansion with 15 additional models and research findings from a preliminary run.

- `information/CLI_DIAGRAMS.md` — CLI reference and diagrams:
	- Command summaries and Mermaid flowcharts for `uv run dqb run`, `uv run dqb download_data`, `uv run dqb compare`, `uv run dqb generate_graphs`, and `uv run dqb benchmark_models`.
	- Global CLI flags and the recommended output directory layout.

Read the full documents for architecture details, hypotheses, and example commands:

[DOCUMENTATION.md](information/DOCUMENTATION.md)

[CLI_DIAGRAMS.md](information/CLI_DIAGRAMS.md)


## Available commands (uv run dqb)

The CLI exposes several helpful subcommands. See `information/CLI_DIAGRAMS.md` for flowcharts and full details.

- `uv run dqb run`
	- Train models across one or more conditions. By default runs all models & conditions defined in the project.
	- Useful flags: `--models`, `--conditions`, `--results-root`, `--results-directory`, `--comparison-root`, `--ignore-saved-models`, `--allow-PQAT`.
	- Examples:
        ```bash
        uv run dqb run
        uv run dqb --results-directory experiment_a run
        uv run dqb run --models lenet5 textcnn
        uv run dqb run --conditions base_fp32 dendrites_fp32
        uv run dqb run --allow-PQAT
        uv run dqb run --ignore-saved-models
        ```

- `uv run dqb download_data`
	- Pre-downloads and prepares datasets required by the selected models.
	- Useful flags: `--models` (subset), `--strict` (fail on any download error), `--results-root`, `--results-directory` (accepted but not used by this command).
	- Examples:
        ```bash
        uv run dqb download_data
        uv run dqb download_data --models lenet5 mpnn
        uv run dqb download_data --strict
        ```

- `uv run dqb compare`
	- Rebuilds comparison charts and summary reports from saved `record.json` files in `results/` without retraining.
	- Useful flags: `--manifest` (write a manifest CSV), `--results-root`, `--results-directory`, `--comparison-root`.
	- Examples:
        ```bash
        uv run dqb compare
        uv run dqb --results-directory experiment_a compare
        uv run dqb compare --manifest
        uv run dqb compare --results-root results --comparison-root comparison
        ```

- `uv run dqb generate_graphs`
	- Renders per-epoch training curves and other plots from saved `history.csv` files.
	- Useful flags: `--results-root`, `--results-directory`, `--regenerate-graphs` (force re-render even if plots exist).
	- Comparison outputs are intentionally not managed here; use `uv run dqb compare` for `comparison/`.
	- Examples:
        ```bash
        uv run dqb generate_graphs
        uv run dqb --results-directory experiment_a generate_graphs
        uv run dqb generate_graphs --regenerate-graphs
        ```

- `uv run dqb benchmark_models`
	- Measures wall-clock inference latency for all trained models using `torch.utils.benchmark.Timer`.
	- Results are saved to `benchmarks/<model>/` with per-condition latency measurements.
	- Useful flags: `--models` (subset), `--conditions` (subset), `--batch-sizes` (e.g., `1 8 32`), `--num-runs` (averaging runs), `--results-root`, `--results-directory`, `--benchmark-root`.
	- Examples:
        ```bash
        uv run dqb benchmark_models
        uv run dqb --results-directory experiment_a benchmark_models
        uv run dqb benchmark_models --models lenet5 resnet18_cifar10
        uv run dqb benchmark_models --batch-sizes 1 32 --num-runs 20
        uv run dqb benchmark_models --benchmark-root my_benchmarks
        ```

- `uv run dqb --help`
	- Show help for the `dqb` command and available subcommands/flags.

For full command flow diagrams and more flags, open `information/CLI_DIAGRAMS.md`.
