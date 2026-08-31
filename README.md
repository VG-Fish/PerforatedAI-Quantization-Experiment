# Dendritic Quantization Benchmark

This repo contains a `uv`-managed benchmark scaffold for a 25-model / 12-condition dendritic quantization experiment.

## How It Works

The benchmark automates training neural networks under different quantization and pruning conditions to measure the impact of [PerforatedAI](https://github.com/PerforatedAI/PerforatedAI) techniques. Here's the workflow:

1. **Setup**: Initialize a Python environment with all dependencies (`uv sync`)
2. **Download Data** (optional): Pre-download datasets for specific models with `uv run dqb download_data`
3. **Train**: Run `uv run dqb run` to train models across 12 conditions that isolate baseline vs dendritic models at each quantization level
4. **Results**: Training metrics are saved to `results/<model>/<condition>/` with per-epoch histories and final performance records
5. **Compare**: Generate comparison charts and summary reports using `uv run dqb compare`
6. **Visualize**: Render training curves and analysis plots with `uv run dqb generate_graphs`
7. **Benchmark**: Measure inference latency on your hardware with `uv run dqb benchmark_models`
8. **Clean**: Remove generated outputs recorded from previous commands with `uv run dqb clean`

Each condition applies only two experimental factors to the same models: quantization level and whether the model is dendritic, allowing cleaner side-by-side comparison of efficiency vs. accuracy tradeoffs.

## Baseline Quality

Every result here is a comparison against `base_fp32`, so an under-trained
baseline distorts both the dendrite delta and the measured quantization
robustness — a model still far from its own optimum has slack that either
intervention can take up. The FP32 recipes therefore track each model's
published training setup rather than a shared default, and each entry in
`BenchmarkRunner._training_hyperparameters` cites its reference.

Practically, that means every model carries its own learning-rate schedule
(`constant`, `step`, `cosine`, or `linear`, with optional warmup, label
smoothing, and gradient clipping) instead of a flat rate for the whole run.
`information/DOCUMENTATION.md` has the full account of what was changed and why;
`information/MODEL_REFERENCE.md` lists the per-model settings.

Dendritic FP32 training defaults to the same fixed epoch budget as the matching non-dendritic model. PerforatedAI uses HISTORY plateau detection for every model; dendrite insertion is active during the first 80% of the budget, then frozen for the final 20% so the selected architecture can settle. Pass `--dynamic-dendritic-training` to keep the same HISTORY schedule open until `training_complete=True`; any epochs beyond the canonical budget are written under `results/<model>/<condition>/continued_until_complete/`. Fixed switching is available only as an explicit schedule diagnostic through `--pai-fixed-switch-interval EPOCHS`, and requested versus observed switches are recorded in `pai_summary.json`.

For Apple Silicon runs, the training path selects MPS automatically, disables CUDA-only pinned memory, keeps DataLoader workers persistent, uses larger per-model batch sizes, sets high float32 matmul precision where supported, and compiles non-dendritic MPS models with `torch.compile(..., backend="aot_eager")` when available. Long dendritic runs periodically clear PerforatedAI processor buffers and the accelerator cache to avoid MPS memory pressure during late epochs.

## Setup

```bash
git clone https://github.com/VG-Fish/PerforatedAI-Quantization-Experiment.git
uv venv .venv
uv sync
```

The benchmark downloads public datasets on first use and caches them under `data/` by default. Set `DQB_DATA_ROOT=/path/to/cache` if you want the datasets stored somewhere else.

`uv sync` installs the runtime dependencies plus the `dev` group (`pytest`, `ty`). The
static-analysis tools are a separate group: `uv sync --group audit` adds `vulture`,
`bandit`, and `deadcode`.

## Checks

```bash
./scripts/ci.sh
```

Runs `ty check`, the test suite, and the generated-documentation check — the same three
steps `.github/workflows/ci.yml` runs on push and pull request. Everything in it is
offline and CPU-only: no dataset is downloaded, no model is built, and no result tree is
written. The suite includes a smoke matrix over all 24 models × 12 conditions, property
tests on the artifact manifest that decides what is reportable, and the seed-paired
statistics that gate every dendrite claim.

## Run

```bash
uv run dqb run
```

Results are written to:
- `results/<model>/<condition>/`
- `comparison/`
- `data/` for downloaded datasets, unless `DQB_DATA_ROOT` is set
- `logs/` for command logs
- `.dqb/command_config.json`, a local registry of generated output paths used by `uv run dqb clean`

To scope outputs for a specific experiment, use `--results-directory`:

```bash
uv run dqb --results-directory experiment_a run
uv run dqb --results-directory experiment_a compare
uv run dqb --results-directory experiment_a generate_graphs
uv run dqb --results-directory experiment_a benchmark_models
```

When set, results are read/written under `results/<results-directory>/...`.

`--results-root`, `--results-directory` and `--logging-dir` apply to every
subcommand and may be given either before or after it, so both of these are
equivalent:

```bash
uv run dqb --results-root results_archive compare
uv run dqb compare --results-root results_archive
```

If a flag is given in both positions, the one after the subcommand wins.

Generated output paths must resolve under the current working directory or the
system temp directory. To use another scratch root explicitly, set
`DQB_ALLOWED_OUTPUT_ROOTS=/path/to/scratch` before running `dqb`.

Dendritic runs pass PerforatedAI save names under `PAI/`, so library-created
checkpoints and sidecars stay in the `PAI/` tree. They also snapshot the
active PerforatedAI config to
`results/<model>/<condition>/PAI_config.json` and
`PAI/<model>_<condition>_PAI_config.json`, so each model/condition keeps its
own reproducibility config instead of relying only on the latest global
`PAI/PAI_config.json`.

Every condition attempt receives a unique artifact ID and PAI namespace. A
completed `artifact_manifest.json` binds that identity to hashes of the model,
metrics, history, and record files. Reporting, reuse, source loading, and
latency benchmarking reject missing, incomplete, mismatched, or modified
manifests. An interrupted run resumes only when its `artifact_attempt.json`
token matches the existing epoch checkpoint; otherwise use `--fresh`.

When `--allow-PQAT` is supplied, PQAT is applied to all quantized conditions
after their source checkpoint has been trained. Each quantized run saves a PTQ
evaluation under `before_pqat/`, fine-tunes for the model-aware PQAT budget, and
saves the post-PQAT artifacts under `after_pqat/`.

The model implementations are part of the experimental definition. After
changing architectures, rerun affected models with `--ignore-saved-models` or a
fresh `--results-directory` so old checkpoints and records do not mix with the
new model definitions.

The same applies to the dataset builders and the training recipes. The
2026-08-06 baseline-quality pass changed model input widths (TextCNN's vocabulary,
the molecular and IMDB-BINARY node features), dataset preprocessing (SMILES
parsing, target standardization, Cora feature normalization, MNIST augmentation
for the classifiers), CapsNet's loss, and every model's learning-rate schedule,
and the 2026-08-07 pass then rewrote `ppo_bipedalwalker` (real PPO), `gcn`
(full-graph transductive Cora), DistilBERT's validation split, and PointNet's
dataloader.

**No results in this repository reflect any of that yet.** The pre-fix result
trees were archived on 2026-08-08 and removed from the working tree:

| Archive | Was |
|---|---|
| `archive/old_models_pre-fix_20260808.zip` | `results/old_models` |
| `archive/results_dynamic_pre-fix_20260808.zip` | `results_dynamic/`, `comparison_dynamic/`, `logs_dynamic/` |

Each zip stores original relative paths, so `unzip archive/<name>.zip` from the
repository root restores the tree where it was. They are kept for provenance
only — the numbers in them are not comparable to anything produced by the
current code. See `information/REMAINING_FIXES.md` §3.4 for the retraining that
is still owed.

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
uv run dqb clean --help

# Download datasets
uv run dqb download_data
uv run dqb download_data --models lenet5 mpnn
uv run dqb download_data --strict

# Train runs
uv run dqb run
uv run dqb run --models lenet5 textcnn

# Explicitly opt into the full exploratory 24-model roster
uv run dqb run --models all
uv run dqb run --conditions base_fp32 dendrites_fp32
uv run dqb run --allow-PQAT
uv run dqb run --dynamic-dendritic-training
uv run dqb run --pai-fixed-switch-interval 8  # diagnostic only
uv run dqb run --ignore-saved-models

# Control how a run is parallelised (default: 4 workers + live progress table)
uv run dqb run --jobs 1                 # train in this process, print to terminal
uv run dqb run --jobs 8
uv run dqb run --fresh                  # drop stale epoch checkpoints first
uv run dqb run --detach                 # launch workers and exit
uv run dqb run --status                 # report on a running (or finished) run
uv run dqb run --logging-dir logs_run7 -i 120

# Compare outputs (includes per-model benchmark timing plots when benchmarks/manifest.csv exists)
uv run dqb compare
uv run dqb compare --manifest
uv run dqb compare --benchmark-root benchmarks

# Generate training graphs
uv run dqb generate_graphs
uv run dqb generate_graphs --regenerate-graphs

# Run latency benchmarks
uv run dqb benchmark_models
uv run dqb benchmark_models --models lenet5 resnet18_cifar10
uv run dqb benchmark_models --batch-sizes 1 32 --num-runs 10
uv run dqb benchmark_models --benchmark-root my_benchmarks
uv run dqb benchmark_models --comparison-root comparison

# Regenerate the current-state guide (and verify it in CI)
uv run dqb docs
uv run dqb docs --check

# Inventory the generated evidence trees before archiving or deleting them
uv run dqb evidence_index

# Remove generated outputs recorded in .dqb/command_config.json
uv run dqb clean --dry-run
uv run dqb clean

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

Start with the generated current-state guide. It is rendered from the code that runs
the experiment — the model/condition registries, the artifact-validity rules, and this
CLI's own option registry — so it cannot drift from what the repository actually does:

- [information/CURRENT_GUIDE.md](information/CURRENT_GUIDE.md) — model roster, condition
  grid, what makes a result reportable, and the full command reference.
  Regenerate with `uv run dqb docs`; CI runs `uv run dqb docs --check`.

Everything else under `information/` is hand-written and indexed by status:

- [information/HISTORICAL_INDEX.md](information/HISTORICAL_INDEX.md) — every document,
  whether it is current, historical, or superseded, and what it may still be cited for.
- [information/RETENTION_POLICY.md](information/RETENTION_POLICY.md) — which generated
  trees may be archived or deleted, and what must exist first.
- [information/EVIDENCE_INDEX.md](information/EVIDENCE_INDEX.md) — the generated
  inventory of every training record on disk, its run namespace, and its manifest
  verdict. Rebuild with `uv run dqb evidence_index`.
- [information/DENDRITE_EFFECT_AUDIT_2026-08-30.md](information/DENDRITE_EFFECT_AUDIT_2026-08-30.md)
  — the standing verdict on whether the dendrite effect beats noise and more training.
- [information/audit/audit_report.md](information/audit/audit_report.md) — the cleanup
  priority ledger and its implementation updates.

Historical documents keep their original numbers on purpose. When one disagrees with
the generated guide, the generated guide is the current state.

## Available commands (uv run dqb)

The CLI exposes several helpful subcommands. Every option, with its default, is in the generated command reference in `information/CURRENT_GUIDE.md`; `information/CLI_DIAGRAMS.md` has the per-command flowcharts.

- `uv run dqb run`
	- Train models across one or more conditions. A bare run uses the evidence-backed default roster (`lenet5`, `tcn_forecaster`, `pointnet_modelnet40`, `resnet18_cifar10`, and `saint_adult`) across every condition. Use `--models all` to opt into the full exploratory 24-model roster.
	- Splits the selected models across 4 worker processes by default and prints a live progress table until they all exit. Training is compute-bound rather than data-bound, so this cuts wall-clock close to linearly — the full 23-model FP32 pass is ~24h sequentially. Each model keeps all of its conditions in one worker, so the `dendrites_q8` → `dendrites_fp32` dependency order still holds. Worker output goes to `<logging-dir>/streams/stream_N.log`, and every progress table is appended to `<logging-dir>/run_progress.log`.
	- Ctrl-C detaches the progress display without stopping training. Stop training with `pkill -f 'dqb run'`.
	- Useful flags: `--models`, `--conditions`, `--results-root`, `--results-directory`, `--comparison-root`, `--ignore-saved-models`, `--allow-PQAT`, `--dynamic-dendritic-training`, `--jobs` (1 trains in-process and prints to the terminal), `--fresh` (delete stale `epoch_checkpoint.pt` files first — `--ignore-saved-models` does *not* cover those), `--detach`, `--status`, `-i/--interval`.
	- Examples:
        ```bash
        uv run dqb run
        uv run dqb --results-directory experiment_a run
        uv run dqb run --models lenet5 textcnn
        uv run dqb run --conditions base_fp32 dendrites_fp32
        uv run dqb run --allow-PQAT
        uv run dqb run --dynamic-dendritic-training
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
	- Also writes two audit tables to `--comparison-root`: `dendrite_audit.csv` (per-arm validity, requested vs observed switch schedule, termination reason, effect verdict) and `dendrite_effect_statistics.csv` (paired seeds, noise floor, mean improvement, p-value, verdict).
	- Useful flags: `--manifest` (write a manifest CSV), `--seed-roots` (other seeds' results roots), `--results-root`, `--results-directory`, `--comparison-root`.
	- `--seed-roots` is what makes the statistics usable: one results root holds one seed, and a dendrite effect is only claimable on three paired seeds. Without it every effect verdict stays `insufficient_seeds`.
	- Examples:
        ```bash
        uv run dqb compare
        uv run dqb --results-directory experiment_a compare
        uv run dqb compare --manifest
        uv run dqb compare --results-root results --comparison-root comparison
        uv run dqb --results-root experiments/dynamic12/tcn_audited_default/seed_0/results compare \
            --seed-roots experiments/dynamic12/tcn_audited_default/seed_1/results \
                         experiments/dynamic12/tcn_audited_default/seed_2/results
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
	- Results are saved to `benchmarks/<model>/` with per-condition latency measurements. Per-model latency charts are written to `comparison/<model>/`.
	- Already-benchmarked model/condition pairs are skipped by default (existing `{condition}.json` files are reused). Pass `--re-run` to force re-measurement.
	- Useful flags: `--models` (subset), `--conditions` (subset), `--batch-sizes` (e.g., `1 8 32`), `--num-runs` (independent timing runs per batch, default 5), `--re-run` (ignore existing results), `--results-root`, `--results-directory`, `--benchmark-root`, `--comparison-root`.
	- Examples:
        ```bash
        uv run dqb benchmark_models
        uv run dqb --results-directory experiment_a benchmark_models
        uv run dqb benchmark_models --models lenet5 resnet18_cifar10
        uv run dqb benchmark_models --batch-sizes 1 32 --num-runs 10
        uv run dqb benchmark_models --benchmark-root my_benchmarks
        uv run dqb benchmark_models --comparison-root my_comparison
        uv run dqb benchmark_models --re-run
        ```

- `uv run dqb clean`
	- Removes generated outputs listed in `.dqb/command_config.json`. The registry is updated automatically by `run`, `download_data`, `compare`, `generate_graphs`, and `benchmark_models`.
	- Recorded paths include user-supplied output locations such as `--results-root`, `--results-directory`, `--comparison-root`, `--benchmark-root`, `--logging-dir`, and `DQB_DATA_ROOT`.
	- Use `--dry-run` to inspect what would be deleted before removing files.
	- Examples:
        ```bash
        uv run dqb clean --dry-run
        uv run dqb clean
        ```

- `uv run dqb docs`
	- Regenerates `information/CURRENT_GUIDE.md` from the model/condition registries, the artifact-validity rules, and the CLI option registry. Hand-written history under `information/` is never regenerated.
	- `--check` writes nothing and exits non-zero when the checked-in guide no longer matches the code. This is what CI runs.
	- Examples:
        ```bash
        uv run dqb docs
        uv run dqb docs --check
        ```

- `uv run dqb evidence_index`
	- Walks the generated roots (`results`, `experiments`, `comparison`, `logs*`, `archive`) and writes `information/evidence_index.json` plus `information/EVIDENCE_INDEX.md`: every training record on disk with its run namespace, artifact id, seed, metric, and manifest verdict.
	- `--verify` re-hashes every manifest-owned file. Run it before archiving or deleting any tree — see `information/RETENTION_POLICY.md`.
	- Examples:
        ```bash
        uv run dqb evidence_index
        uv run dqb evidence_index --verify --roots experiments/dynamic12
        ```

- `uv run dqb --help`
	- Show help for the `dqb` command and available subcommands/flags.

For every flag and its default, see the command reference in `information/CURRENT_GUIDE.md`; for the command flow diagrams, `information/CLI_DIAGRAMS.md`.
