# Script Usage Guide

This repository exposes a small command-line interface for running the dendritic quantization benchmark and rebuilding reports from saved results.

## Prerequisites

Create or activate the `uv` environment first:

```bash
uv venv .venv
```

Then run commands through `uv` so the project uses the managed environment:

```bash
uv run dqb --help
```

If you have PerforatedAI beta credentials, place them in a local `.env` file at the repo root. The CLI will load them automatically.

Example:

```dotenv
PAITOKEN=your-token-here
PAIEMAIL=you@example.com
```

The loader also recognizes the following aliases:

- `PERFORATEDAI_API_KEY`
- `PERFORATEDAI_TOKEN`
- `PERFORATEDBP_API_KEY`
- `PERFORATEDBP_TOKEN`
- `PERFORATEDAI_EMAIL`
- `PERFORATEDBP_EMAIL`
- `PAITOKEN`
- `PAIEMAIL`

## Commands

### `dqb run`

Runs the benchmark pipeline for selected models and conditions.

```bash
uv run dqb run
```

Useful options:

```bash
uv run dqb run --models lenet5 textcnn
uv run dqb run --conditions base_fp32 base_q8 dendrites_fp32
uv run dqb run --results-root results
uv run dqb run --comparison-root comparison
```

What it does:

- Downloads and caches datasets under `data/` by default.
- Builds task bundles for each selected model.
- Trains and evaluates each requested condition.
- Saves per-condition results under `results/<model>/<condition>/`.
- Writes per-model charts and cross-model comparison outputs.

### `dqb compare`

Rebuilds comparison outputs from previously saved records.

```bash
uv run dqb compare
```

Useful options:

```bash
uv run dqb compare --manifest
uv run dqb compare --results-root results
uv run dqb compare --comparison-root comparison
```

What it does:

- Loads saved `record.json` files from the results directory.
- Optionally rewrites `manifest.csv` when `--manifest` is provided.
- Regenerates per-model charts from the loaded records.
- Regenerates cross-model comparison charts in the comparison directory.

### `dqb generate_graphs`

Generates training curves from saved result histories without retraining.

```bash
uv run dqb generate_graphs
```

Useful options:

```bash
uv run dqb generate_graphs --results-root results
```

What it does:

- Scans all `results/<model>/<condition>/history.csv` files.
- Recreates `plots/` directories in each condition folder.
- Writes metric and loss curves for each history file.
- For dendritic runs, also generates architecture evolution plots from `best_arch_scores.csv`.

## Output Layout

After a run, the repository writes:

```text
results/
  <model>/
    <condition>/
      record.json
      record.csv
      metrics.json
      history.csv
      model.pt
      best_model              # dendritic runs only
      final_clean_pai         # dendritic runs only
      best_arch_scores.csv    # dendritic runs only
      paramCounts.csv         # dendritic runs only
      plots/
        training_curve.svg
        primary_metric.svg
        loss_curves.svg
        architecture_evolution.svg  # dendritic runs only
comparison/
  accuracy_retention_heatmap.svg
  size_tradeoff_scatter.svg
  dendrite_delta.svg
  best_quantization_heatmap.svg
  summary.csv
```

## Model and Condition Keys

Available model keys:

- `lenet5`
- `m5`
- `lstm_forecaster`
- `textcnn`
- `gcn`
- `tabnet`
- `mpnn`
- `actor_critic`
- `lstm_autoencoder`
- `distilbert`
- `dqn_lunarlander`
- `ppo_bipedalwalker`
- `attentivefp_freesolv`
- `gin_imdbb`
- `tcn_forecaster`
- `gru_forecaster`
- `pointnet_modelnet40`
- `vae_mnist`
- `snn_nmnist`
- `unet_isic`
- `resnet18_cifar10`
- `mobilenetv2_cifar10`
- `saint_adult`
- `capsnet_mnist`
- `convlstm_movingmnist`

Available condition keys:

- `base_fp32`
- `base_q8`
- `base_q4`
- `base_q2`
- `base_q1_58`
- `base_q1`
- `dendrites_fp32`
- `dendrites_pruned`
- `dendrites_pruned_q8`
- `dendrites_pruned_q4`
- `dendrites_pruned_q2`
- `dendrites_pruned_q1_58`
- `dendrites_pruned_q1`

## Common Workflows

### Run a small smoke test

```bash
uv run dqb run --models lenet5 --conditions base_fp32 base_q8
```

### Regenerate plots from existing results

```bash
uv run dqb compare --manifest
```

## Notes

- Real datasets are downloaded automatically into `data/` by default. Set `DQB_DATA_ROOT` to keep the cache outside the repository.
- The CLI accepts the same `--results-root` and `--comparison-root` options across most commands.
- Dendritic runs are expected to produce additional PerforatedAI sidecar artifacts.
- If you add new models or conditions, update `src/dendritic_benchmark/specs.py` and rerun the CLI.
