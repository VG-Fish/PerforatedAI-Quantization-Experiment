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

The loader also recognizes `PERFORATEDAI_API_KEY`, `PERFORATEDAI_TOKEN`, `PERFORATEDBP_API_KEY`, and `PERFORATEDBP_TOKEN` as aliases for the token, and `PERFORATEDAI_EMAIL` / `PERFORATEDBP_EMAIL` as aliases for the email.

## Commands

### `dqb run`

Runs the benchmark pipeline for all configured models and conditions.

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

- Downloads and caches the real benchmark datasets on first use.
- Builds task bundles for each model from those cached datasets.
- Trains/evaluates each selected condition.
- Saves one result folder per model and condition.
- Writes per-model charts and cross-model comparison charts.

### `dqb compare`

Rebuilds comparison outputs from previously saved records without re-running training.

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

- Loads all saved `record.json` files under the results directory.
- Rewrites the manifest CSV if requested.
- Regenerates per-model plots in each `results/<model>/` directory.
- Regenerates the cross-model plots in the comparison directory.

### `dqb generate_graphs`

Generates training loss/metric curves from saved training data without re-running training.

```bash
uv run dqb generate_graphs
```

Useful options:

```bash
uv run dqb generate_graphs --results-root results
```

What it does:

- Scans all `results/<model>/<condition>/` directories for `history.csv` files.
- Creates a `plots/` subdirectory in each model-condition folder.
- Generates training curves showing validation metrics over epochs.
- For dendritic models, also generates architecture evolution plots from `best_arch_scores.csv`.
- Saves SVG plots to `results/<model>/<condition>/plots/`.

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
      best_arch_scores.csv     # dendritic runs only
      paramCounts.csv          # dendritic runs only
      plots/
        training_curve.svg
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

### Inspect saved records

Look at the JSON and CSV files in each `results/<model>/<condition>/` folder. They contain the metric value, parameter count, file size, and the artifact path used for that run.

## Notes

- Real datasets are downloaded automatically into `data/` by default. Set `DQB_DATA_ROOT` to keep the cache outside the repository.
- First runs need network access for the selected datasets; later runs reuse the local cache.
- Dendritic runs are expected to produce extra sidecar files.
- If you add new models or conditions later, update `src/dendritic_benchmark/specs.py` and rerun the CLI.
- If your beta credentials unlock additional PerforatedAI capabilities, the code will pick them up through the loaded environment automatically.
