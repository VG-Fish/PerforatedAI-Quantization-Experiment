from __future__ import annotations

import argparse
import os
import time
from datetime import datetime
from pathlib import Path

from .compat import load_project_environment, perforatedai_credentials_present
from .data import DATA_ROOT_ENV, DEFAULT_DATA_ROOT, build_task_bundle, dataset_exists
from .pipeline import BenchmarkRunner
from .results import load_training_records, write_comparison_reports, write_manifest, write_model_reports, generate_training_graphs
from .specs import MODEL_SPECS


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


_MODEL_KEYS = (
    "lenet5, m5, lstm_forecaster, textcnn, gcn, tabnet, mpnn, actor_critic, "
    "lstm_autoencoder, distilbert, dqn_lunarlander, ppo_bipedalwalker, "
    "attentivefp_freesolv, gin_imdbb, tcn_forecaster, gru_forecaster, "
    "pointnet_modelnet40, vae_mnist, snn_nmnist, unet_isic, resnet18_cifar10, "
    "mobilenetv2_cifar10, saint_adult, capsnet_mnist, convlstm_movingmnist"
)

_CONDITION_KEYS = (
    "base_fp32, base_q8, base_q4, base_q2, base_q1_58, base_q1, "
    "dendrites_fp32, dendrites_pruned, dendrites_pruned_q8, dendrites_pruned_q4, "
    "dendrites_pruned_q2, dendrites_pruned_q1_58, dendrites_pruned_q1"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Dendritic quantization benchmark runner.\n\n"
            "Trains 25 models across 13 conditions that combine baseline FP32 training, "
            "PerforatedAI dendritic augmentation, magnitude pruning, and post-training "
            "quantization (INT8 down to binary). Results are saved under --results-root "
            "and cross-model comparisons under --comparison-root."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results-root",
        default="results",
        metavar="DIR",
        help=(
            "Root directory where per-model result folders are written. "
            "Each model gets a subdirectory named by its key containing JSON "
            "training records and PNG plots. (default: results)"
        ),
    )
    parser.add_argument(
        "--comparison-root",
        default="comparison",
        metavar="DIR",
        help=(
            "Root directory for cross-model comparison outputs such as summary "
            "CSVs and aggregate plots. (default: comparison)"
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Train models across all (or a subset of) conditions and save results.",
        description=(
            "Runs the full benchmark pipeline: trains each selected model under each "
            "selected condition, evaluates it, and writes JSON records plus plots to "
            "--results-root.\n\n"
            "Conditions are executed in dependency order — e.g. dendrites_pruned_q8 "
            "requires dendrites_pruned, which in turn requires dendrites_fp32 — so "
            "omitting an upstream condition will cause its dependents to be skipped.\n\n"
            f"Available model keys:\n  {_MODEL_KEYS}\n\n"
            f"Available condition keys:\n  {_CONDITION_KEYS}"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    run_parser.add_argument(
        "--models",
        nargs="*",
        metavar="KEY",
        help=(
            "Space-separated list of model keys to include. "
            "Omit to run all 25 models. "
            f"Valid keys: {_MODEL_KEYS}"
        ),
    )
    run_parser.add_argument(
        "--conditions",
        nargs="*",
        metavar="KEY",
        help=(
            "Space-separated list of condition keys to include. "
            "Omit to run all 13 conditions. "
            f"Valid keys: {_CONDITION_KEYS}"
        ),
    )
    run_parser.add_argument(
        "--ignore-saved-models",
        action="store_true",
        help=(
            "Redo training for all selected model/condition pairs even if a "
            "record.json already exists on disk. By default, existing records "
            "are loaded and that combination is skipped."
        ),
    )

    download_parser = subparsers.add_parser(
        "download_data",
        help="Pre-download and cache datasets so that 'run' works offline.",
        description=(
            "Downloads and caches all datasets required by the selected models. "
            "Datasets are stored under the directory given by the DQB_DATA_ROOT "
            "environment variable (default: ./data). Already-cached datasets are "
            "skipped automatically.\n\n"
            "Run this before 'run' if you want to separate the download step, work "
            "offline, or verify that all data sources are reachable.\n\n"
            f"Available model keys:\n  {_MODEL_KEYS}"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    download_parser.add_argument(
        "--models",
        nargs="*",
        metavar="KEY",
        help=(
            "Space-separated list of model keys whose datasets should be downloaded. "
            "Omit to download datasets for all 25 models. "
            f"Valid keys: {_MODEL_KEYS}"
        ),
    )
    download_parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Abort immediately on the first download failure instead of continuing "
            "and reporting all failures at the end."
        ),
    )

    compare_parser = subparsers.add_parser(  # noqa: F841
        "compare",
        help="Generate cross-model comparison reports and plots from saved training records.",
        description=(
            "Reads all JSON training records from --results-root and produces:\n"
            "  • Per-model summary CSVs and bar charts (written to each model's subfolder)\n"
            "  • Aggregate cross-model comparison plots and CSVs (written to --comparison-root)\n\n"
            "Run this after 'run' completes (or partially completes) to visualise results "
            "without re-training. Safe to re-run multiple times."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    compare_parser.add_argument(
        "--manifest",
        action="store_true",
        help=(
            "Rewrite the manifest CSV (results/manifest.csv) from the current training "
            "records before generating plots. Useful after adding new results or manually "
            "editing record files."
        ),
    )

    generate_graphs_parser = subparsers.add_parser(
        "generate_graphs",
        help="Render per-epoch training-curve plots from saved results.",
        description=(
            "Walks --results-root and renders a training-curve PNG for every saved "
            "training record that does not yet have one. Useful for regenerating plots "
            "after a style change or if graphs were accidentally deleted.\n\n"
            "By default, existing graph files are left untouched."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    generate_graphs_parser.add_argument(
        "--regenerate-graphs",
        action="store_true",
        help=(
            "Overwrite graph files that already exist on disk. "
            "Without this flag only missing graphs are created."
        ),
    )

    return parser


def main() -> None:
    load_project_environment()
    parser = build_parser()
    args = parser.parse_args()
    results_root = Path(args.results_root)
    comparison_root = Path(args.comparison_root)

    if perforatedai_credentials_present():
        _log("PerforatedAI credentials detected in environment; beta-capable features can be used if installed.")

    if args.command == "run":
        runner = BenchmarkRunner(results_root=results_root, comparison_root=comparison_root)
        runner.run(
            model_keys=args.models,
            condition_keys=args.conditions,
            ignore_saved=args.ignore_saved_models,
        )
        return

    if args.command == "download_data":
        selected = args.models or [spec.key for spec in MODEL_SPECS]
        total = len(selected)
        data_root = Path(os.environ.get(DATA_ROOT_ENV, DEFAULT_DATA_ROOT)).resolve()
        _log(f"[data] root        : {data_root}")
        _log(f"[data] models      : {total}")
        print(f"{'─' * 60}")
        skipped = 0
        downloaded = 0
        failures: list[tuple[str, str]] = []
        for i, model_key in enumerate(selected, 1):
            prefix = f"[{i}/{total}] {model_key}"
            if dataset_exists(model_key):
                _log(f"{prefix} — already cached, skipping.")
                skipped += 1
                continue
            _log(f"{prefix} — downloading / preparing…")
            t0 = time.monotonic()
            try:
                build_task_bundle(model_key)
                elapsed = time.monotonic() - t0
                _log(f"{prefix} — done ({elapsed:.1f}s).")
                downloaded += 1
            except Exception as exc:
                elapsed = time.monotonic() - t0
                if args.strict:
                    raise
                failures.append((model_key, str(exc)))
                _log(f"{prefix} — FAILED after {elapsed:.1f}s: {exc}")
        print(f"{'─' * 60}")
        _log(
            f"[data] finished — {downloaded} downloaded, "
            f"{skipped} already cached, {len(failures)} failed."
        )
        if failures:
            _log("[data] failures:")
            for model_key, message in failures:
                print(f"  - {model_key}: {message}")
        return

    if args.command == "compare":
        records = load_training_records(results_root)
        if args.manifest:
            write_manifest(records, results_root / "manifest.csv")
        for model_spec in MODEL_SPECS:
            model_records = [record for record in records if record["model_key"] == model_spec.key]
            if model_records:
                write_model_reports(model_spec.display_name, model_records, results_root / model_spec.key)
        write_comparison_reports(records, comparison_root)
        return

    if args.command == "generate_graphs":
        generate_training_graphs(results_root, regenerate=args.regenerate_graphs)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
