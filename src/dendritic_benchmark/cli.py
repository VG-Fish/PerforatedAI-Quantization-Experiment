from __future__ import annotations

import argparse
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .benchmark import BenchmarkOrchestrator
from .compat import load_project_environment, perforatedai_credentials_present
from .data import DATA_ROOT_ENV, DEFAULT_DATA_ROOT, build_task_bundle, dataset_exists
from .log_utils import setup_logging
from .pipeline import BenchmarkRunner
from .results import load_training_records, write_comparison_reports, write_manifest, write_model_reports, generate_training_graphs
from .specs import MODEL_SPECS


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


_MODEL_KEYS: str = (
    "lenet5, m5, lstm_forecaster, textcnn, gcn, tabnet, mpnn, actor_critic, "
    "lstm_autoencoder, distilbert, dqn_lunarlander, ppo_bipedalwalker, "
    "attentivefp_freesolv, gin_imdbb, tcn_forecaster, gru_forecaster, "
    "pointnet_modelnet40, vae_mnist, snn_nmnist, unet_isic, resnet18_cifar10, "
    "mobilenetv2_cifar10, saint_adult, capsnet_mnist, convlstm_movingmnist"
)

_CONDITION_KEYS: str = (
    "base_fp32, base_q8, base_q4, base_q2, base_q1_58, base_q1, "
    "dendrites_fp32, dendrites_q8, dendrites_q4, dendrites_q2, "
    "dendrites_q1_58, dendrites_q1"
)


def build_parser() -> argparse.ArgumentParser:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description=(
            "Dendritic quantization benchmark runner.\n\n"
            "Trains 25 models across 12 conditions that isolate two factors: "
            "PerforatedAI dendritic augmentation and post-training quantization "
            "(INT8 down to binary). Results are saved under --results-root "
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
        "--results-directory",
        default=None,
        metavar="NAME",
        help=(
            "Optional directory name under --results-root to scope a training run "
            "or analysis set. When provided, commands read/write results under "
            "<results-root>/<results-directory>."
        ),
    )
    parser.add_argument(
        "--logging-dir",
        default="logs",
        metavar="DIR",
        help=(
            "Directory where timestamped log files are written. Each invocation "
            "creates a new file named <command>_YYYYMMDD_HHMMSS.txt. "
            "All stdout and stderr are teed to this file. (default: logs)"
        ),
    )
    subparsers: argparse._SubParsersAction = parser.add_subparsers(dest="command", required=True)

    run_parser: argparse.ArgumentParser = subparsers.add_parser(
        "run",
        help="Train models across all (or a subset of) conditions and save results.",
        description=(
            "Runs the full benchmark pipeline: trains each selected model under each "
            "selected condition, evaluates it, and writes JSON records plus plots to "
            "--results-root.\n\n"
            "Conditions are executed in dependency order — e.g. dendrites_q8 "
            "requires dendrites_fp32 — so "
            "omitting an upstream condition will cause its dependents to be skipped.\n\n"
            f"Available model keys:\n  {_MODEL_KEYS}\n\n"
            f"Available condition keys:\n  {_CONDITION_KEYS}"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    run_parser.add_argument(
        "--comparison-root",
        default="comparison",
        metavar="DIR",
        help=(
            "Root directory for cross-model comparison outputs such as summary "
            "CSVs and aggregate plots. (default: comparison)"
        ),
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
            "Omit to run all 12 conditions. "
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
    run_parser.add_argument(
        "--allow-PQAT",
        dest="allow_pqat",
        action="store_true",
        help=(
            "Enable post-quantization-aware training for quantized conditions. "
            "When set, quantized runs save a PTQ evaluation to "
            "`before_pqat/`, fine-tune for a model-specific PQAT epoch budget, "
            "and save the post-PQAT artifacts to `after_pqat/`."
        ),
    )

    download_parser: argparse.ArgumentParser = subparsers.add_parser(
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

    compare_parser: argparse.ArgumentParser = subparsers.add_parser(
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
        "--comparison-root",
        default="comparison",
        metavar="DIR",
        help=(
            "Root directory for cross-model comparison outputs such as summary "
            "CSVs and aggregate plots. (default: comparison)"
        ),
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

    generate_graphs_parser: argparse.ArgumentParser = subparsers.add_parser(
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

    bench_parser: argparse.ArgumentParser = subparsers.add_parser(
        "bench",
        help="Benchmark inference latency of trained models using torch.utils.benchmark.Timer.",
        description=(
            "Measures wall-clock inference latency for all trained models and conditions. "
            "Benchmarks each model using batch sizes 1 and 32 (configurable). Results are "
            "saved to --benchmark-root with per-model subdirectories containing latency "
            "measurements and a computer_info.json file with system specifications.\n\n"
            "Requires trained models to exist in --results-root.\n\n"
            f"Available model keys:\n  {_MODEL_KEYS}\n\n"
            f"Available condition keys:\n  {_CONDITION_KEYS}"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    bench_parser.add_argument(
        "--benchmark-root",
        default="benchmarks",
        metavar="DIR",
        help=(
            "Root directory where benchmark results are written. "
            "Created if it does not exist. (default: benchmarks)"
        ),
    )
    bench_parser.add_argument(
        "--models",
        nargs="*",
        metavar="KEY",
        help=(
            "Space-separated list of model keys to benchmark. "
            "Omit to benchmark all 25 models. "
            f"Valid keys: {_MODEL_KEYS}"
        ),
    )
    bench_parser.add_argument(
        "--conditions",
        nargs="*",
        metavar="KEY",
        help=(
            "Space-separated list of condition keys to benchmark. "
            "Omit to benchmark all 12 conditions. "
            f"Valid keys: {_CONDITION_KEYS}"
        ),
    )
    bench_parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="*",
        default=[1, 32],
        metavar="SIZE",
        help=(
            "Space-separated list of batch sizes to test. "
            "(default: 1 32)"
        ),
    )
    bench_parser.add_argument(
        "--num-runs",
        type=int,
        default=10,
        metavar="N",
        help=(
            "Number of timing runs per benchmark to average. "
            "(default: 10)"
        ),
    )

    return parser


def _handle_run(args: Any, results_root: Path, comparison_root: Path) -> None:
    runner = BenchmarkRunner(results_root=results_root, comparison_root=comparison_root)
    runner.run(
        model_keys=args.models,
        condition_keys=args.conditions,
        ignore_saved=args.ignore_saved_models,
        allow_pqat=args.allow_pqat,
    )


def _handle_download_data(args: Any) -> None:
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


def _handle_compare(args: Any, results_root: Path, comparison_root: Path) -> None:
    records = load_training_records(results_root)
    if args.manifest:
        write_manifest(records, results_root / "manifest.csv")
    for model_spec in MODEL_SPECS:
        model_records = [record for record in records if record["model_key"] == model_spec.key]
        if model_records:
            write_model_reports(model_spec.display_name, model_records, results_root / model_spec.key)
    write_comparison_reports(records, comparison_root)


def _handle_bench(args: Any, results_root: Path, benchmark_root: Path) -> None:
    orchestrator = BenchmarkOrchestrator(results_root=results_root)
    orchestrator.benchmark_all(
        model_keys=args.models,
        condition_keys=args.conditions,
        batch_sizes=args.batch_sizes,
        num_runs=args.num_runs,
        benchmark_root=benchmark_root,
    )


def main() -> None:
    load_project_environment()
    parser = build_parser()
    args = parser.parse_args()
    results_root = Path(args.results_root)
    if args.results_directory:
        results_root = results_root / args.results_directory
    comparison_root = Path(getattr(args, "comparison_root", "comparison"))
    benchmark_root = Path(getattr(args, "benchmark_root", "benchmarks"))

    setup_logging(output_dir=args.logging_dir, script_name=args.command)

    if perforatedai_credentials_present():
        _log("PerforatedAI credentials detected in environment; beta-capable features can be used if installed.")

    if args.command == "run":
        _handle_run(args, results_root, comparison_root)
    elif args.command == "download_data":
        _handle_download_data(args)
    elif args.command == "compare":
        _handle_compare(args, results_root, comparison_root)
    elif args.command == "generate_graphs":
        generate_training_graphs(results_root, regenerate=args.regenerate_graphs)
    elif args.command == "bench":
        _handle_bench(args, results_root, benchmark_root)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
