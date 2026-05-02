from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from .benchmark import BenchmarkOrchestrator
from .compat import load_project_environment, perforatedai_credentials_present
from .data import DATA_ROOT_ENV, DEFAULT_DATA_ROOT, build_task_bundle, dataset_exists
from .log_utils import setup_logging
from .pipeline import BenchmarkRunner
from .results import (
    generate_training_graphs,
    load_training_records,
    write_comparison_reports,
    write_manifest,
    write_model_reports,
    write_per_model_benchmark_plots,
)
from .specs import MODEL_SPECS

argcomplete: Optional[Any] = None
try:
    import argcomplete
except Exception:  # pragma: no cover - optional runtime enhancement
    pass


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

_CLEAN_CONFIG_PATH = Path(".dqb") / "command_config.json"
_CLEAN_CONFIG_VERSION = 1


def _path_entry(path: Path | str, kind: str) -> dict[str, str]:
    resolved = Path(path).expanduser().resolve()
    return {"path": str(resolved), "kind": kind}


def _load_clean_config() -> dict[str, Any]:
    if not _CLEAN_CONFIG_PATH.exists():
        return {"version": _CLEAN_CONFIG_VERSION, "invocations": []}
    try:
        with _CLEAN_CONFIG_PATH.open("r", encoding="utf-8") as fh:
            config = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        raise RuntimeError(f"Could not read clean config at {_CLEAN_CONFIG_PATH}: {exc}") from exc
    if not isinstance(config, dict):
        return {"version": _CLEAN_CONFIG_VERSION, "invocations": []}
    invocations = config.get("invocations")
    if not isinstance(invocations, list):
        config["invocations"] = []
    config["version"] = _CLEAN_CONFIG_VERSION
    return config


def _write_clean_config(config: dict[str, Any]) -> None:
    _CLEAN_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _CLEAN_CONFIG_PATH.open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2, sort_keys=True)
        fh.write("\n")


def _record_clean_config(args: Any, results_root: Path, comparison_root: Path, benchmark_root: Path) -> None:
    if args.command == "clean":
        return
    paths: list[dict[str, str]] = [_path_entry(args.logging_dir, "logs")]
    if args.command == "run":
        paths.extend(
            [
                _path_entry(results_root, "results"),
                _path_entry(comparison_root, "comparison"),
                _path_entry("PAI", "perforatedai"),
                _path_entry(Path(os.environ.get(DATA_ROOT_ENV, DEFAULT_DATA_ROOT)), "data"),
            ]
        )
    elif args.command == "download_data":
        paths.append(_path_entry(Path(os.environ.get(DATA_ROOT_ENV, DEFAULT_DATA_ROOT)), "data"))
    elif args.command == "compare":
        paths.extend(
            [
                _path_entry(results_root, "results_reports"),
                _path_entry(comparison_root, "comparison"),
            ]
        )
    elif args.command == "generate_graphs":
        paths.append(_path_entry(results_root, "results_graphs"))
    elif args.command == "benchmark_models":
        paths.extend(
            [
                _path_entry(benchmark_root, "benchmarks"),
                _path_entry(comparison_root, "comparison"),
            ]
        )
    config = _load_clean_config()
    config["invocations"].append(
        {
            "command": args.command,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "results_root": str(results_root.expanduser().resolve()),
            "results_directory": args.results_directory,
            "logging_dir": str(Path(args.logging_dir).expanduser().resolve()),
            "comparison_root": str(comparison_root.expanduser().resolve()),
            "benchmark_root": str(benchmark_root.expanduser().resolve()),
            "data_root": str(Path(os.environ.get(DATA_ROOT_ENV, DEFAULT_DATA_ROOT)).expanduser().resolve()),
            "paths": paths,
        }
    )
    _write_clean_config(config)


def _iter_recorded_clean_paths(config: dict[str, Any]) -> list[dict[str, str]]:
    by_path: dict[str, str] = {}
    for invocation in config.get("invocations", []):
        if not isinstance(invocation, dict):
            continue
        for entry in invocation.get("paths", []):
            if not isinstance(entry, dict):
                continue
            raw_path = entry.get("path")
            if not isinstance(raw_path, str):
                continue
            path = str(Path(raw_path).expanduser().resolve())
            by_path[path] = str(entry.get("kind", "generated"))
    return [{"path": path, "kind": kind} for path, kind in sorted(by_path.items())]


def _is_dangerous_clean_target(path: Path) -> bool:
    resolved = path.expanduser().resolve()
    cwd = Path.cwd().resolve()
    home = Path.home().resolve()
    anchors = {resolved.anchor, str(cwd), str(home)}
    return str(resolved) in anchors or resolved == _CLEAN_CONFIG_PATH.parent.resolve()


def _remove_clean_target(path: Path) -> str:
    if path.is_symlink():
        path.unlink()
        return "symlink"
    if path.is_dir():
        shutil.rmtree(path)
        return "directory"
    if path.exists():
        path.unlink()
        return "file"
    return "missing"


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
            "Enable post-quantization-aware training for all quantized "
            "conditions. When set, quantized runs save a PTQ evaluation to "
            "`before_pqat/`, fine-tune for a model-specific PQAT epoch budget, "
            "and save the post-PQAT artifacts to `after_pqat/`."
        ),
    )
    run_parser.add_argument(
        "--dynamic-dendritic-training",
        action="store_true",
        help=(
            "Use PerforatedAI's open-ended dynamic FP32 dendritic training mode. "
            "By default, dendritic FP32 runs use the same fixed epoch budget as "
            "the matching non-dendritic run and freeze dendrite insertion for "
            "the final 20%% of epochs."
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
    compare_parser.add_argument(
        "--benchmark-root",
        default="benchmarks",
        metavar="DIR",
        help=(
            "Root directory containing benchmark outputs (manifest.csv) used to build "
            "latency comparison plots in --comparison-root. (default: benchmarks)"
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
        "benchmark_models",
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
        "--comparison-root",
        default="comparison",
        metavar="DIR",
        help=(
            "Root directory for per-model comparison plots. "
            "A subdirectory is created for each model. (default: comparison)"
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
        default=5,
        metavar="N",
        help=(
            "Number of independent timing runs per batch size. Mean and median "
            "are computed across these runs. (default: 5)"
        ),
    )
    bench_parser.add_argument(
        "--re-run",
        action="store_true",
        help=(
            "Re-benchmark all selected model/condition pairs even if a result "
            "JSON already exists on disk. By default, existing results are loaded "
            "and that combination is skipped."
        ),
    )

    clean_parser: argparse.ArgumentParser = subparsers.add_parser(
        "clean",
        help="Remove files generated by previous dqb commands.",
        description=(
            "Reads .dqb/command_config.json and removes the output directories "
            "and files recorded by previous 'uv run dqb' commands. The registry "
            "tracks user-supplied locations such as --results-root, "
            "--results-directory, --comparison-root, --benchmark-root, "
            "--logging-dir, and DQB_DATA_ROOT."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    clean_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the paths that would be removed without deleting anything.",
    )

    return parser


def _handle_run(args: Any, results_root: Path, comparison_root: Path) -> None:
    runner = BenchmarkRunner(results_root=results_root, comparison_root=comparison_root)
    runner.run(
        model_keys=args.models,
        condition_keys=args.conditions,
        ignore_saved=args.ignore_saved_models,
        allow_pqat=args.allow_pqat,
        dynamic_dendritic_training=args.dynamic_dendritic_training,
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
    write_per_model_benchmark_plots(Path(getattr(args, "benchmark_root", "benchmarks")), comparison_root)


def _handle_bench(args: Any, results_root: Path, benchmark_root: Path, comparison_root: Path) -> None:
    orchestrator = BenchmarkOrchestrator(results_root=results_root)
    orchestrator.benchmark_all(
        model_keys=args.models,
        condition_keys=args.conditions,
        batch_sizes=args.batch_sizes,
        num_runs=args.num_runs,
        benchmark_root=benchmark_root,
        comparison_root=comparison_root,
        re_run=args.re_run,
    )


def _handle_clean(args: Any) -> None:
    config = _load_clean_config()
    entries = _iter_recorded_clean_paths(config)
    if not entries:
        _log(f"[clean] No generated paths recorded in {_CLEAN_CONFIG_PATH}.")
        return
    removed = 0
    skipped = 0
    for entry in entries:
        path = Path(entry["path"])
        label = entry["kind"]
        if _is_dangerous_clean_target(path):
            _log(f"[clean] Skipping unsafe {label} target: {path}")
            skipped += 1
            continue
        if args.dry_run:
            status = "exists" if path.exists() else "missing"
            _log(f"[clean] Would remove {label} target ({status}): {path}")
            continue
        status = _remove_clean_target(path)
        if status == "missing":
            _log(f"[clean] Already missing {label} target: {path}")
            skipped += 1
        else:
            _log(f"[clean] Removed {status} {label} target: {path}")
            removed += 1
    if args.dry_run:
        _log(f"[clean] Dry run complete. {len(entries)} recorded target(s) inspected.")
        return
    config["invocations"] = []
    _write_clean_config(config)
    _log(f"[clean] Complete. Removed {removed} target(s), skipped {skipped}.")


def main() -> None:
    load_project_environment()
    parser = build_parser()
    if argcomplete is not None:
        argcomplete.autocomplete(parser)
    args = parser.parse_args()
    results_root = Path(args.results_root)
    if args.results_directory:
        results_root = results_root / args.results_directory
    comparison_root = Path(getattr(args, "comparison_root", "comparison"))
    benchmark_root = Path(getattr(args, "benchmark_root", "benchmarks"))

    _record_clean_config(args, results_root, comparison_root, benchmark_root)
    if args.command != "clean":
        setup_logging(output_dir=args.logging_dir, script_name=args.command)

    if args.command != "clean" and perforatedai_credentials_present():
        _log("PerforatedAI credentials detected in environment; beta-capable features can be used if installed.")

    if args.command == "run":
        _handle_run(args, results_root, comparison_root)
    elif args.command == "download_data":
        _handle_download_data(args)
    elif args.command == "compare":
        _handle_compare(args, results_root, comparison_root)
    elif args.command == "generate_graphs":
        generate_training_graphs(results_root, regenerate=args.regenerate_graphs)
    elif args.command == "benchmark_models":
        _handle_bench(args, results_root, benchmark_root, comparison_root)
    elif args.command == "clean":
        _handle_clean(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
