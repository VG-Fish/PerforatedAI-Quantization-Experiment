import argparse
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from .benchmark import BenchmarkOrchestrator
from .compat import (
    PAI_DIRECTORY_NAME,
    load_project_environment,
    perforatedai_credentials_present,
    set_pai_root,
)
from .data import DATA_ROOT_ENV, DEFAULT_DATA_ROOT, build_task_bundle, dataset_exists
from .docs import CURRENT_GUIDE_PATH, current_guide_is_current, write_current_guide
from .evidence import (
    EVIDENCE_INDEX_JSON,
    EVIDENCE_INDEX_MARKDOWN,
    build_evidence_index,
    index_roots,
    write_evidence_index,
)
from .log_utils import setup_logging, validate_output_path
from .model_adapters import ALL_MODEL_KEYS, selected_model_keys
from .pipeline import (
    DEFAULT_JOBS,
    DEFAULT_PROGRESS_INTERVAL,
    BenchmarkRunner,
    clear_epoch_checkpoints,
    run_parallel,
)
from .plans import PAIOverride, RecipeOverride
from .results import (
    generate_training_graphs,
    load_training_records,
    write_comparison_reports,
    write_manifest,
    write_model_reports,
    write_per_model_benchmark_plots,
)
from .specs import CONDITION_SPECS, MODEL_SPECS
from .statistics import MINIMUM_PAIRED_SEEDS

try:
    import argcomplete as _argcomplete
except Exception:  # pragma: no cover - optional runtime enhancement
    argcomplete = None
else:
    argcomplete: Any | None = _argcomplete


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


# Help text is rendered from the registries so that adding a model or a
# condition cannot leave the CLI describing a roster that no longer exists.
_MODEL_KEYS: str = ", ".join(ALL_MODEL_KEYS)

_CONDITION_KEYS: str = ", ".join(spec.key for spec in CONDITION_SPECS)

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
    # Workers are spawned by a 'run' that has already recorded these exact
    # paths. Letting each of them record too would add nothing and would have
    # several processes read-modify-writing one JSON file at the same moment,
    # which is how entries get lost.
    if getattr(args, "worker", False):
        return
    paths: list[dict[str, str]] = [_path_entry(args.logging_dir, "logs")]
    if args.command == "run":
        paths.extend(
            [
                _path_entry(results_root, "results"),
                _path_entry(comparison_root, "comparison"),
                _path_entry(results_root / PAI_DIRECTORY_NAME, "perforatedai"),
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
            "logging_dir": str(args.logging_dir),
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


def _validated_results_directory(value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(
            "--results-directory must be a relative directory name that stays under --results-root."
        )
    cleaned_parts = [part for part in path.parts if part not in ("", ".")]
    if not cleaned_parts:
        raise ValueError("--results-directory must not be empty.")
    return Path(*cleaned_parts)


def _normalize_output_args(args: Any) -> tuple[Path, Path, Path]:
    args.logging_dir = validate_output_path(Path(args.logging_dir), label="logging_dir")
    results_root = validate_output_path(Path(args.results_root), label="results_root")
    args.results_root = results_root
    results_directory = _validated_results_directory(args.results_directory)
    if results_directory is not None:
        args.results_directory = str(results_directory)
        results_root = validate_output_path(
            results_root / results_directory,
            label="results_root/results_directory",
        )

    comparison_root = validate_output_path(
        Path(getattr(args, "comparison_root", "comparison")),
        label="comparison_root",
    )
    if hasattr(args, "comparison_root"):
        args.comparison_root = comparison_root
    benchmark_root = validate_output_path(
        Path(getattr(args, "benchmark_root", "benchmarks")),
        label="benchmark_root",
    )
    if hasattr(args, "benchmark_root"):
        args.benchmark_root = benchmark_root
    return results_root, comparison_root, benchmark_root


def _add_common_options(parser: argparse.ArgumentParser, *, is_subcommand: bool) -> None:
    """Register the options that every command accepts.

    These are added twice: once on the top-level parser (owning the real
    defaults) and once on every subparser, so they may be given either before
    or after the subcommand. The subcommand copies use ``SUPPRESS`` defaults so
    that an absent flag leaves the top-level value alone; when given in both
    positions the one after the subcommand wins.
    """
    position_note = " May be given before or after the subcommand."

    parser.add_argument(
        "--results-root",
        default=argparse.SUPPRESS if is_subcommand else "results",
        metavar="DIR",
        help=(
            "Root directory where per-model result folders are written. "
            "Each model gets a subdirectory named by its key containing JSON "
            "training records and PNG plots. (default: results)" + position_note
        ),
    )
    parser.add_argument(
        "--results-directory",
        default=argparse.SUPPRESS if is_subcommand else None,
        metavar="NAME",
        help=(
            "Optional directory name under --results-root to scope a training run "
            "or analysis set. When provided, commands read/write results under "
            "<results-root>/<results-directory>." + position_note
        ),
    )
    parser.add_argument(
        "--logging-dir",
        default=argparse.SUPPRESS if is_subcommand else "logs",
        metavar="DIR",
        help=(
            "Directory where timestamped log files are written. Each invocation "
            "creates a new file named <command>_YYYYMMDD_HHMMSS.txt. "
            "All stdout and stderr are teed to this file. A parallel 'run' also "
            "writes <dir>/streams/stream_N.log and <dir>/run_progress.log here — "
            "if <dir>/streams already holds worker logs from an earlier launch, "
            "that launch goes to <dir>2, <dir>3, ... instead, so a new run never "
            "truncates a previous one's stream logs. --status auto-detects the "
            "most recent of these. (default: logs)" + position_note
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description=(
            "Dendritic quantization benchmark runner.\n\n"
            "Trains the registered models across 12 conditions that isolate two factors: "
            "PerforatedAI dendritic augmentation and post-training quantization "
            "(INT8 down to binary). Results are saved under --results-root "
            "and cross-model comparisons under --comparison-root.\n\n"
            "--results-root, --results-directory and --logging-dir are accepted "
            "either before or after the subcommand."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_common_options(parser, is_subcommand=False)

    common_options: argparse.ArgumentParser = argparse.ArgumentParser(add_help=False)
    _add_common_options(common_options, is_subcommand=True)

    subparsers: argparse._SubParsersAction = parser.add_subparsers(dest="command", required=True)

    run_parser: argparse.ArgumentParser = subparsers.add_parser(
        "run",
        parents=[common_options],
        help="Train models across all (or a subset of) conditions and save results.",
        description=(
            "Runs the full benchmark pipeline: trains each selected model under each "
            "selected condition, evaluates it, and writes JSON records plus plots to "
            "--results-root.\n\n"
            "Conditions are executed in dependency order — e.g. dendrites_q8 "
            "requires dendrites_fp32 — so "
            "omitting an upstream condition will cause its dependents to be skipped.\n\n"
            f"By default the selected models are split across {DEFAULT_JOBS} worker "
            "processes and a live progress table is printed until they all exit; "
            "training is compute-bound rather than data-bound, so this cuts "
            "wall-clock close to linearly. Each model keeps all of its conditions "
            "in one worker, so the dependency order above still holds. Worker "
            "output goes to <logging-dir>/streams/stream_N.log, and every progress "
            "table is appended to <logging-dir>/run_progress.log — a launch whose "
            "<logging-dir>/streams already has logs from an earlier run uses "
            "<logging-dir>2 (then 3, ...) instead, so it never truncates them. "
            "Use --jobs 1 to train in this process and print straight to the "
            "terminal instead.\n\n"
            "Ctrl-C stops every worker (each is SIGTERMed, then SIGKILLed after a "
            "10s grace period) and exits. Use --detach beforehand if you want "
            "training to keep running after your terminal disconnects.\n\n"
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
            "Omit to run the evidence-backed default roster; pass `all` to "
            "opt into every registered exploratory model. "
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
    run_parser.add_argument(
        "--pai-capacity-check",
        action="store_true",
        help=(
            "Run PAI's built-in seven-epoch dendrite-capacity diagnostic. "
            "This is restricted to the FP32 dendritic source condition, keeps "
            "PAI live until it reports completion, and records a diagnostic "
            "artifact that is not reusable as a production result."
        ),
    )
    run_parser.add_argument(
        "--model-scale",
        type=float,
        default=1.0,
        metavar="FACTOR",
        help=(
            "Scale the width of the compact-capable benchmark models before "
            "training. Use a value in (0, 1]; 0.75 is the recommended "
            "parameter-efficiency follow-up. (default: 1.0)"
        ),
    )
    run_parser.add_argument(
        "--pai-variant",
        choices=(
            "default",
            "distilbert_classifier_only",
            "gru_gate_ablation",
            "tcn_head_both",
            "tcn_head_output",
            "vae_latent",
        ),
        default="default",
        help=(
            "Select a measured PAI targeting ablation. `default` uses the "
            "current targeted settings; `tcn_head_output`/`tcn_head_both` "
            "screen alternate TCN head targets; `gru_gate_ablation` restores "
            "the prior gate search; `vae_latent` perforates VAE latent heads; "
            "`distilbert_classifier_only` perforates only DistilBERT's final "
            "classifier instead of the pre-classifier+classifier pair. "
            "(default: default)"
        ),
    )
    run_parser.add_argument(
        "--pai-fixed-switch-interval",
        type=int,
        default=None,
        metavar="EPOCHS",
        help=(
            "Diagnostic only: force PAI's fixed-switch mode at the requested "
            "interval for dendritic search. HISTORY plateau detection is the "
            "default for every model because prior fixed requests were not "
            "honored by observed switch epochs."
        ),
    )
    run_parser.add_argument(
        "--recipe-override",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Load a dendritic_benchmark.plans.RecipeOverride from this JSON "
            "file and apply it on top of the selected model's hard-coded "
            "ModelTrainingRecipe. Requires exactly one selected --models key: "
            "an override is one sweep trial for one model, per "
            "information/optimization/03_execution_matrix.md."
        ),
    )
    run_parser.add_argument(
        "--pai-override",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "Load a dendritic_benchmark.plans.PAIOverride from this JSON file "
            "to override the selected model's target modules or explicitly "
            "override PerforatedAI's library-owned schedule. Requires exactly "
            "one selected --models key, for the same reason as "
            "--recipe-override."
        ),
    )
    run_parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=DEFAULT_JOBS,
        metavar="N",
        help=(
            "Number of worker processes to split the selected models across. "
            "Never exceeds the number of models selected. 1 trains in this "
            f"process and prints straight to the terminal. (default: {DEFAULT_JOBS})"
        ),
    )
    run_parser.add_argument(
        "-i",
        "--interval",
        type=int,
        default=DEFAULT_PROGRESS_INTERVAL,
        metavar="SECONDS",
        help=(
            "Seconds between progress tables while waiting on the workers. "
            f"(default: {DEFAULT_PROGRESS_INTERVAL})"
        ),
    )
    run_parser.add_argument(
        "--fresh",
        action="store_true",
        help=(
            "Delete any leftover epoch_checkpoint.pt (and its condition directory) "
            "for the selected pairs before launching. --ignore-saved-models does "
            "not cover these: training resumes from an epoch checkpoint whenever "
            "one exists, so after a model definition change a stale checkpoint "
            "would silently continue an old-architecture run."
        ),
    )
    run_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Seed Python, NumPy and torch RNGs before every model/condition. "
            "Omit to leave them unseeded (the historical behaviour). Without a "
            "seed, run-to-run variance can exceed the effect being measured: "
            "gcn's base_fp32 moved 3.4pp between two runs of the identical "
            "command, against a dendrite effect of 0.30pp. Re-seeding per "
            "condition also makes a model's base_* and dendrites_* arms start "
            "from the same weights, so the two are a paired comparison. Vary "
            "this across runs to get the error bars that variance demands."
        ),
    )
    run_mode = run_parser.add_mutually_exclusive_group()
    run_mode.add_argument(
        "--detach",
        dest="mode",
        action="store_const",
        const="detach",
        default="watch",
        help=(
            "Launch the workers and exit immediately. Training continues, but the "
            "final manifest and comparison reports are skipped — run "
            "'dqb compare --manifest' by hand once every worker has exited."
        ),
    )
    run_mode.add_argument(
        "--status",
        dest="mode",
        action="store_const",
        const="status",
        help=(
            "Report on the workers recorded in --logging-dir without launching "
            "anything, then exit. Pointing this at an old run's log directory "
            "replays that run's final state."
        ),
    )
    run_parser.add_argument(
        "--worker",
        action="store_true",
        help=argparse.SUPPRESS,  # internal: set on the processes 'run' spawns
    )

    download_parser: argparse.ArgumentParser = subparsers.add_parser(
        "download_data",
        parents=[common_options],
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
            "Omit to download datasets for all registered models. "
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

    kd_parser: argparse.ArgumentParser = subparsers.add_parser(
        "pretrain_kd_teacher",
        parents=[common_options],
        help="Fine-tune the ResNet-50 distillation teacher that resnet18_kd_cifar100 needs.",
        description=(
            "Reproduces `train_perforated_resnet_KD.py --pre-train-teacher` from "
            "PerforatedAI's resnet example: an ImageNet ResNet-50 with a fresh "
            "100-way head, fine-tuned on the same CIFAR-100 train subset the "
            "student sees, checkpointed at its best validation accuracy.\n\n"
            "This is a one-time step outside the model x condition matrix. The "
            "student refuses to train without it, exactly as upstream refuses "
            "--use-kd on a non-ImageNet dataset with no teacher checkpoint."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    kd_parser.add_argument(
        "--epochs",
        type=int,
        default=90,
        help="Teacher fine-tuning epochs (upstream default: 90).",
    )
    kd_parser.add_argument(
        "--batch-size", type=int, default=32, help="Teacher batch size (upstream: 32)."
    )
    kd_parser.add_argument(
        "--lr",
        type=float,
        default=0.0125,
        help="Teacher SGD learning rate (upstream: 0.0125 with StepLR(30, 0.1)).",
    )
    kd_parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain and overwrite an existing teacher checkpoint.",
    )

    compare_parser: argparse.ArgumentParser = subparsers.add_parser(
        "compare",
        parents=[common_options],
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
    compare_parser.add_argument(
        "--models",
        nargs="*",
        metavar="KEY",
        help=(
            "Space-separated list of model keys to restrict the aggregate "
            "comparison plots (heatmaps, dendrite delta, summary.csv) to — "
            "e.g. so a heatmap over a handful of trained models doesn't carry "
            "blank rows for every model that was never run. Per-model reports "
            "are unaffected; they are always written for whichever models have "
            "records. Omit to include every model in the roster. "
            f"Valid keys: {_MODEL_KEYS}"
        ),
    )
    compare_parser.add_argument(
        "--seed-roots",
        nargs="+",
        metavar="DIR",
        help=(
            "Additional results roots holding the same models and conditions "
            "under different seeds. Their records join the seed-paired "
            "statistics in comparison/dendrite_effect_statistics.csv and the "
            "effect columns of dendrite_audit.csv; they do not change the "
            f"plots. A dendrite effect is only claimable with {MINIMUM_PAIRED_SEEDS} "
            "paired seeds, so without this the statistics stay at "
            "'insufficient_seeds' whenever each seed has its own root."
        ),
    )

    generate_graphs_parser: argparse.ArgumentParser = subparsers.add_parser(
        "generate_graphs",
        parents=[common_options],
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
        parents=[common_options],
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
            "Omit to benchmark all registered models. "
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
        parents=[common_options],
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

    docs_parser: argparse.ArgumentParser = subparsers.add_parser(
        "docs",
        help="Regenerate the current-state guide from the code registries.",
        description=(
            "Renders information/CURRENT_GUIDE.md from the model/condition specs, "
            "the artifact-validity rules, and this CLI's own option registry, so "
            "the current documentation cannot drift from the code that runs the "
            "experiment. Hand-written history under information/ is indexed by "
            "information/HISTORICAL_INDEX.md and is never regenerated."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    docs_parser.add_argument(
        "--output",
        type=Path,
        default=CURRENT_GUIDE_PATH,
        help=(
            "Path to write the generated guide to. "
            f"(default: {CURRENT_GUIDE_PATH})"
        ),
    )
    docs_parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "Do not write anything; exit non-zero when the checked-in guide no "
            "longer matches the registries. Intended for CI."
        ),
    )

    evidence_parser: argparse.ArgumentParser = subparsers.add_parser(
        "evidence_index",
        help="Index the generated result trees before archiving or deleting them.",
        description=(
            "Walks the generated roots, records every training record it finds "
            "with its run namespace, artifact id, seed, metric, and manifest "
            "verdict, and writes both a machine-readable JSON index and a "
            "markdown summary. Run this before removing any generated tree: the "
            "index is what survives the deletion."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    evidence_parser.add_argument(
        "--roots",
        nargs="+",
        default=None,
        help=(
            "Generated roots to index. (default: results, experiments, "
            "comparison, logs, logs_top10, logs_dynamic5, archive)"
        ),
    )
    evidence_parser.add_argument(
        "--verify",
        action="store_true",
        help=(
            "Re-hash every manifest-owned file instead of only reading the "
            "provenance files. Slow, and required before archiving or deleting "
            "a tree."
        ),
    )
    evidence_parser.add_argument(
        "--json-output",
        type=Path,
        default=EVIDENCE_INDEX_JSON,
        help=f"Path for the machine-readable index. (default: {EVIDENCE_INDEX_JSON})",
    )
    evidence_parser.add_argument(
        "--markdown-output",
        type=Path,
        default=EVIDENCE_INDEX_MARKDOWN,
        help=f"Path for the readable summary. (default: {EVIDENCE_INDEX_MARKDOWN})",
    )

    return parser


def _handle_run(args: Any, results_root: Path, comparison_root: Path) -> None:
    recipe_override = (
        RecipeOverride.from_json_file(args.recipe_override)
        if args.recipe_override is not None
        else None
    )
    pai_override = (
        PAIOverride.from_json_file(args.pai_override)
        if args.pai_override is not None
        else None
    )
    runner = BenchmarkRunner(
        results_root=results_root,
        comparison_root=comparison_root,
        model_scale=args.model_scale,
        pai_variant=args.pai_variant,
        pai_fixed_switch_interval=args.pai_fixed_switch_interval,
        recipe_override=recipe_override,
        pai_override=pai_override,
    )
    selected_models = selected_model_keys(args.models)
    if args.pai_capacity_check:
        # The runner expands this public source condition with its required
        # ``base_fp32`` dependency.  Validate the user-facing request here,
        # not that expanded execution list, otherwise the only valid capacity
        # invocation rejects itself before it can reuse the dense source.
        if args.conditions != ["dendrites_fp32"]:
            raise SystemExit(
                "--pai-capacity-check requires exactly "
                "--conditions dendrites_fp32; run production/PQAT conditions "
                "only after the diagnostic succeeds."
            )
        if args.allow_pqat:
            raise SystemExit("--pai-capacity-check cannot be combined with --allow-PQAT")
    if (recipe_override is not None or pai_override is not None) and len(
        selected_models
    ) != 1:
        # Catch this here, not only inside BenchmarkRunner.run(): a
        # multi-model --jobs>1 invocation never calls .run() on this
        # coordinator-process runner (it dispatches to run_parallel instead),
        # so each spawned worker would otherwise apply the same override file
        # to a different one of the several selected models.
        raise SystemExit(
            "--recipe-override/--pai-override require exactly one --models "
            f"key; got {len(selected_models)}: {', '.join(selected_models)}"
        )

    if (
        not args.worker
        and args.fresh
        and (args.jobs <= 1 or len(selected_models) == 1)
    ):
        def emit_fresh(text: str = "", *, stderr: bool = False) -> None:
            print(text, file=sys.stderr if stderr else sys.stdout)

        clear_epoch_checkpoints(
            results_root,
            selected_models,
            runner._expand_condition_keys(args.conditions),
            fresh=True,
            emit=emit_fresh,
        )

    # A worker is one of the processes a parallel run spawned. It trains its
    # slice in this process and leaves the manifest and comparison reports to
    # the coordinator, which builds them once from every worker's records.
    if args.worker:
        runner.run(
            model_keys=selected_models,
            condition_keys=args.conditions,
            ignore_saved=args.ignore_saved_models,
            allow_pqat=args.allow_pqat,
            dynamic_dendritic_training=args.dynamic_dendritic_training,
            pai_capacity_check=args.pai_capacity_check,
            write_reports=False,
            seed=args.seed,
        )
        return

    if args.mode == "watch" and (args.jobs <= 1 or len(selected_models) == 1):
        runner.run(
            model_keys=selected_models,
            condition_keys=args.conditions,
            ignore_saved=args.ignore_saved_models,
            allow_pqat=args.allow_pqat,
            dynamic_dendritic_training=args.dynamic_dendritic_training,
            pai_capacity_check=args.pai_capacity_check,
            seed=args.seed,
        )
        return

    run_parallel(
        results_root=results_root,
        comparison_root=comparison_root,
        model_keys=selected_models,
        condition_keys=args.conditions,
        expanded_condition_keys=runner._expand_condition_keys(args.conditions),
        log_root=args.logging_dir,
        passthrough=_run_passthrough(args, comparison_root),
        jobs=args.jobs,
        mode=args.mode,
        interval=args.interval,
        fresh=args.fresh,
    )


def _run_passthrough(args: Any, comparison_root: Path) -> list[str]:
    """The flags a worker needs to reproduce this invocation's paths and options.

    --models and --conditions are omitted: the coordinator supplies each worker
    with its own slice of the models, and passes the conditions separately so it
    can leave them unset (meaning "all") exactly as this invocation did.
    """
    flags = [
        "--results-root", str(args.results_root),
        "--logging-dir", str(args.logging_dir),
        "--comparison-root", str(comparison_root),
    ]
    if args.results_directory:
        flags += ["--results-directory", str(args.results_directory)]
    if args.ignore_saved_models:
        flags.append("--ignore-saved-models")
    if args.allow_pqat:
        flags.append("--allow-PQAT")
    if args.dynamic_dendritic_training:
        flags.append("--dynamic-dendritic-training")
    if args.pai_capacity_check:
        flags.append("--pai-capacity-check")
    if not math.isclose(args.model_scale, 1.0):
        flags += ["--model-scale", str(args.model_scale)]
    if args.pai_variant != "default":
        flags += ["--pai-variant", args.pai_variant]
    if args.recipe_override is not None:
        # In practice unreachable today: _handle_run rejects an override with
        # more than one selected model before run_parallel (and therefore
        # this function) is ever called. Kept so a worker still inherits the
        # override correctly if that guard is ever relaxed.
        flags += ["--recipe-override", str(args.recipe_override)]
    if args.pai_override is not None:
        flags += ["--pai-override", str(args.pai_override)]
    if args.pai_fixed_switch_interval is not None:
        flags += [
            "--pai-fixed-switch-interval",
            str(args.pai_fixed_switch_interval),
        ]
    if args.seed is not None:
        flags += ["--seed", str(args.seed)]
    return flags



def _handle_pretrain_kd_teacher(args: Any) -> None:
    """Fine-tune and checkpoint the CIFAR-100 ResNet-50 distillation teacher.

    Mirrors ``train_perforated_resnet_KD.pretrain_teacher``: train on the same
    stratified 25% subset the student uses, validate on the same validation
    half, and keep the best-validation checkpoint under the key ``"model"``
    that ``_kd_teacher`` reads back.
    """
    import torch
    from tqdm import tqdm

    from .data import build_task_bundle
    from .models import build_kd_teacher_resnet50, kd_teacher_checkpoint_path
    from .training import _resolve_device

    checkpoint = kd_teacher_checkpoint_path()
    if checkpoint.exists() and not args.force:
        _log(f"[kd] teacher already present at {checkpoint}; pass --force to retrain.")
        return

    device = _resolve_device("resnet18_kd_cifar100", torch)
    bundle = build_task_bundle("resnet18_kd_cifar100", batch_size=args.batch_size)
    teacher = build_kd_teacher_resnet50(num_classes=100).to(device)

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.SGD(
        teacher.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-3
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    _log(f"[kd] device        : {device}")
    _log(f"[kd] epochs        : {args.epochs}")
    _log(f"[kd] checkpoint    : {checkpoint}")

    best_accuracy = -1.0
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        teacher.train()
        for images, targets in tqdm(
            bundle.train_loader, desc=f"kd-teacher epoch {epoch}/{args.epochs}"
        ):
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            criterion(teacher(images), targets).backward()
            optimizer.step()
        scheduler.step()

        teacher.eval()
        correct = total = 0
        with torch.no_grad():
            for images, targets in bundle.val_loader:
                images, targets = images.to(device), targets.to(device)
                correct += int((teacher(images).argmax(dim=1) == targets).sum().item())
                total += int(targets.numel())
        accuracy = correct / max(1, total)
        _log(f"[kd] epoch {epoch:3d}  val acc {accuracy:.4f}")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(
                {
                    "model": teacher.state_dict(),
                    "epoch": epoch,
                    "val_accuracy": accuracy,
                    "num_classes": 100,
                },
                str(checkpoint),
            )
            _log(f"[kd] saved new best ({accuracy:.4f}) to {checkpoint}")

    _log(f"[kd] done. best val accuracy {best_accuracy:.4f}")


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
    # One results root holds one seed. The other seeds' roots are loaded only
    # for the seed-paired statistics; the plots stay on this run's records.
    seed_records: list[dict[str, Any]] = []
    for raw_root in getattr(args, "seed_roots", None) or []:
        seed_root = Path(raw_root)
        if seed_root.resolve() == results_root.resolve():
            continue
        loaded = load_training_records(seed_root)
        _log(f"[compare] Loaded {len(loaded)} record(s) from seed root {seed_root}.")
        seed_records.extend(loaded)
    write_comparison_reports(
        records,
        comparison_root,
        model_keys=getattr(args, "models", None),
        seed_records=seed_records or None,
    )
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


def _handle_docs(args: Any) -> None:
    output = Path(args.output)
    if args.check:
        if current_guide_is_current(output):
            _log(f"[docs] {output} is current.")
            return
        _log(
            f"[docs] {output} is stale or missing. "
            "Run 'uv run dqb docs' and commit the result."
        )
        sys.exit(1)
    written = write_current_guide(output)
    _log(f"[docs] Wrote {written}.")


def _handle_evidence_index(args: Any) -> None:
    roots = index_roots(args.roots)
    _log(
        f"[evidence] Indexing {', '.join(roots)}"
        + (" with manifest hash verification." if args.verify else ".")
    )
    index = build_evidence_index(roots, verify=args.verify)
    json_path, markdown_path = write_evidence_index(
        index,
        json_path=Path(args.json_output),
        markdown_path=Path(args.markdown_output),
    )
    _log(
        f"[evidence] Indexed {len(index.artifacts)} training record(s) across "
        f"{sum(1 for summary in index.roots if summary.exists)} root(s)."
    )
    _log(f"[evidence] Wrote {json_path} and {markdown_path}.")


def main() -> None:
    os.environ.update(load_project_environment())
    torch.multiprocessing.set_sharing_strategy("file_system")
    parser = build_parser()
    if argcomplete is not None:
        argcomplete.autocomplete(parser)
    args = parser.parse_args()
    # These commands describe or inventory the repository itself. They own no
    # result namespace, so they run before any results root is resolved or any
    # log directory is created.
    if args.command == "clean":
        _handle_clean(args)
        return
    if args.command == "docs":
        _handle_docs(args)
        return
    if args.command == "evidence_index":
        _handle_evidence_index(args)
        return

    try:
        results_root, comparison_root, benchmark_root = _normalize_output_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    # PerforatedAI's artifact tree belongs to the result set it was produced
    # for, so it is scoped to --results-root rather than the working directory.
    set_pai_root(results_root / PAI_DIRECTORY_NAME)

    _record_clean_config(args, results_root, comparison_root, benchmark_root)
    setup_logging(output_dir=str(args.logging_dir), script_name=args.command)

    if perforatedai_credentials_present():
        _log("PerforatedAI credentials detected in environment; beta-capable features can be used if installed.")

    if args.command == "run":
        _handle_run(args, results_root, comparison_root)
    elif args.command == "download_data":
        _handle_download_data(args)
    elif args.command == "pretrain_kd_teacher":
        _handle_pretrain_kd_teacher(args)
    elif args.command == "compare":
        _handle_compare(args, results_root, comparison_root)
    elif args.command == "generate_graphs":
        generate_training_graphs(results_root, regenerate=args.regenerate_graphs)
    elif args.command == "benchmark_models":
        _handle_bench(args, results_root, benchmark_root, comparison_root)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
