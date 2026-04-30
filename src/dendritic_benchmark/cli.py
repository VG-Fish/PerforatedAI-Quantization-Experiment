from __future__ import annotations

import argparse
from pathlib import Path

from .compat import load_project_environment, perforatedai_credentials_present
from .data import build_task_bundle
from .pipeline import BenchmarkRunner
from .results import load_training_records, write_comparison_reports, write_manifest, write_model_reports, generate_training_graphs
from .specs import MODEL_SPECS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dendritic quantization benchmark runner")
    parser.add_argument("--results-root", default="results", help="Directory for per-model result folders")
    parser.add_argument("--comparison-root", default="comparison", help="Directory for cross-model comparison outputs")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run the benchmark")
    run_parser.add_argument("--models", nargs="*", help="Optional subset of model keys")
    run_parser.add_argument("--conditions", nargs="*", help="Optional subset of condition keys")

    download_parser = subparsers.add_parser("download_data", help="Download/cache datasets for selected models")
    download_parser.add_argument("--models", nargs="*", help="Optional subset of model keys")
    download_parser.add_argument("--strict", action="store_true", help="Stop on the first dataset download error")

    compare_parser = subparsers.add_parser("compare", help="Build comparison outputs from saved records")
    compare_parser.add_argument("--manifest", action="store_true", help="Rewrite the manifest CSV before plotting")

    generate_graphs_parser = subparsers.add_parser("generate_graphs", help="Generate training curves from saved results")
    generate_graphs_parser.add_argument(
        "--regenerate-graphs",
        action="store_true",
        help="Recreate graphs even if they already exist",
    )

    return parser


def main() -> None:
    load_project_environment()
    parser = build_parser()
    args = parser.parse_args()
    results_root = Path(args.results_root)
    comparison_root = Path(args.comparison_root)

    if perforatedai_credentials_present():
        print("PerforatedAI credentials detected in environment; beta-capable features can be used if installed.")

    if args.command == "run":
        runner = BenchmarkRunner(results_root=results_root, comparison_root=comparison_root)
        runner.run(model_keys=args.models, condition_keys=args.conditions)
        return

    if args.command == "download_data":
        selected = args.models or [spec.key for spec in MODEL_SPECS]
        failures = []
        for model_key in selected:
            print(f"[data] preparing {model_key}")
            try:
                build_task_bundle(model_key)
            except Exception as exc:
                if args.strict:
                    raise
                failures.append((model_key, str(exc)))
                print(f"[data] FAILED {model_key}: {exc}")
        if failures:
            print("[data] completed with failures:")
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
