from __future__ import annotations

import argparse
from pathlib import Path

from .compat import load_project_environment, perforatedai_credentials_present
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
