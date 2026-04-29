from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .plots import bar_chart, grouped_bar_chart, heatmap, scatter, line_chart, multi_line_chart
from .specs import CONDITION_SPECS, MODEL_SPECS, condition_by_key
from .training import TrainingRecord


def save_training_record(record: TrainingRecord, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = record.to_dict()
    (output_dir / "record.json").write_text(json.dumps(payload, indent=2))
    with (output_dir / "record.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(payload.keys()))
        writer.writeheader()
        writer.writerow(payload)


def load_training_records(results_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not results_root.exists():
        return records
    for record_file in results_root.glob("*/*/record.json"):
        records.append(json.loads(record_file.read_text()))
    return records


def _baseline_lookup(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    baselines: dict[str, dict[str, Any]] = {}
    for record in records:
        if record["condition_key"] == "base_fp32":
            baselines[record["model_key"]] = record
    return baselines


def _normalization_score(record: dict[str, Any], baseline: dict[str, Any]) -> float:
    if not baseline:
        return 0.0
    baseline_metric = float(baseline["metric_value"])
    current_metric = float(record["metric_value"])
    if baseline_metric == 0:
        return 0.0
    if baseline["metric_direction"] == "maximize":
        return 100.0 * current_metric / baseline_metric
    if current_metric == 0:
        return 0.0
    return 100.0 * baseline_metric / current_metric


def _size_reduction(baseline: dict[str, Any], record: dict[str, Any]) -> float:
    baseline_size = float(baseline["file_size_mb"])
    current_size = float(record["file_size_mb"])
    if baseline_size <= 0:
        return 0.0
    return 100.0 * (1.0 - current_size / baseline_size)


def write_manifest(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as fh:
        if not records:
            fh.write("")
            return
        writer = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def write_model_reports(model_display_name: str, records: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    condition_order = [spec.key for spec in CONDITION_SPECS]
    by_condition = {record["condition_key"]: record for record in records}
    metric_values = [float(by_condition[key]["metric_value"]) if key in by_condition else 0.0 for key in condition_order]
    param_values = [float(by_condition[key]["param_count"]) if key in by_condition else 0.0 for key in condition_order]
    size_values = [float(by_condition[key]["file_size_mb"]) if key in by_condition else 0.0 for key in condition_order]
    labels = [condition_by_key(key).display_name for key in condition_order]
    colors = ["#2b6cb0" if index < 6 else "#2f855a" for index in range(len(condition_order))]
    metric_name = records[0]["metric_name"] if records else "Metric"
    bar_chart(output_dir / "metric_comparison.svg", f"{model_display_name}: {metric_name}", labels, metric_values, metric_name, colors=colors)
    bar_chart(output_dir / "parameter_count.svg", f"{model_display_name}: Parameter Count", labels, param_values, "Parameters", colors=colors)
    bar_chart(output_dir / "model_size.svg", f"{model_display_name}: File Size", labels, size_values, "MB", colors=colors)


def write_comparison_reports(records: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    baselines = _baseline_lookup(records)
    model_order = [spec.key for spec in MODEL_SPECS]
    condition_order = [spec.key for spec in CONDITION_SPECS]
    quantization_groups = [
        ["base_fp32", "dendrites_fp32"],
        ["base_q8", "dendrites_pruned_q8"],
        ["base_q4", "dendrites_pruned_q4"],
        ["base_q2", "dendrites_pruned_q2"],
        ["base_q1_58", "dendrites_pruned_q1_58"],
        ["base_q1", "dendrites_pruned_q1"],
    ]
    retention_rows: list[list[float]] = []
    best_quant_rows: list[list[float]] = []
    tradeoff_points: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for model_key in model_order:
        model_records = [record for record in records if record["model_key"] == model_key]
        by_condition = {record["condition_key"]: record for record in model_records}
        baseline = baselines.get(model_key, {})
        retention_rows.append([
            _normalization_score(by_condition[key], baseline) if key in by_condition else 0.0
            for key in condition_order
        ])
        best_quant_rows.append(
            [
                max(
                    [
                        _normalization_score(by_condition[key], baseline)
                        for key in group
                        if key in by_condition
                    ]
                    or [0.0]
                )
                for group in quantization_groups
            ]
        )
        for condition_key, record in by_condition.items():
            if not baseline:
                continue
            retention = _normalization_score(record, baseline)
            size_reduction = _size_reduction(baseline, record)
            summary_rows.append(
                {
                    "model_key": model_key,
                    "condition_key": condition_key,
                    "metric_name": record["metric_name"],
                    "metric_value": record["metric_value"],
                    "normalized_score_percent": retention,
                    "size_reduction_percent": size_reduction,
                    "file_size_mb": record["file_size_mb"],
                    "param_count": record["param_count"],
                    "nonzero_params": record["nonzero_params"],
                }
            )
            tradeoff_points.append(
                {
                    "x": size_reduction,
                    "y": retention,
                    "label": f"{model_key}:{condition_key}",
                    "color": "#2b6cb0" if "dendrites" not in condition_key else "#2f855a",
                    "shape": "square" if "dendrites" in condition_key else "circle",
                }
            )

    heatmap(
        output_dir / "accuracy_retention_heatmap.svg",
        "Accuracy Retention Heatmap",
        [spec.display_name for spec in MODEL_SPECS],
        [spec.display_name for spec in CONDITION_SPECS],
        retention_rows,
        subtitle="Score vs. Base FP32 (%)",
    )
    scatter(
        output_dir / "size_tradeoff_scatter.svg",
        "Size Reduction vs. Accuracy Retention",
        tradeoff_points,
        x_label="File Size Reduction %",
        y_label="Normalized Score %",
    )
    grouped_bar_chart(
        output_dir / "dendrite_delta.svg",
        "Dendrite Delta",
        [spec.display_name for spec in MODEL_SPECS],
        [
            ("Base FP32", [row[0] for row in retention_rows], "#2b6cb0"),
            ("+Dendrites FP32", [row[6] for row in retention_rows], "#2f855a"),
        ],
        "Retention %",
    )
    heatmap(
        output_dir / "best_quantization_heatmap.svg",
        "Best Quantization Level per Domain",
        [spec.display_name for spec in MODEL_SPECS],
        ["FP32", "Q8", "Q4", "Q2", "Q1.58", "Q1"],
        best_quant_rows,
        subtitle="Best retention among baseline and dendritic pruned variants (%)",
    )
    with (output_dir / "summary.csv").open("w", newline="") as fh:
        if summary_rows:
            writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)


def generate_training_graphs(results_root: Path) -> None:
    """Generate training curves from saved history.csv files in all model/condition directories."""
    import shutil

    results_root = Path(results_root)
    if not results_root.exists():
        print(f"Results root {results_root} does not exist")
        return

    graph_count = 0
    for history_file in sorted(results_root.glob("*/*/history.csv")):
        condition_dir = history_file.parent
        model_key = condition_dir.parent.name
        condition_key = condition_dir.name
        plots_dir = condition_dir / "plots"

        # Clean and recreate the plots directory
        if plots_dir.exists():
            shutil.rmtree(plots_dir)
        plots_dir.mkdir(parents=True)

        history: list[dict[str, Any]] = []
        try:
            with history_file.open("r", newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    parsed_row: dict[str, Any] = {}
                    for key, value in row.items():
                        if value is None or value == "":
                            parsed_row[key] = None
                            continue
                        if key == "epoch":
                            parsed_row[key] = int(float(value))
                            continue
                        try:
                            parsed_row[key] = float(value)
                        except ValueError:
                            parsed_row[key] = value
                    history.append(parsed_row)
        except Exception as e:
            print(f"Error reading {history_file}: {e}")
            continue

        if not history:
            continue

        epochs = [row["epoch"] for row in history]

        metrics_file = condition_dir / "metrics.json"
        metric_name = "Validation Metric"
        primary_metric_key = "metric"
        if metrics_file.exists():
            try:
                metrics_data = json.loads(metrics_file.read_text())
                metric_name = metrics_data.get("metric_name", "Validation Metric")
                primary_metric_key = metrics_data.get("primary_metric_key", primary_metric_key)
            except Exception:
                pass

        def numeric_series(column: str) -> list[float]:
            values: list[float] = []
            for row in history:
                value = row.get(column)
                values.append(float(value) if isinstance(value, (int, float)) else float("nan"))
            return values

        def has_real_values(values: list[float]) -> bool:
            return any(value == value for value in values)

        def slugify(name: str) -> str:
            cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in name)
            while "__" in cleaned:
                cleaned = cleaned.replace("__", "_")
            return cleaned.strip("_") or "metric"

        if "val_metric" in history[0] and "val_primary_metric" not in history[0]:
            val_metrics = numeric_series("val_metric")
            title = f"{model_key.upper()} - {condition_key}: Training {metric_name}"
            line_chart(
                plots_dir / "training_curve.svg",
                title,
                "Epoch",
                metric_name,
                epochs,
                val_metrics,
            )
            graph_count += 1
            continue

        primary_series: list[tuple[str, list[float], str | None]] = []
        for column, label, color in (
            ("train_primary_metric", f"Train {metric_name}", "#2b6cb0"),
            ("val_primary_metric", f"Validation {metric_name}", "#2f855a"),
            ("test_primary_metric", f"Test {metric_name}", "#c05621"),
        ):
            if any(column in row for row in history):
                values = numeric_series(column)
                if has_real_values(values):
                    primary_series.append((label, values, color))
        if primary_series:
            multi_line_chart(
                plots_dir / "primary_metric.svg",
                f"{model_key.upper()} - {condition_key}: Primary Metric ({primary_metric_key})",
                "Epoch",
                metric_name,
                epochs,
                primary_series,
            )
            graph_count += 1

        loss_series: list[tuple[str, list[float], str | None]] = []
        for column, label, color in (
            ("train_loss", "Train Loss", "#2b6cb0"),
            ("val_loss", "Validation Loss", "#2f855a"),
            ("test_loss", "Test Loss", "#c05621"),
        ):
            if any(column in row for row in history):
                values = numeric_series(column)
                if has_real_values(values):
                    loss_series.append((label, values, color))
        if loss_series:
            multi_line_chart(
                plots_dir / "loss_curves.svg",
                f"{model_key.upper()} - {condition_key}: Loss Curves",
                "Epoch",
                "Loss",
                epochs,
                loss_series,
            )
            graph_count += 1

        grouped_suffixes: dict[str, list[tuple[str, list[float], str | None]]] = {}
        for key in sorted({column for row in history for column in row.keys()}):
            for prefix, color, label_prefix in (
                ("train_", "#2b6cb0", "Train"),
                ("val_", "#2f855a", "Validation"),
                ("test_", "#c05621", "Test"),
            ):
                if not key.startswith(prefix):
                    continue
                suffix = key[len(prefix) :]
                if suffix in {"loss", "primary_metric", "metric"}:
                    continue
                values = numeric_series(key)
                if not has_real_values(values):
                    continue
                grouped_suffixes.setdefault(suffix, []).append((f"{label_prefix} {suffix.replace('_', ' ').title()}", values, color))
                break

        for suffix, series in sorted(grouped_suffixes.items()):
            multi_line_chart(
                plots_dir / f"metric_{slugify(suffix)}.svg",
                f"{model_key.upper()} - {condition_key}: {suffix.replace('_', ' ').title()}",
                "Epoch",
                suffix.replace("_", " ").title(),
                epochs,
                series,
            )
            graph_count += 1

        best_arch_file = condition_dir / "best_arch_scores.csv"
        if best_arch_file.exists():
            try:
                arch_history = []
                with best_arch_file.open("r", newline="") as fh:
                    reader = csv.DictReader(fh)
                    for row in reader:
                        arch_history.append(
                            {
                                "cycle": int(row["cycle"]),
                                "best_metric_value": float(row["best_metric_value"]),
                            }
                        )
                if arch_history:
                    cycles = [row["cycle"] for row in arch_history]
                    best_metrics = [row["best_metric_value"] for row in arch_history]
                    line_chart(
                        plots_dir / "architecture_evolution.svg",
                        f"{model_key.upper()} - {condition_key}: Architecture Evolution",
                        "Cycle",
                        f"Best {metric_name}",
                        cycles,
                        best_metrics,
                    )
                    graph_count += 1
            except Exception as e:
                print(f"Error reading {best_arch_file}: {e}")

    print(f"Generated {graph_count} training graphs in {results_root}")
