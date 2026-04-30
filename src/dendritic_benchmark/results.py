from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .plots import (
    bar_chart,
    grouped_bar_chart,
    heatmap,
    line_chart,
    multi_line_chart,
    scatter,
    winner_heatmap,
)
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


def write_model_reports(
    model_display_name: str, records: list[dict[str, Any]], output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    condition_order = [spec.key for spec in CONDITION_SPECS]
    by_condition = {record["condition_key"]: record for record in records}
    metric_values = [
        float(by_condition[key]["metric_value"]) if key in by_condition else 0.0
        for key in condition_order
    ]
    param_values = [
        float(by_condition[key]["param_count"]) if key in by_condition else 0.0
        for key in condition_order
    ]
    size_values = [
        float(by_condition[key]["file_size_mb"]) if key in by_condition else 0.0
        for key in condition_order
    ]
    labels = [condition_by_key(key).display_name for key in condition_order]
    colors = [
        "#2b6cb0" if index < 6 else "#2f855a" for index in range(len(condition_order))
    ]
    metric_name = records[0]["metric_name"] if records else "Metric"
    bar_chart(
        output_dir / "metric_comparison.svg",
        f"{model_display_name}: {metric_name}",
        labels,
        metric_values,
        metric_name,
        colors=colors,
    )
    bar_chart(
        output_dir / "parameter_count.svg",
        f"{model_display_name}: Parameter Count",
        labels,
        param_values,
        "Parameters",
        colors=colors,
    )
    bar_chart(
        output_dir / "model_size.svg",
        f"{model_display_name}: File Size",
        labels,
        size_values,
        "MB",
        colors=colors,
    )


def _process_model_comparison(
    model_key: str,
    records: list[dict[str, Any]],
    baselines: dict[str, dict[str, Any]],
    condition_order: list[str],
    quantization_groups: list[list[str]],
) -> tuple[list[float], list[float], list[int], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (retention_row, quant_row, winner_row, summary_rows, tradeoff_points) for one model."""
    model_records = [r for r in records if r["model_key"] == model_key]
    by_condition = {r["condition_key"]: r for r in model_records}
    baseline = baselines.get(model_key, {})
    retention_row = [
        _normalization_score(by_condition[key], baseline) if key in by_condition else 0.0
        for key in condition_order
    ]
    quant_row: list[float] = []
    winner_row: list[int] = []
    for group in quantization_groups:
        scores = [
            (i, _normalization_score(by_condition[key], baseline))
            for i, key in enumerate(group)
            if key in by_condition
        ]
        if scores:
            best_i, best_score = max(scores, key=lambda t: t[1])
            quant_row.append(best_score)
            winner_row.append(0 if best_i == 0 else 1)
        else:
            quant_row.append(0.0)
            winner_row.append(0)
    summary_rows: list[dict[str, Any]] = []
    tradeoff_points: list[dict[str, Any]] = []
    for condition_key, record in by_condition.items():
        if not baseline:
            continue
        retention = _normalization_score(record, baseline)
        size_reduction = _size_reduction(baseline, record)
        summary_rows.append({
            "model_key": model_key,
            "condition_key": condition_key,
            "metric_name": record["metric_name"],
            "metric_value": record["metric_value"],
            "normalized_score_percent": retention,
            "size_reduction_percent": size_reduction,
            "file_size_mb": record["file_size_mb"],
            "param_count": record["param_count"],
            "nonzero_params": record["nonzero_params"],
        })
        tradeoff_points.append({
            "x": size_reduction,
            "y": retention,
            "label": f"{model_key}:{condition_key}",
            "color": "#2b6cb0" if "dendrites" not in condition_key else "#2f855a",
            "shape": "square" if "dendrites" in condition_key else "circle",
        })
    return retention_row, quant_row, winner_row, summary_rows, tradeoff_points


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
    best_quant_winners: list[list[int]] = []
    tradeoff_points: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for model_key in model_order:
        ret_row, quant_row, winner_row, s_rows, t_points = _process_model_comparison(
            model_key, records, baselines, condition_order, quantization_groups
        )
        retention_rows.append(ret_row)
        best_quant_rows.append(quant_row)
        best_quant_winners.append(winner_row)
        summary_rows.extend(s_rows)
        tradeoff_points.extend(t_points)

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
    winner_heatmap(
        output_dir / "best_quantization_heatmap.svg",
        "Best Quantization Level per Domain",
        [spec.display_name for spec in MODEL_SPECS],
        ["FP32", "Q8", "Q4", "Q2", "Q1.58", "Q1"],
        best_quant_winners,
        best_quant_rows,
        subtitle="Which variant achieves the best retention per quantization level (%)",
    )
    with (output_dir / "summary.csv").open("w", newline="") as fh:
        if summary_rows:
            writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)


def _graphs_numeric_series(
    history: list[dict[str, Any]], column: str
) -> list[float]:
    values: list[float] = []
    for row in history:
        value = row.get(column)
        values.append(float(value) if isinstance(value, (int, float)) else float("nan"))
    return values


def _graphs_has_real_values(values: list[float]) -> bool:
    import math
    return any(not math.isnan(v) for v in values)


def _graphs_slugify(name: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in name)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "metric"


def _load_history_csv(history_file: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Load and parse history.csv. Returns (history_rows, log_lines)."""
    logs: list[str] = []
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
        logs.append(f"  Loaded {len(history)} epoch(s) from history.csv")
    except Exception as e:
        logs.append(f"  ERROR reading {history_file}: {e} — skipping.")
    return history, logs


def _build_metric_series(
    history: list[dict[str, Any]],
    columns: list[tuple[str, str, str]],
) -> list[tuple[str, list[float], str | None]]:
    """Build a list of (label, values, color) series for columns present in history."""
    series: list[tuple[str, list[float], str | None]] = []
    for column, label, color in columns:
        if any(column in row for row in history):
            values = _graphs_numeric_series(history, column)
            if _graphs_has_real_values(values):
                series.append((label, values, color))
    return series


def _write_arch_evolution(
    plots_dir: Path,
    best_arch_file: Path,
    model_key: str,
    condition_key: str,
    metric_name: str,
) -> tuple[list[str], int]:
    logs: list[str] = []
    graph_count = 0
    logs.append("  Found best_arch_scores.csv — generating architecture evolution plot.")
    try:
        arch_history = []
        with best_arch_file.open("r", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                arch_history.append(
                    {"cycle": int(row["cycle"]), "best_metric_value": float(row["best_metric_value"])}
                )
        if arch_history:
            out_path = plots_dir / "architecture_evolution.svg"
            line_chart(
                out_path,
                f"{model_key.upper()} - {condition_key}: Architecture Evolution",
                "Cycle",
                f"Best {metric_name}",
                [row["cycle"] for row in arch_history],
                [row["best_metric_value"] for row in arch_history],
            )
            logs.append(f"  Wrote: {out_path.name}  ({len(arch_history)} cycle(s))")
            graph_count += 1
        else:
            logs.append("  best_arch_scores.csv is empty — skipping architecture evolution plot.")
    except Exception as e:
        logs.append(f"  ERROR reading {best_arch_file}: {e}")
    return logs, graph_count


def _load_condition_metrics(condition_dir: Path) -> tuple[str, str, list[str]]:
    logs: list[str] = []
    metric_name = "Validation Metric"
    primary_metric_key = "metric"
    metrics_file = condition_dir / "metrics.json"
    if metrics_file.exists():
        try:
            metrics_data = json.loads(metrics_file.read_text())
            metric_name = metrics_data.get("metric_name", "Validation Metric")
            primary_metric_key = metrics_data.get("primary_metric_key", primary_metric_key)
            logs.append(f"  Metric: {metric_name} (key: {primary_metric_key})")
        except Exception:
            logs.append("  WARNING: could not parse metrics.json — using defaults.")
    else:
        logs.append("  metrics.json not found — using default metric labels.")
    return metric_name, primary_metric_key, logs


def _write_standard_charts(
    history: list[dict[str, Any]],
    epochs: list[Any],
    plots_dir: Path,
    model_key: str,
    condition_key: str,
    metric_name: str,
    primary_metric_key: str,
) -> tuple[list[str], int]:
    logs: list[str] = []
    graph_count = 0
    primary_series = _build_metric_series(history, [
        ("train_primary_metric", f"Train {metric_name}", "#2b6cb0"),
        ("val_primary_metric", f"Validation {metric_name}", "#2f855a"),
        ("test_primary_metric", f"Test {metric_name}", "#c05621"),
    ])
    if primary_series:
        out_path = plots_dir / "primary_metric.svg"
        multi_line_chart(out_path,
                         f"{model_key.upper()} - {condition_key}: Primary Metric ({primary_metric_key})",
                         "Epoch", metric_name, epochs, primary_series)
        logs.append(f"  Wrote: {out_path.name}  ({len(primary_series)} series)")
        graph_count += 1
    else:
        logs.append("  Skipped primary_metric.svg — no valid series found.")
    loss_series = _build_metric_series(history, [
        ("train_loss", "Train Loss", "#2b6cb0"),
        ("val_loss", "Validation Loss", "#2f855a"),
        ("test_loss", "Test Loss", "#c05621"),
    ])
    if loss_series:
        out_path = plots_dir / "loss_curves.svg"
        multi_line_chart(out_path, f"{model_key.upper()} - {condition_key}: Loss Curves",
                         "Epoch", "Loss", epochs, loss_series)
        logs.append(f"  Wrote: {out_path.name}  ({len(loss_series)} series)")
        graph_count += 1
    else:
        logs.append("  Skipped loss_curves.svg — no valid series found.")
    return logs, graph_count


def _write_grouped_metric_charts(
    history: list[dict[str, Any]],
    epochs: list[Any],
    plots_dir: Path,
    model_key: str,
    condition_key: str,
) -> tuple[list[str], int]:
    logs: list[str] = []
    graph_count = 0
    grouped_suffixes: dict[str, list[tuple[str, list[float], str | None]]] = {}
    for key in sorted({column for row in history for column in row.keys()}):
        for prefix, color, label_prefix in (
            ("train_", "#2b6cb0", "Train"),
            ("val_", "#2f855a", "Validation"),
            ("test_", "#c05621", "Test"),
        ):
            if not key.startswith(prefix):
                continue
            suffix = key[len(prefix):]
            if suffix in {"loss", "primary_metric", "metric"}:
                continue
            values = _graphs_numeric_series(history, key)
            if not _graphs_has_real_values(values):
                continue
            grouped_suffixes.setdefault(suffix, []).append(
                (f"{label_prefix} {suffix.replace('_', ' ').title()}", values, color)
            )
            break
    for suffix, series in sorted(grouped_suffixes.items()):
        out_path = plots_dir / f"metric_{_graphs_slugify(suffix)}.svg"
        multi_line_chart(out_path,
                         f"{model_key.upper()} - {condition_key}: {suffix.replace('_', ' ').title()}",
                         "Epoch", suffix.replace("_", " ").title(), epochs, series)
        logs.append(f"  Wrote: {out_path.name}  ({len(series)} series)")
        graph_count += 1
    return logs, graph_count


def _process_condition_graphs(history_file_str: str, regenerate: bool = False) -> tuple[list[str], int]:
    """Process one condition directory and return (log_lines, graph_count).
    Module-level so it is picklable by ProcessPoolExecutor."""
    import shutil

    logs: list[str] = []
    graph_count = 0
    history_file = Path(history_file_str)
    condition_dir = history_file.parent
    model_key = condition_dir.parent.name
    condition_key = condition_dir.name
    plots_dir = condition_dir / "plots"

    if plots_dir.exists() and any(plots_dir.iterdir()):
        if not regenerate:
            logs.append("  Skipping — plots already exist (use --regenerate-graphs to overwrite).")
            return logs, graph_count
        logs.append(f"  Clearing existing plots directory: {plots_dir}")
        shutil.rmtree(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    logs.append(f"  Plots directory ready: {plots_dir}")

    history, load_logs = _load_history_csv(history_file)
    logs.extend(load_logs)
    if not history or any("ERROR" in line for line in load_logs):
        return logs, graph_count

    epochs = [row["epoch"] for row in history]
    metric_name, primary_metric_key, metric_logs = _load_condition_metrics(condition_dir)
    logs.extend(metric_logs)

    if "val_metric" in history[0] and "val_primary_metric" not in history[0]:
        val_metrics = _graphs_numeric_series(history, "val_metric")
        out_path = plots_dir / "training_curve.svg"
        line_chart(out_path, f"{model_key.upper()} - {condition_key}: Training {metric_name}",
                   "Epoch", metric_name, epochs, val_metrics)
        logs.append(f"  Wrote: {out_path.name}")
        graph_count += 1
        logs.append(f"  Done — {graph_count} graph(s) written.")
        return logs, graph_count

    std_logs, std_count = _write_standard_charts(
        history, epochs, plots_dir, model_key, condition_key, metric_name, primary_metric_key
    )
    logs.extend(std_logs)
    graph_count += std_count

    grp_logs, grp_count = _write_grouped_metric_charts(
        history, epochs, plots_dir, model_key, condition_key
    )
    logs.extend(grp_logs)
    graph_count += grp_count

    best_arch_file = condition_dir / "best_arch_scores.csv"
    if best_arch_file.exists():
        arch_logs, arch_count = _write_arch_evolution(
            plots_dir, best_arch_file, model_key, condition_key, metric_name
        )
        logs.extend(arch_logs)
        graph_count += arch_count

    logs.append(f"  Done — {graph_count} graph(s) written.")
    return logs, graph_count


def generate_training_graphs(results_root: Path, regenerate: bool = False) -> None:
    """Generate training curves from saved history.csv files in all model/condition directories."""
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed

    results_root = Path(results_root)
    if not results_root.exists():
        print(f"[generate_graphs] Results root '{results_root}' does not exist — nothing to do.")
        return

    all_history_files = sorted(results_root.glob("*/*/history.csv"))
    total_conditions = len(all_history_files)
    if total_conditions == 0:
        print(f"[generate_graphs] No history.csv files found under '{results_root}'.")
        return

    workers = os.cpu_count() or 4
    print(f"[generate_graphs] Found {total_conditions} condition(s) — using {workers} parallel workers.")
    print(f"{'─' * 64}")

    graph_count = 0
    completed = 0
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_process_condition_graphs, str(f), regenerate): f
            for f in all_history_files
        }
        for future in as_completed(futures):
            history_file = futures[future]
            condition_dir = history_file.parent
            model_key = condition_dir.parent.name
            condition_key = condition_dir.name
            completed += 1
            try:
                logs, count = future.result()
                print(f"[{completed}/{total_conditions}] {model_key} / {condition_key}")
                for line in logs:
                    print(line)
                graph_count += count
            except Exception as e:
                print(f"[{completed}/{total_conditions}] {model_key} / {condition_key}  ERROR: {e}")

    print(f"{'─' * 64}")
    print(f"[generate_graphs] Finished. {graph_count} graph(s) written across {total_conditions} condition(s).")
