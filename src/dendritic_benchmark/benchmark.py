from __future__ import annotations

import csv
import json
import os
import platform
from datetime import datetime
from pathlib import Path
from typing import Any

from .compat import choose_device, require_torch
from .models import build_model
from .specs import CONDITION_SPECS, MODEL_SPECS, condition_by_key, model_by_key

_GRAPH_MODELS = {"gcn", "gin_imdbb", "mpnn", "attentivefp_freesolv"}
_TEXT_MODELS = {"textcnn", "distilbert"}
_LATENCY_CSV_FIELDS = ["condition_key", "batch_size", "mean_latency_ms", "median_latency_ms"]
_MANIFEST_CSV_FIELDS = ["model_key", "condition_key", "batch_size", "mean_latency_ms", "median_latency_ms"]


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


def get_system_info() -> dict[str, Any]:
    torch = require_torch()
    return {
        "timestamp": datetime.now().isoformat(),
        "platform": platform.system(),
        "platform_version": platform.version(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "cpu_count": os.cpu_count(),
    }


def get_model_input_shapes(model_key: str) -> tuple:
    shapes_map = {
        "lenet5": (1, 28, 28),
        "m5": (1, 16000),
        "lstm_forecaster": (24, 1),
        "textcnn": (64,),
        "gcn": ((50, 1433), (50, 50)),
        "tabnet": (14,),
        "mpnn": ((24, 9), (24, 24)),
        "actor_critic": (4,),
        "lstm_autoencoder": (128, 1),
        "distilbert": (96,),
        "dqn_lunarlander": (8,),
        "ppo_bipedalwalker": (24,),
        "attentivefp_freesolv": ((24, 9), (24, 24)),
        "gin_imdbb": ((96, 8), (96, 96)),
        "tcn_forecaster": (96, 7),
        "gru_forecaster": (96, 21),
        "pointnet_modelnet40": (1024, 3),
        "vae_mnist": (1, 28, 28),
        "snn_nmnist": (2, 34, 34),
        "unet_isic": (3, 128, 128),
        "resnet18_cifar10": (3, 32, 32),
        "mobilenetv2_cifar10": (3, 32, 32),
        "saint_adult": (14,),
        "capsnet_mnist": (1, 28, 28),
        "convlstm_movingmnist": (10, 1, 64, 64),
    }
    if model_key not in shapes_map:
        raise KeyError(f"Unknown model key: {model_key}")
    return shapes_map[model_key]


def generate_sample_inputs(model_key: str, batch_size: int) -> tuple[Any, Any]:
    """Return a 2-tuple (primary_input, adjacency). adjacency is None for non-graph models."""
    torch = require_torch()
    device = choose_device()
    shape = get_model_input_shapes(model_key)

    if model_key in _GRAPH_MODELS:
        node_features_shape, adjacency_shape = shape
        return (
            torch.randn(batch_size, *node_features_shape, device=device),
            torch.randn(batch_size, *adjacency_shape, device=device),
        )

    if model_key in _TEXT_MODELS:
        vocab_size = 5000 if model_key == "textcnn" else 30522
        return (torch.randint(0, vocab_size, (batch_size, *shape), device=device), None)

    return (torch.randn(batch_size, *shape, device=device), None)


def benchmark_model_latency(
    model: Any,
    inputs: tuple[Any, Any],
    batch_size: int,
    num_runs: int = 10,
) -> dict[str, Any]:
    torch = require_torch()
    from torch.utils.benchmark import Timer

    primary, adjacency = inputs
    model.eval()
    with torch.no_grad():
        warmup_runs = 3
        if adjacency is not None:
            for _ in range(warmup_runs):
                model(primary, adjacency)
            timer = Timer(
                stmt="model(x, adj)",
                globals={"model": model, "x": primary, "adj": adjacency},
            )
        else:
            for _ in range(warmup_runs):
                model(primary)
            timer = Timer(
                stmt="model(x)",
                globals={"model": model, "x": primary},
            )

        result = timer.timeit(number=num_runs)

    return {
        "batch_size": batch_size,
        "num_runs": num_runs,
        "mean_latency_ms": result.mean * 1000,
        "median_latency_ms": result.median * 1000,
        "stdev_latency_ms": (result.stdev if hasattr(result, "stdev") else 0) * 1000,
    }


class BenchmarkOrchestrator:
    def __init__(self, results_root: Path | str = "results"):
        self.results_root = Path(results_root)

    def _load_model_state(self, model: Any, condition_dir: Path) -> bool:
        import torch

        model_path = condition_dir / "model.pt"
        if not model_path.exists():
            return False

        try:
            state = torch.load(model_path, map_location="cpu", weights_only=False)
            model_state = model.state_dict()
            compatible_state = {
                key: value
                for key, value in state.items()
                if not key.endswith("tracker_string")
                and model_state.get(key) is not None
                and hasattr(value, "shape")
                and hasattr(model_state[key], "shape")
                and value.shape == model_state[key].shape
            }
            model.load_state_dict(compatible_state, strict=False)
            return True
        except Exception as exc:
            _log(f"Failed to load model state: {exc}")
            return False

    def benchmark_condition(
        self,
        model_key: str,
        condition_key: str,
        batch_sizes: list[int],
        num_runs: int = 10,
    ) -> dict[str, Any]:
        device = choose_device()
        condition_spec = condition_by_key(condition_key)
        condition_dir = self.results_root / model_key / condition_key

        if not condition_dir.exists():
            return {"model_key": model_key, "condition_key": condition_key, "error": "condition directory not found"}

        model = build_model(model_key).to(device)
        if not self._load_model_state(model, condition_dir):
            return {"model_key": model_key, "condition_key": condition_key, "error": "failed to load model state"}

        results: dict[str, Any] = {
            "model_key": model_key,
            "condition_key": condition_key,
            "display_name": condition_spec.display_name,
            "timestamp": datetime.now().isoformat(),
            "batch_sizes": {},
        }

        for batch_size in batch_sizes:
            try:
                primary, adjacency = generate_sample_inputs(model_key, batch_size)
                inputs: tuple[Any, Any] = (
                    primary.to(device),
                    adjacency.to(device) if adjacency is not None else None,
                )
                results["batch_sizes"][batch_size] = benchmark_model_latency(
                    model, inputs, batch_size, num_runs
                )
            except Exception as exc:
                results["batch_sizes"][batch_size] = {"error": str(exc)}

        return results

    def _collect_manifest_rows(self, result: dict[str, Any], model_key: str) -> list[dict[str, Any]]:
        if "error" in result:
            return []
        rows = []
        for batch_size, stats in result["batch_sizes"].items():
            if "error" not in stats:
                rows.append({
                    "model_key": model_key,
                    "condition_key": result["condition_key"],
                    "batch_size": batch_size,
                    "mean_latency_ms": stats["mean_latency_ms"],
                    "median_latency_ms": stats["median_latency_ms"],
                })
        return rows

    def _write_latency_summary(self, model_dir: Path, model_results: list[dict[str, Any]]) -> None:
        latency_summary_file = model_dir / "latency_summary.csv"
        with latency_summary_file.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=_LATENCY_CSV_FIELDS)
            writer.writeheader()
            for result in model_results:
                if "error" in result:
                    continue
                for batch_size, stats in result["batch_sizes"].items():
                    if "error" not in stats:
                        writer.writerow({
                            "condition_key": result["condition_key"],
                            "batch_size": batch_size,
                            "mean_latency_ms": stats["mean_latency_ms"],
                            "median_latency_ms": stats["median_latency_ms"],
                        })

    def _write_manifest(self, benchmark_root: Path, manifest_data: list[dict[str, Any]]) -> None:
        if not manifest_data:
            return
        manifest_file = benchmark_root / "manifest.csv"
        with manifest_file.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=_MANIFEST_CSV_FIELDS)
            writer.writeheader()
            writer.writerows(manifest_data)

    def _benchmark_model(
        self,
        model_key: str,
        condition_keys: list[str],
        batch_sizes: list[int],
        num_runs: int,
        benchmark_root: Path,
        total_benchmarks: int,
        counter: list[int],
    ) -> list[dict[str, Any]]:
        model_dir = benchmark_root / model_key
        model_dir.mkdir(parents=True, exist_ok=True)
        model_results = []

        for condition_key in condition_keys:
            counter[0] += 1
            _log(f"[{counter[0]}/{total_benchmarks}] {model_key} / {condition_key}…")
            result = self.benchmark_condition(model_key, condition_key, batch_sizes, num_runs)
            model_results.append(result)
            (model_dir / f"{condition_key}.json").write_text(json.dumps(result, indent=2))

        self._write_latency_summary(model_dir, model_results)
        return model_results

    def benchmark_all(
        self,
        model_keys: list[str] | None = None,
        condition_keys: list[str] | None = None,
        batch_sizes: list[int] | None = None,
        num_runs: int = 10,
        benchmark_root: Path | str = "benchmarks",
        comparison_root: Path | str | None = None,
    ) -> None:
        if batch_sizes is None:
            batch_sizes = [1, 32]
        if model_keys is None:
            model_keys = [spec.key for spec in MODEL_SPECS]
        if condition_keys is None:
            condition_keys = [spec.key for spec in CONDITION_SPECS]

        benchmark_root = Path(benchmark_root)
        benchmark_root.mkdir(parents=True, exist_ok=True)
        (benchmark_root / "computer_info.json").write_text(json.dumps(get_system_info(), indent=2))

        total_benchmarks = len(model_keys) * len(condition_keys)
        counter = [0]
        manifest_data: list[dict[str, Any]] = []

        for model_key in model_keys:
            model_results = self._benchmark_model(
                model_key, condition_keys, batch_sizes, num_runs, benchmark_root, total_benchmarks, counter
            )
            for result in model_results:
                manifest_data.extend(self._collect_manifest_rows(result, model_key))

        self._write_manifest(benchmark_root, manifest_data)
        if comparison_root is not None:
            from .results import write_per_model_benchmark_plots
            write_per_model_benchmark_plots(benchmark_root, Path(comparison_root))
        _log(f"Benchmarking complete. Results written to {benchmark_root}")
