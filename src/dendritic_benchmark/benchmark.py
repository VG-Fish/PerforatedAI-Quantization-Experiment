import csv
import json
import os
import platform
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from .compat import choose_device
from .data import (
    AG_NEWS_SEQ_LEN,
    CORA_NODE_FEATURES,
    CORA_NODES,
    ESOL_MAX_ATOMS,
    FREESOLV_MAX_ATOMS,
)
from .models import (
    ADULT_CATEGORICAL_CARDINALITIES,
    ADULT_FEATURES,
    FORECAST_SEQ_LEN,
    MOLECULE_EDGE_FEATURES,
    MOLECULE_NODE_FEATURES,
    SOCIAL_GRAPH_NODE_FEATURES,
    build_model,
)
from .specs import CONDITION_SPECS, MODEL_SPECS, condition_by_key

# Take (node_features, adjacency); the molecular pair additionally take a dense
# edge-feature tensor and so are handled separately.
_GRAPH_MODELS = {"gcn", "gin_imdbb"}
_MOLECULAR_MODELS = {"mpnn", "attentivefp_freesolv"}
_TABULAR_MODELS = {"tabnet", "saint_adult"}
_TEXT_MODELS = {"textcnn", "distilbert"}
# Largest latency batch a model's input can meaningfully be stacked to; see
# _supported_batch_sizes. Absent means unconstrained.
_MAX_LATENCY_BATCH_SIZE = {"gcn": 1}
_LATENCY_CSV_FIELDS = [
    "condition_key",
    "batch_size",
    "mean_latency_ms",
    "median_latency_ms",
]
_MANIFEST_CSV_FIELDS = [
    "model_key",
    "condition_key",
    "batch_size",
    "mean_latency_ms",
    "median_latency_ms",
]


def _log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


def get_system_info() -> dict[str, Any]:
    return {
        "timestamp": datetime.now().isoformat(),
        "platform": platform.system(),
        "platform_version": platform.version(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.mps.is_available(),
        "cpu_count": os.cpu_count(),
    }


def get_model_input_shapes(model_key: str) -> tuple:
    """Per-sample input shapes used to synthesise latency-benchmark batches.

    Derived from the loaders' own constants wherever one exists. Several entries
    here were hand-written copies that had drifted from the featurisers — Cora
    at 50 nodes against a 64-node ego graph, the molecular sets at 9 node
    features against 20 — so latency was being measured on tensors no training
    run ever sees.
    """
    shapes_map: dict[str, tuple] = {
        "lenet5": (1, 28, 28),
        "m5": (1, 16000),
        "lstm_forecaster": (FORECAST_SEQ_LEN, 1),
        "textcnn": (AG_NEWS_SEQ_LEN,),
        "gcn": ((CORA_NODES, CORA_NODE_FEATURES), (CORA_NODES, CORA_NODES)),
        "tabnet": (ADULT_FEATURES,),
        "mpnn": (
            (ESOL_MAX_ATOMS, MOLECULE_NODE_FEATURES),
            (ESOL_MAX_ATOMS, ESOL_MAX_ATOMS),
            (ESOL_MAX_ATOMS, ESOL_MAX_ATOMS, MOLECULE_EDGE_FEATURES),
        ),
        "actor_critic": (4,),
        "lstm_autoencoder": (128, 1),
        "distilbert": (128,),
        "dqn_lunarlander": (8,),
        "ppo_bipedalwalker": (24,),
        "attentivefp_freesolv": (
            (FREESOLV_MAX_ATOMS, MOLECULE_NODE_FEATURES),
            (FREESOLV_MAX_ATOMS, FREESOLV_MAX_ATOMS),
            (FREESOLV_MAX_ATOMS, FREESOLV_MAX_ATOMS, MOLECULE_EDGE_FEATURES),
        ),
        "gin_imdbb": ((96, SOCIAL_GRAPH_NODE_FEATURES), (96, 96)),
        "tcn_forecaster": (FORECAST_SEQ_LEN, 7),
        "gru_forecaster": (FORECAST_SEQ_LEN, 21),
        "pointnet_modelnet40": (1024, 3),
        "vae_mnist": (1, 28, 28),
        "snn_nmnist": (2, 34, 34),
        "unet_isic": (3, 128, 128),
        "resnet18_cifar10": (3, 32, 32),
        "resnet18_hf_perforated_cifar10": (3, 32, 32),
        "mobilenetv2_cifar10": (3, 32, 32),
        "saint_adult": (ADULT_FEATURES,),
        "capsnet_mnist": (1, 28, 28),
    }
    return shapes_map[model_key]


def _supported_batch_sizes(model_key: str, requested: list[int]) -> list[int]:
    """Clamp requested latency batch sizes to what the model can actually take.

    Only transductive Cora needs this. Its "batch" is the entire 2708-node
    graph, so a batch of 32 would mean 32 copies of a 2708x2708 adjacency —
    939 MB of synthetic input for a measurement that means nothing, because no
    training or inference path ever stacks the graph. Everything above 1 is
    clamped to 1 and de-duplicated, so gcn reports a single batch-1 latency
    instead of an out-of-memory error at every other size.
    """
    ceiling = _MAX_LATENCY_BATCH_SIZE.get(model_key)
    if ceiling is None:
        return list(requested)
    clamped = [min(batch_size, ceiling) for batch_size in requested]
    return list(dict.fromkeys(clamped))


def generate_sample_inputs(model_key: str, batch_size: int) -> tuple[Any, Any]:
    """Return a 2-tuple (primary_input, adjacency).

    ``adjacency`` is None for non-graph models. When the model takes more than
    two positional tensors, ``primary_input`` is a tuple and ``adjacency`` stays
    None; ``benchmark_model_latency`` splats it.
    """
    device = choose_device()
    shape: Any = get_model_input_shapes(model_key)

    if model_key in _MOLECULAR_MODELS:
        node_shape, adjacency_shape, edge_shape = shape
        return (
            (
                torch.randn([batch_size, *node_shape], device=device),
                torch.randn([batch_size, *adjacency_shape], device=device),
                torch.randn([batch_size, *edge_shape], device=device),
            ),
            None,
        )

    if model_key in _TABULAR_MODELS:
        # Categorical columns index embedding tables, so they need in-range
        # integer codes; Gaussian noise would index with a negative float.
        row = torch.randn([batch_size, *shape], device=device)
        for column, cardinality in ADULT_CATEGORICAL_CARDINALITIES.items():
            row[:, column] = torch.randint(
                0, cardinality, (batch_size,), device=device
            ).float()
        return (row, None)

    if model_key in _GRAPH_MODELS:
        node_features_shape, adjacency_shape = shape
        return (
            torch.randn([batch_size, *node_features_shape], device=device),
            torch.randn([batch_size, *adjacency_shape], device=device),
        )

    if model_key == "distilbert":
        input_ids = torch.randint(0, 30522, [batch_size, *shape], device=device)
        attention_mask = torch.ones_like(input_ids)
        return ((input_ids, attention_mask), None)

    if model_key in _TEXT_MODELS:
        vocab_size = 5000 if model_key == "textcnn" else 30522
        return (torch.randint(0, vocab_size, [batch_size, *shape], device=device), None)

    return (torch.randn([batch_size, *shape], device=device), None)


def benchmark_model_latency(
    model: Any,
    inputs: tuple[Any, Any],
    batch_size: int,
    num_runs: int = 5,
) -> dict[str, Any]:
    import statistics

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
        elif isinstance(primary, tuple):
            for _ in range(warmup_runs):
                model(*primary)
            timer = Timer(
                stmt="model(*x)",
                globals={"model": model, "x": primary},
            )
        else:
            for _ in range(warmup_runs):
                model(primary)
            timer = Timer(
                stmt="model(x)",
                globals={"model": model, "x": primary},
            )

        run_times_ms = [timer.timeit(number=1).mean * 1000 for _ in range(num_runs)]

    mean_ms = statistics.mean(run_times_ms)
    median_ms = statistics.median(run_times_ms)
    stdev_ms = statistics.stdev(run_times_ms) if num_runs > 1 else 0.0

    return {
        "batch_size": batch_size,
        "num_runs": num_runs,
        "mean_latency_ms": mean_ms,
        "median_latency_ms": median_ms,
        "stdev_latency_ms": stdev_ms,
    }


def _move_to_device(value: Any, device: Any) -> Any:
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    move = getattr(value, "to", None)
    return move(device) if move is not None else value


class BenchmarkOrchestrator:
    def __init__(self, results_root: Path | str = "results"):
        self.results_root = Path(results_root)

    def _load_model_state(self, model: Any, condition_dir: Path) -> bool:
        import torch

        model_path = condition_dir / "model.pt"
        if not model_path.exists():
            return False

        try:
            state = torch.load(model_path, map_location="cpu", weights_only=True)
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
            # A dendritic model.pt carries PerforatedAI wrapper key names, so
            # against a plain build_model() skeleton almost nothing matches.
            # The old strict=False load then "succeeded" on a handful of
            # tensors and the latency row silently described an unperforated,
            # mostly randomly-initialized network — dendrites_* latencies were
            # base-architecture numbers. Refuse instead: a missing row is
            # honest, a wrong-architecture row is not.
            skipped_source = [
                key for key in state
                if key not in compatible_state and not key.endswith("tracker_string")
            ]
            unfilled_target = [
                key for key in model_state
                if key not in compatible_state and not key.endswith("tracker_string")
            ]
            if skipped_source or unfilled_target:
                _log(
                    f"model.pt in {condition_dir.name} does not match the plain "
                    f"model architecture ({len(skipped_source)} checkpoint "
                    f"tensor(s) unloadable, {len(unfilled_target)} model "
                    "tensor(s) unfilled — likely a PerforatedAI dendritic "
                    "checkpoint); refusing to benchmark the wrong architecture."
                )
                return False
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
        num_runs: int = 5,
    ) -> dict[str, Any]:
        device = choose_device()
        condition_spec = condition_by_key(condition_key)
        condition_dir = self.results_root / model_key / condition_key

        if not condition_dir.exists():
            return {
                "model_key": model_key,
                "condition_key": condition_key,
                "error": "condition directory not found",
            }

        model = build_model(model_key).to(device)
        if not self._load_model_state(model, condition_dir):
            return {
                "model_key": model_key,
                "condition_key": condition_key,
                "error": "failed to load model state",
            }

        results: dict[str, Any] = {
            "model_key": model_key,
            "condition_key": condition_key,
            "display_name": condition_spec.display_name,
            "timestamp": datetime.now().isoformat(),
            "batch_sizes": {},
        }

        for batch_size in _supported_batch_sizes(model_key, batch_sizes):
            try:
                primary, adjacency = generate_sample_inputs(model_key, batch_size)
                inputs: tuple[Any, Any] = (
                    _move_to_device(primary, device),
                    adjacency.to(device) if adjacency is not None else None,
                )
                results["batch_sizes"][batch_size] = benchmark_model_latency(
                    model, inputs, batch_size, num_runs
                )
            except Exception as exc:
                results["batch_sizes"][batch_size] = {"error": str(exc)}

        return results

    def _collect_manifest_rows(
        self, result: dict[str, Any], model_key: str
    ) -> list[dict[str, Any]]:
        if "error" in result:
            return []
        rows = []
        for batch_size, stats in result["batch_sizes"].items():
            if "error" not in stats:
                rows.append(
                    {
                        "model_key": model_key,
                        "condition_key": result["condition_key"],
                        "batch_size": batch_size,
                        "mean_latency_ms": stats["mean_latency_ms"],
                        "median_latency_ms": stats["median_latency_ms"],
                    }
                )
        return rows

    def _write_latency_summary(
        self, model_dir: Path, model_results: list[dict[str, Any]]
    ) -> None:
        latency_summary_file = model_dir / "latency_summary.csv"
        with latency_summary_file.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=_LATENCY_CSV_FIELDS)
            writer.writeheader()
            for result in model_results:
                if "error" in result:
                    continue
                for batch_size, stats in result["batch_sizes"].items():
                    if "error" not in stats:
                        writer.writerow(
                            {
                                "condition_key": result["condition_key"],
                                "batch_size": batch_size,
                                "mean_latency_ms": stats["mean_latency_ms"],
                                "median_latency_ms": stats["median_latency_ms"],
                            }
                        )

    def _write_manifest(
        self, benchmark_root: Path, manifest_data: list[dict[str, Any]]
    ) -> None:
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
        re_run: bool = False,
    ) -> list[dict[str, Any]]:
        model_dir = benchmark_root / model_key
        model_dir.mkdir(parents=True, exist_ok=True)
        model_results = []

        for condition_key in condition_keys:
            counter[0] += 1
            result_file = model_dir / f"{condition_key}.json"
            if not re_run and result_file.exists():
                _log(
                    f"[{counter[0]}/{total_benchmarks}] {model_key} / {condition_key} — already benchmarked, skipping."
                )
                model_results.append(json.loads(result_file.read_text()))
                continue
            _log(f"[{counter[0]}/{total_benchmarks}] {model_key} / {condition_key}…")
            result = self.benchmark_condition(
                model_key, condition_key, batch_sizes, num_runs
            )
            model_results.append(result)
            result_file.write_text(json.dumps(result, indent=2))

        self._write_latency_summary(model_dir, model_results)
        return model_results

    def benchmark_all(
        self,
        model_keys: list[str] | None = None,
        condition_keys: list[str] | None = None,
        batch_sizes: list[int] | None = None,
        num_runs: int = 5,
        benchmark_root: Path | str = "benchmarks",
        comparison_root: Path | str | None = None,
        re_run: bool = False,
    ) -> None:
        if batch_sizes is None:
            batch_sizes = [1, 32]
        if model_keys is None:
            model_keys = [spec.key for spec in MODEL_SPECS]
        if condition_keys is None:
            condition_keys = [spec.key for spec in CONDITION_SPECS]

        benchmark_root = Path(benchmark_root)
        benchmark_root.mkdir(parents=True, exist_ok=True)
        (benchmark_root / "computer_info.json").write_text(
            json.dumps(get_system_info(), indent=2)
        )

        total_benchmarks = len(model_keys) * len(condition_keys)
        counter = [0]
        manifest_data: list[dict[str, Any]] = []

        for model_key in model_keys:
            model_results = self._benchmark_model(
                model_key,
                condition_keys,
                batch_sizes,
                num_runs,
                benchmark_root,
                total_benchmarks,
                counter,
                re_run=re_run,
            )
            for result in model_results:
                manifest_data.extend(self._collect_manifest_rows(result, model_key))

        self._write_manifest(benchmark_root, manifest_data)
        if comparison_root is not None:
            from .results import write_per_model_benchmark_plots

            write_per_model_benchmark_plots(benchmark_root, Path(comparison_root))
        _log(f"Benchmarking complete. Results written to {benchmark_root}")
