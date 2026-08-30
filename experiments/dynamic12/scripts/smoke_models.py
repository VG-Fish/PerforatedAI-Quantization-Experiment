#!/usr/bin/env python3
"""Fast end-to-end smoke test for Dynamic12's ResNet, SAINT, and PointNet runs.

This deliberately uses one real training batch for each model.  It checks the
same forward/loss/backward path and PTQ helper that the benchmark runner uses,
without creating checkpoints or launching an epoch-level benchmark.
"""

from __future__ import annotations

import argparse
import copy
import random
import tempfile
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from dendritic_benchmark.compat import choose_device
from dendritic_benchmark.data import build_task_bundle
from dendritic_benchmark.models import build_model
from dendritic_benchmark.pipeline import BenchmarkRunner
from dendritic_benchmark.specs import condition_by_key
from dendritic_benchmark.training import (
    TrainingConfig,
    _binary_or_multi_loss,
    _compute_loss,
    _finalize_quantized_model_for_eval,
    _forward,
    _make_quantized_copy,
    _move_batch_to_device,
    _qat_init_shadow,
    _run_training_batch,
)

DEFAULT_MODELS = (
    "resnet18_cifar10",
    "resnet18_hf_perforated_cifar10",
    "saint_adult",
    "pointnet_modelnet40",
)
QUANTIZATION_CASES = (
    ("fp32", None, None, "tensor"),
    ("q8", 8, None, "tensor"),
    ("q4", 4, None, "tensor"),
    ("q2", 2, None, "tensor"),
    ("q1.58", 2, "ternary", "tensor"),
    ("q1", 1, "binary", "tensor"),
)


def _all_tensors(value: Any) -> Iterable[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _all_tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _all_tensors(item)


def _require_finite(label: str, value: Any) -> None:
    tensors = list(_all_tensors(value))
    if not tensors:
        raise AssertionError(f"{label} did not contain a tensor")
    bad = [
        tuple(tensor.shape) for tensor in tensors if not torch.isfinite(tensor).all()
    ]
    if bad:
        raise AssertionError(f"{label} contains non-finite values in tensors {bad}")


def _tensor_signature(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return tuple(value.shape)
    if isinstance(value, tuple):
        return tuple(_tensor_signature(item) for item in value)
    if isinstance(value, list):
        return [_tensor_signature(item) for item in value]
    if isinstance(value, dict):
        return {key: _tensor_signature(item) for key, item in value.items()}
    return type(value).__name__


def _training_config(model_key: str, runner: BenchmarkRunner) -> TrainingConfig:
    recipe = runner._training_hyperparameters(model_key, condition_by_key("base_fp32"))
    return TrainingConfig(
        learning_rate=recipe.learning_rate,
        optimizer_name=recipe.optimizer_name,
        momentum=recipe.momentum,
        weight_decay=recipe.weight_decay,
        regression_loss=recipe.regression_loss,
        grad_clip_norm=recipe.grad_clip_norm,
        quantization_granularity=runner._quantization_granularity(
            model_key, condition_by_key("base_q1_58")
        ),
    )


def _qat_finalization_smoke(
    model_key: str,
    model: torch.nn.Module,
    batch: tuple[Any, ...],
    config: TrainingConfig,
    device: torch.device,
) -> str:
    """Exercise one ternary QAT batch and assert finalization is idempotent.

    Validation occurs after the epoch loop leaves projected weights in ``.data``.
    The final artifact must retain exactly those weights instead of recalibrating
    a second ternary grid before test evaluation.
    """
    qat_model = copy.deepcopy(model).to(device)
    qat_granularity = config.quantization_granularity
    qat_config = TrainingConfig(
        bit_width=2,
        quantization_mode="ternary",
        quantization_granularity=qat_granularity,
        use_qat=True,
        learning_rate=config.learning_rate,
        optimizer_name="adam",
        regression_loss=config.regression_loss,
        grad_clip_norm=config.grad_clip_norm,
    )
    _qat_init_shadow(qat_model)
    _make_quantized_copy(
        qat_model,
        2,
        mode="ternary",
        granularity=qat_granularity,
    )
    optimizer = torch.optim.Adam(qat_model.parameters(), lr=qat_config.learning_rate)
    criterion = _binary_or_multi_loss(model_key, qat_config)
    qat_model.train()
    _run_training_batch(
        model=qat_model,
        model_key=model_key,
        batch=batch,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        config=qat_config,
        clear_pai_buffers=False,
        retain_graph_for_optimizer_step=False,
    )
    projected_state = {
        name: value.detach().clone() for name, value in qat_model.state_dict().items()
    }
    finalized = _finalize_quantized_model_for_eval(qat_model, qat_config)
    for name, value in finalized.state_dict().items():
        if not torch.equal(projected_state[name], value):
            raise AssertionError(
                f"{model_key} QAT finalization changed already-projected parameter {name}"
            )
    finalized.eval()
    with torch.no_grad():
        outputs, _, _ = _forward(model_key, finalized, batch)
    _require_finite(f"{model_key} QAT final outputs", outputs)
    return "q1.58 QAT single-projection"


def smoke_model(
    model_key: str, model_scale: float, device: torch.device, smoke_root: Path
) -> dict[str, Any]:
    # The runner is used only to keep the smoke batch-size and loss recipe in
    # lockstep with the actual benchmark. Its output roots are never exercised.
    runner = BenchmarkRunner(
        results_root=smoke_root / model_key / "results",
        comparison_root=smoke_root / model_key / "comparison",
        model_scale=model_scale,
    )
    config = _training_config(model_key, runner)
    recipe = runner._training_hyperparameters(model_key, condition_by_key("base_fp32"))
    bundle = build_task_bundle(model_key, batch_size=recipe.batch_size)
    batch = _move_batch_to_device(next(iter(bundle.train_loader)), device)

    model = build_model(model_key, model_scale=model_scale).to(device)
    model.train()
    criterion = _binary_or_multi_loss(model_key, config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    outputs, targets, _ = _forward(model_key, model, batch)
    loss = _compute_loss(model_key, criterion, outputs, targets, model)
    _require_finite(f"{model_key} train outputs", outputs)
    _require_finite(f"{model_key} loss", loss)
    loss.backward()
    missing_grads = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
    if missing_grads:
        raise AssertionError(f"{model_key} did not backpropagate to {missing_grads}")
    _require_finite(
        f"{model_key} gradients",
        [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ],
    )
    optimizer.step()

    model.eval()
    with torch.no_grad():
        reference_outputs, _, _ = _forward(model_key, model, batch)
    _require_finite(f"{model_key} updated outputs", reference_outputs)
    reference_signature = _tensor_signature(reference_outputs)

    quantized_cases: list[str] = []
    with torch.no_grad():
        for label, bit_width, mode, granularity in QUANTIZATION_CASES:
            candidate = _make_quantized_copy(
                copy.deepcopy(model), bit_width, mode=mode, granularity=granularity
            )
            candidate.eval()
            candidate_outputs, _, _ = _forward(model_key, candidate, batch)
            _require_finite(f"{model_key} {label} outputs", candidate_outputs)
            if _tensor_signature(candidate_outputs) != reference_signature:
                raise AssertionError(
                    f"{model_key} {label} changed output signature "
                    f"{reference_signature!r} -> {_tensor_signature(candidate_outputs)!r}"
                )
            quantized_cases.append(label)
    quantized_cases.append(
        _qat_finalization_smoke(model_key, model, batch, config, device)
    )

    return {
        "model": model_key,
        "loss": float(loss.detach().cpu()),
        "output": reference_signature,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "quantization_cases": quantized_cases,
        "recipe": asdict(recipe),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", nargs="+", choices=DEFAULT_MODELS, default=DEFAULT_MODELS
    )
    parser.add_argument("--model-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.model_scale <= 1:
        raise ValueError("--model-scale must be greater than zero and at most one")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = choose_device()
    print(
        f"Dynamic12 model smoke test on {device} (scale={args.model_scale}, seed={args.seed})"
    )
    print(
        "model             loss         parameters  output                         PTQ"
    )
    with tempfile.TemporaryDirectory(
        prefix="dynamic12_model_smoke_"
    ) as temporary_directory:  # type: ignore[no-matching-overload]
        smoke_root = Path(temporary_directory)
        for model_key in args.models:
            result = smoke_model(model_key, args.model_scale, device, smoke_root)
            print(
                f"{result['model']:<17} {result['loss']:>10.5f} "
                f"{result['parameters']:>10d}  {result['output']!s:<30} "
                f"{', '.join(result['quantization_cases'])}"
            )
    print("PASS: forward, backward, optimizer, and all Dynamic12 PTQ paths are finite.")


if __name__ == "__main__":
    main()
