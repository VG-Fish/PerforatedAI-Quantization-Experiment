#!/usr/bin/env python3
"""Validate every Dynamic12 PerforatedAI target before an expensive PAI run.

The check verifies that each configured target exists, is eligible for PAI,
does not overlap a track-only module, and receives a concrete inferred output
dimension vector from a real task batch. It intentionally does not call
``perforate_model``: that operation initializes PAI's on-disk state and can
start candidate bookkeeping that belongs in the full benchmark.
"""

from __future__ import annotations

import argparse
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch.nn as nn

from dendritic_benchmark.data import build_task_bundle
from dendritic_benchmark.models import build_model
from dendritic_benchmark.pipeline import BenchmarkRunner
from dendritic_benchmark.specs import condition_by_key
from dendritic_benchmark.training import infer_module_output_dimensions


TARGET_CASES = (
    ("resnet18_cifar10", "default"),
    ("saint_adult", "default"),
    ("pointnet_modelnet40", "default"),
)
PAI_ELIGIBLE_TYPES = (nn.Linear, nn.Conv1d, nn.Conv2d)


def _normalise(module_id: str) -> str:
    return module_id.lstrip(".")


def _target_cases(selected_models: Iterable[str]) -> tuple[tuple[str, str], ...]:
    selected = set(selected_models)
    return tuple(case for case in TARGET_CASES if case[0] in selected)


def smoke_case(model_key: str, pai_variant: str, model_scale: float, root: Path) -> dict[str, Any]:
    case_root = root / f"{model_key}_{pai_variant}"
    runner = BenchmarkRunner(
        results_root=case_root / "results",
        comparison_root=case_root / "comparison",
        model_scale=model_scale,
        pai_variant=pai_variant,
    )
    recipe = runner._training_hyperparameters(model_key, condition_by_key("dendrites_fp32"))
    model = build_model(model_key, model_scale=model_scale)
    named_modules = dict(model.named_modules())
    targets = runner._perforation_module_ids_to_perforate(model_key)
    tracked_only = runner._perforation_track_only_module_ids(model_key)
    if not targets:
        raise AssertionError(f"{model_key}/{pai_variant} unexpectedly has no PAI targets")
    overlap = set(targets) & set(tracked_only)
    if overlap:
        raise AssertionError(f"{model_key}/{pai_variant} targets also marked track-only: {overlap}")
    unavailable = [target for target in targets if _normalise(target) not in named_modules]
    if unavailable:
        raise AssertionError(f"{model_key}/{pai_variant} missing targets: {unavailable}")
    ineligible = [
        target
        for target in targets
        if not isinstance(named_modules[_normalise(target)], PAI_ELIGIBLE_TYPES)
    ]
    if ineligible:
        raise AssertionError(f"{model_key}/{pai_variant} has non-eligible targets: {ineligible}")

    bundle = build_task_bundle(model_key, batch_size=recipe.batch_size)
    dimensions = infer_module_output_dimensions(
        model,
        model_key,
        bundle,
        [],
        module_names=targets,
    )
    missing_dimensions = [target for target in targets if target not in dimensions]
    if missing_dimensions:
        raise AssertionError(
            f"{model_key}/{pai_variant} did not infer target dimensions for {missing_dimensions}; "
            f"got {dimensions}"
        )
    malformed = {
        target: dimensions[target]
        for target in targets
        if not dimensions[target] or 0 not in dimensions[target]
    }
    if malformed:
        raise AssertionError(f"{model_key}/{pai_variant} malformed target dimensions: {malformed}")
    return {"model": model_key, "variant": pai_variant, "targets": targets, "dimensions": dimensions}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("resnet18_cifar10", "saint_adult", "pointnet_modelnet40"),
        default=("resnet18_cifar10", "saint_adult", "pointnet_modelnet40"),
    )
    parser.add_argument("--model-scale", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 < args.model_scale <= 1:
        raise ValueError("--model-scale must be greater than zero and at most one")
    cases = _target_cases(args.models)
    with tempfile.TemporaryDirectory(prefix="dynamic12_pai_target_smoke_") as temporary_directory:  # type: ignore[no-matching-overload]
        root = Path(temporary_directory)
        print(f"Dynamic12 PAI target smoke test (scale={args.model_scale})")
        print("model             variant            target -> inferred output dimensions")
        for model_key, pai_variant in cases:
            result = smoke_case(model_key, pai_variant, args.model_scale, root)
            rendered = ", ".join(
                f"{target}:{result['dimensions'][target]}" for target in result["targets"]
            )
            print(f"{model_key:<17} {pai_variant:<18} {rendered}")
    print("PASS: every configured PAI target exists, is eligible, and has output dimensions.")


if __name__ == "__main__":
    main()
