#!/usr/bin/env python3
"""Fail unless every requested Dynamic12 quantized artifact completed PQAT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dendritic_benchmark.specs import condition_supported_by_model


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise RuntimeError(f"missing artifact: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"invalid JSON artifact: {path}: {exc}") from exc


def _is_quantized_condition(condition: str) -> bool:
    return "_q" in condition


def verify_pqat(
    results_root: Path,
    models: list[str],
    conditions: list[str],
) -> list[str]:
    failures: list[str] = []
    checked = 0
    for model in models:
        for condition in conditions:
            quantized = _is_quantized_condition(condition)
            supported = condition_supported_by_model(model, condition)
            if not quantized or not supported:
                continue
            checked += 1
            artifact_dir = results_root / model / condition
            try:
                metrics = _read_json(artifact_dir / "metrics.json")
                record = _read_json(artifact_dir / "record.json")
                before = _read_json(artifact_dir / "before_pqat" / "metrics.json")
                after = _read_json(artifact_dir / "after_pqat" / "metrics.json")
            except RuntimeError as exc:
                failures.append(f"{model}/{condition}: {exc}")
                continue

            fine_tune_epochs = metrics.get("fine_tune_epochs", 0)
            if metrics.get("use_qat") is not True:
                failures.append(f"{model}/{condition}: metrics use_qat is not true")
            if not isinstance(fine_tune_epochs, int) or fine_tune_epochs <= 0:
                failures.append(
                    f"{model}/{condition}: invalid "
                    f"fine_tune_epochs={fine_tune_epochs!r}"
                )
            if record.get("training_skipped") is True:
                failures.append(f"{model}/{condition}: training was skipped")
            if before.get("use_qat") is not False:
                failures.append(
                    f"{model}/{condition}: before_pqat is not a PTQ snapshot"
                )
            if after.get("use_qat") is not True:
                failures.append(f"{model}/{condition}: after_pqat is not marked as QAT")

    if checked == 0:
        failures.append("no requested quantized conditions were found to verify")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--conditions", nargs="+", required=True)
    args = parser.parse_args()

    failures = verify_pqat(args.results_root, args.models, args.conditions)
    if failures:
        detail = "\n".join(f"  - {failure}" for failure in failures)
        raise SystemExit(f"Dynamic12 PQAT verification failed:\n{detail}")
    print("Dynamic12 PQAT verification passed for every quantized artifact.")


if __name__ == "__main__":
    main()
