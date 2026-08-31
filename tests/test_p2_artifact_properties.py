"""Property tests for the artifact manifest that decides what is reportable.

`test_p0_validity.py` pins the specific statuses the reporting layer accepts.
These tests attack the manifest itself: whatever file is touched, whichever
identity field disagrees, whichever stage the artifact was interrupted in, the
verdict has to fail closed.  They are written as loops over the file inventory
and the condition registry rather than as one example each, so a new
manifest-owned file or a new condition is covered the day it is added.
"""

import csv
import json
import tempfile
import unittest
from pathlib import Path

import torch

from dendritic_benchmark.artifacts import (
    ARTIFACT_MANIFEST_NAME,
    ARTIFACT_MANIFEST_SCHEMA_VERSION,
    finalize_artifact_manifest,
    validate_artifact_manifest,
    write_artifact_manifest,
)
from dendritic_benchmark.results import _record_is_reportable
from dendritic_benchmark.specs import CONDITION_SPECS
from dendritic_benchmark.training import QUANTIZATION_EVALUATION_REVISION

_MANIFEST_OWNED_FILES = (
    "model.pt",
    "metrics.json",
    "history.csv",
    "record.json",
    "record.csv",
    "best_model_stats.csv",
)


def _seal(
    condition_dir: Path,
    *,
    model_key: str = "lenet5",
    condition_key: str = "dendrites_fp32",
    dendrite_status: str = "verified_retained",
    quantization_revision: str | None = QUANTIZATION_EVALUATION_REVISION,
    seed: int | None = 0,
    metric_value: float = 0.5,
    finalize: bool = True,
) -> dict[str, object]:
    """Write one complete, sealed artifact and return its record."""
    artifact_id = f"artifact-{model_key}-{condition_key}-{seed}"
    condition_dir.mkdir(parents=True, exist_ok=True)
    torch.save(torch.nn.Linear(2, 1).state_dict(), condition_dir / "model.pt")
    (condition_dir / "metrics.json").write_text(json.dumps({"artifact_id": artifact_id}))
    (condition_dir / "history.csv").write_text("epoch\n1\n")
    record: dict[str, object] = {
        "artifact_id": artifact_id,
        "artifact_dir": str(condition_dir),
        "model_key": model_key,
        "condition_key": condition_key,
        "metric_name": "Accuracy",
        "metric_value": metric_value,
        "dendrite_audit_status": dendrite_status,
    }
    (condition_dir / "record.json").write_text(json.dumps(record))
    with (condition_dir / "record.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(record))
        writer.writeheader()
        writer.writerow(record)
    (condition_dir / "best_model_stats.csv").write_text("metric_value\n0.5\n")
    quantized = not condition_key.endswith("fp32")
    write_artifact_manifest(
        condition_dir,
        artifact_id=artifact_id,
        identity={
            "model_key": model_key,
            "condition_key": condition_key,
            "seed": seed,
            "quantization_evaluation_revision": quantization_revision,
        },
        pai_save_name=f"{model_key}_{condition_key}",
        validity={
            "dendrite_status": dendrite_status,
            "quantization_status": (
                "not_applicable"
                if not quantized
                else (
                    "current"
                    if quantization_revision == QUANTIZATION_EVALUATION_REVISION
                    else "unknown"
                )
            ),
        },
    )
    if finalize:
        finalize_artifact_manifest(condition_dir, artifact_id=artifact_id)
    return record


def _rewrite_manifest(condition_dir: Path, **changes: object) -> None:
    path = condition_dir / ARTIFACT_MANIFEST_NAME
    manifest = json.loads(path.read_text())
    manifest.update(changes)
    path.write_text(json.dumps(manifest))


class ArtifactPropertyTests(unittest.TestCase):
    def test_a_sealed_artifact_owns_every_file_reporting_reads(self) -> None:
        with tempfile.TemporaryDirectory() as root:  # type: ignore[no-matching-overload]
            condition_dir = Path(root) / "dendrites_fp32"
            _seal(condition_dir)
            verdict = validate_artifact_manifest(condition_dir)
            self.assertTrue(verdict.valid, verdict.reason)
            self.assertIsNotNone(verdict.manifest)
            assert verdict.manifest is not None
            self.assertEqual(
                set(_MANIFEST_OWNED_FILES) - set(verdict.manifest["files"]), set()
            )
            self.assertEqual(verdict.manifest["state"], "complete")

    def test_touching_any_manifest_owned_file_invalidates_the_artifact(self) -> None:
        for name in _MANIFEST_OWNED_FILES:
            with self.subTest(file=name), tempfile.TemporaryDirectory() as root:  # type: ignore[no-matching-overload]
                condition_dir = Path(root) / "dendrites_fp32"
                record = _seal(condition_dir)
                target = condition_dir / name
                target.write_bytes(target.read_bytes() + b"\n")
                verdict = validate_artifact_manifest(condition_dir)
                self.assertEqual(verdict.status, "invalid", name)
                self.assertFalse(_record_is_reportable(record))

    def test_removing_any_manifest_owned_file_invalidates_the_artifact(self) -> None:
        for name in _MANIFEST_OWNED_FILES:
            with self.subTest(file=name), tempfile.TemporaryDirectory() as root:  # type: ignore[no-matching-overload]
                condition_dir = Path(root) / "dendrites_fp32"
                record = _seal(condition_dir)
                (condition_dir / name).unlink()
                self.assertEqual(
                    validate_artifact_manifest(condition_dir).status, "invalid", name
                )
                self.assertFalse(_record_is_reportable(record))

    def test_an_unfinalized_or_missing_manifest_is_unknown_not_valid(self) -> None:
        with tempfile.TemporaryDirectory() as root:  # type: ignore[no-matching-overload]
            unfinalized = Path(root) / "unfinalized"
            record = _seal(unfinalized, finalize=False)
            verdict = validate_artifact_manifest(unfinalized)
            self.assertEqual(verdict.status, "unknown")
            self.assertFalse(_record_is_reportable(record))

            missing = Path(root) / "missing"
            missing.mkdir()
            self.assertEqual(validate_artifact_manifest(missing).status, "unknown")

    def test_every_identity_field_reporting_trusts_is_checked(self) -> None:
        expectations = (
            ("expected_artifact_id", "someone-elses-artifact"),
            ("expected_model_key", "resnet18_cifar10"),
            ("expected_condition_key", "base_q8"),
        )
        for field, wrong_value in expectations:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as root:  # type: ignore[no-matching-overload]
                condition_dir = Path(root) / "dendrites_fp32"
                _seal(condition_dir)
                verdict = validate_artifact_manifest(
                    condition_dir, **{field: wrong_value}
                )
                self.assertEqual(verdict.status, "invalid", field)

    def test_a_corrupted_manifest_is_never_silently_accepted(self) -> None:
        corruptions = {
            "schema": {"schema_version": ARTIFACT_MANIFEST_SCHEMA_VERSION + 1},
            "artifact_id": {"artifact_id": ""},
            "identity": {"identity": "not-a-mapping"},
            "files": {"files": []},
            "traversal": {
                "files": {"../escape.json": {"sha256": "0" * 64}},
            },
            "incomplete_inventory": {"files": {"model.pt": {"sha256": "0" * 64}}},
        }
        for name, changes in corruptions.items():
            with self.subTest(corruption=name), tempfile.TemporaryDirectory() as root:  # type: ignore[no-matching-overload]
                condition_dir = Path(root) / "dendrites_fp32"
                _seal(condition_dir)
                _rewrite_manifest(condition_dir, **changes)
                self.assertEqual(
                    validate_artifact_manifest(condition_dir).status, "invalid", name
                )

        with tempfile.TemporaryDirectory() as root:  # type: ignore[no-matching-overload]
            condition_dir = Path(root) / "dendrites_fp32"
            _seal(condition_dir)
            (condition_dir / ARTIFACT_MANIFEST_NAME).write_text("{not json")
            self.assertEqual(
                validate_artifact_manifest(condition_dir).status, "invalid"
            )

    def test_finalizing_refuses_a_foreign_or_incomplete_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as root:  # type: ignore[no-matching-overload]
            condition_dir = Path(root) / "dendrites_fp32"
            _seal(condition_dir, finalize=False)
            with self.assertRaises(RuntimeError):
                finalize_artifact_manifest(condition_dir, artifact_id="a-different-id")

            (condition_dir / "record.csv").unlink()
            with self.assertRaises(RuntimeError):
                finalize_artifact_manifest(
                    condition_dir,
                    artifact_id="artifact-lenet5-dendrites_fp32-0",
                )

            empty = Path(root) / "empty"
            empty.mkdir()
            with self.assertRaises(RuntimeError):
                finalize_artifact_manifest(empty, artifact_id="anything")

    def test_reportability_holds_for_every_condition_in_the_registry(self) -> None:
        with tempfile.TemporaryDirectory() as root:  # type: ignore[no-matching-overload]
            root_path = Path(root)
            for condition in CONDITION_SPECS:
                with self.subTest(condition=condition.key):
                    current = _seal(
                        root_path / f"current_{condition.key}",
                        condition_key=condition.key,
                        dendrite_status=(
                            "verified_retained"
                            if condition.use_dendrites
                            else "not_applicable"
                        ),
                    )
                    self.assertTrue(_record_is_reportable(current))

                    if condition.use_dendrites:
                        unverified = _seal(
                            root_path / f"unverified_{condition.key}",
                            condition_key=condition.key,
                            dendrite_status="legacy_unchecked",
                        )
                        self.assertFalse(_record_is_reportable(unverified))

                    stale = _seal(
                        root_path / f"stale_{condition.key}",
                        condition_key=condition.key,
                        dendrite_status=(
                            "verified_retained"
                            if condition.use_dendrites
                            else "not_applicable"
                        ),
                        quantization_revision="an_older_projection_revision",
                    )
                    # A stale quantization revision may only cost a quantized
                    # arm its reportability; FP32 arms are unaffected.
                    self.assertEqual(
                        _record_is_reportable(stale), not condition.quantized
                    )


if __name__ == "__main__":
    unittest.main()
