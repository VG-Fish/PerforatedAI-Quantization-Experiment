import csv
import json
import tempfile
import unittest
from pathlib import Path

import torch

from dendritic_benchmark.artifacts import (
    finalize_artifact_manifest,
    write_artifact_manifest,
)
from dendritic_benchmark.checkpointing import (
    CheckpointMismatchError,
    inspect_state_dict,
    load_state_dict_checked,
)
from dendritic_benchmark.pipeline import BenchmarkRunner
from dendritic_benchmark.results import (
    _dendrite_audit_status,
    _quantization_evaluation_status,
    _record_is_reportable,
)
from dendritic_benchmark.training import QUANTIZATION_EVALUATION_REVISION


def _sealed_record(
    condition_dir: Path,
    *,
    condition_key: str,
    dendrite_status: str,
    quantization_revision: str | None = None,
) -> dict[str, object]:
    artifact_id = f"artifact-{condition_key}-{dendrite_status}"
    condition_dir.mkdir(parents=True, exist_ok=True)
    torch.save(torch.nn.Linear(2, 1).state_dict(), condition_dir / "model.pt")
    (condition_dir / "metrics.json").write_text(
        json.dumps(
            {
                "artifact_id": artifact_id,
                "quantization_evaluation_revision": quantization_revision,
            }
        )
    )
    (condition_dir / "history.csv").write_text("epoch\n1\n")
    record: dict[str, object] = {
        "artifact_id": artifact_id,
        "artifact_dir": str(condition_dir),
        "model_key": "lenet5",
        "condition_key": condition_key,
        "dendrite_audit_status": dendrite_status,
    }
    (condition_dir / "record.json").write_text(json.dumps(record))
    with (condition_dir / "record.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(record))
        writer.writeheader()
        writer.writerow(record)
    (condition_dir / "best_model_stats.csv").write_text("metric_value\n0.5\n")
    quantized = not condition_key.endswith("fp32")
    quantization_status = (
        "current"
        if quantized and quantization_revision == QUANTIZATION_EVALUATION_REVISION
        else ("unknown" if quantized else "not_applicable")
    )
    write_artifact_manifest(
        condition_dir,
        artifact_id=artifact_id,
        identity={
            "model_key": "lenet5",
            "condition_key": condition_key,
            "quantization_evaluation_revision": quantization_revision,
        },
        pai_save_name=f"lenet5_{condition_key}_{artifact_id}",
        validity={
            "dendrite_status": dendrite_status,
            "quantization_status": quantization_status,
        },
    )
    finalize_artifact_manifest(condition_dir, artifact_id=artifact_id)
    return record


class P0ValidityTests(unittest.TestCase):
    def test_dendritic_reportability_requires_only_verified_statuses(self) -> None:
        statuses = {
            "verified_retained": True,
            "inherited_verified_retained": True,
            "no_retained_insertion": False,
            "inherited_no_retained_insertion": False,
            "unverified": False,
            "inherited_unverified": False,
            "legacy_unchecked": False,
            "unknown": False,
            "invalid": False,
        }
        with tempfile.TemporaryDirectory() as root:  # type: ignore[no-matching-overload]
            root_path = Path(root)
            for status, expected in statuses.items():
                with self.subTest(status=status):
                    record = _sealed_record(
                        root_path / status,
                        condition_key="dendrites_fp32",
                        dendrite_status=status,
                    )
                    self.assertEqual(_record_is_reportable(record), expected)

    def test_legacy_or_tampered_artifacts_fail_closed(self) -> None:
        legacy = {
            "artifact_dir": "/missing",
            "model_key": "lenet5",
            "condition_key": "dendrites_fp32",
        }
        self.assertEqual(_dendrite_audit_status(legacy), "unknown")
        self.assertFalse(_record_is_reportable(legacy))

        with tempfile.TemporaryDirectory() as root:  # type: ignore[no-matching-overload]
            condition_dir = Path(root) / "dendrites_fp32"
            record = _sealed_record(
                condition_dir,
                condition_key="dendrites_fp32",
                dendrite_status="verified_retained",
            )
            self.assertTrue(_record_is_reportable(record))
            (condition_dir / "metrics.json").write_text("{}")
            self.assertFalse(_record_is_reportable(record))

    def test_quantized_dendrite_requires_current_projection_revision(self) -> None:
        with tempfile.TemporaryDirectory() as root:  # type: ignore[no-matching-overload]
            root_path = Path(root)
            current = _sealed_record(
                root_path / "current",
                condition_key="dendrites_q8",
                dendrite_status="inherited_verified_retained",
                quantization_revision=QUANTIZATION_EVALUATION_REVISION,
            )
            self.assertEqual(_quantization_evaluation_status(current), "current")
            self.assertTrue(_record_is_reportable(current))

            stale = _sealed_record(
                root_path / "stale",
                condition_key="dendrites_q8",
                dendrite_status="inherited_verified_retained",
                quantization_revision="double_projection_v0",
            )
            self.assertEqual(_quantization_evaluation_status(stale), "invalid_revision")
            self.assertFalse(_record_is_reportable(stale))

    def test_checkpoint_report_is_bidirectional_and_load_is_atomic(self) -> None:
        model = torch.nn.Linear(2, 1)
        original_weight = model.weight.detach().clone()
        incomplete = {"weight": torch.ones_like(model.weight)}
        report = inspect_state_dict(model.state_dict(), incomplete)
        self.assertEqual(report.missing, ("bias",))
        self.assertFalse(report.complete)
        with self.assertRaises(CheckpointMismatchError) as raised:
            load_state_dict_checked(model, incomplete, context="test checkpoint")
        self.assertEqual(raised.exception.report.missing, ("bias",))
        torch.testing.assert_close(model.weight, original_weight)

        bias = model.bias
        assert isinstance(bias, torch.Tensor)
        complete = {
            "weight": torch.ones_like(model.weight),
            "bias": torch.zeros_like(bias),
        }
        loaded = load_state_dict_checked(model, complete)
        self.assertTrue(loaded.complete)
        torch.testing.assert_close(model.weight, complete["weight"])

    def test_resume_reuses_only_the_persisted_attempt_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as root:  # type: ignore[no-matching-overload]
            runner = BenchmarkRunner(results_root=Path(root) / "results")
            condition_dir = Path(root) / "results" / "lenet5" / "dendrites_fp32"
            first = runner._artifact_attempt(condition_dir, "lenet5", "dendrites_fp32")
            (condition_dir / "epoch_checkpoint.pt").write_bytes(b"checkpoint")
            self.assertEqual(
                runner._artifact_attempt(condition_dir, "lenet5", "dendrites_fp32"),
                first,
            )
            (condition_dir / "artifact_attempt.json").unlink()
            with self.assertRaisesRegex(RuntimeError, "use --fresh"):
                runner._artifact_attempt(condition_dir, "lenet5", "dendrites_fp32")


if __name__ == "__main__":
    unittest.main()
