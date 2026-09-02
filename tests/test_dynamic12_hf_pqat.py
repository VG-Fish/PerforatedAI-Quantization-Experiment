import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import torch

from dendritic_benchmark.artifacts import (
    finalize_artifact_manifest,
    write_artifact_manifest,
)
from dendritic_benchmark.data import DATA_PIPELINE_REVISION
from dendritic_benchmark.models import (
    HF_PERFORATED_RESNET18_REPO_ID,
    HF_PERFORATED_RESNET18_SHA256,
    build_model,
)
from dendritic_benchmark.pipeline import BenchmarkRunner
from dendritic_benchmark.specs import (
    CONDITION_SPECS,
    condition_by_key,
    condition_supported_by_model,
)
from dendritic_benchmark.training import QUANTIZATION_EVALUATION_REVISION

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HF_CHECKPOINT = (
    PROJECT_ROOT
    / "data"
    / "huggingface"
    / "perforated-ai"
    / "resnet-18-perforated-gd"
    / "model.safetensors"
)


#: The dynamic12 sweep's own post-run checker. It lives with that sweep's
#: scripts, which are deleted once the sweep is archived, so the test that
#: exercises it skips rather than fails when the tree is gone.
VERIFY_PQAT_SCRIPT = (
    PROJECT_ROOT / "experiments" / "dynamic12" / "scripts" / "verify_pqat.py"
)


def _load_verifier() -> Any:
    path = VERIFY_PQAT_SCRIPT
    spec = importlib.util.spec_from_file_location("dynamic12_verify_pqat", path)
    if spec is None:
        raise RuntimeError(f"could not load {path}")
    loader = spec.loader
    if loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


class Dynamic12HFPQATTests(unittest.TestCase):
    @unittest.skipUnless(HF_CHECKPOINT.exists(), "HF checkpoint is not cached")
    def test_published_hf_model_preserves_dendrites_and_adapts_cifar_io(self) -> None:
        model = build_model("resnet18_hf_perforated_cifar10")

        self.assertEqual(model.hf_repo_id, HF_PERFORATED_RESNET18_REPO_ID)
        self.assertEqual(model.hf_checkpoint_sha256, HF_PERFORATED_RESNET18_SHA256)
        self.assertEqual(tuple(model.conv1.weight.shape), (64, 3, 3, 3))
        self.assertEqual(model.conv1.stride, (1, 1))
        self.assertIsInstance(model.maxpool, torch.nn.Identity)
        self.assertEqual(model.fc.out_features, 10)
        self.assertEqual(int(model.pre_fc.num_cycles), 8)
        self.assertEqual(len(model.pre_fc.layer_array), 5)
        self.assertEqual(len(model.pre_fc.skip_weights), 4)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        self.assertEqual(parameter_count, 12_492_362)

        model.eval()
        with torch.no_grad():
            output = model(torch.randn(2, 3, 32, 32))
        self.assertEqual(tuple(output.shape), (2, 10))
        self.assertTrue(torch.isfinite(output).all())

    def test_every_supported_quantized_condition_gets_a_pqat_plan(self) -> None:
        with tempfile.TemporaryDirectory() as root:  # type: ignore[no-matching-overload]
            runner = BenchmarkRunner(results_root=Path(root) / "results")
            for model_key in (
                "resnet18_hf_perforated_cifar10",
                "saint_adult",
                "pointnet_modelnet40",
            ):
                for condition in CONDITION_SPECS:
                    if not condition.quantized or not condition_supported_by_model(
                        model_key, condition.key
                    ):
                        continue
                    recipe = runner._training_hyperparameters(model_key, condition)
                    plan = runner._condition_training_plan(
                        model_key,
                        condition,
                        recipe,
                        allow_pqat=True,
                    )
                    self.assertTrue(plan.use_qat, f"{model_key}/{condition.key}")
                    self.assertGreater(plan.fine_tune_epochs, 0)
                    self.assertEqual(plan.max_epochs, plan.fine_tune_epochs)

    def test_hf_model_rejects_redundant_second_dendrite_graph(self) -> None:
        model_key = "resnet18_hf_perforated_cifar10"
        self.assertTrue(condition_supported_by_model(model_key, "base_q1"))
        self.assertFalse(condition_supported_by_model(model_key, "dendrites_fp32"))
        self.assertFalse(condition_supported_by_model(model_key, "dendrites_q1"))

    def test_pqat_artifact_reuse_requires_both_valid_stage_snapshots(self) -> None:
        with tempfile.TemporaryDirectory() as root:  # type: ignore[no-matching-overload]
            root_path = Path(root)
            runner = BenchmarkRunner(results_root=root_path / "results")
            condition = condition_by_key("base_q8")
            condition_dir = root_path / "results" / "saint_adult" / condition.key
            condition_dir.mkdir(parents=True)
            recipe = runner._training_hyperparameters("saint_adult", condition)
            metadata = {
                "artifact_id": "test-artifact",
                "model_scale": 1.0,
                "model_revision": runner._model_artifact_revision("saint_adult"),
                "dataset_revision": DATA_PIPELINE_REVISION,
                "lr_schedule_epochs": recipe.lr_schedule_epochs,
                "quantization_granularity": "tensor",
                "quantization_evaluation_revision": QUANTIZATION_EVALUATION_REVISION,
                "use_qat": True,
                "fine_tune_epochs": 10,
            }
            (condition_dir / "metrics.json").write_text(json.dumps(metadata))
            (condition_dir / "model.pt").write_bytes(b"test checkpoint")
            (condition_dir / "history.csv").write_text("epoch\n1\n")
            for stage_name, use_qat in (("before_pqat", False), ("after_pqat", True)):
                stage_dir = condition_dir / stage_name
                stage_dir.mkdir()
                (stage_dir / "metrics.json").write_text(
                    json.dumps({"use_qat": use_qat})
                )
            record = {
                "artifact_id": "test-artifact",
                "model_key": "saint_adult",
                "condition_key": condition.key,
            }
            (condition_dir / "record.json").write_text(json.dumps(record))
            (condition_dir / "record.csv").write_text(
                "artifact_id,model_key,condition_key\n"
                f"test-artifact,saint_adult,{condition.key}\n"
            )
            (condition_dir / "best_model_stats.csv").write_text("metric_value\n0.5\n")
            write_artifact_manifest(
                condition_dir,
                artifact_id="test-artifact",
                identity={
                    "model_key": "saint_adult",
                    "condition_key": condition.key,
                },
                pai_save_name=None,
                validity={
                    "dendrite_status": "not_applicable",
                    "quantization_status": "current",
                },
            )
            finalize_artifact_manifest(
                condition_dir, artifact_id="test-artifact"
            )

            self.assertTrue(
                runner._condition_metadata_current(
                    "saint_adult",
                    condition,
                    condition_dir,
                    allow_pqat=True,
                )
            )
            (condition_dir / "after_pqat" / "metrics.json").unlink()
            self.assertFalse(
                runner._condition_metadata_current(
                    "saint_adult",
                    condition,
                    condition_dir,
                    allow_pqat=True,
                )
            )

    @unittest.skipUnless(
        VERIFY_PQAT_SCRIPT.exists(), "dynamic12 sweep scripts are not checked out"
    )
    def test_post_run_verifier_rejects_missing_pqat_stage(self) -> None:
        verifier = _load_verifier()
        with tempfile.TemporaryDirectory() as root:  # type: ignore[no-matching-overload]
            results_root = Path(root)
            artifact_dir = results_root / "saint_adult" / "base_q1"
            (artifact_dir / "before_pqat").mkdir(parents=True)
            (artifact_dir / "after_pqat").mkdir()
            (artifact_dir / "metrics.json").write_text(
                json.dumps({"use_qat": True, "fine_tune_epochs": 10})
            )
            (artifact_dir / "record.json").write_text(
                json.dumps({"training_skipped": False})
            )
            (artifact_dir / "before_pqat" / "metrics.json").write_text(
                json.dumps({"use_qat": False})
            )
            (artifact_dir / "after_pqat" / "metrics.json").write_text(
                json.dumps({"use_qat": True})
            )

            self.assertEqual(
                verifier.verify_pqat(
                    results_root,
                    ["saint_adult"],
                    ["base_q1"],
                ),
                [],
            )
            (artifact_dir / "after_pqat" / "metrics.json").unlink()
            failures = verifier.verify_pqat(
                results_root,
                ["saint_adult"],
                ["base_q1"],
            )
            self.assertEqual(len(failures), 1)
            self.assertIn("missing artifact", failures[0])


if __name__ == "__main__":
    unittest.main()
