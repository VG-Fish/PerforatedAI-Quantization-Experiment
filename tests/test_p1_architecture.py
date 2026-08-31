import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

import torch

from dendritic_benchmark.data import _read_off_mesh
from dendritic_benchmark.model_adapters import (
    ALL_MODEL_KEYS,
    DEFAULT_MODEL_KEYS,
    MODEL_ADAPTERS,
    model_adapter,
    selected_model_keys,
)
from dendritic_benchmark.quantization import (
    make_quantized_copy,
    qat_init_shadow,
    qat_project_for_forward,
    qat_restore_shadow_for_step,
    qat_sync_shadow_after_step,
)
from dendritic_benchmark.plans import ExperimentPlan
from dendritic_benchmark.specs import MODEL_SPECS
from dendritic_benchmark.training import TrainingConfig


class P1ArchitectureTests(unittest.TestCase):
    def test_every_registered_model_has_one_adapter(self) -> None:
        registered = tuple(spec.key for spec in MODEL_SPECS)
        self.assertEqual(ALL_MODEL_KEYS, registered)
        self.assertEqual(set(MODEL_ADAPTERS), set(registered))
        for key in registered:
            with self.subTest(key=key):
                self.assertEqual(model_adapter(key).spec.key, key)
                self.assertTrue(model_adapter(key).primary_metric_key)

    def test_default_roster_is_small_and_exploratory_models_are_opt_in(self) -> None:
        self.assertEqual(
            DEFAULT_MODEL_KEYS,
            (
                "lenet5",
                "tcn_forecaster",
                "pointnet_modelnet40",
                "resnet18_cifar10",
                "saint_adult",
            ),
        )
        self.assertEqual(selected_model_keys(None), list(DEFAULT_MODEL_KEYS))
        self.assertEqual(selected_model_keys(["all"]), list(ALL_MODEL_KEYS))
        self.assertEqual(selected_model_keys(["lenet5", "lenet5"]), ["lenet5"])
        with self.assertRaises(ValueError):
            selected_model_keys(["all", "lenet5"])
        with self.assertRaises(KeyError):
            selected_model_keys(["not-a-model"])

    def test_qat_shadow_accumulates_updates_between_projections(self) -> None:
        model = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[0.2, -0.7]]))
        config = TrainingConfig(
            bit_width=2,
            quantization_mode="ternary",
            use_qat=True,
        )
        qat_init_shadow(model)
        qat_project_for_forward(model, config)
        projected = model.weight.detach().clone()
        qat_restore_shadow_for_step(model, config)
        with torch.no_grad():
            model.weight.add_(0.05)
        qat_sync_shadow_after_step(model, config)
        qat_project_for_forward(model, config)
        self.assertFalse(torch.equal(model.weight, projected + 0.05))

        expected = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            expected.weight.copy_(torch.tensor([[0.25, -0.65]]))
        make_quantized_copy(expected, 2, "ternary")
        torch.testing.assert_close(model.weight, expected.weight)

    def test_experiment_plan_is_immutable_and_complete(self) -> None:
        plan = ExperimentPlan(
            artifact_id="attempt-1",
            model_key="lenet5",
            condition_key="dendrites_q8",
            source_condition_key="dendrites_fp32",
            output_dir=Path("results/lenet5/dendrites_q8"),
            pai_save_name="lenet5_dendrites_q8_attempt",
            model_revision="revision-1",
            dataset_revision="data-1",
            model_scale=1.0,
            seed=7,
            quantization_evaluation_revision="quant-1",
            pai_variant="default",
            pai_fixed_switch_interval=None,
            pai_dynamic_schedule={"max_dendrites": 1},
        )
        self.assertEqual(plan.identity()["dataset_revision"], "data-1")
        self.assertEqual(plan.identity()["source_condition_key"], "dendrites_fp32")
        with self.assertRaises(FrozenInstanceError):
            setattr(plan, "seed", 8)

    def test_off_parser_rejects_unbounded_or_truncated_counts(self) -> None:
        with tempfile.TemporaryDirectory() as root:  # type: ignore[no-matching-overload]
            path = Path(root) / "malformed.off"
            path.write_text("OFF\n3 1000001 0\n0 0 0\n1 0 0\n0 1 0\n")
            with self.assertRaisesRegex(ValueError, "face count"):
                _read_off_mesh(path)

            path.write_text("OFF\n3 1 0\n0 0 0\n1 0 0\n0 1 0\n3 0 1\n")
            with self.assertRaisesRegex(ValueError, "face width"):
                _read_off_mesh(path)


if __name__ == "__main__":
    unittest.main()
