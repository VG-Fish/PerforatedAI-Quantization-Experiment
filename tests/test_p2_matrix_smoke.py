"""Smoke the whole model x condition matrix without training anything.

The suite that existed before this pass covered one experiment generation and a
handful of private helpers, so a per-model policy branch could be wrong for 23
of the 24 registered models and still ship green.  These tests walk every
registered pair and assert the policy each pair resolves to: its recipe, its
condition plan, its constructor kwargs, its dependency order, and its PAI
namespace.  Nothing here builds a dataset or a model, so the matrix stays
runnable offline in CI.
"""

import tempfile
import unittest
from pathlib import Path
from typing import get_args

from dendritic_benchmark.model_adapters import ALL_MODEL_KEYS, model_adapter
from dendritic_benchmark.pipeline import BenchmarkRunner
from dendritic_benchmark.plans import (
    ConditionTrainingPlan,
    LRScheduleName,
    ModelTrainingRecipe,
    OptimizerName,
)
from dendritic_benchmark.specs import (
    CONDITION_SPECS,
    MODEL_SPECS,
    PRE_PERFORATED_MODEL_KEYS,
    condition_by_key,
    condition_supported_by_model,
    model_by_key,
)

_OPTIMIZER_NAMES = frozenset(get_args(OptimizerName))
_LR_SCHEDULE_NAMES = frozenset(get_args(LRScheduleName))

_CONDITION_KEYS = [spec.key for spec in CONDITION_SPECS]


def _supported_pairs() -> list[tuple[str, str]]:
    return [
        (model_key, condition_key)
        for model_key in ALL_MODEL_KEYS
        for condition_key in _CONDITION_KEYS
        if condition_supported_by_model(model_key, condition_key)
    ]


class MatrixSmokeTests(unittest.TestCase):
    """One runner for the whole matrix; none of these tests write results."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary = tempfile.TemporaryDirectory()  # type: ignore[no-matching-overload]
        root = Path(cls._temporary.name)
        cls.runner = BenchmarkRunner(
            results_root=root / "results", comparison_root=root / "comparison"
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temporary.cleanup()

    def test_matrix_size_is_the_registry_product_minus_known_exclusions(self) -> None:
        pairs = _supported_pairs()
        excluded = [
            (model_key, condition_key)
            for model_key in ALL_MODEL_KEYS
            for condition_key in _CONDITION_KEYS
            if not condition_supported_by_model(model_key, condition_key)
        ]
        self.assertEqual(
            len(pairs), len(MODEL_SPECS) * len(CONDITION_SPECS) - len(excluded)
        )
        # The only intended exclusions: the published HF checkpoints already
        # carry a trained dendrite graph, so a second one would not be a
        # control -- and neither are the two control families, which are both
        # defined relative to a dendrites_fp32 run these models never have.
        self.assertEqual(
            sorted(excluded),
            sorted(
                (model_key, key)
                for model_key in PRE_PERFORATED_MODEL_KEYS
                for key in _CONDITION_KEYS
                if key.startswith("dendrites_")
                or condition_by_key(key).control_kind is not None
            ),
        )

    def test_pre_perforated_models_keep_exactly_the_six_base_conditions(self) -> None:
        """A pre-perforated model's matrix is base_* and nothing else.

        Regression guard for a run that trained all six base conditions and
        then died in ``_prepare_control_model`` with "capacity controls require
        dendrites_fp32" -- the refusal was right, but it came hours after the
        planner should have excluded the pair.
        """
        for model_key in PRE_PERFORATED_MODEL_KEYS:
            supported = [
                key
                for key in _CONDITION_KEYS
                if condition_supported_by_model(model_key, key)
            ]
            self.assertEqual(
                supported,
                [key for key in _CONDITION_KEYS if key.startswith("base_")
                 and condition_by_key(key).control_kind is None],
                msg=model_key,
            )

    def test_every_pair_resolves_a_usable_training_recipe(self) -> None:
        for model_key, condition_key in _supported_pairs():
            with self.subTest(model=model_key, condition=condition_key):
                recipe = self.runner._training_hyperparameters(
                    model_key, condition_by_key(condition_key)
                )
                self.assertIsInstance(recipe, ModelTrainingRecipe)
                self.assertGreater(recipe.batch_size, 0)
                self.assertGreater(recipe.max_epochs, 0)
                self.assertGreater(recipe.learning_rate, 0.0)
                # Derived from the Literals rather than retyped: a recipe naming an
                # optimizer the builders do not implement is the bug worth catching,
                # and a hard-coded set here only catches "the registry grew".
                self.assertIn(recipe.optimizer_name, _OPTIMIZER_NAMES)
                self.assertIn(recipe.lr_schedule, _LR_SCHEDULE_NAMES)
                self.assertGreaterEqual(recipe.warmup_epochs, 0)

    def test_condition_plans_follow_the_two_factor_design(self) -> None:
        for model_key, condition_key in _supported_pairs():
            condition = condition_by_key(condition_key)
            recipe = self.runner._training_hyperparameters(model_key, condition)
            with self.subTest(model=model_key, condition=condition_key):
                without_pqat = self.runner._condition_training_plan(
                    model_key, condition, recipe, allow_pqat=False
                )
                self.assertIsInstance(without_pqat, ConditionTrainingPlan)
                derived_quantized = (
                    condition.quantized and condition.source_key != condition.key
                )
                if derived_quantized:
                    # Post-training quantization takes no gradient step unless
                    # PQAT is explicitly requested.
                    self.assertEqual(without_pqat.max_epochs, 0)
                else:
                    self.assertEqual(without_pqat.max_epochs, recipe.max_epochs)
                # A dendrite graph may only grow in the FP32 phase; the
                # quantized arms inherit it.
                self.assertEqual(
                    without_pqat.update_dendrites_during_training,
                    condition.use_dendrites and not condition.quantized,
                )

                with_pqat = self.runner._condition_training_plan(
                    model_key, condition, recipe, allow_pqat=True
                )
                if derived_quantized:
                    self.assertTrue(with_pqat.use_qat)
                    self.assertGreater(with_pqat.fine_tune_epochs, 0)
                    self.assertEqual(with_pqat.max_epochs, with_pqat.fine_tune_epochs)
                else:
                    self.assertEqual(with_pqat.max_epochs, without_pqat.max_epochs)

    def test_every_model_declares_consistent_constructor_capabilities(self) -> None:
        for model_key in ALL_MODEL_KEYS:
            with self.subTest(model=model_key):
                adapter = model_adapter(model_key)
                spec = model_by_key(model_key)
                kwargs = self.runner._model_kwargs(model_key)
                self.assertEqual(kwargs["model_scale"], 1.0)
                if adapter.num_classes is None:
                    self.assertNotIn("num_classes", kwargs)
                else:
                    self.assertEqual(kwargs["num_classes"], adapter.num_classes)
                    self.assertGreater(adapter.num_classes, 1)
                self.assertEqual(adapter.spec, spec)
                self.assertIn(spec.metric_direction, {"maximize", "minimize"})
                self.assertTrue(spec.dataset)

    def test_condition_sources_are_registered_and_stay_within_their_arm(self) -> None:
        for condition in CONDITION_SPECS:
            with self.subTest(condition=condition.key):
                source = condition_by_key(condition.source_key)
                self.assertFalse(source.quantized, "a source arm must be FP32")
                if condition.quantized:
                    # A quantized arm quantizes its own FP32 arm, so the two
                    # sides of the dendrite factor never cross.
                    self.assertEqual(source.use_dendrites, condition.use_dendrites)
                    self.assertIsNotNone(condition.bit_width)
                    self.assertIsNotNone(condition.quantization_mode)
                elif condition.use_dendrites:
                    # The dendritic FP32 arm grows out of the dense FP32
                    # baseline: that shared start is what makes it a control.
                    self.assertEqual(condition.source_key, "base_fp32")
                elif condition.control_kind is not None:
                    # The two FP32 validity controls intentionally fork from
                    # the audited dendritic source; their own quantized
                    # descendants still source their corresponding FP32 arm.
                    self.assertEqual(condition.source_key, "dendrites_fp32")
                else:
                    self.assertEqual(condition.source_key, condition.key)

    def test_expanding_any_condition_selection_pulls_its_source_in_first(self) -> None:
        for condition in CONDITION_SPECS:
            with self.subTest(condition=condition.key):
                expanded = self.runner._expand_condition_keys([condition.key])
                self.assertIn(condition.key, expanded)
                self.assertIn(condition.source_key, expanded)
                self.assertLessEqual(
                    expanded.index(condition.source_key), expanded.index(condition.key)
                )
                self.assertEqual(expanded, sorted(expanded, key=_CONDITION_KEYS.index))
        self.assertEqual(self.runner._expand_condition_keys(None), _CONDITION_KEYS)
        with self.assertRaises(KeyError):
            self.runner._expand_condition_keys(["not-a-condition"])

    def test_pai_namespaces_are_unique_per_pair_and_per_attempt(self) -> None:
        namespaces = {
            self.runner._pai_save_name(model_key, condition_key)
            for model_key, condition_key in _supported_pairs()
        }
        self.assertEqual(len(namespaces), len(_supported_pairs()))
        first = self.runner._pai_save_name("lenet5", "dendrites_fp32", "a" * 32)
        second = self.runner._pai_save_name("lenet5", "dendrites_fp32", "b" * 32)
        self.assertNotEqual(first, second)
        self.assertTrue(first.startswith("lenet5_dendrites_fp32_"))


if __name__ == "__main__":
    unittest.main()
