import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import torch
import torchvision.models as tv_models

from dendritic_benchmark.compat import PAIRuntimeOptions
from dendritic_benchmark.data import _kd_mixup_cutmix_collate
from dendritic_benchmark.models import (
    _build_resnet18_kd_cifar100,
    build_kd_teacher_resnet50,
)
from dendritic_benchmark.pipeline import BenchmarkRunner
from dendritic_benchmark.specs import condition_by_key
from dendritic_benchmark.training import (
    TrainingConfig,
    _pai_validation_score,
    _run_epoch_batches,
    _setup_pai_optimizer,
)


class _RecordingTracker:
    def __init__(self) -> None:
        self.optimizer_class: Any = None
        self.scheduler_class = None
        self.optimizer_instance = None
        self.optimizer_args = None
        self.scheduler_args = None

    def set_optimizer(self, optimizer_class):
        self.optimizer_class = optimizer_class

    def set_scheduler(self, scheduler_class):
        self.scheduler_class = scheduler_class

    def set_optimizer_instance(self, optimizer):
        self.optimizer_instance = optimizer

    def setup_optimizer(self, model, optimizer_args, scheduler_args=None):
        del model
        self.optimizer_args = optimizer_args
        self.scheduler_args = scheduler_args
        return self.optimizer_class(**optimizer_args), None


class UpstreamBaseEquivalenceTests(unittest.TestCase):
    def _runner(self) -> BenchmarkRunner:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        return BenchmarkRunner(root / "results", root / "comparison")

    def test_upstream_recipes_are_not_substituted(self) -> None:
        runner = self._runner()
        base = condition_by_key("base_fp32")
        mnist = runner._training_hyperparameters("mnist_pai", base)
        self.assertEqual(mnist.optimizer_name, "adadelta")
        self.assertTrue(mnist.pai_owns_lr_schedule)
        self.assertEqual(mnist.dendrite_lr_min_factor, 0.0)

        kd = runner._training_hyperparameters("resnet18_kd_cifar100", base)
        self.assertTrue(kd.pai_owns_lr_schedule)
        self.assertEqual(kd.pai_restructure_lr_multiplier, 10.0)

        carvana = runner._training_hyperparameters("unet_carvana", base)
        self.assertEqual(
            (carvana.batch_size, carvana.learning_rate, carvana.optimizer_name),
            (1, 1.0e-5, "rmsprop"),
        )
        self.assertEqual(carvana.lr_schedule, "plateau")

        supervisely = runner._training_hyperparameters("unet_supervisely", base)
        self.assertEqual(supervisely.lr_schedule, "poly")
        self.assertFalse(supervisely.pai_setup_optimizer)

    def test_upstream_pai_growth_policies_are_model_specific(self) -> None:
        runner = self._runner()
        self.assertIsNone(runner._pai_dynamic_schedule("mnist_pai"))
        carvana = runner._pai_dynamic_schedule("unet_carvana")
        assert carvana is not None
        self.assertEqual(
            (carvana.max_dendrites, carvana.n_epochs_to_switch, carvana.p_epochs_to_switch),
            (2, 25, 25),
        )
        kd = runner._pai_dynamic_schedule("resnet18_kd_cifar100")
        assert kd is not None
        self.assertEqual((kd.max_dendrites, kd.n_epochs_to_switch), (5, 40))
        self.assertEqual(kd.initial_history_after_switches, 2)
        self.assertEqual(runner._pai_fixed_switch_interval("unet_supervisely"), 80)

    def test_pai_owned_step_scheduler_is_registered_with_tracker(self) -> None:
        tracker = _RecordingTracker()
        model = torch.nn.Linear(3, 2)
        config = TrainingConfig(
            use_dendrites=True,
            enable_pai_dendrite_updates=True,
            max_epochs=20,
            learning_rate=1.0,
            optimizer_name="adadelta",
            lr_schedule="step",
            lr_decay_every=1,
            lr_decay_gamma=0.7,
            pai_owns_lr_schedule=True,
        )
        with (
            patch("dendritic_benchmark.training._pai_tracker", return_value=tracker),
            patch("dendritic_benchmark.training._validate_pai_training_model"),
            patch("dendritic_benchmark.training.pai_working_directory"),
        ):
            _setup_pai_optimizer(model, torch, config)
        self.assertIs(tracker.scheduler_class, torch.optim.lr_scheduler.StepLR)
        self.assertEqual(tracker.scheduler_args, {"step_size": 1, "gamma": 0.7})

    def test_supervisely_uses_optimizer_instance_and_loss_signal(self) -> None:
        tracker = _RecordingTracker()
        model = torch.nn.Linear(3, 2)
        config = TrainingConfig(
            use_dendrites=True,
            enable_pai_dendrite_updates=True,
            max_epochs=80,
            learning_rate=0.01,
            optimizer_name="sgd",
            pai_setup_optimizer=False,
        )
        with (
            patch("dendritic_benchmark.training._pai_tracker", return_value=tracker),
            patch("dendritic_benchmark.training._validate_pai_training_model"),
            patch("dendritic_benchmark.training.pai_working_directory"),
        ):
            optimizer, _ = _setup_pai_optimizer(model, torch, config)
        self.assertIs(tracker.optimizer_instance, optimizer)
        context = cast(Any, SimpleNamespace(model_key="unet_supervisely"))
        self.assertEqual(
            _pai_validation_score(context, val_metric=0.8, val_loss=0.25), 0.25
        )

    def test_kd_mixup_cutmix_produces_soft_targets(self) -> None:
        batch = [(torch.rand(3, 32, 32), index) for index in range(4)]
        images, targets = _kd_mixup_cutmix_collate(batch)
        self.assertEqual(tuple(images.shape), (4, 3, 32, 32))
        self.assertEqual(tuple(targets.shape), (4, 100))
        torch.testing.assert_close(targets.sum(dim=1), torch.ones(4))

    def test_kd_models_keep_upstream_imagenet_stems(self) -> None:
        original_resnet18 = tv_models.resnet18
        original_resnet50 = tv_models.resnet50
        with (
            patch.object(
                tv_models,
                "resnet18",
                side_effect=lambda **kwargs: original_resnet18(weights=None),
            ),
            patch.object(
                tv_models,
                "resnet50",
                side_effect=lambda **kwargs: original_resnet50(weights=None),
            ),
        ):
            student = _build_resnet18_kd_cifar100()
            teacher = build_kd_teacher_resnet50()
        self.assertEqual(student.conv1.kernel_size, (7, 7))
        self.assertEqual(student.conv1.stride, (2, 2))
        self.assertIsInstance(student.maxpool, torch.nn.MaxPool2d)
        self.assertEqual(teacher.conv1.kernel_size, (7, 7))
        self.assertIsInstance(teacher.maxpool, torch.nn.MaxPool2d)
        identity = torch.eye(student.pre_fc.out_features)
        self.assertFalse(torch.equal(student.pre_fc.weight.detach(), identity))

        # Upstream builds resnet_double.ResNetPAI, which adds a 512x512
        # pre_fc; resnet.py's same-named class does not, and reading that one
        # instead moves the dendrite target onto the stem. The README's
        # 11,490,981-parameter Food-101 baseline is what tells the two apart,
        # so assert the arithmetic that identifies it.
        reference = original_resnet18(weights=None)
        reference.fc = torch.nn.Linear(reference.fc.in_features, 101)
        upstream_101_way = sum(p.numel() for p in reference.parameters())
        self.assertEqual(upstream_101_way, 11_228_325)
        pre_fc = sum(p.numel() for p in student.pre_fc.parameters())
        self.assertEqual(pre_fc, 512 * 512 + 512)
        self.assertEqual(upstream_101_way + pre_fc, 11_490_981)
        self.assertEqual(student(torch.rand(2, 3, 32, 32)).shape, (2, 100))

    def test_intra_epoch_validation_clock_runs_five_times(self) -> None:
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.rand(10, 2), torch.arange(10) % 2),
            batch_size=1,
        )
        bundle = SimpleNamespace(train_loader=loader)
        model = torch.nn.Linear(2, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        calls = 0

        def callback(current_model, current_optimizer):
            nonlocal calls
            calls += 1
            return current_model, current_optimizer, False, False

        _run_epoch_batches(
            model,
            "lenet5",
            bundle,
            torch.device("cpu"),
            torch.nn.CrossEntropyLoss(),
            optimizer,
            torch,
            0,
            1,
            "interval test",
            TrainingConfig(),
            "Accuracy",
            intra_epoch_callback=callback,
            intra_epoch_interval=2,
        )
        self.assertEqual(calls, 5)


if __name__ == "__main__":
    unittest.main()


def _carvana_config(recipe: Any, condition: Any) -> TrainingConfig:
    """A TrainingConfig carrying this recipe's schedule fields for a condition."""
    return TrainingConfig(
        max_epochs=recipe.max_epochs,
        learning_rate=recipe.learning_rate,
        optimizer_name=recipe.optimizer_name,
        momentum=recipe.momentum,
        weight_decay=recipe.weight_decay,
        lr_schedule=recipe.lr_schedule,
        lr_plateau_mode=recipe.lr_plateau_mode,
        lr_plateau_factor=recipe.lr_plateau_factor,
        lr_plateau_patience=recipe.lr_plateau_patience,
        pai_owns_lr_schedule=recipe.pai_owns_lr_schedule,
        pai_setup_optimizer=recipe.pai_setup_optimizer,
        use_dendrites=condition.use_dendrites,
        enable_pai_dendrite_updates=condition.use_dendrites,
    )


class UpstreamFidelityCorrectionTests(unittest.TestCase):
    """Regressions for the four places the port diverged from upstream."""

    def _runner(self) -> BenchmarkRunner:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        return BenchmarkRunner(root / "results", root / "comparison")

    # -- the KD student is ResNetPAI, not the transfer-learning pre_fc form ---

    def test_kd_perforates_pre_fc_and_tracks_the_rest(self) -> None:
        runner = self._runner()
        self.assertEqual(
            runner._default_module_ids_to_perforate("resnet18_kd_cifar100"),
            [".pre_fc"],
        )
        self.assertEqual(
            runner._default_track_only_module_ids("resnet18_kd_cifar100"),
            [".conv1", ".bn1", ".layer1", ".layer2", ".layer3", ".layer4", ".fc"],
        )
        # The lists have to name modules that exist, and between them cover
        # every parameter -- PAI leaves anything in neither list untyped.
        original_resnet18 = tv_models.resnet18
        with patch.object(
            tv_models, "resnet18", side_effect=lambda **_: original_resnet18(weights=None)
        ):
            student = _build_resnet18_kd_cifar100()
        named = dict(student.named_modules())
        prefixes = ("pre_fc", "conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc")
        for prefix in prefixes:
            self.assertIn(prefix, named, prefix)
        covered = {
            name
            for prefix in prefixes
            for name, _ in named[prefix].named_parameters(prefix=prefix)
        }
        self.assertEqual(covered, {name for name, _ in student.named_parameters()})

    # ------------------------------- Carvana Dice reduces over the batch ----

    def test_carvana_dice_term_is_upstream_reduce_batch_first(self) -> None:
        from dendritic_benchmark.training import CarvanaUNetLoss

        torch.manual_seed(0)
        logits = torch.randn(4, 2, 8, 8)
        targets = (torch.rand(4, 8, 8) > 0.5).long()

        # utils/dice_score.dice_loss(probabilities, onehot, multiclass=True),
        # transcribed from upstream rather than called: one global Dice over
        # the flattened (N*C, H, W) tensor.
        probabilities = torch.softmax(logits, dim=1).flatten(0, 1)
        onehot = (
            torch.nn.functional.one_hot(targets, 2).permute(0, 3, 1, 2).float().flatten(0, 1)
        )
        inter = 2 * (probabilities * onehot).sum()
        sets_sum = probabilities.sum() + onehot.sum()
        sets_sum = torch.where(sets_sum == 0, inter, sets_sum)
        expected = torch.nn.functional.cross_entropy(logits, targets) + (
            1 - (inter + 1e-6) / (sets_sum + 1e-6)
        )

        self.assertTrue(
            torch.allclose(CarvanaUNetLoss()(logits, targets), expected, atol=1e-6)
        )

        # And it is genuinely a different number from the per-(sample, class)
        # mean the port used to compute, so the test cannot pass by accident.
        per_pair_inter = 2 * (probabilities * onehot).sum(dim=(-1, -2))
        per_pair_sets = probabilities.sum(dim=(-1, -2)) + onehot.sum(dim=(-1, -2))
        per_pair = ((per_pair_inter + 1e-6) / (per_pair_sets + 1e-6)).mean()
        self.assertFalse(
            torch.allclose(per_pair, (inter + 1e-6) / (sets_sum + 1e-6), atol=1e-4)
        )

    # ------------------- the dense arms own the plateau schedule themselves --

    def test_carvana_dense_arms_get_a_plateau_scheduler(self) -> None:
        from dendritic_benchmark.training import _build_trainer_scheduler

        runner = self._runner()
        parameters = [torch.nn.Parameter(torch.zeros(1))]
        for condition_key in (
            "base_fp32",
            "base_more_training_fp32",
            "capacity_dense_fp32",
        ):
            with self.subTest(condition=condition_key):
                condition = condition_by_key(condition_key)
                recipe = runner._training_hyperparameters("unet_carvana", condition)
                self.assertEqual(recipe.lr_schedule, "plateau")
                self.assertTrue(recipe.pai_owns_lr_schedule)
                self.assertFalse(condition.use_dendrites)
                config = _carvana_config(recipe, condition)
                scheduler = _build_trainer_scheduler(
                    torch.optim.SGD(parameters, lr=0.1), torch, config, None
                )
                self.assertIsInstance(
                    scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                )
                plateau = cast(Any, scheduler)
                self.assertEqual(plateau.mode, "max")
                self.assertEqual(plateau.patience, 5)

    def test_pai_keeps_the_plateau_schedule_on_the_dendritic_arm(self) -> None:
        from dendritic_benchmark.training import _build_trainer_scheduler

        runner = self._runner()
        condition = condition_by_key("dendrites_fp32")
        config = _carvana_config(
            runner._training_hyperparameters("unet_carvana", condition), condition
        )
        self.assertTrue(config.use_dendrites)
        self.assertIsNone(
            _build_trainer_scheduler(
                torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1),
                torch,
                config,
                _RecordingTracker(),
            )
        )

    # ---------------------- container targets report a channels-first axis --

    def test_grouped_conv_norm_targets_report_the_channel_axis(self) -> None:
        from dendritic_benchmark.training import _dimension_vector_for_module_output

        # Upstream's set_output_dimensions([-1, 0, -1, -1]) for both U-Nets,
        # and PAISequential([conv1, bn1]) for the ResNet stem.
        block = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3, padding=1), torch.nn.BatchNorm2d(8))
        self.assertEqual(
            _dimension_vector_for_module_output(block, block(torch.rand(2, 3, 8, 8))),
            [-1, 0, -1, -1],
        )
        # A Linear head is unchanged: its feature axis is already the last one.
        head = torch.nn.Sequential(torch.nn.LayerNorm(4), torch.nn.Linear(4, 3))
        self.assertEqual(
            _dimension_vector_for_module_output(head, head(torch.rand(2, 5, 4))),
            [-1, -1, 0],
        )

    # ------------------------------ an unreadable pair names both its files --

    def test_supervisely_unreadable_pair_names_both_files(self) -> None:
        from dendritic_benchmark.data import _SuperviselyDataset

        with tempfile.TemporaryDirectory() as root:  # type: ignore[no-matching-overload]
            missing = Path(root) / "not-an-image.png"
            missing.write_text("not a png")
            dataset = _SuperviselyDataset([(missing, missing)], is_training=False)
            with self.assertRaises(RuntimeError) as caught:
                dataset[0]
            self.assertIn("not-an-image.png", str(caught.exception))

    # ---------------- Perforated Backpropagation survives a default config ---

    def test_default_runtime_options_keep_perforated_backpropagation(self) -> None:
        """A model with no runtime kwargs must not turn PB off.

        ``pc.perforated_backpropagation`` is the *declared* default (False);
        the value in force on a licensed install is ``get_...()`` (True).
        Restoring the attribute writes False back through the setter, which
        would demote all 27 models that pass no runtime kwargs from dendrites
        trained by Perforated Backpropagation to plain gradient descent -- the
        benchmark's entire independent variable.
        """
        from dendritic_benchmark import compat

        class _Config:
            def __init__(self) -> None:
                self.perforated_backpropagation = False  # declared default
                self.weight_decay_accepted = False
                self.cap_at_n = False
                self.test_saves = True
                self._live = True  # what the licensed library is running with
                self.writes: list[bool] = []

            def get_perforated_backpropagation(self) -> bool:
                return self._live

            def set_perforated_backpropagation(self, value: bool) -> None:
                self.writes.append(value)
                self._live = value

            def set_weight_decay_accepted(self, value: bool) -> None: ...
            def set_cap_at_n(self, value: bool) -> None: ...
            def set_test_saves(self, value: bool) -> None: ...

        config = _Config()
        with patch.object(compat, "_PAI_RUNTIME_OPTION_BASELINE", None):
            gpa = SimpleNamespace(pc=config)
            compat._configure_pai_runtime_options(gpa, PAIRuntimeOptions())
            self.assertTrue(config.get_perforated_backpropagation())
            self.assertEqual(config.writes, [True])

            # An explicit opt-out is still honoured ...
            compat._configure_pai_runtime_options(
                gpa, PAIRuntimeOptions(perforated_backpropagation=False)
            )
            self.assertFalse(config.get_perforated_backpropagation())

            # ... and does not leak into the next model in the same worker,
            # because the restore target is the once-per-process snapshot of
            # the live value rather than whatever the previous model left.
            compat._configure_pai_runtime_options(gpa, PAIRuntimeOptions())
            self.assertTrue(config.get_perforated_backpropagation())

    def test_live_config_value_prefers_the_getter(self) -> None:
        from dendritic_benchmark.compat import _pai_live_config_value

        class _Diverging:
            cap_at_n = False

            def get_cap_at_n(self) -> bool:
                return True

        class _GetterRaises:
            improvement_threshold = [0.1]

            def get_improvement_threshold(self) -> float:
                raise IndexError("needs a dendrite count")

        self.assertIs(_pai_live_config_value(_Diverging(), "cap_at_n"), True)
        # A getter that cannot answer falls back to the attribute rather than
        # propagating -- PAI's list settings are indexed by dendrite count and
        # their getters raise before any dendrite exists.
        self.assertEqual(
            _pai_live_config_value(_GetterRaises(), "improvement_threshold"), [0.1]
        )

    def test_schedule_restore_resets_fields_absent_from_a_fresh_config(self) -> None:
        """``p_epochs_to_switch`` must not leak between models in one worker.

        A worker perforates several models in one process. ``p_epochs_to_switch``
        exists only on the live config, not on a fresh ``PAIConfig()``, so
        restoring from a fresh instance skips it and leaves
        ``resnet18_kd_cifar100``'s 40 (or ``unet_carvana``'s 25) in force for
        every model perforated afterwards.
        """
        from dendritic_benchmark import compat

        library_defaults = {"n_epochs_to_switch": 10, "p_epochs_to_switch": 2}

        class _Live:
            def __init__(self) -> None:
                self._values = dict(library_defaults)

            def __getattr__(self, name: str) -> Any:
                if name.startswith("get_") and name[4:] in self._values:
                    return lambda: self._values[name[4:]]
                if name.startswith("set_"):
                    field = name[4:]
                    return lambda value: self._values.__setitem__(field, value)
                raise AttributeError(name)

        class _Fresh:
            # A fresh PAIConfig() declares n_epochs_to_switch but not
            # p_epochs_to_switch -- the latter arrives with perforatedbp.
            n_epochs_to_switch = 10

        live = _Live()
        gpa = SimpleNamespace(pc=live, PAIConfig=_Fresh)

        with patch.object(
            compat, "_PAI_LIBRARY_SCHEDULE_FIELDS", tuple(library_defaults)
        ), patch.object(compat, "_PAI_SCHEDULE_BASELINE", None):
            compat._restore_pai_library_schedule_defaults(gpa)  # snapshot
            live.set_n_epochs_to_switch(40)  # what the KD model configures
            live.set_p_epochs_to_switch(40)
            compat._restore_pai_library_schedule_defaults(gpa)  # next model

        self.assertEqual(live._values, library_defaults)
