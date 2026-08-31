"""Tests for RecipeOverride/PAIOverride: the sweep-trial config objects
information/optimization/03_execution_matrix.md asks for so a base-recipe or
PAI-targeting trial (R1, RP1, AP0, ...) is a JSON file instead of a hand-edit
to pipeline.py. Covers JSON loading, merge/apply semantics, the
BenchmarkRunner wiring that actually applies them, the single-model guard,
and the CLI flags that load them.
"""

import json
import tempfile
import unittest
from pathlib import Path

from dendritic_benchmark.cli import build_parser
from dendritic_benchmark.compat import PAIDynamicSchedule
from dendritic_benchmark.pipeline import BenchmarkRunner
from dendritic_benchmark.plans import (
    CLEAR,
    CLEAR_JSON_VALUE,
    ModelTrainingRecipe,
    PAIOverride,
    RecipeOverride,
)


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload))
    return path


class RecipeOverrideTests(unittest.TestCase):
    def test_from_json_file_round_trips_set_fields_only(self) -> None:
        with tempfile.TemporaryDirectory() as root:  # type: ignore[no-matching-overload]
            path = _write_json(
                Path(root) / "override.json",
                {"learning_rate": 0.05, "weight_decay": 5e-4},
            )
            override = RecipeOverride.from_json_file(path)
        self.assertEqual(
            override.to_dict(), {"learning_rate": 0.05, "weight_decay": 5e-4}
        )
        self.assertIsNone(override.max_epochs)

    def test_a_nullable_field_can_be_cleared_to_none(self) -> None:
        # "None" already means "unset" on an override, so disabling gradient
        # clipping, the step schedule, or the LR-schedule horizon needed an
        # explicit sentinel. Without it those sweep arms were unreachable.
        recipe = ModelTrainingRecipe(
            batch_size=128,
            max_epochs=40,
            learning_rate=1e-2,
            lr_schedule="step",
            lr_decay_every=20,
            lr_schedule_epochs=40,
            grad_clip_norm=5.0,
        )
        with tempfile.TemporaryDirectory() as root:  # type: ignore[no-matching-overload]
            path = _write_json(
                Path(root) / "override.json",
                {
                    "grad_clip_norm": CLEAR_JSON_VALUE,
                    "lr_decay_every": CLEAR_JSON_VALUE,
                    "learning_rate": 0.003,
                },
            )
            override = RecipeOverride.from_json_file(path)
        self.assertIs(override.grad_clip_norm, CLEAR)
        applied = override.apply(recipe)
        self.assertIsNone(applied.grad_clip_norm)
        self.assertIsNone(applied.lr_decay_every)
        self.assertEqual(applied.learning_rate, 0.003)
        # an untouched nullable field keeps the recipe's own value
        self.assertEqual(applied.lr_schedule_epochs, 40)

    def test_a_cleared_field_survives_the_metrics_json_round_trip(self) -> None:
        # to_dict is written to metrics.json and read back by
        # _condition_metadata_current. Emitting CLEAR as JSON null would read
        # back as "unset", so the trial would judge its own artifact stale and
        # retrain it on every invocation -- PAIOverride's tuple bug again.
        override = RecipeOverride(grad_clip_norm=CLEAR, learning_rate=0.05)
        recorded = override.to_dict()
        self.assertEqual(recorded["grad_clip_norm"], CLEAR_JSON_VALUE)
        self.assertEqual(json.loads(json.dumps(recorded)), recorded)
        with tempfile.TemporaryDirectory() as root:  # type: ignore[no-matching-overload]
            path = _write_json(Path(root) / "recorded.json", recorded)
            self.assertEqual(RecipeOverride.from_json_file(path), override)

    def test_clearing_a_non_nullable_field_is_rejected(self) -> None:
        # ModelTrainingRecipe.learning_rate is not Optional; clearing it would
        # build a recipe the trainer cannot run. The field's annotation already
        # rejects CLEAR statically -- hence the ignore -- but an override
        # arriving from a JSON file is never type-checked, so the runtime
        # guard is the one that actually fires.
        with self.assertRaises(ValueError) as ctx:
            RecipeOverride(learning_rate=CLEAR)  # ty: ignore[invalid-argument-type]
        self.assertIn("cannot be cleared to None", str(ctx.exception))

    def test_clearing_a_non_nullable_field_is_rejected_from_json(self) -> None:
        with tempfile.TemporaryDirectory() as root:  # type: ignore[no-matching-overload]
            path = _write_json(
                Path(root) / "override.json", {"learning_rate": CLEAR_JSON_VALUE}
            )
            with self.assertRaises(ValueError) as ctx:
                RecipeOverride.from_json_file(path)
        self.assertIn("cannot be cleared to None", str(ctx.exception))

    def test_from_json_file_rejects_unknown_keys(self) -> None:
        with tempfile.TemporaryDirectory() as root:  # type: ignore[no-matching-overload]
            path = _write_json(Path(root) / "override.json", {"leraning_rate": 0.05})
            with self.assertRaises(ValueError) as ctx:
                RecipeOverride.from_json_file(path)
        self.assertIn("leraning_rate", str(ctx.exception))

    def test_apply_replaces_only_set_fields(self) -> None:
        base = ModelTrainingRecipe(128, 200, 0.1, "sgd", 0.9, 5e-4)
        override = RecipeOverride(learning_rate=0.05)
        updated = override.apply(base)
        self.assertEqual(updated.learning_rate, 0.05)
        self.assertEqual(updated.batch_size, base.batch_size)
        self.assertEqual(updated.weight_decay, base.weight_decay)

    def test_apply_with_no_fields_set_returns_recipe_unchanged(self) -> None:
        base = ModelTrainingRecipe(128, 200, 0.1)
        self.assertEqual(RecipeOverride().apply(base), base)

    def test_dendrite_lr_min_factor_is_a_recipe_field_not_a_pai_field(self) -> None:
        # information/optimization/01_initial_five_plan.md groups "dendrite LR
        # floor" with the PAI tuning grid, but it is implemented on
        # ModelTrainingRecipe -- confirm RecipeOverride, not PAIOverride, is
        # what carries it.
        override = RecipeOverride(dendrite_lr_min_factor=0.05)
        base = ModelTrainingRecipe(128, 40, 1e-2)
        self.assertEqual(override.apply(base).dendrite_lr_min_factor, 0.05)


class PAIOverrideTests(unittest.TestCase):
    def test_from_json_file_converts_lists_to_tuples(self) -> None:
        with tempfile.TemporaryDirectory() as root:  # type: ignore[no-matching-overload]
            path = _write_json(
                Path(root) / "override.json",
                {"module_ids_to_perforate": [".fc1"], "max_dendrites": 2},
            )
            override = PAIOverride.from_json_file(path)
        self.assertEqual(override.module_ids_to_perforate, (".fc1",))
        self.assertEqual(override.max_dendrites, 2)

    def test_from_json_file_rejects_unknown_keys(self) -> None:
        with tempfile.TemporaryDirectory() as root:  # type: ignore[no-matching-overload]
            path = _write_json(Path(root) / "override.json", {"max_dendrite": 2})
            with self.assertRaises(ValueError):
                PAIOverride.from_json_file(path)

    def test_history_lookback_and_initial_history_must_be_set_together(self) -> None:
        with self.assertRaises(ValueError):
            PAIOverride(history_lookback=12)
        with self.assertRaises(ValueError):
            PAIOverride(initial_history_after_switches=12)

    def test_history_lookback_and_initial_history_must_be_equal(self) -> None:
        # information/optimization/03_execution_matrix.md's zero-seeded EMA
        # bug guard (see memory pai-zero-seeded-ema-corrupts-best-tracking).
        with self.assertRaises(ValueError):
            PAIOverride(history_lookback=12, initial_history_after_switches=8)
        # Equal values are fine.
        PAIOverride(history_lookback=12, initial_history_after_switches=12)

    def test_to_dict_only_includes_set_fields(self) -> None:
        override = PAIOverride(max_dendrites=2)
        self.assertEqual(override.to_dict(), {"max_dendrites": 2})

    def test_to_dict_survives_a_json_round_trip(self) -> None:
        # to_dict() is written to metrics.json and read back by
        # BenchmarkRunner._condition_metadata_current to decide whether a saved
        # artifact still matches. JSON has no tuple, so emitting one here would
        # make every override run judge its own artifact stale and retrain it
        # on every invocation.
        override = PAIOverride(
            module_ids_to_perforate=(".readout.0", ".readout_gate"),
            track_only_module_ids=(".layers",),
            improvement_threshold=(0.005, 0.002),
            max_dendrites=1,
        )
        recorded = override.to_dict()
        self.assertEqual(json.loads(json.dumps(recorded)), recorded)
        self.assertEqual(
            recorded["module_ids_to_perforate"], [".readout.0", ".readout_gate"]
        )
        self.assertEqual(recorded["improvement_threshold"], [0.005, 0.002])

    def test_to_dict_round_trips_through_from_json_file(self) -> None:
        override = PAIOverride(
            module_ids_to_perforate=(".fc1",), improvement_threshold=(0.005, 0.002)
        )
        with tempfile.TemporaryDirectory() as root:  # type: ignore[no-matching-overload]
            path = _write_json(Path(root) / "pai.json", override.to_dict())
            self.assertEqual(PAIOverride.from_json_file(path), override)

    def test_empty_sequences_are_rejected(self) -> None:
        # An empty module_ids_to_perforate would make
        # _perforation_modules_to_perforate fall back to type-selecting every
        # Linear/Conv, i.e. silently widen the target set rather than narrow it.
        for name, build in (
            ("module_ids_to_perforate", lambda: PAIOverride(module_ids_to_perforate=())),
            ("track_only_module_ids", lambda: PAIOverride(track_only_module_ids=())),
            ("improvement_threshold", lambda: PAIOverride(improvement_threshold=())),
        ):
            with self.subTest(field=name), self.assertRaises(ValueError):
                build()

    def test_apply_to_schedule_returns_base_unchanged_when_no_schedule_field_set(
        self,
    ) -> None:
        override = PAIOverride(module_ids_to_perforate=(".fc1",))
        base = PAIDynamicSchedule(max_dendrites=1)
        self.assertIs(override.apply_to_schedule(base), base)
        self.assertIsNone(override.apply_to_schedule(None))

    def test_apply_to_schedule_merges_set_fields_onto_base(self) -> None:
        override = PAIOverride(p_epochs_to_switch=8)
        base = PAIDynamicSchedule(max_dendrites=1, p_epochs_to_switch=10)
        merged = override.apply_to_schedule(base)
        assert merged is not None
        self.assertEqual(merged.p_epochs_to_switch, 8)
        self.assertEqual(merged.max_dendrites, 1)

    def test_apply_to_schedule_from_no_base_schedule(self) -> None:
        override = PAIOverride(max_dendrites=2)
        merged = override.apply_to_schedule(None)
        assert merged is not None
        self.assertEqual(merged.max_dendrites, 2)
        self.assertIsNone(merged.p_epochs_to_switch)

    def test_resolved_module_ids_falls_back_to_defaults_independently(self) -> None:
        override = PAIOverride(module_ids_to_perforate=(".fc1",))
        perforate, track_only = override.resolved_module_ids(
            [".conv4", ".fc1"], [".conv1", ".bn1"]
        )
        self.assertEqual(perforate, [".fc1"])
        # track_only_module_ids was not set on the override, so it falls back.
        self.assertEqual(track_only, [".conv1", ".bn1"])


class BenchmarkRunnerOverrideWiringTests(unittest.TestCase):
    def test_recipe_override_changes_resolved_training_hyperparameters(self) -> None:
        with tempfile.TemporaryDirectory() as root:  # type: ignore[no-matching-overload]
            runner = BenchmarkRunner(
                results_root=Path(root) / "results",
                recipe_override=RecipeOverride(learning_rate=0.05),
            )
            recipe = runner._training_hyperparameters(
                "m5", _condition("base_fp32")
            )
        self.assertEqual(recipe.learning_rate, 0.05)

    def test_pai_override_changes_resolved_target_modules(self) -> None:
        with tempfile.TemporaryDirectory() as root:  # type: ignore[no-matching-overload]
            runner = BenchmarkRunner(
                results_root=Path(root) / "results",
                pai_override=PAIOverride(module_ids_to_perforate=(".fc1",)),
            )
            self.assertEqual(
                runner._perforation_module_ids_to_perforate("m5"), [".fc1"]
            )
            # track_only_module_ids was not overridden, so m5's default stands.
            self.assertEqual(
                runner._perforation_track_only_module_ids("m5"),
                [".conv1", ".bn1", ".conv2", ".bn2", ".conv3", ".bn3", ".bn4"],
            )

    def test_pai_override_changes_resolved_dynamic_schedule(self) -> None:
        with tempfile.TemporaryDirectory() as root:  # type: ignore[no-matching-overload]
            runner = BenchmarkRunner(
                results_root=Path(root) / "results",
                pai_override=PAIOverride(p_epochs_to_switch=6),
            )
            schedule = runner._pai_dynamic_schedule("resnet18_cifar10")
        assert schedule is not None
        self.assertEqual(schedule.p_epochs_to_switch, 6)
        # resnet18_cifar10's own default (max_dendrites=1) is preserved.
        self.assertEqual(schedule.max_dendrites, 1)

    def test_no_override_leaves_resolved_values_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as root:  # type: ignore[no-matching-overload]
            plain = BenchmarkRunner(results_root=Path(root) / "plain")
            overridden = BenchmarkRunner(
                results_root=Path(root) / "overridden", recipe_override=RecipeOverride()
            )
            condition = _condition("base_fp32")
            self.assertEqual(
                plain._training_hyperparameters("m5", condition),
                overridden._training_hyperparameters("m5", condition),
            )

    def test_an_override_target_set_that_leaves_a_parameter_untyped_is_rejected(
        self,
    ) -> None:
        # PAI cannot type a parameter in neither list, and this benchmark
        # suppresses the warning that would say so, so the check has to raise.
        from dendritic_benchmark.models import build_model
        from dendritic_benchmark.specs import condition_by_key

        with tempfile.TemporaryDirectory() as root:  # type: ignore[no-matching-overload]
            runner = BenchmarkRunner(
                results_root=Path(root) / "results",
                # .conv4 moves out of the perforate list without joining the
                # track-only list, orphaning conv4.weight/conv4.bias.
                pai_override=PAIOverride(module_ids_to_perforate=(".fc1",)),
            )
            model = build_model("m5", model_scale=1.0, num_classes=12)
            with self.assertRaises(ValueError) as ctx:
                runner._dendrite_initialization_metadata(
                    model, "m5", None, condition_by_key("dendrites_fp32")
                )
        self.assertIn("conv4.weight", str(ctx.exception))

    def test_a_complete_override_target_set_is_accepted(self) -> None:
        # AP1 from the execution matrix: .fc1 alone, with .conv4 moved into the
        # track-only list so no parameter is left untyped.
        from dendritic_benchmark.compat import PAIModuleSelection
        from dendritic_benchmark.models import build_model

        with tempfile.TemporaryDirectory() as root:  # type: ignore[no-matching-overload]
            runner = BenchmarkRunner(
                results_root=Path(root) / "results",
                pai_override=PAIOverride(
                    module_ids_to_perforate=(".fc1",),
                    track_only_module_ids=(
                        ".conv1", ".bn1", ".conv2", ".bn2",
                        ".conv3", ".bn3", ".conv4", ".bn4",
                    ),
                ),
            )
            model = build_model("m5", model_scale=1.0, num_classes=12)
            selection = PAIModuleSelection(
                module_ids_to_perforate=runner._perforation_module_ids_to_perforate(
                    "m5"
                ),
                track_only_module_ids=runner._perforation_track_only_module_ids("m5"),
                parameter_ids_to_track=runner._perforation_parameter_ids_to_track("m5"),
            )
            self.assertEqual(selection.module_ids_to_perforate, [".fc1"])
            runner._reject_uncovered_parameters(model, "m5", selection)

    def test_the_guard_also_covers_a_checked_in_default_target_set(self) -> None:
        # The guard was override-only while mpnn/gru_forecaster/vae_mnist still
        # shipped incomplete defaults. They are covered now, so it applies to
        # every ID-based selection -- a default that regresses must fail the
        # same way an override does, and the message must say which it was.
        from dendritic_benchmark.compat import PAIModuleSelection
        from dendritic_benchmark.models import build_model
        from dendritic_benchmark.pipeline import _uncovered_parameter_names

        with tempfile.TemporaryDirectory() as root:  # type: ignore[no-matching-overload]
            plain = BenchmarkRunner(results_root=Path(root) / "results")
            scheduled = BenchmarkRunner(
                results_root=Path(root) / "scheduled",
                pai_override=PAIOverride(max_dendrites=1),
            )
            model = build_model("mpnn", model_scale=1.0)
            selection = PAIModuleSelection(
                module_ids_to_perforate=plain._perforation_module_ids_to_perforate(
                    "mpnn"
                ),
                track_only_module_ids=plain._perforation_track_only_module_ids("mpnn"),
                parameter_ids_to_track=plain._perforation_parameter_ids_to_track(
                    "mpnn"
                ),
            )
            # mpnn's own default is complete, so it passes on both runners --
            # including the schedule-only override, which changes no target.
            self.assertEqual(_uncovered_parameter_names(model, selection), [])
            plain._reject_uncovered_parameters(model, "mpnn", selection)
            scheduled._reject_uncovered_parameters(model, "mpnn", selection)

            # Drop one track-only ID to simulate a regressed default. No
            # override is set, so the message must blame the checked-in set.
            regressed = PAIModuleSelection(
                module_ids_to_perforate=selection.module_ids_to_perforate,
                track_only_module_ids=[
                    i
                    for i in (selection.track_only_module_ids or [])
                    if i != ".node_encoder"
                ],
                parameter_ids_to_track=selection.parameter_ids_to_track,
            )
            with self.assertRaises(ValueError) as ctx:
                plain._reject_uncovered_parameters(model, "mpnn", regressed)
            message = str(ctx.exception)
            self.assertIn("the checked-in mpnn target set", message)
            self.assertNotIn("--pai-override", message)
            # and it must name the module to add, not only the parameter
            self.assertIn(".node_encoder.0", message)

    def test_run_rejects_an_override_with_more_than_one_selected_model(self) -> None:
        with tempfile.TemporaryDirectory() as root:  # type: ignore[no-matching-overload]
            runner = BenchmarkRunner(
                results_root=Path(root) / "results",
                recipe_override=RecipeOverride(learning_rate=0.05),
            )
            with self.assertRaises(ValueError):
                runner.run(model_keys=["m5", "lenet5"], write_reports=False)


class ArtifactIdentityTests(unittest.TestCase):
    def test_m5_has_an_artifact_revision_for_its_new_target_set(self) -> None:
        # m5's PAI targets moved from type-selected Linear/Conv1d to the AP0
        # pair. The recorded module-ID fields are compared only when present,
        # so an artifact predating them reads as matching; the revision is the
        # only thing that invalidates a pre-AP0 m5 dendritic artifact.
        from dendritic_benchmark.pipeline import _MODEL_ARTIFACT_REVISIONS

        self.assertIn("m5", _MODEL_ARTIFACT_REVISIONS)

    def test_coverage_repairs_bumped_their_models_artifact_revisions(self) -> None:
        # mpnn/gru_forecaster/vae_mnist gained the track-only modules that
        # complete their parameter coverage. A tracked module is wrapped where
        # an untyped one is not, so a pre-coverage artifact has a different
        # topology and must not be reused -- and it was trained with
        # parameters PAI could not type, so it must not be reused anyway.
        from dendritic_benchmark.pipeline import _MODEL_ARTIFACT_REVISIONS

        for model_key in ("mpnn", "gru_forecaster", "vae_mnist"):
            with self.subTest(model_key=model_key):
                self.assertIn(model_key, _MODEL_ARTIFACT_REVISIONS)
        # the two that already had one must not silently keep the old value
        self.assertNotEqual(
            _MODEL_ARTIFACT_REVISIONS["gru_forecaster"],
            "dynamic11_multiscale_decoder_v2",
        )
        self.assertNotEqual(
            _MODEL_ARTIFACT_REVISIONS["vae_mnist"], "dynamic11_fair_ternary_v2"
        )

    def test_paired_control_identity_names_the_dense_arm_of_a_dendritic_run(
        self,
    ) -> None:
        # 00_assessment.md's validity protocol reads a dendritic result
        # against a dense control. The field was a hardcoded None, so nothing
        # on the record said which dense run that is.
        from dendritic_benchmark.specs import condition_by_key

        with tempfile.TemporaryDirectory() as root:  # type: ignore[no-matching-overload]
            results_root = Path(root) / "results"
            runner = BenchmarkRunner(results_root=results_root)
            runner._seed = 1  # set by run(), not the constructor
            recipe = ModelTrainingRecipe(
                batch_size=32, max_epochs=200, learning_rate=1e-3
            )

            dense = runner._paired_control_identity(
                "mpnn", condition_by_key("base_fp32"), recipe
            )
            self.assertIsNone(dense, "a dense run is paired with nothing")

            identity = runner._paired_control_identity(
                "mpnn", condition_by_key("dendrites_fp32"), recipe
            )
            assert identity is not None
            self.assertEqual(identity["control_condition_key"], "base_fp32")
            self.assertEqual(identity["control_model_key"], "mpnn")
            self.assertEqual(identity["seed"], 1)
            self.assertEqual(identity["dendritic_max_epochs"], 200)
            # no dense record on disk yet: say so rather than imply one exists
            self.assertEqual(identity["control_status"], "missing")
            self.assertIsNone(identity["control_artifact_id"])
            # the two unimplemented controls stay explicitly unset
            self.assertIsNone(identity["matched_continuation_control"])
            self.assertIsNone(identity["capacity_matched_control"])

            control_dir = results_root / "mpnn" / "base_fp32"
            control_dir.mkdir(parents=True)
            _write_json(control_dir / "record.json", {"artifact_id": "deadbeef"})
            linked = runner._paired_control_identity(
                "mpnn", condition_by_key("dendrites_fp32"), recipe
            )
            assert linked is not None
            self.assertEqual(linked["control_status"], "present")
            self.assertEqual(linked["control_artifact_id"], "deadbeef")

    def test_source_topology_hash_is_copied_from_the_fp32_source_manifest(
        self,
    ) -> None:
        # "The same FP32 source topology for PTQ and PQAT" was unverifiable
        # from the manifests: a run's own topology_hash describes its
        # quantized result, and the before_pqat snapshot predates it. Copying
        # the source's hash forward makes the two arms comparable.
        from dendritic_benchmark.artifacts import ARTIFACT_MANIFEST_NAME
        from dendritic_benchmark.specs import condition_by_key

        with tempfile.TemporaryDirectory() as root:  # type: ignore[no-matching-overload]
            runner = BenchmarkRunner(results_root=Path(root) / "results")
            source_dir = Path(root) / "source"
            source_dir.mkdir()
            q4 = condition_by_key("dendrites_q4")

            # no source directory recorded for this condition
            self.assertIsNone(runner._source_topology_hash(q4, {}))
            # a source with no manifest reports absence, not this run's hash
            self.assertIsNone(
                runner._source_topology_hash(q4, {q4.source_key: source_dir})
            )

            _write_json(
                source_dir / ARTIFACT_MANIFEST_NAME,
                {"telemetry": {"topology_hash": "abc123"}},
            )
            self.assertEqual(
                runner._source_topology_hash(q4, {q4.source_key: source_dir}),
                "abc123",
            )
            # a source that predates topology hashing stays None
            _write_json(source_dir / ARTIFACT_MANIFEST_NAME, {"telemetry": {}})
            self.assertIsNone(
                runner._source_topology_hash(q4, {q4.source_key: source_dir})
            )

    def test_dense_artifacts_get_a_topology_hash(self) -> None:
        # The hash was derived only from PAI's prepare_final_model, so every
        # non-dendritic artifact went without one -- including the dense
        # controls, which are exactly where comparing topologies pays off.
        from dendritic_benchmark.models import build_model
        from dendritic_benchmark.training import _topology_hash

        dense = build_model("lenet5")
        self.assertTrue(_topology_hash(dense))
        self.assertEqual(_topology_hash(dense), _topology_hash(build_model("lenet5")))

    def test_source_commit_marks_a_dirty_working_tree(self) -> None:
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as root:  # type: ignore[no-matching-overload]
            runner = BenchmarkRunner(results_root=Path(root) / "results")
            outputs = {
                ("rev-parse", "HEAD"): "abc123",
                ("status", "--porcelain", "--untracked-files=no"): " M src/x.py",
            }
            with patch.object(
                BenchmarkRunner, "_git_output", staticmethod(lambda *a: outputs[a])
            ):
                self.assertEqual(runner._source_commit(), "abc123-dirty")

            clean = BenchmarkRunner(results_root=Path(root) / "clean")
            outputs[("status", "--porcelain", "--untracked-files=no")] = ""
            with patch.object(
                BenchmarkRunner, "_git_output", staticmethod(lambda *a: outputs[a])
            ):
                self.assertEqual(clean._source_commit(), "abc123")

    def test_an_unavailable_git_status_is_not_reported_as_clean(self) -> None:
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as root:  # type: ignore[no-matching-overload]
            runner = BenchmarkRunner(results_root=Path(root) / "results")
            with patch.object(
                BenchmarkRunner,
                "_git_output",
                staticmethod(lambda *a: "abc123" if a[0] == "rev-parse" else None),
            ):
                self.assertEqual(runner._source_commit(), "abc123-dirty")

    def test_no_git_at_all_records_no_commit(self) -> None:
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as root:  # type: ignore[no-matching-overload]
            runner = BenchmarkRunner(results_root=Path(root) / "results")
            with patch.object(
                BenchmarkRunner, "_git_output", staticmethod(lambda *a: None)
            ):
                self.assertIsNone(runner._source_commit())


class CLIOverrideFlagTests(unittest.TestCase):
    def test_run_parser_accepts_override_flags(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "run",
                "--recipe-override",
                "recipe.json",
                "--pai-override",
                "pai.json",
            ]
        )
        self.assertEqual(args.recipe_override, Path("recipe.json"))
        self.assertEqual(args.pai_override, Path("pai.json"))

    def test_run_parser_defaults_overrides_to_none(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["run"])
        self.assertIsNone(args.recipe_override)
        self.assertIsNone(args.pai_override)

    def test_pai_variant_choices_include_distilbert_classifier_only(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["run", "--pai-variant", "distilbert_classifier_only"])
        self.assertEqual(args.pai_variant, "distilbert_classifier_only")


def _condition(key: str):
    from dendritic_benchmark.specs import condition_by_key

    return condition_by_key(key)


if __name__ == "__main__":
    unittest.main()
