import json
import tempfile
import unittest
from copy import deepcopy
from contextlib import nullcontext
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch

from dendritic_benchmark.artifacts import (
    finalize_artifact_manifest,
    write_artifact_manifest,
)
from dendritic_benchmark.compat import (
    PAIDynamicSchedule,
    _configure_dynamic_pai_schedule,
    _configure_pai_training_schedule,
    pai_save_path,
    set_pai_root,
    ternary_quantize_tensor,
    ternary_quantize_tensor_per_channel,
)
from dendritic_benchmark.data import _make_loader
from dendritic_benchmark.models import MPNNLayer, GRUForecaster, TCNForecaster, build_model
from dendritic_benchmark.pipeline import BenchmarkRunner
from dendritic_benchmark.specs import condition_by_key
from dendritic_benchmark.training import (
    ArtifactMetadata,
    EpochTrainingContext,
    PAI_DENDRITE_PARAM_GROUP_KEY,
    TrainingConfig,
    _apply_lr_schedule,
    _binary_or_multi_loss,
    _build_optimizer,
    _dendrite_audit,
    _finalize_quantized_model_for_eval,
    _final_clean_pai_parameter_stats,
    _dendrite_learning_rate,
    _is_dendrite_parameter_name,
    _make_quantized_copy,
    _optimizer_param_groups,
    _read_pai_architecture_log,
    _run_training_epochs,
    _scheduled_learning_rate,
    _write_pai_summary,
)


def _write_test_artifact(
    condition_dir: Path,
    *,
    model_key: str,
    condition_key: str,
    dendrite_status: str,
) -> None:
    artifact_id = f"test-{dendrite_status}"
    condition_dir.mkdir(parents=True, exist_ok=True)
    (condition_dir / "model.pt").write_bytes(b"test checkpoint")
    (condition_dir / "metrics.json").write_text(
        json.dumps({"artifact_id": artifact_id})
    )
    (condition_dir / "history.csv").write_text("epoch\n1\n")
    record = {
        "artifact_id": artifact_id,
        "model_key": model_key,
        "condition_key": condition_key,
        "dendrite_audit_status": dendrite_status,
    }
    (condition_dir / "record.json").write_text(json.dumps(record))
    (condition_dir / "record.csv").write_text(
        "artifact_id,model_key,condition_key,dendrite_audit_status\n"
        f"{artifact_id},{model_key},{condition_key},{dendrite_status}\n"
    )
    (condition_dir / "best_model_stats.csv").write_text("metric_value\n0.5\n")
    write_artifact_manifest(
        condition_dir,
        artifact_id=artifact_id,
        identity={"model_key": model_key, "condition_key": condition_key},
        pai_save_name="test-pai-namespace",
        validity={
            "dendrite_status": dendrite_status,
            "quantization_status": "not_applicable",
        },
    )
    finalize_artifact_manifest(condition_dir, artifact_id=artifact_id)


class _RecordingPC:
    DOING_HISTORY = "history"
    DOING_FIXED_SWITCH = "fixed"

    def __init__(self) -> None:
        self.values: dict[str, object] = {}

    def __getattr__(self, name: str):
        if not name.startswith("set_"):
            raise AttributeError(name)

        def setter(value: object) -> None:
            self.values[name] = value

        return setter


class DynamicPAIFollowupTests(unittest.TestCase):
    def test_mpnn_edge_projection_avoids_concat_without_changing_messages(self) -> None:
        torch.manual_seed(0)
        layer = MPNNLayer(hidden=4, edge_features=3)
        h = torch.randn(2, 5, 4, requires_grad=True)
        edge_features = torch.randn(2, 5, 5, 3)
        source = h.unsqueeze(2).expand(2, 5, 5, 4)
        target = h.unsqueeze(1).expand(2, 5, 5, 4)
        expected = layer.edge_mlp(torch.cat([target, source, edge_features], dim=-1))
        expected.square().sum().backward()
        h_grad = h.grad
        assert h_grad is not None
        expected_input_grad = h_grad.detach().clone()
        expected_parameter_grads = {}
        for name, parameter in layer.edge_mlp.named_parameters():
            parameter_grad = parameter.grad
            assert parameter_grad is not None
            expected_parameter_grads[name] = parameter_grad.detach().clone()

        layer.zero_grad(set_to_none=True)
        h = h.detach().clone().requires_grad_(True)
        actual = layer._edge_messages(h, edge_features)
        actual.square().sum().backward()

        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(h.grad, expected_input_grad, rtol=1e-5, atol=1e-6)
        for name, parameter in layer.edge_mlp.named_parameters():
            torch.testing.assert_close(
                parameter.grad, expected_parameter_grads[name], rtol=1e-5, atol=1e-6
            )

    def test_qat_final_evaluation_does_not_project_twice(self) -> None:
        model = torch.nn.Linear(3, 2, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[0.15, -0.55, 0.9], [0.2, 0.3, -0.1]]))
        projected = _make_quantized_copy(
            deepcopy(model), bit_width=2, mode="ternary", granularity="channel"
        )
        before = {name: value.detach().clone() for name, value in projected.state_dict().items()}
        config = TrainingConfig(
            bit_width=2,
            quantization_mode="ternary",
            quantization_granularity="channel",
            use_qat=True,
        )
        with patch(
            "dendritic_benchmark.training._make_quantized_copy",
            side_effect=AssertionError("QAT model must not be projected a second time"),
        ):
            finalized = _finalize_quantized_model_for_eval(projected, config)

        self.assertIs(finalized, projected)
        self.assertEqual(set(before), set(finalized.state_dict()))
        for name, value in finalized.state_dict().items():
            self.assertTrue(torch.equal(before[name], value), name)

    def test_dendrite_audit_requires_switch_and_parameter_evidence(self) -> None:
        metadata = ArtifactMetadata(
            model_key="tcn_forecaster",
            condition_key="dendrites_fp32",
            display_name="+Dendrites",
            metric_name="MAE",
            metric_direction="minimize",
            primary_metric_key="mae",
            use_dendrites=True,
            use_pruning=False,
            bit_width=32,
            use_qat=False,
            fine_tune_epochs=0,
            regression_loss="smooth_l1",
            enable_pai_dendrite_updates=True,
            train_dendrites_until_complete=True,
            freeze_dendrite_updates_fraction=0.2,
            pai_candidate_graph_batch_limit=None,
            memory_cleanup_interval_batches=None,
            model_scale=0.75,
            pai_variant="default",
            pai_fixed_switch_interval=6,
            pai_dynamic_schedule={"max_dendrites": 1},
            pai_save_name="tcn_forecaster_dendrites_fp32",
            dense_param_count=100,
        )
        no_insertion = _dendrite_audit(
            metadata=metadata,
            param_count=100,
            raw_architecture={"status": "available", "max_param_count": 100},
            raw_switches={"status": "available", "row_count": 0, "switch_epochs": []},
        )
        self.assertEqual(no_insertion["status"], "no_retained_insertion")

        verified = _dendrite_audit(
            metadata=metadata,
            param_count=120,
            raw_architecture={"status": "available", "max_param_count": 120},
            raw_switches={"status": "available", "row_count": 2, "switch_epochs": [1, 9]},
        )
        self.assertEqual(verified["status"], "verified_retained")

    def test_dendrite_audit_reads_pai_param_counts_not_stale_best_scores(self) -> None:
        save_name = "dendrite_csv_test"
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            set_pai_root(root_path / "PAI")
            raw_pai_dir = root_path / "PAI" / save_name
            raw_pai_dir.mkdir(parents=True)
            (raw_pai_dir / f"{save_name}_best_arch_scores.csv").write_text(
                "Param Counts,Max Valid Scores\n1000,0.8\n"
            )
            (raw_pai_dir / f"{save_name}param_counts.csv").write_text(
                "Switch Number,Param Count\n0,1000\n1,1000\n2,1120\n"
            )
            (raw_pai_dir / f"{save_name}switch_epochs.csv").write_text(
                "Switch Number,Switch Epoch\n0,9\n1,17\n"
            )
            metadata = ArtifactMetadata(
                model_key="gcn",
                condition_key="dendrites_fp32",
                display_name="+Dendrites",
                metric_name="MAE",
                metric_direction="minimize",
                primary_metric_key="mae",
                use_dendrites=True,
                use_pruning=False,
                bit_width=32,
                use_qat=False,
                fine_tune_epochs=0,
                regression_loss="smooth_l1",
                enable_pai_dendrite_updates=True,
                train_dendrites_until_complete=True,
                freeze_dendrite_updates_fraction=0.2,
                pai_candidate_graph_batch_limit=None,
                memory_cleanup_interval_batches=None,
                model_scale=0.75,
                pai_variant="default",
                pai_fixed_switch_interval=None,
                pai_dynamic_schedule={"max_dendrites": 1},
                pai_save_name=save_name,
                dense_param_count=1000,
            )
            raw_architecture = _read_pai_architecture_log(save_name)
            self.assertTrue(raw_architecture["path"].endswith("param_counts.csv"))
            self.assertEqual(raw_architecture["max_param_count"], 1120)
            self.assertEqual(
                _dendrite_audit(metadata=metadata, param_count=1120)["status"],
                "verified_retained",
            )
        set_pai_root(Path.cwd() / "PAI")

    def test_dendritic_parameter_stats_use_final_clean_pai_model(self) -> None:
        wrapped = torch.nn.Linear(2, 2, bias=False)
        clean = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            clean.weight.copy_(torch.tensor([[1.0, 0.0]]))

        class _PAIUtils:
            @staticmethod
            def prepare_final_model(model: torch.nn.Module) -> torch.nn.Module:
                self.assertIs(model, wrapped)
                return clean

        with patch("dendritic_benchmark.training.pai_runtime_guard", nullcontext):
            with patch(
                "dendritic_benchmark.training.importlib.import_module",
                return_value=_PAIUtils,
            ):
                self.assertEqual(_final_clean_pai_parameter_stats(wrapped), (2, 1))

    def test_data_worker_override_disables_multiprocessing(self) -> None:
        dataset = torch.utils.data.TensorDataset(torch.arange(4))
        with patch.dict("os.environ", {"DQB_DATA_NUM_WORKERS": "0"}):
            loader = _make_loader(dataset, batch_size=2, num_workers=2)

        self.assertEqual(loader.num_workers, 0)
        self.assertFalse(loader.persistent_workers)

    def test_schedule_override_scales_thresholds_with_dendrite_cap(self) -> None:
        pc = _RecordingPC()
        _configure_dynamic_pai_schedule(
            pc,
            schedule=PAIDynamicSchedule(max_dendrites=1, p_epochs_to_switch=6),
        )

        self.assertEqual(pc.values["set_max_dendrites"], 1)
        self.assertEqual(pc.values["set_p_epochs_to_switch"], 6)
        self.assertEqual(pc.values["set_improvement_threshold"], [0.005, 0.002])

    def test_bounded_and_open_ended_training_share_history_schedule(self) -> None:
        pc = _RecordingPC()

        class _GPA:
            def __init__(self, pai_config: _RecordingPC) -> None:
                self.pc = pai_config

        gpa = _GPA(pc)
        _configure_pai_training_schedule(gpa)

        self.assertEqual(pc.values["set_switch_mode"], "history")

    def test_dynamic_completion_stops_immediately(self) -> None:
        """An open-ended PAI run must stop on its completion epoch."""

        class _EpochProgress:
            def __iter__(self):
                return iter(range(5))

            def close(self) -> None:
                pass

        context = EpochTrainingContext(
            model=torch.nn.Linear(1, 1),
            model_key="gcn",
            bundle=None,
            device=torch.device("cpu"),
            criterion=None,
            torch=torch,
            max_epochs=3,
            run_label="gcn | dendrites_fp32",
            config=TrainingConfig(
                use_dendrites=True,
                enable_pai_dendrite_updates=True,
                train_dendrites_until_complete=True,
            ),
            metric_name="Accuracy",
            primary_metric_key="accuracy",
            metric_direction="maximize",
        )

        def record_epoch(*, state, epoch, **_kwargs):
            row = {"epoch": epoch + 1, "val_metric": 0.5}
            state.history.append(row)
            return row, 0.5

        def complete_pai_once(*, optimizer, pai_tracker, **_kwargs):
            return optimizer, None, pai_tracker is not None

        with (
            patch("dendritic_benchmark.training._epoch_progress", return_value=_EpochProgress()),
            patch("dendritic_benchmark.training._apply_lr_schedule"),
            patch(
                "dendritic_benchmark.training._run_training_pass_oom_guarded",
                return_value=(0.0, {}),
            ),
            patch("dendritic_benchmark.training._run_validation_pass", return_value=(0.0, {})),
            patch("dendritic_benchmark.training._record_epoch_result", side_effect=record_epoch),
            patch(
                "dendritic_benchmark.training._apply_pai_epoch_update",
                side_effect=complete_pai_once,
            ),
            patch("dendritic_benchmark.training._run_memory_guard_cleanup_if_needed"),
            patch("dendritic_benchmark.training._update_epoch_progress"),
            patch("dendritic_benchmark.training._training_collapsed", return_value=False),
            patch("dendritic_benchmark.training._set_pai_candidate_graph_for_context"),
        ):
            history, *_ = _run_training_epochs(context, optimizer=object(), pai_tracker=object())

        self.assertEqual([row["epoch"] for row in history], [1])
        self.assertEqual(history[-1]["training_termination_reason"], "pai_training_complete")

    def test_fixed_switching_requires_an_explicit_diagnostic_interval(self) -> None:
        pc = _RecordingPC()
        _configure_dynamic_pai_schedule(pc, fixed_switch_interval=8)

        self.assertEqual(pc.values["set_switch_mode"], "fixed")
        self.assertEqual(pc.values["set_first_fixed_switch_num"], 8)

    def test_targeted_profiles_keep_only_measured_modules(self) -> None:
        with tempfile.TemporaryDirectory() as root:  # type: ignore[no-matching-overload]
            runner = BenchmarkRunner(results_root=Path(root) / "results")
            self.assertEqual(
                runner._perforation_track_only_module_ids("actor_critic"),
                [".value", ".backbone.0", ".policy"],
            )
            self.assertEqual(
                runner._perforation_module_ids_to_perforate("mpnn"),
                [
                    ".readout.0",
                    ".readout_gate",
                    ".layers.2.update.hidden_gates",
                    ".layers.2.update.input_gates",
                    ".layers.3.update.hidden_gates",
                    ".layers.3.update.input_gates",
                ],
            )
            latent_runner = BenchmarkRunner(
                results_root=Path(root) / "latent-results",
                pai_variant="vae_latent",
            )
            self.assertEqual(
                latent_runner._perforation_module_ids_to_perforate("vae_mnist"),
                [".mu", ".logvar"],
            )

    def test_priority_model_profiles_use_safe_late_targets(self) -> None:
        with tempfile.TemporaryDirectory() as root:  # type: ignore[no-matching-overload]
            runner = BenchmarkRunner(results_root=Path(root) / "results")
            self.assertEqual(
                runner._perforation_module_ids_to_perforate("resnet18_cifar10"),
                [".pre_fc"],
            )
            self.assertEqual(
                runner._perforation_track_only_module_ids("resnet18_cifar10"),
                [".conv1", ".bn1", ".layer1", ".layer2", ".layer3", ".layer4", ".fc"],
            )
            self.assertEqual(
                runner._perforation_module_ids_to_perforate("saint_adult"),
                [".head"],
            )
            self.assertIsNone(runner._pai_fixed_switch_interval("saint_adult"))
            diagnostic_runner = BenchmarkRunner(
                results_root=Path(root) / "diagnostic-results",
                pai_fixed_switch_interval=100,
            )
            self.assertEqual(
                diagnostic_runner._pai_fixed_switch_interval("saint_adult"), 100
            )
            self.assertEqual(
                runner._perforation_module_ids_to_perforate("pointnet_modelnet40"),
                [".conv3.0", ".head.0"],
            )
            schedule = runner._pai_dynamic_schedule("resnet18_cifar10")
            self.assertIsNotNone(schedule)
            assert schedule is not None
            self.assertEqual(schedule.max_dendrites, 1)
            gcn_schedule = runner._pai_dynamic_schedule("gcn")
            self.assertIsNotNone(gcn_schedule)
            assert gcn_schedule is not None
            self.assertEqual(gcn_schedule.max_dendrites, 1)

    def test_priority_models_leave_no_parameter_untyped(self) -> None:
        """Every parameter must be perforated or tracked, or PAI drops into pdb.

        PAI assigns a ``parameter_type`` from the module that owns each
        parameter. One that is in neither list gets none, which it warns about
        on every p-phase step and follows with ``pdb.set_trace`` -- a hang in a
        non-interactive worker. This caught ``.head.1``/``.head.4`` on PointNet
        (132,352 parameters, the 512->256 Linear among them) and the row-block
        and head LayerNorms on SAINT.
        """

        def covers(module_id: str, parameter_name: str) -> bool:
            dotted = "." + parameter_name
            return dotted == module_id or dotted.startswith(module_id + ".")

        cases = {
            "resnet18_cifar10": {},
            "saint_adult": {"num_classes": 2},
            "pointnet_modelnet40": {"num_classes": 40},
        }
        with tempfile.TemporaryDirectory() as root:  # type: ignore[no-matching-overload]
            runner = BenchmarkRunner(results_root=Path(root) / "results")
            for model_key, kwargs in cases.items():
                model = build_model(model_key, **kwargs)
                ids = [
                    *runner._perforation_module_ids_to_perforate(model_key),
                    *runner._perforation_track_only_module_ids(model_key),
                    *runner._perforation_parameter_ids_to_track(model_key),
                ]
                uncovered = [
                    name
                    for name, _ in model.named_parameters()
                    if not any(covers(i, name) for i in ids)
                ]
                self.assertEqual(uncovered, [], f"{model_key} leaves parameters untyped")

    def test_dendritic_pqat_requires_a_verified_fp32_source(self) -> None:
        with tempfile.TemporaryDirectory() as root:  # type: ignore[no-matching-overload]
            root_path = Path(root)
            runner = BenchmarkRunner(results_root=root_path / "results")
            source_dir = root_path / "results" / "saint_adult" / "dendrites_fp32"
            source_dir.mkdir(parents=True)
            _write_test_artifact(
                source_dir,
                model_key="saint_adult",
                condition_key="dendrites_fp32",
                dendrite_status="no_retained_insertion",
            )
            with self.assertRaisesRegex(RuntimeError, "verified retained"):
                runner._require_verified_dendritic_pqat_source(
                    "saint_adult",
                    condition_by_key("dendrites_q8"),
                    {"dendrites_fp32": source_dir},
                )

            _write_test_artifact(
                source_dir,
                model_key="saint_adult",
                condition_key="dendrites_fp32",
                dendrite_status="verified_retained",
            )
            runner._require_verified_dendritic_pqat_source(
                "saint_adult",
                condition_by_key("dendrites_q8"),
                {"dendrites_fp32": source_dir},
            )

    def test_resnet_prefc_is_in_both_arms_and_starts_as_identity(self) -> None:
        model = build_model("resnet18_cifar10")
        self.assertIsInstance(model.pre_fc, torch.nn.Linear)
        self.assertEqual(model.pre_fc.in_features, model.pre_fc.out_features)
        torch.testing.assert_close(
            model.pre_fc.weight.detach(),
            torch.eye(model.pre_fc.in_features),
        )
        torch.testing.assert_close(
            model.pre_fc.bias.detach(),
            torch.zeros(model.pre_fc.out_features),
        )

    def test_compact_models_have_fewer_parameters(self) -> None:
        full = build_model("mpnn", model_scale=1.0)
        compact = build_model("mpnn", model_scale=0.75)
        self.assertLess(
            sum(parameter.numel() for parameter in compact.parameters()),
            sum(parameter.numel() for parameter in full.parameters()),
        )

    def test_tcn_uses_multiscale_nonlinear_readout(self) -> None:
        model = TCNForecaster(hidden=16)
        output = model(torch.randn(3, 96, 7))

        self.assertEqual(output.shape, (3, 24, 7))
        self.assertEqual(model.readout_windows, (8, 16, 32))
        self.assertIsInstance(model.head, torch.nn.Sequential)

    def test_tcn_uses_smooth_l1_and_history_pai_switching(self) -> None:
        config = TrainingConfig(regression_loss="smooth_l1")
        self.assertIsInstance(
            _binary_or_multi_loss("tcn_forecaster", config), torch.nn.SmoothL1Loss
        )
        with tempfile.TemporaryDirectory() as root:  # type: ignore[no-matching-overload]
            runner = BenchmarkRunner(results_root=Path(root) / "results")
            self.assertIsNone(runner._pai_fixed_switch_interval("tcn_forecaster"))
            ternary = condition_by_key("base_q1_58")
            recipe = runner._training_hyperparameters("tcn_forecaster", ternary)
            plan = runner._condition_training_plan(
                "tcn_forecaster", ternary, recipe, allow_pqat=True
            )
            self.assertEqual(recipe.learning_rate, 3.0e-4)
            self.assertEqual(plan.max_epochs, 36)

    def test_gru_uses_multiscale_decoder_projection(self) -> None:
        torch.manual_seed(0)
        model = GRUForecaster(hidden=16, use_revin=False)
        inputs = torch.randn(3, 12, 21)
        outputs = model(inputs)

        self.assertEqual(outputs.shape, (3, model.horizon, model.input_size))
        self.assertEqual(model.readout_windows, (8, 16, 32))
        self.assertIsInstance(model.head[1], torch.nn.Linear)
        self.assertIsInstance(model.head[4], torch.nn.Linear)
        wider = GRUForecaster(hidden=16, decoder_hidden=24)
        self.assertEqual(wider.decoder_hidden, 24)
        self.assertGreater(
            sum(parameter.numel() for parameter in wider.parameters()),
            sum(parameter.numel() for parameter in model.parameters()),
        )

    def test_gru_revin_is_on_by_default_and_free(self) -> None:
        torch.manual_seed(0)
        plain = GRUForecaster(hidden=16, use_revin=False).eval()
        torch.manual_seed(0)
        revin = GRUForecaster(hidden=16).eval()
        self.assertTrue(revin.use_revin)
        # RevIN is stateless, so it costs no parameters and leaves PAI nothing
        # extra to perforate.
        self.assertEqual(
            sum(p.numel() for p in plain.parameters()),
            sum(p.numel() for p in revin.parameters()),
        )

        # A window shifted and scaled away from the training distribution is
        # exactly the chronological-split case RevIN exists to absorb: the
        # prediction should track the shift rather than ignore it.
        base = torch.randn(2, 12, 21)
        shifted = base * 3.0 + 7.0
        self.assertGreater(
            (plain(shifted) - plain(base)).abs().mean().item(),
            0.0,
        )
        torch.testing.assert_close(
            revin(shifted), revin(base) * 3.0 + 7.0, rtol=1.0e-4, atol=1.0e-4
        )

    def test_gru_state_dropout_is_variational_and_off_by_default(self) -> None:
        torch.manual_seed(0)
        model = GRUForecaster(hidden=16)
        self.assertEqual(model.state_dropout, 0.0)
        self.assertIsNone(model._state_mask(torch.zeros(2, 16)))

        model = GRUForecaster(hidden=16, state_dropout=0.5).train()
        mask = model._state_mask(torch.zeros(4, 16))
        assert mask is not None
        # Inverted dropout: kept units are scaled by 1/keep, dropped are zero.
        self.assertEqual(
            sorted(set(mask.flatten().tolist())), [0.0, 2.0]
        )

    def test_gru_pai_targets_decoder_and_keeps_gate_run_as_ablation(self) -> None:
        with tempfile.TemporaryDirectory() as root:  # type: ignore[no-matching-overload]
            runner = BenchmarkRunner(results_root=Path(root) / "results")
            self.assertEqual(
                runner._perforation_module_ids_to_perforate("gru_forecaster"),
                [".head.1"],
            )
            self.assertEqual(
                runner._perforation_track_only_module_ids("gru_forecaster"),
                [
                    ".cells.0.input_gates",
                    ".cells.0.hidden_gates",
                    ".cells.1.input_gates",
                    ".cells.1.hidden_gates",
                    ".head.4",
                ],
            )
            self.assertIsNone(runner._pai_fixed_switch_interval("gru_forecaster"))
            gate_runner = BenchmarkRunner(
                results_root=Path(root) / "gate-results",
                pai_variant="gru_gate_ablation",
            )
            self.assertEqual(
                gate_runner._perforation_module_ids_to_perforate("gru_forecaster"),
                [".cells.0.input_gates", ".cells.1.input_gates"],
            )
            self.assertIsNone(gate_runner._pai_fixed_switch_interval("gru_forecaster"))
            diagnostic_runner = BenchmarkRunner(
                results_root=Path(root) / "fixed-results",
                pai_fixed_switch_interval=8,
            )
            self.assertEqual(
                diagnostic_runner._pai_fixed_switch_interval("gru_forecaster"), 8
            )

    def test_tcn_targets_best_scoring_head_projection_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as root:  # type: ignore[no-matching-overload]
            runner = BenchmarkRunner(results_root=Path(root) / "results")
            self.assertEqual(
                runner._perforation_module_ids_to_perforate("tcn_forecaster"),
                [".head.0"],
            )
            self.assertEqual(
                runner._perforation_track_only_module_ids("tcn_forecaster"),
                [".net", ".head.3"],
            )

    def test_vae_uses_fair_horizon_and_channelwise_ternary_pqat(self) -> None:
        with tempfile.TemporaryDirectory() as root:  # type: ignore[no-matching-overload]
            runner = BenchmarkRunner(results_root=Path(root) / "results")
            recipe = runner._training_hyperparameters(
                "vae_mnist", condition_by_key("base_fp32")
            )
            ternary = condition_by_key("base_q1_58")
            ternary_recipe = runner._training_hyperparameters("vae_mnist", ternary)
            self.assertEqual(recipe.max_epochs, 150)
            self.assertEqual(recipe.lr_schedule_epochs, 150)
            self.assertEqual(ternary_recipe.learning_rate, 2.0e-4)
            self.assertEqual(runner._pqat_epoch_budget("vae_mnist", ternary), 40)
            self.assertEqual(
                runner._quantization_granularity("vae_mnist", ternary), "channel"
            )

    def test_channelwise_ternary_preserves_small_output_rows(self) -> None:
        weights = torch.tensor([[0.10, -0.10], [10.0, -10.0]])
        per_tensor = ternary_quantize_tensor(weights)
        per_channel = ternary_quantize_tensor_per_channel(weights)

        self.assertEqual(per_tensor[0].abs().sum().item(), 0.0)
        torch.testing.assert_close(per_channel[0].abs(), torch.tensor([0.10, 0.10]))
        torch.testing.assert_close(per_channel[1].abs(), torch.tensor([10.0, 10.0]))

    def test_learning_rate_horizon_survives_dynamic_overrun(self) -> None:
        config = TrainingConfig(
            learning_rate=1.0e-3,
            lr_schedule="cosine",
            lr_min_factor=0.02,
            lr_schedule_epochs=150,
        )
        self.assertGreater(
            _scheduled_learning_rate(config, 86, 50) or 0.0,
            2.0e-5,
        )

    # Parameter names read off a real perforated PointNet checkpoint
    # (dendrites_fp32/model.pt, 4,128,369 params after one retained dendrite).
    _DENDRITE_NAMES = (
        "conv3.0.dendrites_to_top.0",
        "conv3.0.dendrite_module.layers.0.weight",
        "conv3.0.dendrite_module.layers.0.bias",
    )
    _BACKBONE_NAMES = (
        "conv3.0.main_module.weight",
        "conv3.0.dendrite_module.parent_module.weight",
        "conv3.0.dendrite_module.parent_module.bias",
        "head.0.main_module.weight",
        "input_transform.conv1.weight",
    )

    def test_dendrite_parameter_predicate_matches_real_checkpoint_names(self) -> None:
        for name in self._DENDRITE_NAMES:
            self.assertTrue(_is_dendrite_parameter_name(name), name)
        for name in self._BACKBONE_NAMES:
            self.assertFalse(_is_dendrite_parameter_name(name), name)

    def test_optimizer_groups_split_only_once_dendrites_exist(self) -> None:
        class _Fake(torch.nn.Module):
            def __init__(self, names: tuple[str, ...]) -> None:
                super().__init__()
                self._names = names
                self._params = torch.nn.ParameterList(
                    [torch.nn.Parameter(torch.zeros(2)) for _ in names]
                )

            def named_parameters(self, *args, **kwargs):  # type: ignore[override]
                return zip(self._names, self._params)

        dendritic = TrainingConfig(use_dendrites=True)
        # Before any dendrite is retained the split must be a no-op, so the
        # optimizer is built exactly as it was before grouping existed.
        groups = _optimizer_param_groups(_Fake(self._BACKBONE_NAMES), dendritic)
        self.assertNotIsInstance(groups, list)

        groups = _optimizer_param_groups(
            _Fake(self._BACKBONE_NAMES + self._DENDRITE_NAMES), dendritic
        )
        self.assertIsInstance(groups, list)
        self.assertEqual(len(groups), 2)
        by_flag = {g[PAI_DENDRITE_PARAM_GROUP_KEY]: g["params"] for g in groups}
        self.assertEqual(len(by_flag[True]), len(self._DENDRITE_NAMES))
        self.assertEqual(len(by_flag[False]), len(self._BACKBONE_NAMES))

        # A non-dendritic condition never groups, whatever the names look like.
        self.assertNotIsInstance(
            _optimizer_param_groups(
                _Fake(self._BACKBONE_NAMES + self._DENDRITE_NAMES),
                TrainingConfig(use_dendrites=False),
            ),
            list,
        )

    def test_dendrite_group_keeps_a_live_rate_past_the_cosine_floor(self) -> None:
        """The ResNet-18 recipe annealed the dendrite group to exactly 0.0.

        Measured before this fix: 13 of 19 epochs at lr=0.0 with validation
        flat inside 0.004.  The backbone must keep the schedule its dense
        control runs; only the inserted dendrite gets the floor.
        """
        config = TrainingConfig(
            use_dendrites=True, learning_rate=0.1, lr_schedule="cosine",
            warmup_epochs=5, lr_min_factor=0.0, dendrite_lr_min_factor=0.1,
        )
        # The floor is opt-in: the default must reproduce the old schedule.
        self.assertEqual(TrainingConfig().dendrite_lr_min_factor, 0.0)
        # A minimal stand-in for torch.optim.Optimizer: _apply_lr_schedule only
        # ever reads and writes param_groups.
        class _Opt:
            param_groups: list[dict[str, Any]]

        optimizer = _Opt()
        optimizer.param_groups = [
            {"lr": 0.0, PAI_DENDRITE_PARAM_GROUP_KEY: False},
            {"lr": 0.0, PAI_DENDRITE_PARAM_GROUP_KEY: True},
        ]
        _apply_lr_schedule(optimizer, config, 200, 200)
        backbone, dendrite = optimizer.param_groups
        self.assertEqual(backbone["lr"], 0.0)
        self.assertAlmostEqual(dendrite["lr"], 0.01)

        # While the schedule is still above the floor both groups agree, so
        # an early insertion behaves exactly as it did before this change.
        _apply_lr_schedule(optimizer, config, 120, 200)
        backbone, dendrite = optimizer.param_groups
        self.assertEqual(backbone["lr"], dendrite["lr"])

    def test_priority_recipes_opt_into_the_dendrite_lr_floor(self) -> None:
        """The three dynamic12 priority models must not anneal dendrites to nil."""
        with tempfile.TemporaryDirectory() as root:  # type: ignore[no-matching-overload]
            runner = BenchmarkRunner(results_root=Path(root) / "results")
            base = condition_by_key("base_fp32")
            for model_key in (
                "resnet18_cifar10", "saint_adult", "pointnet_modelnet40",
            ):
                recipe = runner._training_hyperparameters(model_key, base)
                self.assertGreater(
                    recipe.dendrite_lr_min_factor, 0.0, model_key
                )
                floor = recipe.learning_rate * recipe.dendrite_lr_min_factor
                config = TrainingConfig(
                    use_dendrites=True,
                    learning_rate=recipe.learning_rate,
                    lr_schedule=recipe.lr_schedule,
                    lr_decay_every=recipe.lr_decay_every,
                    lr_decay_gamma=recipe.lr_decay_gamma,
                    lr_min_factor=recipe.lr_min_factor,
                    warmup_epochs=recipe.warmup_epochs,
                    dendrite_lr_min_factor=recipe.dendrite_lr_min_factor,
                )
                # At the far end of the dynamic tail the backbone is at or
                # below its floor; the dendrite group must still be trainable.
                tail = recipe.max_epochs + 28
                scheduled = _scheduled_learning_rate(config, tail, recipe.max_epochs)
                self.assertIsNotNone(scheduled)
                self.assertAlmostEqual(
                    _dendrite_learning_rate(config, float(scheduled or 0.0)), floor
                )

    def test_optimizer_step_moves_dendrites_after_the_backbone_freezes(self) -> None:
        """End-to-end: a real step must still move a dendrite at lr-floor time.

        This is the behaviour the whole fix exists for.  At epoch 228 -- inside
        the dynamic tail, where ResNet-18's cosine sits at exactly 0.0 -- the
        backbone and PAI's frozen shadow copy must not move, and the dendrite
        and its dendrite-to-neuron mixing weight must.
        """
        class _Model(torch.nn.Module):
            _NAMES = (
                "layer.main_module.weight",
                "layer.dendrite_module.parent_module.weight",
                "layer.dendrite_module.layers.0.weight",
                "layer.dendrites_to_top.0",
            )

            def __init__(self) -> None:
                super().__init__()
                self.p = torch.nn.ParameterList(
                    [torch.nn.Parameter(torch.zeros(2)) for _ in self._NAMES]
                )

            def named_parameters(self, *args, **kwargs):  # type: ignore[override]
                return zip(self._NAMES, self.p)

        config = TrainingConfig(
            use_dendrites=True, learning_rate=0.1, lr_schedule="cosine",
            warmup_epochs=5, lr_min_factor=0.0, dendrite_lr_min_factor=0.1,
            optimizer_name="sgd", momentum=0.9, nesterov=True,
        )
        model = _Model()
        optimizer = _build_optimizer(model, torch, config)
        self.assertEqual(
            [g.get(PAI_DENDRITE_PARAM_GROUP_KEY) for g in optimizer.param_groups],
            [False, True],
        )
        _apply_lr_schedule(optimizer, config, 228, 200)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.0)

        for parameter in model.p:
            parameter.grad = torch.ones(2)
        before = [parameter.detach().clone() for parameter in model.p]
        optimizer.step()
        moved = [
            not torch.equal(parameter.detach(), original)
            for parameter, original in zip(model.p, before)
        ]
        self.assertEqual(moved, [False, False, True, True])

    def test_dynamic_training_has_no_total_epoch_cap(self) -> None:
        self.assertNotIn("max_dynamic_training_epochs", TrainingConfig.__dataclass_fields__)
        self.assertFalse(hasattr(BenchmarkRunner, "_dynamic_training_epoch_cap"))

    def test_changed_model_revision_invalidates_prior_dynamic11_record(self) -> None:
        with tempfile.TemporaryDirectory() as root:  # type: ignore[no-matching-overload]
            root_path = Path(root)
            runner = BenchmarkRunner(results_root=root_path / "results")
            condition_dir = root_path / "results" / "gru_forecaster" / "base_fp32"
            condition_dir.mkdir(parents=True)
            (condition_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "model_scale": 1.0,
                        "model_revision": None,
                        "lr_schedule_epochs": None,
                        "quantization_granularity": "tensor",
                    }
                )
            )
            self.assertFalse(
                runner._condition_metadata_current(
                    "gru_forecaster", condition_by_key("base_fp32"), condition_dir
                )
            )

    def test_summary_marks_stale_raw_architecture_log(self) -> None:
        with tempfile.TemporaryDirectory() as root:  # type: ignore[no-matching-overload]
            root_path = Path(root)
            set_pai_root(root_path / "PAI")
            raw_pai_dir = root_path / "PAI" / "gcn_dendrites_fp32"
            raw_pai_dir.mkdir(parents=True)
            (raw_pai_dir / "gcn_dendrites_fp32param_counts.csv").write_text(
                "Switch Number,Param Count\n0,100\n"
            )
            metadata = ArtifactMetadata(
                model_key="gcn",
                condition_key="dendrites_fp32",
                display_name="GCN",
                metric_name="Accuracy",
                metric_direction="maximize",
                primary_metric_key="accuracy",
                use_dendrites=True,
                use_pruning=False,
                bit_width=None,
                use_qat=False,
                fine_tune_epochs=0,
                regression_loss="mse",
                enable_pai_dendrite_updates=True,
                train_dendrites_until_complete=True,
                freeze_dendrite_updates_fraction=0.2,
                pai_candidate_graph_batch_limit=None,
                memory_cleanup_interval_batches=None,
                model_scale=0.75,
                pai_variant="default",
                pai_fixed_switch_interval=None,
                pai_dynamic_schedule={"max_dendrites": 1},
                pai_save_name="gcn_dendrites_fp32",
            )
            output_dir = root_path / "result"
            output_dir.mkdir()
            _write_pai_summary(
                output_dir=output_dir,
                history=[
                    {
                        "epoch": 10,
                        "pai_restructured": True,
                        "pai_switch_reason": "candidate_phase_timeout",
                    },
                    {
                        "epoch": 20,
                        "pai_training_complete": True,
                        "training_termination_reason": "pai_training_complete",
                    },
                ],
                metadata=metadata,
                param_count=123,
                nonzero_params=120,
            )

            summary = json.loads((output_dir / "pai_summary.json").read_text())
            self.assertEqual(summary["authoritative_source"], "benchmark_history_and_final_checkpoint")
            self.assertEqual(summary["history"]["restructured_epochs"], [10])
            self.assertEqual(summary["history"]["training_complete_epochs"], [20])
            self.assertEqual(summary["architecture_log_consistency"], "stale")
            self.assertIsNone(summary["fixed_switch_interval"])
            self.assertEqual(summary["requested_schedule"]["mode"], "history")
            self.assertEqual(
                summary["observed_schedule"]["forced_switches"],
                [{"epoch": 10, "reason": "candidate_phase_timeout"}],
            )
            self.assertEqual(
                summary["observed_schedule"]["termination_reason"],
                "pai_training_complete",
            )


if __name__ == "__main__":
    unittest.main()
