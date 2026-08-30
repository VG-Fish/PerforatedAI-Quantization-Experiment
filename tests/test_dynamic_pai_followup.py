import json
import tempfile
import unittest
from copy import deepcopy
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import torch

from dendritic_benchmark.compat import (
    PAIDynamicSchedule,
    _configure_dynamic_pai_schedule,
    set_pai_root,
    ternary_quantize_tensor,
    ternary_quantize_tensor_per_channel,
)
from dendritic_benchmark.data import _make_loader
from dendritic_benchmark.models import GRUForecaster, TCNForecaster, build_model
from dendritic_benchmark.pipeline import BenchmarkRunner
from dendritic_benchmark.specs import condition_by_key
from dendritic_benchmark.training import (
    ArtifactMetadata,
    TrainingConfig,
    _binary_or_multi_loss,
    _dendrite_audit,
    _finalize_quantized_model_for_eval,
    _final_clean_pai_parameter_stats,
    _make_quantized_copy,
    _scheduled_learning_rate,
    _write_pai_summary,
)


class _RecordingPC:
    DOING_HISTORY = "history"

    def __init__(self) -> None:
        self.values: dict[str, object] = {}

    def __getattr__(self, name: str):
        if not name.startswith("set_"):
            raise AttributeError(name)

        def setter(value: object) -> None:
            self.values[name] = value

        return setter


class DynamicPAIFollowupTests(unittest.TestCase):
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

    def test_targeted_profiles_keep_only_measured_modules(self) -> None:
        with tempfile.TemporaryDirectory() as root:
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
        with tempfile.TemporaryDirectory() as root:
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
                [
                    ".row_blocks.0.attn.qkv",
                    ".row_blocks.1.attn.qkv",
                    ".head.1",
                ],
            )
            self.assertEqual(
                runner._perforation_module_ids_to_perforate("pointnet_modelnet40"),
                [".conv3.0", ".head.0"],
            )
            self.assertEqual(
                runner._pai_dynamic_schedule("resnet18_cifar10").max_dendrites,
                1,
            )

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
        with tempfile.TemporaryDirectory() as root:
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

    def test_tcn_uses_smooth_l1_and_early_fixed_pai_switch(self) -> None:
        config = TrainingConfig(regression_loss="smooth_l1")
        self.assertIsInstance(
            _binary_or_multi_loss("tcn_forecaster", config), torch.nn.SmoothL1Loss
        )
        with tempfile.TemporaryDirectory() as root:
            runner = BenchmarkRunner(results_root=Path(root) / "results")
            self.assertEqual(runner._pai_fixed_switch_interval("tcn_forecaster"), 6)
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
        with tempfile.TemporaryDirectory() as root:
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
            self.assertEqual(
                runner._pai_fixed_switch_interval("gru_forecaster"),
                8,
            )
            gate_runner = BenchmarkRunner(
                results_root=Path(root) / "gate-results",
                pai_variant="gru_gate_ablation",
            )
            self.assertEqual(
                gate_runner._perforation_module_ids_to_perforate("gru_forecaster"),
                [".cells.0.input_gates", ".cells.1.input_gates"],
            )
            self.assertIsNone(gate_runner._pai_fixed_switch_interval("gru_forecaster"))

    def test_tcn_targets_best_scoring_head_projection_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as root:
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
        with tempfile.TemporaryDirectory() as root:
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

    def test_changed_model_revision_invalidates_prior_dynamic11_record(self) -> None:
        with tempfile.TemporaryDirectory() as root:
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
        with tempfile.TemporaryDirectory() as root:
            root_path = Path(root)
            set_pai_root(root_path / "PAI")
            raw_pai_dir = root_path / "PAI" / "gcn_dendrites_fp32"
            raw_pai_dir.mkdir(parents=True)
            (raw_pai_dir / "gcn_dendrites_fp32_best_arch_scores.csv").write_text(
                "Param Counts,Max Valid Scores\n100,0.8\n"
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
                    {"epoch": 10, "pai_restructured": True},
                    {"epoch": 20, "pai_training_complete": True},
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


if __name__ == "__main__":
    unittest.main()
