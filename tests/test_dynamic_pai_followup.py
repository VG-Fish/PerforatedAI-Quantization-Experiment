import json
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

import torch

from dendritic_benchmark.compat import PAIDynamicSchedule, _configure_dynamic_pai_schedule, set_pai_root
from dendritic_benchmark.data import _make_loader
from dendritic_benchmark.models import GRUForecaster, TCNForecaster, build_model
from dendritic_benchmark.pipeline import BenchmarkRunner
from dendritic_benchmark.specs import condition_by_key
from dendritic_benchmark.training import (
    ArtifactMetadata,
    TrainingConfig,
    _binary_or_multi_loss,
    _final_clean_pai_parameter_stats,
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
            self.assertEqual(plan.max_epochs, 24)

    def test_gru_hoisted_input_projection_is_equivalent(self) -> None:
        torch.manual_seed(0)
        model = GRUForecaster(hidden=16)
        inputs = torch.randn(3, 12, 21)

        states = [inputs.new_zeros(inputs.shape[0], model.hidden) for _ in model.cells]
        for timestep in range(inputs.shape[1]):
            step = inputs[:, timestep]
            for index, cell in enumerate(model.cells):
                states[index] = cell(step, states[index])
                step = states[index]
        original = model.head(states[-1]).view(-1, model.horizon, model.input_size)

        # Batched matrix multiplication changes floating-point accumulation
        # order relative to 96 independent vector projections.
        torch.testing.assert_close(model(inputs), original, rtol=1.0e-5, atol=1.0e-6)

    def test_gru_pai_targets_hoisted_input_gates(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            runner = BenchmarkRunner(results_root=Path(root) / "results")
            self.assertEqual(
                runner._perforation_module_ids_to_perforate("gru_forecaster"),
                [".cells.0.input_gates", ".cells.1.input_gates"],
            )
            self.assertEqual(
                runner._perforation_track_only_module_ids("gru_forecaster"),
                [".cells.0.hidden_gates", ".cells.1.hidden_gates", ".head"],
            )
            self.assertEqual(
                runner._pai_dynamic_schedule("gru_forecaster").n_epochs_to_switch,
                8,
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
