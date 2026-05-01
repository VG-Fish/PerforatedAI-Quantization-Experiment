from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from .compat import choose_device, nn, perforate_model, require_torch
from .data import build_task_bundle
from .models import build_model
from .results import (
    save_training_record,
    write_comparison_reports,
    write_manifest,
    write_model_reports,
)
from .specs import CONDITION_SPECS, MODEL_SPECS, ConditionSpec, condition_by_key, model_by_key
from .training import OptimizerName, TrainingConfig, TrainingRecord, train_and_evaluate

EPOCH_MULTIPLIER = 10
_RECORD_JSON = "record.json"
_MODEL_PT = "model.pt"


@dataclass(frozen=True)
class ModelTrainingRecipe:
    batch_size: int
    max_epochs: int
    learning_rate: float
    optimizer_name: OptimizerName = "adam"
    momentum: float = 0.9
    weight_decay: float = 0.0


def _log(msg: str, *, before: bool = False, after: bool = False) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    if before:
        print()
    print(f"[{ts}] {msg}")
    if after:
        print()


class BenchmarkRunner:
    def __init__(
        self,
        results_root: Path | str = "results",
        comparison_root: Path | str = "comparison",
    ):
        self.results_root = Path(results_root)
        self.comparison_root = Path(comparison_root)
        self.results_root.mkdir(parents=True, exist_ok=True)
        self.comparison_root.mkdir(parents=True, exist_ok=True)

    def _load_state(
        self, model: Any, checkpoint_path: Path, *, strict: bool = True
    ) -> Any:
        torch = require_torch()
        if checkpoint_path.exists():
            state = torch.load(checkpoint_path, map_location=choose_device())
            if strict:
                model.load_state_dict(state, strict=True)
            else:
                current_state = model.state_dict()
                compatible_state = {
                    key: value
                    for key, value in state.items()
                    if key in current_state
                    and tuple(current_state[key].shape) == tuple(value.shape)
                }
                skipped = sorted(set(state) - set(compatible_state))
                model.load_state_dict(compatible_state, strict=False)
                if skipped:
                    print(
                        "[state] skipped incompatible source-checkpoint tensors: "
                        + ", ".join(skipped[:5])
                        + ("..." if len(skipped) > 5 else "")
                    )
        return model

    def _load_source_checkpoint(
        self,
        model: Any,
        model_key: str,
        source_key: str,
        checkpoint_path: Path,
        target_uses_dendrites: bool,
        save_name: str,
        maximizing_score: bool,
        config_snapshot_path: Path | str | None = None,
    ) -> Any:
        source_condition = condition_by_key(source_key)

        # Dendritic checkpoints contain PerforatedAI wrapper keys, so the target model
        # must be perforated before we load them. Base checkpoints still load into the
        # plain model first, then we perforate the model afterward if needed.
        if source_condition.use_dendrites and target_uses_dendrites:
            model = perforate_model(
                model,
                save_name=save_name,
                doing_pai=True,
                maximizing_score=maximizing_score,
                modules_to_track=self._perforation_track_modules(model_key),
                config_snapshot_path=config_snapshot_path,
            )
            model = self._configure_perforated_model(model, model_key)
            return self._load_state(model, checkpoint_path, strict=False)

        model = self._load_state(model, checkpoint_path)
        if target_uses_dendrites:
            model = perforate_model(
                model,
                save_name=save_name,
                doing_pai=True,
                maximizing_score=maximizing_score,
                modules_to_track=self._perforation_track_modules(model_key),
                config_snapshot_path=config_snapshot_path,
            )
            model = self._configure_perforated_model(model, model_key)
        return model

    def _artifact_path(
        self, condition_dir: Path, prefer_dendritic: bool = False
    ) -> Path:
        preferred = condition_dir / _MODEL_PT
        if preferred.exists():
            return preferred
        # Backwards compatibility for older runs that wrote multiple checkpoint names.
        if prefer_dendritic:
            for name in ["best_model", "final_clean_pai"]:
                path = condition_dir / name
                if path.exists():
                    return path
        return preferred

    def _expand_condition_keys(self, condition_keys: list[str] | None) -> list[str]:
        requested = condition_keys or [spec.key for spec in CONDITION_SPECS]
        lookup = {spec.key: spec for spec in CONDITION_SPECS}
        ordered: list[str] = []
        seen: set[str] = set()

        def visit(key: str) -> None:
            if key in seen:
                return
            spec = lookup[key]
            if spec.source_key and spec.source_key != key:
                visit(spec.source_key)
            seen.add(key)
            ordered.append(key)

        for key in requested:
            if key not in lookup:
                raise KeyError(f"Unknown condition key: {key}")
            visit(key)
        return [key for key in [spec.key for spec in CONDITION_SPECS] if key in ordered]

    def _model_kwargs(self, model_key: str) -> dict[str, Any]:
        if model_key in {"lenet5", "vae_mnist", "snn_nmnist", "capsnet_mnist"}:
            return {"num_classes": 10}
        if model_key == "m5":
            return {"num_classes": 12}
        if model_key == "textcnn":
            return {"num_classes": 4}
        if model_key == "gcn":
            return {"num_classes": 7}
        if model_key in {"tabnet", "gin_imdbb", "saint_adult"}:
            return {"num_classes": 2}
        if model_key == "pointnet_modelnet40":
            return {"num_classes": 40}
        if model_key == "distilbert":
            return {"num_classes": 2}
        return {}

    def _perforation_track_modules(self, model_key: str) -> list[Any]:
        if nn is None:
            return []
        if model_key in {"lstm_forecaster", "lstm_autoencoder"}:
            return [nn.LSTM]
        if model_key in {"distilbert", "gru_forecaster"}:
            return [nn.GRU]
        if model_key == "attentivefp_freesolv":
            return [nn.GRUCell]
        if model_key == "saint_adult":
            return [nn.MultiheadAttention]
        return []

    def _configure_perforated_model(self, model: Any, model_key: str) -> Any:
        if (
            model_key == "gcn"
            and hasattr(model, "conv2")
            and hasattr(model.conv2, "linear")
        ):
            linear = model.conv2.linear
            if hasattr(linear, "set_this_output_dimensions"):
                linear.set_this_output_dimensions([-1, 0])
        return model

    def _training_hyperparameters(
        self, model_key: str, _condition: ConditionSpec
    ) -> ModelTrainingRecipe:
        """Return model-specific training knobs adapted from canonical recipes."""
        recipes: dict[str, ModelTrainingRecipe] = {
            "lenet5": ModelTrainingRecipe(256, 20, 1.0e-2, "sgd", 0.9, 0.0),
            "m5": ModelTrainingRecipe(128, 30, 1.0e-2, "adam", 0.9, 1.0e-4),
            "lstm_forecaster": ModelTrainingRecipe(256, 40, 1.0e-3),
            "textcnn": ModelTrainingRecipe(128, 10, 1.0e-3, "adam", 0.9, 1.0e-4),
            "gcn": ModelTrainingRecipe(32, 200, 1.0e-2, "adam", 0.9, 5.0e-4),
            "tabnet": ModelTrainingRecipe(1024, 100, 2.0e-3, "adamw", 0.9, 1.0e-5),
            "mpnn": ModelTrainingRecipe(32, 100, 1.0e-3, "adam", 0.9, 1.0e-5),
            "actor_critic": ModelTrainingRecipe(512, 40, 3.0e-4),
            "lstm_autoencoder": ModelTrainingRecipe(128, 50, 1.0e-3),
            "distilbert": ModelTrainingRecipe(32, 4, 1.0e-4, "adamw", 0.9, 1.0e-2),
            "dqn_lunarlander": ModelTrainingRecipe(128, 120, 6.3e-4),
            "ppo_bipedalwalker": ModelTrainingRecipe(64, 120, 3.0e-4),
            "attentivefp_freesolv": ModelTrainingRecipe(32, 100, 1.0e-3, "adam", 0.9, 1.0e-5),
            "gin_imdbb": ModelTrainingRecipe(32, 100, 1.0e-2, "adam", 0.9, 5.0e-4),
            "tcn_forecaster": ModelTrainingRecipe(128, 60, 1.0e-3, "adam", 0.9, 1.0e-4),
            "gru_forecaster": ModelTrainingRecipe(24, 50, 1.0e-3),
            "pointnet_modelnet40": ModelTrainingRecipe(32, 60, 1.0e-3, "adam", 0.9, 1.0e-4),
            "vae_mnist": ModelTrainingRecipe(128, 20, 1.0e-3),
            "snn_nmnist": ModelTrainingRecipe(16, 50, 1.0e-3),
            "unet_isic": ModelTrainingRecipe(8, 100, 1.0e-3, "adam", 0.9, 1.0e-5),
            "resnet18_cifar10": ModelTrainingRecipe(128, 90, 5.0e-2, "sgd", 0.9, 5.0e-4),
            "mobilenetv2_cifar10": ModelTrainingRecipe(128, 150, 5.0e-2, "sgd", 0.9, 4.0e-5),
            "saint_adult": ModelTrainingRecipe(256, 100, 1.0e-4, "adamw", 0.9, 1.0e-5),
            "capsnet_mnist": ModelTrainingRecipe(128, 30, 3.0e-3, "adam", 0.9, 0.0),
            "convlstm_movingmnist": ModelTrainingRecipe(16, 50, 1.0e-3),
        }
        return recipes.get(
            model_key,
            ModelTrainingRecipe(64, 4 * EPOCH_MULTIPLIER, 1.0e-3),
        )

    def _pqat_epoch_budget(self, model_key: str) -> int:
        """Allocate a short PQAT phase from the model's canonical epoch recipe."""
        recipe = self._training_hyperparameters(
            model_key, condition_by_key("base_fp32")
        )
        return max(1, min(10, math.ceil(recipe.max_epochs * 0.30)))

    def _load_saved_condition(
        self,
        model_key: str,
        condition: ConditionSpec,
        model_records: list[dict[str, Any]],
        all_records: list[dict[str, Any]],
        saved_dirs: dict[str, Path],
    ) -> None:
        condition_dir = self.results_root / model_key / condition.key
        record_path = condition_dir / _RECORD_JSON
        _log(f"[skip] {model_key} / {condition.key} — record.json found, skipping training.")
        record = TrainingRecord(**json.loads(record_path.read_text()))
        model_records.append(record.to_dict())
        all_records.append(record.to_dict())
        saved_dirs[condition.key] = condition_dir

    def _train_pending_condition(
        self,
        model_spec: Any,
        condition: ConditionSpec,
        bundle: Any,
        ignore_saved: bool,
        model_records: list[dict[str, Any]],
        all_records: list[dict[str, Any]],
        saved_dirs: dict[str, Path],
        allow_pqat: bool,
    ) -> bool:
        condition_dir = self.results_root / model_spec.key / condition.key
        record_path = condition_dir / _RECORD_JSON
        if record_path.exists() and not ignore_saved:
            _log(f"[skip] {model_spec.key} / {condition.key} — record.json found, skipping training.")
            record = TrainingRecord(**json.loads(record_path.read_text()))
            newly_trained = False
        else:
            _log(f"[train] {model_spec.key} / {condition.key} — starting…", before=True)
            record = self._run_condition(
                model_spec.key,
                model_spec.metric_name,
                model_spec.metric_direction,
                bundle,
                condition,
                saved_dirs,
                allow_pqat,
            )
            save_training_record(record, condition_dir)
            _log(
                f"[done] {model_spec.key} / {condition.key} — "
                f"{model_spec.metric_name}: {record.metric_value:.4f}",
                after=True,
            )
            newly_trained = True
        model_records.append(record.to_dict())
        all_records.append(record.to_dict())
        saved_dirs[condition.key] = condition_dir
        return newly_trained

    def _process_one_model_spec(
        self,
        model_spec: Any,
        selected_conditions: list[Any],
        ignore_saved: bool,
        all_records: list[dict[str, Any]],
        allow_pqat: bool,
    ) -> bool:
        pending = [
            cond for cond in selected_conditions
            if ignore_saved or not (
                self.results_root / model_spec.key / cond.key / _RECORD_JSON
            ).exists()
        ]
        already_done = [cond for cond in selected_conditions if cond not in pending]

        if not pending:
            _log(
                f"[skip] {model_spec.key} — all conditions already recorded, "
                "skipping dataset load.",
                before=True,
            )
        else:
            _log(
                f"[data] {model_spec.key} — loading dataset "
                f"({len(pending)} condition(s) to train)…",
                before=True,
            )

        model_records: list[dict[str, Any]] = []
        saved_dirs: dict[str, Path] = {}

        for condition in already_done:
            self._load_saved_condition(
                model_spec.key, condition, model_records, all_records, saved_dirs
            )

        newly_trained = False
        if pending:
            recipe = self._training_hyperparameters(model_spec.key, selected_conditions[0])
            bundle = build_task_bundle(model_spec.key, batch_size=recipe.batch_size)
            for condition in pending:
                if self._train_pending_condition(
                    model_spec, condition, bundle, ignore_saved,
                    model_records, all_records, saved_dirs, allow_pqat,
                ):
                    newly_trained = True

        write_model_reports(
            model_spec.display_name,
            model_records,
            self.results_root / model_spec.key,
        )
        return newly_trained

    def run(
        self,
        model_keys: list[str] | None = None,
        condition_keys: list[str] | None = None,
        ignore_saved: bool = False,
        allow_pqat: bool = False,
    ) -> list[dict[str, Any]]:
        selected_models = [
            model_by_key(key)
            for key in (model_keys or [spec.key for spec in MODEL_SPECS])
        ]
        selected_condition_keys = self._expand_condition_keys(condition_keys)
        selected_conditions = [condition_by_key(key) for key in selected_condition_keys]
        all_records: list[dict[str, Any]] = []

        for model_spec in selected_models:
            newly_trained = self._process_one_model_spec(
                model_spec, selected_conditions, ignore_saved, all_records, allow_pqat
            )
            print("-" * 50)
            if newly_trained:
                completed_model_keys = {r["model_key"] for r in all_records}
                if len(completed_model_keys) >= 2:
                    _log(
                        f"[compare] {len(completed_model_keys)} models complete — "
                        "regenerating comparison reports…",
                        after=True,
                    )
                    write_manifest(all_records, self.results_root / "manifest.csv")
                    write_comparison_reports(all_records, self.comparison_root)

        write_manifest(all_records, self.results_root / "manifest.csv")
        write_comparison_reports(all_records, self.comparison_root)
        return all_records

    def _run_condition(
        self,
        model_key: str,
        metric_name: str,
        metric_direction: str,
        bundle: Any,
        condition: ConditionSpec,
        saved_dirs: dict[str, Path],
        allow_pqat: bool,
    ) -> TrainingRecord:
        require_torch()
        model = build_model(model_key, **self._model_kwargs(model_key))
        condition_dir = self.results_root / model_key / condition.key
        pai_config_snapshot = condition_dir / "PAI_config.json"
        if condition.source_key in saved_dirs:
            checkpoint = self._artifact_path(
                saved_dirs[condition.source_key],
                prefer_dendritic="dendrites" in condition.source_key,
            )
            model = self._load_source_checkpoint(
                model,
                model_key,
                condition.source_key,
                checkpoint,
                condition.use_dendrites,
                save_name=f"{model_key}_{condition.key}",
                maximizing_score=metric_direction == "maximize",
                config_snapshot_path=pai_config_snapshot,
            )
        elif condition.use_dendrites:
            model = perforate_model(
                model,
                save_name=f"{model_key}_{condition.key}",
                doing_pai=True,
                maximizing_score=metric_direction == "maximize",
                modules_to_track=self._perforation_track_modules(model_key),
                config_snapshot_path=pai_config_snapshot,
            )
            model = self._configure_perforated_model(model, model_key)

        training_hyperparameters = self._training_hyperparameters(model_key, condition)
        max_epochs = training_hyperparameters.max_epochs
        use_qat = condition.use_qat
        fine_tune_epochs = condition.fine_tune_epochs
        if condition.quantized and condition.source_key != condition.key:
            if allow_pqat:
                fine_tune_epochs = self._pqat_epoch_budget(model_key)
                max_epochs = fine_tune_epochs
                use_qat = True
            else:
                max_epochs = 0
        training_config = TrainingConfig(
            bit_width=condition.bit_width,
            quantization_mode=condition.quantization_mode,
            use_dendrites=condition.use_dendrites,
            use_pruning=condition.use_pruning,
            prune_amount=condition.prune_amount,
            use_qat=use_qat,
            fine_tune_epochs=fine_tune_epochs,
            max_epochs=max_epochs,
            learning_rate=training_hyperparameters.learning_rate,
            optimizer_name=training_hyperparameters.optimizer_name,
            momentum=training_hyperparameters.momentum,
            weight_decay=training_hyperparameters.weight_decay,
            source_condition_key=condition.source_key,
        )
        return train_and_evaluate(
            model_key=model_key,
            condition_key=condition.key,
            display_name=condition.display_name,
            metric_name=metric_name,
            metric_direction=metric_direction,
            model=model,
            bundle=bundle,
            output_dir=condition_dir,
            config=training_config,
        )
