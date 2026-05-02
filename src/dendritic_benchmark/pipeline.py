from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from .compat import (
    PAIModuleSelection,
    attach_module_output_dimensions,
    choose_device,
    latest_pai_switch_checkpoint,
    load_pai_system_checkpoint,
    nn,
    perforate_model,
    require_torch,
    set_module_output_dimensions,
)
from .data import build_task_bundle
from .models import build_model
from .results import (
    save_training_record,
    write_comparison_reports,
    write_manifest,
    write_model_reports,
)
from .specs import CONDITION_SPECS, MODEL_SPECS, ConditionSpec, condition_by_key, model_by_key
from .training import (
    OptimizerName,
    TrainingConfig,
    TrainingRecord,
    infer_module_output_dimensions,
    train_and_evaluate,
)

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


@dataclass(frozen=True)
class ConditionTrainingPlan:
    max_epochs: int
    use_qat: bool
    fine_tune_epochs: int
    update_dendrites_during_training: bool


@dataclass(frozen=True)
class SourceCheckpointLoadConfig:
    save_name: str
    maximizing_score: bool
    module_selection: PAIModuleSelection
    config_snapshot_path: Path | str | None = None
    dendrite_training_max_epochs: int | None = None
    dynamic_dendritic_training: bool = False
    freeze_dendrite_updates_fraction: float = 0.20
    batches_per_epoch: int | None = None
    module_output_dimensions: dict[str, list[int]] | None = None


def _log(msg: str, *, before: bool = False, after: bool = False) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    if before:
        print()
    print(f"[{ts}] {msg}")
    if after:
        print()


def _is_ignorable_state_key(key: str) -> bool:
    return key.endswith("tracker_string")


def _tensor_shape(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(shape)
    except TypeError:
        return None


def _is_compatible_state_value(current_value: Any, source_value: Any) -> bool:
    current_shape = _tensor_shape(current_value)
    source_shape = _tensor_shape(source_value)
    return (
        current_shape is not None
        and source_shape is not None
        and current_shape == source_shape
    )


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

    def _split_compatible_state(
        self, state: dict[str, Any], current_state: dict[str, Any]
    ) -> tuple[dict[str, Any], list[str]]:
        compatible_state: dict[str, Any] = {}
        skipped: list[str] = []
        for key, value in state.items():
            if _is_ignorable_state_key(key):
                continue
            current_value = current_state.get(key)
            if not _is_compatible_state_value(current_value, value):
                skipped.append(key)
                continue
            compatible_state[key] = value
        return compatible_state, skipped

    def _load_compatible_state(self, model: Any, state: dict[str, Any]) -> None:
        compatible_state, skipped = self._split_compatible_state(
            state, model.state_dict()
        )
        model.load_state_dict(compatible_state, strict=False)
        if skipped:
            print(
                "[state] skipped incompatible source-checkpoint tensors: "
                + ", ".join(sorted(skipped)[:5])
                + ("..." if len(skipped) > 5 else "")
            )

    def _load_state(
        self, model: Any, checkpoint_path: Path, *, strict: bool = True
    ) -> Any:
        torch = require_torch()
        if checkpoint_path.exists():
            state = torch.load(checkpoint_path, map_location=choose_device())
            if strict:
                model.load_state_dict(state, strict=True)
            else:
                self._load_compatible_state(model, cast(dict[str, Any], state))
        return model

    def _load_source_checkpoint(
        self,
        model: Any,
        model_key: str,
        source_key: str,
        checkpoint_path: Path,
        target_uses_dendrites: bool,
        load_config: SourceCheckpointLoadConfig,
    ) -> Any:
        source_condition = condition_by_key(source_key)

        # Dendritic checkpoints contain PerforatedAI wrapper keys, so the target model
        # must be perforated before we load them. Base checkpoints still load into the
        # plain model first, then we perforate the model afterward if needed.
        if source_condition.use_dendrites and target_uses_dendrites:
            model = perforate_model(
                model,
                save_name=load_config.save_name,
                doing_pai=True,
                maximizing_score=load_config.maximizing_score,
                module_selection=load_config.module_selection,
                config_snapshot_path=load_config.config_snapshot_path,
                use_runtime_guard=self._use_pai_runtime_guard(),
                dendrite_training_max_epochs=(
                    load_config.dendrite_training_max_epochs
                ),
                dynamic_dendritic_training=load_config.dynamic_dendritic_training,
                freeze_dendrite_updates_fraction=(
                    load_config.freeze_dendrite_updates_fraction
                ),
                batches_per_epoch=load_config.batches_per_epoch,
                no_backward_workaround=False,
            )
            model = self._configure_perforated_model(
                model, load_config.module_output_dimensions
            )
            source_save_name = f"{model_key}_{source_key}"
            pai_checkpoint_name = latest_pai_switch_checkpoint(source_save_name)
            if pai_checkpoint_name is not None:
                model = load_pai_system_checkpoint(
                    model,
                    source_save_name,
                    pai_checkpoint_name,
                )
                model = self._configure_perforated_model(
                    model, load_config.module_output_dimensions
                )
            return self._load_state(model, checkpoint_path, strict=False)

        model = self._load_state(model, checkpoint_path)
        if target_uses_dendrites:
            model = perforate_model(
                model,
                save_name=load_config.save_name,
                doing_pai=True,
                maximizing_score=load_config.maximizing_score,
                module_selection=load_config.module_selection,
                config_snapshot_path=load_config.config_snapshot_path,
                use_runtime_guard=self._use_pai_runtime_guard(),
                dendrite_training_max_epochs=(
                    load_config.dendrite_training_max_epochs
                ),
                dynamic_dendritic_training=load_config.dynamic_dendritic_training,
                freeze_dendrite_updates_fraction=(
                    load_config.freeze_dendrite_updates_fraction
                ),
                batches_per_epoch=load_config.batches_per_epoch,
                no_backward_workaround=False,
            )
            model = self._configure_perforated_model(
                model, load_config.module_output_dimensions
            )
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
        if model_key in {"lenet5", "snn_nmnist", "capsnet_mnist"}:
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

    def _perforation_track_modules(self) -> list[Any]:
        if nn is None:
            return []
        # PerforatedAI is configured on tensor-returning Conv/Linear modules.
        # Recurrent and attention benchmark models expose their gates/projections
        # as explicit Linear layers rather than handing tuple-returning LSTM/GRU
        # or MultiheadAttention modules directly to PAI.
        return [nn.Linear, nn.Conv1d, nn.Conv2d]

    def _perforation_track_only_module_ids(self, model_key: str) -> list[str]:
        return {
            "actor_critic": [".value"],
            "ppo_bipedalwalker": [".critic"],
            "capsnet_mnist": [".decoder.0", ".decoder.2"],
        }.get(model_key, [])

    def _use_pai_runtime_guard(self) -> bool:
        return True

    def _configure_perforated_model(
        self,
        model: Any,
        module_output_dimensions: dict[str, list[int]] | None = None,
    ) -> Any:
        if module_output_dimensions:
            attach_module_output_dimensions(model, module_output_dimensions)
            set_module_output_dimensions(model, module_output_dimensions)
        return model

    def _condition_training_plan(
        self,
        model_key: str,
        condition: ConditionSpec,
        training_hyperparameters: ModelTrainingRecipe,
        allow_pqat: bool,
    ) -> ConditionTrainingPlan:
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
        return ConditionTrainingPlan(
            max_epochs=max_epochs,
            use_qat=use_qat,
            fine_tune_epochs=fine_tune_epochs,
            update_dendrites_during_training=(
                condition.use_dendrites and not condition.quantized
            ),
        )

    def _dendrite_initialization_metadata(
        self,
        model: Any,
        model_key: str,
        bundle: Any,
        condition: ConditionSpec,
    ) -> tuple[PAIModuleSelection, dict[str, list[int]] | None]:
        if not condition.use_dendrites:
            return PAIModuleSelection(), None
        modules_to_perforate = self._perforation_track_modules()
        module_selection = PAIModuleSelection(
            modules_to_perforate=modules_to_perforate,
            track_only_module_ids=self._perforation_track_only_module_ids(model_key),
        )
        module_output_dimensions = infer_module_output_dimensions(
            model, model_key, bundle, modules_to_perforate
        )
        return module_selection, module_output_dimensions

    def _prepare_condition_model(
        self,
        *,
        model: Any,
        model_key: str,
        metric_direction: str,
        condition: ConditionSpec,
        saved_dirs: dict[str, Path],
        pai_config_snapshot: Path,
        training_plan: ConditionTrainingPlan,
        dynamic_dendritic_training: bool,
        batches_per_epoch: int | None,
        module_selection: PAIModuleSelection,
        module_output_dimensions: dict[str, list[int]] | None,
    ) -> Any:
        dendrite_training_max_epochs = (
            training_plan.max_epochs
            if training_plan.update_dendrites_during_training
            else None
        )
        if condition.source_key in saved_dirs:
            checkpoint = self._artifact_path(
                saved_dirs[condition.source_key],
                prefer_dendritic="dendrites" in condition.source_key,
            )
            return self._load_source_checkpoint(
                model,
                model_key,
                condition.source_key,
                checkpoint,
                condition.use_dendrites,
                SourceCheckpointLoadConfig(
                    save_name=f"{model_key}_{condition.key}",
                    maximizing_score=metric_direction == "maximize",
                    module_selection=module_selection,
                    config_snapshot_path=pai_config_snapshot,
                    dendrite_training_max_epochs=dendrite_training_max_epochs,
                    dynamic_dendritic_training=dynamic_dendritic_training,
                    freeze_dendrite_updates_fraction=0.20,
                    batches_per_epoch=batches_per_epoch,
                    module_output_dimensions=module_output_dimensions,
                ),
            )
        if not condition.use_dendrites:
            return model
        model = perforate_model(
            model,
            save_name=f"{model_key}_{condition.key}",
            doing_pai=True,
            maximizing_score=metric_direction == "maximize",
            module_selection=module_selection,
            config_snapshot_path=pai_config_snapshot,
            use_runtime_guard=self._use_pai_runtime_guard(),
            dendrite_training_max_epochs=dendrite_training_max_epochs,
            dynamic_dendritic_training=dynamic_dendritic_training,
            freeze_dendrite_updates_fraction=0.20,
            batches_per_epoch=batches_per_epoch,
            no_backward_workaround=False,
        )
        return self._configure_perforated_model(model, module_output_dimensions)

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
        dynamic_dendritic_training: bool,
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
                dynamic_dendritic_training,
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
        dynamic_dendritic_training: bool,
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
                    dynamic_dendritic_training,
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
        dynamic_dendritic_training: bool = False,
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
                model_spec,
                selected_conditions,
                ignore_saved,
                all_records,
                allow_pqat,
                dynamic_dendritic_training,
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

    @staticmethod
    def _batches_per_epoch(bundle: Any) -> int | None:
        train_loader = getattr(bundle, "train_loader", None)
        if train_loader is None:
            return None
        try:
            return len(train_loader)
        except TypeError:
            return None

    def _run_condition(
        self,
        model_key: str,
        metric_name: str,
        metric_direction: str,
        bundle: Any,
        condition: ConditionSpec,
        saved_dirs: dict[str, Path],
        allow_pqat: bool,
        dynamic_dendritic_training: bool,
    ) -> TrainingRecord:
        require_torch()
        training_hyperparameters = self._training_hyperparameters(model_key, condition)
        training_plan = self._condition_training_plan(
            model_key, condition, training_hyperparameters, allow_pqat
        )
        model = build_model(model_key, **self._model_kwargs(model_key))
        condition_dir = self.results_root / model_key / condition.key
        pai_config_snapshot = condition_dir / "PAI_config.json"
        batches_per_epoch = self._batches_per_epoch(bundle)
        module_selection, module_output_dimensions = (
            self._dendrite_initialization_metadata(model, model_key, bundle, condition)
        )
        model = self._prepare_condition_model(
            model=model,
            model_key=model_key,
            metric_direction=metric_direction,
            condition=condition,
            saved_dirs=saved_dirs,
            pai_config_snapshot=pai_config_snapshot,
            training_plan=training_plan,
            dynamic_dendritic_training=dynamic_dendritic_training,
            batches_per_epoch=batches_per_epoch,
            module_selection=module_selection,
            module_output_dimensions=module_output_dimensions,
        )

        weight_decay = 0.0 if condition.use_dendrites else training_hyperparameters.weight_decay
        training_config = TrainingConfig(
            bit_width=condition.bit_width,
            quantization_mode=condition.quantization_mode,
            use_dendrites=condition.use_dendrites,
            use_pruning=condition.use_pruning,
            prune_amount=condition.prune_amount,
            use_qat=training_plan.use_qat,
            fine_tune_epochs=training_plan.fine_tune_epochs,
            max_epochs=training_plan.max_epochs,
            learning_rate=training_hyperparameters.learning_rate,
            optimizer_name=training_hyperparameters.optimizer_name,
            momentum=training_hyperparameters.momentum,
            weight_decay=weight_decay,
            source_condition_key=condition.source_key,
            enable_pai_dendrite_updates=training_plan.update_dendrites_during_training,
            train_dendrites_until_complete=(
                training_plan.update_dendrites_during_training
                and dynamic_dendritic_training
            ),
            freeze_dendrite_updates_fraction=0.20,
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
