from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
from .training import TrainingRecord, train_and_evaluate

EPOCH_MULTIPLIER = 10


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
            model.load_state_dict(state, strict=strict)
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
            )
            model = self._configure_perforated_model(model, model_key)
        return model

    def _artifact_path(
        self, condition_dir: Path, prefer_dendritic: bool = False
    ) -> Path:
        candidates = ["model.pt"]
        if prefer_dendritic:
            candidates = ["final_clean_pai", "best_model", "model.pt"]
        for name in candidates:
            path = condition_dir / name
            if path.exists():
                return path
        return condition_dir / "model.pt"

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

    def _base_epoch_budget(self, condition: ConditionSpec) -> int:
        return 4 * EPOCH_MULTIPLIER

    def run(
        self,
        model_keys: list[str] | None = None,
        condition_keys: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        selected_models = [
            model_by_key(key)
            for key in (model_keys or [spec.key for spec in MODEL_SPECS])
        ]
        selected_condition_keys = self._expand_condition_keys(condition_keys)
        selected_conditions = [condition_by_key(key) for key in selected_condition_keys]
        all_records: list[dict[str, Any]] = []

        for model_spec in selected_models:
            bundle = build_task_bundle(model_spec.key)
            model_records: list[dict[str, Any]] = []
            saved_dirs: dict[str, Path] = {}
            for condition in selected_conditions:
                condition_dir = self.results_root / model_spec.key / condition.key
                record_path = condition_dir / "record.json"
                if record_path.exists():
                    print(
                        f"[skip] {model_spec.key} / {condition.key} "
                        "— record.json found, skipping training."
                    )
                    record = TrainingRecord(**json.loads(record_path.read_text()))
                else:
                    record = self._run_condition(
                        model_spec.key,
                        model_spec.display_name,
                        model_spec.metric_name,
                        model_spec.metric_direction,
                        bundle,
                        condition,
                        saved_dirs,
                    )
                    save_training_record(record, condition_dir)
                model_records.append(record.to_dict())
                all_records.append(record.to_dict())
                saved_dirs[condition.key] = condition_dir
            write_model_reports(
                model_spec.display_name,
                model_records,
                self.results_root / model_spec.key,
            )

            # Eagerly regenerate comparison outputs as soon as at least 2 models have
            # finished training, so results are visible without waiting for all models.
            completed_model_keys = {r["model_key"] for r in all_records}
            if len(completed_model_keys) >= 2:
                print(
                    f"[compare] {len(completed_model_keys)} models complete — "
                    "regenerating comparison reports…"
                )
                write_manifest(all_records, self.results_root / "manifest.csv")
                write_comparison_reports(all_records, self.comparison_root)

        # Final write covers the single-model case and ensures the manifest and
        # comparison outputs always reflect every completed model.
        write_manifest(all_records, self.results_root / "manifest.csv")
        write_comparison_reports(all_records, self.comparison_root)
        return all_records

    def _run_condition(
        self,
        model_key: str,
        display_name: str,
        metric_name: str,
        metric_direction: str,
        bundle: Any,
        condition: ConditionSpec,
        saved_dirs: dict[str, Path],
    ) -> TrainingRecord:
        torch = require_torch()
        model = build_model(model_key, **self._model_kwargs(model_key))
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
            )
        elif condition.use_dendrites:
            model = perforate_model(
                model,
                save_name=f"{model_key}_{condition.key}",
                doing_pai=True,
                maximizing_score=metric_direction == "maximize",
                modules_to_track=self._perforation_track_modules(model_key),
            )
            model = self._configure_perforated_model(model, model_key)

        return train_and_evaluate(
            model_key=model_key,
            condition_key=condition.key,
            display_name=condition.display_name,
            metric_name=metric_name,
            metric_direction=metric_direction,
            model=model,
            bundle=bundle,
            output_dir=self.results_root / model_key / condition.key,
            bit_width=condition.bit_width,
            quantization_mode=condition.quantization_mode,
            use_dendrites=condition.use_dendrites,
            use_pruning=condition.use_pruning,
            prune_amount=condition.prune_amount,
            use_qat=condition.use_qat,
            fine_tune_epochs=condition.fine_tune_epochs,
            max_epochs=self._base_epoch_budget(condition),
            source_condition_key=condition.source_key,
        )
