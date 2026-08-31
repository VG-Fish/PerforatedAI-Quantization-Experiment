"""Immutable experiment-plan value objects.

These types describe policy selected before a model is built or training starts.
Keeping them outside the runner makes plan construction independently testable
and prevents worker/artifact concerns from leaking into recipe definitions.
"""

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

from .compat import PAIDynamicSchedule, PAIModuleSelection

OptimizerName = Literal["adam", "adamw", "sgd"]
RegressionLossName = Literal["mse", "mae", "smooth_l1"]
LRScheduleName = Literal["constant", "step", "cosine", "linear"]


@dataclass(frozen=True)
class ModelTrainingRecipe:
    batch_size: int
    max_epochs: int
    learning_rate: float
    optimizer_name: OptimizerName = "adam"
    momentum: float = 0.9
    weight_decay: float = 0.0
    lr_schedule: LRScheduleName = "constant"
    lr_decay_every: int | None = None
    lr_decay_gamma: float = 1.0
    lr_min_factor: float = 0.0
    lr_schedule_epochs: int | None = None
    dendrite_lr_min_factor: float = 0.0
    warmup_epochs: int = 0
    label_smoothing: float = 0.0
    regression_loss: RegressionLossName = "mse"
    grad_clip_norm: float | None = None
    nesterov: bool = False

    def with_batch_size(
        self, batch_size: int, *, scale_learning_rate: bool = True
    ) -> "ModelTrainingRecipe":
        """Copy with a per-sample-equivalent learning rate."""
        scale = batch_size / self.batch_size if scale_learning_rate else 1.0
        return replace(
            self, batch_size=batch_size, learning_rate=self.learning_rate * scale
        )


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
    batches_per_epoch: int | None = None
    module_output_dimensions: dict[str, list[int]] | None = None
    candidate_graph_enabled: bool = True
    initial_correlation_batches_limit: int | None = None
    fixed_switch_interval: int | None = None
    dynamic_schedule: PAIDynamicSchedule | None = None


@dataclass(frozen=True)
class ExperimentPlan:
    """Complete immutable identity for one model/condition attempt."""

    artifact_id: str
    model_key: str
    condition_key: str
    source_condition_key: str
    output_dir: Path
    pai_save_name: str
    model_revision: str | None
    dataset_revision: str
    model_scale: float
    seed: int | None
    quantization_evaluation_revision: str | None
    pai_variant: str
    pai_fixed_switch_interval: int | None
    pai_dynamic_schedule: dict[str, Any] | None

    def identity(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "model_key": self.model_key,
            "condition_key": self.condition_key,
            "source_condition_key": self.source_condition_key,
            "output_dir": str(self.output_dir),
            "pai_save_name": self.pai_save_name,
            "model_revision": self.model_revision,
            "dataset_revision": self.dataset_revision,
            "model_scale": self.model_scale,
            "seed": self.seed,
            "quantization_evaluation_revision": (
                self.quantization_evaluation_revision
            ),
            "pai_variant": self.pai_variant,
            "pai_fixed_switch_interval": self.pai_fixed_switch_interval,
            "pai_dynamic_schedule": self.pai_dynamic_schedule,
        }
