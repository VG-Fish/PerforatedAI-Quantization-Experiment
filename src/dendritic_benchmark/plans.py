"""Immutable experiment-plan value objects.

These types describe policy selected before a model is built or training starts.
Keeping them outside the runner makes plan construction independently testable
and prevents worker/artifact concerns from leaking into recipe definitions.
"""

import json
from dataclasses import dataclass, fields, replace
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


class _ClearSentinel:
    """Singleton meaning "set this recipe field to ``None``", not "leave it".

    An override uses ``None`` to mean "unset -- keep the recipe's own value",
    which leaves no way to express the opposite for the three
    :class:`ModelTrainingRecipe` fields that are themselves nullable. A trial
    that wants to *disable* gradient clipping, the step schedule, or the
    LR-schedule horizon could not be written down at all, so those arms of the
    sweep were unreachable from a JSON file.
    """

    _instance: "_ClearSentinel | None" = None

    def __new__(cls) -> "_ClearSentinel":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "CLEAR"

    def __reduce__(self) -> tuple[Any, ...]:
        return (_ClearSentinel, ())


#: Sentinel for "override this field to ``None``". See :class:`_ClearSentinel`.
CLEAR = _ClearSentinel()

#: How ``CLEAR`` is spelled in an override JSON file and in the ``to_dict``
#: form written to ``metrics.json``. A plain JSON ``null`` cannot carry the
#: meaning: it is already how an unset field is written.
CLEAR_JSON_VALUE = "__null__"

#: The RecipeOverride fields that accept ``CLEAR``. Exactly the
#: ``ModelTrainingRecipe`` fields whose own type admits ``None`` -- clearing
#: any other field would produce a recipe the trainer cannot run.
_CLEARABLE_RECIPE_FIELDS = frozenset(
    {"lr_decay_every", "lr_schedule_epochs", "grad_clip_norm"}
)


def _load_override_json(cls: Any, path: Path | str) -> dict[str, Any]:
    """Parse an override JSON file and reject any key the dataclass has no field for.

    A silently-ignored typo (``"leraning_rate"``) would make a sweep trial
    quietly run with the base recipe instead of the intended override, so an
    unknown key is a hard error naming the offending key(s) rather than a
    warning.
    """
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path}: override file must contain a JSON object")
    valid = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - valid)
    if unknown:
        raise ValueError(
            f"{path}: unknown {cls.__name__} field(s) {unknown}; valid fields are "
            f"{sorted(valid)}"
        )
    return data


@dataclass(frozen=True)
class RecipeOverride:
    """Optional per-sweep-trial overrides layered onto a model's hard-coded
    :class:`ModelTrainingRecipe`.

    Every field is ``None`` by default, meaning "use the recipe's own value".
    This is the ``RecipeOverride`` object required by
    ``information/optimization/03_execution_matrix.md`` so a base-recipe sweep
    trial (e.g. ``R1=(.05,5e-4,.1)``) can be expressed as a JSON file instead of
    a hand-edit to ``BenchmarkRunner._training_hyperparameters``. Load one with
    :meth:`from_json_file` and apply it with :meth:`apply`.

    ``dendrite_lr_min_factor`` lives here rather than on a PAI-side override
    even though the tuning-grid table in
    ``information/optimization/01_initial_five_plan.md`` groups "dendrite LR
    floor" with PAI knobs: it is implemented as a ``ModelTrainingRecipe``
    field, and keeping one field in exactly one override type avoids an
    ambiguous "which override wins" question.
    """

    batch_size: int | None = None
    max_epochs: int | None = None
    learning_rate: float | None = None
    optimizer_name: OptimizerName | None = None
    momentum: float | None = None
    weight_decay: float | None = None
    lr_schedule: LRScheduleName | None = None
    lr_decay_every: int | _ClearSentinel | None = None
    lr_decay_gamma: float | None = None
    lr_min_factor: float | None = None
    lr_schedule_epochs: int | _ClearSentinel | None = None
    dendrite_lr_min_factor: float | None = None
    warmup_epochs: int | None = None
    label_smoothing: float | None = None
    regression_loss: RegressionLossName | None = None
    grad_clip_norm: float | _ClearSentinel | None = None
    nesterov: bool | None = None

    def __post_init__(self) -> None:
        for f in fields(self):
            if (
                isinstance(getattr(self, f.name), _ClearSentinel)
                and f.name not in _CLEARABLE_RECIPE_FIELDS
            ):
                raise ValueError(
                    f"RecipeOverride.{f.name} cannot be cleared to None: only "
                    f"{sorted(_CLEARABLE_RECIPE_FIELDS)} are nullable on "
                    "ModelTrainingRecipe"
                )

    @classmethod
    def from_json_file(cls, path: Path | str) -> "RecipeOverride":
        data = _load_override_json(cls, path)
        for key, value in data.items():
            if value == CLEAR_JSON_VALUE:
                data[key] = CLEAR
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        """The fields this override actually sets, for artifact identity.

        ``CLEAR`` is emitted as :data:`CLEAR_JSON_VALUE` rather than as JSON
        ``null``. This value is written to ``metrics.json`` and read back by
        ``BenchmarkRunner._condition_metadata_current`` to decide whether a
        saved artifact still matches; a ``null`` there would read back as an
        unset field, so a "disable gradient clipping" trial would judge its own
        artifact stale and retrain it on every invocation -- the same failure
        ``PAIOverride.to_dict``'s tuple/list conversion exists to prevent.
        """
        return {
            f.name: (CLEAR_JSON_VALUE if isinstance(value, _ClearSentinel) else value)
            for f in fields(self)
            if (value := getattr(self, f.name)) is not None
        }

    def apply(self, recipe: "ModelTrainingRecipe") -> "ModelTrainingRecipe":
        """Return ``recipe`` with only this override's set fields replaced."""
        changes = {
            f.name: (None if isinstance(value, _ClearSentinel) else value)
            for f in fields(self)
            if (value := getattr(self, f.name)) is not None
        }
        return replace(recipe, **changes) if changes else recipe


# The PAIDynamicSchedule-shaped subset of PAIOverride's fields, in the order
# PAIDynamicSchedule itself declares them. Shared by to_dict-style helpers
# below so the "which fields count as schedule fields" list is written once.
_PAI_SCHEDULE_OVERRIDE_FIELDS = (
    "max_dendrites",
    "n_epochs_to_switch",
    "history_lookback",
    "initial_history_after_switches",
    "p_epochs_to_switch",
    "improvement_threshold",
    "candidate_weight_initialization_multiplier",
)

#: PAIOverride fields the JSON file supplies as arrays and this module stores as
#: tuples (frozen dataclasses must be hashable). ``to_dict`` converts them back
#: to lists before they reach artifact metadata -- see its docstring.
_PAI_SEQUENCE_OVERRIDE_FIELDS = (
    "module_ids_to_perforate",
    "track_only_module_ids",
    "improvement_threshold",
)


@dataclass(frozen=True)
class PAIOverride:
    """Optional per-sweep-trial overrides for one model's PAI configuration.

    Required by ``information/optimization/03_execution_matrix.md`` alongside
    :class:`RecipeOverride`. Covers the "PAI tuning grid" in
    ``information/optimization/01_initial_five_plan.md``: target-module
    selection plus the six :class:`~dendritic_benchmark.compat.PAIDynamicSchedule`
    fields. A ``BenchmarkRunner`` applies at most one ``PAIOverride`` per run,
    and only when exactly one model is selected -- these are per-model sweep
    trials (``RP0``, ``AP1``, ``NP1``, ...), never a blanket override applied
    identically across a multi-model run.
    """

    module_ids_to_perforate: tuple[str, ...] | None = None
    track_only_module_ids: tuple[str, ...] | None = None
    max_dendrites: int | None = None
    n_epochs_to_switch: int | None = None
    history_lookback: int | None = None
    initial_history_after_switches: int | None = None
    p_epochs_to_switch: int | None = None
    improvement_threshold: tuple[float, ...] | None = None
    candidate_weight_initialization_multiplier: float | None = None

    def __post_init__(self) -> None:
        # information/optimization/03_execution_matrix.md: "The initial-history
        # value must equal the lookback when lookback changes, to avoid the
        # known zero-seeded EMA bug." Enforced here rather than left to a
        # downstream PAI warning, since that bug silently corrupts best-epoch
        # tracking rather than raising.
        lookback_set = self.history_lookback is not None
        initial_set = self.initial_history_after_switches is not None
        if lookback_set != initial_set:
            raise ValueError(
                "PAIOverride must set history_lookback and "
                "initial_history_after_switches together, or set neither"
            )
        if lookback_set and self.history_lookback != self.initial_history_after_switches:
            raise ValueError(
                "PAIOverride.initial_history_after_switches "
                f"({self.initial_history_after_switches}) must equal "
                f"history_lookback ({self.history_lookback}) -- see the "
                "zero-seeded EMA bug note in "
                "information/optimization/03_execution_matrix.md"
            )
        # An empty list is not "no override": it is indistinguishable from an
        # unset field everywhere downstream, and for module_ids_to_perforate it
        # is actively dangerous -- BenchmarkRunner._perforation_modules_to_perforate
        # falls back to type-selecting *every* Linear/Conv1d/Conv2d when the ID
        # list is empty, which is the blanket wrapping
        # information/optimization/01_initial_five_plan.md forbids as a primary
        # comparison. Reject it rather than let a "[]" typo widen the target set.
        for name in _PAI_SEQUENCE_OVERRIDE_FIELDS:
            value = getattr(self, name)
            if value is not None and len(value) == 0:
                raise ValueError(
                    f"PAIOverride.{name} must be non-empty when set; omit the "
                    "field to keep the model's default"
                )

    @classmethod
    def from_json_file(cls, path: Path | str) -> "PAIOverride":
        data = _load_override_json(cls, path)
        for key in _PAI_SEQUENCE_OVERRIDE_FIELDS:
            if data.get(key) is not None:
                data[key] = tuple(data[key])
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        """The fields this override actually sets, for artifact identity.

        Sequence fields are emitted as lists, not tuples, exactly as
        :meth:`PAIDynamicSchedule.to_dict` does. This value is written to
        ``metrics.json`` and then read back by
        ``BenchmarkRunner._condition_metadata_current`` to decide whether a
        saved artifact still matches the requested configuration; JSON has no
        tuple, so a tuple here would never compare equal to its own round-trip
        and every rerun of a sweep trial would discard and retrain the artifact
        it had just written.
        """
        return {
            f.name: (list(value) if f.name in _PAI_SEQUENCE_OVERRIDE_FIELDS else value)
            for f in fields(self)
            if (value := getattr(self, f.name)) is not None
        }

    def apply_to_schedule(
        self, base: PAIDynamicSchedule | None
    ) -> PAIDynamicSchedule | None:
        """Merge this override's schedule fields onto ``base``.

        Returns ``base`` unchanged if this override sets no schedule field.
        A field left unset here falls back to ``base``'s own value (``None``
        if ``base`` is itself ``None``), matching how
        :class:`PAIDynamicSchedule` already treats ``None`` as "use the
        global default" via ``compat.PAI_DYNAMIC_SCHEDULE_DEFAULTS``.
        """
        overrides = {
            name: getattr(self, name)
            for name in _PAI_SCHEDULE_OVERRIDE_FIELDS
            if getattr(self, name) is not None
        }
        if not overrides:
            return base
        merged = {
            name: overrides.get(name, getattr(base, name) if base is not None else None)
            for name in _PAI_SCHEDULE_OVERRIDE_FIELDS
        }
        return PAIDynamicSchedule(**merged)

    def resolved_module_ids(
        self, default_perforate: list[str], default_track_only: list[str]
    ) -> tuple[list[str], list[str]]:
        """Return the target/track-only ID lists after this override, if any."""
        perforate = (
            list(self.module_ids_to_perforate)
            if self.module_ids_to_perforate is not None
            else default_perforate
        )
        track_only = (
            list(self.track_only_module_ids)
            if self.track_only_module_ids is not None
            else default_track_only
        )
        return perforate, track_only


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
