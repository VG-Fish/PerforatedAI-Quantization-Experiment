import functools
import gc
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, cast

import torch
import torch.nn as nn

from .compat import (
    PAI_ARTIFACT_NAME,
    PAI_DIRECTORY_NAME,
    PAI_RESUME_NAME,
    PAIDynamicSchedule,
    PAIModuleSelection,
    PAIRuntimeOptions,
    attach_module_output_dimensions,
    choose_device,
    configure_pai_candidate_graph,
    latest_pai_switch_checkpoint,
    load_pai_system_checkpoint,
    pai_system_checkpoint_exists,
    perforate_model,
    seed_everything,
    set_module_output_dimensions,
    set_pai_root,
)
from .data import build_task_bundle
from .log_utils import validate_output_path
from .models import ADULT_CATEGORICAL_CARDINALITIES, build_model
from .results import (
    load_training_records,
    save_training_record,
    write_comparison_reports,
    write_manifest,
    write_model_reports,
)
from .specs import CONDITION_SPECS, MODEL_SPECS, ConditionSpec, condition_by_key, model_by_key
from .training import (
    DENDRITE_AUDIT_REVISION,
    LRScheduleName,
    OptimizerName,
    QUANTIZATION_EVALUATION_REVISION,
    RegressionLossName,
    TrainingConfig,
    TrainingRecord,
    infer_module_output_dimensions,
    train_and_evaluate,
)

EPOCH_MULTIPLIER = 10
_RECORD_JSON = "record.json"
_MODEL_PT = "model.pt"
_DEFAULT_PAI_INITIAL_CORRELATION_BATCH_LIMIT = 32
_MODEL_PAI_INITIAL_CORRELATION_BATCH_LIMITS = {
    "distilbert": 4,
}
_MODEL_DENDRITIC_BATCH_SIZES = {
    "distilbert": 4,
}
_DEFAULT_DENDRITIC_MEMORY_CLEANUP_INTERVAL = 512
_MODEL_DENDRITIC_MEMORY_CLEANUP_INTERVALS = {
    "distilbert": 128,
}
# PAI's HISTORY mode needs n_epochs_to_switch (10) quiet epochs to decide a switch,
# and its running average keeps resetting that counter (see
# _configure_interval_pai_schedule). At DistilBERT's ~23 min/epoch the 2026-07-29
# run never got past switch 0 in 40 epochs. Models listed here switch on a fixed
# interval instead, sized to the epochs their baseline recipe needs to converge.
_MODEL_DENDRITIC_FIXED_SWITCH_INTERVALS = {
    "distilbert": 3,
    # Dynamic10's TCN reached its best validation epoch at 9, while HISTORY
    # mode first restructured at epoch 20. Insert the single permitted
    # dendrite early enough for it to affect the selected checkpoint.
    "tcn_forecaster": 6,
    # Dynamic11's gate search began after the GRU cosine had effectively
    # bottomed out.  The decoder-projection search now switches while the
    # weather model still has time to adapt.
    "gru_forecaster": 8,
    # The old VAE candidate did not arrive until epoch 86 of a 50-epoch cosine
    # recipe, so it trained at the LR floor.  Give its decoder target a single,
    # early fixed switch and a shared 150-epoch adaptation budget.
    "vae_mnist": 20,
}
_PAI_VARIANTS = frozenset(
    {
        "default",
        "gru_gate_ablation",
        "mpnn_capacity",
        "tcn_head_both",
        "tcn_head_output",
        "vae_latent",
    }
)
# A revision is part of an artifact's identity, just like model_scale. It keeps
# a prior Dynamic11 record from being reused after an architecture, target, or
# optimization change that record cannot represent.
_MODEL_ARTIFACT_REVISIONS = {
    "tcn_forecaster": "dynamic11_targeted_head_v2",
    "gru_forecaster": "dynamic11_multiscale_decoder_v2",
    "vae_mnist": "dynamic11_fair_ternary_v2",
    "resnet18_cifar10": "dynamic12_prefc_target_v1",
    "saint_adult": "dynamic12_row_qkv_target_v1",
    "pointnet_modelnet40": "dynamic12_late_feature_target_v1",
}
# Dynamic9 showed that the global three-dendrite schedule adds unnecessary
# capacity to several models. These are deliberately sparse overrides; fields
# omitted here keep the well-tested global defaults in compat.py.
_MODEL_DYNAMIC_PAI_SCHEDULES = {
    "gcn": PAIDynamicSchedule(max_dendrites=1, p_epochs_to_switch=6),
    "actor_critic": PAIDynamicSchedule(max_dendrites=2, p_epochs_to_switch=6),
    "lenet5": PAIDynamicSchedule(max_dendrites=1),
    "tcn_forecaster": PAIDynamicSchedule(max_dendrites=1),
    "mpnn": PAIDynamicSchedule(max_dendrites=3),
    "vae_mnist": PAIDynamicSchedule(max_dendrites=1, p_epochs_to_switch=6),
    "gru_forecaster": PAIDynamicSchedule(max_dendrites=1, p_epochs_to_switch=6),
    # Keep the first three replacement experiments capacity-conservative.  A
    # single retained dendrite makes a parameter-matched dense control practical
    # if any configuration shows a credible gain.  Note this diverges from
    # upstream for ResNet-18: PerforatedAI's published model retains 5
    # dendrites on .pre_fc (--dendrite-mode 1 -> max_dendrites=5), so a result
    # here is not a reproduction of their ImageNet number even before the
    # dataset difference.
    "resnet18_cifar10": PAIDynamicSchedule(max_dendrites=1, p_epochs_to_switch=10),
    "saint_adult": PAIDynamicSchedule(max_dendrites=1, p_epochs_to_switch=10),
    "pointnet_modelnet40": PAIDynamicSchedule(max_dendrites=1, p_epochs_to_switch=10),
}
_PAI_VARIANT_SCHEDULES = {
    "mpnn_capacity": {
        "mpnn": PAIDynamicSchedule(max_dendrites=4),
    },
    # Retain the recurrent-gate configuration only as a labelled ablation; it
    # must not silently stand in for the new decoder-target default.
    "gru_gate_ablation": {
        "gru_forecaster": PAIDynamicSchedule(max_dendrites=1, n_epochs_to_switch=8),
    },
}
# Full-transformer PAI wrapping makes DistilBERT's candidate forward exceed
# Apple Silicon MPS memory. Keep dendrite search on the task-specific head.
_DISTILBERT_PAI_CLASSIFICATION_HEAD = [
    ".model.pre_classifier",
    ".model.classifier",
]


@dataclass(frozen=True)
class ModelTrainingRecipe:
    batch_size: int
    max_epochs: int
    learning_rate: float
    optimizer_name: OptimizerName = "adam"
    momentum: float = 0.9
    weight_decay: float = 0.0
    # "constant" | "step" | "cosine" | "linear". See LRScheduleName in training.py.
    lr_schedule: LRScheduleName = "constant"
    # Step decay: multiply lr by lr_decay_gamma every lr_decay_every epochs.
    # Only read when lr_schedule == "step".
    lr_decay_every: int | None = None
    lr_decay_gamma: float = 1.0
    # Floor for "cosine"/"linear", as a fraction of learning_rate.
    lr_min_factor: float = 0.0
    # Dynamic PAI may legitimately continue beyond max_epochs while a
    # candidate completes.  Decay over this explicit horizon instead of
    # prematurely pinning the learning rate at its floor.
    lr_schedule_epochs: int | None = None
    # Linear warmup ramp, in epochs, applied under every schedule.
    warmup_epochs: int = 0
    label_smoothing: float = 0.0
    regression_loss: RegressionLossName = "mse"
    grad_clip_norm: float | None = None
    nesterov: bool = False

    def with_batch_size(
        self, batch_size: int, *, scale_learning_rate: bool = True
    ) -> "ModelTrainingRecipe":
        """Copy at a different batch size, keeping the per-sample step constant.

        Dendrite conditions run at a smaller batch so the candidate forward fits
        in memory. Carrying the learning rate over unchanged multiplies the
        per-sample step by ``batch_size / dendritic_batch_size``, which is what
        diverged DistilBERT into a constant classifier on its second epoch
        (AdamW at 1e-4, batch 32 -> 4: an 8x effective step, on a recipe already
        5x the canonical 2e-5 for BERT fine-tuning).
        """
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
    dynamic_dendritic_training: bool = False
    freeze_dendrite_updates_fraction: float = 0.20
    batches_per_epoch: int | None = None
    module_output_dimensions: dict[str, list[int]] | None = None
    candidate_graph_enabled: bool = True
    initial_correlation_batches_limit: int | None = None
    fixed_switch_interval: int | None = None
    dynamic_schedule: PAIDynamicSchedule | None = None


def _log(msg: str, *, before: bool = False, after: bool = False) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    if before:
        print()
    print(f"[{ts}] {msg}")
    if after:
        print()


def _release_accelerator_memory() -> None:
    gc.collect()
    mps = getattr(torch, "mps", None)
    if mps is not None and torch.backends.mps.is_available():
        empty = getattr(mps, "empty_cache", None)
        if empty is not None:
            empty()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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
        *,
        model_scale: float = 1.0,
        pai_variant: str = "default",
    ):
        if not 0 < model_scale <= 1:
            raise ValueError("model_scale must be greater than zero and at most one")
        if pai_variant not in _PAI_VARIANTS:
            choices = ", ".join(sorted(_PAI_VARIANTS))
            raise ValueError(f"Unknown PAI variant {pai_variant!r}; choose one of {choices}")
        self.results_root = validate_output_path(Path(results_root), label="results_root")
        self.comparison_root = validate_output_path(Path(comparison_root), label="comparison_root")
        self.results_root.mkdir(parents=True, exist_ok=True)
        self.comparison_root.mkdir(parents=True, exist_ok=True)
        # Keep PAI artifacts with the results they belong to. The CLI already
        # does this; repeating it here covers programmatic callers.
        set_pai_root(self.results_root / PAI_DIRECTORY_NAME)
        # Overwritten by run(); defined here so _train_pending_condition can be
        # called directly by programmatic callers that never went through run().
        self._seed: int | None = None
        self._model_scale = model_scale
        self._pai_variant = pai_variant

    def _split_compatible_state(
        self, state: dict[str, Any], current_state: dict[str, Any]
    ) -> tuple[dict[str, Any], list[str]]:
        """Split a source ``state_dict`` into the part that fits ``current_state``
        and the keys that do not.

        Mismatches are reported in *both* directions. Iterating only over
        ``state`` catches source tensors the target has no slot for, but is
        blind to the opposite and more damaging case: target parameters the
        source never supplies. Those survive ``load_state_dict(strict=False)``
        at whatever ``perforate_model`` initialized them to, so an untrained,
        randomly-initialized dendrite gets quantized and scored as though it
        were part of the trained model. ``actor_critic`` and ``m5`` shipped a
        whole phantom dendrite this way. See MEASUREMENT_CAVEATS.md #5.
        """
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
        for key in current_state:
            if _is_ignorable_state_key(key) or key in state:
                continue
            skipped.append(key)
        return compatible_state, skipped

    def _load_compatible_state(self, model: Any, state: dict[str, Any]) -> None:
        """Load a dendritic source checkpoint's weights onto a model whose
        structure was independently reconstructed from PAI's *latest* switch
        checkpoint (see ``_load_source_checkpoint`` below).

        Those two things should always agree: with the fix in
        ``training.py::_load_compatible_best_state``, ``state`` (loaded from
        ``model.pt``) is always self-consistent -- it is either the true
        best-epoch model or the true final-epoch model, never a shape-filtered
        hybrid. If reconstructing from the latest switch checkpoint still
        doesn't match it, that means the two independent checkpointing systems
        (this benchmark's own best-epoch tracking vs. PAI's own switch
        checkpoints) have genuinely diverged -- see
        information/MEASUREMENT_CAVEATS.md #3. Previously this was a silent
        partial load, which produced a `dendrites_q*` record with a different
        (and wrong) param_count than its own `dendrites_fp32` record. Failing
        loudly here instead surfaces the mismatch at the point it happens,
        rather than downstream in a comparison table.
        """
        current_state = model.state_dict()
        compatible_state, skipped = self._split_compatible_state(state, current_state)
        if skipped:
            unfilled = sorted(k for k in skipped if k not in state)
            mismatched = sorted(k for k in skipped if k in state)
            detail = []
            if unfilled:
                detail.append(
                    f"{len(unfilled)} target tensor(s) the source never supplies "
                    f"(would stay at init: {', '.join(unfilled[:3])}"
                    + ("..." if len(unfilled) > 3 else "")
                    + ")"
                )
            if mismatched:
                detail.append(
                    f"{len(mismatched)} shape mismatch(es) "
                    f"({', '.join(mismatched[:3])}"
                    + ("..." if len(mismatched) > 3 else "")
                    + ")"
                )
            raise RuntimeError(
                "[state] source-checkpoint structure does not match the "
                "PAI-reconstructed model -- refusing a partial load. "
                + "; ".join(detail)
                + ". See information/MEASUREMENT_CAVEATS.md #3 and #5."
            )
        model.load_state_dict(compatible_state, strict=False)

    def _load_state(
        self, model: Any, checkpoint_path: Path, *, strict: bool = True
    ) -> Any:
        if checkpoint_path.exists():
            # weights_only=True: this file only ever holds a plain
            # state_dict() of tensors (written in training._persist_stage_artifacts),
            # so the restricted unpickler is safe here and closes off arbitrary
            # code execution from a malicious/corrupted checkpoint.
            state = torch.load(checkpoint_path, map_location=choose_device(), weights_only=True)
            if strict:
                model.load_state_dict(state, strict=True)
            else:
                self._load_compatible_state(model, cast(dict[str, Any], state))
        return model

    def _pai_save_name(self, model_key: str, condition_key: str) -> str:
        if model_key == "distilbert" and "dendrites" in condition_key:
            return f"{model_key}_{condition_key}_head_only"
        return f"{model_key}_{condition_key}"

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
                dendrite_training_max_epochs=(
                    load_config.dendrite_training_max_epochs
                ),
                dynamic_dendritic_training=load_config.dynamic_dendritic_training,
                freeze_dendrite_updates_fraction=(
                    load_config.freeze_dendrite_updates_fraction
                ),
                batches_per_epoch=load_config.batches_per_epoch,
                runtime_options=PAIRuntimeOptions(
                    use_runtime_guard=self._use_pai_runtime_guard(),
                    candidate_graph_enabled=load_config.candidate_graph_enabled,
                    initial_correlation_batches_limit=(
                        load_config.initial_correlation_batches_limit
                    ),
                    fixed_switch_interval=load_config.fixed_switch_interval,
                    dynamic_schedule=load_config.dynamic_schedule,
                ),
            )
            model = self._configure_perforated_model(
                model, load_config.module_output_dimensions
            )
            source_save_name = self._pai_save_name(model_key, source_key)
            pai_checkpoint_name = self._source_pai_checkpoint_name(source_save_name)
            if pai_checkpoint_name is not None:
                model = load_pai_system_checkpoint(
                    model,
                    source_save_name,
                    pai_checkpoint_name,
                )
                model = self._configure_perforated_model(
                    model, load_config.module_output_dimensions
                )
                configure_pai_candidate_graph(load_config.candidate_graph_enabled)
            model = self._load_state(model, checkpoint_path, strict=False)
            configure_pai_candidate_graph(load_config.candidate_graph_enabled)
            return model

        model = self._load_state(model, checkpoint_path)
        if target_uses_dendrites:
            model = perforate_model(
                model,
                save_name=load_config.save_name,
                doing_pai=True,
                maximizing_score=load_config.maximizing_score,
                module_selection=load_config.module_selection,
                config_snapshot_path=load_config.config_snapshot_path,
                dendrite_training_max_epochs=(
                    load_config.dendrite_training_max_epochs
                ),
                dynamic_dendritic_training=load_config.dynamic_dendritic_training,
                freeze_dendrite_updates_fraction=(
                    load_config.freeze_dendrite_updates_fraction
                ),
                batches_per_epoch=load_config.batches_per_epoch,
                runtime_options=PAIRuntimeOptions(
                    use_runtime_guard=self._use_pai_runtime_guard(),
                    candidate_graph_enabled=load_config.candidate_graph_enabled,
                    initial_correlation_batches_limit=(
                        load_config.initial_correlation_batches_limit
                    ),
                    fixed_switch_interval=load_config.fixed_switch_interval,
                    dynamic_schedule=load_config.dynamic_schedule,
                ),
            )
            model = self._configure_perforated_model(
                model, load_config.module_output_dimensions
            )
            configure_pai_candidate_graph(load_config.candidate_graph_enabled)
        return model

    def _source_pai_checkpoint_name(self, source_save_name: str) -> str | None:
        """Return the PAI checkpoint that should reconstruct a source artifact.

        Post-training quantized dendritic conditions load the benchmark's
        ``model.pt`` from their FP32 source. For dynamic PAI runs, the latest
        ``switch_N`` checkpoint can be structurally stale by the time training
        exits; the final/resume snapshot is the one saved alongside the final
        benchmark checkpoint and should therefore be tried first. ``switch_N``
        remains the fallback for older results that predate final snapshots.

        ``PAI_ARTIFACT_NAME`` is preferred over all of them: it is written at
        the one instant that is guaranteed to describe ``model.pt``, whereas
        ``PAI_RESUME_NAME`` is written inside the epoch loop and can describe a
        structure the artifact never had. See MEASUREMENT_CAVEATS.md #5.
        """
        for checkpoint_name in (
            PAI_ARTIFACT_NAME,
            PAI_RESUME_NAME,
            "latest",
            "best_model",
            "final_clean_pai",
        ):
            if pai_system_checkpoint_exists(source_save_name, checkpoint_name):
                return checkpoint_name
        return latest_pai_switch_checkpoint(source_save_name)

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
        kwargs: dict[str, Any] = {"model_scale": self._model_scale}
        if model_key in {"lenet5", "snn_nmnist", "capsnet_mnist"}:
            kwargs["num_classes"] = 10
        elif model_key == "m5":
            kwargs["num_classes"] = 12
        elif model_key == "textcnn":
            kwargs["num_classes"] = 4
        elif model_key == "gcn":
            kwargs["num_classes"] = 7
        elif model_key in {"tabnet", "saint_adult"}:
            # Adult's 8 nominal columns reach the models as integer codes; both
            # size embedding tables from this schema rather than treating a code
            # as a number. See TabularColumnEmbedding.
            kwargs.update(
                {
                    "num_classes": 2,
                    "categorical_cardinalities": ADULT_CATEGORICAL_CARDINALITIES,
                }
            )
        elif model_key == "gin_imdbb":
            kwargs["num_classes"] = 2
        elif model_key == "pointnet_modelnet40":
            kwargs["num_classes"] = 40
        elif model_key == "distilbert":
            kwargs["num_classes"] = 2
        return kwargs

    def _perforation_track_modules(self) -> list[Any]:
        if nn is None:
            return []
        # PerforatedAI is configured on tensor-returning Conv/Linear modules.
        # Recurrent and attention benchmark models expose their gates/projections
        # as explicit Linear layers rather than handing tuple-returning LSTM/GRU
        # or MultiheadAttention modules directly to PAI.
        return [nn.Linear, nn.Conv1d, nn.Conv2d]

    def _perforation_modules_to_perforate(self, model_key: str) -> list[Any]:
        if self._perforation_module_ids_to_perforate(model_key):
            return []
        return list(self._perforation_track_modules())

    def _perforation_module_ids_to_perforate(self, model_key: str) -> list[str]:
        if model_key == "resnet18_cifar10":
            # Mirrors upstream exactly. PerforatedAI's published ResNet-18
            # (LPA.ResNetPAIPreFC; examples/imagenet/resnet_prefc.py, and the
            # perforated-ai/resnet-18-perforated-gd card) adds a 512 -> 512
            # pre-classifier projection after global pooling and perforates
            # only that, leaving the residual backbone tracked. Their training
            # command is `--convert-count 0`, which tracks .layer1..4 outright,
            # plus .conv1/.bn1/.fc -- so .pre_fc is the sole perforated module.
            # Our dense arm carries the same layer, so the comparison isolates
            # the retained dendrite rather than the extra dense projection.
            return [".pre_fc"]
        if model_key == "saint_adult":
            # Not a new selection: this is exactly the complement of the
            # track-only list SAINT already carried, stated explicitly so the
            # target set stops depending on which modules happen to be
            # Linear/Conv. Verified equivalent -- both spellings wrap
            # {.head.1, .row_blocks.0.attn.qkv, .row_blocks.1.attn.qkv} and
            # add the same 29,120 parameters. The stored PBScores from the
            # top10 run back the set: .row_blocks.0.attn.qkv peaks at 0.092
            # and .head.1 at 0.060 correlation, both well over the 0.02
            # "good placement" bar in the perforatedai-analyze skill.
            return [
                ".row_blocks.0.attn.qkv",
                ".row_blocks.1.attn.qkv",
                ".head.1",
            ]
        if model_key == "pointnet_modelnet40":
            # Late per-point features and the first classifier projection are
            # wide, tensor-returning layers.  The two T-Nets are deliberately
            # excluded: a dendrite there perturbs the learned coordinate basis
            # every downstream point is expressed in, so it is not the local
            # change a late-layer dendrite is.  They are also where the
            # parameters are -- perforating all 18 eligible Linear/Conv modules
            # (what type-based registration did) costs +3,459,569 parameters,
            # +100%, for one dendrite, against +656,896 (+18.9%) for these two.
            return [".conv3.0", ".head.0"]
        if model_key == "distilbert":
            return list(_DISTILBERT_PAI_CLASSIFICATION_HEAD)
        if model_key == "mpnn":
            return [
                ".readout.0",
                ".readout_gate",
                ".layers.2.update.hidden_gates",
                ".layers.2.update.input_gates",
                ".layers.3.update.hidden_gates",
                ".layers.3.update.input_gates",
            ]
        if model_key == "vae_mnist":
            if self._pai_variant == "vae_latent":
                return [".mu", ".logvar"]
            return [".decoder.4"]
        if model_key == "tcn_forecaster":
            # Dynamic11's fresh PBscores favour the input projection of the
            # nonlinear head.  Keep alternate output/both-target runs explicit
            # so their added capacity never contaminates the default result.
            if self._pai_variant == "tcn_head_output":
                return [".head.3"]
            if self._pai_variant == "tcn_head_both":
                return [".head.0", ".head.3"]
            return [".head.0"]
        if model_key == "gru_forecaster":
            if self._pai_variant == "gru_gate_ablation":
                return [".cells.0.input_gates", ".cells.1.input_gates"]
            # The recurrent-gate run did not retain useful capacity and its
            # candidates trained after the cosine had decayed.  The new
            # multiscale decoder exposes this narrow bottleneck exactly once;
            # a dendrite here is cheap, survives final cleanup, and can affect
            # every forecasted timestep.
            return [".head.1"]
        return []

    def _perforation_track_only_module_ids(self, model_key: str) -> list[str]:
        return {
            # Dynamic9's only clearly efficient actor-critic candidate was the
            # second shared-backbone projection. Holding the policy/value heads
            # out also avoids changing outputs represented in an active buffer.
            "actor_critic": [".value", ".backbone.0", ".policy"],
            # Dynamic9's useful LeNet signal came from the classifier. Keeping
            # both convolutions tracked-only turns this into a classifier-only
            # dendrite experiment rather than spending capacity on early maps.
            "lenet5": [".features.0", ".features.3"],
            # Mirror PerforatedAI's ResNet wrapper: retain the entire residual
            # backbone as ordinary neurons and restrict candidate growth to the
            # added pre-classifier projection.
            "resnet18_cifar10": [
                ".conv1",
                ".bn1",
                ".layer1",
                ".layer2",
                ".layer3",
                ".layer4",
                ".fc",
            ],
            # Dendrites go on the shared .backbone only. The two heads are held
            # out for a reason specific to on-policy training, not for the old
            # PBScore reason (which measured a behaviour-cloning run in which
            # .critic received no gradient at all and is therefore void):
            # inserting a dendrite mid-run changes a module's output the moment
            # it switches in, and both heads sit at a point where a step change
            # invalidates data already collected. A jump in .actor_mean moves
            # the policy outside PPO's clip range (0.18) against the log-probs
            # the current buffer was recorded under, so the surrogate saturates
            # and stops producing gradient; a jump in .critic changes the value
            # baseline the buffer's advantages were computed against. Behind
            # them, the backbone can absorb capacity without either effect.
            "ppo_bipedalwalker": [".critic", ".actor_mean"],
            "gru_forecaster": (
                [
                    ".cells.0.hidden_gates",
                    ".cells.1.hidden_gates",
                    ".head",
                ]
                if self._pai_variant == "gru_gate_ablation"
                else [
                    ".cells.0.input_gates",
                    ".cells.0.hidden_gates",
                    ".cells.1.input_gates",
                    ".cells.1.hidden_gates",
                    ".head.4",
                ]
            ),
            # MobileNetV2's final classifier Linear sits inside nn.Sequential(Dropout, Linear),
            # and the initial Conv2d sits inside Conv2dNormActivation (also an nn.Sequential);
            # perforating either leaves DendriteValueTracker.shape uninitialized at the PA switch.
            "mobilenetv2_cifar10": [".classifier.1", ".features.0.0"],
            # Every conv inside the 4 TemporalBlocks (.net.*) scored 0.017-0.023
            # Best-PBScore on the 2026-08-03 dynamic run, 7-10x below .head's
            # 0.166; perforating all of them just spreads the max_dendrites=3
            # budget across near-noise candidates instead of the layer that
            # actually correlates with the learning signal.
            "tcn_forecaster": (
                [".net", ".head.0"]
                if self._pai_variant == "tcn_head_output"
                else [".net", ".head.3"]
                if self._pai_variant != "tcn_head_both"
                else [".net"]
            ),
            # gcn is deliberately absent, and it is worth recording why so nobody adds
            # it back on the same reasoning that failed here.
            #
            # .conv1 holds 91,776 of GCN's 92,231 parameters (99.5%) yet scored the
            # *lower* Best-PBScore on the 2026-08-03 dynamic run -- 0.0635 against
            # .conv2's 0.0925. Both the perforatedai-analyze PBScore guidance and the
            # "convert only top layers" advice in the perforatedai skill therefore say
            # to hold .conv1 out and let dendrites go to the 455-parameter .conv2.
            #
            # Measured, that is wrong. Three paired dqb runs per arm, dynamic mode,
            # comparing dendrites_fp32 against the same run's own base_fp32:
            #
            #   .conv1 track-only   +0.0030 +0.0050 +0.0040  -> +0.40pp, 1.01x params
            #   .conv1 perforated   +0.0190 +0.0290 +0.0220  -> +2.33pp, 3.67x params
            #
            # Welch t ~ 6.4 on ~2 dof against a 4.30 critical value. Excluding .conv1
            # cuts the parameter cost 3.6x but keeps only 17% of the accuracy gain, so
            # nearly all of the dendritic benefit comes from the layer PBScore ranked
            # *lower*. PBScore ranks candidates within a layer's own signal; it does not
            # predict how much headroom a layer has to absorb new capacity, and on a
            # 1433-input bag-of-words first layer that headroom is where the gain is.
            # See experiments/dynamic5/config/PERFORATION.md.
            # Only .head.1 (0.229) and the row_blocks attn.qkv layers (0.10-0.15)
            # cleared the >0.02 PBScore bar the perforatedai-analyze guidance
            # treats as "efficient dendrite user" on the 2026-08-03 dynamic run;
            # every column_blocks/ffn/attn.out/feature_embed/head.3 layer sat at
            # 0.009-0.03. Track the ones that showed no real signal.
            # The row blocks' LayerNorms and the head's input LayerNorm
            # (`.head.0`) were previously in neither list, so their 640
            # parameters carried no parameter_type -- the same p-phase warning
            # and pdb.set_trace path documented for distilbert below. They are
            # leaf norms with no perforated child, so tracking the module is
            # the right remedy here rather than parameter_ids_to_track.
            "saint_adult": [
                ".feature_embed",
                ".column_blocks",
                ".row_blocks.0.attn.out",
                ".row_blocks.0.ffn",
                ".row_blocks.0.norm1",
                ".row_blocks.0.norm2",
                ".row_blocks.1.attn.out",
                ".row_blocks.1.ffn",
                ".row_blocks.1.norm1",
                ".row_blocks.1.norm2",
                ".head.0",
                ".head.3",
            ],
            # Every parameter-bearing head child except the perforated
            # `.head.0`. `.head.1` (BatchNorm1d) and `.head.4` (the 512->256
            # Linear, 131k parameters) have to be named explicitly: a parameter
            # in neither list gets no parameter_type, which PAI warns on for
            # every p-phase step and follows with pdb.set_trace -- fatal in a
            # non-interactive worker. `.head.{2,3,6,7}` are ReLU/Dropout and
            # hold no parameters, so they are omitted rather than tracked.
            "pointnet_modelnet40": [
                ".input_transform",
                ".conv1",
                ".feature_transform",
                ".conv2",
                ".conv3.1",
                ".head.1",
                ".head.4",
                ".head.5",
                ".head.8",
            ],
            # Only the classification head is perforated (see
            # _DISTILBERT_PAI_CLASSIFICATION_HEAD), which leaves all 100 backbone
            # parameters neither perforated nor tracked. PAI cannot assign those a
            # parameter_type, so in p-phase it warns on each one and calls
            # pdb.set_trace. Tracking the backbone tags them "neuron" instead.
            "distilbert": [".model.distilbert"],
        }.get(model_key, [])

    def _perforation_module_names_to_not_save(self, model_key: str) -> list[str]:
        # HuggingFace's DistilBertForSequenceClassification exposes both
        # `.distilbert` and `.base_model` pointing at the same submodule;
        # PAI requires one of the duplicate pointers be excluded from saving.
        return {
            "distilbert": [".model.base_model"],
        }.get(model_key, [])

    def _perforation_parameter_ids_to_track(self, model_key: str) -> list[str]:
        """Parameters PAI must type explicitly because no wrapper owns them.

        Named as ``model.named_parameters()`` reports them but with a leading
        dot, matching the module-id convention: PAI validates these through the
        same checker and rejects ``conv1.bias`` with "Module ID 'conv1.bias'
        must start with '.'".
        """
        return {
            # GraphConv holds its bias itself so that the propagation can be
            # applied to `x @ W` alone and the bias added afterwards, which is
            # Kipf & Welling's formulation and 15x cheaper on the full graph.
            # The child `.linear` is perforated, so GraphConv itself cannot be
            # tracked, which leaves these two biases untyped and warning on
            # every p-phase step. See PAIModuleSelection.parameter_ids_to_track.
            "gcn": [".conv1.bias", ".conv2.bias"],
            # This learned positional term is a raw parameter rather than a
            # child module, so tag it explicitly when the selected SAINT
            # projections are perforated.
            "saint_adult": [".column_embedding"],
        }.get(model_key, [])

    def _use_pai_runtime_guard(self) -> bool:
        return True

    def _pai_initial_correlation_batches_limit(self, model_key: str) -> int:
        return _MODEL_PAI_INITIAL_CORRELATION_BATCH_LIMITS.get(
            model_key, _DEFAULT_PAI_INITIAL_CORRELATION_BATCH_LIMIT
        )

    def _pai_fixed_switch_interval(self, model_key: str) -> int | None:
        if model_key == "gru_forecaster" and self._pai_variant == "gru_gate_ablation":
            return None
        return _MODEL_DENDRITIC_FIXED_SWITCH_INTERVALS.get(model_key)

    def _pai_dynamic_schedule(self, model_key: str) -> PAIDynamicSchedule | None:
        """Return the measured schedule for a model and optional ablation variant."""
        variant_schedule = _PAI_VARIANT_SCHEDULES.get(self._pai_variant, {}).get(
            model_key
        )
        if variant_schedule is not None:
            return variant_schedule
        return _MODEL_DYNAMIC_PAI_SCHEDULES.get(model_key)

    @staticmethod
    def _model_artifact_revision(model_key: str) -> str | None:
        return _MODEL_ARTIFACT_REVISIONS.get(model_key)

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
                fine_tune_epochs = self._pqat_epoch_budget(model_key, condition)
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
        modules_to_perforate = self._perforation_modules_to_perforate(model_key)
        module_ids_to_perforate = self._perforation_module_ids_to_perforate(model_key)
        module_selection = PAIModuleSelection(
            modules_to_perforate=modules_to_perforate,
            module_ids_to_perforate=module_ids_to_perforate,
            track_only_module_ids=self._perforation_track_only_module_ids(model_key),
            module_names_to_not_save=self._perforation_module_names_to_not_save(model_key),
            parameter_ids_to_track=self._perforation_parameter_ids_to_track(model_key),
        )
        module_output_dimensions = infer_module_output_dimensions(
            model,
            model_key,
            bundle,
            modules_to_perforate,
            module_names=[*module_ids_to_perforate],
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
        initial_correlation_batches_limit = (
            self._pai_initial_correlation_batches_limit(model_key)
            if training_plan.update_dendrites_during_training
            else None
        )
        fixed_switch_interval = (
            self._pai_fixed_switch_interval(model_key)
            if training_plan.update_dendrites_during_training
            else None
        )
        dynamic_schedule = (
            self._pai_dynamic_schedule(model_key)
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
                    save_name=self._pai_save_name(model_key, condition.key),
                    maximizing_score=metric_direction == "maximize",
                    module_selection=module_selection,
                    config_snapshot_path=pai_config_snapshot,
                    dendrite_training_max_epochs=dendrite_training_max_epochs,
                    dynamic_dendritic_training=dynamic_dendritic_training,
                    freeze_dendrite_updates_fraction=0.20,
                    batches_per_epoch=batches_per_epoch,
                    module_output_dimensions=module_output_dimensions,
                    candidate_graph_enabled=training_plan.update_dendrites_during_training,
                    initial_correlation_batches_limit=(
                        initial_correlation_batches_limit
                    ),
                    fixed_switch_interval=fixed_switch_interval,
                    dynamic_schedule=dynamic_schedule,
                ),
            )
        if not condition.use_dendrites:
            return model
        model = perforate_model(
            model,
            save_name=self._pai_save_name(model_key, condition.key),
            doing_pai=True,
            maximizing_score=metric_direction == "maximize",
            module_selection=module_selection,
            config_snapshot_path=pai_config_snapshot,
            dendrite_training_max_epochs=dendrite_training_max_epochs,
            dynamic_dendritic_training=dynamic_dendritic_training,
            freeze_dendrite_updates_fraction=0.20,
            batches_per_epoch=batches_per_epoch,
            runtime_options=PAIRuntimeOptions(
                use_runtime_guard=self._use_pai_runtime_guard(),
                candidate_graph_enabled=training_plan.update_dendrites_during_training,
                initial_correlation_batches_limit=initial_correlation_batches_limit,
                fixed_switch_interval=fixed_switch_interval,
                dynamic_schedule=dynamic_schedule,
            ),
        )
        configure_pai_candidate_graph(training_plan.update_dendrites_during_training)
        return self._configure_perforated_model(model, module_output_dimensions)

    def _training_hyperparameters(
        self, model_key: str, condition: ConditionSpec
    ) -> ModelTrainingRecipe:
        """Return model-specific training knobs adapted from canonical recipes.

        The dendritic-vs-quantization question this benchmark exists to answer
        only means something if the FP32 baselines are near their published
        accuracy, so each recipe below tracks the reference training setup for
        that model/dataset pair rather than a shared default. Comments record
        the reference and, where the recipe changed, what the old one left on
        the table on the 2026-08-05 run (`results/<model>/base_fp32/`).
        """
        recipes: dict[str, ModelTrainingRecipe] = {
            # 98.39% under a flat lr for 20 epochs. LeCun et al. report ~99.05%
            # and modern LeNet-5 runs reach ~99.2%; the gap was purely schedule
            # and budget, so anneal to zero over twice as many (1s/epoch) epochs.
            "lenet5": ModelTrainingRecipe(
                128, 40, 1.0e-2, "sgd", 0.9, 5.0e-4,
                lr_schedule="cosine", nesterov=True,
            ),
            # torchaudio's Speech Commands tutorial (the source of this
            # architecture) pairs Adam 1e-2 / wd 1e-4 with StepLR(20, 0.1). The
            # step was missing here, which is why val accuracy oscillated
            # between 0.885 and 0.909 for the last 15 epochs instead of settling.
            "m5": ModelTrainingRecipe(
                128, 40, 1.0e-2, "adam", 0.9, 1.0e-4,
                lr_schedule="step", lr_decay_every=20, lr_decay_gamma=0.1,
            ),
            "lstm_forecaster": ModelTrainingRecipe(
                256, 60, 1.0e-3, lr_schedule="cosine", lr_min_factor=0.01,
                grad_clip_norm=1.0,
            ),
            # 10 epochs was under-training a model that costs 0.2s/epoch. Kim
            # (2014) trains TextCNN to convergence with dropout+early stopping;
            # 30 annealed epochs is the equivalent here. See TextDataSets.ag_news
            # for the matching vocab/sequence-length widening.
            "textcnn": ModelTrainingRecipe(
                128, 30, 1.0e-3, "adam", 0.9, 1.0e-4,
                lr_schedule="cosine", lr_min_factor=0.02,
            ),
            # Kipf & Welling: Adam 1e-2, wd 5e-4, dropout 0.5, 200 epochs of
            # full-batch descent over 140 labelled nodes — 200 optimiser steps.
            # GraphDatasets.cora is now transductive, so batch_size=1 means one
            # whole-graph step per epoch and this recipe reproduces those 200
            # steps exactly rather than landing at 1000 mini-batch steps over
            # independent ego graphs. Kipf uses a constant rate with early
            # stopping on a 10-epoch window; the cosine floor does the same job
            # inside a fixed budget and keeps both arms on one schedule.
            "gcn": ModelTrainingRecipe(
                1, 200, 1.0e-2, "adam", 0.9, 5.0e-4,
                lr_schedule="cosine", lr_min_factor=0.05,
            ),
            # TabNet (Arik & Pfister) reports 85.7% on Adult. Still improving at
            # epoch 100 (0.8489 -> 0.8501 over the last three), so extend and
            # anneal rather than stop on a flat lr.
            "tabnet": ModelTrainingRecipe(
                1024, 200, 2.0e-3, "adamw", 0.9, 1.0e-5,
                lr_schedule="cosine", lr_min_factor=0.02,
            ),
            # MoleculeNet's MPNN reaches ~0.58 RMSE on ESOL. Targets are now
            # standardised in GraphDatasets.esol, which also makes the MSE scale
            # sane. 200 epochs, not more: with only ~790 training molecules the
            # richer featuriser lets this memorise them, and a measured 300-epoch
            # run reached train RMSE 0.29 against val 0.80 and tested at 0.8117,
            # while the same recipe at 200 epochs tested at 0.6665.
            "mpnn": ModelTrainingRecipe(
                32, 200, 1.0e-3, "adam", 0.9, 1.0e-5,
                lr_schedule="cosine", lr_min_factor=0.02, grad_clip_norm=5.0,
            ),
            "actor_critic": ModelTrainingRecipe(
                512, 60, 3.0e-4, lr_schedule="cosine", lr_min_factor=0.05
            ),
            "lstm_autoencoder": ModelTrainingRecipe(
                128, 60, 1.0e-3, lr_schedule="cosine", lr_min_factor=0.02,
                grad_clip_norm=1.0,
            ),
            # Sanh et al. and the reference distilbert-base-uncased-finetuned-sst-2
            # card both fine-tune at AdamW 2e-5 for 3 epochs with linear warmup
            # and decay, reaching 91.3% dev accuracy. This ran at 1e-4 — 5x too
            # high — on a flat schedule and stopped at 87.96%.
            # Warmup stays off: this loop's ramp has per-epoch granularity, and
            # over a 3-epoch budget one warmup epoch is a third of training at a
            # reduced rate rather than the ~6% of *steps* BERT recipes intend.
            # HuggingFace's Trainer default is likewise linear-decay, no warmup.
            "distilbert": ModelTrainingRecipe(
                32, 3, 2.0e-5, "adamw", 0.9, 1.0e-2,
                lr_schedule="linear", grad_clip_norm=1.0,
            ),
            "dqn_lunarlander": ModelTrainingRecipe(
                128, 120, 6.3e-4, lr_schedule="cosine", lr_min_factor=0.05
            ),
            # On-policy: one "epoch" is one PPO iteration — 2048 fresh
            # environment steps, then 10 passes over that buffer in minibatches
            # of 64. Everything but the budget is Stable-Baselines3 RL Zoo's
            # tuned BipedalWalker-v3 entry (lr 3e-4, batch 64, max_grad_norm
            # 0.5, plus gamma/GAE/clip/entropy set at the rollout and loss).
            #
            # The budget is not: the Zoo trains 5M steps for its reported ~213,
            # which is many hours per condition here. 800 iterations is ~1.6M
            # steps, roughly 2h, and matters for a specific reason — an
            # untrained BipedalWalker policy stands still and scores about -9,
            # while a policy that has started moving and still falls scores
            # about -100. A run cut off inside that trough would select the
            # do-nothing first epoch as its best checkpoint and compare two
            # arms' untrained weights. ~1.6M steps clears the trough. It is
            # still short of the 300-point solved threshold; the number is a
            # real, published-comparable return either way, which is the point
            # of the change, and both arms get the same budget.
            "ppo_bipedalwalker": ModelTrainingRecipe(
                64, 800, 3.0e-4, lr_schedule="cosine", lr_min_factor=0.05,
                grad_clip_norm=0.5,
            ),
            # FreeSolv's targets span roughly -25..+4 kcal/mol. Trained on raw
            # values the model spent its first 10 epochs sitting at the target
            # variance (train MSE ~14.8) and finished at RMSE 2.14 against
            # MoleculeNet's ~1.15 for AttentiveFP. GraphDatasets.freesolv now
            # standardises the target; the recipe supplies the longer annealed
            # budget the 642-molecule set needs.
            "attentivefp_freesolv": ModelTrainingRecipe(
                32, 300, 1.0e-3, "adam", 0.9, 1.0e-5,
                lr_schedule="cosine", lr_min_factor=0.02, grad_clip_norm=5.0,
            ),
            # Xu et al. train GIN with Adam 1e-2 decayed by 0.5 every 50 epochs
            # and report 75.1% on IMDB-BINARY. Train loss only moved 0.705 ->
            # 0.622 in 100 epochs here because mean-pooling over 96 padded node
            # slots washed out the graph embedding; see GIN.forward for the mask.
            "gin_imdbb": ModelTrainingRecipe(
                32, 200, 1.0e-2, "adam", 0.9, 5.0e-4,
                lr_schedule="step", lr_decay_every=50, lr_decay_gamma=0.5,
            ),
            "tcn_forecaster": ModelTrainingRecipe(
                128, 80, 1.0e-3, "adam", 0.9, 1.0e-4,
                lr_schedule="cosine", lr_min_factor=0.01, grad_clip_norm=1.0,
                regression_loss="smooth_l1",
            ),
            # Batch 24 -> 128 (see _BATCH_SIZES) cuts the step count per epoch by
            # 5.3x. Weather runs Autoformer's 96-step horizon.
            #
            # 80 epochs at 1e-3 was wrong in a way the stored Dynamic10 history
            # shows plainly: validation MAE bottomed at epoch 5 and then rose for
            # 75 straight epochs (0.3305 -> 0.3957) while train MAE fell to
            # 0.184. Cosine barely moves in that window -- the LR was still
            # 9.9e-4 at the epoch the model was at its best -- so the entire
            # budget was spent overfitting past the checkpoint that got reported.
            # At 3e-4 over 24 epochs the model peaks at epoch 14 and *stays*
            # there (best 0.2586, final 0.2587) instead of peaking early and
            # decaying. The peak value itself is a wash; ending training at the
            # optimum rather than 75 epochs beyond it is the point, and it is
            # what lets the warm-started dendritic arm be scored against a
            # converged baseline. In the full pipeline the best epoch moves
            # 5 -> 16 of 24 and training drops 1716s -> 356s. See
            # GRUForecaster for the RevIN measurements.
            "gru_forecaster": ModelTrainingRecipe(
                128, 24, 3.0e-4, lr_schedule="cosine", lr_min_factor=0.01,
                grad_clip_norm=1.0, regression_loss="smooth_l1",
            ),
            # Decay by 0.7 every 20 epochs, matching the reference PointNet
            # implementation's schedule (Qi et al., provider.py). A constant 1e-3
            # for all 60 epochs left train loss still drifting down at the end and
            # let ReLU units die off in the wide (1024-channel) BatchNorm1d layers
            # (~21% of feature_transform.conv.7 at running_var < 1e-6 by epoch 60).
            # NB: this is a convergence improvement, not the fix for the run that
            # reported ~7% val accuracy — that was a corrupted-eval bug, see
            # _move_batch_to_device in training.py.
            #
            # 100 -> 200 epochs came free. An epoch used to cost ~200s, of which
            # ~190s was re-parsing OFF meshes in the dataloader; caching the
            # sampled clouds (see _ModelNet40Dataset) put it at ~36s, so 200
            # epochs now runs in a third of the wall clock the old 100 did. The
            # reference trains 250, and the 100-epoch run was still climbing
            # (best val accuracy arrived at epoch 81 of 100).
            "pointnet_modelnet40": ModelTrainingRecipe(
                32, 200, 1.0e-3, "adam", 0.9, 1.0e-4,
                lr_schedule="step", lr_decay_every=20, lr_decay_gamma=0.7,
            ),
            "vae_mnist": ModelTrainingRecipe(
                # Dynamic11's dendritic VAE ran to epoch 149 while its dense
                # control stopped at 50, and the first candidate arrived after
                # the cosine schedule was already at its floor.  The shared
                # 150-epoch horizon makes the FP32 control fair and leaves a
                # useful rate for a decoder dendrite to adapt.
                128, 150, 1.0e-3, lr_schedule="cosine", lr_min_factor=0.02,
                lr_schedule_epochs=150,
            ),
            # 242s/epoch is the most expensive model in the suite, so the budget
            # stays at 50; the annealed tail is what buys the accuracy (val was
            # still bouncing 0.978-0.981 over the last ten epochs at a flat lr).
            "snn_nmnist": ModelTrainingRecipe(
                16, 50, 1.0e-3, "adam", 0.9, 1.0e-5,
                lr_schedule="cosine", lr_min_factor=0.01, grad_clip_norm=5.0,
            ),
            "unet_isic": ModelTrainingRecipe(
                8, 100, 1.0e-3, "adam", 0.9, 1.0e-5,
                lr_schedule="cosine", lr_min_factor=0.02,
            ),
            # The standard CIFAR-10 ResNet recipe (He et al. as adapted by
            # kuangliu/pytorch-cifar and the PyTorch Lightning ~94% baseline):
            # SGD 0.1 / momentum 0.9 / wd 5e-4, batch 128, 200 epochs, cosine to
            # zero, random-crop+flip. Held at a flat 0.05 for 90 epochs this
            # plateaued at 88.85% with train loss stuck at 0.18 — the anneal is
            # exactly the missing piece. 24s/epoch, so 200 epochs is ~80 min.
            "resnet18_cifar10": ModelTrainingRecipe(
                128, 200, 1.0e-1, "sgd", 0.9, 5.0e-4,
                lr_schedule="cosine", warmup_epochs=5, label_smoothing=0.1,
                nesterov=True,
            ),
            # Same story at 89.14% over 150 flat epochs; published MobileNetV2
            # CIFAR-10 runs reach ~94.1% with SGD 0.1 + cosine over 200 epochs.
            # Weight decay stays at 4e-5 (the MobileNetV2 paper's value) rather
            # than 5e-4 — depthwise-separable stacks are much smaller.
            "mobilenetv2_cifar10": ModelTrainingRecipe(
                128, 200, 1.0e-1, "sgd", 0.9, 4.0e-5,
                lr_schedule="cosine", warmup_epochs=5, label_smoothing=0.1,
                nesterov=True,
            ),
            # 2.2s/epoch, and val accuracy was still ticking up at epoch 100
            # (0.8467 -> 0.8495). SAINT (Somepalli et al.) reports ~86% on Adult.
            "saint_adult": ModelTrainingRecipe(
                256, 200, 1.0e-4, "adamw", 0.9, 1.0e-5,
                lr_schedule="cosine", warmup_epochs=5, lr_min_factor=0.02,
                grad_clip_norm=1.0,
            ),
            # Sabour et al. train with Adam 1e-3 and exponential decay; the real
            # fix here is the margin loss (see CapsuleMarginLoss in training.py),
            # since CrossEntropy over capsule *lengths* barely produced gradient.
            "capsnet_mnist": ModelTrainingRecipe(
                128, 30, 1.0e-3, "adam", 0.9, 0.0,
                lr_schedule="step", lr_decay_every=1, lr_decay_gamma=0.96,
            ),
        }
        recipe = recipes.get(
            model_key,
            ModelTrainingRecipe(64, 4 * EPOCH_MULTIPLIER, 1.0e-3),
        )
        if model_key == "tcn_forecaster" and self._model_scale < 1.0:
            # The compact TCN follow-up has less base capacity but the same
            # chronological validation gap seen in Dynamic9. Keep the measured
            # dropout=0.2 architecture and modestly raise L2 for both paired
            # arms before asking a dendrite to recover capacity.
            recipe = replace(recipe, weight_decay=2.0e-4)
        if model_key == "tcn_forecaster" and condition.key.endswith("q1_58"):
            # Ternary QAT was still improving at the end of Dynamic10's
            # ten-epoch phase. A lower step avoids repeatedly jumping between
            # quantization bins while the longer phase can recover accuracy.
            recipe = replace(recipe, learning_rate=3.0e-4)
        if model_key == "vae_mnist" and condition.key.endswith("q1_58"):
            # Ternary VAE PQAT has a much sharper reconstruction loss than
            # the FP32 phase.  A smaller step lets its full-precision shadow
            # accumulate movement between ternary projections.
            recipe = replace(recipe, learning_rate=2.0e-4)
        dendritic_batch_size = _MODEL_DENDRITIC_BATCH_SIZES.get(model_key)
        if condition.use_dendrites and dendritic_batch_size is not None:
            return recipe.with_batch_size(dendritic_batch_size)
        return recipe

    def _pqat_epoch_budget(self, model_key: str, condition: ConditionSpec) -> int:
        """Allocate a short PQAT phase from the model's canonical epoch recipe."""
        if model_key == "tcn_forecaster" and condition.key.endswith("q1_58"):
            return 36
        if model_key == "vae_mnist" and condition.key.endswith("q1_58"):
            return 40
        recipe = self._training_hyperparameters(
            model_key, condition_by_key("base_fp32")
        )
        return max(1, min(10, math.ceil(recipe.max_epochs * 0.30)))

    @staticmethod
    def _dynamic_training_epoch_cap(
        training_plan: ConditionTrainingPlan,
    ) -> int | None:
        """Bound a dynamic PAI run without undercutting its paired control.

        One permitted dendrite needs only a short post-switch adaptation phase.
        Sixteen extra epochs exceed the configured six-epoch PAI phase twice
        over, while preventing a no-improvement tracker from consuming an
        unbounded amount of compute and invalidating the matched budget.
        """
        if not training_plan.update_dendrites_during_training:
            return None
        return training_plan.max_epochs + 16

    @staticmethod
    def _quantization_granularity(
        model_key: str, condition: ConditionSpec
    ) -> Literal["tensor", "channel"]:
        """Choose the quantizer scale layout without changing its bit codes."""
        if (
            condition.quantization_mode == "ternary"
            and model_key in {"tcn_forecaster", "vae_mnist"}
        ):
            return "channel"
        return "tensor"

    def _memory_cleanup_interval_batches(
        self,
        model_key: str,
        condition: ConditionSpec,
        batches_per_epoch: int | None,
    ) -> int | None:
        if not condition.use_dendrites:
            return None
        configured = _MODEL_DENDRITIC_MEMORY_CLEANUP_INTERVALS.get(
            model_key, _DEFAULT_DENDRITIC_MEMORY_CLEANUP_INTERVAL
        )
        if batches_per_epoch is None:
            return configured
        if batches_per_epoch <= configured:
            return None
        return configured

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
        record = cast(Any, TrainingRecord)(**json.loads(record_path.read_text()))
        model_records.append(record.to_dict())
        all_records.append(record.to_dict())
        saved_dirs[condition.key] = condition_dir

    def _distilbert_dendritic_config_current(self, condition_dir: Path) -> bool:
        config_path = condition_dir / "PAI_config.json"
        if not config_path.exists():
            return False
        try:
            config = json.loads(config_path.read_text())
        except json.JSONDecodeError:
            return False
        expected_ids = set(_DISTILBERT_PAI_CLASSIFICATION_HEAD)
        module_ids = set(config.get("module_ids_to_perforate") or [])
        modules_to_perforate = config.get("modules_to_perforate") or []
        correlation_batches = config.get("initial_correlation_batches")
        # Records written before the backbone was tracked have untagged
        # parameters, which changes both the p-phase parameter filtering and the
        # state_dict key names, so they cannot be reused against today's config.
        tracked_ids = set(config.get("module_ids_to_track") or [])
        expected_tracked_ids = set(
            self._perforation_track_only_module_ids("distilbert")
        )
        return (
            module_ids == expected_ids
            and tracked_ids == expected_tracked_ids
            and not modules_to_perforate
            and isinstance(correlation_batches, int)
            and correlation_batches <= self._pai_initial_correlation_batches_limit(
                "distilbert"
            )
        )

    def _condition_metadata_current(
        self, model_key: str, condition: ConditionSpec, condition_dir: Path
    ) -> bool:
        """Reject saved artifacts built with a different compact/PAI profile."""
        metrics_path = condition_dir / "metrics.json"
        if not metrics_path.exists():
            return False
        try:
            metadata = json.loads(metrics_path.read_text())
        except json.JSONDecodeError:
            return False
        try:
            recorded_scale = float(metadata.get("model_scale", 1.0))
        except (TypeError, ValueError):
            return False
        if recorded_scale != self._model_scale:
            return False
        if metadata.get("model_revision") != self._model_artifact_revision(model_key):
            return False
        recipe = self._training_hyperparameters(model_key, condition)
        if metadata.get("lr_schedule_epochs") != recipe.lr_schedule_epochs:
            return False
        if (
            metadata.get("quantization_granularity", "tensor")
            != self._quantization_granularity(model_key, condition)
        ):
            return False
        if (
            condition.quantized
            and metadata.get("quantization_evaluation_revision")
            != QUANTIZATION_EVALUATION_REVISION
        ):
            return False
        if not condition.use_dendrites:
            return True
        if metadata.get("dendrite_audit_revision") != DENDRITE_AUDIT_REVISION:
            return False
        if metadata.get("pai_variant") != self._pai_variant:
            return False
        expected_schedule = self._pai_dynamic_schedule(model_key)
        recorded_schedule = metadata.get("pai_dynamic_schedule")
        if (
            recorded_schedule is not None
            and expected_schedule is not None
            and recorded_schedule != expected_schedule.to_dict()
        ):
            return False
        return True

    @staticmethod
    def _saved_dendrite_audit_status(
        source_key: str, saved_dirs: dict[str, Path]
    ) -> str | None:
        source_dir = saved_dirs.get(source_key)
        if source_dir is None:
            return None
        record_path = source_dir / _RECORD_JSON
        try:
            record = json.loads(record_path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        status = record.get("dendrite_audit_status")
        return str(status) if status else None

    def _condition_record_usable(
        self,
        model_key: str,
        condition: ConditionSpec,
        *,
        ignore_saved: bool,
    ) -> bool:
        if ignore_saved:
            return False
        condition_dir = self.results_root / model_key / condition.key
        if not (condition_dir / _RECORD_JSON).exists():
            return False
        if not self._condition_metadata_current(model_key, condition, condition_dir):
            _log(
                f"[stale] {model_key} / {condition.key} — model scale or PAI "
                "targeting profile changed; retraining."
            )
            return False
        if (
            model_key == "distilbert"
            and condition.use_dendrites
            and not self._distilbert_dendritic_config_current(condition_dir)
        ):
            _log(
                f"[stale] {model_key} / {condition.key} — PAI config predates the "
                "current setup (memory-safe head-only perforation with the "
                "backbone tracked); retraining."
            )
            return False
        return True

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
        # This method is called only for conditions that
        # ``_condition_record_usable`` rejected.  Do not let the mere presence
        # of a stale record bypass a model/PAI revision and silently keep old
        # results in a new experiment directory.
        _log(f"[train] {model_spec.key} / {condition.key} — starting…", before=True)
        # Re-seed per condition so a model's base_* and dendrites_* arms draw
        # the same initial weights, making them a paired comparison.
        if self._seed is not None:
            seed_everything(self._seed)
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
            if not self._condition_record_usable(
                model_spec.key, cond, ignore_saved=ignore_saved
            )
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
            bundles_by_batch_size: dict[int, Any] = {}
            for condition in pending:
                recipe = self._training_hyperparameters(model_spec.key, condition)
                bundle = bundles_by_batch_size.get(recipe.batch_size)
                if bundle is None:
                    # Seed before the bundle too: dataset splits and shuffle
                    # order are drawn here, and a different train/val split is
                    # just as much a different experiment as a different init.
                    if self._seed is not None:
                        seed_everything(self._seed)
                    bundle = build_task_bundle(
                        model_spec.key, batch_size=recipe.batch_size
                    )
                    bundles_by_batch_size[recipe.batch_size] = bundle
                if self._train_pending_condition(
                    model_spec, condition, bundle, ignore_saved,
                    model_records, all_records, saved_dirs, allow_pqat,
                    dynamic_dendritic_training,
                ):
                    newly_trained = True
                _release_accelerator_memory()

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
        write_reports: bool = True,
        seed: int | None = None,
    ) -> list[dict[str, Any]]:
        """Train the selected models sequentially in this process.

        ``seed`` is re-applied before every (model, condition) — see
        :func:`compat.seed_everything` for why per-condition and not per-process.
        ``None`` leaves every RNG unseeded, which is the historical behaviour.

        ``write_reports=False`` suppresses the manifest and the cross-model
        comparison reports, which are built from *this* runner's records and so
        would each hold only one worker's share when several workers run at
        once. :func:`run_parallel` writes them once, from every record on disk,
        after the last worker exits. Per-model reports stay on: each lands in
        its own model's directory, which exactly one worker ever writes to.
        """
        selected_models = [
            model_by_key(key)
            for key in (model_keys or [spec.key for spec in MODEL_SPECS])
        ]
        selected_condition_keys = self._expand_condition_keys(condition_keys)
        selected_conditions = [condition_by_key(key) for key in selected_condition_keys]
        all_records: list[dict[str, Any]] = []
        self._seed = seed
        if seed is not None:
            _log(f"[seed] every model/condition seeded with {seed}")

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
            if newly_trained and write_reports:
                completed_model_keys = {r["model_key"] for r in all_records}
                if len(completed_model_keys) >= 2:
                    _log(
                        f"[compare] {len(completed_model_keys)} models complete — "
                        "regenerating comparison reports…",
                        after=True,
                    )
                    write_manifest(all_records, self.results_root / "manifest.csv")
                    write_comparison_reports(all_records, self.comparison_root)

        if write_reports:
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
        training_hyperparameters = self._training_hyperparameters(model_key, condition)
        training_plan = self._condition_training_plan(
            model_key, condition, training_hyperparameters, allow_pqat
        )
        model = build_model(model_key, **self._model_kwargs(model_key))
        dense_param_count = sum(parameter.numel() for parameter in model.parameters())
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

        weight_decay = training_hyperparameters.weight_decay
        pai_candidate_graph_batch_limit = (
            self._pai_initial_correlation_batches_limit(model_key)
            if model_key in _MODEL_PAI_INITIAL_CORRELATION_BATCH_LIMITS
            and training_plan.update_dendrites_during_training
            else None
        )
        memory_cleanup_interval_batches = self._memory_cleanup_interval_batches(
            model_key, condition, batches_per_epoch
        )
        dynamic_schedule = (
            self._pai_dynamic_schedule(model_key)
            if training_plan.update_dendrites_during_training
            else None
        )
        fixed_switch_interval = (
            self._pai_fixed_switch_interval(model_key)
            if training_plan.update_dendrites_during_training
            else None
        )
        training_config = TrainingConfig(
            bit_width=condition.bit_width,
            quantization_mode=condition.quantization_mode,
            quantization_granularity=self._quantization_granularity(
                model_key, condition
            ),
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
            nesterov=training_hyperparameters.nesterov,
            lr_schedule=training_hyperparameters.lr_schedule,
            lr_decay_every=training_hyperparameters.lr_decay_every,
            lr_decay_gamma=training_hyperparameters.lr_decay_gamma,
            lr_min_factor=training_hyperparameters.lr_min_factor,
            lr_schedule_epochs=training_hyperparameters.lr_schedule_epochs,
            max_dynamic_training_epochs=self._dynamic_training_epoch_cap(
                training_plan
            ),
            quantization_evaluation_revision=(
                QUANTIZATION_EVALUATION_REVISION if condition.quantized else None
            ),
            dendrite_audit_revision=(
                DENDRITE_AUDIT_REVISION if condition.use_dendrites else None
            ),
            dense_param_count=dense_param_count,
            source_dendrite_audit_status=(
                self._saved_dendrite_audit_status(condition.source_key, saved_dirs)
                if condition.use_dendrites and condition.source_key != condition.key
                else None
            ),
            warmup_epochs=training_hyperparameters.warmup_epochs,
            label_smoothing=training_hyperparameters.label_smoothing,
            regression_loss=training_hyperparameters.regression_loss,
            grad_clip_norm=training_hyperparameters.grad_clip_norm,
            source_condition_key=condition.source_key,
            enable_pai_dendrite_updates=training_plan.update_dendrites_during_training,
            train_dendrites_until_complete=(
                training_plan.update_dendrites_during_training
                and dynamic_dendritic_training
            ),
            freeze_dendrite_updates_fraction=0.20,
            pai_candidate_graph_batch_limit=pai_candidate_graph_batch_limit,
            memory_cleanup_interval_batches=memory_cleanup_interval_batches,
            pai_save_name=self._pai_save_name(model_key, condition.key),
            model_scale=self._model_scale,
            pai_variant=self._pai_variant,
            model_revision=self._model_artifact_revision(model_key),
            pai_fixed_switch_interval=fixed_switch_interval,
            pai_dynamic_schedule=(
                dynamic_schedule.to_dict() if dynamic_schedule is not None else None
            ),
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


# ============================================================== parallel run ===
# `run` trains one model at a time in this process. Splitting the selected models
# across several worker processes cuts wall-clock close to linearly on this
# machine: training is accelerator-compute-bound rather than data-bound, so a
# second concurrent job costs the first only a few percent (ResNet-18 held
# 3.80 -> 3.61 batch/s alongside another run). The full 23-model FP32 sweep is
# ~24h sequentially, and ResNet-18 alone is ~155s/epoch x 200 epochs.
#
# Partitioning is by *model*, never by condition: conditions form a dependency
# chain within a model (dendrites_q8 is quantized from dendrites_fp32), so a
# model and all of its conditions must stay in one worker to keep that order.

_STREAM_LOG_DIRNAME = "streams"


def _log_root_variant(base: Path, n: int) -> Path:
    """The nth candidate log root: base itself for n == 1, else base2, base3, ..."""
    return base if n == 1 else base.parent / f"{base.name}{n}"


def _log_root_has_stream_logs(log_root: Path) -> bool:
    stream_dir = log_root / _STREAM_LOG_DIRNAME
    return stream_dir.exists() and any(stream_dir.glob("stream_*.log"))


def _next_log_root(base: Path) -> Path:
    """The first ``base``/``base2``/``base3``/... whose streams/ has no worker
    logs yet, so launching a new run never truncates a previous run's.

    ``_launch_worker`` opens each stream_N.log with ``"wb"`` (truncating) on
    purpose, so the reporter never double-counts a prior run's ``[done]``
    lines — that only stays safe if a fresh launch always lands in a log root
    that has never had workers write to it.
    """
    n = 1
    while _log_root_has_stream_logs(_log_root_variant(base, n)):
        n += 1
    return _log_root_variant(base, n)


def _latest_log_root(base: Path) -> Path:
    """The most recently created ``base``/``base2``/``base3``/... that exists,
    for read-only callers (``--status``) that want the active run's logs
    without minting a new directory."""
    n = 1
    latest = base
    while _log_root_variant(base, n).exists():
        latest = _log_root_variant(base, n)
        n += 1
    return latest


_PROGRESS_LOG_NAME = "run_progress.log"
# What the launching invocation selected, so that a later --status against this
# log directory reports progress against that run's workload rather than against
# whatever the --status invocation itself happens to select.
_PLAN_NAME = "run_plan.json"
DEFAULT_JOBS = 4
DEFAULT_PROGRESS_INTERVAL = 60

# Rough FP32 training cost per model, in approximate hours, used *only* to
# balance the workers. Measured where a recent record exists (snn_nmnist 4.4h,
# capsnet_mnist 3.8h, gcn ~1min) and estimated from the epoch budget and the
# relative costs in results/old_models otherwise. An entry being wrong costs
# wall-clock balance, never correctness; an unlisted model is treated as
# _DEFAULT_COST_HOURS.
_MODEL_COST_HOURS: dict[str, float] = {
    "resnet18_cifar10": 8.6,
    "mobilenetv2_cifar10": 8.0,
    "snn_nmnist": 4.4,
    "capsnet_mnist": 3.8,
    "pointnet_modelnet40": 3.0,
    "gru_forecaster": 2.5,
    "distilbert": 2.0,
    "m5": 1.0,
    "ppo_bipedalwalker": 0.3,
    "tcn_forecaster": 0.3,
    "saint_adult": 0.2,
    "vae_mnist": 0.2,
    "dqn_lunarlander": 0.2,
    "lstm_autoencoder": 0.2,
}
_DEFAULT_COST_HOURS = 0.1

# The interesting state lives in tqdm bars that are rewritten in place, and the
# bar's postfix carries the running best metric, so worker logs are parsed
# rather than skimmed.
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_EPOCH = re.compile(r"^(\S+) \| (\S+):\s+(\d+)%\|[^|]*\|\s*(\d+)/(\d+) \[([^<\]]+)<([^,\]]+),\s*(.*)\]\s*$")
_DONE = re.compile(r"\[done\] (\S+) / (\S+) [—-]+ (.+?): (-?[\d.]+)")
_START = re.compile(r"\[train\] (\S+) / (\S+) [—-]+ starting")
_BEST = re.compile(r"best_\w+=(-?[\d.]+)")

# Checked against full command lines. Workers are spawned through the `dqb`
# console script when one is on PATH and through `python -m` otherwise, so both
# spellings have to be recognised — including runs started from another terminal.
_WORKER_PATTERNS = ("dqb run", "dendritic_benchmark.cli run")


def partition_models(model_keys: list[str], jobs: int) -> list[list[str]]:
    """Split models across ``jobs`` workers, longest-first, balancing by cost.

    Greedy longest-processing-time: repeatedly hand the most expensive model to
    whichever worker is cheapest so far. With the default four jobs this keeps
    the two CIFAR models — the long poles, together nearly 17 of the sweep's 24
    hours — out of the same worker, which is the split that actually decides
    wall-clock.
    """
    jobs = max(1, min(jobs, len(model_keys)))
    streams: list[list[str]] = [[] for _ in range(jobs)]
    loads = [0.0] * jobs
    for key in sorted(model_keys, key=lambda k: -_MODEL_COST_HOURS.get(k, _DEFAULT_COST_HOURS)):
        cheapest = loads.index(min(loads))
        streams[cheapest].append(key)
        loads[cheapest] += _MODEL_COST_HOURS.get(key, _DEFAULT_COST_HOURS)
    # A worker with nothing to do would still pay interpreter and import startup
    # and then show up in the progress table as a stream stuck at "starting…".
    return [stream for stream in streams if stream]


class _Emitter:
    """Print a block and append it to the progress log.

    Worker logs are unreadable after the fact — tqdm rewrites a single line in
    place for hours — so the progress log is the only durable record of how the
    run advanced, and it is what makes ``--detach`` usable.
    """

    def __init__(self, progress_log: Path) -> None:
        self._progress_log = progress_log

    def __call__(self, text: str = "", *, stderr: bool = False) -> None:
        print(text, file=sys.stderr if stderr else sys.stdout)
        try:
            with self._progress_log.open("a", encoding="utf-8") as handle:
                handle.write(text + "\n")
        except OSError as exc:
            print(f"  (could not append to {self._progress_log}: {exc.strerror})", file=sys.stderr)


def _dqb_command() -> list[str]:
    """Resolve the command used to spawn worker processes.

    Preferring the console script keeps ``pkill -f 'dqb run'`` — printed by this
    command and used everywhere else — matching the processes it launches. The
    ``python -m`` form is only a fallback for when nothing named ``dqb`` is
    reachable; either way the workers stay inside the interpreter running now.
    """
    argv0 = Path(sys.argv[0])
    if argv0.name == "dqb" and argv0.exists():
        return [str(argv0.resolve())]
    found = shutil.which("dqb")
    if found:
        return [found]
    return [sys.executable, "-m", "dendritic_benchmark.cli"]


def _stop_pattern(dqb_cmd: list[str]) -> str:
    return "dqb run" if Path(dqb_cmd[0]).name == "dqb" else "dendritic_benchmark.cli run"


def _workers_running() -> bool:
    """Whether any `dqb run` process is active, including ones we did not start.

    Only used before launching and by ``--status``; the coordinator tracks its
    own children instead, so a worker that has exited but not yet been reaped
    cannot keep the progress loop alive.
    """
    for pattern in _WORKER_PATTERNS:
        try:
            completed = subprocess.run(
                ["pgrep", "-f", pattern],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except OSError:
            return False
        if completed.returncode == 0:
            return True
    return False


def _stream_logs(log_dir: Path) -> list[Path]:
    return sorted(log_dir.glob("stream_*.log"), key=lambda p: int(p.stem.split("_")[1]))


def _render_progress(log_dir: Path, results_root: Path, total_pairs: int | None) -> str:
    logs = _stream_logs(log_dir)
    if not logs:
        return "  (no worker logs yet)"

    # The condition gets its own column rather than being folded into the model
    # label: `run` now covers all twelve, and the widest pairing
    # (attentivefp_freesolv / dendrites_q1_58) would otherwise overflow the
    # column and knock every row after it out of alignment.
    lines: list[str] = [
        f"  {'stream':<9} {'model':<21} {'condition':<16} {'epoch':>9} {'%':>4}  "
        f"{'remaining':>10}  {'best':>9}",
        f"  {'-' * 9} {'-' * 21} {'-' * 16} {'-' * 9} {'-' * 4}  {'-' * 10}  {'-' * 9}",
    ]
    total_done = 0

    for log in logs:
        # These files belong to the workers, not to us, and the coordinator
        # rereads them every interval for hours. A log that is rotated,
        # truncated or removed between the listing above and this read must not
        # take the coordinator down mid-run.
        try:
            text = _ANSI.sub("", log.read_text(errors="replace")).replace("\r", "\n")
        except OSError as exc:
            lines.append(f"  {log.stem:<10} (unreadable: {exc.strerror})")
            continue

        done = _DONE.findall(text)
        total_done += len(done)

        current, current_condition, last_epoch = None, None, None
        for line in text.splitlines():
            started = _START.search(line)
            if started:
                current, current_condition, last_epoch = started.group(1), started.group(2), None
            epoch = _EPOCH.match(line.strip())
            if epoch:
                last_epoch = epoch
        if (current, current_condition) in {(entry[0], entry[1]) for entry in done}:
            current = None

        if current is None:
            state = "all queued work finished" if done else "starting…"
            lines.append(f"  {log.stem:<9} {state}")
        elif last_epoch is None:
            lines.append(
                f"  {log.stem:<9} {current:<21} {current_condition or '—':<16} "
                f"{'—':>9} {'—':>4}  {'warming up':>10}  {'—':>9}"
            )
        else:
            _, _, pct, cur, tot, _elapsed, remaining, postfix = last_epoch.groups()
            best = _BEST.search(postfix)
            lines.append(
                f"  {log.stem:<9} {current:<21} {current_condition or '—':<16} "
                f"{cur + '/' + tot:>9} {pct + '%':>4}  "
                f"{remaining:>10}  {(best.group(1) if best else '—'):>9}"
            )

        for model, condition, metric, value in done:
            lines.append(f"  {'':<9} ✓ {model:<21} {condition:<16} {metric}: {value}")

    # total_done only counts [done] lines in *this launch's* (truncated)
    # stream logs, so on a resumed run it silently omits every pair that was
    # already complete before this run started and got [skip]ped instead of
    # retrained. The summary line means "how much of the whole plan is done",
    # so prefer counting record.json on disk against the full planned set —
    # that's accurate across restarts. Only fall back to the log-derived count
    # when the plan file isn't readable (e.g. a bare --status with no prior
    # launch in this log dir).
    planned_pairs = _planned_pairs_list(log_dir.parent)
    on_disk = _count_complete_on_disk(results_root, planned_pairs)
    if on_disk is not None and planned_pairs is not None:
        completed = f"{on_disk}/{len(planned_pairs)}"
    else:
        completed = f"{total_done}/{total_pairs}" if total_pairs else str(total_done)
    lines.append("")
    lines.append(
        f"  {completed} model/condition pairs complete   "
        f"({time.strftime('%H:%M:%S')})   -> {results_root}"
    )
    return "\n".join(lines)


def _write_plan(log_root: Path, model_keys: list[str], condition_keys: list[str]) -> None:
    payload = {"models": model_keys, "conditions": condition_keys}
    try:
        (log_root / _PLAN_NAME).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    except OSError:
        # Only costs --status its denominator; never worth failing a launch over.
        pass


def _planned_pairs(log_root: Path) -> int | None:
    try:
        payload = json.loads((log_root / _PLAN_NAME).read_text(encoding="utf-8"))
        return len(payload["models"]) * len(payload["conditions"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _planned_pairs_list(log_root: Path) -> list[tuple[str, str]] | None:
    """Every (model, condition) pair in the launching run's plan, in order."""
    try:
        payload = json.loads((log_root / _PLAN_NAME).read_text(encoding="utf-8"))
        return [(model, condition) for model in payload["models"] for condition in payload["conditions"]]
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _count_complete_on_disk(
    results_root: Path, pairs: list[tuple[str, str]] | None
) -> int | None:
    """How many planned pairs already have a record.json, regardless of which run wrote it."""
    if pairs is None:
        return None
    return sum(
        1 for model, condition in pairs
        if (results_root / model / condition / _RECORD_JSON).is_file()
    )


def _emit_progress(emit: _Emitter, log_dir: Path, results_root: Path, total_pairs: int | None) -> None:
    emit()
    emit("=" * 88)
    emit(f"  {datetime.now():%Y-%m-%d %H:%M:%S}")
    emit(_render_progress(log_dir, results_root, total_pairs))


def clear_epoch_checkpoints(
    results_root: Path,
    model_keys: list[str],
    condition_keys: list[str],
    *,
    fresh: bool,
    emit: _Emitter,
) -> None:
    """Deal with epoch checkpoints left behind by an earlier run.

    ``--ignore-saved-models`` does NOT prevent epoch-level resume: training.py
    calls ``_load_epoch_checkpoint()`` unconditionally whenever the output
    directory exists, and never consults that flag. The flag only stops a
    finished record.json from causing a skip. So after any model-definition
    change, a leftover epoch_checkpoint.pt would silently continue an
    old-architecture run.
    """
    stale = [
        (model, condition)
        for model in model_keys
        for condition in condition_keys
        if (results_root / model / condition / "epoch_checkpoint.pt").is_file()
    ]
    if not stale:
        return

    labels = " ".join(f"{model}/{condition}" for model, condition in stale)
    if fresh:
        # Logged, not just printed: which checkpoints were discarded is the one
        # fact that decides whether a later result is a clean run or a resumed one.
        emit(f"removing {len(stale)} stale epoch checkpoint(s): {labels}")
        for model, condition in stale:
            shutil.rmtree(results_root / model / condition, ignore_errors=True)
        return

    # Same reasoning as the --fresh branch: still stderr for the operator, but
    # also on disk, so a resumed run cannot later be mistaken for a clean one.
    emit(f"WARNING: {len(stale)} model/condition pair(s) have an epoch_checkpoint.pt", stderr=True)
    emit(f"  and WILL resume mid-run rather than start clean: {labels}", stderr=True)
    emit("  --ignore-saved-models does not override this. If the model definitions", stderr=True)
    emit("  changed since those checkpoints, the run would be invalid.", stderr=True)
    emit("  Re-run with --fresh to delete them first, or accept the resume.", stderr=True)
    emit("", stderr=True)


def _launch_worker(
    index: int,
    models: list[str],
    *,
    dqb_cmd: list[str],
    passthrough: list[str],
    condition_keys: list[str] | None,
    log_dir: Path,
) -> subprocess.Popen[bytes]:
    command = [*dqb_cmd, "run", "--worker", *passthrough]
    if condition_keys:
        command += ["--conditions", *condition_keys]
    command += ["--models", *models]
    # Truncated, not appended to: the reporter reads the whole file, and a
    # previous run's [done] lines would be counted as this one's.
    handle = (log_dir / f"stream_{index}.log").open("wb")
    try:
        # start_new_session detaches the worker from the controlling terminal, so
        # Ctrl-C in the coordinator and closing the terminal both leave training
        # running.
        return subprocess.Popen(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    finally:
        handle.close()


def _build_final_reports(
    results_root: Path, comparison_root: Path, *, emit: _Emitter
) -> None:
    """Write the manifest and comparison reports from every record on disk.

    Workers run with ``write_reports=False`` precisely so this can happen once,
    here, across all of their records at the same time. Only reached when the
    coordinator saw every worker exit — ``--detach`` and ``--status`` leave it to
    the operator, since neither knows when (or whether) training finished.
    """
    emit()
    emit("building manifest.csv and comparison reports from all per-model records")
    try:
        records = load_training_records(results_root)
        write_manifest(records, results_root / "manifest.csv")
        write_comparison_reports(records, comparison_root)
    except Exception as exc:
        # Non-fatal on purpose: training results are already on disk, and a
        # failed report build must not make the run look like it lost them.
        emit(f"  WARNING: report build failed ({exc}). Results are intact; rerun with:")
        emit(f"    dqb compare --manifest --results-root {results_root}")
        return
    emit(f"  {len(records)} record(s) -> {results_root / 'manifest.csv'}")


# 2026-08-10: three of four workers were SIGKILLed within a few seconds of each
# other (no traceback in any stream log — almost certainly an MPS/Metal crash
# that never reached Python's exception handling) and this loop kept waiting on
# them for ~7 hours, re-printing their last-known epoch forever, because it
# only ever checked whether *every* worker had exited. A worker is now given a
# bounded number of chances to be respawned before it's treated as failed.
_MAX_WORKER_RESTARTS = 5
_RESTART_BACKOFF_SECONDS = 30


def _watch(
    procs: list[subprocess.Popen[bytes]],
    *,
    streams: list[list[str]],
    dqb_cmd: list[str],
    passthrough: list[str],
    condition_keys: list[str] | None,
    emit: _Emitter,
    log_dir: Path,
    results_root: Path,
    total_pairs: int,
    interval: int,
    stop_pattern: str,
    on_finish: Callable[[], None],
) -> None:
    """Poll workers, respawning any that die before finishing their queue.

    A worker that exits with code 0 worked through every model/condition it
    was given. Anything else — a nonzero code, or death by signal — is treated
    as a crash and respawned with the same model list. `_launch_worker` picks
    up any leftover ``epoch_checkpoint.pt`` automatically (see
    ``clear_epoch_checkpoints``), so the respawn resumes mid-epoch rather than
    losing the run. Each stream gets at most ``_MAX_WORKER_RESTARTS`` attempts
    so a model that crashes deterministically (a bad checkpoint, not a
    transient fault) doesn't spin forever — it's reported and left stopped
    instead of silently retried, and everything else keeps going.
    """
    restarts = [0] * len(procs)
    given_up = [False] * len(procs)

    def _settled(i: int) -> bool:
        return given_up[i] or procs[i].poll() == 0

    try:
        while True:
            _emit_progress(emit, log_dir, results_root, total_pairs)

            for i, proc in enumerate(procs):
                # poll() also reaps, which keeps exited workers from lingering
                # as zombies for the lifetime of the run.
                code = proc.poll()
                if code is None or code == 0 or given_up[i]:
                    continue
                if restarts[i] >= _MAX_WORKER_RESTARTS:
                    given_up[i] = True
                    emit(
                        f"stream_{i + 1} crashed {restarts[i]} time(s) (last exit {code}) "
                        f"and hit the restart limit — giving up on: {' '.join(streams[i])}",
                        stderr=True,
                    )
                    continue
                restarts[i] += 1
                emit(
                    f"stream_{i + 1} (pid {proc.pid}) exited with code {code} — restarting "
                    f"(attempt {restarts[i]}/{_MAX_WORKER_RESTARTS}) in "
                    f"{_RESTART_BACKOFF_SECONDS}s: {' '.join(streams[i])}",
                    stderr=True,
                )
                time.sleep(_RESTART_BACKOFF_SECONDS)
                procs[i] = _launch_worker(
                    i + 1,
                    streams[i],
                    dqb_cmd=dqb_cmd,
                    passthrough=passthrough,
                    condition_keys=condition_keys,
                    log_dir=log_dir,
                )

            if all(_settled(i) for i in range(len(procs))):
                emit()
                emit("all workers finished.")
                on_finish()
                return
            time.sleep(interval)
    except KeyboardInterrupt:
        print()
        print("Ctrl-C — stopping all workers…")
        _terminate_all(procs)
        print("stopped. resume later with the same `dqb run` command.")


def _terminate_all(procs: list[subprocess.Popen[bytes]], *, grace_seconds: float = 10.0) -> None:
    """SIGTERM every worker's process group, then SIGKILL whatever is still alive.

    Workers are launched with ``start_new_session=True`` (so an unrelated Ctrl-C
    in this terminal wouldn't reach them via normal job control), which makes
    each one the leader of its own process group. Signalling ``proc.pid``
    directly would hit only that leader and orphan the DataLoader worker
    subprocesses it spawned — signalling the group via ``os.killpg`` reaches
    those too.
    """
    for proc in procs:
        if proc.poll() is not None:
            continue
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass

    deadline = time.monotonic() + grace_seconds
    for proc in procs:
        try:
            proc.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()


def run_parallel(
    *,
    results_root: Path,
    comparison_root: Path,
    model_keys: list[str],
    condition_keys: list[str] | None,
    expanded_condition_keys: list[str],
    log_root: Path,
    passthrough: list[str],
    jobs: int = DEFAULT_JOBS,
    mode: str = "watch",
    interval: int = DEFAULT_PROGRESS_INTERVAL,
    fresh: bool = False,
) -> None:
    """Train ``model_keys`` across parallel worker processes, reporting progress.

    ``passthrough`` carries the flags a worker needs to reproduce this
    invocation's paths and training options verbatim; ``condition_keys`` is
    passed separately because it is the one list the coordinator may leave
    unset, while ``expanded_condition_keys`` is the resolved set used for
    counting work and clearing stale checkpoints.
    """
    if mode == "status":
        # Read-only: report on whatever log root the most recent launch used,
        # without minting a new one. The launching run's plan, not this
        # invocation's — --status is normally typed without --models/
        # --conditions, which would otherwise measure a two-model run against
        # all 276 pairs.
        active_log_root = _latest_log_root(log_root)
        log_dir = active_log_root / _STREAM_LOG_DIRNAME
        progress_log = active_log_root / _PROGRESS_LOG_NAME
        emit = _Emitter(progress_log)
        _emit_progress(emit, log_dir, results_root, _planned_pairs(active_log_root))
        if not _workers_running():
            emit("  (no dqb run processes active)")
        return

    total_pairs = len(model_keys) * len(expanded_condition_keys)

    if _workers_running():
        print(
            "refusing to launch: 'dqb run' is already active — two runs would race on the",
            file=sys.stderr,
        )
        print(
            f"same {results_root} paths. Use --status to watch, or pkill -f 'dqb run'.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    # A fresh base/base2/base3/... whose streams/ has never had a worker write
    # to it, so this launch can never truncate a previous run's stream logs
    # (_launch_worker opens each stream_N.log with "wb" on purpose — see there).
    log_root = _next_log_root(log_root)
    log_dir = log_root / _STREAM_LOG_DIRNAME
    # Deliberately a level above log_dir: every stream_*.log inside log_dir is a
    # worker log to the reporter, and a progress file sitting there would be
    # reported as an extra worker stuck at "starting…".
    progress_log = log_root / _PROGRESS_LOG_NAME
    log_dir.mkdir(parents=True, exist_ok=True)
    emit = _Emitter(progress_log)

    dqb_cmd = _dqb_command()
    clear_epoch_checkpoints(
        results_root, model_keys, expanded_condition_keys, fresh=fresh, emit=emit
    )

    streams = partition_models(model_keys, jobs)
    _write_plan(log_root, model_keys, expanded_condition_keys)
    emit()
    emit(f"=== launch {datetime.now():%Y-%m-%d %H:%M:%S} -> {results_root}, logs -> {log_root}/")

    procs: list[subprocess.Popen[bytes]] = []
    for index, models in enumerate(streams, 1):
        proc = _launch_worker(
            index,
            models,
            dqb_cmd=dqb_cmd,
            passthrough=passthrough,
            condition_keys=condition_keys,
            log_dir=log_dir,
        )
        procs.append(proc)
        cost = sum(_MODEL_COST_HOURS.get(key, _DEFAULT_COST_HOURS) for key in models)
        emit(f"  stream_{index:<3} pid {proc.pid:<7} ~{cost:>4.1f}h  {' '.join(models)}")

    stop_pattern = _stop_pattern(dqb_cmd)
    emit()
    emit(f"logs:     tail -f {log_dir}/stream_*.log")
    emit(f"progress: tail -f {progress_log}")
    emit(f"status:   dqb run --logging-dir {log_root} --status")
    emit(f"stop:     pkill -f '{stop_pattern}'")

    if mode == "detach":
        emit(f"reports:  dqb compare --manifest --results-root {results_root}")
        emit("          (run once every worker has exited — detached runs skip the automatic")
        emit("           report build, so manifest.csv is not written at all)")
        return

    print()
    print(f"progress every {interval}s — Ctrl-C stops all workers (use --detach to keep them running)")
    print(f"(also appended to {progress_log})")
    _watch(
        procs,
        streams=streams,
        dqb_cmd=dqb_cmd,
        passthrough=passthrough,
        condition_keys=condition_keys,
        emit=emit,
        log_dir=log_dir,
        results_root=results_root,
        total_pairs=total_pairs,
        interval=interval,
        stop_pattern=stop_pattern,
        on_finish=functools.partial(
            _build_final_reports, results_root, comparison_root, emit=emit
        ),
    )
