import csv
import gc
import hashlib
import importlib
import itertools
import json
import math
import os
import shutil
import subprocess
import time
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from tqdm.auto import tqdm

from .artifacts import write_artifact_manifest
from .checkpointing import (
    inspect_state_dict,
    load_state_dict_checked,
)
from .compat import (
    MODULE_OUTPUT_DIMENSIONS_ATTR,
    PAI_ARTIFACT_NAME,
    attach_module_output_dimensions,
    choose_device,
    clear_pai_processor_buffers,
    clear_pai_tracker_state,
    configure_pai_candidate_graph,
    load_pai_system,
    pai_resume_state_exists,
    pai_runtime_guard,
    pai_save_path,
    pai_working_directory,
    save_pai_system,
    set_module_output_dimensions,
)
from .data import _explained_variance
from .model_adapters import model_adapter
from .plans import LRScheduleName, OptimizerName, RegressionLossName
from .quantization import (
    make_quantized_copy as _make_quantized_copy,
    qat_init_shadow as _qat_init_shadow,
    qat_project_for_forward as _qat_project_for_forward,
    qat_restore_shadow_for_step as _qat_restore_shadow_for_step,
    qat_sync_shadow_after_step as _qat_sync_shadow_after_step,
    should_quantize_for_eval as _should_quantize_for_eval,
    should_quantize_for_training as _should_quantize_for_training,
)

_MODEL_PT: str = "model.pt"
_BEST_MODEL_STATS_CSV: str = "best_model_stats.csv"
_EPOCH_CHECKPOINT_PT: str = "epoch_checkpoint.pt"
_MEMORY_GUARD_THRESHOLD_BYTES = 20 * 1024**3
_MEMORY_GUARD_CHECK_INTERVAL_BATCHES = 16
# Consecutive epochs of a bit-for-bit identical validation metric that count as a
# collapsed run rather than a plateau (see _training_collapsed). Generous enough
# that a coarse metric on a small validation split can sit still without tripping.
_COLLAPSE_GUARD_EPOCHS = 12
QUANTIZATION_EVALUATION_REVISION = "single_projection_v1"
# ``v2`` verifies final retained topology from PAI's raw switch-count log,
# rather than its best-score table. The latter can retain only the dense row.
DENDRITE_AUDIT_REVISION = "retained_dendrite_v2"
_FALLBACK_MODULE_OUTPUT_DIMENSIONS: dict[str, dict[str, list[int]]] = {
    "gcn": {
        ".conv1.linear": [-1, -1, 0],
        ".conv2.linear": [-1, -1, 0],
    },
    "mpnn": {
        ".message.0": [-1, -1, 0],
        ".message.2": [-1, -1, 0],
    },
}
# "constant" leaves the learning rate alone for the whole run.
# "step"     multiplies it by lr_decay_gamma every lr_decay_every epochs.
# "cosine"   anneals it from learning_rate down to learning_rate*lr_min_factor.
# "linear"   decays it linearly to learning_rate*lr_min_factor (BERT-style).
# All three honour warmup_epochs first. See ModelTrainingRecipe in pipeline.py.

# Hard ceiling on a single PAI dendrite ("p") phase, in epochs.  PAI's own
# pai_improvement_threshold gates do not reliably end the phase (raising them
# changed nothing on 91 of 92 switch checks), so this is what actually bounds
# one candidate phase. It is not a cap on total dynamic training time.
MAX_DENDRITE_PHASE_EPOCHS = 8


@dataclass
class TrainingConfig:
    bit_width: int | None = None
    quantization_mode: str | None = None
    # ``channel`` keeps ternary codes but gives every Linear/Conv output row
    # its own scale.  It is reserved for the low-bit TCN/VAE follow-ups where
    # one global scale erased small decoder channels.
    quantization_granularity: Literal["tensor", "channel"] = "tensor"
    use_dendrites: bool = False
    use_pruning: bool = False
    prune_amount: float = 0.4
    use_qat: bool = False
    fine_tune_epochs: int = 0
    max_epochs: int = 8
    learning_rate: float = 1e-3
    optimizer_name: OptimizerName = "adam"
    momentum: float = 0.9
    weight_decay: float = 0.0
    nesterov: bool = False
    lr_schedule: LRScheduleName = "constant"
    # Step decay: multiply lr by lr_decay_gamma every lr_decay_every epochs.
    # Only read when lr_schedule == "step".
    lr_decay_every: int | None = None
    lr_decay_gamma: float = 1.0
    # Floor for "cosine"/"linear", as a fraction of the base learning rate.
    lr_min_factor: float = 0.0
    # Optional planned horizon for an annealing schedule. Dynamic PAI can run
    # beyond ``max_epochs``; without this it reaches the LR floor before a
    # late candidate has a chance to adapt.
    lr_schedule_epochs: int | None = None
    # Floor for the dendrite param group only, as a fraction of learning_rate.
    # Applies from the epoch PAI retains a dendrite; the backbone group is
    # untouched. See _dendrite_learning_rate.
    #
    # Defaults to 0.0 -- a no-op that reproduces the pre-2026-08-30 schedule
    # exactly -- so that enabling it stays an explicit, per-recipe decision and
    # no stored result silently changes meaning. The dynamic12 priority models
    # opt in; every other model keeps the behaviour its results were measured
    # under until it is re-run.
    dendrite_lr_min_factor: float = 0.0
    # Linear ramp from 0 to learning_rate over this many epochs, applied under
    # every schedule. Fractional epochs are not supported; the ramp is per-epoch.
    warmup_epochs: int = 0
    label_smoothing: float = 0.0
    # Regression tasks default to MSE for backwards compatibility. Forecasting
    # models may opt into a loss that matches their reported MAE metric.
    regression_loss: RegressionLossName = "mse"
    grad_clip_norm: float | None = None
    source_condition_key: str | None = None
    enable_pai_dendrite_updates: bool = False
    train_dendrites_until_complete: bool = False
    freeze_dendrite_updates_fraction: float = 0.20
    pai_candidate_graph_batch_limit: int | None = None
    memory_cleanup_interval_batches: int | None = None
    pai_save_name: str | None = None
    # Stored with every artifact so a compact-base or PAI-targeting run can be
    # reproduced without inferring intent from its output directory name.
    model_scale: float = 1.0
    pai_variant: str = "default"
    # Explicitly invalidates artifacts when a model architecture or its
    # experiment-critical training plan changes without a model-scale change.
    model_revision: str | None = None
    dataset_revision: str | None = None
    pai_fixed_switch_interval: int | None = None
    pai_dynamic_schedule: dict[str, Any] | None = None
    # Ceiling on a single dendrite ("p") phase in dynamic mode, and the only
    # thing that bounds one.  PAI leaves the phase once no node's correlation
    # has improved for p_epochs_to_switch epochs, but those scores keep drifting
    # up on noise long after they have converged, so the patience counter resets
    # almost every epoch: measured at 91 of 92 dendrite-mode switch checks, vs
    # 27 of 52 in neuron mode where the counter behaves.  Left alone the phase
    # runs unbounded — 207+ epochs on lstm_autoencoder without ever adding a
    # dendrite, and 198/175/169 on mpnn/tabnet/lstm_forecaster.  Raising PAI's
    # own pai_improvement_threshold(_raw) does not help (see compat.py).
    #
    # Lowered 50 -> 8 on 2026-08-28.  At 50 this guard could never fire at all:
    # a dendrite phase freezes validation bit-for-bit by construction (the parent
    # net is frozen and the candidates are not wired into the output yet), so
    # _training_collapsed tripped at _COLLAPSE_GUARD_EPOCHS = 12 and killed the
    # run 38 epochs before the ceiling was reached.  Across the 2026-08-28 seed-0
    # run that cost six of seven models their entire dendrite schedule -- mpnn
    # stopped at 24 epochs against its base arm's 200, having never switched a
    # single dendrite in -- and "Forcing the switch" has never once appeared in
    # any log.  8 sits above the longest phase that has ever ended on its own
    # (3 epochs, gcn, the one model that did complete) and below the collapse
    # guard, so the forced switch happens while the run is still alive.
    # 0 disables the guard.
    max_dendrite_phase_epochs: int = MAX_DENDRITE_PHASE_EPOCHS
    # Stored only for quantized artifacts. QAT validation already evaluates the
    # projected weights; final evaluation must not quantize them a second time.
    quantization_evaluation_revision: str | None = None
    # A dendritic result is reportable only after its final topology is backed
    # by raw PAI switch evidence. The source status is inherited by quantized
    # descendants, which do not run their own candidate-search phase.
    dendrite_audit_revision: str | None = None
    dense_param_count: int | None = None
    source_dendrite_audit_status: str | None = None
    # Unique namespace minted for this condition attempt. It binds the final
    # files and the PAI side tree; rerunning into the same output directory can
    # therefore never consume an earlier attempt's switch logs.
    artifact_id: str = ""
    seed: int | None = None
    # Identifies the weight-only projection rule (quantization.QUANTIZER_REVISION).
    # Set only for quantized conditions, mirroring quantization_evaluation_revision.
    quantizer_revision: str | None = None
    # The exact PAI target-module selection used this run (PAIModuleSelection's
    # three ID lists), recorded so a later change to a model's default targets
    # -- or to a PAIOverride -- invalidates artifacts trained under different
    # targets instead of silently reusing them. See
    # information/optimization/03_execution_matrix.md's manifest-fields list.
    module_ids_to_perforate: tuple[str, ...] | None = None
    track_only_module_ids: tuple[str, ...] | None = None
    parameter_ids_to_track: tuple[str, ...] | None = None
    # RecipeOverride/PAIOverride.to_dict(): only the fields a sweep trial's
    # override file actually set, empty/None when no override was supplied.
    recipe_override: dict[str, Any] | None = None
    pai_override: dict[str, Any] | None = None
    # The fully-resolved ModelTrainingRecipe actually used this run -- after
    # any RecipeOverride and after dendritic batch-size rescaling -- so a
    # sweep trial's real training configuration is on the record even when it
    # differs from both the base recipe and the override file alone.
    effective_recipe: dict[str, Any] | None = None
    # Best-effort `git rev-parse HEAD` at the time this attempt started.
    source_commit: str | None = None
    # Placeholder linking a dendritic result to its matched dense-continuation
    # and capacity-matched dense controls (information/optimization/00_assessment.md
    # validity protocol, step 4). Left unset until those control runs exist and
    # can be linked by artifact_id; not populated by this training pass.
    paired_control_identity: dict[str, Any] | None = None


@dataclass
class TrainingRecord:
    model_key: str
    condition_key: str
    display_name: str
    metric_name: str
    metric_value: float
    metric_direction: str
    best_metric_value: float
    best_epoch: int
    param_count: int
    nonzero_params: int
    file_size_mb: float
    train_seconds: float
    artifact_dir: str
    # Set to True when max_epochs==0 (post-training quantization — no gradient updates).
    training_skipped: bool = False
    # Human-readable explanation of why training was skipped (empty string when training ran).
    skip_reason: str = ""
    dendrite_audit_status: str = "not_applicable"
    dendrite_audit_reason: str = ""
    artifact_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ArtifactMetadata:
    model_key: str
    condition_key: str
    display_name: str
    metric_name: str
    metric_direction: str
    primary_metric_key: str
    use_dendrites: bool
    use_pruning: bool
    bit_width: int | None
    use_qat: bool
    fine_tune_epochs: int
    regression_loss: RegressionLossName
    enable_pai_dendrite_updates: bool
    train_dendrites_until_complete: bool
    freeze_dendrite_updates_fraction: float
    pai_candidate_graph_batch_limit: int | None
    memory_cleanup_interval_batches: int | None
    model_scale: float
    pai_variant: str
    pai_fixed_switch_interval: int | None
    pai_dynamic_schedule: dict[str, Any] | None
    pai_save_name: str | None
    quantization_granularity: str = "tensor"
    dataset_revision: str | None = None
    lr_schedule_epochs: int | None = None
    model_revision: str | None = None
    quantization_evaluation_revision: str | None = None
    dendrite_audit_revision: str | None = None
    dense_param_count: int | None = None
    source_dendrite_audit_status: str | None = None
    artifact_id: str = ""
    seed: int | None = None
    source_condition_key: str | None = None
    quantizer_revision: str | None = None
    module_ids_to_perforate: tuple[str, ...] | None = None
    track_only_module_ids: tuple[str, ...] | None = None
    parameter_ids_to_track: tuple[str, ...] | None = None
    recipe_override: dict[str, Any] | None = None
    pai_override: dict[str, Any] | None = None
    effective_recipe: dict[str, Any] | None = None
    source_commit: str | None = None
    paired_control_identity: dict[str, Any] | None = None
    max_dendrite_phase_epochs: int | None = None


@dataclass(frozen=True)
class ArtifactPayload:
    best_metric: float
    final_metric: float
    best_epoch: int
    history: list[dict[str, Any]]
    test_loss: float
    test_metrics: dict[str, Any]
    training_skipped: bool
    skip_reason: str
    stage_name: str | None = None


@dataclass
class EpochTrainingContext:
    model: Any
    model_key: str
    bundle: Any
    device: Any
    criterion: Any
    torch: Any
    max_epochs: int
    run_label: str
    config: TrainingConfig
    metric_name: str
    primary_metric_key: str
    metric_direction: str
    output_dir: "Path | None" = None


@dataclass
class EpochTrainingState:
    history: list[dict[str, Any]]
    best_metric: float
    best_epoch: int
    best_state: dict[str, Any] | None


@dataclass
class TrainingBatchAccumulator:
    running_loss_t: Any
    examples: int
    outputs: list[Any]
    targets: list[Any]
    metric_targets: list[Any]


@dataclass(frozen=True)
class PAIUpdateStatus:
    frozen: bool
    active: bool


@dataclass(frozen=True)
class ArtifactStats:
    param_count: int
    nonzero_params: int
    file_size_mb: float
    artifact_path: Path


def _write_best_model_stats_csv(output_dir: Path, record: TrainingRecord) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = record.to_dict()
    with (output_dir / _BEST_MODEL_STATS_CSV).open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(payload.keys()))
        writer.writeheader()
        writer.writerow(payload)


def _metric_is_better(new: float, old: float, direction: str) -> bool:
    return new > old if direction == "maximize" else new < old


def _auc(scores: Any, targets: Any) -> float:
    scores = scores.detach().flatten()
    targets = targets.detach().flatten().long()
    positives = scores[targets == 1]
    negatives = scores[targets == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return 0.5
    comparisons = (positives[:, None] > negatives[None, :]).float()
    ties = (positives[:, None] == negatives[None, :]).float() * 0.5
    return (comparisons + ties).mean().item()


def _dice_from_logits(logits: Any, targets: Any) -> float:
    probs = logits.sigmoid()
    preds = (probs >= 0.5).float()
    intersection = (preds * targets).sum(dim=tuple(range(1, preds.dim())))
    union = preds.sum(dim=tuple(range(1, preds.dim()))) + targets.sum(
        dim=tuple(range(1, targets.dim()))
    )
    return float(((2.0 * intersection + 1e-6) / (union + 1e-6)).mean().item())


def _vae_loss(outputs: Any, targets: Any) -> Any:
    recon, mu, logvar = outputs
    bce = torch.nn.functional.binary_cross_entropy(recon, targets, reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (bce + kld) / max(1, targets.shape[0])


def _vae_metrics(outputs: Any, targets: Any) -> dict[str, float]:
    recon, mu, logvar = outputs
    bce = torch.nn.functional.binary_cross_entropy(recon, targets, reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    elbo = -float(((bce + kld) / max(1, targets.shape[0])).item())
    return {
        "elbo": elbo,
        "reconstruction_bce": float((bce / max(1, targets.shape[0])).item()),
        "kl_divergence": float((kld / max(1, targets.shape[0])).item()),
    }


# --------------------------------------------------------------- PPO --------
# Models whose training data is produced by the current policy rather than read
# from a split. Their bundle carries a `.on_policy` rollout source, their
# training loader is rebuilt every epoch, and validation/test are episodic
# rollouts rather than passes over a loader.
_ON_POLICY_MODELS: frozenset[str] = frozenset({"ppo_bipedalwalker"})
# Stable-Baselines3 RL Zoo's tuned BipedalWalker-v3 PPO entry.
_PPO_CLIP_RANGE: float = 0.18
_PPO_VALUE_COEF: float = 0.5
_PPO_ENTROPY_COEF: float = 0.001
# Deterministic evaluation episodes. Validation runs every epoch, so it is kept
# small; test runs once. Seeds are disjoint so the test return is not the
# validation return the best checkpoint was selected on.
_PPO_VAL_EPISODES: int = 5
_PPO_VAL_SEED: int = 4242
_PPO_TEST_SEED: int = 12345


def _is_on_policy(model_key: str) -> bool:
    return model_key in _ON_POLICY_MODELS


def _unpack_ppo_targets(targets: Any, action_dim: int) -> tuple[Any, Any, Any, Any]:
    """Split the packed PPO target tensor into its four columns.

    The rollout yields five tensors per row; ``_forward`` concatenates the last
    four into one ``[B, action_dim + 3]`` tensor so that everything downstream —
    ``_batch_size``, the metric accumulator's ``torch.cat``, the detach helper —
    keeps working on a plain tensor instead of needing a tuple special case at
    every step.
    """
    action = targets[:, :action_dim]
    old_log_prob = targets[:, action_dim]
    advantage = targets[:, action_dim + 1]
    returns = targets[:, action_dim + 2]
    return action, old_log_prob, advantage, returns


def _ppo_terms(outputs: Any, targets: Any) -> dict[str, Any]:
    """The pieces of PPO's objective, as tensors, for one minibatch.

    Shared by the loss and the metrics so the two cannot describe different
    quantities. Everything here is Schulman et al.'s clipped-surrogate PPO as
    implemented in Stable-Baselines3:

    * advantages are standardised **per minibatch**, which is SB3's default and
      keeps the policy-gradient step scale-free as the reward magnitude grows;
    * the ratio is ``exp(new_log_prob - old_log_prob)`` over the *unclipped*
      Gaussian sample, matching how the rollout recorded ``old_log_prob``;
    * ``approx_kl`` is Schulman's low-variance estimator ``(r - 1) - log r``,
      which unlike ``-log r`` is non-negative and much less noisy.
    """
    mean, log_std, value = outputs
    action, old_log_prob, advantage, returns = _unpack_ppo_targets(
        targets, mean.shape[-1]
    )
    distribution = torch.distributions.Normal(mean, log_std.exp())
    log_prob = distribution.log_prob(action).sum(dim=-1)
    entropy = distribution.entropy().sum(dim=-1).mean()

    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
    log_ratio = log_prob - old_log_prob
    ratio = log_ratio.exp()
    clipped = torch.clamp(ratio, 1.0 - _PPO_CLIP_RANGE, 1.0 + _PPO_CLIP_RANGE)
    policy_loss = -torch.min(ratio * advantage, clipped * advantage).mean()
    value_loss = 0.5 * (value - returns).square().mean()
    return {
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy,
        "approx_kl": ((ratio - 1.0) - log_ratio).mean(),
        "clip_fraction": (
            (ratio - 1.0).abs() > _PPO_CLIP_RANGE
        ).to(ratio.dtype).mean(),
        "value": value,
        "returns": returns,
    }


def _ppo_loss(outputs: Any, targets: Any) -> Any:
    terms = _ppo_terms(outputs, targets)
    return (
        terms["policy_loss"]
        + _PPO_VALUE_COEF * terms["value_loss"]
        - _PPO_ENTROPY_COEF * terms["entropy"]
    )


def _ppo_metrics(outputs: Any, targets: Any) -> dict[str, float]:
    """Diagnostics for a PPO epoch.

    None of these is the selection metric — that is the episodic return, which
    only an environment rollout can produce and which is merged in separately
    (see ``_run_epoch_batches`` and ``_rollout_evaluation``). These are the
    numbers that say *why* a return is or is not moving: a ``clip_fraction``
    near zero means the updates are too small to matter, an ``approx_kl`` that
    runs away means they are too large, and an ``explained_variance`` at or
    below zero means the critic is not predicting returns at all, so every
    advantage the policy is trained on is noise.
    """
    with torch.no_grad():
        terms = _ppo_terms(outputs, targets)
        return {
            "policy_loss": float(terms["policy_loss"]),
            "value_loss": float(terms["value_loss"]),
            "entropy": float(terms["entropy"]),
            "approx_kl": float(terms["approx_kl"]),
            "clip_fraction": float(terms["clip_fraction"]),
            "explained_variance": _explained_variance(
                terms["value"], terms["returns"]
            ),
        }


_PRIMARY_METRIC_KEY: dict[str, str] = {
    key: model_adapter(key).primary_metric_key
    for key in (
        "lenet5", "m5", "lstm_forecaster", "textcnn", "gcn", "tabnet",
        "mpnn", "actor_critic", "lstm_autoencoder", "distilbert",
        "dqn_lunarlander", "ppo_bipedalwalker", "attentivefp_freesolv",
        "gin_imdbb", "tcn_forecaster", "gru_forecaster",
        "pointnet_modelnet40", "vae_mnist", "snn_nmnist",
        "resnet18_cifar10", "resnet18_hf_perforated_cifar10",
        "mobilenetv2_cifar10", "saint_adult", "capsnet_mnist",
    )
}
# Kept outside the registered roster until its dataset support is promoted.
_PRIMARY_METRIC_KEY["unet_isic"] = "dice"


class CapsuleMarginLoss(torch.nn.Module):
    """Margin loss from Sabour et al., "Dynamic Routing Between Capsules" (sec. 3).

    CapsNet's output is a per-class capsule *length* in [0, 1), not a logit.
    Feeding those to ``CrossEntropyLoss`` squeezes every class into a softmax over
    a range narrower than one nat, so the gradient is tiny and nearly uniform:
    the 2026-08-05 run's train loss moved from 1.578 to 1.468 across 30 epochs
    and validation accuracy stalled at 98.7% against the paper's 99.5%. The
    margin loss scores each capsule independently against m+ / m-, which is what
    the architecture's routing was designed for.
    """

    def __init__(
        self,
        num_classes: int,
        m_positive: float = 0.9,
        m_negative: float = 0.1,
        down_weight: float = 0.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.m_positive = m_positive
        self.m_negative = m_negative
        self.down_weight = down_weight

    def forward(self, lengths: Any, targets: Any) -> Any:
        onehot = torch.nn.functional.one_hot(
            targets.long().flatten(), num_classes=self.num_classes
        ).to(lengths.dtype)
        present = torch.clamp(self.m_positive - lengths, min=0.0).square()
        absent = torch.clamp(lengths - self.m_negative, min=0.0).square()
        loss = onehot * present + self.down_weight * (1.0 - onehot) * absent
        return loss.sum(dim=-1).mean()


def _binary_or_multi_loss(model_key: str, config: TrainingConfig | None = None) -> Any:
    if model_key in {"tcn_forecaster", "gru_forecaster"}:
        # Both forecasters report MAE, so the training loss is a recipe choice
        # rather than a fixed property of the task.
        regression_loss = config.regression_loss if config is not None else "mse"
        if regression_loss == "mae":
            return torch.nn.L1Loss()
        if regression_loss == "smooth_l1":
            return torch.nn.SmoothL1Loss(beta=0.1)
        if regression_loss != "mse":
            raise ValueError(
                f"Unknown {model_key} regression loss {regression_loss!r}"
            )
        return torch.nn.MSELoss()
    if model_key in {"lstm_forecaster", "mpnn", "attentivefp_freesolv"}:
        return torch.nn.MSELoss()
    if model_key in {"lstm_autoencoder"}:
        return torch.nn.MSELoss()
    if model_key == "unet_isic":
        return torch.nn.BCEWithLogitsLoss()
    # These build their objective from the model's own multi-part output rather
    # than a criterion over (prediction, target): the VAE's ELBO, and PPO's
    # clipped surrogate + value + entropy. See _compute_loss.
    if model_key in {"vae_mnist", "ppo_bipedalwalker"}:
        return None
    if model_key == "capsnet_mnist":
        return CapsuleMarginLoss(num_classes=10)
    label_smoothing = float(getattr(config, "label_smoothing", 0.0) or 0.0)
    return torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)


def _safe_ratio(numerator: float, denominator: float, default: float = 0.0) -> float:
    if abs(denominator) < 1e-12:
        return default
    return numerator / denominator


def _batch_size(targets: Any) -> int:
    if hasattr(targets, "shape") and len(targets.shape) > 0:
        return int(targets.shape[0])
    return 1


def _detach_metric_payload(
    model_key: str, outputs: Any, targets: Any, metric_targets: Any | None
) -> tuple[Any, Any, Any | None]:
    if model_key == "actor_critic":
        outputs = outputs[0]
    # vae_mnist returns (reconstruction, mu, logvar) and ppo_bipedalwalker
    # returns (mean, log_std, value); anything multi-headed detaches per head.
    if isinstance(outputs, tuple):
        outputs = tuple(item.detach().cpu() for item in outputs)
    else:
        outputs = outputs.detach().cpu()
    targets = targets.detach().cpu()
    if metric_targets is not None:
        metric_targets = metric_targets.detach().cpu()
    return outputs, targets, metric_targets


def _average_precision(scores: Any, targets: Any) -> float:
    scores = scores.flatten().float()
    targets = targets.flatten().long()
    positives = int((targets == 1).sum().item())
    if positives == 0:
        return 0.0
    order = torch.argsort(scores, descending=True)
    sorted_targets = targets[order]
    tp = sorted_targets.cumsum(dim=0).float()
    precision = tp / torch.arange(1, len(sorted_targets) + 1, dtype=torch.float32)
    positive_positions = sorted_targets == 1
    if not positive_positions.any():
        return 0.0
    return float(precision[positive_positions].sum().item() / positives)


def _best_f1_threshold(
    scores: Any, targets: Any
) -> tuple[float, float, float, float, float]:
    scores = scores.flatten().float()
    targets = targets.flatten().long()
    positives = int((targets == 1).sum().item())
    negatives = int((targets == 0).sum().item())
    if positives == 0 or negatives == 0:
        threshold = float(scores.median().item()) if scores.numel() else 0.0
        return threshold, 0.0, 0.0, 0.0, 0.0

    order = torch.argsort(scores, descending=True)
    sorted_scores = scores[order]
    sorted_targets = targets[order]
    tp = sorted_targets.cumsum(dim=0).float()
    fp = torch.arange(1, len(sorted_targets) + 1, dtype=torch.float32) - tp
    fn = float(positives) - tp
    precision = tp / (tp + fp).clamp_min(1e-12)
    recall = tp / max(1, positives)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    best_index = int(torch.argmax(f1).item())
    tp_best = float(tp[best_index].item())
    fp_best = float(fp[best_index].item())
    fn_best = float(fn[best_index].item())
    tn_best = float(negatives - fp_best)
    precision_best = _safe_ratio(tp_best, tp_best + fp_best)
    recall_best = _safe_ratio(tp_best, tp_best + fn_best)
    specificity_best = _safe_ratio(tn_best, tn_best + fp_best)
    return (
        float(sorted_scores[best_index].item()),
        float(f1[best_index].item()),
        precision_best,
        recall_best,
        specificity_best,
    )


def _classification_metrics(logits: Any, targets: Any) -> dict[str, float]:
    logits = logits.float()
    targets = targets.long().flatten()
    predictions = logits.argmax(dim=-1)
    probs = torch.softmax(logits, dim=-1)
    num_classes = int(logits.shape[-1])
    metrics: dict[str, float] = {}
    metrics["accuracy"] = float((predictions == targets).float().mean().item())
    metrics["error_rate"] = 1.0 - metrics["accuracy"]
    metrics["confidence_mean"] = float(probs.max(dim=-1).values.mean().item())
    if num_classes >= 2:
        metrics["top2_accuracy"] = float(
            (
                torch.topk(logits, k=min(2, num_classes), dim=-1).indices
                == targets.unsqueeze(-1)
            )
            .any(dim=-1)
            .float()
            .mean()
            .item()
        )
    if num_classes >= 3:
        metrics["top3_accuracy"] = float(
            (torch.topk(logits, k=3, dim=-1).indices == targets.unsqueeze(-1))
            .any(dim=-1)
            .float()
            .mean()
            .item()
        )
    if num_classes >= 5:
        metrics["top5_accuracy"] = float(
            (torch.topk(logits, k=5, dim=-1).indices == targets.unsqueeze(-1))
            .any(dim=-1)
            .float()
            .mean()
            .item()
        )

    confusion = torch.zeros((num_classes, num_classes), dtype=torch.float64)
    cm_targets = targets.cpu()
    cm_predictions = predictions.cpu()
    valid_mask = (cm_targets >= 0) & (cm_targets < num_classes)
    if not valid_mask.all():
        cm_targets = cm_targets[valid_mask]
        cm_predictions = cm_predictions[valid_mask]
    indices = cm_targets * num_classes + cm_predictions
    confusion += (
        torch.bincount(indices, minlength=num_classes * num_classes)
        .reshape(num_classes, num_classes)
        .to(torch.float64)
    )
    total = float(confusion.sum().item())
    diag = confusion.diag()
    supports = confusion.sum(dim=1)
    predicted_counts = confusion.sum(dim=0)
    precisions = diag / predicted_counts.clamp_min(1.0)
    recalls = diag / supports.clamp_min(1.0)
    f1_scores = 2 * precisions * recalls / (precisions + recalls).clamp_min(1e-12)
    class_accuracy = (diag + (total - supports - predicted_counts + diag)) / max(
        total, 1.0
    )

    metrics["precision_macro"] = float(precisions.mean().item())
    metrics["recall_macro"] = float(recalls.mean().item())
    metrics["f1_macro"] = float(f1_scores.mean().item())
    support_total = supports.sum().clamp_min(1.0)
    support_weights = supports / support_total
    metrics["precision_weighted"] = float((precisions * support_weights).sum().item())
    metrics["recall_weighted"] = float((recalls * support_weights).sum().item())
    metrics["f1_weighted"] = float((f1_scores * support_weights).sum().item())
    metrics["balanced_accuracy"] = float(recalls.mean().item())

    expected = float(
        (supports * predicted_counts).sum().item() / max(total * total, 1.0)
    )
    observed = float(diag.sum().item() / max(total, 1.0))
    metrics["cohens_kappa"] = _safe_ratio(observed - expected, 1.0 - expected)

    cov_ytyp = float(
        diag.sum().item() * total - (supports * predicted_counts).sum().item()
    )
    cov_ypyp = float(total * total - (predicted_counts * predicted_counts).sum().item())
    cov_ytyt = float(total * total - (supports * supports).sum().item())
    metrics["mcc"] = _safe_ratio(cov_ytyp, math.sqrt(max(cov_ypyp * cov_ytyt, 1e-12)))

    for class_index in range(num_classes):
        metrics[f"class_{class_index}_support"] = float(supports[class_index].item())
        metrics[f"class_{class_index}_precision"] = float(
            precisions[class_index].item()
        )
        metrics[f"class_{class_index}_recall"] = float(recalls[class_index].item())
        metrics[f"class_{class_index}_f1"] = float(f1_scores[class_index].item())
        metrics[f"class_{class_index}_accuracy"] = float(
            class_accuracy[class_index].item()
        )

    if num_classes == 2:
        positive_scores = probs[:, 1]
        metrics["roc_auc"] = _auc(positive_scores, targets)
        metrics["average_precision"] = _average_precision(positive_scores, targets)
        threshold, best_f1, best_precision, best_recall, best_specificity = (
            _best_f1_threshold(positive_scores, targets)
        )
        metrics["best_threshold"] = threshold
        metrics["best_f1"] = best_f1
        metrics["best_precision"] = best_precision
        metrics["best_recall"] = best_recall
        metrics["specificity"] = best_specificity
        positive_preds = predictions == 1
        positive_targets = targets == 1
        true_positive = float((positive_preds & positive_targets).sum().item())
        false_positive = float((positive_preds & ~positive_targets).sum().item())
        false_negative = float((~positive_preds & positive_targets).sum().item())
        true_negative = float((~positive_preds & ~positive_targets).sum().item())
        metrics["precision"] = _safe_ratio(
            true_positive, true_positive + false_positive
        )
        metrics["recall"] = _safe_ratio(true_positive, true_positive + false_negative)
        metrics["f1"] = _safe_ratio(
            2 * metrics["precision"] * metrics["recall"],
            metrics["precision"] + metrics["recall"],
        )
        metrics["specificity_at_argmax"] = _safe_ratio(
            true_negative, true_negative + false_positive
        )

    return metrics


def _regression_metrics(preds: Any, targets: Any) -> dict[str, float]:
    preds = preds.float().flatten()
    targets = targets.float().flatten()
    errors = preds - targets
    abs_errors = errors.abs()
    squared_errors = errors.square()
    target_mean = targets.mean()
    centered_targets = targets - target_mean
    metrics: dict[str, float] = {
        "mae": float(abs_errors.mean().item()),
        "mse": float(squared_errors.mean().item()),
        "rmse": float(torch.sqrt(squared_errors.mean()).item()),
        "max_error": float(abs_errors.max().item()),
        "median_ae": float(abs_errors.median().item()),
    }
    denominator = float(centered_targets.square().sum().item())
    metrics["r2"] = 1.0 - _safe_ratio(
        float(squared_errors.sum().item()), denominator, default=0.0
    )
    variance_targets = float(targets.var(unbiased=False).item())
    variance_residual = float(errors.var(unbiased=False).item())
    metrics["explained_variance"] = 1.0 - _safe_ratio(
        variance_residual, variance_targets, default=0.0
    )
    nonzero_mask = targets.abs() > 1e-8
    if bool(nonzero_mask.any().item()):
        metrics["mape"] = float(
            (abs_errors[nonzero_mask] / targets[nonzero_mask].abs()).mean().item()
        )
    else:
        metrics["mape"] = 0.0
    denominator_smape = (preds.abs() + targets.abs()).clamp_min(1e-8)
    metrics["smape"] = float((2.0 * abs_errors / denominator_smape).mean().item())
    return metrics


def _anomaly_metrics(
    reconstructions: Any, targets: Any, labels: Any | None
) -> dict[str, float]:
    reductions = tuple(range(1, reconstructions.dim()))
    reconstruction_error = ((reconstructions.float() - targets.float()) ** 2).mean(
        dim=reductions
    )
    metrics: dict[str, float] = {
        "reconstruction_mse": float(reconstruction_error.mean().item()),
        "reconstruction_rmse": float(torch.sqrt(reconstruction_error.mean()).item()),
        "reconstruction_mae": float(
            (reconstructions.float() - targets.float()).abs().mean().item()
        ),
        "error_std": float(reconstruction_error.std(unbiased=False).item()),
        "error_max": float(reconstruction_error.max().item()),
    }
    if labels is None:
        return metrics
    labels = labels.long().flatten()
    metrics["auc"] = _auc(reconstruction_error, labels)
    metrics["average_precision"] = _average_precision(reconstruction_error, labels)
    threshold, best_f1, best_precision, best_recall, best_specificity = (
        _best_f1_threshold(reconstruction_error, labels)
    )
    metrics["best_threshold"] = threshold
    metrics["precision"] = best_precision
    metrics["recall"] = best_recall
    metrics["f1"] = best_f1
    metrics["specificity"] = best_specificity
    predictions = (reconstruction_error >= threshold).long()
    metrics["accuracy"] = float((predictions == labels).float().mean().item())
    return metrics


_RL_ENVIRONMENTS: dict[str, tuple[str, bool]] = {
    # model key -> (gymnasium id, continuous action space)
    "actor_critic": ("CartPole-v1", False),
    "dqn_lunarlander": ("LunarLander-v3", False),
    "ppo_bipedalwalker": ("BipedalWalker-v3", True),
}
# Reference returns for the environments above, for reading the number against:
# CartPole-v1 caps at 500, LunarLander counts 200 as solved, BipedalWalker 300.
_RL_EVAL_EPISODES: int = 20
_RL_EVAL_STEP_CAP: int = 2000


def _evaluate_episodic_return(
    model_key: str,
    model: Any,
    device: Any,
    *,
    episodes: int = _RL_EVAL_EPISODES,
    seed: int = 12345,
) -> dict[str, float]:
    """Roll the trained policy out in its gym environment and summarise returns.

    ``actor_critic`` and ``dqn_lunarlander`` are behaviour cloning. The metric
    they train and select on is agreement with a scripted policy over cached
    observations, which has no published counterpart at all — the suite's own
    docs warn that the scores must not be read against published
    CartPole/LunarLander results. Actually stepping the environment produces an
    episodic return that can be, so it is recorded here alongside the training
    metric. For those two it is deliberately *not* the selection metric:
    promoting it would put a different objective in front of the model than the
    loss it minimises, and a different one in front of the dendritic arm than
    the baseline. It is an extra column, evaluated once, after the best weights
    are restored.

    ``ppo_bipedalwalker`` is the exception: it trains on policy, so the return
    *is* the objective, and this function is also its validation and test
    evaluation by way of ``_rollout_evaluation``.

    Actions are taken deterministically — the Gaussian policy's mean, the
    Q-network's argmax — and continuous ones are clipped into the action space,
    since an unsquashed mean is not bounded by it.

    Episode seeds are fixed, so the same policy scores the same return in both
    arms. Returns an empty dict when gymnasium or the Box2D backend the two
    harder environments need is missing, so an optional dependency degrades to
    "no return recorded" rather than failing a finished training run.
    """
    entry = _RL_ENVIRONMENTS.get(model_key)
    if entry is None:
        return {}
    env_id, continuous = entry
    try:
        gymnasium: Any = importlib.import_module("gymnasium")
        env = getattr(gymnasium, "make")(env_id)
    except Exception as exc:  # missing gymnasium, missing Box2D, bad env id
        print(f"[rl-eval] {model_key}: {env_id} unavailable ({exc}); skipping return")
        return {}

    was_training = model.training
    model.eval()
    returns: list[float] = []
    # low/high live on Box, not on the generic Space that `make` is declared to
    # return, and `continuous` is exactly the flag saying this env has a Box.
    # gymnasium's annotation cannot express that, so the narrowing happens here.
    action_space: Any = env.action_space
    action_low = (
        torch.as_tensor(action_space.low, dtype=torch.float32) if continuous else None
    )
    action_high = (
        torch.as_tensor(action_space.high, dtype=torch.float32) if continuous else None
    )
    try:
        with torch.no_grad():
            for episode in range(episodes):
                observation, _ = env.reset(seed=seed + episode)
                total = 0.0
                for _ in range(_RL_EVAL_STEP_CAP):
                    batch = torch.as_tensor(
                        observation, dtype=torch.float32, device=device
                    ).unsqueeze(0)
                    output = model(batch)
                    if isinstance(output, tuple):
                        # ActorCritic returns (logits, value); PPOPolicy returns
                        # (action mean, log std, value). The first element is
                        # the deterministic action in both cases.
                        output = output[0]
                    if continuous:
                        action = torch.clamp(
                            output.squeeze(0).float().cpu(), action_low, action_high
                        ).numpy()
                    else:
                        action = int(output.argmax(dim=-1).item())
                    observation, reward, terminated, truncated, _ = env.step(action)
                    total += float(reward)
                    if terminated or truncated:
                        break
                returns.append(total)
    except Exception as exc:
        print(f"[rl-eval] {model_key}: rollout failed ({exc}); skipping return")
        return {}
    finally:
        env.close()
        model.train(was_training)

    mean_return = sum(returns) / len(returns)
    variance = sum((value - mean_return) ** 2 for value in returns) / len(returns)
    return {
        "episodic_return_mean": mean_return,
        "episodic_return_std": math.sqrt(variance),
        "episodic_return_min": min(returns),
        "episodic_return_max": max(returns),
        "episodic_return_episodes": float(len(returns)),
    }


def _rescale_to_dataset_units(value: Any, offset: float, scale: float) -> Any:
    """Undo a dataset's target standardisation so metrics keep their real units."""
    if math.isclose(scale, 1.0) and math.isclose(offset, 0.0):
        return value
    return value.float() * scale + offset


def _compute_all_metrics(
    model_key: str,
    outputs: Any,
    targets: Any,
    metric_targets: Any | None,
    *,
    metric_name: str,
    target_offset: float = 0.0,
    target_scale: float = 1.0,
) -> dict[str, float]:
    task_kind = (
        "segmentation" if model_key == "unet_isic" else model_adapter(model_key).task_kind
    )
    if model_key == "actor_critic" and isinstance(outputs, tuple):
        outputs = outputs[0]
    if task_kind == "regression":
        return _regression_metrics(
            _rescale_to_dataset_units(outputs, target_offset, target_scale),
            _rescale_to_dataset_units(targets, target_offset, target_scale),
        )
    if task_kind == "on_policy":
        return _ppo_metrics(outputs, targets)
    if task_kind == "vae":
        return _vae_metrics(outputs, targets)
    if task_kind == "segmentation":
        return {"dice": _dice_from_logits(outputs, targets)}
    if task_kind == "anomaly":
        return _anomaly_metrics(outputs, targets, metric_targets)
    metrics = _classification_metrics(outputs, targets)
    if model_key in {"actor_critic", "dqn_lunarlander"}:
        metrics["reward_proxy"] = metrics["accuracy"]
    if metric_name.lower() == "accuracy":
        metrics["primary_alias"] = metrics["accuracy"]
    return metrics


def _prefix_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def _cat_payload(items: list[Any]) -> Any:
    if not items:
        return items
    first = items[0]
    if isinstance(first, tuple):
        return tuple(torch.cat([item[index] for item in items], dim=0) for index in range(len(first)))
    return torch.cat(items, dim=0)


def _history_fieldnames(history: list[dict[str, Any]]) -> list[str]:
    preferred_order = [
        "epoch",
        "primary_metric_name",
        "primary_metric_key",
        "metric_direction",
        "learning_rate",
        "epoch_seconds",
        "train_loss",
        "train_primary_metric",
        "val_loss",
        "val_primary_metric",
        "test_loss",
        "test_primary_metric",
    ]
    seen = set()
    fieldnames: list[str] = []
    for name in preferred_order:
        if any(name in row for row in history):
            fieldnames.append(name)
            seen.add(name)
    extras = sorted({key for row in history for key in row.keys() if key not in seen})
    fieldnames.extend(extras)
    return fieldnames


def _forward(model_key: str, model: Any, batch: tuple[Any, ...]) -> tuple[Any, Any, Any]:
    if model_key == "gcn":
        # Transductive: the batch is the whole graph (batch dimension 1) plus
        # the node indices this split scores. The model returns a logit row per
        # node; selecting the split's rows here keeps the loss and every metric
        # downstream working on a plain [n_split, num_classes] tensor.
        x, adjacency, node_indices, targets = batch
        logits = model(x, adjacency)
        return logits[0].index_select(0, node_indices[0]), targets[0], None
    if model_key == "gin_imdbb":
        x, adjacency, targets = batch
        return model(x, adjacency), targets, None
    if model_key in {"mpnn", "attentivefp_freesolv"}:
        node_features, adjacency, edge_features, targets = batch
        return model(node_features, adjacency, edge_features), targets, None
    if model_key == "lstm_autoencoder":
        x, target, metric_targets = batch
        return model(x), target, metric_targets
    if model_key == "actor_critic":
        x, targets = batch
        return model(x), targets, None
    if model_key == "distilbert":
        input_ids, attention_mask, targets = batch
        return model(input_ids, attention_mask), targets, None
    if model_key == "vae_mnist":
        x, _ = batch
        return model(x), x, None
    if model_key == "ppo_bipedalwalker":
        # The rollout yields (obs, action, old_log_prob, advantage, return).
        # The last four are packed into one tensor so the accumulator, the
        # batch-size helper and the detach helper all keep seeing a tensor;
        # _unpack_ppo_targets reverses it inside the loss and the metrics.
        observation, action, old_log_prob, advantage, returns = batch
        targets = torch.cat(
            [
                action,
                old_log_prob.unsqueeze(-1),
                advantage.unsqueeze(-1),
                returns.unsqueeze(-1),
            ],
            dim=-1,
        )
        return model(observation), targets, None
    x, targets = batch
    return model(x), targets, None


def _first_tensor(value: Any) -> Any | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(value, dict):
        for item in value.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _dimension_vector_for_module_output(module: Any, output: Any) -> list[int] | None:
    tensor = _first_tensor(output)
    if tensor is None:
        return None
    rank = getattr(tensor, "ndim", 0)
    if rank < 2:
        return None
    dimensions = [-1] * rank
    module_type = type(module).__name__.lower()
    output_axis = 1 if "conv" in module_type and rank > 1 else rank - 1
    dimensions[output_axis] = 0
    return dimensions


def _module_matches_any(module: Any, module_classes: tuple[Any, ...]) -> bool:
    return any(isinstance(module, module_class) for module_class in module_classes)


def _sample_batch_from_loader(train_loader: Any) -> Any:
    dataset = getattr(train_loader, "dataset", None)
    collate_fn = getattr(train_loader, "collate_fn", None)
    batch_size = getattr(train_loader, "batch_size", None) or 1
    if (
        dataset is not None
        and hasattr(dataset, "__getitem__")
        and hasattr(dataset, "__len__")
    ):
        sample_count = min(max(1, int(batch_size)), len(dataset), 2)
        samples = [dataset[index] for index in range(sample_count)]
        if collate_fn is not None:
            return collate_fn(samples)
        return samples[0] if sample_count == 1 else samples
    return next(iter(train_loader))


def infer_module_output_dimensions(
    model: Any,
    model_key: str,
    bundle: Any,
    module_classes: list[Any],
    module_names: list[str] | None = None,
) -> dict[str, list[int]]:
    valid_classes = tuple(
        module_class for module_class in module_classes if isinstance(module_class, type)
    )
    selected_module_names = {
        module_name.lstrip(".") for module_name in (module_names or [])
    }
    if not valid_classes and not selected_module_names:
        return {}

    device = next(
        (parameter.device for parameter in model.parameters()),
        torch.device("cpu"),
    )
    dimensions: dict[str, list[int]] = {}
    handles = []

    def make_hook(module_name: str) -> Any:
        def hook(module: Any, _inputs: Any, output: Any) -> None:
            vector = _dimension_vector_for_module_output(module, output)
            if vector is not None:
                dimensions[f".{module_name}"] = vector

        return hook

    for module_name, module in model.named_modules():
        if not module_name:
            continue
        if (
            module_name in selected_module_names
            or _module_matches_any(module, valid_classes)
        ):
            handles.append(module.register_forward_hook(make_hook(module_name)))

    was_training = bool(getattr(model, "training", False))
    try:
        model.eval()
        batch = _sample_batch_from_loader(bundle.train_loader)
        batch = _move_batch_to_device(batch, device)
        with torch.no_grad():
            _forward(model_key, model, batch)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    return dimensions


def _pointnet_feature_transform_penalty(model: Any) -> Any | None:
    """Orthogonality penalty for PointNet's unconstrained 64x64 feature-transform.

    Without this term the feature-transform T-Net is free to drift far from
    orthogonal, which was blowing up val_loss to ~15 (vs. train_loss ~0.3) and
    capping val accuracy near random (~8%) while train accuracy hit ~92% —
    the model was fitting itself around whatever a wildly non-orthogonal
    matrix did on a given batch rather than a stable point-cloud feature
    space. This is the regularizer from the original PointNet paper (Qi et
    al., https://arxiv.org/abs/1612.00593, sec 3.4), applied only to the
    feature transform (not the 3x3 input transform, which is small enough
    to stay stable unregularized).
    """
    base = _unwrap_compiled(model)
    matrix = getattr(base, "_feature_transform_matrix", None)
    if matrix is None:
        return None
    k = matrix.shape[-1]
    eye = torch.eye(k, device=matrix.device, dtype=matrix.dtype).unsqueeze(0)
    return torch.mean(
        torch.norm(torch.bmm(matrix, matrix.transpose(2, 1)) - eye, dim=(1, 2))
    )


def _compute_loss(
    model_key: str, criterion: Any, outputs: Any, targets: Any, model: Any = None
) -> Any:
    if model_key == "actor_critic":
        return criterion(outputs[0], targets)
    if model_key == "vae_mnist":
        return _vae_loss(outputs, targets)
    if model_key == "ppo_bipedalwalker":
        return _ppo_loss(outputs, targets)
    if model_key == "pointnet_modelnet40" and model is not None:
        penalty = _pointnet_feature_transform_penalty(model)
        loss = criterion(outputs, targets)
        return loss if penalty is None else loss + 0.001 * penalty
    return criterion(outputs, targets)


def _unwrap_compiled(model: Any) -> Any:
    """Return the underlying ``nn.Module`` when the model is a ``torch.compile``
    wrapper (``torch._dynamo.OptimizedModule``).

    ``torch.compile`` stores the original module as ``self._orig_mod`` and
    prefixes every key in ``state_dict()`` with ``_orig_mod.``.  Always saving
    and loading checkpoints through the unwrapped module keeps key names clean
    so downstream conditions can load the file into a fresh, uncompiled model.
    """
    return getattr(model, "_orig_mod", model)


def _finalize_quantized_model_for_eval(model: Any, config: "TrainingConfig") -> Any:
    """Return deployment weights without accidentally re-quantizing a QAT model.

    A QAT batch ends by projecting the full-precision shadow into ``.data`` so
    validation, checkpoint selection, and the next batch all observe precisely
    the deployment grid. Reapplying PTQ after restoring the selected state is a
    *second* calibration/projection, which is not generally idempotent (most
    visibly for ternary scales). PTQ-only conditions still need their one final
    projection here.
    """
    if not _should_quantize_for_eval(config):
        return model
    if _should_quantize_for_training(config):
        return model
    return _make_quantized_copy(
        model,
        config.bit_width,
        config.quantization_mode,
        config.quantization_granularity,
    )


def _artifact_path(output_dir: Path, use_dendrites: bool) -> Path:
    preferred = output_dir / _MODEL_PT
    if preferred.exists():
        return preferred
    # Backwards compatibility for older runs that wrote multiple checkpoint names.
    if use_dendrites:
        for candidate in ("best_model", "final_clean_pai"):
            path = output_dir / candidate
            if path.exists():
                return path
    return preferred


def _count_parameters(model: Any) -> tuple[int, int]:
    # Materialize model.parameters() once and derive both counts from that
    # single snapshot. Calling model.parameters() twice (once per sum())
    # let capsnet_mnist/dendrites_q1 report nonzero_params (24,453,036)
    # greater than param_count (22,795,776): PerforatedAI's dendrite wrapper
    # can lazily materialize/mutate its module tree on traversal, so the two
    # generator calls silently walked different parameter sets. Nonzero can
    # never exceed total when both are summed from the same list.
    params = list(model.parameters())
    param_count = sum(p.numel() for p in params)
    nonzero_params = sum((p != 0).sum().item() for p in params)
    return param_count, nonzero_params


def _topology_hash(model: Any) -> str:
    """A deterministic identity for a model's architecture, not its weights.

    Hashes each parameter's dotted name, owning-module class name, and shape
    -- never its values -- so retraining the identical architecture reproduces
    the same hash while a real shape change (a retained-vs-rejected dendrite)
    changes it. information/optimization/00_assessment.md: "A retained
    dendrite is an architectural change, not merely a metric event."
    """
    owners = dict(model.named_modules())
    entries = []
    for name, param in model.named_parameters():
        module_name, _, _leaf = name.rpartition(".")
        owner = owners.get(module_name)
        owner_class = type(owner).__name__ if owner is not None else ""
        entries.append(f"{name}|{owner_class}|{tuple(param.shape)}")
    canonical = "\n".join(sorted(entries))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _final_clean_pai_parameter_stats(model: Any) -> tuple[int, int, str]:
    """Count the inference model PAI writes as ``final_clean_pai.pt``.

    A live PAI wrapper retains the next candidate dendrite and its training
    bookkeeping. Those tensors are not part of the final inference topology,
    but ``model.parameters()`` includes them. PAI's own final export calls
    ``prepare_final_model`` to deep-copy, blockwise-convert, and strip that
    scaffolding, so use the same representation for benchmark parameter
    reporting. This must run only after all training and evaluation complete:
    PAI clears processor state while constructing the copy.

    Also returns this same final-clean model's :func:`_topology_hash`, so the
    one (expensive, state-clearing) ``prepare_final_model`` call serves both
    parameter accounting and the manifest's topology-hash field.
    """
    try:
        UPA = importlib.import_module("perforatedai.utils_perforatedai")
        prepare_final_model = getattr(UPA, "prepare_final_model")
    except Exception as exc:
        raise RuntimeError(
            "PerforatedAI's final-clean export is required to count a dendritic "
            "model accurately."
        ) from exc
    try:
        with pai_runtime_guard():
            final_clean_model = prepare_final_model(model)
    except Exception as exc:
        raise RuntimeError(
            "PerforatedAI could not prepare the final-clean inference model for "
            "parameter accounting."
        ) from exc
    param_count, nonzero_params = _count_parameters(final_clean_model)
    return param_count, nonzero_params, _topology_hash(final_clean_model)


def _write_metrics_and_history(
    *,
    output_dir: Path,
    metadata: ArtifactMetadata,
    payload: ArtifactPayload,
    stats: ArtifactStats,
) -> None:
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "model_key": metadata.model_key,
                "condition_key": metadata.condition_key,
                "display_name": metadata.display_name,
                "metric_name": metadata.metric_name,
                "metric_direction": metadata.metric_direction,
                "primary_metric_key": metadata.primary_metric_key,
                "best_metric_value": payload.best_metric,
                "metric_value": payload.final_metric,
                "best_epoch": payload.best_epoch,
                "train_history_columns": _history_fieldnames(payload.history),
                "test_loss": payload.test_loss,
                "test_metrics": payload.test_metrics,
                "param_count": stats.param_count,
                "nonzero_params": stats.nonzero_params,
                "file_size_mb": stats.file_size_mb,
                "use_dendrites": metadata.use_dendrites,
                "use_pruning": metadata.use_pruning,
                "bit_width": metadata.bit_width,
                "quantization_granularity": metadata.quantization_granularity,
                "use_qat": metadata.use_qat,
                "fine_tune_epochs": metadata.fine_tune_epochs,
                "regression_loss": metadata.regression_loss,
                "lr_schedule_epochs": metadata.lr_schedule_epochs,
                "quantization_evaluation_revision": (
                    metadata.quantization_evaluation_revision
                ),
                "dendrite_audit_revision": metadata.dendrite_audit_revision,
                "dense_param_count": metadata.dense_param_count,
                "source_dendrite_audit_status": (
                    metadata.source_dendrite_audit_status
                ),
                "enable_pai_dendrite_updates": metadata.enable_pai_dendrite_updates,
                "train_dendrites_until_complete": metadata.train_dendrites_until_complete,
                "freeze_dendrite_updates_fraction": (
                    metadata.freeze_dendrite_updates_fraction
                ),
                "memory_cleanup_interval_batches": (
                    metadata.memory_cleanup_interval_batches
                ),
                "model_scale": metadata.model_scale,
                "model_revision": metadata.model_revision,
                "pai_variant": metadata.pai_variant,
                "pai_fixed_switch_interval": metadata.pai_fixed_switch_interval,
                "pai_dynamic_schedule": metadata.pai_dynamic_schedule,
                "pai_save_name": metadata.pai_save_name,
                "artifact_id": metadata.artifact_id,
                "seed": metadata.seed,
                "source_condition_key": metadata.source_condition_key,
                "quantizer_revision": metadata.quantizer_revision,
                "module_ids_to_perforate": metadata.module_ids_to_perforate,
                "track_only_module_ids": metadata.track_only_module_ids,
                "parameter_ids_to_track": metadata.parameter_ids_to_track,
                "recipe_override": metadata.recipe_override,
                "pai_override": metadata.pai_override,
                "effective_recipe": metadata.effective_recipe,
                "max_dendrite_phase_epochs": metadata.max_dendrite_phase_epochs,
                "source_commit": metadata.source_commit,
                "paired_control_identity": metadata.paired_control_identity,
                "artifact_path": str(stats.artifact_path),
                "training_skipped": payload.training_skipped,
                "skip_reason": payload.skip_reason,
                "stage_name": payload.stage_name,
            },
            indent=2,
        )
    )
    with (output_dir / "history.csv").open("w", newline="") as fh:
        fieldnames = _history_fieldnames(payload.history)
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(payload.history)


def _persist_stage_artifacts(
    *,
    output_dir: Path,
    plain_model: Any,
    metadata: ArtifactMetadata,
    payload: ArtifactPayload,
    parameter_stats: tuple[int, int] | None = None,
    topology_hash: str | None = None,
) -> tuple[Path, float, int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / _MODEL_PT
    torch.save(plain_model.state_dict(), checkpoint_path)
    artifact_path = _artifact_path(output_dir, metadata.use_dendrites)
    file_size_mb = artifact_path.stat().st_size / (1024 * 1024)
    param_count, nonzero_params = parameter_stats or _count_parameters(plain_model)
    if metadata.use_dendrites:
        _write_dendritic_sidecars(
            output_dir=output_dir,
            history=payload.history,
            best_metric=payload.best_metric,
            best_epoch=payload.best_epoch,
            param_count=param_count,
            nonzero_params=nonzero_params,
            metric_name=metadata.metric_name,
            metric_direction=metadata.metric_direction,
        )
        _write_pai_summary(
            output_dir=output_dir,
            history=payload.history,
            metadata=metadata,
            param_count=param_count,
            nonzero_params=nonzero_params,
        )
    stats = ArtifactStats(
        param_count=param_count,
        nonzero_params=nonzero_params,
        file_size_mb=file_size_mb,
        artifact_path=artifact_path,
    )
    _write_metrics_and_history(
        output_dir=output_dir,
        metadata=metadata,
        payload=payload,
        stats=stats,
    )
    if not metadata.artifact_id:
        raise RuntimeError("artifact_id is required before persisting an artifact")
    dendrite_status = "not_applicable"
    pai_schedule_telemetry: dict[str, Any] = {}
    pai_epoch_milestones: dict[str, Any] = {}
    if metadata.use_dendrites:
        try:
            summary = json.loads((output_dir / "pai_summary.json").read_text())
            dendrite_status = str(
                summary.get("dendrite_audit", {}).get("status", "unknown")
            )
            pai_schedule_telemetry = {
                "requested": summary.get("requested_schedule", {}),
                "observed": summary.get("observed_schedule", {}),
            }
            pai_epoch_milestones = summary.get("pai_epoch_milestones", {}) or {}
        except (OSError, json.JSONDecodeError):
            dendrite_status = "unknown"
    bit_width = metadata.bit_width
    quantized = bit_width is not None and bit_width < 32
    quantization_status = (
        "not_applicable"
        if not quantized
        else (
            "current"
            if metadata.quantization_evaluation_revision
            == QUANTIZATION_EVALUATION_REVISION
            else "unknown"
        )
    )
    write_artifact_manifest(
        output_dir,
        artifact_id=metadata.artifact_id,
        identity={
            "model_key": metadata.model_key,
            "condition_key": metadata.condition_key,
            "source_condition_key": metadata.source_condition_key,
            "model_revision": metadata.model_revision,
            "dataset_revision": metadata.dataset_revision,
            "model_scale": metadata.model_scale,
            "seed": metadata.seed,
            "pai_variant": metadata.pai_variant,
            "pai_switch_mode": (
                "fixed_diagnostic"
                if metadata.pai_fixed_switch_interval is not None
                else "history"
            )
            if metadata.enable_pai_dendrite_updates
            else "not_applicable",
            "pai_fixed_switch_interval": metadata.pai_fixed_switch_interval,
            "pai_dynamic_schedule": metadata.pai_dynamic_schedule,
            "dendrite_audit_revision": metadata.dendrite_audit_revision,
            "quantization_evaluation_revision": (
                metadata.quantization_evaluation_revision
            ),
            "quantizer_revision": metadata.quantizer_revision,
            "module_ids_to_perforate": metadata.module_ids_to_perforate,
            "track_only_module_ids": metadata.track_only_module_ids,
            "parameter_ids_to_track": metadata.parameter_ids_to_track,
            "recipe_override": metadata.recipe_override,
            "pai_override": metadata.pai_override,
            "max_dendrite_phase_epochs": metadata.max_dendrite_phase_epochs,
            "source_commit": metadata.source_commit,
            "paired_control_identity": metadata.paired_control_identity,
        },
        pai_save_name=metadata.pai_save_name,
        validity={
            "dendrite_status": dendrite_status,
            "quantization_status": quantization_status,
        },
        telemetry={
            "pai_schedule": pai_schedule_telemetry,
            "pai_epoch_milestones": pai_epoch_milestones,
            "topology_hash": topology_hash,
            "effective_recipe": metadata.effective_recipe,
            "learning_rates": {
                "backbone": [
                    row.get("backbone_learning_rate")
                    for row in payload.history
                    if row.get("backbone_learning_rate") is not None
                ],
                "dendrite": [
                    row.get("dendrite_learning_rate")
                    for row in payload.history
                    if row.get("dendrite_learning_rate") is not None
                ],
            },
        },
        additional_files=tuple(
            name
            for name in ("pai_summary.json", "PAI_config.json", "artifact_attempt.json")
            if (output_dir / name).is_file()
        ),
    )
    return artifact_path, file_size_mb, param_count, nonzero_params


def _format_metric_value(value: float) -> str:
    if math.isfinite(value):
        return f"{value:.4f}"
    return "n/a"


def _metric_display_key(metric_name: str) -> str:
    return metric_name.strip().lower().replace(" ", "_")


def _write_dendritic_sidecars(
    output_dir: Path,
    history: list[dict[str, Any]],
    best_metric: float,
    best_epoch: int,
    param_count: int,
    nonzero_params: int,
    metric_name: str,
    metric_direction: str,
) -> None:
    best_arch_rows = [
        {
            "cycle": row["epoch"],
            "best_metric_value": best_metric
            if row["epoch"] == best_epoch
            else row.get("val_primary_metric", row.get("val_metric", best_metric)),
            "best_epoch": best_epoch,
            "metric_name": metric_name,
            "metric_direction": metric_direction,
            "param_count": param_count,
            "nonzero_params": nonzero_params,
        }
        for row in history
    ]
    with (output_dir / "best_arch_scores.csv").open("w", newline="") as fh:
        if best_arch_rows:
            writer = csv.DictWriter(fh, fieldnames=list(best_arch_rows[0].keys()))
            writer.writeheader()
            writer.writerows(best_arch_rows)
    with (output_dir / "paramCounts.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["epoch", "param_count", "nonzero_params"]
        )
        writer.writeheader()
        for row in history:
            writer.writerow(
                {
                    "epoch": row["epoch"],
                    "param_count": param_count,
                    "nonzero_params": nonzero_params,
                }
            )


def _history_flag(row: dict[str, Any], field: str) -> bool:
    value = row.get(field, False)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes"}


def _load_continued_history(output_dir: Path) -> list[dict[str, Any]]:
    """Load the over-budget PAI phase retained beside the canonical artifact."""
    path = output_dir / "continued_until_complete" / "history.csv"
    if not path.exists():
        return []
    try:
        with path.open(newline="") as fh:
            return list(csv.DictReader(fh))  # type: ignore[no-matching-overload]
    except (OSError, csv.Error):
        return []


def _read_pai_architecture_log(save_name: str | None) -> dict[str, Any]:
    """Read PAI's raw retained-topology count CSV.

    ``*_best_arch_scores.csv`` is a score-selection report, not a topology
    ledger: after a retained dendrite performs worse than its dense parent it
    can contain only the earlier dense row. PAI's ``param_counts.csv`` records
    the architecture at each switch and is the raw evidence needed to compare
    with the exported final-clean model.
    """
    if not save_name:
        return {"status": "not_configured"}
    folder = pai_save_path(save_name)
    # PAI 3.x writes ``<save>param_counts.csv``. Retain the camel-case form
    # solely for older PAI artifacts; do not fall back to best_arch_scores.
    path = folder / f"{folder.name}param_counts.csv"
    if not path.exists():
        legacy_path = folder / f"{folder.name}paramCounts.csv"
        if legacy_path.exists():
            path = legacy_path
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    try:
        with path.open(newline="") as fh:
            rows = list(csv.DictReader(fh))  # type: ignore[no-matching-overload]
    except (OSError, csv.Error):
        return {"status": "unreadable", "path": str(path)}
    if not rows:
        return {"status": "empty", "path": str(path), "row_count": 0}
    param_column = next(
        (name for name in rows[0] if "param" in name.lower()), None
    )
    switch_column = next(
        (name for name in rows[0] if "number" in name.lower()), None
    )
    counts: list[int] = []
    # {switch_number: param_count}, for joining against _read_pai_switch_log's
    # per-switch epochs (see _pai_epoch_milestones). Only populated when both
    # columns parse; a row with an unparsable switch number is excluded rather
    # than guessed from row order, since param_counts.csv and switch_epochs.csv
    # are independently-lengthed logs for the same run (see MEASUREMENT_CAVEATS.md
    # and _write_pai_summary's docstring).
    by_switch: dict[int, int] = {}
    if param_column is not None:
        for row in rows:
            try:
                count = int(float(row[param_column]))
            except (TypeError, ValueError):
                continue
            counts.append(count)
            if switch_column is not None:
                try:
                    by_switch[int(float(row[switch_column]))] = count
                except (TypeError, ValueError):
                    pass
    return {
        "status": "available",
        "path": str(path),
        "row_count": len(rows),
        "max_param_count": max(counts) if counts else None,
        "param_count_by_switch": by_switch,
    }


def _read_pai_switch_log(save_name: str | None) -> dict[str, Any]:
    if not save_name:
        return {"status": "not_configured"}
    folder = pai_save_path(save_name)
    path = folder / f"{folder.name}switch_epochs.csv"
    if not path.exists():
        return {"status": "missing", "path": str(path)}
    try:
        with path.open(newline="") as fh:
            rows = list(csv.DictReader(fh))  # type: ignore[no-matching-overload]
    except (OSError, csv.Error):
        return {"status": "unreadable", "path": str(path)}
    epoch_column = next(
        (name for name in (rows[0] if rows else {}) if "epoch" in name.lower()),
        None,
    )
    switch_column = next(
        (name for name in (rows[0] if rows else {}) if "number" in name.lower()),
        None,
    )
    epochs: list[int] = []
    # {switch_number: epoch}, keyed by the CSV's own "Switch Number" column
    # rather than row order -- see _read_pai_architecture_log's by_switch.
    epochs_by_switch: dict[int, int] = {}
    if epoch_column is not None:
        for row in rows:
            try:
                epoch = int(float(row[epoch_column]))
            except (TypeError, ValueError):
                continue
            epochs.append(epoch)
            if switch_column is not None:
                try:
                    epochs_by_switch[int(float(row[switch_column]))] = epoch
                except (TypeError, ValueError):
                    pass
    return {
        "status": "available",
        "path": str(path),
        "row_count": len(rows),
        "switch_epochs": epochs,
        "epoch_by_switch": epochs_by_switch,
    }


def _pai_epoch_milestones(
    *,
    raw_architecture: dict[str, Any],
    raw_switches: dict[str, Any],
    dense_param_count: int | None,
    complete_epochs: list[int],
) -> dict[str, int | None]:
    """Name the three PAI lifecycle epochs the raw logs only imply.

    ``first_candidate_epoch`` is the epoch of PAI's first logged switch -- "the
    closest existing proxy" per information/optimization/03_execution_matrix.md
    -- now given an explicit name instead of left as
    ``observed_schedule["switch_epochs"][0]`` for every caller to reach into.
    It falls back to the ordered ``switch_epochs`` list when the log has no
    "Switch Number" column to key on.

    ``first_retention_epoch`` is the epoch of the first switch whose
    ``param_counts.csv`` row grew the parameter count versus the previous
    switch (or versus ``dense_param_count``, for switch 0) -- i.e. the first
    switch that actually kept a dendrite rather than discarding a candidate.
    This joins ``param_counts.csv`` and ``switch_epochs.csv`` by their shared
    "Switch Number" column rather than assuming matched row order: the two
    logs can have different lengths for the same run (observed on disk, e.g.
    ``results/dynamic5/PAI/actor_critic_dendrites_fp32/``), so a retained
    switch with no matching epoch row returns ``None`` rather than a guess.

    ``completion_epoch`` is the first epoch the benchmark's own
    ``pai_training_complete`` flag was observed (authoritative history, per
    ``_write_pai_summary``'s docstring), not derived from either raw PAI log.

    Best-effort telemetry, not a verdict: see ``_dendrite_audit`` for the
    actual retained-vs-not status this benchmark treats as evidence.
    """
    epoch_by_switch: dict[int, int] = raw_switches.get("epoch_by_switch") or {}
    if epoch_by_switch:
        # The lowest switch *number*, not the lowest epoch: those coincide in
        # every log seen so far, but the question asked is "when did the first
        # switch happen", and the switch number is what answers it.
        first_candidate_epoch = epoch_by_switch[min(epoch_by_switch)]
    else:
        # No usable "Switch Number" column (an older or reworded log). The
        # ordered epoch list is still a valid answer for this one field, and
        # returning None here while switch_epochs plainly holds the epoch
        # would be a worse report than the value it already carries.
        switch_epochs: list[int] = raw_switches.get("switch_epochs") or []
        first_candidate_epoch = min(switch_epochs) if switch_epochs else None

    param_count_by_switch: dict[int, int] = (
        raw_architecture.get("param_count_by_switch") or {}
    )
    first_retention_epoch: int | None = None
    if param_count_by_switch:
        previous_count = dense_param_count
        for switch_number in sorted(param_count_by_switch):
            count = param_count_by_switch[switch_number]
            if previous_count is not None and count > previous_count:
                first_retention_epoch = epoch_by_switch.get(switch_number)
                break
            previous_count = count

    return {
        "first_candidate_epoch": first_candidate_epoch,
        "first_retention_epoch": first_retention_epoch,
        "completion_epoch": min(complete_epochs) if complete_epochs else None,
    }


def _write_pai_summary(
    *,
    output_dir: Path,
    history: list[dict[str, Any]],
    metadata: ArtifactMetadata,
    param_count: int,
    nonzero_params: int,
) -> None:
    """Persist a condition-local, final-checkpoint PAI summary.

    PAI's own CSVs may stop before the final continuation phase. The benchmark
    history and the final model checkpoint are therefore authoritative here;
    the PAI CSVs are included only as auditable raw telemetry.
    """
    continued_history = _load_continued_history(output_dir)
    all_history = [*history, *continued_history]
    restructured_epochs = [
        int(row["epoch"])
        for row in all_history
        if _history_flag(row, "pai_restructured") and str(row.get("epoch", "")).isdigit()
    ]
    complete_epochs = [
        int(row["epoch"])
        for row in all_history
        if _history_flag(row, "pai_training_complete")
        and str(row.get("epoch", "")).isdigit()
    ]
    raw_architecture = _read_pai_architecture_log(metadata.pai_save_name)
    raw_switches = _read_pai_switch_log(metadata.pai_save_name)
    observed_switch_epochs = list(raw_switches.get("switch_epochs", []))
    observed_intervals = [
        later - earlier
        for earlier, later in zip(observed_switch_epochs, observed_switch_epochs[1:])
    ]
    forced_switches = [
        {
            "epoch": int(row["epoch"]),
            "reason": str(row["pai_switch_reason"]),
        }
        for row in all_history
        if row.get("pai_switch_reason") == "candidate_phase_timeout"
        and str(row.get("epoch", "")).isdigit()
    ]
    termination_reason = next(
        (
            str(row["training_termination_reason"])
            for row in reversed(all_history)
            if row.get("training_termination_reason")
        ),
        "unknown",
    )
    dendrite_audit = _dendrite_audit(
        metadata=metadata,
        param_count=param_count,
        raw_architecture=raw_architecture,
        raw_switches=raw_switches,
    )
    epoch_milestones = _pai_epoch_milestones(
        raw_architecture=raw_architecture,
        raw_switches=raw_switches,
        dense_param_count=metadata.dense_param_count,
        complete_epochs=complete_epochs,
    )
    raw_param_count = raw_architecture.get("max_param_count")
    if raw_param_count is None:
        consistency = "unavailable"
    elif raw_param_count == param_count:
        consistency = "match"
    elif raw_param_count < param_count:
        consistency = "stale"
    else:
        consistency = "inconsistent"
    (output_dir / "pai_summary.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "authoritative_source": "benchmark_history_and_final_checkpoint",
                "model_key": metadata.model_key,
                "condition_key": metadata.condition_key,
                "pai_variant": metadata.pai_variant,
                "fixed_switch_interval": metadata.pai_fixed_switch_interval,
                "dynamic_schedule": metadata.pai_dynamic_schedule,
                "requested_schedule": {
                    "mode": (
                        "fixed_diagnostic"
                        if metadata.pai_fixed_switch_interval is not None
                        else "history"
                    ),
                    "fixed_switch_interval": metadata.pai_fixed_switch_interval,
                    "dynamic_schedule": metadata.pai_dynamic_schedule,
                },
                "observed_schedule": {
                    "switch_epochs": observed_switch_epochs,
                    "switch_intervals": observed_intervals,
                    "forced_switches": forced_switches,
                    "termination_reason": termination_reason,
                },
                "final_model": {
                    "param_count": param_count,
                    "nonzero_params": nonzero_params,
                },
                "history": {
                    "canonical_epoch_count": len(history),
                    "continued_epoch_count": len(continued_history),
                    "restructured_epochs": restructured_epochs,
                    "training_complete_epochs": complete_epochs,
                },
                "raw_pai_logs": {
                    "architecture": raw_architecture,
                    "switches": raw_switches,
                },
                "architecture_log_consistency": consistency,
                "dendrite_audit": dendrite_audit,
                "pai_epoch_milestones": epoch_milestones,
            },
            indent=2,
        )
    )


def _dendrite_audit(
    *,
    metadata: ArtifactMetadata,
    param_count: int,
    raw_architecture: dict[str, Any] | None = None,
    raw_switches: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the evidence-based status for a dendritic artifact.

    ``pai_restructured`` is an implementation event, not proof that PAI kept a
    dendrite in the final inference topology. A retained run needs all three:
    a larger final model, PAI's initial-and-insertion switch records, and a raw
    architecture count agreeing with the exported model. Quantized descendants
    do not search again, so they inherit their source FP32 result's status.
    """
    if not metadata.use_dendrites:
        return {
            "revision": metadata.dendrite_audit_revision,
            "status": "not_applicable",
            "reason": "condition does not use dendrites",
        }
    if not metadata.enable_pai_dendrite_updates:
        source_status = metadata.source_dendrite_audit_status
        if source_status in {"verified_retained", "inherited_verified_retained"}:
            return {
                "revision": metadata.dendrite_audit_revision,
                "status": "inherited_verified_retained",
                "reason": "inherits verified retained topology from source FP32 artifact",
                "source_status": source_status,
            }
        inherited_status = (
            "inherited_no_retained_insertion"
            if source_status == "no_retained_insertion"
            else "inherited_unverified"
        )
        return {
            "revision": metadata.dendrite_audit_revision,
            "status": inherited_status,
            "reason": "source FP32 artifact has no verified retained dendrite",
            "source_status": source_status,
        }

    raw_architecture = raw_architecture or _read_pai_architecture_log(
        metadata.pai_save_name
    )
    raw_switches = raw_switches or _read_pai_switch_log(metadata.pai_save_name)
    switch_count = int(raw_switches.get("row_count", 0) or 0)
    raw_param_count = raw_architecture.get("max_param_count")
    dense_param_count = metadata.dense_param_count
    audit = {
        "revision": metadata.dendrite_audit_revision,
        "dense_param_count": dense_param_count,
        "final_param_count": param_count,
        "raw_switch_count": switch_count,
    }
    if raw_switches.get("status") != "available":
        return {
            **audit,
            "status": "unverified",
            "reason": "raw PAI switch log is unavailable",
        }
    # The first row marks PAI's initial transition into candidate tracking; a
    # second is required to show that a candidate was actually inserted.
    if switch_count < 2:
        return {
            **audit,
            "status": "no_retained_insertion",
            "reason": "raw PAI switch log has no candidate-insertion switch",
        }
    if dense_param_count is None:
        return {
            **audit,
            "status": "unverified",
            "reason": "dense reference parameter count was not recorded",
        }
    if param_count <= dense_param_count:
        return {
            **audit,
            "status": "no_retained_insertion",
            "reason": "final model did not retain parameters beyond the dense reference",
        }
    if raw_architecture.get("status") != "available" or raw_param_count is None:
        return {
            **audit,
            "status": "unverified",
            "reason": "raw PAI architecture log is unavailable",
        }
    if int(raw_param_count) != param_count:
        return {
            **audit,
            "status": "unverified",
            "reason": "raw PAI architecture count disagrees with final exported model",
            "raw_param_count": raw_param_count,
        }
    return {
        **audit,
        "status": "verified_retained",
        "reason": "raw switch and architecture logs match a larger final topology",
        "raw_param_count": raw_param_count,
    }


def _apply_pruning(model: Any, torch: Any, prune_amount: float) -> None:
    try:
        import torch.nn.utils.prune as prune

        parameters_to_prune = [
            (module, "weight")
            for module in model.modules()
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d))
        ]
        if parameters_to_prune:
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=prune_amount,
            )
            for module, _ in parameters_to_prune:
                if hasattr(module, "weight_orig"):
                    prune.remove(module, "weight")
    except Exception:
        pass


# Models where torch.compile(aot_eager) triggers MPS allocator faults
# (malloc "pointer being freed was not allocated").  PointNet's TransformNet
# uses torch.bmm with [B,3,3]/[B,64,64] matrices against [B,*,1024] inputs;
# AOT-eager tracing of that pattern double-frees an MPS buffer during the
# first eval-mode forward, so we keep it in eager mode.
_TORCH_COMPILE_MPS_BLOCKLIST: frozenset[str] = frozenset(
    {
        "pointnet_modelnet40",
        "resnet18_hf_perforated_cifar10",
        "snn_nmnist",
    }
)

# Models that crash with an MPS allocator double-free regardless of
# torch.compile.  SpikingConvNet runs a 10-step BPTT loop through a custom
# autograd Function (SurrogateSpike), and the MPS backend double-frees an
# intermediate buffer near the end of the second epoch.  Forcing CPU
# evaluation/training avoids the fault; the model is small enough (32+64
# conv channels + FC) that CPU is viable.
_MPS_TO_CPU_FALLBACK: frozenset[str] = frozenset({"snn_nmnist"})


def _resolve_device(model_key: str, torch: Any) -> Any:
    device = choose_device()
    if (
        getattr(device, "type", "") == "mps"
        and model_key in _MPS_TO_CPU_FALLBACK
    ):
        print(
            f"[device] {model_key}: forcing CPU (MPS allocator double-free "
            f"observed during training)"
        )
        return torch.device("cpu")
    return device


def _apply_torch_compile(
    model: Any, torch: Any, model_key: str, condition_key: str, device: Any, use_dendrites: bool
) -> Any:
    if use_dendrites or not hasattr(torch, "compile") or getattr(device, "type", "") != "mps":
        return model
    if model_key in _TORCH_COMPILE_MPS_BLOCKLIST:
        print(f"[compile] torch.compile skipped for {model_key}/{condition_key}: known MPS allocator issue")
        return model
    try:
        model = torch.compile(model, backend="aot_eager", fullgraph=False)
        print(f"[compile] torch.compile(aot_eager) applied to {model_key}/{condition_key}")
    except Exception as _compile_exc:
        print(f"[compile] torch.compile skipped for {model_key}/{condition_key}: {_compile_exc}")
    return model


# Marker key written into a param group so _apply_lr_schedule can tell the
# dendrite group apart from the backbone group. PyTorch preserves unknown keys
# in param_groups verbatim, and PAI hands optimArgs straight to the optimizer
# constructor, so the marker survives both construction paths.
PAI_DENDRITE_PARAM_GROUP_KEY = "pai_dendrite_group"


def _is_dendrite_parameter_name(name: str) -> bool:
    """True for parameters PerforatedAI adds when it retains a dendrite.

    Keyed on module layout rather than on ``parameter_type`` because the type
    attribute only distinguishes ``neuron`` from ``ignored`` before a switch --
    it does not single out the newly inserted tensors. The two names below are
    what a retained dendrite actually contributes, read off a perforated
    PointNet checkpoint (``dendrites_fp32/model.pt``):

        conv3.0.dendrites_to_top.0                      (1,024)
        conv3.0.dendrite_module.layers.0.weight       (131,072)
        conv3.0.dendrite_module.layers.0.bias           (1,024)

    ``dendrite_module.parent_module.*`` is deliberately excluded: it is the
    frozen shadow copy of the neuron PAI carries for candidate scoring, typed
    ``ignored``, and it must keep following the backbone schedule.
    """
    if ".dendrite_module." in name and ".parent_module." not in name:
        return True
    return ".dendrites_to_top" in name or name.startswith("dendrites_to_top")


def _split_dendrite_parameters(model: Any) -> tuple[list[Any], list[Any]]:
    """Partition every parameter into (backbone, dendrite-side).

    Deliberately does **not** filter on ``requires_grad``. PAI freezes the
    parent network for the duration of a dendrite phase and unfreezes it
    afterwards; an optimizer rebuilt mid-phase against only the then-trainable
    tensors would drop the whole backbone permanently, and it would never train
    again once PAI unfroze it. Passing frozen parameters to an optimizer is
    harmless -- their grad stays ``None`` and the step skips them -- and it is
    what ``model.parameters()`` did before this split existed.
    """
    backbone: list[Any] = []
    dendrite: list[Any] = []
    for name, parameter in model.named_parameters():
        (dendrite if _is_dendrite_parameter_name(name) else backbone).append(parameter)
    return backbone, dendrite


def _optimizer_param_groups(model: Any, config: TrainingConfig) -> Any:
    """Return ``params`` for the optimizer, split by group when dendrites exist.

    Before any dendrite is retained -- and for every non-dendritic condition --
    the dendrite list is empty and this returns a plain parameter iterable, so
    the optimizer is built exactly as it was before this split existed.
    """
    if not config.use_dendrites:
        return model.parameters()
    backbone, dendrite = _split_dendrite_parameters(model)
    if not dendrite:
        return model.parameters()
    floor = config.learning_rate * config.dendrite_lr_min_factor
    print(
        f"[pai-lr] dendrite parameter group: {len(dendrite)} tensors / "
        f"{sum(p.numel() for p in dendrite)} parameters held at a floor of "
        f"{floor:g} ({len(backbone)} backbone tensors follow the schedule)"
    )
    return [
        {"params": backbone, PAI_DENDRITE_PARAM_GROUP_KEY: False},
        {"params": dendrite, PAI_DENDRITE_PARAM_GROUP_KEY: True},
    ]


def _build_optimizer(model: Any, torch: Any, config: TrainingConfig) -> Any:
    params = _optimizer_param_groups(model, config)
    if config.optimizer_name == "sgd":
        return torch.optim.SGD(
            params,
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            nesterov=config.nesterov and config.momentum > 0.0,
        )
    if config.optimizer_name == "adamw":
        return torch.optim.AdamW(
            params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    return torch.optim.Adam(
        params,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def _optimizer_class(torch: Any, config: TrainingConfig) -> Any:
    if config.optimizer_name == "sgd":
        return torch.optim.SGD
    if config.optimizer_name == "adamw":
        return torch.optim.AdamW
    return torch.optim.Adam


def _optimizer_args(model: Any, config: TrainingConfig) -> dict[str, Any]:
    args: dict[str, Any] = {
        "params": _optimizer_param_groups(model, config),
        "lr": config.learning_rate,
        "weight_decay": config.weight_decay,
    }
    if config.optimizer_name == "sgd":
        args["momentum"] = config.momentum
        args["nesterov"] = config.nesterov and config.momentum > 0.0
    return args


def _pai_tracker() -> Any | None:
    try:
        gpa = importlib.import_module("perforatedai.globals_perforatedai")
    except Exception:
        return None
    tracker = getattr(gpa, "pai_tracker", None)
    if tracker is None or not hasattr(tracker, "add_validation_score"):
        return None
    return tracker


def _pai_module_count(model: Any) -> int | None:
    try:
        upa: Any = importlib.import_module("perforatedai.utils_perforatedai")
        return len(getattr(upa, "get_pai_modules")(model, 0))
    except Exception:
        return None


def _validate_pai_training_model(model: Any) -> None:
    pai_module_count = _pai_module_count(model)
    if pai_module_count is None or pai_module_count > 0:
        return
    raise RuntimeError(
        "PerforatedAI did not create any PAINeuronModule wrappers for this "
        "dendritic run. The model would fail at the first dynamic switch with "
        "'does not have any pai_modules'. Check the PerforatedAI module "
        "registration and ensure layers are registered for perforation, not "
        "track-only wrapping."
    )


def _pai_updates_enabled(config: TrainingConfig) -> bool:
    return bool(
        config.enable_pai_dendrite_updates or config.train_dendrites_until_complete
    )


def _copy_pai_graphs_to_output(pai_save_name: str, output_dir: Path) -> None:
    src = pai_save_path(pai_save_name)
    if not src.exists():
        return
    dst = output_dir / "pai_plots"
    dst.mkdir(parents=True, exist_ok=True)
    for ext in ("*.png", "*.svg", "*.pdf"):
        for f in src.glob(ext):
            shutil.copy2(f, dst / f.name)


def _post_pai_run_config_event(config: TrainingConfig) -> None:
    try:
        gpa: Any = importlib.import_module("perforatedai.globals_perforatedai")
        events_url = getattr(getattr(gpa, "pc"), "events_url", None)
        if not events_url:
            return
        import requests

        total = None if config.train_dendrites_until_complete else config.max_epochs
        requests.post(
            events_url,
            json={"type": "run_config", "total_epochs": total},
            timeout=1.0,
        )
    except Exception:
        pass


@contextmanager
def _pai_pdb_suppressed() -> "Any":
    """Neutralize pdb.set_trace for the duration of a PerforatedAI call.

    PAI drops into pdb for conditions it treats as "come look at this" rather
    than as errors — notably once per parameter it cannot classify while
    filtering optimizer param groups in p-phase. Under a non-interactive stdin
    that raises BdbQuit, which is an ordinary Exception and so gets swallowed
    by the fallbacks below, silently handing back a non-PAI optimizer.
    """
    import pdb as _pdb

    # Typed as Any so reassigning set_trace below is a plain attribute write
    # rather than a redefinition of the stdlib symbol's declared type.
    pdb_module: Any = _pdb
    original = pdb_module.set_trace

    def _no_set_trace(*, header: str | None = None) -> None:
        _ = header

    setattr(pdb_module, "set_trace", _no_set_trace)
    try:
        yield
    finally:
        setattr(pdb_module, "set_trace", original)


def _warn_pai_optimizer_fallback(exc: BaseException) -> None:
    print(
        "[pai] WARNING: PerforatedAI's setup_optimizer failed "
        f"({type(exc).__name__}: {exc}); falling back to a standard optimizer. "
        "PAI's dendrite step will NOT run, so candidate dendrites train on "
        "nothing until this is resolved."
    )


def _setup_pai_optimizer(
    model: Any,
    torch: Any,
    config: TrainingConfig,
) -> tuple[Any, Any | None]:
    optimizer = _build_optimizer(model, torch, config)
    if (
        not config.use_dendrites
        or config.max_epochs <= 0
        or not _pai_updates_enabled(config)
    ):
        return optimizer, None
    tracker = _pai_tracker()
    if tracker is None:
        return optimizer, None
    _validate_pai_training_model(model)
    try:
        with _pai_pdb_suppressed(), pai_working_directory():
            tracker.set_optimizer(_optimizer_class(torch, config))
            setup_result = tracker.setup_optimizer(
                model, _optimizer_args(model, config), {}
            )
    except TypeError:
        try:
            with _pai_pdb_suppressed(), pai_working_directory():
                setup_result = tracker.setup_optimizer(
                    model, _optimizer_args(model, config)
                )
        except Exception as exc:
            _warn_pai_optimizer_fallback(exc)
            return optimizer, tracker
    except Exception as exc:
        _warn_pai_optimizer_fallback(exc)
        return optimizer, tracker
    if isinstance(setup_result, tuple) and setup_result:
        return setup_result[0], tracker
    if setup_result is not None:
        return setup_result, tracker
    return optimizer, tracker


def _rollout_evaluation(
    model_key: str, model: Any, device: Any, split: str
) -> tuple[float, dict[str, Any]]:
    """Evaluate an on-policy model by stepping the environment, not a split.

    There is no held-out set to read: a policy's quality is what it scores when
    it acts. The validation and test seeds are disjoint so that the return the
    best checkpoint was selected on is not the return reported for it, and
    validation uses fewer episodes because it runs every epoch.

    The returned "loss" is the negated mean return, so the loop's loss column
    keeps pointing the same way it does everywhere else (lower is better) while
    the selection metric stays the return itself.
    """
    validating = split == "val"
    metrics = _evaluate_episodic_return(
        model_key,
        model,
        device,
        episodes=_PPO_VAL_EPISODES if validating else _RL_EVAL_EPISODES,
        seed=_PPO_VAL_SEED if validating else _PPO_TEST_SEED,
    )
    if not metrics:
        # gymnasium or Box2D missing: reported as a flat zero rather than a
        # crash, matching how _evaluate_episodic_return degrades elsewhere.
        return 0.0, {}
    mean_return = float(metrics["episodic_return_mean"])
    metrics["episodic_return"] = mean_return
    return -mean_return, metrics


def _eval_on_loader(
    model: Any,
    model_key: str,
    loader: Any,
    device: Any,
    criterion: Any,
    metric_name: str,
    torch: Any,
    target_offset: float = 0.0,
    target_scale: float = 1.0,
    split: str = "test",
) -> tuple[float, dict[str, Any]]:
    """Run evaluation on a dataloader, return (loss, metrics)."""
    if _is_on_policy(model_key):
        return _rollout_evaluation(model_key, model, device, split)
    running_loss_t = torch.zeros(1, device=device)
    examples = 0
    outputs_list: list[Any] = []
    targets_list: list[Any] = []
    metric_targets_list: list[Any] = []
    with torch.no_grad():
        for batch in loader:
            batch = _move_batch_to_device(batch, device)
            outputs, targets, metric_targets = _forward(model_key, model, batch)
            loss = _compute_loss(model_key, criterion, outputs, targets, model=model)
            batch_examples = _batch_size(targets)
            running_loss_t = running_loss_t + loss.detach() * batch_examples
            examples += batch_examples
            det_outputs, det_targets, det_metric_targets = _detach_metric_payload(
                model_key, outputs, targets, metric_targets
            )
            outputs_list.append(det_outputs)
            targets_list.append(det_targets)
            if det_metric_targets is not None:
                metric_targets_list.append(det_metric_targets)
    loss_val = (running_loss_t / max(1, examples)).item()
    metrics: dict[str, Any] = {}
    if outputs_list:
        metrics = _compute_all_metrics(
            model_key,
            _cat_payload(outputs_list),
            torch.cat(targets_list, dim=0),
            torch.cat(metric_targets_list, dim=0) if metric_targets_list else None,
            metric_name=metric_name,
            target_offset=target_offset,
            target_scale=target_scale,
        )
    return loss_val, metrics


def _configure_dendrite_output_dimensions(
    model: Any, model_key: str, use_dendrites: bool, device: Any
) -> None:
    if not use_dendrites:
        return
    module_dimensions = getattr(model, MODULE_OUTPUT_DIMENSIONS_ATTR, None)
    if not module_dimensions:
        module_dimensions = _FALLBACK_MODULE_OUTPUT_DIMENSIONS.get(model_key, {})
        attach_module_output_dimensions(model, module_dimensions)
    set_module_output_dimensions(model, module_dimensions, device=device)


def _determine_skip_info(
    max_epochs: int,
    bit_width: int | None,
    use_qat: bool,
    quantization_mode: str | None,
) -> tuple[bool, str]:
    training_skipped = max_epochs == 0
    if not training_skipped:
        return False, ""
    if bit_width is not None and bit_width < 32 and not use_qat:
        _quant_desc = f"{bit_width}-bit {quantization_mode or 'int'}"
        skip_reason = (
            f"post-training quantization ({_quant_desc})"
            " — weights are quantized without any gradient updates"
        )
    else:
        skip_reason = "no training epochs configured"
    return True, skip_reason


def _print_skip_banner(
    run_label: str,
    skip_reason: str,
    source_condition_key: str | None,
    condition_key: str,
    bit_width: int | None,
    quantization_mode: str | None,
) -> None:
    _source_info = (
        f"condition '{source_condition_key}'"
        if source_condition_key and source_condition_key != condition_key
        else "the current model state"
    )
    _quant_info = (
        f"{bit_width}-bit {quantization_mode or 'int'} quantization"
        if bit_width is not None and bit_width < 32
        else "no quantization"
    )
    print(
        f"\n{'─' * 64}\n"
        f"[SKIP TRAINING]  {run_label}\n"
        f"  Reason  : {skip_reason.capitalize()}\n"
        f"  Source  : checkpoint loaded from {_source_info}\n"
        f"  Quant.  : {_quant_info} will be applied to the loaded weights\n"
        f"  Next    : proceeding directly to test-set evaluation\n"
        f"{'─' * 64}\n"
    )


def _optimizer_step_requires_retained_graph(optimizer: Any) -> bool:
    step = getattr(optimizer, "step", None)
    step_func = getattr(step, "__func__", None)
    step_module = (
        getattr(step, "__module__", "")
        or getattr(step_func, "__module__", "")
    )
    step_name = (
        getattr(step, "__name__", "")
        or getattr(step_func, "__name__", "")
    )
    return (
        step_module.startswith("perforatedbp.")
        and step_name in {"closure_pai_step", "pai_step"}
    )


def _new_training_batch_accumulator(torch: Any, device: Any) -> TrainingBatchAccumulator:
    return TrainingBatchAccumulator(
        running_loss_t=torch.zeros(1, device=device),
        examples=0,
        outputs=[],
        targets=[],
        metric_targets=[],
    )


def _epoch_limit_label(config: "TrainingConfig", max_epochs: int) -> str:
    if config.train_dendrites_until_complete:
        return "until PAI complete"
    return str(max_epochs)


def _training_batch_progress(
    bundle: Any,
    *,
    run_label: str,
    epoch: int,
    epoch_limit_label: str,
) -> Any:
    return tqdm(
        bundle.train_loader,
        desc=f"{run_label} | epoch {epoch + 1}/{epoch_limit_label}",
        unit="batch",
        leave=False,
        dynamic_ncols=True,
        miniters=max(1, len(bundle.train_loader) // 10),
    )


def _pai_model_has_dendrites(model: Any) -> bool:
    modules = getattr(model, "modules", None)
    if modules is None:
        return False
    for module in modules():
        added = getattr(module, "dendrite_modules_added", None)
        if isinstance(added, int) and added > 0:
            return True
    return False


def _pai_tracker_in_neuron_mode() -> bool:
    """True only while PAI is in an all-neuron ("n") phase.

    In "p" phase the candidate dendrites are being trained, and every
    optimizer.step backpropagates through their graph, so it cannot be torn
    down mid-epoch.  An unreadable tracker is treated as "not neuron mode"
    because leaving the graph on only costs memory, while turning it off at
    the wrong moment crashes the run.
    """
    member_vars = getattr(_pai_tracker(), "member_vars", None)
    if not isinstance(member_vars, dict):
        return False
    return member_vars.get("mode") == "n"


def _candidate_graph_batch_limit(
    config: "TrainingConfig",
    *,
    clear_pai_buffers: bool,
    model: Any,
) -> int | None:
    candidate_graph_batch_limit = config.pai_candidate_graph_batch_limit
    if (
        config.use_dendrites
        and not clear_pai_buffers
        and candidate_graph_batch_limit is not None
        and candidate_graph_batch_limit > 0
        # In p-phase PAI's optimizer.step (closure_pai_step) backwards through
        # the candidate graph. Disabling that graph mid-epoch frees the saved
        # tensors and the next p-step raises "Trying to backward through the
        # graph a second time", so the limit is only safe during an all-neuron
        # correlation phase. dendrite_modules_added stays 0 through the *first*
        # p-phase, so the module check alone does not exclude it — the tracker
        # mode is what actually distinguishes the two phases.
        and _pai_tracker_in_neuron_mode()
        and not _pai_model_has_dendrites(model)
    ):
        return candidate_graph_batch_limit
    return None


def _maybe_disable_candidate_graph_for_batch(
    batch_index: int,
    candidate_graph_batch_limit: int | None,
) -> None:
    if batch_index == candidate_graph_batch_limit:
        configure_pai_candidate_graph(False)


def _move_batch_to_device(batch: Any, device: Any) -> tuple[Any, ...]:
    """Copy a batch to ``device``, asynchronously only when that is actually safe.

    ``non_blocking=True`` only buys overlap when the *source* is page-locked, and
    ``_make_loader`` deliberately sets ``pin_memory=False`` (MPS uses unified
    memory, so pinning is pure overhead there).  From pageable memory the flag
    therefore bought nothing — but on MPS it still returned before the copy had
    landed, so a caller that dropped its reference to the CPU tensor let the
    allocator recycle those bytes underneath the in-flight copy.  ``_eval_on_loader``
    did exactly that (``batch = tuple(item.to(...) for item in batch)`` rebinds the
    only reference), which fed PointNet's eval passes partly-garbage point clouds:
    val accuracy pinned near chance for all 60 epochs while train accuracy climbed
    to 78%, test_loss came out at 1.9e33, and two identical eval runs disagreed
    (5.47% vs 5.02%) — the non-determinism that gives a memory race away.
    Gating on ``is_pinned()`` keeps the async path for any future pinned loader
    while making the pageable case a plain synchronous copy.
    """
    return tuple(item.to(device, non_blocking=_is_pinned(item)) for item in batch)


def _is_pinned(item: Any) -> bool:
    """True only for page-locked CPU tensors, where async H2D copies are safe."""
    if getattr(item, "device", None) is None or item.device.type != "cpu":
        return False
    try:
        return bool(item.is_pinned())
    except (AttributeError, RuntimeError):
        return False


def _backward_and_step(
    loss: Any,
    optimizer: Any,
    *,
    retain_graph_for_optimizer_step: bool,
    model: Any = None,
    grad_clip_norm: float | None = None,
    qat_config: "TrainingConfig | None" = None,
) -> None:
    # PerforatedAI's optimizer step may run its own backward pass after the
    # benchmark's loss backward. Standard torch optimizers do not need the
    # graph retained; keeping it on long MPS runs causes per-batch memory growth.
    loss.backward(retain_graph=retain_graph_for_optimizer_step)
    if grad_clip_norm and model is not None:
        # Clipped before the step so PerforatedAI's own step (which may run a
        # second backward) still sees the clipped gradients on the first pass.
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
    if model is not None and qat_config is not None:
        # PQAT fine-tune: the gradient above was computed against the
        # quantized .data written by _qat_project_for_forward. Swap in the
        # full-precision shadow before stepping so the update accumulates
        # there instead of on the discrete quantized value (which the very
        # next projection would just overwrite anyway).
        _qat_restore_shadow_for_step(model, qat_config)
    optimizer.step()
    if model is not None and qat_config is not None:
        _qat_sync_shadow_after_step(model, qat_config)


def _record_training_batch_metrics(
    accumulator: TrainingBatchAccumulator,
    *,
    model_key: str,
    outputs: Any,
    targets: Any,
    metric_targets: Any | None,
    loss: Any,
) -> None:
    batch_examples = _batch_size(targets)
    accumulator.running_loss_t = (
        accumulator.running_loss_t + loss.detach() * batch_examples
    )
    accumulator.examples += batch_examples
    det_out, det_tgt, det_mt = _detach_metric_payload(
        model_key, outputs, targets, metric_targets
    )
    accumulator.outputs.append(det_out)
    accumulator.targets.append(det_tgt)
    if det_mt is not None:
        accumulator.metric_targets.append(det_mt)


def _memory_cleanup_due(batch_index: int, config: "TrainingConfig") -> bool:
    interval = config.memory_cleanup_interval_batches
    return interval is not None and interval > 0 and (batch_index + 1) % interval == 0


def _run_periodic_training_memory_cleanup(
    *,
    model: Any,
    torch: Any,
    config: "TrainingConfig",
) -> None:
    if config.use_dendrites:
        clear_pai_processor_buffers(model)
    _release_accelerator_cache(torch)


def _run_training_batch(
    *,
    model: Any,
    model_key: str,
    batch: Any,
    device: Any,
    criterion: Any,
    optimizer: Any,
    config: "TrainingConfig",
    clear_pai_buffers: bool,
    retain_graph_for_optimizer_step: bool,
) -> tuple[Any, Any, Any, Any | None]:
    batch = _move_batch_to_device(batch, device)
    optimizer.zero_grad(set_to_none=True)
    if clear_pai_buffers:
        clear_pai_processor_buffers(model)
    # PQAT fine-tune: project the full-precision shadow onto the quantization
    # grid so forward/backward see the same weights that will actually ship.
    # No-op for every other condition (_should_quantize_for_training is False).
    _qat_project_for_forward(model, config)
    outputs, targets, metric_targets = _forward(model_key, model, batch)
    loss = _compute_loss(model_key, criterion, outputs, targets, model=model)
    _backward_and_step(
        loss,
        optimizer,
        retain_graph_for_optimizer_step=retain_graph_for_optimizer_step,
        model=model,
        grad_clip_norm=config.grad_clip_norm,
        qat_config=config if _should_quantize_for_training(config) else None,
    )
    # Leave .data on the quantization grid: matches what mid-epoch validation
    # and the next batch's forward pass should see. Cheap and idempotent when
    # not PQAT fine-tuning (_qat_project_for_forward no-ops).
    _qat_project_for_forward(model, config)
    if clear_pai_buffers:
        clear_pai_processor_buffers(model)
    return outputs, targets, metric_targets, loss


def _finalize_training_batch_metrics(
    accumulator: TrainingBatchAccumulator,
    *,
    model_key: str,
    torch: Any,
    metric_name: str,
    target_offset: float = 0.0,
    target_scale: float = 1.0,
) -> tuple[float, dict[str, Any]]:
    train_loss = (
        accumulator.running_loss_t / max(1, accumulator.examples)
    ).item()
    if not accumulator.outputs:
        return train_loss, {}
    train_metrics = _compute_all_metrics(
        model_key,
        _cat_payload(accumulator.outputs),
        torch.cat(accumulator.targets, dim=0),
        torch.cat(accumulator.metric_targets, dim=0)
        if accumulator.metric_targets
        else None,
        metric_name=metric_name,
        target_offset=target_offset,
        target_scale=target_scale,
    )
    return train_loss, train_metrics


def _refresh_on_policy_batches(
    *,
    model: Any,
    model_key: str,
    bundle: Any,
    device: Any,
    config: "TrainingConfig",
    candidate_graph_active: bool,
) -> dict[str, float]:
    """Replace an on-policy model's training loader with a fresh rollout.

    Off-policy models read the same cached tensors every epoch. PPO's training
    data is a function of the weights it is about to update, so the buffer has
    to be regenerated here, before the epoch's progress bar is built over
    ``bundle.train_loader``.

    Collection runs ``n_steps`` single-observation forward passes. Those are
    data generation, not training, so PerforatedAI's candidate graph is turned
    off around them and its processor buffers are cleared afterwards: left on,
    the dendrite candidates would be correlated against batch-of-one
    activations that no optimizer step ever backpropagates through.

    Returns the rollout's own statistics — including ``episodic_return``, the
    epoch's train-side selection number — for merging into the train metrics.
    """
    source = getattr(bundle, "on_policy", None)
    if source is None or not _is_on_policy(model_key):
        return {}
    if candidate_graph_active:
        configure_pai_candidate_graph(False)
    try:
        loader, stats = source.collect(model, device)
    finally:
        if config.use_dendrites:
            clear_pai_processor_buffers(model)
        if candidate_graph_active:
            configure_pai_candidate_graph(True)
    bundle.train_loader = loader
    return stats


def _run_epoch_batches(
    model: Any,
    model_key: str,
    bundle: Any,
    device: Any,
    criterion: Any,
    optimizer: Any,
    torch: Any,
    epoch: int,
    max_epochs: int,
    run_label: str,
    config: "TrainingConfig",
    metric_name: str,
    clear_pai_buffers: bool = False,
) -> tuple[float, dict[str, Any]]:
    model.train()
    if clear_pai_buffers:
        clear_pai_processor_buffers(model)
    _run_memory_guard_cleanup_if_needed(
        model=model,
        torch=torch,
        config=config,
        location=f"{run_label} epoch {epoch + 1} start",
    )
    rollout_metrics = _refresh_on_policy_batches(
        model=model,
        model_key=model_key,
        bundle=bundle,
        device=device,
        config=config,
        candidate_graph_active=config.use_dendrites and not clear_pai_buffers,
    )
    accumulator = _new_training_batch_accumulator(torch, device)
    batch_progress = _training_batch_progress(
        bundle,
        run_label=run_label,
        epoch=epoch,
        epoch_limit_label=_epoch_limit_label(config, max_epochs),
    )
    retain_graph_for_optimizer_step = _optimizer_step_requires_retained_graph(
        optimizer
    )
    candidate_graph_batch_limit = _candidate_graph_batch_limit(
        config, clear_pai_buffers=clear_pai_buffers, model=model
    )
    for batch_index, batch in enumerate(batch_progress):
        _maybe_disable_candidate_graph_for_batch(
            batch_index, candidate_graph_batch_limit
        )
        outputs, targets, metric_targets, loss = _run_training_batch(
            model=model,
            model_key=model_key,
            batch=batch,
            device=device,
            criterion=criterion,
            optimizer=optimizer,
            config=config,
            clear_pai_buffers=clear_pai_buffers,
            retain_graph_for_optimizer_step=retain_graph_for_optimizer_step,
        )
        _record_training_batch_metrics(
            accumulator,
            model_key=model_key,
            outputs=outputs,
            targets=targets,
            metric_targets=metric_targets,
            loss=loss,
        )
        del outputs, targets, metric_targets, loss
        guard_cleaned = False
        if _memory_guard_check_due(batch_index):
            guard_cleaned = _run_memory_guard_cleanup_if_needed(
                model=model,
                torch=torch,
                config=config,
                location=(
                    f"{run_label} epoch {epoch + 1} "
                    f"batch {batch_index + 1}"
                ),
            )
        if _memory_cleanup_due(batch_index, config) and not guard_cleaned:
            _run_periodic_training_memory_cleanup(
                model=model,
                torch=torch,
                config=config,
            )
    batch_progress.close()
    train_loss, train_metrics = _finalize_training_batch_metrics(
        accumulator,
        model_key=model_key,
        torch=torch,
        metric_name=metric_name,
        target_offset=getattr(bundle, "target_offset", 0.0),
        target_scale=getattr(bundle, "target_scale", 1.0),
    )
    # The rollout's numbers describe the same epoch as the surrogate-loss
    # diagnostics computed above, and are the only ones expressed in the
    # environment's own units.
    train_metrics.update(rollout_metrics)
    return train_loss, train_metrics


def _release_accelerator_cache(torch: Any, *, collect_python: bool = True) -> None:
    if collect_python:
        gc.collect()
    mps = getattr(torch, "mps", None)
    if mps is not None and torch.backends.mps.is_available():
        empty_cache = getattr(mps, "empty_cache", None)
        if empty_cache is not None:
            empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _process_rss_bytes_from_proc() -> int | None:
    statm_path = Path("/proc/self/statm")
    if not statm_path.exists():
        return None
    try:
        fields = statm_path.read_text().split()
        resident_pages = int(fields[1])
        return resident_pages * int(os.sysconf("SC_PAGE_SIZE"))
    except (IndexError, OSError, ValueError):
        return None


def _process_rss_bytes_from_ps() -> int | None:
    try:
        result = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(os.getpid())],
            capture_output=True,
            check=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    lines = result.stdout.strip().splitlines()
    if not lines:
        return None
    try:
        return int(lines[0].strip()) * 1024
    except ValueError:
        return None


def _process_resident_memory_bytes() -> int | None:
    return _process_rss_bytes_from_proc() or _process_rss_bytes_from_ps()


def _safe_backend_available(backend: Any) -> bool:
    try:
        return bool(backend is not None and backend.is_available())
    except Exception:
        return False


def _safe_memory_method_reading(target: Any, method_name: str) -> int | None:
    method = getattr(target, method_name, None)
    if method is None:
        return None
    try:
        return int(method())
    except Exception:
        return None


def _memory_readings_from_methods(
    target: Any, method_names: tuple[str, ...]
) -> list[int]:
    return [
        reading
        for method_name in method_names
        if (reading := _safe_memory_method_reading(target, method_name)) is not None
    ]


def _cuda_memory_readings(torch: Any) -> list[int]:
    cuda = getattr(torch, "cuda", None)
    if not _safe_backend_available(cuda):
        return []
    return _memory_readings_from_methods(
        cuda, ("memory_reserved", "memory_allocated")
    )


def _mps_memory_readings(torch: Any) -> list[int]:
    mps = getattr(torch, "mps", None)
    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is None or not _safe_backend_available(mps_backend):
        return []
    return _memory_readings_from_methods(
        mps, ("driver_allocated_memory", "current_allocated_memory")
    )


def _max_memory_reading(readings: list[int]) -> int | None:
    return max(readings) if readings else None


def _torch_accelerator_memory_bytes(torch: Any) -> int | None:
    return _max_memory_reading([
        *_cuda_memory_readings(torch),
        *_mps_memory_readings(torch),
    ])


def _current_memory_usage_bytes(torch: Any) -> int | None:
    readings = [
        value
        for value in (
            _process_resident_memory_bytes(),
            _torch_accelerator_memory_bytes(torch),
        )
        if value is not None
    ]
    if not readings:
        return None
    # MPS uses unified memory, so use the highest observed source instead of
    # adding process RSS and driver allocation together.
    return max(readings)


def _format_memory_usage(memory_bytes: int) -> str:
    return f"{memory_bytes / (1024**3):.2f} GB"


def _memory_guard_check_due(batch_index: int) -> bool:
    return (batch_index + 1) % _MEMORY_GUARD_CHECK_INTERVAL_BATCHES == 0


def _run_memory_guard_cleanup_if_needed(
    *,
    model: Any,
    torch: Any,
    config: TrainingConfig,
    location: str,
) -> bool:
    usage_bytes = _current_memory_usage_bytes(torch)
    if usage_bytes is None or usage_bytes <= _MEMORY_GUARD_THRESHOLD_BYTES:
        return False
    if config.use_dendrites:
        clear_pai_processor_buffers(model)
    _release_accelerator_cache(torch)
    after_bytes = _current_memory_usage_bytes(torch)
    message = (
        f"[memory] {location}: usage reached "
        f"{_format_memory_usage(usage_bytes)}; cleared caches after crossing "
        f"{_format_memory_usage(_MEMORY_GUARD_THRESHOLD_BYTES)}."
    )
    if after_bytes is not None:
        message += f" Current usage: {_format_memory_usage(after_bytes)}."
    print(message)
    return True


def _initial_epoch_state(metric_direction: str) -> EpochTrainingState:
    best_metric = -math.inf if metric_direction == "maximize" else math.inf
    return EpochTrainingState([], best_metric, 0, None)


def _save_epoch_checkpoint(
    output_dir: "Path",
    epoch: int,
    state: "EpochTrainingState",
    optimizer: Any,
    model: Any,
    torch: Any,
) -> None:
    ckpt = {
        "epoch": epoch,
        "model_state_dict": {
            k: v.detach().cpu().clone()
            for k, v in _unwrap_compiled(model).state_dict().items()
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "history": state.history,
        "best_metric": state.best_metric,
        "best_epoch": state.best_epoch,
        "best_state": state.best_state,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, output_dir / _EPOCH_CHECKPOINT_PT)


def _load_epoch_checkpoint(
    output_dir: "Path",
    torch: Any,
) -> "dict | None":
    path = output_dir / _EPOCH_CHECKPOINT_PT
    if not path.exists():
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise RuntimeError(
            f"epoch checkpoint is unreadable ({exc}); use --fresh rather than "
            "continuing inside an unknown artifact namespace"
        ) from exc


def _apply_epoch_checkpoint(
    ckpt: dict,
    state: "EpochTrainingState",
    model: Any,
    optimizer: Any,
) -> int:
    resume_epoch = int(ckpt["epoch"]) + 1
    if not _load_compatible_best_state(model, ckpt["model_state_dict"]):
        raise RuntimeError(
            "epoch checkpoint topology does not match the restored model; "
            "refusing to resume partially. Use --fresh to start a new attempt."
        )
    try:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    except Exception as exc:
        raise RuntimeError(
            "epoch checkpoint optimizer state is incompatible; refusing to "
            "resume with a reset optimizer. Use --fresh to start a new attempt."
        ) from exc
    state.history = list(ckpt.get("history", []))
    state.best_metric = ckpt["best_metric"]
    state.best_epoch = int(ckpt["best_epoch"])
    state.best_state = ckpt.get("best_state")
    print(
        f"[checkpoint] resuming from epoch {resume_epoch} "
        f"(last completed: {ckpt['epoch'] + 1}, "
        f"best so far: epoch {state.best_epoch}, metric {state.best_metric:.4f})"
    )
    return resume_epoch


def _persistable_pai_save_name(context: "EpochTrainingContext") -> str | None:
    """The save name for this run's PAI state, or None if there's none worth keeping.

    Returns the name rather than a bool so callers get a genuinely non-optional
    ``str`` to hand to the ``*_pai_system`` helpers; a bool predicate left them
    passing ``config.pai_save_name`` (``str | None``) on a branch only a human
    could see was already guarded.
    """
    if not (
        context.config.use_dendrites
        and _pai_updates_enabled(context.config)
        and _pai_tracker() is not None
    ):
        return None
    return context.config.pai_save_name or None


def _save_pai_resume_state(context: "EpochTrainingContext") -> None:
    """Snapshot PAI's dendrite structure and schedule beside the epoch checkpoint."""
    save_name = _persistable_pai_save_name(context)
    if save_name is None:
        return
    save_pai_system(_unwrap_compiled(context.model), save_name)


def _restore_pai_resume_state(
    context: "EpochTrainingContext",
    optimizer: Any,
) -> Any:
    """Rebuild the saved dendrite structure and schedule before resuming.

    Returns an optimizer matching the restored model.  The one handed in was
    built against the freshly perforated (dendrite-free) network, so its
    parameter groups stop lining up as soon as dendrites come back — that
    mismatch is what silently reset the optimizer on every earlier resume.
    """
    save_name = _persistable_pai_save_name(context)
    if save_name is None:
        return optimizer
    if not pai_resume_state_exists(save_name):
        return optimizer
    module_dimensions = getattr(context.model, MODULE_OUTPUT_DIMENSIONS_ATTR, None)
    restored = load_pai_system(_unwrap_compiled(context.model), save_name)
    if restored is None:
        return optimizer
    attach_module_output_dimensions(restored, module_dimensions)
    context.model = restored.to(context.device)
    _configure_dendrite_output_dimensions(
        context.model,
        context.model_key,
        context.config.use_dendrites,
        context.device,
    )
    clear_pai_processor_buffers(context.model)
    optimizer, _ = _setup_pai_optimizer(context.model, context.torch, context.config)
    member_vars = getattr(_pai_tracker(), "member_vars", None) or {}
    print(
        "[pai-state] restored PAI schedule: "
        f"{member_vars.get('num_dendrites_added', '?')} dendrite(s), "
        f"cycle {member_vars.get('num_cycles', '?')}, "
        f"mode {member_vars.get('mode', '?')}, "
        f"PAI epoch {member_vars.get('num_epochs_run', '?')}."
    )
    return optimizer


def _build_history_row(
    *,
    epoch: int,
    epoch_start: float,
    optimizer: Any,
    train_loss: float,
    train_metrics: dict[str, Any],
    val_loss: float,
    val_metrics: dict[str, Any],
    context: EpochTrainingContext,
) -> dict[str, Any]:
    primary_metric_key = context.primary_metric_key
    val_metric = float(val_metrics.get(primary_metric_key, 0.0))
    backbone_group = next(
        (
            group
            for group in optimizer.param_groups
            if not group.get(PAI_DENDRITE_PARAM_GROUP_KEY, False)
        ),
        optimizer.param_groups[0],
    )
    dendrite_group = next(
        (
            group
            for group in optimizer.param_groups
            if group.get(PAI_DENDRITE_PARAM_GROUP_KEY, False)
        ),
        None,
    )
    history_row: dict[str, Any] = {
        "epoch": epoch + 1,
        "primary_metric_name": context.metric_name,
        "primary_metric_key": primary_metric_key,
        "metric_direction": context.metric_direction,
        "learning_rate": float(backbone_group["lr"]),
        "backbone_learning_rate": float(backbone_group["lr"]),
        "dendrite_learning_rate": (
            float(dendrite_group["lr"]) if dendrite_group is not None else None
        ),
        "epoch_seconds": time.perf_counter() - epoch_start,
        "train_loss": train_loss,
        "train_primary_metric": float(train_metrics.get(primary_metric_key, 0.0)),
        "val_loss": val_loss,
        "val_primary_metric": val_metric,
        "val_metric": val_metric,
    }
    history_row.update(_prefix_metrics("train", train_metrics))
    history_row.update(_prefix_metrics("val", val_metrics))
    return history_row


def _record_best_epoch(
    state: EpochTrainingState,
    model: Any,
    epoch: int,
    val_metric: float,
    metric_direction: str,
) -> None:
    is_first_best = state.best_state is None
    if not is_first_best and not _metric_is_better(
        val_metric, state.best_metric, metric_direction
    ):
        return
    state.best_metric = val_metric
    state.best_epoch = epoch + 1
    state.best_state = {
        k: v.detach().cpu().clone()
        for k, v in _unwrap_compiled(model).state_dict().items()
    }


def _navigate_to_module_attr(model: Any, key: str) -> tuple[Any, str] | None:
    parts = key.split(".")
    target = model
    for part in parts[:-1]:
        try:
            target = getattr(target, part)
        except AttributeError:
            if not part.isdigit():
                return None
            try:
                target = target[int(part)]
            except (IndexError, KeyError, TypeError):
                return None
    return target, parts[-1]


def _is_pai_lazy_dendrite_buffer_key(key: str) -> bool:
    return ".dendrite_module." in key or ".dendrite_values." in key


def _adopt_missing_pai_dendrite_buffers(
    model: Any, best_state: dict[str, Any]
) -> list[str]:
    # PAI registers some DendriteValueTracker buffers (`shape`, score history,
    # etc.) lazily inside create_new_dendrite_module rather than at perforation.
    # A checkpoint saved after the n→p switch carries them, but the freshly
    # perforated model on resume does not, so a plain load_state_dict silently
    # drops the values. The `initialized` flag IS present in both and gets
    # set to True, which makes PAI skip its first-forward re-init and later
    # AttributeError on `.shape`. Register the missing buffers on the parent
    # modules so the imminent load_state_dict can populate them.
    plain_model = _unwrap_compiled(model)
    current_keys = set(plain_model.state_dict().keys())
    adopted: list[str] = []
    for key, value in best_state.items():
        if key in current_keys:
            continue
        if not _is_pai_lazy_dendrite_buffer_key(key):
            continue
        if not hasattr(value, "clone"):
            continue
        nav = _navigate_to_module_attr(plain_model, key)
        if nav is None:
            continue
        parent, leaf = nav
        register_buffer = getattr(parent, "register_buffer", None)
        if register_buffer is None:
            continue
        try:
            register_buffer(leaf, value.detach().clone())
            adopted.append(key)
        except Exception:
            continue
    return adopted


def _load_compatible_best_state(model: Any, best_state: dict[str, Any]) -> bool:
    """Restore ``best_state`` onto ``model`` if, and only if, doing so would not
    silently paper over a dendrite-structure change since the best epoch.

    A plateau-triggered dendrite switch can add a dendrite *after* the best
    validation epoch was recorded, without ever beating that score again before
    training ends — normal, expected behaviour. ``best_state`` (a value-only
    snapshot taken at the best epoch) then has no tensor, or a differently
    shaped one, for the new dendrite's parameters. The previous behaviour here
    was to silently skip just those tensors and load everything else: the
    "restored best model" ended up a hybrid of best-epoch values (for anything
    shape-compatible) and leftover post-best-epoch training values (for
    anything that wasn't) — a state that was never actually validated at any
    epoch, matching neither ``best_metric`` nor the true final-epoch metric.
    See information/MEASUREMENT_CAVEATS.md #3 for the measured symptom (a test
    metric that regresses even though the validation curve improved).

    Returns True if the restore was applied (``model`` now holds the
    best-epoch weights). Returns False, leaving ``model`` completely
    untouched, if the structures diverge — the caller should then evaluate and
    persist the live, final-epoch model instead, which is self-consistent by
    construction.
    """
    plain_model = _unwrap_compiled(model)
    report = inspect_state_dict(
        plain_model.state_dict(),
        best_state,
        allowed_unexpected=_is_pai_lazy_dendrite_buffer_key,
    )
    if not report.complete:
        print(
            "[state] best-epoch structure does not match the final trained "
            "structure (a dendrite was likely added after the best epoch) -- "
            "keeping the final model instead of a partial restore. "
            + report.summary(limit=5)
        )
        return False
    adopted = _adopt_missing_pai_dendrite_buffers(plain_model, best_state)
    load_state_dict_checked(plain_model, best_state, context="best-epoch checkpoint")
    if adopted:
        print(
            "[state] adopted lazy PAI dendrite buffers: "
            + ", ".join(adopted[:5])
            + ("..." if len(adopted) > 5 else "")
        )
    return True


def _pai_dendrite_phase_epochs(pai_tracker: Any) -> int | None:
    """Epochs PAI has spent in the current dendrite phase, or None if not in one."""
    member_vars = getattr(pai_tracker, "member_vars", None)
    if not isinstance(member_vars, dict) or member_vars.get("mode") != "p":
        return None
    switch_epochs = member_vars.get("switch_epochs")
    num_epochs_run = member_vars.get("num_epochs_run")
    if not switch_epochs or not isinstance(num_epochs_run, int):
        return None
    try:
        return num_epochs_run - int(switch_epochs[-1])
    except (TypeError, ValueError):
        return None


def _pai_dendrite_phase_stalled(
    context: EpochTrainingContext, pai_tracker: Any
) -> bool:
    """True when the dendrite phase has overrun its ceiling and must be cut short."""
    limit = context.config.max_dendrite_phase_epochs
    if limit <= 0:
        return False
    phase_epochs = _pai_dendrite_phase_epochs(pai_tracker)
    if phase_epochs is None or phase_epochs < limit:
        return False
    print(
        f"[pai] {context.run_label}: dendrite phase has run {phase_epochs} epochs "
        f"(limit {limit}) without PAI electing to switch. Forcing the switch so "
        "the candidate dendrites are evaluated instead of training on noise."
    )
    return True


def _run_dynamic_dendrite_update(
    *,
    context: EpochTrainingContext,
    optimizer: Any,
    pai_tracker: Any,
    val_metric: float,
    force_switch: bool = False,
) -> tuple[Any, Any | None, bool, bool]:
    import pdb as _pdb
    from typing import Any, Callable

    def _no_set_trace(*, header: str | None = None) -> None:
        _ = header

    pdb_module: Any = _pdb
    _orig_set_trace: Callable[..., None] = pdb_module.set_trace
    setattr(pdb_module, "set_trace", _no_set_trace)
    try:
        module_dimensions = getattr(
            context.model, MODULE_OUTPUT_DIMENSIONS_ATTR, None
        )
        with pai_working_directory():
            # The unforced path passes no third argument at all, so PAI builds
            # without force_switch behave exactly as before; the forced path
            # passes it positionally so it does not depend on the parameter name.
            model, restructured, training_complete = (
                pai_tracker.add_validation_score(val_metric, context.model, True)
                if force_switch
                else pai_tracker.add_validation_score(val_metric, context.model)
            )
        attach_module_output_dimensions(model, module_dimensions)
        context.model = model.to(context.device)
        _configure_dendrite_output_dimensions(
            context.model,
            context.model_key,
            context.config.use_dendrites,
            context.device,
        )
        if restructured:
            optimizer, _ = _setup_pai_optimizer(context.model, context.torch, context.config)
        return optimizer, pai_tracker, bool(restructured), bool(training_complete)
    except SystemExit as pai_exit:
        if pai_exit.code != -1:
            raise
        raise RuntimeError(
            "PerforatedAI aborted during dynamic dendrite insertion "
            "(SystemExit -1). This usually means the model handed to "
            "add_validation_score is not a valid perforated model for the "
            "current PAI state. The dendritic run is invalid, so training has "
            "been stopped instead of continuing forever."
        ) from pai_exit
    except Exception as pai_exc:
        raise RuntimeError(
            "PerforatedAI failed during dynamic dendrite insertion. The "
            "dendritic run is invalid, so training has been stopped instead "
            "of continuing without PAI."
        ) from pai_exc
    finally:
        setattr(pdb_module, "set_trace", _orig_set_trace)


def _dendrite_freeze_start_epoch(max_epochs: int, freeze_fraction: float) -> int | None:
    if max_epochs <= 1 or freeze_fraction <= 0:
        return None
    freeze_epochs = max(1, min(max_epochs - 1, math.ceil(max_epochs * freeze_fraction)))
    return max_epochs - freeze_epochs


def _pai_updates_frozen(
    context: EpochTrainingContext,
    epoch: int,
    *,
    run_until_pai_complete: bool,
) -> bool:
    if run_until_pai_complete or not context.config.enable_pai_dendrite_updates:
        return False
    freeze_start = _dendrite_freeze_start_epoch(
        context.max_epochs, context.config.freeze_dendrite_updates_fraction
    )
    return freeze_start is not None and epoch >= freeze_start


def _copy_optimizer_learning_rates(source: Any, target: Any) -> None:
    source_groups = getattr(source, "param_groups", None)
    target_groups = getattr(target, "param_groups", None)
    if not source_groups or not target_groups:
        return
    if len(source_groups) == len(target_groups):
        for source_group, target_group in zip(source_groups, target_groups):
            if "lr" in source_group:
                target_group["lr"] = source_group["lr"]
        return
    source_lr = source_groups[0].get("lr")
    if source_lr is None:
        return
    for target_group in target_groups:
        target_group["lr"] = source_lr


def _freeze_pai_live_updates(
    context: EpochTrainingContext,
    optimizer: Any,
) -> Any:
    _set_pai_candidate_graph_for_context(context, False)
    clear_pai_processor_buffers(context.model)
    clear_pai_tracker_state()
    _release_accelerator_cache(context.torch)
    standard_optimizer = _build_optimizer(context.model, context.torch, context.config)
    _copy_optimizer_learning_rates(optimizer, standard_optimizer)
    print(
        "[pai] live dendrite updates frozen; continuing with a standard optimizer."
    )
    return standard_optimizer


def _set_epoch_progress(
    epoch_progress: Any,
    metric_name: str,
    val_metric: float,
    best_metric: float,
    best_epoch: int,
) -> None:
    metric_key = _metric_display_key(metric_name)
    epoch_progress.set_postfix(
        **{
            f"val_{metric_key}": _format_metric_value(val_metric),
            f"best_{metric_key}": _format_metric_value(best_metric),
        },
        best_epoch=best_epoch,
    )


def _epoch_progress(
    context: EpochTrainingContext,
    run_until_pai_complete: bool,
    start_epoch: int = 0,
) -> Any:
    epoch_iterable = (
        itertools.count(start_epoch)
        if run_until_pai_complete
        else range(start_epoch, context.max_epochs)
    )
    remaining = None if run_until_pai_complete else context.max_epochs - start_epoch
    return tqdm(
        epoch_iterable,
        total=remaining,
        desc=context.run_label,
        unit="epoch",
        leave=True,
        dynamic_ncols=True,
    )


def _pai_update_status(
    context: EpochTrainingContext,
    epoch: int,
    pai_tracker: Any | None,
    run_until_pai_complete: bool,
) -> PAIUpdateStatus:
    frozen = _pai_updates_frozen(
        context, epoch, run_until_pai_complete=run_until_pai_complete
    )
    return PAIUpdateStatus(frozen=frozen, active=bool(pai_tracker and not frozen))


def _set_pai_candidate_graph_for_context(
    context: EpochTrainingContext, enabled: bool
) -> None:
    if context.config.use_dendrites:
        configure_pai_candidate_graph(enabled)


def _clear_pai_buffers_when_inactive(
    context: EpochTrainingContext, pai_updates_active: bool
) -> None:
    if context.config.use_dendrites and not pai_updates_active:
        clear_pai_processor_buffers(context.model)


def _run_training_pass(
    context: EpochTrainingContext,
    optimizer: Any,
    epoch: int,
    pai_status: PAIUpdateStatus,
) -> tuple[float, dict[str, Any]]:
    _set_pai_candidate_graph_for_context(context, pai_status.active)
    try:
        return _run_epoch_batches(
            context.model, context.model_key, context.bundle, context.device,
            context.criterion, optimizer, context.torch, epoch, context.max_epochs,
            context.run_label, context.config, context.metric_name,
            clear_pai_buffers=context.config.use_dendrites and not pai_status.active,
        )
    finally:
        _set_pai_candidate_graph_for_context(context, False)
        _clear_pai_buffers_when_inactive(context, pai_status.active)


def _run_validation_pass(
    context: EpochTrainingContext,
) -> tuple[float, dict[str, Any]]:
    context.model.eval()
    return _eval_on_loader(
        context.model, context.model_key, context.bundle.val_loader,
        context.device, context.criterion, context.metric_name, context.torch,
        target_offset=getattr(context.bundle, "target_offset", 0.0),
        target_scale=getattr(context.bundle, "target_scale", 1.0),
        split="val",
    )


def _record_epoch_result(
    *,
    state: EpochTrainingState,
    context: EpochTrainingContext,
    optimizer: Any,
    epoch: int,
    epoch_start: float,
    train_loss: float,
    train_metrics: dict[str, Any],
    val_loss: float,
    val_metrics: dict[str, Any],
    pai_status: PAIUpdateStatus,
) -> tuple[dict[str, Any], float]:
    history_row = _build_history_row(
        epoch=epoch, epoch_start=epoch_start, optimizer=optimizer,
        train_loss=train_loss, train_metrics=train_metrics,
        val_loss=val_loss, val_metrics=val_metrics, context=context,
    )
    state.history.append(history_row)
    val_metric = float(history_row["val_metric"])
    _record_best_epoch(
        state, context.model, epoch, val_metric, context.metric_direction
    )
    history_row["pai_dynamic_insertion_active"] = pai_status.active
    history_row["pai_dendrite_updates_frozen"] = pai_status.frozen
    history_row["pai_restructured"] = False
    history_row["pai_training_complete"] = False
    history_row["pai_dendrite_phase"] = False
    return history_row, val_metric


def _run_active_pai_update(
    *,
    context: EpochTrainingContext,
    optimizer: Any,
    pai_tracker: Any,
    val_metric: float,
    force_switch: bool,
) -> tuple[Any, Any | None, bool, bool]:
    _set_pai_candidate_graph_for_context(context, True)
    try:
        return _run_dynamic_dendrite_update(
            context=context, optimizer=optimizer, pai_tracker=pai_tracker,
            val_metric=val_metric, force_switch=force_switch,
        )
    finally:
        _set_pai_candidate_graph_for_context(context, False)


def _apply_pai_epoch_update(
    *,
    context: EpochTrainingContext,
    optimizer: Any,
    pai_tracker: Any | None,
    history_row: dict[str, Any],
    val_metric: float,
    pai_status: PAIUpdateStatus,
) -> tuple[Any, Any | None, bool]:
    if not pai_status.active:
        return optimizer, pai_tracker, False
    # Read before add_validation_score, so the flag describes the phase this
    # epoch's train and validation passes actually ran under rather than the one
    # the tracker moves to as a result of them.  _training_collapsed keys off it.
    history_row["pai_dendrite_phase"] = (
        _pai_dendrite_phase_epochs(pai_tracker) is not None
    )
    force_switch = _pai_dendrite_phase_stalled(context, pai_tracker)
    optimizer, pai_tracker, restructured, training_complete = _run_active_pai_update(
        context=context, optimizer=optimizer, pai_tracker=pai_tracker,
        val_metric=val_metric, force_switch=force_switch,
    )
    history_row["pai_switch_reason"] = (
        "candidate_phase_timeout"
        if force_switch
        else ("pai_schedule" if restructured else "")
    )
    history_row["pai_restructured"] = restructured
    history_row["pai_training_complete"] = training_complete
    # The model may have been replaced (e.g. best-model import on PAI switch) or
    # restructured, so any buffered tensors referencing the old computation graphs
    # are now stale.  Clear them so the next training epoch starts fresh.
    clear_pai_processor_buffers(context.model)
    _release_accelerator_cache(context.torch)
    if training_complete:
        _set_pai_candidate_graph_for_context(context, False)
        return optimizer, None, True
    return optimizer, pai_tracker, False


def _update_epoch_progress(
    epoch_progress: Any,
    context: EpochTrainingContext,
    state: EpochTrainingState,
    val_metric: float,
) -> None:
    _set_epoch_progress(
        epoch_progress, context.metric_name, val_metric,
        state.best_metric, state.best_epoch,
    )


def _run_training_pass_oom_guarded(
    context: EpochTrainingContext,
    optimizer: Any,
    epoch: int,
    pai_status: "PAIUpdateStatus",
) -> tuple[float, dict[str, Any]]:
    try:
        return _run_training_pass(context, optimizer, epoch, pai_status)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            _release_accelerator_cache(context.torch)
            print(
                f"[oom] {context.run_label} epoch {epoch + 1}: out of memory — "
                "releasing caches. Resume training to continue from the last "
                "completed epoch checkpoint."
            )
        raise


def _training_collapsed(state: EpochTrainingState, metric_direction: str) -> bool:
    """True when validation froze bit-for-bit at a value worse than the best seen.

    An identical float across this many epochs means the model's outputs stopped
    changing at all, which is a different thing from a plateau. The 2026-07-29
    dendrite run diverged DistilBERT on its second epoch and then sat on one
    constant class -- exactly the validation split's majority-class fraction --
    for 39 more, because nothing in the loop was watching for it.

    Requiring the frozen value to be *worse* than ``best_metric`` keeps a
    genuinely converged run, whose frozen value is its best, from tripping this.

    Epochs inside a PAI dendrite ("p") phase are exempt.  During that phase the
    parent network is frozen and the candidate dendrites are not yet wired into
    the output, so identical validation is the *expected* signature, not a dead
    network -- the two are indistinguishable from the metric alone.  Without
    this exemption the guard fired on six of seven models in the 2026-08-28
    run and truncated every one of them mid-phase.  ``max_dendrite_phase_epochs``
    (8) bounds the phase well inside this window, so exempting these epochs
    cannot let a genuinely stuck run train forever.
    """
    if len(state.history) < _COLLAPSE_GUARD_EPOCHS:
        return False
    window = state.history[-_COLLAPSE_GUARD_EPOCHS:]
    if any(row.get("pai_dendrite_phase") for row in window):
        return False
    recent = [float(row["val_metric"]) for row in window]
    frozen_metric = recent[0]
    if any(metric != frozen_metric for metric in recent[1:]):
        return False
    if metric_direction == "maximize":
        return frozen_metric < state.best_metric
    return frozen_metric > state.best_metric


def _scheduled_learning_rate(
    config: "TrainingConfig", epoch: int, max_epochs: int
) -> float | None:
    """Learning rate this schedule prescribes at ``epoch``, or None for no-op.

    Always computed from the base learning rate and the epoch index rather than
    mutated in place, so it stays correct regardless of checkpoint resume (which
    re-enters the loop mid-way) or PerforatedAI recreating the optimizer on
    dendrite restructuring (which resets param groups back to the base lr).

    ``lr_schedule_epochs`` is the recipe's planned horizon when provided.  It
    lets a dynamic dendritic run preserve useful learning rate through its
    candidate phase while still holding at the floor after that horizon rather
    than wrapping a cosine curve back up.
    """
    base = config.learning_rate
    warmup = max(0, config.warmup_epochs)
    schedule_epochs = config.lr_schedule_epochs or max_epochs
    if warmup and epoch < warmup:
        # epoch 0 gets base/warmup rather than 0, which would be a dead epoch.
        return base * float(epoch + 1) / float(warmup)
    if config.lr_schedule == "step":
        lr_decay_every = config.lr_decay_every
        if not lr_decay_every:
            return None
        return base * (config.lr_decay_gamma ** (epoch // lr_decay_every))
    if config.lr_schedule not in {"cosine", "linear"}:
        return None
    floor = base * config.lr_min_factor
    decay_span = max(1, schedule_epochs - warmup)
    progress = min(1.0, max(0.0, float(epoch - warmup) / float(decay_span)))
    if config.lr_schedule == "linear":
        return floor + (base - floor) * (1.0 - progress)
    return floor + 0.5 * (base - floor) * (1.0 + math.cos(math.pi * progress))


def _dendrite_learning_rate(
    config: "TrainingConfig", scheduled_lr: float
) -> float:
    """Return the lr for newly inserted dendrite parameters.

    The backbone schedule is annealed on the assumption that the network it
    trains has been learning since epoch 0. A dendrite inserted at epoch 190 of
    200 has not: it is a freshly initialized module handed whatever the anneal
    has left, which for a cosine run with ``lr_min_factor=0.0`` is exactly zero
    (measured on ResNet-18: 13 of 19 epochs at lr=0.0, val flat within 0.004).
    Holding the dendrite group at a floor keeps it trainable no matter when the
    plateau detector fires, while leaving the backbone on the schedule its
    dense control also runs -- so a dendritic gain stays attributable to the
    dendrite rather than to a warm restart the control never got.
    """
    floor = config.learning_rate * config.dendrite_lr_min_factor
    return max(scheduled_lr, floor)


def _apply_lr_schedule(
    optimizer: Any, config: "TrainingConfig", epoch: int, max_epochs: int
) -> None:
    """Set each param group's lr from the schedule, floored for dendrites."""
    target_lr = _scheduled_learning_rate(config, epoch, max_epochs)
    if target_lr is None:
        return
    for group in optimizer.param_groups:
        group["lr"] = (
            _dendrite_learning_rate(config, target_lr)
            if group.get(PAI_DENDRITE_PARAM_GROUP_KEY, False)
            else target_lr
        )


def _run_training_epochs(
    context: EpochTrainingContext,
    optimizer: Any,
    pai_tracker: Any | None = None,
) -> tuple[list[dict[str, Any]], float, int, dict[str, Any] | None]:
    state = _initial_epoch_state(context.metric_direction)
    run_until_pai_complete = bool(
        context.config.train_dendrites_until_complete and pai_tracker is not None
    )
    start_epoch = 0
    output_dir = context.output_dir
    if output_dir is not None:
        ckpt = _load_epoch_checkpoint(output_dir, context.torch)
        if ckpt is not None:
            # Restore the dendrite structure first so the checkpoint's tensors
            # and optimizer groups have something shaped like them to land in.
            optimizer = _restore_pai_resume_state(context, optimizer)
            start_epoch = _apply_epoch_checkpoint(ckpt, state, context.model, optimizer)
    epoch_progress = _epoch_progress(context, run_until_pai_complete, start_epoch)
    for epoch in epoch_progress:
        epoch_start = time.perf_counter()
        pai_status = _pai_update_status(
            context, epoch, pai_tracker, run_until_pai_complete
        )
        if pai_status.frozen and pai_tracker is not None:
            optimizer = _freeze_pai_live_updates(context, optimizer)
            pai_tracker = None
            _release_accelerator_cache(context.torch)
            pai_status = PAIUpdateStatus(frozen=True, active=False)
        _apply_lr_schedule(optimizer, context.config, epoch, context.max_epochs)
        train_loss, train_metrics = _run_training_pass_oom_guarded(
            context, optimizer, epoch, pai_status
        )
        val_loss, val_metrics = _run_validation_pass(context)
        history_row, val_metric = _record_epoch_result(
            state=state, context=context, optimizer=optimizer, epoch=epoch,
            epoch_start=epoch_start, train_loss=train_loss,
            train_metrics=train_metrics, val_loss=val_loss,
            val_metrics=val_metrics, pai_status=pai_status,
        )
        optimizer, pai_tracker, pai_training_complete = _apply_pai_epoch_update(
            context=context, optimizer=optimizer, pai_tracker=pai_tracker,
            history_row=history_row, val_metric=val_metric, pai_status=pai_status,
        )
        _run_memory_guard_cleanup_if_needed(
            model=context.model,
            torch=context.torch,
            config=context.config,
            location=f"{context.run_label} epoch {epoch + 1} end",
        )
        if output_dir is not None:
            # Written before the epoch checkpoint so the PAI snapshot's
            # tracker_string buffer is already on the model when its
            # state_dict is captured, keeping the two files consistent.
            _save_pai_resume_state(context)
            _save_epoch_checkpoint(
                output_dir, epoch, state, optimizer, context.model, context.torch
            )
        _update_epoch_progress(epoch_progress, context, state, val_metric)
        if _training_collapsed(state, context.metric_direction):
            history_row["training_termination_reason"] = "validation_collapse"
            print(
                f"[collapse] {context.run_label}: validation "
                f"{context.metric_name} frozen at {val_metric:.6f} for "
                f"{_COLLAPSE_GUARD_EPOCHS} epochs, worse than the best "
                # best_epoch is already 1-indexed (_record_best_epoch stores
                # epoch + 1), so no further offset here.
                f"{state.best_metric:.6f} from epoch {state.best_epoch} — "
                "stopping this condition rather than training a dead network."
            )
            break
        # In dynamic dendritic mode PAI owns the stopping decision. Its
        # completion signal is emitted only once, immediately before the
        # tracker is cleared, so stop on this epoch rather than continuing an
        # open-ended iterator without a live PAI schedule.
        if pai_training_complete and run_until_pai_complete:
            history_row["training_termination_reason"] = "pai_training_complete"
            break
    epoch_progress.close()
    if state.history and not state.history[-1].get("training_termination_reason"):
        state.history[-1]["training_termination_reason"] = "epoch_budget"
    _set_pai_candidate_graph_for_context(context, False)
    return state.history, state.best_metric, state.best_epoch, state.best_state


def _build_artifact_metadata(
    *,
    model_key: str,
    condition_key: str,
    display_name: str,
    metric_name: str,
    metric_direction: str,
    primary_metric_key: str,
    config: TrainingConfig,
) -> ArtifactMetadata:
    return ArtifactMetadata(
        model_key=model_key,
        condition_key=condition_key,
        display_name=display_name,
        metric_name=metric_name,
        metric_direction=metric_direction,
        primary_metric_key=primary_metric_key,
        use_dendrites=config.use_dendrites,
        use_pruning=config.use_pruning,
        bit_width=config.bit_width,
        use_qat=config.use_qat,
        fine_tune_epochs=config.fine_tune_epochs,
        regression_loss=config.regression_loss,
        enable_pai_dendrite_updates=config.enable_pai_dendrite_updates,
        train_dendrites_until_complete=config.train_dendrites_until_complete,
        freeze_dendrite_updates_fraction=config.freeze_dendrite_updates_fraction,
        pai_candidate_graph_batch_limit=config.pai_candidate_graph_batch_limit,
        memory_cleanup_interval_batches=config.memory_cleanup_interval_batches,
        model_scale=config.model_scale,
        model_revision=config.model_revision,
        dataset_revision=config.dataset_revision,
        pai_variant=config.pai_variant,
        pai_fixed_switch_interval=config.pai_fixed_switch_interval,
        pai_dynamic_schedule=config.pai_dynamic_schedule,
        pai_save_name=config.pai_save_name,
        quantization_granularity=config.quantization_granularity,
        lr_schedule_epochs=config.lr_schedule_epochs,
        quantization_evaluation_revision=config.quantization_evaluation_revision,
        dendrite_audit_revision=config.dendrite_audit_revision,
        dense_param_count=config.dense_param_count,
        source_dendrite_audit_status=config.source_dendrite_audit_status,
        artifact_id=config.artifact_id,
        seed=config.seed,
        source_condition_key=config.source_condition_key,
        quantizer_revision=config.quantizer_revision,
        module_ids_to_perforate=config.module_ids_to_perforate,
        track_only_module_ids=config.track_only_module_ids,
        parameter_ids_to_track=config.parameter_ids_to_track,
        recipe_override=config.recipe_override,
        pai_override=config.pai_override,
        effective_recipe=config.effective_recipe,
        source_commit=config.source_commit,
        paired_control_identity=config.paired_control_identity,
        max_dendrite_phase_epochs=config.max_dendrite_phase_epochs,
    )


def _metadata_for_stage(
    metadata: ArtifactMetadata,
    *,
    use_qat: bool | None = None,
    fine_tune_epochs: int | None = None,
) -> ArtifactMetadata:
    return ArtifactMetadata(
        model_key=metadata.model_key,
        condition_key=metadata.condition_key,
        display_name=metadata.display_name,
        metric_name=metadata.metric_name,
        metric_direction=metadata.metric_direction,
        primary_metric_key=metadata.primary_metric_key,
        use_dendrites=metadata.use_dendrites,
        use_pruning=metadata.use_pruning,
        bit_width=metadata.bit_width,
        use_qat=metadata.use_qat if use_qat is None else use_qat,
        fine_tune_epochs=(
            metadata.fine_tune_epochs
            if fine_tune_epochs is None
            else fine_tune_epochs
        ),
        regression_loss=metadata.regression_loss,
        enable_pai_dendrite_updates=metadata.enable_pai_dendrite_updates,
        train_dendrites_until_complete=metadata.train_dendrites_until_complete,
        freeze_dendrite_updates_fraction=metadata.freeze_dendrite_updates_fraction,
        pai_candidate_graph_batch_limit=metadata.pai_candidate_graph_batch_limit,
        memory_cleanup_interval_batches=metadata.memory_cleanup_interval_batches,
        model_scale=metadata.model_scale,
        model_revision=metadata.model_revision,
        dataset_revision=metadata.dataset_revision,
        pai_variant=metadata.pai_variant,
        pai_fixed_switch_interval=metadata.pai_fixed_switch_interval,
        pai_dynamic_schedule=metadata.pai_dynamic_schedule,
        pai_save_name=metadata.pai_save_name,
        quantization_granularity=metadata.quantization_granularity,
        lr_schedule_epochs=metadata.lr_schedule_epochs,
        quantization_evaluation_revision=metadata.quantization_evaluation_revision,
        dendrite_audit_revision=metadata.dendrite_audit_revision,
        dense_param_count=metadata.dense_param_count,
        source_dendrite_audit_status=metadata.source_dendrite_audit_status,
        artifact_id=metadata.artifact_id,
        seed=metadata.seed,
        source_condition_key=metadata.source_condition_key,
        quantizer_revision=metadata.quantizer_revision,
        module_ids_to_perforate=metadata.module_ids_to_perforate,
        track_only_module_ids=metadata.track_only_module_ids,
        parameter_ids_to_track=metadata.parameter_ids_to_track,
        recipe_override=metadata.recipe_override,
        pai_override=metadata.pai_override,
        effective_recipe=metadata.effective_recipe,
        source_commit=metadata.source_commit,
        paired_control_identity=metadata.paired_control_identity,
        max_dendrite_phase_epochs=metadata.max_dendrite_phase_epochs,
    )


def _capture_before_pqat_snapshot(
    *,
    model: Any,
    model_key: str,
    bundle: Any,
    device: Any,
    criterion: Any,
    metric_name: str,
    torch: Any,
    primary_metric_key: str,
    metric_direction: str,
    output_dir: Path,
    metadata: ArtifactMetadata,
) -> None:
    before_test_loss, before_test_metrics = _eval_on_loader(
        model, model_key, bundle.test_loader, device, criterion, metric_name, torch,
        target_offset=getattr(bundle, "target_offset", 0.0),
        target_scale=getattr(bundle, "target_scale", 1.0),
    )
    before_final_metric = float(before_test_metrics.get(primary_metric_key, 0.0))
    payload = ArtifactPayload(
        best_metric=before_final_metric,
        final_metric=before_final_metric,
        best_epoch=0,
        history=[
            {
                "epoch": 0,
                "primary_metric_name": metric_name,
                "primary_metric_key": primary_metric_key,
                "metric_direction": metric_direction,
                "test_loss": before_test_loss,
                "test_primary_metric": before_final_metric,
                **_prefix_metrics("test", before_test_metrics),
            }
        ],
        test_loss=before_test_loss,
        test_metrics=before_test_metrics,
        training_skipped=True,
        skip_reason="pre-PQAT PTQ snapshot",
        stage_name="before_pqat",
    )
    before_metadata = _metadata_for_stage(
        metadata, use_qat=False, fine_tune_epochs=0
    )
    _persist_stage_artifacts(
        output_dir=output_dir / "before_pqat",
        plain_model=_unwrap_compiled(model),
        metadata=before_metadata,
        payload=payload,
    )


def _attach_test_metrics_to_history(
    history: list[dict[str, Any]],
    *,
    metric_name: str,
    primary_metric_key: str,
    metric_direction: str,
    test_loss: float,
    final_metric: float,
    test_metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    if not history:
        history.append(
            {
                "epoch": 0,
                "primary_metric_name": metric_name,
                "primary_metric_key": primary_metric_key,
                "metric_direction": metric_direction,
            }
        )
    history[-1]["test_loss"] = test_loss
    history[-1]["test_primary_metric"] = final_metric
    history[-1].update(_prefix_metrics("test", test_metrics))
    return history


def _persist_post_pqat_snapshot(
    *,
    enabled: bool,
    output_dir: Path,
    plain_model: Any,
    metadata: ArtifactMetadata,
    payload: ArtifactPayload,
    parameter_stats: tuple[int, int] | None = None,
    topology_hash: str | None = None,
) -> None:
    if not enabled:
        return
    _persist_stage_artifacts(
        output_dir=output_dir / "after_pqat",
        plain_model=plain_model,
        metadata=metadata,
        payload=payload,
        parameter_stats=parameter_stats,
        topology_hash=topology_hash,
    )


def _persist_over_budget_snapshot(
    *,
    enabled: bool,
    output_dir: Path,
    plain_model: Any,
    metadata: ArtifactMetadata,
    payload: ArtifactPayload,
    max_epochs: int,
    parameter_stats: tuple[int, int] | None = None,
    topology_hash: str | None = None,
) -> ArtifactPayload:
    if not enabled:
        return payload
    canonical_history = [
        row for row in payload.history if int(row.get("epoch", 0)) <= max_epochs
    ]
    over_budget_history = [
        row for row in payload.history if int(row.get("epoch", 0)) > max_epochs
    ]
    if not over_budget_history:
        return payload
    if canonical_history:
        final_test_fields = {
            key: value
            for key, value in payload.history[-1].items()
            if key.startswith("test_")
        }
        canonical_history[-1].update(final_test_fields)
    over_payload = ArtifactPayload(
        best_metric=payload.best_metric,
        final_metric=payload.final_metric,
        best_epoch=payload.best_epoch,
        history=over_budget_history,
        test_loss=payload.test_loss,
        test_metrics=payload.test_metrics,
        training_skipped=payload.training_skipped,
        skip_reason=payload.skip_reason,
        stage_name="continued_until_complete",
    )
    _persist_stage_artifacts(
        output_dir=output_dir / "continued_until_complete",
        plain_model=plain_model,
        metadata=metadata,
        payload=over_payload,
        parameter_stats=parameter_stats,
        topology_hash=topology_hash,
    )
    return ArtifactPayload(
        best_metric=payload.best_metric,
        final_metric=payload.final_metric,
        best_epoch=payload.best_epoch,
        history=canonical_history,
        test_loss=payload.test_loss,
        test_metrics=payload.test_metrics,
        training_skipped=payload.training_skipped,
        skip_reason=payload.skip_reason,
        stage_name=payload.stage_name,
    )


def _configure_mps_matmul_precision(torch: Any, device: Any) -> None:
    if getattr(device, "type", "") == "mps" and hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def _is_pqat_enabled(config: TrainingConfig, condition_key: str) -> bool:
    bit_width = config.bit_width
    return (
        bit_width is not None
        and bit_width < 32
        and config.use_qat
        and config.fine_tune_epochs > 0
        and config.source_condition_key is not None
        and config.source_condition_key != condition_key
    )


def _use_pai_runtime_guard() -> bool:
    return True


def _prepare_model_for_training(
    model: Any,
    *,
    torch: Any,
    device: Any,
    model_key: str,
    condition_key: str,
    config: TrainingConfig,
) -> Any:
    model = model.to(device)
    _configure_dendrite_output_dimensions(
        model, model_key, config.use_dendrites, device
    )
    if config.use_pruning:
        _apply_pruning(model, torch, config.prune_amount)
    if _should_quantize_for_training(config):
        # Snapshot full precision *before* the first hard quantization below —
        # this is the shadow PQAT fine-tuning trains, see _qat_init_shadow.
        _qat_init_shadow(model)
        model = _make_quantized_copy(
            model,
            config.bit_width,
            config.quantization_mode,
            config.quantization_granularity,
        )
    if config.max_epochs <= 0:
        return model
    return _apply_torch_compile(
        model, torch, model_key, condition_key, device, config.use_dendrites
    )


def _run_or_skip_training(
    *,
    context: EpochTrainingContext,
    optimizer: Any,
    pai_tracker: Any | None,
    skip_reason: str,
    source_condition_key: str | None,
    condition_key: str,
) -> tuple[list[dict[str, Any]], float, int, dict[str, Any] | None]:
    if context.max_epochs > 0:
        return _run_training_epochs(context, optimizer, pai_tracker)
    _print_skip_banner(
        context.run_label,
        skip_reason,
        source_condition_key,
        condition_key,
        context.config.bit_width,
        context.config.quantization_mode,
    )
    return [], _initial_epoch_state(context.metric_direction).best_metric, 0, None


def train_and_evaluate(
    *,
    model_key: str,
    condition_key: str,
    display_name: str,
    metric_name: str,
    metric_direction: str,
    model: Any,
    bundle: Any,
    output_dir: Path,
    config: TrainingConfig | None = None,
) -> TrainingRecord:
    if config is None:
        config = TrainingConfig()
    bit_width = config.bit_width
    quantization_mode = config.quantization_mode
    use_dendrites = config.use_dendrites
    use_qat = config.use_qat
    max_epochs = config.max_epochs
    source_condition_key = config.source_condition_key

    device = _resolve_device(model_key, torch)
    _configure_mps_matmul_precision(torch, device)
    output_dir.mkdir(parents=True, exist_ok=True)
    if use_dendrites and _pai_updates_enabled(config):
        _post_pai_run_config_event(config)
    primary_metric_key = _PRIMARY_METRIC_KEY.get(model_key, "accuracy")
    metadata = _build_artifact_metadata(
        model_key=model_key,
        condition_key=condition_key,
        display_name=display_name,
        metric_name=metric_name,
        metric_direction=metric_direction,
        primary_metric_key=primary_metric_key,
        config=config,
    )

    pqat_enabled = _is_pqat_enabled(config, condition_key)
    model = _prepare_model_for_training(
        model,
        torch=torch,
        device=device,
        model_key=model_key,
        condition_key=condition_key,
        config=config,
    )

    pai_guard = (
        pai_runtime_guard()
        if use_dendrites and _use_pai_runtime_guard()
        else nullcontext(None)  # type: ignore[no-matching-overload]
    )
    with pai_guard:
        optimizer, pai_tracker = _setup_pai_optimizer(model, torch, config)
        criterion = _binary_or_multi_loss(model_key, config)
        start_time = time.perf_counter()
        run_label = f"{model_key} | {condition_key}"

        training_skipped, skip_reason = _determine_skip_info(
            max_epochs, bit_width, use_qat, quantization_mode
        )

        if pqat_enabled:
            _capture_before_pqat_snapshot(
                model=model,
                model_key=model_key,
                bundle=bundle,
                device=device,
                criterion=criterion,
                metric_name=metric_name,
                torch=torch,
                primary_metric_key=primary_metric_key,
                metric_direction=metric_direction,
                output_dir=output_dir,
                metadata=metadata,
            )

        epoch_context = EpochTrainingContext(
            model=model,
            model_key=model_key,
            bundle=bundle,
            device=device,
            criterion=criterion,
            torch=torch,
            max_epochs=max_epochs,
            run_label=run_label,
            config=config,
            metric_name=metric_name,
            primary_metric_key=primary_metric_key,
            metric_direction=metric_direction,
            output_dir=output_dir,
        )
        history, best_metric, best_epoch, best_state = _run_or_skip_training(
            context=epoch_context,
            optimizer=optimizer,
            pai_tracker=pai_tracker,
            skip_reason=skip_reason,
            source_condition_key=source_condition_key,
            condition_key=condition_key,
        )
        model = epoch_context.model

    if best_state is not None:
        # Load into the underlying module; the compiled wrapper's forward graph
        # reads parameters in-place from the same tensors, so it stays in sync.
        # If the dendrite structure changed after the best epoch, this is a
        # deliberate no-op -- `model` stays the self-consistent final-epoch
        # model rather than becoming a best/final hybrid. best_metric_value and
        # best_epoch below still describe the true validation peak either way;
        # metric_value below describes whichever model was actually kept. See
        # information/MEASUREMENT_CAVEATS.md #3.
        _load_compatible_best_state(model, best_state)

    pai_save_name = config.pai_save_name
    if use_dendrites and pai_save_name:
        # Pin the PAI structure to the artifact, not to the epoch loop.
        #
        # model.pt is written from `model` as it stands right here: after the
        # best-epoch restore decision above, before the quantization below.
        # The PAI_RESUME_NAME snapshot, by contrast, is written inside the
        # epoch loop, so if the final epoch added a candidate dendrite (or the
        # best-state restore above declined a structure change) the two
        # disagree -- and every dendrites_q* condition rebuilds its skeleton
        # from the PAI snapshot but takes its weights from model.pt. When the
        # snapshot had *fewer* tensors the load raised (caveat #4); when it had
        # *more*, the extras were silently left at random init, quantized, and
        # scored. That is how actor_critic reported 52,617 params in fp32 and
        # 71,059 in every quantized arm, and m5 50,456 vs 75,696.
        #
        # Snapshotting here makes structure and weights come from the same
        # instant by construction, which is the only way the two independent
        # checkpoint systems can be kept in agreement.
        save_pai_system(
            _unwrap_compiled(model), pai_save_name, PAI_ARTIFACT_NAME
        )

    model = _finalize_quantized_model_for_eval(model, config)

    eval_device = device
    if (
        _should_quantize_for_eval(config)
        and model_key in _TORCH_COMPILE_MPS_BLOCKLIST
        and getattr(device, "type", "") == "mps"
    ):
        # PointNet's bmm-heavy forward double-frees an MPS buffer when run on
        # post-training-quantized weights; evaluate the quantized copy on CPU.
        eval_device = torch.device("cpu")
        model = model.to(eval_device)

    model.eval()
    test_loss, test_metrics = _eval_on_loader(
        model, model_key, bundle.test_loader, eval_device, criterion,
        metric_name, torch,
        target_offset=getattr(bundle, "target_offset", 0.0),
        target_scale=getattr(bundle, "target_scale", 1.0),
    )
    final_metric = float(test_metrics.get(primary_metric_key, 0.0))
    if best_epoch == 0:
        best_metric = final_metric
    # After final_metric is read, so the rollout can never become the primary
    # metric by accident — for the behaviour-cloning models it is recorded, not
    # selected on. On-policy models already got their return from
    # _eval_on_loader above; rolling out again would only burn a minute to
    # produce the same number under the same seed.
    if "episodic_return_mean" not in test_metrics:
        test_metrics.update(_evaluate_episodic_return(model_key, model, eval_device))

    _plain_model = _unwrap_compiled(model)
    history = _attach_test_metrics_to_history(
        history,
        metric_name=metric_name,
        primary_metric_key=primary_metric_key,
        metric_direction=metric_direction,
        test_loss=test_loss,
        final_metric=final_metric,
        test_metrics=test_metrics,
    )
    payload = ArtifactPayload(
        best_metric=best_metric,
        final_metric=final_metric,
        best_epoch=best_epoch,
        history=history,
        test_loss=test_loss,
        test_metrics=test_metrics,
        training_skipped=training_skipped,
        skip_reason=skip_reason,
        stage_name="after_pqat" if pqat_enabled else None,
    )
    _final_clean_stats = (
        _final_clean_pai_parameter_stats(_plain_model) if use_dendrites else None
    )
    final_parameter_stats = (
        _final_clean_stats[:2] if _final_clean_stats is not None else None
    )
    final_topology_hash = (
        _final_clean_stats[2] if _final_clean_stats is not None else None
    )
    final_param_count = (
        final_parameter_stats[0]
        if final_parameter_stats is not None
        else _count_parameters(_plain_model)[0]
    )
    dendrite_audit = _dendrite_audit(
        metadata=metadata,
        param_count=final_param_count,
    )
    if dendrite_audit["status"] in {
        "no_retained_insertion",
        "inherited_no_retained_insertion",
        "unverified",
        "inherited_unverified",
    }:
        print(
            f"[audit] {run_label}: {dendrite_audit['status']} — "
            f"{dendrite_audit['reason']}"
        )
    _persist_post_pqat_snapshot(
        enabled=pqat_enabled,
        output_dir=output_dir,
        plain_model=_plain_model,
        metadata=metadata,
        payload=payload,
        parameter_stats=final_parameter_stats,
        topology_hash=final_topology_hash,
    )
    payload = _persist_over_budget_snapshot(
        enabled=use_dendrites and config.train_dendrites_until_complete,
        output_dir=output_dir,
        plain_model=_plain_model,
        metadata=metadata,
        payload=payload,
        max_epochs=max_epochs,
        parameter_stats=final_parameter_stats,
        topology_hash=final_topology_hash,
    )

    _, file_size_mb, param_count, nonzero_params = _persist_stage_artifacts(
        output_dir=output_dir,
        plain_model=_plain_model,
        metadata=metadata,
        payload=payload,
        parameter_stats=final_parameter_stats,
        topology_hash=final_topology_hash,
    )

    record = TrainingRecord(
        model_key=model_key,
        condition_key=condition_key,
        display_name=display_name,
        metric_name=metric_name,
        metric_value=final_metric,
        metric_direction=metric_direction,
        best_metric_value=best_metric,
        best_epoch=best_epoch,
        param_count=param_count,
        nonzero_params=nonzero_params,
        file_size_mb=file_size_mb,
        train_seconds=time.perf_counter() - start_time,
        artifact_dir=str(output_dir),
        training_skipped=training_skipped,
        skip_reason=skip_reason,
        dendrite_audit_status=str(dendrite_audit["status"]),
        dendrite_audit_reason=str(dendrite_audit["reason"]),
        artifact_id=config.artifact_id,
    )
    _write_best_model_stats_csv(output_dir, record)
    if use_dendrites and pai_save_name:
        _copy_pai_graphs_to_output(pai_save_name, output_dir)
    return record
