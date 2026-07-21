import csv
import gc
import importlib
import itertools
import json
import math
import os
import shutil
import subprocess
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from tqdm.auto import tqdm

from .compat import (
    MODULE_OUTPUT_DIMENSIONS_ATTR,
    attach_module_output_dimensions,
    binary_quantize_tensor,
    choose_device,
    clear_pai_processor_buffers,
    clear_pai_tracker_state,
    configure_pai_candidate_graph,
    pai_runtime_guard,
    set_module_output_dimensions,
    symmetric_quantize_tensor,
    ternary_quantize_tensor,
)

_MODEL_PT: str = "model.pt"
_BEST_MODEL_STATS_CSV: str = "best_model_stats.csv"
_EPOCH_CHECKPOINT_PT: str = "epoch_checkpoint.pt"
_MEMORY_GUARD_THRESHOLD_BYTES = 10 * 1024**3
_MEMORY_GUARD_CHECK_INTERVAL_BATCHES = 16
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
OptimizerName = Literal["adam", "adamw", "sgd"]


@dataclass
class TrainingConfig:
    bit_width: int | None = None
    quantization_mode: str | None = None
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
    source_condition_key: str | None = None
    enable_pai_dendrite_updates: bool = False
    train_dendrites_until_complete: bool = False
    freeze_dendrite_updates_fraction: float = 0.20
    pai_candidate_graph_batch_limit: int | None = None
    memory_cleanup_interval_batches: int | None = None
    pai_save_name: str | None = None


@dataclass
class TrainingRecord:
    model_key: str
    condition_key: str
    display_name: str
    metric_name: str
    metric_value: float
    metric_direction: str
    best_epoch: int
    param_count: int
    nonzero_params: int
    file_size_mb: float
    # Set to True when max_epochs==0 (post-training quantization — no gradient updates).
    training_skipped: bool = False
    # Human-readable explanation of why training was skipped (empty string when training ran).
    skip_reason: str = ""

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
    enable_pai_dendrite_updates: bool
    train_dendrites_until_complete: bool
    freeze_dendrite_updates_fraction: float
    pai_candidate_graph_batch_limit: int | None
    memory_cleanup_interval_batches: int | None


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


_PRIMARY_METRIC_KEY: dict[str, str] = {
    "lenet5": "accuracy",
    "m5": "accuracy",
    "lstm_forecaster": "mae",
    "textcnn": "accuracy",
    "gcn": "accuracy",
    "tabnet": "accuracy",
    "mpnn": "rmse",
    "actor_critic": "reward_proxy",
    "lstm_autoencoder": "auc",
    "distilbert": "accuracy",
    "dqn_lunarlander": "reward_proxy",
    "ppo_bipedalwalker": "reward_proxy",
    "attentivefp_freesolv": "rmse",
    "gin_imdbb": "accuracy",
    "tcn_forecaster": "mae",
    "gru_forecaster": "mae",
    "pointnet_modelnet40": "accuracy",
    "vae_mnist": "elbo",
    "snn_nmnist": "accuracy",
    "unet_isic": "dice",
    "resnet18_cifar10": "accuracy",
    "mobilenetv2_cifar10": "accuracy",
    "saint_adult": "accuracy",
    "capsnet_mnist": "accuracy",
}


def _binary_or_multi_loss(model_key: str) -> Any:
    if model_key in {"lstm_forecaster", "mpnn", "attentivefp_freesolv", "tcn_forecaster", "gru_forecaster", "ppo_bipedalwalker"}:
        return torch.nn.MSELoss()
    if model_key in {"lstm_autoencoder"}:
        return torch.nn.MSELoss()
    if model_key == "unet_isic":
        return torch.nn.BCEWithLogitsLoss()
    if model_key == "vae_mnist":
        return None
    if model_key == "actor_critic":
        return torch.nn.CrossEntropyLoss()
    return torch.nn.CrossEntropyLoss()


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
    if model_key == "vae_mnist" and isinstance(outputs, tuple):
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


def _compute_all_metrics(
    model_key: str,
    outputs: Any,
    targets: Any,
    metric_targets: Any | None,
    *,
    metric_name: str,
) -> dict[str, float]:
    if model_key == "actor_critic" and isinstance(outputs, tuple):
        outputs = outputs[0]
    if model_key in {"lstm_forecaster", "mpnn", "attentivefp_freesolv", "tcn_forecaster", "gru_forecaster"}:
        return _regression_metrics(outputs, targets)
    if model_key == "ppo_bipedalwalker":
        metrics = _regression_metrics(outputs, targets)
        metrics["reward_proxy"] = -metrics["mae"]
        return metrics
    if model_key == "vae_mnist":
        return _vae_metrics(outputs, targets)
    if model_key == "unet_isic":
        return {"dice": _dice_from_logits(outputs, targets)}
    if model_key == "lstm_autoencoder":
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
    if model_key in {"gcn", "gin_imdbb"}:
        x, adjacency, targets = batch
        return model(x, adjacency), targets, None
    if model_key in {"mpnn", "attentivefp_freesolv"}:
        node_features, adjacency, targets = batch
        return model(node_features, adjacency), targets, None
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
        batch = tuple(item.to(device, non_blocking=True) for item in batch)
        with torch.no_grad():
            _forward(model_key, model, batch)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    return dimensions


def _compute_loss(model_key: str, criterion: Any, outputs: Any, targets: Any) -> Any:
    if model_key == "actor_critic":
        return criterion(outputs[0], targets)
    if model_key == "vae_mnist":
        return _vae_loss(outputs, targets)
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


def _make_quantized_copy(
    model: Any, bit_width: int | None, mode: str | None = None
) -> Any:
    if bit_width is None or bit_width >= 32:
        return model
    # Quantize on CPU and stream results back to each parameter.  Running the
    # quantization kernels on MPS triggered a malloc double-free for PointNet
    # post-training ternary quantization, so we keep the math off-device.
    with torch.no_grad():
        for param in model.parameters():
            if param.numel() == 0:
                continue
            cpu_param = param.detach().cpu()
            if mode == "binary" or bit_width == 1:
                quantized = binary_quantize_tensor(cpu_param)
            elif mode == "ternary":
                quantized = ternary_quantize_tensor(cpu_param)
            else:
                quantized = symmetric_quantize_tensor(cpu_param, bit_width)
            param.copy_(quantized.to(param.device))
    return model


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
    param_count = sum(p.numel() for p in model.parameters())
    nonzero_params = sum((p != 0).sum().item() for p in model.parameters())
    return param_count, nonzero_params


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
                "use_qat": metadata.use_qat,
                "fine_tune_epochs": metadata.fine_tune_epochs,
                "enable_pai_dendrite_updates": metadata.enable_pai_dendrite_updates,
                "train_dendrites_until_complete": metadata.train_dendrites_until_complete,
                "freeze_dendrite_updates_fraction": (
                    metadata.freeze_dendrite_updates_fraction
                ),
                "memory_cleanup_interval_batches": (
                    metadata.memory_cleanup_interval_batches
                ),
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
) -> tuple[Path, float, int, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / _MODEL_PT
    torch.save(plain_model.state_dict(), checkpoint_path)
    artifact_path = _artifact_path(output_dir, metadata.use_dendrites)
    file_size_mb = artifact_path.stat().st_size / (1024 * 1024)
    param_count, nonzero_params = _count_parameters(plain_model)
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
_TORCH_COMPILE_MPS_BLOCKLIST: frozenset[str] = frozenset({"pointnet_modelnet40", "snn_nmnist"})

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


def _build_optimizer(model: Any, torch: Any, config: TrainingConfig) -> Any:
    if config.optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    if config.optimizer_name == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    return torch.optim.Adam(
        model.parameters(),
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
        "params": model.parameters(),
        "lr": config.learning_rate,
        "weight_decay": config.weight_decay,
    }
    if config.optimizer_name == "sgd":
        args["momentum"] = config.momentum
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
        upa = importlib.import_module("perforatedai.utils_perforatedai")
        return len(upa.get_pai_modules(model, 0))
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
    src = Path("PAI") / pai_save_name
    if not src.exists():
        return
    dst = output_dir / "pai_plots"
    dst.mkdir(parents=True, exist_ok=True)
    for ext in ("*.png", "*.svg", "*.pdf"):
        for f in src.glob(ext):
            shutil.copy2(f, dst / f.name)


def _post_pai_run_config_event(config: TrainingConfig) -> None:
    try:
        gpa = importlib.import_module("perforatedai.globals_perforatedai")
        events_url = getattr(gpa.pc, "events_url", None)
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
        tracker.set_optimizer(_optimizer_class(torch, config))
        setup_result = tracker.setup_optimizer(
            model, _optimizer_args(model, config), {}
        )
    except TypeError:
        try:
            setup_result = tracker.setup_optimizer(
                model, _optimizer_args(model, config)
            )
        except Exception:
            return optimizer, tracker
    except Exception:
        return optimizer, tracker
    if isinstance(setup_result, tuple) and setup_result:
        return setup_result[0], tracker
    if setup_result is not None:
        return setup_result, tracker
    return optimizer, tracker


def _eval_on_loader(
    model: Any,
    model_key: str,
    loader: Any,
    device: Any,
    criterion: Any,
    metric_name: str,
    torch: Any,
) -> tuple[float, dict[str, Any]]:
    """Run evaluation on a dataloader, return (loss, metrics)."""
    running_loss_t = torch.zeros(1, device=device)
    examples = 0
    outputs_list: list[Any] = []
    targets_list: list[Any] = []
    metric_targets_list: list[Any] = []
    with torch.no_grad():
        for batch in loader:
            batch = tuple(item.to(device, non_blocking=True) for item in batch)
            outputs, targets, metric_targets = _forward(model_key, model, batch)
            loss = _compute_loss(model_key, criterion, outputs, targets)
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
        # Once any dendrites exist, PAI's optimizer.step (closure_pai_step) does
        # an internal backward through the candidate graph in p-phase. Disabling
        # that graph mid-epoch frees the saved tensors and the next p-step
        # raises "Trying to backward through the graph a second time", so the
        # limit is only safe during the initial all-neuron correlation phase.
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
    return tuple(item.to(device, non_blocking=True) for item in batch)


def _backward_and_step(
    loss: Any,
    optimizer: Any,
    *,
    retain_graph_for_optimizer_step: bool,
) -> None:
    # PerforatedAI's optimizer step may run its own backward pass after the
    # benchmark's loss backward. Standard torch optimizers do not need the
    # graph retained; keeping it on long MPS runs causes per-batch memory growth.
    loss.backward(retain_graph=retain_graph_for_optimizer_step)
    optimizer.step()


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


def _maybe_apply_qat_projection(model: Any, config: "TrainingConfig") -> None:
    if config.bit_width is not None and config.bit_width < 32 and config.use_qat:
        _make_quantized_copy(model, config.bit_width, config.quantization_mode)


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
    outputs, targets, metric_targets = _forward(model_key, model, batch)
    loss = _compute_loss(model_key, criterion, outputs, targets)
    _backward_and_step(
        loss,
        optimizer,
        retain_graph_for_optimizer_step=retain_graph_for_optimizer_step,
    )
    if clear_pai_buffers:
        clear_pai_processor_buffers(model)
    _maybe_apply_qat_projection(model, config)
    return outputs, targets, metric_targets, loss


def _finalize_training_batch_metrics(
    accumulator: TrainingBatchAccumulator,
    *,
    model_key: str,
    torch: Any,
    metric_name: str,
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
    )
    return train_loss, train_metrics


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
    return _finalize_training_batch_metrics(
        accumulator,
        model_key=model_key,
        torch=torch,
        metric_name=metric_name,
    )


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
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        print(f"[checkpoint] failed to load epoch checkpoint ({exc}); starting from scratch.")
        return None


def _apply_epoch_checkpoint(
    ckpt: dict,
    state: "EpochTrainingState",
    model: Any,
    optimizer: Any,
) -> int:
    resume_epoch = int(ckpt["epoch"]) + 1
    _load_compatible_best_state(model, ckpt["model_state_dict"])
    try:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    except Exception as exc:
        print(f"[checkpoint] could not restore optimizer state ({exc}); optimizer reset.")
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
    history_row: dict[str, Any] = {
        "epoch": epoch + 1,
        "primary_metric_name": context.metric_name,
        "primary_metric_key": primary_metric_key,
        "metric_direction": context.metric_direction,
        "learning_rate": float(optimizer.param_groups[0]["lr"]),
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


def _load_compatible_best_state(model: Any, best_state: dict[str, Any]) -> None:
    plain_model = _unwrap_compiled(model)
    adopted = _adopt_missing_pai_dendrite_buffers(plain_model, best_state)
    current_state = plain_model.state_dict()
    compatible_state: dict[str, Any] = {}
    skipped: list[str] = []
    for key, value in best_state.items():
        if _is_ignorable_state_key(key):
            continue
        current_value = current_state.get(key)
        current_shape = _tensor_shape(current_value)
        source_shape = _tensor_shape(value)
        if current_shape is None or source_shape is None or current_shape != source_shape:
            skipped.append(key)
            continue
        compatible_state[key] = value
    missing, unexpected = plain_model.load_state_dict(compatible_state, strict=False)
    if adopted:
        print(
            "[state] adopted lazy PAI dendrite buffers: "
            + ", ".join(adopted[:5])
            + ("..." if len(adopted) > 5 else "")
        )
    if skipped:
        print(
            "[state] skipped incompatible best-state tensors: "
            + ", ".join(skipped[:5])
            + ("..." if len(skipped) > 5 else "")
        )
    if unexpected:
        print(f"[state] ignored unexpected best-state tensors: {unexpected[:5]}")
    real_missing = [key for key in missing if not _is_ignorable_state_key(key)]
    if real_missing:
        print(f"[state] retained current values for missing tensors: {real_missing[:5]}")


def _run_dynamic_dendrite_update(
    *,
    context: EpochTrainingContext,
    optimizer: Any,
    pai_tracker: Any,
    val_metric: float,
) -> tuple[Any, Any | None, bool, bool]:
    import pdb as _pdb
    from typing import Any, Callable

    def _no_set_trace(*, header: str | None = None) -> None:
        _ = header

    pdb_module: Any = _pdb
    _orig_set_trace: Callable[..., None] = pdb_module.set_trace
    pdb_module.set_trace = _no_set_trace
    try:
        module_dimensions = getattr(
            context.model, MODULE_OUTPUT_DIMENSIONS_ATTR, None
        )
        model, restructured, training_complete = pai_tracker.add_validation_score(
            val_metric, context.model
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
        pdb_module.set_trace = _orig_set_trace


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
        context.device, context.criterion, context.metric_name, context.torch
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
    return history_row, val_metric


def _run_active_pai_update(
    *,
    context: EpochTrainingContext,
    optimizer: Any,
    pai_tracker: Any,
    val_metric: float,
) -> tuple[Any, Any | None, bool, bool]:
    _set_pai_candidate_graph_for_context(context, True)
    try:
        return _run_dynamic_dendrite_update(
            context=context, optimizer=optimizer, pai_tracker=pai_tracker,
            val_metric=val_metric,
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
    optimizer, pai_tracker, restructured, training_complete = _run_active_pai_update(
        context=context, optimizer=optimizer, pai_tracker=pai_tracker,
        val_metric=val_metric,
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
    if context.output_dir is not None:
        ckpt = _load_epoch_checkpoint(context.output_dir, context.torch)
        if ckpt is not None:
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
        if context.output_dir is not None:
            _save_epoch_checkpoint(
                context.output_dir, epoch, state, optimizer, context.model, context.torch
            )
        _update_epoch_progress(epoch_progress, context, state, val_metric)
        if pai_training_complete and run_until_pai_complete:
            break
    epoch_progress.close()
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
        enable_pai_dendrite_updates=config.enable_pai_dendrite_updates,
        train_dendrites_until_complete=config.train_dendrites_until_complete,
        freeze_dendrite_updates_fraction=config.freeze_dendrite_updates_fraction,
        pai_candidate_graph_batch_limit=config.pai_candidate_graph_batch_limit,
        memory_cleanup_interval_batches=config.memory_cleanup_interval_batches,
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
        enable_pai_dendrite_updates=metadata.enable_pai_dendrite_updates,
        train_dendrites_until_complete=metadata.train_dendrites_until_complete,
        freeze_dendrite_updates_fraction=metadata.freeze_dendrite_updates_fraction,
        pai_candidate_graph_batch_limit=metadata.pai_candidate_graph_batch_limit,
        memory_cleanup_interval_batches=metadata.memory_cleanup_interval_batches,
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
        model, model_key, bundle.test_loader, device, criterion, metric_name, torch
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
) -> None:
    if not enabled:
        return
    _persist_stage_artifacts(
        output_dir=output_dir / "after_pqat",
        plain_model=plain_model,
        metadata=metadata,
        payload=payload,
    )


def _persist_over_budget_snapshot(
    *,
    enabled: bool,
    output_dir: Path,
    plain_model: Any,
    metadata: ArtifactMetadata,
    payload: ArtifactPayload,
    max_epochs: int,
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
    return (
        config.bit_width is not None
        and config.bit_width < 32
        and config.use_qat
        and config.fine_tune_epochs > 0
        and config.source_condition_key is not None
        and config.source_condition_key != condition_key
    )


def _should_quantize_for_training(config: TrainingConfig) -> bool:
    return config.bit_width is not None and config.bit_width < 32 and config.use_qat


def _should_quantize_for_eval(config: TrainingConfig) -> bool:
    return config.bit_width is not None and config.bit_width < 32


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
        model = _make_quantized_copy(model, config.bit_width, config.quantization_mode)
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
        else nullcontext()
    )
    with pai_guard:
        optimizer, pai_tracker = _setup_pai_optimizer(model, torch, config)
        criterion = _binary_or_multi_loss(model_key)
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
        _load_compatible_best_state(model, best_state)

    if _should_quantize_for_eval(config):
        model = _make_quantized_copy(model, bit_width, quantization_mode)

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
        metric_name, torch
    )
    final_metric = float(test_metrics.get(primary_metric_key, 0.0))
    if best_epoch == 0:
        best_metric = final_metric

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
    _persist_post_pqat_snapshot(
        enabled=pqat_enabled,
        output_dir=output_dir,
        plain_model=_plain_model,
        metadata=metadata,
        payload=payload,
    )
    payload = _persist_over_budget_snapshot(
        enabled=use_dendrites and config.train_dendrites_until_complete,
        output_dir=output_dir,
        plain_model=_plain_model,
        metadata=metadata,
        payload=payload,
        max_epochs=max_epochs,
    )

    _, file_size_mb, param_count, nonzero_params = _persist_stage_artifacts(
        output_dir=output_dir,
        plain_model=_plain_model,
        metadata=metadata,
        payload=payload,
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
    )
    _write_best_model_stats_csv(output_dir, record)
    if use_dendrites and config.pai_save_name:
        _copy_pai_graphs_to_output(config.pai_save_name, output_dir)
    return record
