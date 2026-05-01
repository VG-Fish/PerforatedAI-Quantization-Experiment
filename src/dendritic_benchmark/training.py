from __future__ import annotations

import csv
import importlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from tqdm.auto import tqdm

from .compat import (
    binary_quantize_tensor,
    choose_device,
    require_torch,
    symmetric_quantize_tensor,
    ternary_quantize_tensor,
)

_MODEL_PT: str = "model.pt"
_BEST_MODEL_STATS_CSV: str = "best_model_stats.csv"
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


@dataclass
class EpochTrainingState:
    history: list[dict[str, Any]]
    best_metric: float
    best_epoch: int
    best_state: dict[str, Any] | None


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


def _accuracy(logits: Any, targets: Any) -> float:
    return (logits.argmax(dim=-1) == targets).float().mean().item()


def _mae(preds: Any, targets: Any) -> float:
    return (preds - targets).abs().mean().item()


def _rmse(preds: Any, targets: Any) -> float:
    return math.sqrt(((preds - targets) ** 2).mean().item())


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


def _ssim_proxy(preds: Any, targets: Any) -> float:
    mse = ((preds - targets) ** 2).mean()
    return float((1.0 / (1.0 + mse)).item())


def _vae_loss(outputs: Any, targets: Any) -> Any:
    torch = require_torch()
    recon, mu, logvar = outputs
    bce = torch.nn.functional.binary_cross_entropy(recon, targets, reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return (bce + kld) / max(1, targets.shape[0])


def _vae_metrics(outputs: Any, targets: Any) -> dict[str, float]:
    torch = require_torch()
    recon, mu, logvar = outputs
    bce = torch.nn.functional.binary_cross_entropy(recon, targets, reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    elbo = -float(((bce + kld) / max(1, targets.shape[0])).item())
    return {
        "elbo": elbo,
        "reconstruction_bce": float((bce / max(1, targets.shape[0])).item()),
        "kl_divergence": float((kld / max(1, targets.shape[0])).item()),
    }


def _reward_proxy(preds: Any, targets: Any) -> float:
    if preds.shape == targets.shape and preds.dtype.is_floating_point:
        return 1.0 / (1.0 + _mae(preds, targets))
    return _accuracy(preds, targets)


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
    "convlstm_movingmnist": "ssim",
}


def _binary_or_multi_loss(model_key: str) -> Any:
    torch = require_torch()
    if model_key in {"lstm_forecaster", "mpnn", "attentivefp_freesolv", "tcn_forecaster", "gru_forecaster", "ppo_bipedalwalker", "convlstm_movingmnist"}:
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


def _collapse_metric(
    model_key: str, outputs: Any, targets: Any, metric_targets: Any | None = None
) -> float:
    if model_key in {"actor_critic"}:
        outputs = outputs[0]
    if model_key in {"lstm_forecaster", "tcn_forecaster", "gru_forecaster"}:
        return _mae(outputs, targets)
    if model_key in {"mpnn", "attentivefp_freesolv"}:
        return _rmse(outputs, targets)
    if model_key == "ppo_bipedalwalker":
        return _reward_proxy(outputs, targets)
    if model_key == "unet_isic":
        return _dice_from_logits(outputs, targets)
    if model_key == "convlstm_movingmnist":
        return _ssim_proxy(outputs, targets)
    if model_key == "vae_mnist":
        return _vae_metrics(outputs, targets)["elbo"]
    if model_key == "lstm_autoencoder":
        torch = require_torch()
        if metric_targets is None:
            metric_targets = torch.zeros(outputs.shape[0], device=outputs.device)
        reconstruction_error = ((outputs - targets) ** 2).mean(dim=(1, 2))
        return _auc(-reconstruction_error, metric_targets)
    return _accuracy(outputs, targets)


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
    torch = require_torch()
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
    torch = require_torch()
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
    torch = require_torch()
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
    indices = targets * num_classes + predictions
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
    torch = require_torch()
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
    torch = require_torch()
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
    if model_key == "convlstm_movingmnist":
        metrics = _regression_metrics(outputs, targets)
        metrics["ssim"] = _ssim_proxy(outputs, targets)
        return metrics
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
    torch = require_torch()
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
    x, targets = batch
    return model(x), targets, None


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


def _make_quantized_copy(
    model: Any, bit_width: int | None, mode: str | None = None
) -> Any:
    torch = require_torch()
    if bit_width is None or bit_width >= 32:
        return model
    with torch.no_grad():
        for param in model.parameters():
            if mode == "binary" or bit_width == 1:
                param.copy_(binary_quantize_tensor(param))
            elif mode == "ternary":
                param.copy_(ternary_quantize_tensor(param))
            else:
                param.copy_(symmetric_quantize_tensor(param, bit_width))
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
    torch = require_torch()
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


def _apply_torch_compile(
    model: Any, torch: Any, model_key: str, condition_key: str, device: Any, use_dendrites: bool
) -> Any:
    if use_dendrites or not hasattr(torch, "compile") or getattr(device, "type", "") != "mps":
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


def _setup_pai_optimizer(
    model: Any,
    torch: Any,
    config: TrainingConfig,
) -> tuple[Any, Any | None]:
    optimizer = _build_optimizer(model, torch, config)
    tracker = _pai_tracker()
    if not config.use_dendrites or tracker is None:
        return optimizer, None if not config.use_dendrites else tracker
    try:
        tracker.set_optimizer(_optimizer_class(torch, config))
        setup_result = tracker.setup_optimizer(model, _optimizer_args(model, config), {})
    except TypeError:
        try:
            setup_result = tracker.setup_optimizer(model, _optimizer_args(model, config))
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


def _configure_gcn_dendrites(
    model: Any, model_key: str, use_dendrites: bool, device: Any, torch: Any
) -> None:
    if (
        use_dendrites
        and model_key == "gcn"
        and hasattr(model, "conv2")
        and hasattr(model.conv2, "linear")
    ):
        linear = model.conv2.linear
        if hasattr(linear, "set_this_output_dimensions"):
            linear.set_this_output_dimensions(torch.tensor([-1, 0], device=device))


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
) -> tuple[float, dict[str, Any]]:
    model.train()
    running_loss_t = torch.zeros(1, device=device)
    train_examples = 0
    train_outputs: list[Any] = []
    train_targets: list[Any] = []
    train_metric_targets: list[Any] = []
    batch_progress = tqdm(
        bundle.train_loader,
        desc=f"{run_label} | epoch {epoch + 1}/{max_epochs}",
        unit="batch",
        leave=False,
        dynamic_ncols=True,
        miniters=max(1, len(bundle.train_loader) // 10),
    )
    for batch in batch_progress:
        batch = tuple(item.to(device, non_blocking=True) for item in batch)
        optimizer.zero_grad(set_to_none=True)
        outputs, targets, metric_targets = _forward(model_key, model, batch)
        loss = _compute_loss(model_key, criterion, outputs, targets)
        loss.backward()
        optimizer.step()
        batch_examples = _batch_size(targets)
        running_loss_t = running_loss_t + loss.detach() * batch_examples
        train_examples += batch_examples
        det_out, det_tgt, det_mt = _detach_metric_payload(model_key, outputs, targets, metric_targets)
        train_outputs.append(det_out)
        train_targets.append(det_tgt)
        if det_mt is not None:
            train_metric_targets.append(det_mt)
        if config.bit_width is not None and config.bit_width < 32 and config.use_qat:
            _make_quantized_copy(model, config.bit_width, config.quantization_mode)
    batch_progress.close()
    train_loss = (running_loss_t / max(1, train_examples)).item()
    train_metrics: dict[str, Any] = {}
    if train_outputs:
        train_metrics = _compute_all_metrics(
            model_key,
            _cat_payload(train_outputs),
            torch.cat(train_targets, dim=0),
            torch.cat(train_metric_targets, dim=0) if train_metric_targets else None,
            metric_name=metric_name,
        )
    return train_loss, train_metrics


def _initial_epoch_state(metric_direction: str) -> EpochTrainingState:
    best_metric = -math.inf if metric_direction == "maximize" else math.inf
    return EpochTrainingState([], best_metric, 0, None)


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


def _load_compatible_best_state(model: Any, best_state: dict[str, Any]) -> None:
    plain_model = _unwrap_compiled(model)
    current_state = plain_model.state_dict()
    compatible_state: dict[str, Any] = {}
    skipped: list[str] = []
    for key, value in best_state.items():
        current_value = current_state.get(key)
        if current_value is None or tuple(current_value.shape) != tuple(value.shape):
            skipped.append(key)
            continue
        compatible_state[key] = value
    missing, unexpected = plain_model.load_state_dict(compatible_state, strict=False)
    if skipped:
        print(
            "[state] skipped incompatible best-state tensors: "
            + ", ".join(skipped[:5])
            + ("..." if len(skipped) > 5 else "")
        )
    if unexpected:
        print(f"[state] ignored unexpected best-state tensors: {unexpected[:5]}")
    real_missing = [key for key in missing if not key.endswith("tracker_string")]
    if real_missing:
        print(f"[state] retained current values for missing tensors: {real_missing[:5]}")


def _run_dynamic_dendrite_update(
    *,
    context: EpochTrainingContext,
    optimizer: Any,
    pai_tracker: Any,
    val_metric: float,
) -> tuple[Any, Any | None, bool, bool]:
    try:
        model, restructured, training_complete = pai_tracker.add_validation_score(
            val_metric, context.model
        )
        context.model = model.to(context.device)
        _configure_gcn_dendrites(
            context.model,
            context.model_key,
            context.config.use_dendrites,
            context.device,
            context.torch,
        )
        if restructured:
            optimizer, _ = _setup_pai_optimizer(context.model, context.torch, context.config)
        return optimizer, pai_tracker, bool(restructured), bool(training_complete)
    except Exception as pai_exc:
        print(f"[pai] dynamic dendrite update skipped: {pai_exc}")
        return optimizer, None, False, False


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


def _run_training_epochs(
    context: EpochTrainingContext,
    optimizer: Any,
    pai_tracker: Any | None = None,
) -> tuple[list[dict[str, Any]], float, int, dict[str, Any] | None]:
    state = _initial_epoch_state(context.metric_direction)
    epoch_progress = tqdm(
        range(context.max_epochs),
        desc=context.run_label,
        unit="epoch",
        leave=True,
        dynamic_ncols=True,
    )
    dynamic_freeze_epoch = math.floor(context.max_epochs * 0.8)
    for epoch in epoch_progress:
        epoch_start = time.perf_counter()
        train_loss, train_metrics = _run_epoch_batches(
            context.model, context.model_key, context.bundle, context.device,
            context.criterion, optimizer, context.torch, epoch, context.max_epochs,
            context.run_label, context.config, context.metric_name,
        )
        context.model.eval()
        val_loss, val_metrics = _eval_on_loader(
            context.model, context.model_key, context.bundle.val_loader,
            context.device, context.criterion, context.metric_name, context.torch
        )
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
        history_row["pai_dynamic_insertion_active"] = bool(
            pai_tracker is not None and epoch < dynamic_freeze_epoch
        )
        history_row["pai_restructured"] = False
        history_row["pai_training_complete"] = False
        if pai_tracker is not None and epoch < dynamic_freeze_epoch:
            (
                optimizer,
                pai_tracker,
                restructured,
                training_complete,
            ) = _run_dynamic_dendrite_update(
                context=context, optimizer=optimizer, pai_tracker=pai_tracker,
                val_metric=val_metric,
            )
            history_row["pai_restructured"] = restructured
            history_row["pai_training_complete"] = training_complete
            if training_complete:
                pai_tracker = None
        _set_epoch_progress(
            epoch_progress, context.metric_name, val_metric,
            state.best_metric, state.best_epoch,
        )
    epoch_progress.close()
    return state.history, state.best_metric, state.best_epoch, state.best_state


def _build_artifact_metadata(
    *,
    model_key: str,
    condition_key: str,
    display_name: str,
    metric_name: str,
    metric_direction: str,
    primary_metric_key: str,
    use_dendrites: bool,
    use_pruning: bool,
    bit_width: int | None,
    use_qat: bool,
    fine_tune_epochs: int,
) -> ArtifactMetadata:
    return ArtifactMetadata(
        model_key=model_key,
        condition_key=condition_key,
        display_name=display_name,
        metric_name=metric_name,
        metric_direction=metric_direction,
        primary_metric_key=primary_metric_key,
        use_dendrites=use_dendrites,
        use_pruning=use_pruning,
        bit_width=bit_width,
        use_qat=use_qat,
        fine_tune_epochs=fine_tune_epochs,
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
    _configure_gcn_dendrites(model, model_key, config.use_dendrites, device, torch)
    if config.use_pruning:
        _apply_pruning(model, torch, config.prune_amount)
    if _should_quantize_for_training(config):
        model = _make_quantized_copy(model, config.bit_width, config.quantization_mode)
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
    fine_tune_epochs = config.fine_tune_epochs
    max_epochs = config.max_epochs
    source_condition_key = config.source_condition_key

    torch = require_torch()
    device = choose_device()
    _configure_mps_matmul_precision(torch, device)
    output_dir.mkdir(parents=True, exist_ok=True)
    primary_metric_key = _PRIMARY_METRIC_KEY.get(model_key, "accuracy")
    metadata = _build_artifact_metadata(
        model_key=model_key,
        condition_key=condition_key,
        display_name=display_name,
        metric_name=metric_name,
        metric_direction=metric_direction,
        primary_metric_key=primary_metric_key,
        use_dendrites=use_dendrites,
        use_pruning=config.use_pruning,
        bit_width=bit_width,
        use_qat=use_qat,
        fine_tune_epochs=fine_tune_epochs,
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

    model.eval()
    test_loss, test_metrics = _eval_on_loader(
        model, model_key, bundle.test_loader, device, criterion,
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
    return record
