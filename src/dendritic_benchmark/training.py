from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from .compat import (
    binary_quantize_tensor,
    choose_device,
    require_torch,
    symmetric_quantize_tensor,
    ternary_quantize_tensor,
)


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


def _metric_is_better(new: float, old: float, direction: str) -> bool:
    return new > old if direction == "maximize" else new < old


def _accuracy(logits, targets):
    return (logits.argmax(dim=-1) == targets).float().mean().item()


def _mae(preds, targets):
    return (preds - targets).abs().mean().item()


def _rmse(preds, targets):
    return math.sqrt(((preds - targets) ** 2).mean().item())


def _auc(scores, targets):
    scores = scores.detach().flatten()
    targets = targets.detach().flatten().long()
    positives = scores[targets == 1]
    negatives = scores[targets == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return 0.5
    comparisons = (positives[:, None] > negatives[None, :]).float()
    ties = (positives[:, None] == negatives[None, :]).float() * 0.5
    return (comparisons + ties).mean().item()


def _reward_proxy(preds, targets):
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
}


def _binary_or_multi_loss(model_key: str):
    torch = require_torch()
    if model_key in {"lstm_forecaster", "mpnn"}:
        return torch.nn.MSELoss()
    if model_key in {"lstm_autoencoder"}:
        return torch.nn.MSELoss()
    if model_key == "actor_critic":
        return torch.nn.CrossEntropyLoss()
    return torch.nn.CrossEntropyLoss()


def _collapse_metric(
    model_key: str, outputs: Any, targets: Any, metric_targets: Any | None = None
) -> float:
    if model_key == "actor_critic":
        outputs = outputs[0]
    if model_key == "lstm_forecaster":
        return _mae(outputs, targets)
    if model_key == "mpnn":
        return _rmse(outputs, targets)
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
    if model_key in {"lstm_forecaster", "mpnn"}:
        return _regression_metrics(outputs, targets)
    if model_key == "lstm_autoencoder":
        return _anomaly_metrics(outputs, targets, metric_targets)
    metrics = _classification_metrics(outputs, targets)
    if model_key == "actor_critic":
        metrics["reward_proxy"] = metrics["accuracy"]
    if metric_name.lower() == "accuracy":
        metrics["primary_alias"] = metrics["accuracy"]
    return metrics


def _prefix_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


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


def _forward(model_key: str, model: Any, batch: tuple[Any, ...]) -> tuple[Any, Any]:
    if model_key in {"gcn"}:
        x, adjacency, targets = batch
        return model(x, adjacency), targets
    if model_key in {"mpnn"}:
        node_features, adjacency, targets = batch
        return model(node_features, adjacency), targets
    if model_key == "lstm_autoencoder":
        x, target, metric_targets = batch
        return model(x), target, metric_targets
    if model_key == "actor_critic":
        x, targets = batch
        return model(x), targets
    x, targets = batch
    return model(x), targets


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
    if use_dendrites:
        for candidate in ("final_clean_pai", "best_model", "model.pt"):
            path = output_dir / candidate
            if path.exists():
                return path
    return output_dir / "model.pt"


def _format_metric_value(value: float) -> str:
    if math.isfinite(value):
        return f"{value:.4f}"
    return "n/a"


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
    bit_width: int | None = None,
    quantization_mode: str | None = None,
    use_dendrites: bool = False,
    use_pruning: bool = False,
    prune_amount: float = 0.4,
    use_qat: bool = False,
    fine_tune_epochs: int = 0,
    max_epochs: int = 8,
    source_condition_key: str | None = None,
) -> TrainingRecord:
    torch = require_torch()
    device = choose_device()
    output_dir.mkdir(parents=True, exist_ok=True)
    model = model.to(device)
    primary_metric_key = _PRIMARY_METRIC_KEY.get(model_key, "accuracy")

    if (
        use_dendrites
        and model_key == "gcn"
        and hasattr(model, "conv2")
        and hasattr(model.conv2, "linear")
    ):
        linear = model.conv2.linear
        if hasattr(linear, "set_this_output_dimensions"):
            linear.set_this_output_dimensions(torch.tensor([-1, 0], device=device))

    if use_pruning:
        try:
            import torch.nn.utils.prune as prune

            parameters_to_prune = [
                (module, "weight")
                for module in model.modules()
                if isinstance(
                    module, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d)
                )
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

    if bit_width is not None and bit_width < 32 and use_qat:
        model = _make_quantized_copy(model, bit_width, quantization_mode)

    # Compile non-dendritic models on MPS to fuse operators and cut Python-dispatch
    # overhead.  The "aot_eager" backend is used because the default "inductor"
    # backend requires Triton, which is not available on macOS / Apple Silicon.
    # PerforatedAI-wrapped (dendritic) models are excluded since their custom
    # forward logic may not be fully traceable by the compiler.
    if (
        not use_dendrites
        and hasattr(torch, "compile")
        and getattr(device, "type", "") == "mps"
    ):
        try:
            model = torch.compile(model, backend="aot_eager", fullgraph=False)
            print(
                f"[compile] torch.compile(aot_eager) applied to {model_key}/{condition_key}"
            )
        except Exception as _compile_exc:
            print(
                f"[compile] torch.compile skipped for {model_key}/{condition_key}: "
                f"{_compile_exc}"
            )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = _binary_or_multi_loss(model_key)
    history: list[dict[str, Any]] = []
    best_metric = -math.inf if metric_direction == "maximize" else math.inf
    best_epoch = 0
    best_state = None
    start_time = time.perf_counter()
    run_label = f"{model_key} | {condition_key}"

    # Determine whether the training loop will be skipped entirely (PTQ conditions).
    training_skipped = max_epochs == 0
    if training_skipped:
        if bit_width is not None and bit_width < 32 and not use_qat:
            _quant_desc = f"{bit_width}-bit {quantization_mode or 'int'}"
            skip_reason = (
                f"post-training quantization ({_quant_desc})"
                " — weights are quantized without any gradient updates"
            )
        else:
            skip_reason = "no training epochs configured"
    else:
        skip_reason = ""

    if max_epochs > 0:
        epoch_progress = tqdm(
            range(max_epochs),
            desc=run_label,
            unit="epoch",
            leave=True,
            dynamic_ncols=True,
        )

        for epoch in epoch_progress:
            epoch_start = time.perf_counter()
            model.train()
            # Keep the loss sum on-device to eliminate one CPU–GPU sync per batch.
            # A single .item() call at epoch-end materialises the value.
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
                # Refresh at most ~10 times per epoch to reduce tqdm Python overhead.
                miniters=max(1, len(bundle.train_loader) // 10),
            )
            for batch in batch_progress:
                # non_blocking=True lets the CPU queue the next transfer while the
                # GPU is still computing the current batch (no-op for MPS unified
                # memory, but harmless and future-proof for CUDA).
                batch = tuple(item.to(device, non_blocking=True) for item in batch)
                # set_to_none=True skips the memset and lets the allocator reuse
                # gradient buffers, saving a small but consistent amount of work.
                optimizer.zero_grad(set_to_none=True)
                outputs, targets, *rest = _forward(model_key, model, batch)
                if model_key == "actor_critic":
                    loss = criterion(outputs[0], targets)
                else:
                    loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                batch_examples = _batch_size(targets)
                # Accumulate on-device — no CPU–GPU sync in the hot path.
                running_loss_t = running_loss_t + loss.detach() * batch_examples
                train_examples += batch_examples
                metric_targets = rest[0] if rest else None
                detached_outputs, detached_targets, detached_metric_targets = (
                    _detach_metric_payload(model_key, outputs, targets, metric_targets)
                )
                train_outputs.append(detached_outputs)
                train_targets.append(detached_targets)
                if detached_metric_targets is not None:
                    train_metric_targets.append(detached_metric_targets)
                if bit_width is not None and bit_width < 32 and use_qat:
                    _make_quantized_copy(model, bit_width, quantization_mode)
            batch_progress.close()

            # Single sync per epoch to read the accumulated GPU-side loss.
            train_loss = (running_loss_t / max(1, train_examples)).item()
            train_metrics = {}
            if train_outputs:
                train_metrics = _compute_all_metrics(
                    model_key,
                    torch.cat(train_outputs, dim=0),
                    torch.cat(train_targets, dim=0),
                    torch.cat(train_metric_targets, dim=0)
                    if train_metric_targets
                    else None,
                    metric_name=metric_name,
                )

            model.eval()
            val_running_loss_t = torch.zeros(1, device=device)
            val_examples = 0
            val_outputs: list[Any] = []
            val_targets: list[Any] = []
            val_metric_targets: list[Any] = []
            with torch.no_grad():
                for batch in bundle.val_loader:
                    batch = tuple(item.to(device, non_blocking=True) for item in batch)
                    outputs, targets, *rest = _forward(model_key, model, batch)
                    metric_targets = rest[0] if rest else None
                    if model_key == "actor_critic":
                        loss = criterion(outputs[0], targets)
                    else:
                        loss = criterion(outputs, targets)
                    batch_examples = _batch_size(targets)
                    val_running_loss_t = (
                        val_running_loss_t + loss.detach() * batch_examples
                    )
                    val_examples += batch_examples
                    detached_outputs, detached_targets, detached_metric_targets = (
                        _detach_metric_payload(
                            model_key, outputs, targets, metric_targets
                        )
                    )
                    val_outputs.append(detached_outputs)
                    val_targets.append(detached_targets)
                    if detached_metric_targets is not None:
                        val_metric_targets.append(detached_metric_targets)

            val_loss = (val_running_loss_t / max(1, val_examples)).item()
            val_metrics = {}
            if val_outputs:
                val_metrics = _compute_all_metrics(
                    model_key,
                    torch.cat(val_outputs, dim=0),
                    torch.cat(val_targets, dim=0),
                    torch.cat(val_metric_targets, dim=0)
                    if val_metric_targets
                    else None,
                    metric_name=metric_name,
                )
            val_metric = float(val_metrics.get(primary_metric_key, 0.0))

            history_row: dict[str, Any] = {
                "epoch": epoch + 1,
                "primary_metric_name": metric_name,
                "primary_metric_key": primary_metric_key,
                "metric_direction": metric_direction,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "epoch_seconds": time.perf_counter() - epoch_start,
                "train_loss": train_loss,
                "train_primary_metric": float(
                    train_metrics.get(primary_metric_key, 0.0)
                ),
                "val_loss": val_loss,
                "val_primary_metric": val_metric,
                "val_metric": val_metric,
            }
            history_row.update(_prefix_metrics("train", train_metrics))
            history_row.update(_prefix_metrics("val", val_metrics))
            history.append(history_row)
            if best_state is None or _metric_is_better(
                val_metric, best_metric, metric_direction
            ):
                best_metric = val_metric
                best_epoch = epoch + 1
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
            epoch_progress.set_postfix(
                val_metric=_format_metric_value(val_metric),
                best_metric=_format_metric_value(best_metric),
                best_epoch=best_epoch,
            )
        epoch_progress.close()
    else:
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

    if best_state is not None:
        model.load_state_dict(best_state)

    if bit_width is not None and bit_width < 32:
        model = _make_quantized_copy(model, bit_width, quantization_mode)

    model.eval()
    test_running_loss_t = torch.zeros(1, device=device)
    test_examples = 0
    test_outputs: list[Any] = []
    test_targets: list[Any] = []
    test_metric_targets: list[Any] = []
    with torch.no_grad():
        for batch in bundle.test_loader:
            batch = tuple(item.to(device, non_blocking=True) for item in batch)
            outputs, targets, *rest = _forward(model_key, model, batch)
            metric_targets = rest[0] if rest else None
            if model_key == "actor_critic":
                loss = criterion(outputs[0], targets)
            else:
                loss = criterion(outputs, targets)
            batch_examples = _batch_size(targets)
            test_running_loss_t = test_running_loss_t + loss.detach() * batch_examples
            test_examples += batch_examples
            detached_outputs, detached_targets, detached_metric_targets = (
                _detach_metric_payload(model_key, outputs, targets, metric_targets)
            )
            test_outputs.append(detached_outputs)
            test_targets.append(detached_targets)
            if detached_metric_targets is not None:
                test_metric_targets.append(detached_metric_targets)

    test_loss = (test_running_loss_t / max(1, test_examples)).item()
    test_metrics = {}
    if test_outputs:
        test_metrics = _compute_all_metrics(
            model_key,
            torch.cat(test_outputs, dim=0),
            torch.cat(test_targets, dim=0),
            torch.cat(test_metric_targets, dim=0) if test_metric_targets else None,
            metric_name=metric_name,
        )
    final_metric = float(test_metrics.get(primary_metric_key, 0.0))
    if best_epoch == 0:
        best_metric = final_metric

    checkpoint_path = output_dir / "model.pt"
    torch.save(model.state_dict(), checkpoint_path)
    if use_dendrites:
        torch.save(model.state_dict(), output_dir / "best_model")
        torch.save(model.state_dict(), output_dir / "final_clean_pai")
    artifact_path = _artifact_path(output_dir, use_dendrites)
    file_size_mb = artifact_path.stat().st_size / (1024 * 1024)

    param_count = sum(p.numel() for p in model.parameters())
    nonzero_params = sum((p != 0).sum().item() for p in model.parameters())

    if use_dendrites:
        _write_dendritic_sidecars(
            output_dir=output_dir,
            history=history,
            best_metric=best_metric,
            best_epoch=best_epoch,
            param_count=param_count,
            nonzero_params=nonzero_params,
            metric_name=metric_name,
            metric_direction=metric_direction,
        )

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

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "model_key": model_key,
                "condition_key": condition_key,
                "display_name": display_name,
                "metric_name": metric_name,
                "metric_direction": metric_direction,
                "primary_metric_key": primary_metric_key,
                "best_metric_value": best_metric,
                "metric_value": final_metric,
                "best_epoch": best_epoch,
                "train_history_columns": _history_fieldnames(history),
                "test_loss": test_loss,
                "test_metrics": test_metrics,
                "param_count": param_count,
                "nonzero_params": nonzero_params,
                "file_size_mb": file_size_mb,
                "use_dendrites": use_dendrites,
                "use_pruning": use_pruning,
                "bit_width": bit_width,
                "use_qat": use_qat,
                "fine_tune_epochs": fine_tune_epochs,
                "artifact_path": str(artifact_path),
                "training_skipped": training_skipped,
                "skip_reason": skip_reason,
            },
            indent=2,
        )
    )
    with (output_dir / "history.csv").open("w", newline="") as fh:
        fieldnames = _history_fieldnames(history)
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)

    return TrainingRecord(
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
