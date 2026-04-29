from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from .compat import binary_quantize_tensor, choose_device, require_torch, symmetric_quantize_tensor, ternary_quantize_tensor


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


def _binary_or_multi_loss(model_key: str):
    torch = require_torch()
    if model_key in {"lstm_forecaster", "mpnn"}:
        return torch.nn.MSELoss()
    if model_key in {"lstm_autoencoder"}:
        return torch.nn.MSELoss()
    if model_key == "actor_critic":
        return torch.nn.CrossEntropyLoss()
    return torch.nn.CrossEntropyLoss()


def _collapse_metric(model_key: str, outputs: Any, targets: Any, metric_targets: Any | None = None) -> float:
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


def _make_quantized_copy(model: Any, bit_width: int | None, mode: str | None = None) -> Any:
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
    history: list[dict[str, float]],
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
            "best_metric_value": best_metric if row["epoch"] == best_epoch else row["val_metric"],
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
        writer = csv.DictWriter(fh, fieldnames=["epoch", "param_count", "nonzero_params"])
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
) -> TrainingRecord:
    torch = require_torch()
    device = choose_device()
    output_dir.mkdir(parents=True, exist_ok=True)
    model = model.to(device)

    if use_dendrites and model_key == "gcn" and hasattr(model, "conv2") and hasattr(model.conv2, "linear"):
        linear = model.conv2.linear
        if hasattr(linear, "set_this_output_dimensions"):
            linear.set_this_output_dimensions(torch.tensor([-1, 0], device=device))

    if use_pruning:
        try:
            import torch.nn.utils.prune as prune

            parameters_to_prune = [
                (module, "weight")
                for module in model.modules()
                if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d))
            ]
            if parameters_to_prune:
                prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=prune_amount)
                for module, _ in parameters_to_prune:
                    if hasattr(module, "weight_orig"):
                        prune.remove(module, "weight")
        except Exception:
            pass

    if bit_width is not None and bit_width < 32 and use_qat:
        model = _make_quantized_copy(model, bit_width, quantization_mode)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = _binary_or_multi_loss(model_key)
    history: list[dict[str, float]] = []
    best_metric = -math.inf if metric_direction == "maximize" else math.inf
    best_epoch = 0
    best_state = None
    start_time = time.perf_counter()
    run_label = f"{model_key} | {condition_key}"
    if max_epochs > 0:
        epoch_progress = tqdm(
            range(max_epochs),
            desc=run_label,
            unit="epoch",
            leave=True,
            dynamic_ncols=True,
        )

        for epoch in epoch_progress:
            model.train()
            running_loss = 0.0
            batch_progress = tqdm(
                bundle.train_loader,
                desc=f"{run_label} | epoch {epoch + 1}/{max_epochs}",
                unit="batch",
                leave=False,
                dynamic_ncols=True,
            )
            for batch_index, batch in enumerate(batch_progress, start=1):
                batch = tuple(item.to(device) for item in batch)
                optimizer.zero_grad()
                outputs, targets, *rest = _forward(model_key, model, batch)
                if model_key == "actor_critic":
                    loss = criterion(outputs[0], targets)
                else:
                    loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += float(loss.detach().item())
                average_loss = running_loss / batch_index
                batch_progress.set_postfix(loss=f"{average_loss:.4f}")
                if bit_width is not None and bit_width < 32 and use_qat:
                    _make_quantized_copy(model, bit_width, quantization_mode)
            batch_progress.close()
            model.eval()
            values = []
            with torch.no_grad():
                for batch in bundle.val_loader:
                    batch = tuple(item.to(device) for item in batch)
                    outputs, targets, *rest = _forward(model_key, model, batch)
                    metric_targets = rest[0] if rest else None
                    values.append(_collapse_metric(model_key, outputs, targets, metric_targets))
            val_metric = sum(values) / max(1, len(values))
            history.append({"epoch": epoch + 1, "val_metric": val_metric})
            if best_state is None or _metric_is_better(val_metric, best_metric, metric_direction):
                best_metric = val_metric
                best_epoch = epoch + 1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epoch_progress.set_postfix(
                val_metric=_format_metric_value(val_metric),
                best_metric=_format_metric_value(best_metric),
                best_epoch=best_epoch,
            )
        epoch_progress.close()
    else:
        print(f"{run_label}: evaluating existing checkpoint without additional training")

    if best_state is not None:
        model.load_state_dict(best_state)

    if bit_width is not None and bit_width < 32:
        model = _make_quantized_copy(model, bit_width, quantization_mode)

    test_values = []
    model.eval()
    with torch.no_grad():
        for batch in bundle.test_loader:
            batch = tuple(item.to(device) for item in batch)
            outputs, targets, *rest = _forward(model_key, model, batch)
            metric_targets = rest[0] if rest else None
            test_values.append(_collapse_metric(model_key, outputs, targets, metric_targets))
    final_metric = sum(test_values) / max(1, len(test_values))
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

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "model_key": model_key,
                "condition_key": condition_key,
                "display_name": display_name,
                "metric_name": metric_name,
                "metric_direction": metric_direction,
                "best_metric_value": best_metric,
                "metric_value": final_metric,
                "best_epoch": best_epoch,
                "param_count": param_count,
                "nonzero_params": nonzero_params,
                "file_size_mb": file_size_mb,
                "use_dendrites": use_dendrites,
                "use_pruning": use_pruning,
                "bit_width": bit_width,
                "use_qat": use_qat,
                "fine_tune_epochs": fine_tune_epochs,
                "artifact_path": str(artifact_path),
            },
            indent=2,
        )
    )
    with (output_dir / "history.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["epoch", "val_metric"])
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
    )
