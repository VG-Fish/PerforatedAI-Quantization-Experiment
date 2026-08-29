#!/usr/bin/env python3
"""Dense controls matched to Dynamic12 GRU's retained PAI decoder capacity."""

from __future__ import annotations

import argparse
import math
import time

import torch

from dendritic_benchmark.compat import choose_device
from dendritic_benchmark.data import build_task_bundle
from dendritic_benchmark.models import GRUForecaster


def evaluate(model: torch.nn.Module, loader: object, device: torch.device, scale: float, offset: float) -> float:
    model.eval()
    total_error = 0.0
    examples = 0
    with torch.no_grad():
        for inputs, targets in loader:  # type: ignore[union-attr]
            inputs, targets = inputs.to(device), targets.to(device)
            predictions = model(inputs)
            total_error += (
                ((predictions * scale + offset) - (targets * scale + offset)).abs().mean().item()
                * inputs.size(0)
            )
            examples += inputs.size(0)
    return total_error / max(1, examples)


def run_arm(width: int, seed: int, epochs: int, bundle: object, device: torch.device) -> tuple[int, float, float]:
    torch.manual_seed(seed)
    model = GRUForecaster(hidden=48, decoder_hidden=width).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3.0e-4)
    criterion = torch.nn.SmoothL1Loss(beta=0.1)
    scale = float(getattr(bundle, "target_scale", 1.0))
    offset = float(getattr(bundle, "target_offset", 0.0))
    best_val, selected_test = math.inf, math.inf
    for epoch in range(epochs):
        learning_rate = 3.0e-4 * (0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * epoch / epochs)))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        model.train()
        for inputs, targets in bundle.train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs.to(device)), targets.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        validation = evaluate(model, bundle.val_loader, device, scale, offset)
        if validation < best_val:
            best_val = validation
            selected_test = evaluate(model, bundle.test_loader, device, scale, offset)
    return sum(parameter.numel() for parameter in model.parameters()), best_val, selected_test


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--decoder-widths", type=int, nargs="+", default=[48, 51])
    args = parser.parse_args()
    device = choose_device()
    bundle = build_task_bundle("gru_forecaster", batch_size=128)
    for width in args.decoder_widths:
        started = time.monotonic()
        results = [run_arm(width, seed, args.epochs, bundle, device) for seed in args.seeds]
        tests = torch.tensor([result[2] for result in results])
        print(
            f"decoder={width:3d} params={results[0][0]:7d} test MAE "
            f"{tests.mean():.4f} +/- {tests.std(unbiased=len(tests) > 1):.4f} "
            f"[{time.monotonic() - started:.0f}s]",
            flush=True,
        )


if __name__ == "__main__":
    main()
