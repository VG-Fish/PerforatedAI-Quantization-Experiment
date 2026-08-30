#!/usr/bin/env python3
"""Dense VAE controls matching Dynamic12's 1.07M-parameter PAI artifact."""

from __future__ import annotations

import argparse
import math
import time
from typing import Any

import torch

from dendritic_benchmark.compat import choose_device
from dendritic_benchmark.data import TaskBundle, build_task_bundle
from dendritic_benchmark.models import build_model
from dendritic_benchmark.training import _compute_loss


def evaluate(model: torch.nn.Module, loader: Any, device: torch.device) -> float:
    model.eval()
    total_loss, examples = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            inputs = batch[0].to(device)
            outputs = model(inputs)
            total_loss += _compute_loss("vae_mnist", None, outputs, inputs).item() * inputs.size(0)
            examples += inputs.size(0)
    if examples == 0:
        return -total_loss
    return -total_loss / examples


def run_arm(
    scale: float, seed: int, epochs: int, bundle: TaskBundle, device: torch.device
) -> tuple[int, float, float]:
    torch.manual_seed(seed)
    model = build_model("vae_mnist", model_scale=scale).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    best_val, selected_test = -math.inf, -math.inf
    for epoch in range(epochs):
        learning_rate = 1.0e-3 * (0.02 + 0.98 * 0.5 * (1.0 + math.cos(math.pi * epoch / epochs)))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        model.train()
        for batch in bundle.train_loader:
            inputs = batch[0].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = _compute_loss("vae_mnist", None, model(inputs), inputs)
            loss.backward()
            optimizer.step()
        validation = evaluate(model, bundle.val_loader, device)
        if validation > best_val:
            best_val = validation
            selected_test = evaluate(model, bundle.test_loader, device)
    return sum(parameter.numel() for parameter in model.parameters()), best_val, selected_test


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--model-scales", type=float, nargs="+", default=[0.75, 1.0])
    args = parser.parse_args()
    device = choose_device()
    bundle = build_task_bundle("vae_mnist", batch_size=128)
    for scale in args.model_scales:
        started = time.monotonic()
        results = [run_arm(scale, seed, args.epochs, bundle, device) for seed in args.seeds]
        tests = torch.tensor([result[2] for result in results])
        print(
            f"scale={scale:.2f} params={results[0][0]:7d} test ELBO "
            f"{tests.mean():.3f} +/- {tests.std(unbiased=len(tests) > 1):.3f} "
            f"[{time.monotonic() - started:.0f}s]",
            flush=True,
        )


if __name__ == "__main__":
    main()
