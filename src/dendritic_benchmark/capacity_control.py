"""Topology-matched, ordinary-PyTorch controls for retained PAI branches.

The extractor intentionally works from a saved PAI ``state_dict`` instead of
vendor implementation classes.  This keeps a completed experiment readable
after a PerforatedAI upgrade and, more importantly, makes every supported
control auditable from its artifact alone.  We currently support the branch
form PAI uses for the ResNet first protocol: one retained copy of a Linear or
Conv module and one elementwise output mixing tensor.  Other layouts raise
``UnsupportedTopology``; callers must record that status rather than widen a
model as a substitute control.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn


class UnsupportedTopology(RuntimeError):
    """The saved PAI branch cannot be represented exactly by this control."""


@dataclass(frozen=True)
class DenseBranchSpec:
    target_path: str
    kind: str
    weight_shape: tuple[int, ...]
    bias_shape: tuple[int, ...] | None
    mixing_shape: tuple[int, ...]
    branch_count: int

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key in ("weight_shape", "bias_shape", "mixing_shape"):
            if value[key] is not None:
                value[key] = list(value[key])
        return value


@dataclass(frozen=True)
class RetainedTopologySpec:
    branches: tuple[DenseBranchSpec, ...]
    source_parameter_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "branches": [branch.to_dict() for branch in self.branches],
            "source_parameter_count": self.source_parameter_count,
        }

    @property
    def sha256(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


def _tensor_state(state: dict[str, Any], key: str) -> torch.Tensor:
    value = state.get(key)
    if not isinstance(value, torch.Tensor):
        raise UnsupportedTopology(f"missing tensor {key!r} in saved PAI topology")
    return value


def retained_topology_from_state_dict(state: dict[str, Any]) -> RetainedTopologySpec:
    """Extract retained one-branch PAI modules from a final model state.

    PAI saves a retained copied module at
    ``<path>.dendrite_module.layers.0`` and its output mixer at
    ``<path>.dendrites_to_top.0``.  Multiple retained copies, candidate-only
    graphs, or non-elementwise mixers are deliberately unsupported: treating
    them as a nearby dense architecture would violate the protocol.
    """
    marker = ".dendrite_module.layers.0.weight"
    paths = sorted(key[: -len(marker)] for key in state if key.endswith(marker))
    if not paths:
        raise UnsupportedTopology("no retained PAI branch tensors were found")
    branches: list[DenseBranchSpec] = []
    for path in paths:
        prefix = f"{path}.dendrite_module.layers."
        layer_indices = sorted(
            int(key[len(prefix) :].split(".", 1)[0])
            for key in state
            if key.startswith(prefix) and key.endswith(".weight")
        )
        if layer_indices != [0]:
            raise UnsupportedTopology(
                f"{path}: expected one retained branch, found layers {layer_indices}"
            )
        weight = _tensor_state(state, f"{prefix}0.weight")
        bias = state.get(f"{prefix}0.bias")
        if bias is not None and not isinstance(bias, torch.Tensor):
            raise UnsupportedTopology(f"{path}: branch bias is not a tensor")
        mix = _tensor_state(state, f"{path}.dendrites_to_top.0")
        if mix.ndim < 1 or mix.shape[-1] != weight.shape[0]:
            raise UnsupportedTopology(
                f"{path}: mixing tensor {tuple(mix.shape)} does not match output "
                f"dimension {weight.shape[0]}"
            )
        # The first protocol is ResNet's `.pre_fc` Linear.  Conv branches need
        # PAI's spatial mixer semantics, which are not inferred safely from a
        # state dict, so reject them rather than approximate them.
        if weight.ndim != 2:
            raise UnsupportedTopology(f"{path}: only Linear branches are currently supported")
        branches.append(
            DenseBranchSpec(
                target_path=path,
                kind="linear",
                weight_shape=tuple(weight.shape),
                bias_shape=tuple(bias.shape) if isinstance(bias, torch.Tensor) else None,
                mixing_shape=tuple(mix.shape),
                branch_count=1,
            )
        )
    return RetainedTopologySpec(
        branches=tuple(branches),
        source_parameter_count=sum(value.numel() for value in state.values() if isinstance(value, torch.Tensor)),
    )


def dense_backbone_state_from_pai(state: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Map PAI parent-module names back to an unwrapped dense state dict."""
    dense: dict[str, torch.Tensor] = {}
    for name, value in state.items():
        if not isinstance(value, torch.Tensor) or ".dendrite_module.parent_module." not in name:
            continue
        target, suffix = name.split(".dendrite_module.parent_module.", 1)
        dense[f"{target}.{suffix}"] = value.detach().clone()
    return dense


def _module_at(root: nn.Module, path: str) -> tuple[nn.Module, str, nn.Module]:
    parts = path.lstrip(".").split(".")
    parent: nn.Module = root
    for part in parts[:-1]:
        parent = cast(Any, parent)[int(part)] if part.isdigit() else getattr(parent, part)
    leaf = parts[-1]
    child = cast(Any, parent)[int(leaf)] if leaf.isdigit() else getattr(parent, leaf)
    return parent, leaf, child


def _replace(parent: nn.Module, name: str, replacement: nn.Module) -> None:
    if name.isdigit():
        cast(Any, parent)[int(name)] = replacement
    else:
        setattr(parent, name, replacement)


class _DenseParallelBranch(nn.Module):
    """A normal residual branch with PAI's one-branch output mixer shape."""

    def __init__(self, parent: nn.Module, branch: nn.Module, mixing_shape: tuple[int, ...]):
        super().__init__()
        self.parent = parent
        self.branch = branch
        # PAI's output mixer is trainable capacity, not runtime bookkeeping.
        self.mixing = nn.Parameter(torch.empty(mixing_shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parent = self.parent(x)
        branch = self.branch(x)
        # The supported PAI topology has one retained branch.  Its mixer is
        # elementwise on the output axis; leading singleton axes broadcast.
        return parent + branch * self.mixing


def _ordinary_branch(parent: nn.Module, spec: DenseBranchSpec) -> nn.Module:
    if spec.kind == "linear" and isinstance(parent, nn.Linear):
        return nn.Linear(parent.in_features, parent.out_features, bias=parent.bias is not None)
    raise UnsupportedTopology(
        f"{spec.target_path}: saved {spec.kind} branch does not match dense "
        f"module {type(parent).__name__}"
    )


def apply_capacity_dense_control(
    model: nn.Module,
    topology: RetainedTopologySpec,
    *,
    seed: int | None,
) -> nn.Module:
    """Add ordinary branches and verify exact trainable parameter equality."""
    base_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    with torch.random.fork_rng():
        if seed is not None:
            torch.manual_seed(seed)
        for spec in topology.branches:
            parent, name, original = _module_at(model, spec.target_path)
            branch = _ordinary_branch(original, spec)
            wrapper = _DenseParallelBranch(original, branch, spec.mixing_shape)
            nn.init.normal_(wrapper.mixing, mean=0.0, std=0.005)
            _replace(parent, name, wrapper)
    added = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad) - base_count
    expected = sum(
        int(torch.tensor(branch.weight_shape).prod())
        + (int(torch.tensor(branch.bias_shape).prod()) if branch.bias_shape else 0)
        + int(torch.tensor(branch.mixing_shape).prod())
        for branch in topology.branches
    )
    if added != expected:
        raise UnsupportedTopology(f"ordinary branch count {added} != extracted count {expected}")
    return model


def save_topology_spec(path: Path, topology: RetainedTopologySpec) -> None:
    path.write_text(json.dumps({**topology.to_dict(), "sha256": topology.sha256}, indent=2))
