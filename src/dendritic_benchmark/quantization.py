"""Quantization policy and QAT shadow-state management.

This module owns the rule that a trained model is projected exactly once for
evaluation.  It deliberately knows nothing about epochs, metrics, artifacts,
or PAI lifecycle state, which keeps quantization changes independently
testable from the training orchestrator.
"""

from typing import Any, Literal, Protocol

import torch

from .compat import (
    binary_quantize_tensor,
    symmetric_quantize_tensor,
    ternary_quantize_tensor,
    ternary_quantize_tensor_per_channel,
)

QuantizationGranularity = Literal["tensor", "channel"]

# Identifies the weight-only projection rule itself (symmetric tensor/channel
# scaling, the Q2 robust-scale correction, ternary/binary scale metadata) --
# distinct from training.QUANTIZATION_EVALUATION_REVISION, which gates when a
# *result* is reportable. This lets a future activation-calibrated quantizer
# get its own revision without retroactively invalidating every existing
# weight-only artifact. See information/optimization/00_assessment.md point 5.
QUANTIZER_REVISION = "weight_only_symmetric_v1"


class QuantizationConfig(Protocol):
    bit_width: int | None
    quantization_mode: str | None
    quantization_granularity: QuantizationGranularity
    use_qat: bool


def should_quantize_for_training(config: QuantizationConfig) -> bool:
    bit_width = config.bit_width
    return bit_width is not None and bit_width < 32 and config.use_qat


def should_quantize_for_eval(config: QuantizationConfig) -> bool:
    bit_width = config.bit_width
    return bit_width is not None and bit_width < 32


def quantize_tensor(
    tensor: Any,
    bit_width: int,
    mode: str | None,
    granularity: QuantizationGranularity = "tensor",
) -> Any:
    """Project one tensor on CPU using the configured quantization grid."""
    cpu_tensor = tensor.detach().cpu()
    if mode == "binary" or bit_width == 1:
        return binary_quantize_tensor(cpu_tensor)
    if mode == "ternary":
        if granularity == "channel":
            return ternary_quantize_tensor_per_channel(cpu_tensor)
        return ternary_quantize_tensor(cpu_tensor)
    return symmetric_quantize_tensor(cpu_tensor, bit_width)


def make_quantized_copy(
    model: Any,
    bit_width: int | None,
    mode: str | None = None,
    granularity: QuantizationGranularity = "tensor",
) -> Any:
    """Project model parameters in place and return the model for composition."""
    if bit_width is None or bit_width >= 32:
        return model
    with torch.no_grad():
        for param in model.parameters():
            if param.numel() == 0:
                continue
            quantized = quantize_tensor(param, bit_width, mode, granularity)
            param.copy_(quantized.to(param.device))
    return model


_QAT_SHADOW_ATTR = "_pai_qat_shadow"


def qat_init_shadow(model: Any) -> None:
    """Snapshot full-precision values before QAT starts projecting weights."""
    for param in model.parameters():
        if param.numel() != 0:
            setattr(param, _QAT_SHADOW_ATTR, param.detach().clone())


def _qat_shadow_for(param: Any) -> Any:
    shadow = getattr(param, _QAT_SHADOW_ATTR, None)
    if shadow is None:
        shadow = param.detach().clone()
        setattr(param, _QAT_SHADOW_ATTR, shadow)
    return shadow


def qat_project_for_forward(model: Any, config: QuantizationConfig) -> None:
    """Put projected weights in ``.data`` for a deployment-realistic pass."""
    if not should_quantize_for_training(config):
        return
    bit_width = config.bit_width
    assert bit_width is not None
    with torch.no_grad():
        for param in model.parameters():
            if param.numel() == 0:
                continue
            quantized = quantize_tensor(
                _qat_shadow_for(param),
                bit_width,
                config.quantization_mode,
                config.quantization_granularity,
            )
            param.data.copy_(quantized.to(param.device))


def qat_restore_shadow_for_step(model: Any, config: QuantizationConfig) -> None:
    """Restore continuous weights immediately before the optimizer step."""
    if not should_quantize_for_training(config):
        return
    with torch.no_grad():
        for param in model.parameters():
            if param.numel() != 0:
                param.data.copy_(_qat_shadow_for(param))


def qat_sync_shadow_after_step(model: Any, config: QuantizationConfig) -> None:
    """Persist the optimizer update into each parameter's continuous shadow."""
    if not should_quantize_for_training(config):
        return
    with torch.no_grad():
        for param in model.parameters():
            if param.numel() != 0:
                _qat_shadow_for(param).copy_(param.data)
