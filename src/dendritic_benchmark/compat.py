from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
import builtins

# Module-level flag to ensure the PAI config-saved message is emitted only once
_PAI_CONFIG_SAVED_PRINTED = False
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from dotenv import load_dotenv
else:
    try:  # pragma: no cover - optional dependency
        from dotenv import load_dotenv
    except Exception:  # pragma: no cover - allow import before deps are installed
        load_dotenv = None


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


if TYPE_CHECKING:  # pragma: no cover
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
else:
    try:  # pragma: no cover - optional dependency
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except Exception:  # pragma: no cover - allow import on machines without torch
        torch = None
        nn = None
        F = None


def require_torch() -> Any:
    if torch is None:
        raise RuntimeError(
            "PyTorch is required for training. Install it in the uv-managed environment "
            "before running the benchmark."
        )
    return torch


def has_torchao() -> bool:
    return module_available("torchao")


def has_perforatedai() -> bool:
    return module_available("perforatedai")


def load_project_environment() -> None:
    if load_dotenv is None:
        return
    dotenv_path = Path(__file__).resolve().parents[2] / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path, override=False)


def _mirror_env_aliases() -> dict[str, str]:
    aliases = {
        "PERFORATEDAI_API_KEY": (
            "PERFORATEDAI_API_KEY",
            "PERFORATEDAI_TOKEN",
            "PERFORATEDBP_API_KEY",
            "PERFORATEDBP_TOKEN",
            "PAITOKEN",
        ),
        "PERFORATEDAI_EMAIL": ("PERFORATEDAI_EMAIL", "PERFORATEDBP_EMAIL", "PAIEMAIL"),
    }
    resolved: dict[str, str] = {}
    for canonical, names in aliases.items():
        value = next((os.getenv(name) for name in names if os.getenv(name)), None)
        if not value:
            continue
        resolved[canonical] = value
        for name in names:
            os.environ.setdefault(name, value)
    return resolved


def perforatedai_credentials_present() -> bool:
    return bool(_mirror_env_aliases())


@dataclass
class BackendStatus:
    torch_available: bool
    torchao_available: bool
    perforatedai_available: bool


def backend_status() -> BackendStatus:
    return BackendStatus(
        torch_available=torch is not None,
        torchao_available=has_torchao(),
        perforatedai_available=has_perforatedai(),
    )


def _configure_pai_trackers(
    GPA: Any,
    modules_to_track: list[Any] | None,
    module_names_to_track: list[str] | None,
    confirm_unwrapped_modules: bool,
) -> None:
    for setter_name in (
        "set_modules_to_track",
        "set_module_names_to_track",
        "set_module_ids_to_track",
        "set_modules_to_perforate",
        "set_module_names_to_perforate",
        "set_module_ids_to_perforate",
    ):
        setter = getattr(GPA.pc, setter_name, None)
        if setter is not None:
            setter([])
    if modules_to_track:
        GPA.pc.append_modules_to_track(modules_to_track)
    if module_names_to_track:
        GPA.pc.append_module_names_to_track(module_names_to_track)
    if hasattr(GPA.pc, "set_testing_dendrite_capacity"):
        GPA.pc.set_testing_dendrite_capacity(False)
    if confirm_unwrapped_modules:
        GPA.pc.set_unwrapped_modules_confirmed(True)


def perforate_model(
    model: Any,
    save_name: str,
    doing_pai: bool = True,
    maximizing_score: bool = True,
    modules_to_track: list[Any] | None = None,
    module_names_to_track: list[str] | None = None,
    confirm_unwrapped_modules: bool = True,
) -> Any:
    if not has_perforatedai():
        return model

    # Wrap builtins.print temporarily to filter repeated PerforatedAI config
    # messages that look like: "[PAI Config] Saved ...". Keep a module-level
    # flag so only the first such message is printed across the process.
    global _PAI_CONFIG_SAVED_PRINTED
    original_print = builtins.print

    def _filtered_print(*args: Any, **kwargs: Any) -> None:
        global _PAI_CONFIG_SAVED_PRINTED
        try:
            text = " ".join(str(a) for a in args)
        except Exception:
            return original_print(*args, **kwargs)
        if text.startswith("[PAI Config] Saved"):
            if _PAI_CONFIG_SAVED_PRINTED:
                return
            _PAI_CONFIG_SAVED_PRINTED = True
        return original_print(*args, **kwargs)

    try:  # pragma: no cover - optional dependency
        setattr(builtins, "print", _filtered_print)
        _mirror_env_aliases()
        GPA = importlib.import_module("perforatedai.globals_perforatedai")
        UPA = importlib.import_module("perforatedai.utils_perforatedai")

        _configure_pai_trackers(GPA, modules_to_track, module_names_to_track, confirm_unwrapped_modules)

        return UPA.perforate_model(
            model,
            doing_pai=doing_pai,
            save_name=save_name,
            maximizing_score=maximizing_score,
            making_graphs=False,
        )
    except Exception:
        return model
    finally:
        builtins.print = original_print


def choose_device() -> Any:
    if torch is None:
        return "cpu"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():  # pragma: no cover - CUDA not expected on Mac
        return torch.device("cuda")
    return torch.device("cpu")


def symmetric_quantize_tensor(tensor: Any, bit_width: int) -> Any:
    if torch is None:
        return tensor
    if bit_width >= 16:
        return tensor.clone()
    if bit_width <= 1:
        return tensor.sign().clamp(min=-1, max=1)
    levels = 2 ** bit_width - 1
    max_abs = tensor.abs().max()
    if max_abs == 0:
        return tensor.clone()
    scale = max_abs / (levels // 2)
    return torch.round(tensor / scale).clamp(-(levels // 2), levels // 2) * scale


def ternary_quantize_tensor(tensor: Any) -> Any:
    if torch is None:
        return tensor
    threshold = tensor.std(unbiased=False) * 0.5
    out = torch.zeros_like(tensor)
    out[tensor > threshold] = 1
    out[tensor < -threshold] = -1
    return out


def binary_quantize_tensor(tensor: Any) -> Any:
    if torch is None:
        return tensor
    return torch.where(tensor >= 0, torch.ones_like(tensor), -torch.ones_like(tensor))


def round_robin(value: int, modulo: int) -> int:
    return value % modulo if modulo else value


def clamp_float(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))
