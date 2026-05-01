from __future__ import annotations

import importlib.util
import os
import pdb
import shutil
import sys
from contextlib import contextmanager
from dataclasses import dataclass
import builtins

# Module-level flag to ensure the PAI config-saved message is emitted only once
_PAI_CONFIG_SAVED_PRINTED: bool = False
_PAI_DEBUGGER_SUPPRESS_REMAINING: int = 0
from pathlib import Path
from typing import Any, Callable, Iterator

load_dotenv: Any
try:  # pragma: no cover - optional dependency
    from dotenv import load_dotenv as _load_dotenv

    load_dotenv = _load_dotenv
except Exception:  # pragma: no cover - allow import before deps are installed
    load_dotenv = None


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


torch: Any
nn: Any
F: Any
try:  # pragma: no cover - optional dependency
    import torch as _torch
    import torch.nn as _nn
    import torch.nn.functional as _F

    torch = _torch
    nn = _nn
    F = _F
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
    gpa: Any,
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
        setter = getattr(gpa.pc, setter_name, None)
        if setter is not None:
            setter([])
    if modules_to_track:
        gpa.pc.append_modules_to_track(modules_to_track)
        append_modules_to_perforate = getattr(gpa.pc, "append_modules_to_perforate", None)
        if append_modules_to_perforate is not None:
            append_modules_to_perforate(modules_to_track)
    if module_names_to_track:
        gpa.pc.append_module_names_to_track(module_names_to_track)
        append_module_names_to_perforate = getattr(
            gpa.pc, "append_module_names_to_perforate", None
        )
        if append_module_names_to_perforate is not None:
            append_module_names_to_perforate(module_names_to_track)
    set_device = getattr(gpa.pc, "set_device", None)
    if set_device is not None:
        set_device(choose_device())
    if hasattr(gpa.pc, "set_testing_dendrite_capacity"):
        gpa.pc.set_testing_dendrite_capacity(False)
    if confirm_unwrapped_modules:
        gpa.pc.set_unwrapped_modules_confirmed(True)


def _consume_pai_config_message(text: str) -> bool:
    global _PAI_CONFIG_SAVED_PRINTED
    if not text.startswith("[PAI Config] Saved"):
        return False
    if _PAI_CONFIG_SAVED_PRINTED:
        return True
    _PAI_CONFIG_SAVED_PRINTED = True
    return False


def _print_pai_debugger_notice(text: str) -> None:
    stream = sys.__stderr__ or sys.stderr
    if stream is None:
        return
    stream.write(
        "\033[31m[PAI debugger suppressed] PerforatedAI attempted to print a "
        f"debugger warning: {text.strip()}\033[0m\n"
    )
    stream.flush()


def _consume_pai_debugger_message(text: str) -> bool:
    global _PAI_DEBUGGER_SUPPRESS_REMAINING
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.startswith("WARNING: Parameter does not have parameter_type attribute"):
        _print_pai_debugger_notice(stripped)
        _PAI_DEBUGGER_SUPPRESS_REMAINING = 8
        return True
    if stripped.startswith(
        (
            "(Pdb)",
            "--Call--",
            "You can find this param",
            "UPA.find_param_name_by_id(",
            "Ensure that model is either converted or tracked",
            "Instructions in customization.md",
        )
    ):
        _PAI_DEBUGGER_SUPPRESS_REMAINING = max(_PAI_DEBUGGER_SUPPRESS_REMAINING - 1, 0)
        return True
    if _PAI_DEBUGGER_SUPPRESS_REMAINING:
        if stripped.startswith((">", "->")):
            _PAI_DEBUGGER_SUPPRESS_REMAINING -= 1
            return True
        _PAI_DEBUGGER_SUPPRESS_REMAINING = 0
    return False


def _consume_pai_output_message(text: str) -> bool:
    return _consume_pai_config_message(text) or _consume_pai_debugger_message(text)


class _PaiConfigFilterStream:
    def __init__(self, stream: Any) -> None:
        self._stream = stream
        self._buffer = ""

    def write(self, data: str) -> Any:
        if not data:
            return self._stream.write(data)
        self._buffer += data
        written = 0
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if _consume_pai_output_message(line):
                continue
            written = self._stream.write(f"{line}\n")
        return written

    def flush(self) -> None:
        if self._buffer:
            if not _consume_pai_output_message(self._buffer):
                self._stream.write(self._buffer)
            self._buffer = ""
        self._stream.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


def _filtered_print_factory(original_print: Any) -> Any:
    def _filtered_print(*args: Any, **kwargs: Any) -> None:
        try:
            text = " ".join(str(a) for a in args)
        except Exception:
            return original_print(*args, **kwargs)
        if _consume_pai_output_message(text):
            return
        return original_print(*args, **kwargs)

    return _filtered_print


def _install_pai_output_filters() -> tuple[Any, Any, Any]:
    original_print = builtins.print
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    setattr(builtins, "print", _filtered_print_factory(original_print))
    sys.stdout = _PaiConfigFilterStream(original_stdout)
    sys.stderr = _PaiConfigFilterStream(original_stderr)
    return original_print, original_stdout, original_stderr


def _restore_pai_output_filters(
    original_print: Any,
    original_stdout: Any,
    original_stderr: Any,
) -> None:
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    builtins.print = original_print


@contextmanager
def _suppress_pai_debugger() -> Iterator[None]:
    """Prevent PerforatedAI library warnings from dropping benchmark runs into pdb."""

    original_set_trace: Callable[..., Any] = pdb.set_trace
    original_pdb_set_trace: Callable[..., Any] = pdb.Pdb.set_trace
    original_breakpointhook: Callable[..., Any] = sys.breakpointhook
    original_sys_settrace: Callable[..., Any] = sys.settrace

    def _no_set_trace(*args: Any, **kwargs: Any) -> None:
        _ = args, kwargs

    def _guarded_settrace(trace_function: Any) -> None:
        trace_owner = getattr(trace_function, "__self__", None)
        owner_module = getattr(type(trace_owner), "__module__", "")
        function_module = getattr(trace_function, "__module__", "")
        if owner_module in {"pdb", "bdb"} or function_module in {"pdb", "bdb"}:
            return
        original_sys_settrace(trace_function)

    setattr(pdb, "set_trace", _no_set_trace)
    setattr(pdb.Pdb, "set_trace", _no_set_trace)
    setattr(sys, "breakpointhook", _no_set_trace)
    setattr(sys, "settrace", _guarded_settrace)
    try:
        yield
    finally:
        setattr(pdb, "set_trace", original_set_trace)
        setattr(pdb.Pdb, "set_trace", original_pdb_set_trace)
        setattr(sys, "breakpointhook", original_breakpointhook)
        setattr(sys, "settrace", original_sys_settrace)


@contextmanager
def pai_runtime_guard() -> Iterator[None]:
    original_print, original_stdout, original_stderr = _install_pai_output_filters()
    try:
        with _suppress_pai_debugger():
            yield
    finally:
        _restore_pai_output_filters(original_print, original_stdout, original_stderr)


def perforate_model(
    model: Any,
    save_name: str,
    doing_pai: bool = True,
    maximizing_score: bool = True,
    modules_to_track: list[Any] | None = None,
    module_names_to_track: list[str] | None = None,
    confirm_unwrapped_modules: bool = True,
    config_snapshot_path: Path | str | None = None,
    use_runtime_guard: bool = False,
) -> Any:
    if not has_perforatedai():
        return model

    try:  # pragma: no cover - optional dependency
        _mirror_env_aliases()
        GPA = importlib.import_module("perforatedai.globals_perforatedai")
        UPA = importlib.import_module("perforatedai.utils_perforatedai")
        upa_perforate_model = getattr(UPA, "perforate_model")

        def _run_perforation() -> Any:
            _configure_pai_trackers(
                GPA, modules_to_track, module_names_to_track, confirm_unwrapped_modules
            )
            pai_save_name = str(_pai_save_path(save_name))
            return upa_perforate_model(
                model,
                doing_pai=doing_pai,
                save_name=pai_save_name,
                maximizing_score=maximizing_score,
                making_graphs=False,
            )

        if use_runtime_guard:
            with pai_runtime_guard():
                perforated_model = _run_perforation()
        else:
            perforated_model = _run_perforation()
        _snapshot_pai_config(_snapshot_stem(save_name), config_snapshot_path)
        return perforated_model
    except Exception:
        return model


def _pai_save_path(save_name: str) -> Path:
    path = Path(save_name)
    if path.is_absolute():
        return Path("PAI") / path.name
    if path.parts and path.parts[0] == "PAI":
        return path
    return Path("PAI") / path


def _snapshot_stem(save_name: str) -> str:
    return "_".join(Path(save_name).parts)


def _snapshot_pai_config(save_name: str, config_snapshot_path: Path | str | None) -> None:
    config_path = Path("PAI") / "PAI_config.json"
    if not config_path.exists():
        return
    named_snapshot = config_path.with_name(f"{save_name}_PAI_config.json")
    try:
        shutil.copy2(config_path, named_snapshot)
        if config_snapshot_path is not None:
            artifact_path = Path(config_snapshot_path)
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(config_path, artifact_path)
    except Exception:
        return


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
