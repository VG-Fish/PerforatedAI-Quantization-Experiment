import builtins
import importlib
import math
import os
import pdb
import random
import re
import shutil
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import torch
from dotenv import dotenv_values

# Module-level flag to ensure the PAI config-saved message is emitted only once
_PAI_CONFIG_SAVED_PRINTED: bool = False
_PAI_DEBUGGER_SUPPRESS_REMAINING: int = 0
_PAI_GLOBALS_MODULE = "perforatedai.globals_perforatedai"
MODULE_OUTPUT_DIMENSIONS_ATTR = "_dqb_module_output_dimensions"
# PerforatedAI >= 3.2.3 rejects save names containing "/" and writes every
# artifact to "{save_name}/" relative to the process working directory, so the
# benchmark's PAI/{save_name}/ layout is reached by moving into PAI/ for the
# duration of each library call instead of by nesting the save name.
PAI_DIRECTORY_NAME = "PAI"
_PAI_ROOT = Path(PAI_DIRECTORY_NAME).resolve()
_PAI_WORKING_DIRECTORY_DEPTH = 0
# Name of the PAI system snapshot the benchmark writes alongside its own epoch
# checkpoint. Distinct from PAI's internal "latest", which is written mid-way
# through add_validation_score and so can disagree with the epoch checkpoint.
PAI_RESUME_NAME = "dqb_resume"
# Name of the PAI system snapshot taken at the *exact* moment the contents of
# model.pt are decided -- after the best-epoch restore decision and before any
# quantization. PAI_RESUME_NAME above is written inside the epoch loop and so
# can still describe a different dendrite structure than the artifact that
# training finally persists; downstream quantized conditions rebuild their
# skeleton from this one instead. See MEASUREMENT_CAVEATS.md #5.
PAI_ARTIFACT_NAME = "dqb_artifact"
_PAI_CONFIG_LIST_SETTERS: tuple[str, ...] = (
    "set_modules_to_track",
    "set_module_names_to_track",
    "set_module_ids_to_track",
    "set_modules_to_perforate",
    "set_module_names_to_perforate",
    "set_module_ids_to_perforate",
    # Clear persisted processing config so stale runs don't affect new ones.
    "set_modules_with_processing",
    "set_modules_processing_classes",
    "set_module_names_with_processing",
    "set_module_by_name_processing_classes",
    "set_module_names_to_not_save",
    # Cleared with the rest: parameter ids are per-model, so one model's entries
    # must not survive into the next model's perforation in the same process.
    "set_parameter_ids_to_track",
)


def load_project_environment() -> dict[str, str]:
    """Return the ``.env`` file in the current working directory as a dict.

    The current working directory is where ``uv run dqb ...`` was invoked.
    Returns an empty dict if no ``.env`` is present.
    """
    dotenv_path = Path.cwd() / ".env"
    if not dotenv_path.exists():
        return {}
    return {
        key: value
        for key, value in dotenv_values(dotenv_path).items()
        if value is not None
    }


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


@dataclass(frozen=True)
class PAIModuleSelection:
    modules_to_perforate: list[Any] | None = None
    module_names_to_perforate: list[str] | None = None
    module_ids_to_perforate: list[str] | None = None
    track_only_module_ids: list[str] | None = None
    module_names_to_not_save: list[str] | None = None
    # Fully-qualified parameter names, as `model.named_parameters()` reports
    # them and with no leading dot ("conv1.bias"). PAI assigns every parameter a
    # parameter_type from the module that owns it, so a parameter held directly
    # by a module whose *child* is perforated belongs to neither list and gets
    # "Parameter does not have parameter_type attribute" on every p-phase step.
    # Tracking the owning module is not the fix there — that module must stay
    # unwrapped precisely because its child is perforated — so PAI's own remedy
    # is to name the parameter. See GraphConv, whose bias sits beside a
    # perforated Linear.
    parameter_ids_to_track: list[str] | None = None


@dataclass(frozen=True)
class PAIRuntimeOptions:
    use_runtime_guard: bool = False
    no_backward_workaround: bool = False
    candidate_graph_enabled: bool = True
    initial_correlation_batches_limit: int | None = None
    fixed_switch_interval: int | None = None


def _call_if_available(target: Any, method_name: str, *args: Any) -> None:
    method = getattr(target, method_name, None)
    if method is not None:
        method(*args)


def _clear_pai_tracker_lists(pc: Any) -> None:
    for setter_name in _PAI_CONFIG_LIST_SETTERS:
        _call_if_available(pc, setter_name, [])


def _append_if_configured(pc: Any, method_name: str, value: Any) -> None:
    if value:
        _call_if_available(pc, method_name, value)


def _append_pai_module_selection(pc: Any, selection: PAIModuleSelection) -> None:
    # In PerforatedAI 3.2, entries in *_to_track become tracked-only wrappers.
    # Dendrite insertion requires PAINeuronModule wrappers, so benchmark-selected
    # modules must be registered only with the perforation lists.
    _append_if_configured(
        pc, "append_modules_to_perforate", selection.modules_to_perforate
    )
    _append_if_configured(
        pc,
        "append_module_names_to_perforate",
        selection.module_names_to_perforate,
    )
    _append_if_configured(
        pc,
        "append_module_ids_to_perforate",
        selection.module_ids_to_perforate,
    )
    _append_if_configured(
        pc, "append_module_ids_to_track", selection.track_only_module_ids
    )
    _append_if_configured(
        pc, "append_module_names_to_not_save", selection.module_names_to_not_save
    )
    _append_if_configured(
        pc, "append_parameter_ids_to_track", selection.parameter_ids_to_track
    )


def configure_pai_candidate_graph(candidate_graph_enabled: bool) -> None:
    try:
        gpa = importlib.import_module(_PAI_GLOBALS_MODULE)
    except Exception:
        return
    _call_if_available(
        getattr(gpa, "pc", None),
        "set_candidate_graph_mode",
        candidate_graph_enabled,
    )


def _configure_pai_trackers(
    gpa: Any,
    module_selection: PAIModuleSelection | None,
    confirm_unwrapped_modules: bool,
    no_backward_workaround: bool = False,
    candidate_graph_enabled: bool = True,
) -> None:
    selection = module_selection or PAIModuleSelection()
    pc = gpa.pc
    _clear_pai_tracker_lists(pc)
    _append_pai_module_selection(pc, selection)
    _call_if_available(pc, "set_device", choose_device())
    _call_if_available(pc, "set_testing_dendrite_capacity", False)
    _call_if_available(pc, "set_debugging_memory_leak", False)
    _call_if_available(pc, "set_candidate_graph_mode", candidate_graph_enabled)
    _call_if_available(pc, "set_dashboard_events_enabled", True)
    if confirm_unwrapped_modules:
        _call_if_available(pc, "set_unwrapped_modules_confirmed", True)
    _call_if_available(pc, "set_no_backward_workaround", no_backward_workaround)


def _bounded_dendrite_schedule(
    max_epochs: int,
    freeze_fraction: float,
) -> tuple[int, int, int, int]:
    freeze_epochs = 0
    if max_epochs > 1 and freeze_fraction > 0:
        freeze_epochs = max(
            1, min(max_epochs - 1, math.ceil(max_epochs * freeze_fraction))
        )
    active_epochs = max(1, max_epochs - freeze_epochs)
    target_switches = max(1, min(4, active_epochs // 4))
    switch_interval = max(1, active_epochs // target_switches)
    p_epochs = max(1, min(2, switch_interval // 2))
    return active_epochs, target_switches, switch_interval, p_epochs


def _call_pai_setter(pc: Any, setter_name: str, value: Any) -> None:
    setter = getattr(pc, setter_name, None)
    if setter is not None:
        setter(value)


def _set_pai_switch_mode(pc: Any, mode_name: str) -> None:
    mode = getattr(pc, mode_name, None)
    if mode is not None:
        _call_pai_setter(pc, "set_switch_mode", mode)


def _apply_pai_schedule_values(pc: Any, values: dict[str, Any]) -> None:
    for setter_name, value in values.items():
        _call_pai_setter(pc, setter_name, value)


def _initial_correlation_batches(
    batches_per_epoch: int | None,
    initial_correlation_batches_limit: int | None,
) -> int | None:
    if batches_per_epoch is None:
        return None
    correlation_batches = max(1, batches_per_epoch - 1)
    if initial_correlation_batches_limit is not None:
        correlation_batches = min(
            correlation_batches, max(1, initial_correlation_batches_limit)
        )
    return correlation_batches


def _configure_interval_pai_schedule(pc: Any, *, switch_interval: int) -> None:
    """Switch every ``switch_interval`` epochs instead of on a detected plateau.

    HISTORY mode compares a running average (an EMA over ``history_lookback``
    epochs) that starts at 0 and only ever climbs toward the current score. Even
    against a bit-for-bit frozen score it keeps clearing the relative
    ``improvement_threshold`` every few epochs, so ``epoch_last_improved`` is
    refreshed continuously and ``n_epochs_to_switch`` never counts down. The
    2026-07-29 dynamic run spent 40 epochs and 15 h on DistilBERT that way with
    ``num_cycles`` still 0 and an empty ``switch_epochs.csv``.

    That trap is only affordable to wait out when epochs are cheap. For models
    where they are not, switch on a fixed interval and accept a schedule that
    ignores the plateau rather than one that never fires.
    """
    _set_pai_switch_mode(pc, "DOING_FIXED_SWITCH")
    _apply_pai_schedule_values(
        pc,
        {
            "set_first_fixed_switch_num": switch_interval,
            "set_fixed_switch_num": switch_interval,
            "set_n_epochs_to_switch": switch_interval,
        },
    )


def _configure_dynamic_pai_schedule(
    pc: Any,
    batches_per_epoch: int | None = None,
    initial_correlation_batches_limit: int | None = None,
    fixed_switch_interval: int | None = None,
) -> None:
    if fixed_switch_interval is not None:
        _configure_interval_pai_schedule(pc, switch_interval=fixed_switch_interval)
    else:
        _set_pai_switch_mode(pc, "DOING_HISTORY")
        _apply_pai_schedule_values(
            pc,
            {
                "set_n_epochs_to_switch": 10,
                # PAI names the plateau-detection window "history_lookback"; the
                # default of 1 switches on transient noise.
                "set_history_lookback": 8,
                # Indexed by dendrites added (globals_perforatedai getter_val), so
                # this needs max_dendrites + 1 entries. The final entry must stay
                # above zero: at a threshold of 0 only improvement_threshold_raw
                # (1e-5) separates a real gain from validation jitter, and the
                # plateau detector never sees n_epochs_to_switch quiet epochs.
                "set_improvement_threshold": [0.005, 0.002, 0.001, 0.001],
            },
        )
    _apply_pai_schedule_values(
        pc,
        {
            "set_p_epochs_to_switch": 2,
            # Dendrites stopped paying for themselves well before the sixth on
            # every model measured so far, and each extra one costs ~100 epochs
            # at a steadily worse seconds-per-epoch.
            "set_max_dendrites": 3,
            # Zeroing the best score on switch also zeroes running_accuracy,
            # which is an EMA over history_lookback epochs. Climbing back from
            # 0 to a ~0.99 metric takes ~70 epochs, and every one of them
            # registers as an improvement, so epoch_last_improved is refreshed
            # continuously and the switch trigger cannot fire.
            "set_reset_best_score_on_switch": False,
            "set_candidate_weight_initialization_multiplier": 0.005,
            # Not set here: pai_improvement_threshold / _raw, which gate how much
            # a node's correlation must gain in one epoch to keep the dendrite
            # phase alive.  Raising them from the (0.1, 1e-4) defaults to
            # (0.2, 1e-3) was measured to change nothing — the dendrite-phase
            # patience counter still reset on 91 of 92 switch checks — so the
            # override was dropped rather than left as unexplained config.
            # max_dendrite_phase_epochs in training.py is what actually bounds
            # the phase.
        },
    )
    correlation_batches = _initial_correlation_batches(
        batches_per_epoch, initial_correlation_batches_limit
    )
    if correlation_batches is not None:
        _call_pai_setter(pc, "set_initial_correlation_batches", correlation_batches)


def _configure_bounded_pai_schedule(
    pc: Any,
    *,
    max_epochs: int,
    freeze_fraction: float,
    batches_per_epoch: int | None = None,
    initial_correlation_batches_limit: int | None = None,
) -> None:
    _, target_switches, switch_interval, p_epochs = _bounded_dendrite_schedule(
        max_epochs, freeze_fraction
    )
    _set_pai_switch_mode(pc, "DOING_FIXED_SWITCH")
    _apply_pai_schedule_values(
        pc,
        {
            "set_first_fixed_switch_num": switch_interval,
            "set_fixed_switch_num": switch_interval,
            "set_n_epochs_to_switch": switch_interval,
            "set_p_epochs_to_switch": p_epochs,
            "set_max_dendrites": target_switches,
        },
    )
    correlation_batches = _initial_correlation_batches(
        batches_per_epoch, initial_correlation_batches_limit
    )
    if correlation_batches is not None:
        _call_pai_setter(pc, "set_initial_correlation_batches", correlation_batches)


def _configure_pai_training_schedule(
    gpa: Any,
    *,
    max_epochs: int,
    dynamic_dendritic_training: bool,
    freeze_fraction: float,
    batches_per_epoch: int | None = None,
    initial_correlation_batches_limit: int | None = None,
    fixed_switch_interval: int | None = None,
) -> None:
    pc = gpa.pc
    if dynamic_dendritic_training:
        _configure_dynamic_pai_schedule(
            pc,
            batches_per_epoch=batches_per_epoch,
            initial_correlation_batches_limit=initial_correlation_batches_limit,
            fixed_switch_interval=fixed_switch_interval,
        )
        return

    _configure_bounded_pai_schedule(
        pc,
        max_epochs=max_epochs,
        freeze_fraction=freeze_fraction,
        batches_per_epoch=batches_per_epoch,
        initial_correlation_batches_limit=initial_correlation_batches_limit,
    )


def set_module_output_dimensions(
    model: Any,
    module_dimensions: dict[str, list[int]],
    *,
    device: Any | None = None,
) -> None:
    named_modules = getattr(model, "named_modules", None)
    if named_modules is None:
        return
    modules = dict(named_modules())
    for module_name, dimensions in module_dimensions.items():
        module = modules.get(module_name.lstrip("."))
        if module is None:
            continue
        setter = getattr(module, "set_this_output_dimensions", None)
        if setter is None:
            continue
        value: Any = dimensions
        if device is not None:
            value = torch.tensor(dimensions, device=device)
        setter(value)


def attach_module_output_dimensions(
    model: Any,
    module_dimensions: dict[str, list[int]] | None,
) -> Any:
    if module_dimensions:
        setattr(
            model,
            MODULE_OUTPUT_DIMENSIONS_ATTR,
            {name: list(dimensions) for name, dimensions in module_dimensions.items()},
        )
    return model


def _zero_grad_if_available(target: Any) -> None:
    zero_grad = getattr(target, "zero_grad", None)
    if zero_grad is None:
        return
    try:
        zero_grad(set_to_none=True)
    except TypeError:
        zero_grad()


def _call_noarg_if_available(target: Any, method_name: str) -> bool:
    method = getattr(target, method_name, None)
    if method is None:
        return False
    try:
        method()
        return True
    except Exception:
        return False


def _clear_pai_tracker_buffers(tracker: Any) -> None:
    if tracker is None:
        return
    if isinstance(tracker, (list, tuple, set)):
        for item in tracker:
            _clear_pai_tracker_buffers(item)
        return
    if _call_noarg_if_available(tracker, "clear_all_processors"):
        return
    _call_noarg_if_available(tracker, "clear_processors")


def _clear_pai_tracker_state(tracker: Any) -> None:
    if tracker is None:
        return
    if isinstance(tracker, (list, tuple, set)):
        for item in tracker:
            _clear_pai_tracker_state(item)
        return
    _clear_pai_tracker_buffers(tracker)
    if hasattr(tracker, "add_validation_score"):
        _call_noarg_if_available(tracker, "clear")


def clear_pai_processor_buffers(model: Any) -> None:
    try:
        gpa = importlib.import_module(_PAI_GLOBALS_MODULE)
        _clear_pai_tracker_buffers(getattr(gpa, "pai_tracker", None))
    except Exception:
        pass
    _zero_grad_if_available(model)
    modules = getattr(model, "modules", None)
    if modules is None:
        return
    for module in modules():
        clear_processors = getattr(module, "clear_processors", None)
        if clear_processors is None and not hasattr(module, "apply_pb_grads"):
            continue
        if module is not model:
            _zero_grad_if_available(module)
        if clear_processors is None:
            continue
        try:
            clear_processors()
        except Exception:
            continue


def clear_pai_tracker_state() -> None:
    try:
        gpa = importlib.import_module(_PAI_GLOBALS_MODULE)
        _clear_pai_tracker_state(getattr(gpa, "pai_tracker", None))
    except Exception:
        pass


def _consume_pai_config_message(text: str) -> bool:
    global _PAI_CONFIG_SAVED_PRINTED
    if not text.startswith("[PAI Config] Saved"):
        return False
    if _PAI_CONFIG_SAVED_PRINTED:
        return True
    _PAI_CONFIG_SAVED_PRINTED = True
    return False


def _consume_pai_debugger_message(text: str) -> bool:
    """Swallow PerforatedAI's "parameter_type" warning and its explanatory lines.

    2026-08-10: this used to match the trigger and every follow-up line by
    exact wording (``"WARNING: Parameter does not have..."``, ``"You can find
    this param"``, ``"Ensure that model is either converted or tracked"``,
    ``"Instructions in customization.md"``, plus a pdb "(Pdb)"/"--Call--"
    prompt). None of that text exists in perforatedai==3.2.3 any more — it now
    prints ``"Perforated WARNING: Parameter does not have parameter_type
    attribute in n/p mode"`` followed by exactly 5 reworded explanatory lines
    ending in "...debugging.md", and never actually drops into pdb (that's
    still neutralized unconditionally by ``_suppress_pai_debugger``, which
    doesn't depend on text matching) — so every line of the old filter missed,
    and this warning has been flooding stream_*.log uncaught. Counting a fixed
    number of follow-up lines instead of matching their wording survives the
    next time PerforatedAI rewords the explanation, as long as the line count
    doesn't change. See PAIModuleSelection.parameter_ids_to_track for what
    actually causes the underlying warning.
    """
    global _PAI_DEBUGGER_SUPPRESS_REMAINING
    stripped = text.strip()
    if not stripped:
        return False
    if "Parameter does not have parameter_type attribute" in stripped:
        _PAI_DEBUGGER_SUPPRESS_REMAINING = 5
        return True
    if _PAI_DEBUGGER_SUPPRESS_REMAINING:
        _PAI_DEBUGGER_SUPPRESS_REMAINING -= 1
        return True
    return False


def _consume_pai_noise_message(text: str) -> bool:
    stripped = text.strip()
    return stripped.startswith(
        "For PAI training it is recommended to not use weight decay"
    )


def _consume_pai_output_message(text: str) -> bool:
    return (
        _consume_pai_config_message(text)
        or _consume_pai_debugger_message(text)
        or _consume_pai_noise_message(text)
    )


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


def set_pai_root(root: Path | str) -> Path:
    """Point PerforatedAI's artifact tree at ``root`` and return it resolved.

    Called once per invocation so PAI writes under ``--results-root`` rather
    than into the directory the command happened to be run from. Resolved
    eagerly because :func:`pai_working_directory` chdirs into this path, and a
    relative one would re-resolve against the directory it just moved to.
    """
    global _PAI_ROOT
    _PAI_ROOT = Path(root).expanduser().resolve()
    return _PAI_ROOT


def pai_root() -> Path:
    """Return the directory PerforatedAI artifacts are written under."""
    return _PAI_ROOT


@contextmanager
def pai_working_directory() -> Iterator[None]:
    """Run PerforatedAI file I/O with ``PAI/`` as the working directory.

    PerforatedAI resolves ``save_name`` relative to the process working
    directory and refuses names containing a path separator, so every call that
    reads or writes PAI artifacts has to be made from inside ``PAI/``. Re-entrant
    so nested PAI calls share a single directory change.
    """
    global _PAI_WORKING_DIRECTORY_DEPTH
    if _PAI_WORKING_DIRECTORY_DEPTH:
        yield
        return
    _PAI_ROOT.mkdir(parents=True, exist_ok=True)
    previous = Path.cwd()
    _PAI_WORKING_DIRECTORY_DEPTH += 1
    os.chdir(_PAI_ROOT)
    try:
        yield
    finally:
        _PAI_WORKING_DIRECTORY_DEPTH -= 1
        os.chdir(previous)


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
    module_selection: PAIModuleSelection | None = None,
    confirm_unwrapped_modules: bool = True,
    config_snapshot_path: Path | str | None = None,
    dendrite_training_max_epochs: int | None = None,
    dynamic_dendritic_training: bool = True,
    freeze_dendrite_updates_fraction: float = 0.20,
    batches_per_epoch: int | None = None,
    runtime_options: PAIRuntimeOptions | None = None,
) -> Any:
    try:
        _mirror_env_aliases()
        runtime_options = runtime_options or PAIRuntimeOptions()
        GPA = importlib.import_module(_PAI_GLOBALS_MODULE)
        UPA = importlib.import_module("perforatedai.utils_perforatedai")
        upa_perforate_model = getattr(UPA, "perforate_model")

        modules_mod = importlib.import_module("perforatedai.modules_perforatedai")
        _set_tracked_params = getattr(modules_mod, "set_tracked_params", None)

        def _run_perforation() -> Any:
            _configure_pai_trackers(
                GPA,
                module_selection,
                confirm_unwrapped_modules,
                no_backward_workaround=runtime_options.no_backward_workaround,
                candidate_graph_enabled=runtime_options.candidate_graph_enabled,
            )
            if dendrite_training_max_epochs is not None:
                _configure_pai_training_schedule(
                    GPA,
                    max_epochs=dendrite_training_max_epochs,
                    dynamic_dendritic_training=dynamic_dendritic_training,
                    freeze_fraction=freeze_dendrite_updates_fraction,
                    batches_per_epoch=batches_per_epoch,
                    initial_correlation_batches_limit=(
                        runtime_options.initial_correlation_batches_limit
                    ),
                    fixed_switch_interval=runtime_options.fixed_switch_interval,
                )
            with pai_working_directory():
                perforated = upa_perforate_model(
                    model,
                    doing_pai=doing_pai,
                    save_name=_pai_flat_save_name(save_name),
                    maximizing_score=maximizing_score,
                    making_graphs=True,
                )
            if _set_tracked_params is not None:
                _set_tracked_params(perforated)
            return perforated

        if runtime_options.use_runtime_guard:
            with pai_runtime_guard():
                perforated_model = _run_perforation()
        else:
            perforated_model = _run_perforation()
        _snapshot_pai_config(save_name, config_snapshot_path)
        return perforated_model
    except SystemExit as exc:
        # PerforatedAI reports fatal configuration problems by calling
        # sys.exit(), which would otherwise tear the benchmark down with no
        # traceback and no log line.
        raise RuntimeError(
            f"PerforatedAI aborted perforation with SystemExit({exc.code}). "
            "The preceding PerforatedAI output explains what it rejected; the "
            "dendritic condition is invalid until that is resolved."
        ) from exc
    except Exception as exc:
        if doing_pai:
            raise RuntimeError(
                "PerforatedAI failed to perforate the model. The dendritic "
                "condition is invalid, so the benchmark will not continue with "
                "an unperforated fallback model."
            ) from exc
        return model


def _pai_flat_save_name(save_name: str) -> str:
    """Return ``save_name`` collapsed to a single path-separator-free segment."""
    path = Path(save_name)
    if path.is_absolute():
        return path.name
    parts = [part for part in path.parts if part != PAI_DIRECTORY_NAME]
    return "_".join(parts) if parts else path.name


def pai_save_path(save_name: str) -> Path:
    """Return the ``PAI/`` directory PerforatedAI writes ``save_name`` into."""
    return _PAI_ROOT / _pai_flat_save_name(save_name)


def pai_resume_state_exists(save_name: str, name: str = PAI_RESUME_NAME) -> bool:
    """Report whether a PAI system snapshot is available to resume from."""
    return (pai_save_path(save_name) / f"{name}.pt").exists()


def pai_system_checkpoint_exists(save_name: str, name: str) -> bool:
    """Report whether a named PAI system checkpoint exists for ``save_name``."""
    return (pai_save_path(save_name) / f"{name}.pt").exists()


def save_pai_system(
    model: Any, save_name: str, name: str = PAI_RESUME_NAME
) -> bool:
    """Snapshot the perforated network *and* the PAI tracker's own state.

    The benchmark's epoch checkpoint carries model and optimizer tensors only.
    PAI keeps the dendrite schedule (dendrites added, cycle count, switch
    epochs, plateau bookkeeping) in ``pai_tracker.member_vars``, which is not
    part of any ``state_dict``, so without this snapshot a resumed run rebuilds
    a zero-dendrite model and restarts the schedule from the first cycle.
    """
    try:
        UPA = importlib.import_module("perforatedai.utils_perforatedai")
        with _suppress_pai_debugger(), pai_working_directory():
            UPA.save_system(model, _pai_flat_save_name(save_name), name)
    except Exception as exc:
        print(
            f"[pai-state] could not snapshot PAI state ({exc}); a resumed run "
            "would restart the dendrite schedule from zero."
        )
        return False
    return True


def load_pai_system(
    model: Any, save_name: str, name: str = PAI_RESUME_NAME
) -> Any | None:
    """Restore a :func:`save_pai_system` snapshot, or return ``None`` on failure.

    ``model`` must be a freshly perforated network; PAI reattaches its module
    vector to it and rebuilds the saved dendrite structure, so the returned
    model is not necessarily the object passed in.
    """
    try:
        UPA = importlib.import_module("perforatedai.utils_perforatedai")
        with _suppress_pai_debugger(), pai_working_directory():
            # load_from_manual_save keeps PAI from advancing its epoch counter:
            # its own periodic saves happen before start_epoch, whereas this
            # snapshot is taken after add_validation_score has already run it.
            return UPA.load_system(
                model,
                _pai_flat_save_name(save_name),
                name,
                load_from_manual_save=True,
            )
    except Exception as exc:
        print(
            f"[pai-state] could not restore PAI state ({exc}); continuing with "
            "a fresh dendrite schedule."
        )
        return None


def _snapshot_pai_config(
    save_name: str, config_snapshot_path: Path | str | None
) -> None:
    flat_save_name = _pai_flat_save_name(save_name)
    config_path = pai_save_path(save_name) / f"{flat_save_name}_config.json"
    if not config_path.exists():
        return
    named_snapshot = _PAI_ROOT / f"{flat_save_name}_PAI_config.json"
    try:
        shutil.copy2(config_path, named_snapshot)
        if config_snapshot_path is not None:
            artifact_path = Path(config_snapshot_path)
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(config_path, artifact_path)
    except Exception:
        return


def latest_pai_switch_checkpoint(save_name: str) -> str | None:
    """Return the latest source PAI switch checkpoint name, without ``.pt``."""
    folder = pai_save_path(save_name)
    if not folder.is_dir():
        return None
    latest_switch: int | None = None
    for path in folder.glob("switch_*.pt"):
        match = re.fullmatch(r"switch_(\d+)", path.stem)
        if match is None:
            continue
        switch_num = int(match.group(1))
        if latest_switch is None or switch_num > latest_switch:
            latest_switch = switch_num
    if latest_switch is None:
        return None
    return f"switch_{latest_switch}"


def _installed_pai_versions() -> str:
    """Return the installed perforatedai/perforatedbp versions for diagnostics."""
    from importlib.metadata import PackageNotFoundError, version

    parts = []
    for pkg in ("perforatedai", "perforatedbp"):
        try:
            parts.append(f"{pkg}=={version(pkg)}")
        except PackageNotFoundError:
            parts.append(f"{pkg}=<not installed>")
    return ", ".join(parts)


def load_pai_system_checkpoint(
    model: Any,
    save_name: str,
    checkpoint_name: str,
) -> Any:
    """Rebuild a PerforatedAI model architecture from a saved PAI switch."""
    try:
        UPA = importlib.import_module("perforatedai.utils_perforatedai")
        modules_mod = importlib.import_module("perforatedai.modules_perforatedai")
        load_system = getattr(UPA, "load_system")
        _set_tracked_params = getattr(modules_mod, "set_tracked_params", None)
        with pai_runtime_guard(), pai_working_directory():
            loaded = load_system(
                model,
                _pai_flat_save_name(save_name),
                checkpoint_name,
                True,
            )
        if _set_tracked_params is not None:
            _set_tracked_params(loaded)
        return loaded
    except SystemExit as exc:
        print(
            f"[state] PerforatedAI system load aborted from "
            f"{pai_save_path(save_name)}/{checkpoint_name}.pt: {exc}"
        )
        raise
    except AttributeError as exc:
        # ``load_system`` -> ``simulate_cycles`` -> ``create_new_dendrite_module``
        # reconstructs the dendrite tracker state by replaying it internally.
        # When the checkpoint was written by an older perforatedai/perforatedbp
        # release than the one currently installed, that replay can find
        # attributes on ``DendriteValueTracker``/``PAIDendriteModule`` that the
        # new library expects to already be populated (e.g. ``.shape``) but the
        # older checkpoint never recorded. That surfaces as a generic
        # AttributeError deep in vendor code, which looks like corruption but
        # is actually a checkpoint/library version mismatch.
        installed = _installed_pai_versions()
        raise RuntimeError(
            "PerforatedAI system checkpoint at "
            f"{pai_save_path(save_name)}/{checkpoint_name}.pt could not be "
            f"replayed by the installed library ({installed}): {exc!r}. This "
            "is the signature of a checkpoint written by an older "
            "perforatedai/perforatedbp release than what's installed now — "
            "the on-disk dendrite state doesn't match what the current "
            "library expects to rebuild from it. Fix by either (a) "
            "retraining the source condition (e.g. dendrites_fp32) under "
            "the currently installed perforatedai/perforatedbp so its PAI "
            "checkpoint is rewritten in the current format, or (b) "
            "reinstalling the perforatedai/perforatedbp versions that "
            "originally produced this checkpoint."
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            "PerforatedAI system checkpoint could not be loaded from "
            f"{pai_save_path(save_name)}/{checkpoint_name}.pt. The dendritic "
            "source architecture is required before loading the benchmark "
            "checkpoint."
        ) from exc


def seed_everything(seed: int) -> None:
    """Seed every RNG that can move a benchmark result.

    Without this, two runs of the same config are two different experiments.
    Measured on `gcn`: `base_fp32` moved 0.7960 -> 0.7620 between two runs of
    the identical command -- 3.4pp with no quantization involved, against a
    dendrite effect of +0.30pp -- and PAI settled on 4 dendrites one run and 3
    the other. See information/MEASUREMENT_CAVEATS.md #7.

    Called once per (model, condition) rather than once per process, so each
    condition is reproducible on its own and, more importantly, so a model's
    `base_*` and `dendrites_*` arms draw the *same* initial weights. That makes
    the arms a paired comparison: a difference between them is the dendrites,
    not a different lottery ticket.

    Note this does not force deterministic kernels
    (`torch.use_deterministic_algorithms`). MPS has no deterministic mode for
    several of the ops these models use, so it would fail outright rather than
    silently disagree; seeding removes the large run-to-run swings while
    leaving nondeterministic reduction order, which is worth far less than
    3.4pp.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed % (2**32))
    except Exception:
        pass
    torch.manual_seed(seed)
    try:
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    try:
        torch.mps.manual_seed(seed)
    except Exception:
        pass


def choose_device() -> Any:
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        try:
            torch.empty(1, device="mps")
            return torch.device("mps")
        except Exception:
            pass
    if torch.cuda.is_available():  # pragma: no cover - CUDA not expected on Mac
        return torch.device("cuda")
    return torch.device("cpu")


# Clip ratios searched when calibrating the integer grid. 1.0 is plain
# max-based scaling; lower values trade saturation of the largest weights for
# a finer step over the bulk of the distribution.
_QUANT_CLIP_RATIOS = tuple(round(1.0 - 0.05 * i, 4) for i in range(13))  # 1.00 -> 0.40

# Calibration searches a subsample of very large tensors; the chosen scale is
# then applied to the full tensor.
_QUANT_CALIBRATION_SAMPLE = 1 << 16


def _calibrate_scale(tensor: Any, qmin: int, qmax: int) -> float:
    """Pick the quantization scale that minimizes ``||q(W) - W||^2``.

    Calibrating on ``abs().max()`` lets one outlier weight define the whole
    grid: the step becomes ``max_abs / codes``, so if that weight is 20x the
    next largest, every ordinary weight collapses onto one or two codes. That
    is the mechanism behind information/MEASUREMENT_CAVEATS.md #1, and the
    reason ``m5``'s 4-bit scores fell 21pp while every other model was fine
    at 4-bit -- its first conv has ``absmax 1.90`` against ``std 0.29``.

    A fixed percentile would fix that but is arbitrary in the other
    direction: at 8-bit there are 256 codes, the step is already fine, and
    clipping a genuinely large weight costs more than it buys. Searching clip
    ratios for the lowest reconstruction error picks per tensor *and* per bit
    width -- it lands on ~1.0 where codes are plentiful and clips hard at
    2-bit where they are not -- so no magic constant has to be right for
    every layer in the suite.
    """
    flat = tensor.detach().flatten().float()
    sample = flat
    if sample.numel() > _QUANT_CALIBRATION_SAMPLE:
        # Strided, not random. A torch.randperm subsample here draws from the
        # global RNG, which would make quantization itself nondeterministic --
        # two runs of the same seeded config could quantize the same weights to
        # different grids. That was measurable: gcn's q2 moved by one test node
        # between the recorded value and a recomputation from the same
        # checkpoint. A fixed stride is reproducible, costs nothing, and for
        # picking a clip ratio is as representative as a random draw, since
        # weight order within a flattened tensor carries no relevant structure.
        step = (sample.numel() + _QUANT_CALIBRATION_SAMPLE - 1) // _QUANT_CALIBRATION_SAMPLE
        sample = flat[::step]
    # The clip search runs on the sample, but the grid must still cover the
    # tensor's true range, so the largest magnitude comes from the full tensor.
    max_abs = float(flat.abs().max())
    if max_abs == 0:
        return 0.0
    best_scale = max_abs / abs(qmin)
    best_error = None
    for ratio in _QUANT_CLIP_RATIOS:
        scale = max_abs * ratio / abs(qmin)
        if scale <= 0:
            continue
        approx = torch.clamp(torch.round(sample / scale), qmin, qmax) * scale
        error = float(torch.sum((approx - sample) ** 2))
        if best_error is None or error < best_error:
            best_error = error
            best_scale = scale
    return best_scale


def symmetric_quantize_tensor(tensor: Any, bit_width: int) -> Any:
    """Uniform symmetric integer quantization onto the standard signed grid.

    ``qmin = -2**(b-1)``, ``qmax = 2**(b-1)-1`` -- e.g. {-2..1} at 2-bit,
    {-8..7} at 4-bit, {-128..127} at 8-bit. The scale divides by the largest
    code *magnitude* (``|qmin|``), not ``qmax``; dividing by ``qmax`` would
    leave an ordinary symmetric tensor on only {-scale, 0, +scale} at 2-bit,
    which is the three-level collapse of caveat #1.
    """
    if bit_width >= 16:
        return tensor.clone()
    if bit_width <= 1:
        return binary_quantize_tensor(tensor)
    qmin = -(2 ** (bit_width - 1))
    qmax = 2 ** (bit_width - 1) - 1
    scale = _calibrate_scale(tensor, qmin, qmax)
    if scale <= 0:
        return torch.zeros_like(tensor)
    return torch.clamp(torch.round(tensor / scale), qmin, qmax) * scale


def ternary_quantize_tensor(tensor: Any) -> Any:
    """BitNet b1.58 absmean ternarization: ``round(clamp(W/s, -1, 1)) * s``.

    The previous kernel returned a bare ``{-1, 0, +1}`` indicator with **no
    scale factor**, so a layer whose weights had ``std ~ 0.005`` came back
    with every surviving weight at magnitude 1.0 -- a ~200x amplification
    that compounds multiplicatively through depth. That is why ``mpnn``
    scored 617.0 RMSE at q1.58 against 0.72 in fp32, and why several models
    collapsed to chance rather than degrading. b1.58 keeps the ternary
    *codes* but restores the per-tensor scale ``s = mean(|W|)``, which is the
    published formulation and the only thing that makes the arm comparable to
    its own fp32 baseline.
    """
    scale = tensor.detach().abs().mean()
    if scale == 0:
        return torch.zeros_like(tensor)
    return torch.clamp(torch.round(tensor / scale), -1, 1) * scale


def binary_quantize_tensor(tensor: Any) -> Any:
    """XNOR-Net binarization: ``mean(|W|) * sign(W)``.

    Same missing-scale defect as :func:`ternary_quantize_tensor` (``mpnn``
    reached 1030.7 RMSE at q1), plus a second failure the scale also fixes:
    the old kernel mapped ``0 -> +1``, so an all-zero parameter became an
    all-ones parameter. ``m5``'s ``dendrites_to_top.0`` is exactly that -- a
    genuinely zeroed dendrite output gate that binarization turned fully on.
    With ``scale == 0`` such a tensor now stays zero.
    """
    scale = tensor.detach().abs().mean()
    if scale == 0:
        return torch.zeros_like(tensor)
    return torch.where(
        tensor >= 0,
        torch.full_like(tensor, float(scale)),
        torch.full_like(tensor, -float(scale)),
    )
