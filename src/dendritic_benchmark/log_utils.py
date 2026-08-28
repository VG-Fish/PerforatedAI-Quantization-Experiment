"""
log_utils.py
============
Shared logging utility that tees all stdout/stderr output to a log file.

Usage in any script:
    from log_utils import setup_logging
    setup_logging(output_dir="results", log_file=args.log_file, script_name="train_classifier")

All subsequent print() calls automatically write to both console and log file.
"""

from datetime import datetime
from io import TextIOWrapper
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import IO

OUTPUT_ROOTS_ENV = "DQB_ALLOWED_OUTPUT_ROOTS"
# Directories a `--logging-dir` / `--results-root` / `--comparison-root` value
# must never resolve into or under, even when explicitly listed in
# DQB_ALLOWED_OUTPUT_ROOTS. Entries are resolved lazily because several are
# symlinks on macOS (`/etc` -> `/private/etc`, `/tmp` -> `/private/tmp`).
_DENIED_OUTPUT_ROOT_NAMES = (
    "/bin",
    "/boot",
    "/dev",
    "/etc",
    "/lib",
    "/lib64",
    "/proc",
    "/root",
    "/sbin",
    "/sys",
    "/usr",
    "/System",
    "/Library",
)
_DENIED_WINDOWS_OUTPUT_ROOT_NAMES = (
    "C:\\Windows",
    "C:\\Program Files",
    "C:\\Program Files (x86)",
)
_SAFE_LOG_STEM_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _path_contains(root: Path, candidate: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def _is_denied_output_path(resolved: Path) -> bool:
    root = Path(resolved.anchor)  # "/" on POSIX, "C:\\" on Windows
    if resolved == root:
        return True
    for name in (*_DENIED_OUTPUT_ROOT_NAMES, *_DENIED_WINDOWS_OUTPUT_ROOT_NAMES):
        denied = Path(name)
        if not denied.is_absolute():
            continue
        try:
            denied_resolved = denied.expanduser().resolve(strict=False)
        except OSError:
            continue
        if _path_contains(denied_resolved, resolved):
            return True
    return False


def _allowed_output_roots() -> tuple[Path, ...]:
    raw_roots = [Path.cwd(), Path(tempfile.gettempdir())]
    extra_roots = os.environ.get(OUTPUT_ROOTS_ENV, "")
    raw_roots.extend(Path(part) for part in extra_roots.split(os.pathsep) if part)

    roots: list[Path] = []
    for root in raw_roots:
        resolved = root.expanduser().resolve(strict=False)
        if _is_denied_output_path(resolved):
            continue
        if resolved not in roots:
            roots.append(resolved)
    return tuple(roots)


def _safe_log_stem(script_name: str) -> str:
    stem = _SAFE_LOG_STEM_RE.sub("_", script_name).strip("._-")
    return stem[:80] or "script"


def validate_output_path(path: Path, *, label: str) -> Path:
    """
    Resolve ``path`` and reject it if it escapes the configured output roots.

    Called before any ``mkdir()``/``open()`` on a path built from CLI
    arguments (``--logging-dir``, ``--results-root``, ``--comparison-root``,
    ...), which this codebase treats as untrusted input precisely because
    those arguments are as likely to come from an automated agent as from a
    human typing at a terminal. ``label`` is only used to make the error
    message identify which argument was rejected.
    """
    resolved = path.expanduser().resolve(strict=False)
    if _is_denied_output_path(resolved):
        raise ValueError(
            f"{label} resolves to {resolved!r}, which is not a safe output location. "
            "Pass a project-scoped path or an explicitly allowed scratch path instead."
        )
    allowed_roots = _allowed_output_roots()
    if not any(_path_contains(root, resolved) for root in allowed_roots):
        roots = ", ".join(str(root) for root in allowed_roots)
        raise ValueError(
            f"{label} resolves to {resolved!r}, outside the allowed output roots "
            f"({roots}). Use a relative path, a path under the system temp "
            f"directory, or set {OUTPUT_ROOTS_ENV} to opt in another root."
        )
    return resolved


class TeeStream:
    """A stream wrapper that writes to both the original stream and a log file."""

    def __init__(self, original_stream: IO[str], log_file_handle: IO[str]) -> None:
        self.original: IO[str] = original_stream
        self.log_file: IO[str] = log_file_handle

    def write(self, data: str) -> None:
        self.original.write(data)
        self.log_file.write(data)
        self.log_file.flush()

    def flush(self) -> None:
        self.original.flush()
        self.log_file.flush()

    def fileno(self) -> int:
        return self.original.fileno()

    def isatty(self) -> bool:
        return self.original.isatty()


_log_file_handle: TextIOWrapper | None = None


def setup_logging(
    output_dir: str = ".",
    log_file: str | None = None,
    script_name: str = "script",
) -> Path:
    """
    Set up tee logging so all print() output goes to both console and a log file.

    Parameters
    ----------
    output_dir : str
        Base output directory. Logs go into ``<output_dir>/``.
    log_file : str or None
        Explicit log file path. If None, a timestamped file is created
        in ``<output_dir>/<script_name>_YYYYMMDD_HHMMSS.txt``.
        Existing files are never overwritten — a numeric suffix is added
        if the path already exists.
    script_name : str
        Used to construct the default log filename.

    Returns
    -------
    Path
        The resolved log file path.
    """
    global _log_file_handle

    if log_file is not None:
        log_path = validate_output_path(Path(log_file), label="log_file")
    else:
        log_dir = validate_output_path(Path(output_dir), label="output_dir")
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"{_safe_log_stem(script_name)}_{timestamp}.txt"

    # Never overwrite: add numeric suffix if file exists
    if log_path.exists():
        stem = log_path.stem
        suffix = log_path.suffix
        parent = log_path.parent
        counter = 1
        while log_path.exists():
            log_path = parent / f"{stem}_{counter}{suffix}"
            counter += 1

    log_path.parent.mkdir(parents=True, exist_ok=True)
    _log_file_handle = log_path.open("w", encoding="utf-8")

    original_stdout = sys.__stdout__
    original_stderr = sys.__stderr__
    if original_stdout is None or original_stderr is None:
        raise RuntimeError("Original stdout/stderr streams are unavailable.")
    sys.stdout = TeeStream(original_stdout, _log_file_handle)  # type: ignore[assignment]
    sys.stderr = TeeStream(original_stderr, _log_file_handle)  # type: ignore[assignment]

    print(f"Log file: {log_path.resolve()}")

    return log_path
