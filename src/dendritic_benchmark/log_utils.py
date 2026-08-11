"""
log_utils.py
============
Shared logging utility that tees all stdout/stderr output to a log file.

Usage in any script:
    from log_utils import setup_logging
    setup_logging(output_dir="results", log_file=args.log_file, script_name="train_classifier")

All subsequent print() calls automatically write to both console and log file.
"""

import sys
from datetime import datetime
from io import TextIOWrapper
from pathlib import Path
from typing import IO

# Directories a `--logging-dir` / `--results-root` / `--comparison-root` value
# must never resolve into or under. This tool is routinely invoked by coding
# agents with CLI arguments that are not typed by a human, so a malformed or
# adversarial value (e.g. `--logging-dir /` or `--logging-dir /etc`) should
# fail loudly here rather than silently create directories or write files
# outside the project. This is deliberately a narrow denylist of OS-critical
# roots rather than a "must stay under cwd" jail: legitimate runs already
# point these flags at arbitrary scratch directories (e.g. under /tmp), and
# blocking that would break real usage for no security benefit.
#
# Entries are resolved lazily (not at import time) and compared against the
# *resolved* candidate path, because several of these are themselves symlinks
# on macOS (`/etc` -> `/private/etc`, `/tmp` -> `/private/tmp`) — comparing
# raw strings would silently miss the traversal this exists to catch.
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


def validate_output_path(path: Path, *, label: str) -> Path:
    """
    Resolve ``path`` and reject it if it lands on or under an OS-critical
    directory.

    Called before any ``mkdir()``/``open()`` on a path built from CLI
    arguments (``--logging-dir``, ``--results-root``, ``--comparison-root``,
    ...), which this codebase treats as untrusted input precisely because
    those arguments are as likely to come from an automated agent as from a
    human typing at a terminal. ``label`` is only used to make the error
    message identify which argument was rejected.
    """
    resolved = path.resolve()
    root = Path(resolved.anchor)  # "/" on POSIX, "C:\\" on Windows
    if resolved == root:
        raise ValueError(
            f"{label} resolves to the filesystem root ({resolved!r}). "
            "Refusing to create or write files there — pass a project-scoped path instead."
        )
    for name in (*_DENIED_OUTPUT_ROOT_NAMES, *_DENIED_WINDOWS_OUTPUT_ROOT_NAMES):
        denied = Path(name)
        if not denied.is_absolute():
            continue
        try:
            denied_resolved = denied.resolve()
        except OSError:
            continue  # doesn't exist on this OS (e.g. Windows roots on macOS)
        if resolved == denied_resolved or denied_resolved in resolved.parents:
            raise ValueError(
                f"{label} resolves to {resolved!r}, under the OS-critical "
                f"directory {denied_resolved!r}. Refusing to create or write "
                "files there — pass a project-scoped path instead."
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
        log_path = log_dir / f"{script_name}_{timestamp}.txt"

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
    _log_file_handle = open(log_path, "w")

    assert sys.__stdout__ is not None
    assert sys.__stderr__ is not None
    sys.stdout = TeeStream(sys.__stdout__, _log_file_handle)  # type: ignore[assignment]
    sys.stderr = TeeStream(sys.__stderr__, _log_file_handle)  # type: ignore[assignment]

    print(f"Log file: {log_path.resolve()}")

    return log_path
