"""Generic worker-process supervision.

The scientific pipeline supplies launch/progress callbacks; this module owns
only process lifecycle, restart limits, backoff, and group termination.
"""

import os
import signal
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class WorkerSupervisor:
    processes: list[subprocess.Popen[bytes]]
    streams: list[list[str]]
    launch: Callable[[int, list[str]], subprocess.Popen[bytes]]
    report_progress: Callable[[], None]
    report: Callable[[str, bool], None]
    on_finish: Callable[[], None]
    interval: int
    max_restarts: int = 5
    restart_backoff_seconds: int = 30

    def watch(self) -> None:
        """Poll workers and restart failed streams up to a bounded limit."""
        restarts = [0] * len(self.processes)
        given_up = [False] * len(self.processes)

        def settled(index: int) -> bool:
            return given_up[index] or self.processes[index].poll() == 0

        while True:
            self.report_progress()
            for index, process in enumerate(self.processes):
                code = process.poll()
                if code is None or code == 0 or given_up[index]:
                    continue
                stream = self.streams[index]
                if restarts[index] >= self.max_restarts:
                    given_up[index] = True
                    self.report(
                        f"stream_{index + 1} crashed {restarts[index]} time(s) "
                        f"(last exit {code}) and hit the restart limit — giving "
                        f"up on: {' '.join(stream)}",
                        True,
                    )
                    continue
                restarts[index] += 1
                self.report(
                    f"stream_{index + 1} (pid {process.pid}) exited with code "
                    f"{code} — restarting (attempt {restarts[index]}/"
                    f"{self.max_restarts}) in {self.restart_backoff_seconds}s: "
                    f"{' '.join(stream)}",
                    True,
                )
                time.sleep(self.restart_backoff_seconds)
                self.processes[index] = self.launch(index + 1, stream)

            if all(settled(index) for index in range(len(self.processes))):
                self.report("", False)
                self.report("all workers finished.", False)
                self.on_finish()
                return
            time.sleep(self.interval)


def terminate_process_groups(
    processes: list[subprocess.Popen[bytes]], *, grace_seconds: float = 10.0
) -> None:
    """SIGTERM each worker group, then SIGKILL groups past the grace period."""
    for process in processes:
        if process.poll() is not None:
            continue
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass

    deadline = time.monotonic() + grace_seconds
    for process in processes:
        try:
            process.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
