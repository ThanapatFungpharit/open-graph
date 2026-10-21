"""Background symbolic tasks, run in worker processes so they can be cancelled.

Each *lane* is a FIFO queue served by one long-lived child process
(``python -m opengraph.engine.worker``), started on first use. A lane runs one task at a
time; the others wait their turn. Cancelling a queued task just drops it; cancelling the
running one kills its worker, and the lane starts a fresh worker for the next task.
A reader thread per worker posts its replies to the interface loop (see
:mod:`opengraph.events`), so callbacks may touch the interface. The interface never
imports SymPy and never waits on it.
"""

from __future__ import annotations

import enum
import io
import os
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import cast

from opengraph import events
from opengraph.engine import worker
from opengraph.engine.compiler import CasRequest, SymbolicTarget
from opengraph.engine.errors import MathError

HISTORY_LIMIT = 100  # finished tasks kept for the Tasks window


class TaskState(enum.Enum):
    QUEUED = "Queued"
    RUNNING = "Running"
    DONE = "Done"
    FAILED = "Failed"
    CANCELLED = "Cancelled"


@dataclass(eq=False)
class Task:
    id: int
    title: str
    lane: str
    operation: str
    target: SymbolicTarget | CasRequest
    on_done: Callable[[object], None] = field(repr=False)  # the result, or a MathError
    on_cancel: Callable[[], None] | None = field(default=None, repr=False)
    state: TaskState = TaskState.QUEUED
    detail: str = ""  # why it failed
    submitted: float = field(default_factory=time.monotonic)
    started: float | None = None
    ended: float | None = None

    @property
    def finished(self) -> bool:
        return self.state in (TaskState.DONE, TaskState.FAILED, TaskState.CANCELLED)

    def elapsed(self, now: float | None = None) -> float | None:
        """Seconds spent running (so far, while it runs); None until it starts."""
        if self.started is None:
            return None
        end = self.ended if self.ended is not None else time.monotonic() if now is None else now
        return max(0.0, end - self.started)


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "—"
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, rest = divmod(seconds, 60)
    return f"{int(minutes)}:{rest:04.1f}"


def _worker_environment() -> dict[str, str]:
    """The parent's environment, with OpenGraph importable even when run from source."""
    env = dict(os.environ)
    source_root = str(Path(__file__).resolve().parent.parent)
    current = env.get("PYTHONPATH")
    env["PYTHONPATH"] = source_root + (os.pathsep + current if current else "")
    return env


def _read_replies(process: subprocess.Popen[bytes], lane: _Lane) -> None:
    """Worker thread: forward everything the worker writes, then its exit."""
    stream = process.stdout
    assert isinstance(stream, io.BufferedReader)
    try:
        while chunk := stream.read1(65536):
            events.post(partial(lane._on_output, process, chunk))
    except (OSError, ValueError):  # the pipe was closed under us (killed)
        pass
    events.post(partial(lane._on_exit, process, "The worker stopped"))


class _Lane:
    """One queue and the worker process that serves it."""

    def __init__(self, manager: TaskManager) -> None:
        self.manager = manager
        self.queue: deque[Task] = deque()
        self.current: Task | None = None
        self.process: subprocess.Popen[bytes] | None = None
        self.buffer = bytearray()

    def dispatch(self) -> None:
        if self.current is not None or not self.queue:
            return
        task = self.current = self.queue.popleft()
        task.state, task.started = TaskState.RUNNING, time.monotonic()
        if self.process is None:
            self._start_worker()
        if self.process is not None:
            self._send(task)
        self.manager.changed.emit()

    def _start_worker(self) -> None:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)  # no console window on Windows
        try:
            process = subprocess.Popen(
                [sys.executable, "-m", "opengraph.engine.worker"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                env=_worker_environment(),
                creationflags=flags,
            )
        except OSError:
            # Reported once this call has returned, like a worker that exits at once.
            events.call_soon(self._on_failed_start)
            return
        self.process, self.buffer = process, bytearray()
        threading.Thread(
            target=_read_replies, args=(process, self), name="opengraph-task", daemon=True
        ).start()

    def _on_failed_start(self) -> None:
        task, self.current = self.current, None
        if task is not None:
            task.detail = "The worker could not be started"
            self.manager._finish(task, TaskState.FAILED, MathError(task.detail))
        self.dispatch()

    def _send(self, task: Task) -> None:
        assert self.process is not None and self.process.stdin is not None
        try:
            self.process.stdin.write(worker.encode((task.id, task.operation, task.target)))
            self.process.stdin.flush()
        except OSError:
            pass  # the worker died; its reader reports the exit

    def _on_output(self, process: subprocess.Popen[bytes], data: bytes) -> None:
        if process is not self.process:
            return
        self.buffer += data
        for message in worker.decode(self.buffer):
            task_id, ok, payload = cast(tuple[int, bool, object], message)
            task = self.current
            if task is None or task.id != task_id:
                continue  # cancelled tasks kill their worker, so this shouldn't happen
            self.current = None
            if ok:
                self.manager._finish(task, TaskState.DONE, payload)
            else:
                task.detail = str(payload)
                self.manager._finish(task, TaskState.FAILED, MathError(task.detail))
        self.dispatch()

    def _on_exit(self, process: subprocess.Popen[bytes], reason: str) -> None:
        if process is not self.process:
            return  # a worker we killed on purpose
        self.process = None
        _close(process)
        task, self.current = self.current, None
        if task is not None:
            task.detail = reason
            self.manager._finish(task, TaskState.FAILED, MathError(reason))
        self.dispatch()

    def kill_worker(self) -> None:
        process, self.process = self.process, None
        if process is not None:
            process.kill()
            try:
                process.wait(2)
            except subprocess.TimeoutExpired:
                pass
            _close(process)

    def cancel_current(self) -> Task | None:
        task, self.current = self.current, None
        self.kill_worker()
        return task


def _close(process: subprocess.Popen[bytes]) -> None:
    for stream in (process.stdin, process.stdout):
        if stream is not None:
            try:
                stream.close()
            except OSError:
                pass


class TaskManager:
    """Queues symbolic operations per lane and runs them in worker processes."""

    def __init__(self) -> None:
        self.changed = events.Signal()  # a task was added, started, finished or removed
        self._lanes: dict[str, _Lane] = {}
        self._tasks: list[Task] = []
        self._next_id = 1

    # -- queries ---------------------------------------------------------------------
    def tasks(self) -> list[Task]:
        return list(self._tasks)

    def active(self) -> list[Task]:
        return [t for t in self._tasks if not t.finished]

    def get(self, task_id: int) -> Task | None:
        return next((t for t in self._tasks if t.id == task_id), None)

    # -- commands --------------------------------------------------------------------
    def submit(
        self,
        lane: str,
        title: str,
        operation: str,
        target: SymbolicTarget | CasRequest,
        on_done: Callable[[object], None],
        on_cancel: Callable[[], None] | None = None,
    ) -> Task:
        task = Task(self._next_id, title, lane, operation, target, on_done, on_cancel)
        self._next_id += 1
        self._tasks.append(task)
        self._trim_history()
        queue = self._lanes.setdefault(lane, _Lane(self))
        queue.queue.append(task)
        self.changed.emit()
        queue.dispatch()
        return task

    def cancel(self, task_id: int) -> bool:
        """Cancel a queued or running task; False if it had already finished."""
        task = self.get(task_id)
        if task is None or task.finished:
            return False
        lane = self._lanes[task.lane]
        if lane.current is task:
            lane.cancel_current()
        else:
            lane.queue.remove(task)
        self._finish(task, TaskState.CANCELLED, None)
        lane.dispatch()
        return True

    def cancel_all(self) -> None:
        for task in self.active():
            self.cancel(task.id)

    def clear_finished(self) -> None:
        self._tasks = [t for t in self._tasks if not t.finished]
        self.changed.emit()

    def wait(self, timeout_ms: int = 30_000) -> bool:
        """Process events until every task has finished; False on timeout.

        For tests and scripts: it drives the private event loop (see
        :func:`opengraph.events.process_events`)."""
        return events.wait_until(lambda: not self.active(), timeout_ms / 1000)

    def shutdown(self) -> None:
        """Stop every worker without calling back (the application is closing)."""
        for lane in self._lanes.values():
            lane.queue.clear()
            lane.current = None
            lane.kill_worker()
        now = time.monotonic()
        for task in self.active():
            task.state, task.ended = TaskState.CANCELLED, now
        self.changed.emit()

    # -- internals -------------------------------------------------------------------
    def _finish(self, task: Task, state: TaskState, outcome: object) -> None:
        task.state, task.ended = state, time.monotonic()
        self.changed.emit()
        if state is TaskState.CANCELLED:
            if task.on_cancel is not None:
                task.on_cancel()
        else:
            task.on_done(outcome)

    def _trim_history(self) -> None:
        finished = [t for t in self._tasks if t.finished]
        excess = len(finished) - HISTORY_LIMIT
        if excess > 0:
            drop = {id(t) for t in finished[:excess]}
            self._tasks = [t for t in self._tasks if id(t) not in drop]
