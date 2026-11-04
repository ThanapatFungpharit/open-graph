"""Background symbolic tasks, run in worker processes so they can be cancelled.

Android and iOS don't let an application start processes. There each lane runs its tasks
on a thread instead (:class:`_ThreadLane`), which still keeps the interface responsive.
A thread can't be killed, so cancelling raises an exception inside it (:class:`_ThreadJob`).
SymPy is pure Python, so that stops a computation within moments. Only a single operation
in C can hold it up, such as multiplying integers of millions of digits.

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
import logging
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

log = logging.getLogger(__name__)

HISTORY_LIMIT = 100  # finished tasks kept for the Tasks window


# How often an exception is raised again in a thread that hasn't stopped: SymPy and mpmath
# have a few bare ``except:`` clauses, which can catch the first one.
_STOP_RETRY_S = 0.2


class TaskCancelled(BaseException):
    """Raised inside a worker thread to stop its task.

    Not an ``Exception``, so that neither SymPy's handlers nor :func:`worker.run` take it
    for a failed computation; ``finally`` clauses and ``with`` blocks still clean up.
    """


def _raise_in_thread(thread_id: int) -> bool:
    """Make :class:`TaskCancelled` happen in a thread, at its next bytecode. False if this
    Python can't: the thread then runs on, and only its result is dropped."""
    try:
        import ctypes

        delivered = ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_ulong(thread_id), ctypes.py_object(TaskCancelled)
        )
    except (ImportError, AttributeError, OSError):
        log.warning("This Python can't interrupt a thread: cancelled tasks run to their end")
        return False
    return bool(delivered == 1)


class _ThreadJob:
    """A task on a thread of its own, which can be told to stop.

    The lock makes sure no exception is raised once the thread has finished its work: it
    would escape the thread's function, or hit a new thread with a reused identifier.
    """

    def __init__(self, task: Task, run: Callable[[_ThreadJob], None]) -> None:
        self.task = task
        self._lock = threading.Lock()
        self._finished = False
        self._ended = threading.Event()
        self._thread = threading.Thread(
            target=run, args=(self,), name="opengraph-task", daemon=True
        )
        self._thread.start()

    def finish(self) -> None:
        """Called by the job's own thread when it has nothing left to do. An exception
        raised just before this can still arrive during it, so it starts over then."""
        while True:
            try:
                with self._lock:
                    self._finished = True
                self._ended.set()
                return
            except TaskCancelled:
                continue

    def stop(self) -> None:
        """Keep interrupting the thread, from the side, until it has finished."""
        threading.Thread(target=self._interrupt, name="opengraph-stop", daemon=True).start()

    def _interrupt(self) -> None:
        while True:
            with self._lock:
                thread_id = self._thread.ident
                if self._finished or thread_id is None or not _raise_in_thread(thread_id):
                    return
            if self._ended.wait(_STOP_RETRY_S):
                return

    def wait(self, timeout: float) -> bool:
        """Whether the thread has finished, waiting at most ``timeout`` seconds."""
        return self._ended.wait(timeout)


def _uses_threads() -> bool:
    """Whether tasks run on threads, as they must where processes aren't allowed."""
    return bool(os.environ.get("OPENGRAPH_THREAD_WORKERS")) or sys.platform in ("android", "ios")


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


class _ThreadLane(_Lane):
    """One queue, with each task on a thread of its own (see the module's documentation)."""

    def __init__(self, manager: TaskManager) -> None:
        super().__init__(manager)
        self.job: _ThreadJob | None = None

    def dispatch(self) -> None:
        if self.current is not None or not self.queue:
            return
        task = self.current = self.queue.popleft()
        task.state, task.started = TaskState.RUNNING, time.monotonic()
        self.job = _ThreadJob(task, self._run)
        self.manager.changed.emit()

    def _run(self, job: _ThreadJob) -> None:
        """Worker thread: compute, then report on the interface loop."""
        task = job.task
        try:
            outcome = worker.run(task.operation, task.target)
            job.finish()
            events.post(partial(self._on_result, task, outcome))
        except TaskCancelled:
            pass
        finally:
            job.finish()

    def _on_result(self, task: Task, outcome: tuple[bool, object]) -> None:
        if task is not self.current:
            return  # cancelled, or the application is closing
        self.current, self.job = None, None
        ok, payload = outcome
        if ok:
            self.manager._finish(task, TaskState.DONE, payload)
        else:
            task.detail = str(payload)
            self.manager._finish(task, TaskState.FAILED, MathError(task.detail))
        self.dispatch()

    def kill_worker(self) -> None:
        job, self.job = self.job, None
        if job is not None:
            job.stop()


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
        queue = self._lanes.get(lane)
        if queue is None:
            queue = self._lanes[lane] = _ThreadLane(self) if _uses_threads() else _Lane(self)
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
