"""Symbolic tasks in worker processes: cancelling, timing and the Tasks panel."""

from __future__ import annotations

import subprocess
import sys
import threading

import pytest

from opengraph import events, tasks
from opengraph.engine import worker
from opengraph.engine.compiler import compile_expression
from opengraph.engine.errors import MathError
from opengraph.tasks import TaskManager, TaskState, format_duration
from opengraph.ui.main_window import MainWindow
from tests.support import wait_until

pytest.importorskip("sympy")


def target(text: str):  # type: ignore[no-untyped-def]
    result = compile_expression(text).symbolic
    assert result is not None
    return result


def test_frames_round_trip() -> None:
    buffer = bytearray(worker.encode((1, "solve", "a")) + worker.encode([2, 3]))
    partial = bytearray(buffer[:-2])
    assert worker.decode(partial) == [(1, "solve", "a")]  # the incomplete frame waits
    assert worker.decode(buffer) == [(1, "solve", "a"), [2, 3]]
    assert buffer == b""


def test_worker_reports_errors_as_messages() -> None:
    assert worker.run("derivative", target("y = x^3")) == (True, "3*x^2")
    ok, message = worker.run("solve", target("abs(x) - x"))
    assert not ok and "interval" in str(message)


def test_format_duration() -> None:
    assert format_duration(None) == "—"
    assert format_duration(3.24) == "3.2 s"
    assert format_duration(65.5) == "1:05.5"


def test_tasks_run_in_order_and_record_their_time() -> None:
    manager = TaskManager()
    results: list[object] = []
    first = manager.submit("lane", "a", "derivative", target("y = x^3"), results.append)
    second = manager.submit("lane", "b", "solve", target("abs(x) - x"), results.append)
    assert first.state is TaskState.RUNNING and second.state is TaskState.QUEUED
    assert second.elapsed() is None
    assert manager.wait()
    assert results[0] == "3*x^2"
    assert second.state is TaskState.FAILED and "interval" in second.detail
    assert first.elapsed() is not None and first.elapsed() == first.elapsed()  # frozen
    manager.shutdown()


def test_cancelling_kills_the_running_task_and_the_lane_goes_on() -> None:
    manager = TaskManager()
    done: list[object] = []
    cancelled: list[str] = []
    running = manager.submit(
        "lane", "a", "derivative", target("y = x^3"), done.append, lambda: cancelled.append("a")
    )
    queued = manager.submit(
        "lane", "b", "derivative", target("y = x^2"), done.append, lambda: cancelled.append("b")
    )
    third = manager.submit("lane", "c", "simplify", target("y = 2x - x"), done.append)
    assert manager.cancel(queued.id) and manager.cancel(running.id)
    assert not manager.cancel(running.id)  # already finished
    assert cancelled == ["b", "a"]
    assert running.state is queued.state is TaskState.CANCELLED
    assert third.state is TaskState.RUNNING  # on a fresh worker
    assert manager.wait()
    assert done == ["x"]  # only the task that wasn't cancelled answered
    manager.shutdown()


def test_cancel_from_the_tasks_panel_unlocks_the_controller(window: MainWindow) -> None:
    c = window.controller
    row_id = c.document.items[0].id
    c.set_text(row_id, "y = x^3")
    c.run_symbolic(row_id, "derivative")
    assert c.busy
    wait_until(lambda: window.tasks_button.visible)
    assert str(window.tasks_button.content).startswith("1 task")
    window.show_tasks()
    dialog = window.tasks_dialog
    assert dialog.visible
    (task,) = c.tasks.active()
    assert task.title == "Derivative: y = x^3"
    assert dialog.cancel_button(task.id) is not None
    dialog.cancel(task.id)
    assert task.state is TaskState.CANCELLED
    assert not c.busy and not window.progress.visible and not window.tasks_button.visible
    assert dialog.cancel_button(task.id) is None
    assert dialog.summary.value == "No tasks running"
    assert len(c.document) == 1
    # The next operation runs normally.
    c.run_symbolic(row_id, "derivative")
    wait_until(lambda: len(c.document) == 2, timeout_ms=30_000)
    assert c.document.items[1].text == "y = 3*x^2"
    assert [t.state for t in c.tasks.tasks()][-1] is TaskState.DONE
    c.tasks.clear_finished()
    assert dialog.row_count == len(c.tasks.active()) == 0


def test_cancelled_exact_solve_keeps_the_numerical_solutions(window: MainWindow) -> None:
    c = window.controller
    row_id = c.document.items[0].id
    c.set_text(row_id, "b^3 = 2")
    c.flush()
    (task,) = c.tasks.active()
    assert task.title == "Solve exactly: b^3 = 2"
    c.tasks.cancel(task.id)
    assert c.wait_for_jobs()
    assert c.results[row_id].message.startswith("b ≈ 1.25992")
    c.recompile()
    assert not c.tasks.active()  # not retried


# -- where processes aren't allowed (Android, iOS) -------------------------------------------


@pytest.fixture
def threads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENGRAPH_THREAD_WORKERS", "1")


def _no_processes(monkeypatch: pytest.MonkeyPatch) -> None:
    def refused(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("tried to start a process")

    monkeypatch.setattr(subprocess, "Popen", refused)


@pytest.mark.parametrize("platform", ["android", "ios"])
def test_mobile_platforms_use_threads(monkeypatch: pytest.MonkeyPatch, platform: str) -> None:
    monkeypatch.delenv("OPENGRAPH_THREAD_WORKERS", raising=False)
    monkeypatch.setattr(sys, "platform", platform)
    assert tasks._uses_threads()


@pytest.mark.usefixtures("threads")
def test_a_thread_lane_runs_tasks_in_order(monkeypatch: pytest.MonkeyPatch) -> None:
    _no_processes(monkeypatch)
    manager = TaskManager()
    results: list[object] = []
    target = compile_expression("y = x^2").symbolic
    assert target is not None
    manager.submit("lane", "a", "derivative", target, results.append)
    manager.submit("lane", "b", "integral", target, results.append)
    assert manager.wait(30_000)
    assert len(results) == 2 and all(isinstance(r, str) for r in results)
    assert [t.state for t in manager.tasks()] == [TaskState.DONE, TaskState.DONE]
    manager.shutdown()


@pytest.mark.usefixtures("threads")
def test_cancelling_on_a_thread_drops_the_result(monkeypatch: pytest.MonkeyPatch) -> None:
    release = threading.Event()

    def slow(_operation: str, _target: object) -> tuple[bool, object]:
        release.wait(10)
        return True, "late"

    monkeypatch.setattr(worker, "run", slow)
    manager = TaskManager()
    results: list[object] = []
    cancelled: list[bool] = []
    target = compile_expression("y = x^2").symbolic
    assert target is not None
    task = manager.submit(
        "lane", "t", "derivative", target, results.append, lambda: cancelled.append(True)
    )
    assert task.state is TaskState.RUNNING
    assert manager.cancel(task.id)
    assert task.state is TaskState.CANCELLED and cancelled == [True]

    # The lane goes on at once, and the abandoned result is never delivered.
    monkeypatch.setattr(worker, "run", lambda _o, _t: (True, "next"))
    following = manager.submit("lane", "u", "derivative", target, results.append)
    assert manager.wait(5_000)
    release.set()
    events.process_events(0.05)
    assert results == ["next"] and following.state is TaskState.DONE
    manager.shutdown()


@pytest.mark.usefixtures("threads")
def test_a_failure_on_a_thread_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(worker, "run", lambda _o, _t: (False, "no luck"))
    manager = TaskManager()
    results: list[object] = []
    target = compile_expression("y = x^2").symbolic
    assert target is not None
    task = manager.submit("lane", "t", "derivative", target, results.append)
    assert manager.wait(5_000)
    assert task.state is TaskState.FAILED and task.detail == "no luck"
    assert isinstance(results[0], MathError)
    manager.shutdown()
