"""Edits compile on a background thread; stale results never reach the interface."""

from __future__ import annotations

import threading
from typing import Any

import pytest

from opengraph import controller as controller_module
from opengraph.ui.main_window import MainWindow
from tests.support import first_id, wait_until


def feedback(window: MainWindow) -> str:
    result = window.controller.results.get(first_id(window))
    return "" if result is None else result.message


def test_typing_compiles_off_the_interface_thread(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    threads: list[int] = []
    original = controller_module.compile_with_values

    def recording(*args: Any, **kwargs: Any) -> Any:
        threads.append(threading.get_ident())
        return original(*args, **kwargs)

    monkeypatch.setattr(controller_module, "compile_with_values", recording)
    controller = window.controller
    controller.set_text(first_id(window), "a = 2 + 3")
    assert controller.compiling
    wait_until(lambda: not controller.compiling)
    assert "5" in feedback(window)
    assert threads and threading.get_ident() not in threads


def test_a_result_for_outdated_text_is_discarded(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered, release = threading.Event(), threading.Event()
    original = controller_module.compile_with_values
    calls: list[str] = []

    def blocked(rows: Any, *args: Any) -> Any:
        texts = " ".join(text for _, text in rows)
        calls.append(texts)
        if "first" in texts:  # hold the first compile until the text has changed
            entered.set()
            assert release.wait(5)
        return original(rows, *args)

    monkeypatch.setattr(controller_module, "compile_with_values", blocked)
    controller = window.controller
    controller.set_text(first_id(window), "first = 1")
    wait_until(entered.is_set)
    controller.set_text(first_id(window), "second = 2")
    release.set()
    wait_until(lambda: not controller.compiling)
    assert controller.results[first_id(window)].defines == "second"
    assert any("second" in c for c in calls)


def test_flush_compiles_at_once_even_while_a_background_compile_runs(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    entered, release = threading.Event(), threading.Event()
    original = controller_module.compile_with_values
    ui_thread = threading.get_ident()

    def slow_in_background(*args: Any) -> Any:
        if threading.get_ident() != ui_thread:
            entered.set()
            assert release.wait(5)
        return original(*args)

    monkeypatch.setattr(controller_module, "compile_with_values", slow_in_background)
    controller = window.controller
    controller.set_text(first_id(window), "b = 7")
    wait_until(entered.is_set)
    controller.flush()  # doesn't wait for the worker
    assert controller.results[first_id(window)].defines == "b"
    assert not controller.compiling
    release.set()
