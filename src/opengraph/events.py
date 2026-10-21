"""Signals, timers and the hand-off from worker threads to the interface loop.

The whole interface runs on one asyncio event loop: Flet's, while the application
runs (see :func:`install`), or a private loop that tests and scripts drive with
:func:`process_events`. Model objects talk to each other through :class:`Signal`
(a plain observer list, called synchronously), debounce with :class:`Timer`, and
background threads hand their results back with :func:`post`, so every callback that
touches the interface runs on the loop's thread.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable

log = logging.getLogger(__name__)

_loop: asyncio.AbstractEventLoop | None = None


def install(loop: asyncio.AbstractEventLoop) -> None:
    """Run callbacks, timers and posted results on ``loop`` from now on."""
    global _loop
    _loop = loop


def loop() -> asyncio.AbstractEventLoop:
    """The interface loop (a private one is created on first use outside the app)."""
    global _loop
    if _loop is None or _loop.is_closed():
        _loop = asyncio.new_event_loop()
    return _loop


def _call(fn: Callable[[], object]) -> None:
    try:
        fn()
    except Exception:  # a failing callback must not stop the loop
        log.exception("Callback failed")


def post(fn: Callable[[], object]) -> None:
    """Run ``fn`` on the interface loop soon; safe to call from any thread."""
    target = loop()
    try:
        target.call_soon_threadsafe(_call, fn)
    except RuntimeError:  # the loop was closed (the application is quitting)
        pass


def call_soon(fn: Callable[[], object]) -> None:
    """Run ``fn`` on the interface loop once the current callback has returned."""
    loop().call_soon(_call, fn)


def process_events(seconds: float = 0.0) -> None:
    """Run the private loop for ``seconds`` (tests and scripts; never inside the app)."""
    target = loop()
    if target.is_running():
        raise RuntimeError("process_events() can't run inside the running event loop")
    target.run_until_complete(asyncio.sleep(seconds))


def wait_until(condition: Callable[[], object], timeout: float = 5.0) -> bool:
    """Process events until ``condition()`` holds; False after ``timeout`` seconds."""
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            return False
        process_events(0.005)
    return True


class Signal:
    """A list of callbacks called, in connection order, by :meth:`emit`."""

    def __init__(self) -> None:
        self._slots: list[Callable[..., object]] = []

    def connect(self, slot: Callable[..., object]) -> None:
        self._slots.append(slot)

    def disconnect(self, slot: Callable[..., object]) -> None:
        self._slots.remove(slot)

    def emit(self, *args: object) -> None:
        for slot in list(self._slots):
            slot(*args)


class Timer:
    """A single-shot or repeating timer on the interface loop.

    ``start()`` (re)starts it, so a single-shot timer started on every keystroke fires
    once typing pauses for ``interval_ms``.
    """

    def __init__(self, interval_ms: int = 0, *, single_shot: bool = False) -> None:
        self.interval_ms = interval_ms
        self.single_shot = single_shot
        self.timeout = Signal()
        self._handle: asyncio.TimerHandle | None = None

    def start(self, interval_ms: int | None = None) -> None:
        if interval_ms is not None:
            self.interval_ms = interval_ms
        self.stop()
        self._handle = loop().call_later(self.interval_ms / 1000, self._fire)

    def stop(self) -> None:
        if self._handle is not None:
            self._handle.cancel()
            self._handle = None

    def is_active(self) -> bool:
        return self._handle is not None

    def _fire(self) -> None:
        self._handle = None
        if not self.single_shot:
            self._handle = loop().call_later(self.interval_ms / 1000, self._fire)
        _call(self.timeout.emit)

    @staticmethod
    def single_shot_call(interval_ms: int, fn: Callable[[], object]) -> None:
        """Call ``fn`` once after ``interval_ms``."""
        loop().call_later(interval_ms / 1000, _call, fn)
