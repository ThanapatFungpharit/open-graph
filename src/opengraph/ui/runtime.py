"""Glue between the interface objects and the Flet session.

Interface classes own Flet controls (composition, not subclassing) and work without a
page, which keeps them testable without a client: :func:`push` sends a control's
changes only once it is on a page, and :func:`spawn` runs a coroutine (focus requests,
clipboard and file dialogs) only while the event loop runs.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Any

import flet as ft

from opengraph import events

log = logging.getLogger(__name__)


def page_of(control: ft.BaseControl) -> ft.Page | None:
    try:
        return control.page
    except RuntimeError:
        return None


def push(*controls: ft.BaseControl) -> None:
    """Send the controls' changes to the client (nothing to do before they are shown)."""
    for control in controls:
        if page_of(control) is not None:
            control.update()


_tasks: set[asyncio.Task[Any]] = set()


def spawn(make: Callable[[], Coroutine[Any, Any, object]]) -> bool:
    """Run ``make()`` on the running event loop; False when there is none."""
    loop = events.loop()
    if not loop.is_running():
        return False
    task = loop.create_task(_logged(make()))
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)
    return True


async def _logged(coroutine: Coroutine[Any, Any, object]) -> None:
    try:
        await coroutine
    except Exception:
        log.exception("Background interface task failed")


FOCUS_ATTEMPTS = 20  # passes of the event loop to wait for a new control to be shown


def focus(control: ft.TextField | ft.KeyboardListener | ft.IconButton, attempt: int = 0) -> None:
    """Ask the client to focus ``control``, once it is on a page.

    A control that was just added reaches the page with the next update, so the request
    is retried for a few passes of the event loop."""
    if page_of(control) is not None:
        spawn(lambda: _focus(control))
    elif attempt < FOCUS_ATTEMPTS and events.loop().is_running():
        events.call_soon(lambda: focus(control, attempt + 1))


async def _focus(control: ft.TextField | ft.KeyboardListener | ft.IconButton) -> None:
    try:
        await control.focus()  # type: ignore[no-untyped-call]
    except RuntimeError as exc:  # not mounted on the client (yet, or any more)
        log.debug("Focus request failed: %s", exc)


class Clipboard:
    """The system clipboard, through the client; a local copy serves without one."""

    def __init__(self) -> None:
        self.text = ""
        self.service: ft.Clipboard | None = None  # set by the application

    def set(self, text: str) -> None:
        self.text = text
        service = self.service
        if service is not None:
            spawn(lambda: service.set(text))

    async def get(self) -> str:
        if self.service is not None:
            value = await self.service.get()
            if value is not None:
                self.text = value
        return self.text


clipboard = Clipboard()


class FocusTracker:
    """Which text input has the keyboard focus, as reported by the inputs.

    Keys that no input consumes reach the graph canvas only while nothing is focused,
    so typing ``+`` into a table cell never zooms the graph."""

    def __init__(self) -> None:
        self.owner: object | None = None

    def gained(self, owner: object) -> None:
        self.owner = owner

    def lost(self, owner: object) -> None:
        if self.owner is owner:
            self.owner = None


focus_tracker = FocusTracker()


@dataclass(slots=True)
class Modifiers:
    """Modifier keys held down, as reported by the window's keyboard listener."""

    ctrl: bool = False
    shift: bool = False
    alt: bool = False
    meta: bool = False

    def update(self, key: str, down: bool) -> None:
        name = key.lower()
        if name.startswith("control"):
            self.ctrl = down
        elif name.startswith("shift"):
            self.shift = down
        elif name.startswith("alt"):
            self.alt = down
        elif name.startswith("meta"):
            self.meta = down


modifiers = Modifiers()
