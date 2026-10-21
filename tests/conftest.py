from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path

import pytest

from opengraph import events
from opengraph.ui import dialogs, runtime, theme
from opengraph.ui.main_window import MainWindow
from opengraph.ui.mathedit import widget
from tests.support import make_window


@pytest.fixture(autouse=True)
def _isolated_state(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    """A fresh event loop and process-wide interface state for every test, and
    preferences kept out of the real user profile."""
    monkeypatch.setenv("OPENGRAPH_CONFIG_DIR", str(Path(tmp_path_factory.mktemp("config"))))
    loop = asyncio.new_event_loop()
    events.install(loop)
    theme.set_current(theme.LIGHT)
    theme.set_system_dark(False)
    runtime.clipboard.text = ""
    runtime.clipboard.service = None
    runtime.modifiers.ctrl = runtime.modifiers.shift = False
    runtime.modifiers.alt = runtime.modifiers.meta = False
    runtime.focus_tracker.owner = None
    widget._focused = None
    dialogs.host.page = None
    dialogs.host.files = None
    dialogs.host.shown.clear()
    yield
    loop.run_until_complete(asyncio.sleep(0))
    loop.close()


@pytest.fixture
def window() -> Iterator[MainWindow]:
    """The main window, laid out as the client would report it."""
    win = make_window()
    yield win
    win.shutdown()
