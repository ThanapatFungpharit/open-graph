"""Application entry point: ``opengraph [document.ograph.json] [--web]``."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from pathlib import Path

import flet as ft

from opengraph import events
from opengraph.ui import dialogs, fonts, runtime, theme
from opengraph.ui.main_window import MainWindow, fonts_for_page

DEFAULT_SIZE = (1200, 760)


def build(page: ft.Page, document: Path | None = None) -> MainWindow:
    """Set up ``page`` as the OpenGraph window (runs on the interface loop)."""
    events.install(asyncio.get_running_loop())
    page.title = "OpenGraph"
    page.fonts = fonts_for_page()
    page.padding = 0
    page.spacing = 0
    runtime.clipboard.service = ft.Clipboard()
    dialogs.host.page = page
    dialogs.host.files = ft.FilePicker()
    theme.set_system_dark(page.platform_brightness == ft.Brightness.DARK)
    touch = page.platform is not None and page.platform.is_mobile()
    runtime.device.touch = touch
    runtime.device.native_mobile = touch and not page.web

    window = MainWindow(page)
    if not runtime.device.native_mobile:
        width = window.preferences.get_float("windowWidth") or DEFAULT_SIZE[0]
        height = window.preferences.get_float("windowHeight") or DEFAULT_SIZE[1]
        page.window.width, page.window.height = width, height
        page.window.min_width, page.window.min_height = 640, 420
        page.window.prevent_close = True

        def on_window_event(event: ft.WindowEvent[ft.Window]) -> None:
            if event.type == ft.WindowEventType.CLOSE:
                window.close()

        page.window.on_event = on_window_event
    page.on_keyboard_event = window.on_key
    page.on_resize = lambda event: window.resize(event.width, event.height)
    page.on_app_lifecycle_state_change = lambda event: window.on_lifecycle(event.state)
    page.on_platform_brightness_change = lambda _: window.on_brightness_changed(
        page.platform_brightness == ft.Brightness.DARK
    )
    if page.width and page.height:
        window.resize(page.width, page.height)
    page.add(ft.SafeArea(window.control, expand=True))  # clear of notches and system bars
    window.apply_theme(window.theme_mode)
    if document is not None:
        window.open_path(document)
    elif runtime.device.native_mobile:
        window.restore_autosave()
    if not touch:  # focusing would pop up the soft keyboard
        window.panel.focus_row(window.controller.document.items[0].id)
    page.update()
    return window


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv if argv is None else argv
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    args = argv[1:]
    web = "--web" in args
    paths = [Path(a) for a in args if not a.startswith("--")]
    document = paths[0] if paths else None

    async def start(page: ft.Page) -> None:
        build(page, document)

    if web:  # serve the interface at http://localhost:<port> (needs the flet-web package)
        port = int(next((a.split("=", 1)[1] for a in args if a.startswith("--port=")), "8550"))
        os.environ.setdefault("FLET_FORCE_WEB_SERVER", "true")  # serve; don't open a browser
        ft.run(start, assets_dir=str(fonts.ASSETS_DIR), view=ft.AppView.WEB_BROWSER, port=port)
    else:
        ft.run(start, assets_dir=str(fonts.ASSETS_DIR))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
