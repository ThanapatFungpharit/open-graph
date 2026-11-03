"""Dialogs: messages, questions, the colour chooser and file choosers.

Flet dialogs don't block: each one reports its answer through a callback. They are
shown through :data:`host`, which knows the page once the application has started;
before that (in tests) shown dialogs are only recorded in ``host.shown``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path

import flet as ft

from opengraph import colors
from opengraph.document import PALETTE
from opengraph.ui import runtime, theme


class DialogHost:
    def __init__(self) -> None:
        self.page: ft.Page | None = None
        self.shown: list[ft.DialogControl] = []  # open dialogs, oldest first
        self.files: ft.FilePicker | None = None  # set by the application

    @property
    def saves_through_client(self) -> bool:
        """Whether saving hands data to the client's file chooser, instead of getting a
        path that this process could write to (a browser, Android and iOS)."""
        page = self.page
        return page is not None and (page.web or runtime.device.native_mobile)

    def show(self, dialog: ft.DialogControl) -> None:
        self.shown.append(dialog)
        if self.page is not None:
            self.page.show_dialog(dialog)

    def close(self, dialog: ft.DialogControl) -> None:
        if dialog in self.shown:
            self.shown.remove(dialog)
        if self.page is not None and dialog.open:
            dialog.open = False
            runtime.push(dialog)


host = DialogHost()


def _button(label: str, on_click: Callable[[], object], *, primary: bool = False) -> ft.Control:
    if primary:
        return ft.FilledButton(label, on_click=on_click)
    return ft.TextButton(label, on_click=on_click)


def message(title: str, text: str) -> ft.AlertDialog:
    """An information or warning message with an OK button."""
    dialog = ft.AlertDialog(title=title, content=ft.Text(text, selectable=True))
    dialog.actions = [_button("OK", lambda: host.close(dialog), primary=True)]
    host.show(dialog)
    return dialog


def ask(
    title: str,
    text: str,
    answers: Sequence[str],
    on_answer: Callable[[str | None], object],
    *,
    default: str | None = None,
) -> ft.AlertDialog:
    """A question; ``on_answer`` gets the chosen answer, or None when dismissed."""
    done = False

    def answer(choice: str | None) -> None:
        nonlocal done
        if done:
            return
        done = True
        host.close(dialog)
        on_answer(choice)

    dialog = ft.AlertDialog(
        title=title,
        content=ft.Text(text),
        modal=True,
        on_dismiss=lambda: answer(None),
    )
    dialog.actions = [
        _button(label, partial(answer, label), primary=label == default) for label in answers
    ]
    host.show(dialog)
    return dialog


class ColorDialog:
    """Choose a colour: a palette swatch, or any ``#rrggbb`` typed in."""

    def __init__(self, title: str, current: str, on_chosen: Callable[[str], object]) -> None:
        self._on_chosen = on_chosen
        self.field = ft.TextField(
            value=current,
            label="Hex colour",
            width=160,
            dense=True,
            on_change=lambda _: self._preview(),
            on_submit=lambda _: self.accept(),
        )
        self.sample = ft.Container(width=28, height=28, border_radius=14)
        self.error = ft.Text("", color=theme.current().error, visible=False, size=12)
        swatches: list[ft.Control] = [
            ft.Container(
                width=28,
                height=28,
                border_radius=14,
                bgcolor=color,
                tooltip=color,
                on_click=lambda _, c=color: self.choose(c),
            )
            for color in PALETTE
        ]
        self.dialog = ft.AlertDialog(
            title=title,
            content=ft.Column(
                [
                    ft.Row(swatches, wrap=True, spacing=8, run_spacing=8, width=260),
                    ft.Row([self.sample, self.field], spacing=12),
                    self.error,
                ],
                tight=True,
                spacing=14,
            ),
        )
        self.dialog.actions = [
            _button("Cancel", lambda: host.close(self.dialog)),
            _button("OK", self.accept, primary=True),
        ]
        self._preview()

    def show(self) -> None:
        host.show(self.dialog)

    def _preview(self) -> None:
        text = (self.field.value or "").strip()
        self.sample.bgcolor = colors.name(text) if colors.is_valid(text) else None
        runtime.push(self.sample)

    def choose(self, color: str) -> None:
        host.close(self.dialog)
        self._on_chosen(colors.name(color))

    def accept(self) -> None:
        text = (self.field.value or "").strip()
        if not colors.is_valid(text):
            self.error.value = "Type a colour such as #2563eb"
            self.error.visible = True
            runtime.push(self.error)
            return
        self.choose(text if text.startswith("#") else f"#{text}")


# -- files --------------------------------------------------------------------------------
def _file_picker() -> ft.FilePicker | None:
    return host.files


def ask_open_path(
    title: str, initial: Path, extensions: Sequence[str], on_path: Callable[[Path], object]
) -> None:
    """Choose an existing file; ``on_path`` is called only if one is chosen."""
    picker = _file_picker()
    if picker is None:
        return

    async def run() -> None:
        filtered = bool(extensions) and not runtime.device.touch
        files = await picker.pick_files(
            dialog_title=title,
            initial_directory=str(initial),
            # Android and iOS filter by file types that .ograph.json isn't registered as.
            file_type=ft.FilePickerFileType.CUSTOM if filtered else ft.FilePickerFileType.ANY,
            allowed_extensions=list(extensions) if filtered else None,
        )
        if files and files[0].path:
            on_path(Path(files[0].path))

    runtime.spawn(run)


def ask_save_path(
    title: str, initial: Path, extensions: Sequence[str], on_path: Callable[[Path], object]
) -> None:
    """Choose where to save; ``on_path`` is called only if a path is chosen."""
    picker = _file_picker()
    if picker is None:
        return

    async def run() -> None:
        path = await picker.save_file(
            dialog_title=title,
            file_name=initial.name,
            initial_directory=str(initial.parent),
            file_type=ft.FilePickerFileType.CUSTOM if extensions else ft.FilePickerFileType.ANY,
            allowed_extensions=list(extensions) or None,
        )
        if path:
            on_path(Path(path))

    runtime.spawn(run)


def save_bytes(
    title: str,
    name: str,
    data: bytes,
    on_saved: Callable[[Path | None], object],
) -> None:
    """Let the client's file chooser save ``data``; ``on_saved`` is called if it did, with
    the path if the client tells it (a browser doesn't, and can't report a cancel either)."""
    picker = _file_picker()
    if picker is None:
        return

    async def run() -> None:
        path = await picker.save_file(dialog_title=title, file_name=name, src_bytes=data)
        if path:
            on_saved(Path(path))
        elif host.page is not None and host.page.web:
            on_saved(None)

    runtime.spawn(run)
