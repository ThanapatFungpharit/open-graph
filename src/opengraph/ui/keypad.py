"""A row of keys for what a phone's keyboard makes hard to reach.

The typeset editor moves between its fields with arrow keys, and soft keyboards have none;
``^``, ``/`` and the brackets are a keyboard layer away, and θ and π aren't there at all.
Keys act on the focused editor. They are plain gesture detectors, so pressing one neither
takes the focus from the editor nor closes the soft keyboard.
"""

from __future__ import annotations

from functools import partial

import flet as ft

from opengraph.ui import theme
from opengraph.ui.mathedit.widget import focused_editor

KEY_SIZE = (46, 42)

# The label, and what it does: a key press if the action names a key, else typed text.
KEYS: tuple[tuple[str, str], ...] = (
    ("←", "Arrow Left"),
    ("→", "Arrow Right"),
    ("⌫", "Backspace"),
    ("^", "^"),
    ("/", "/"),
    ("(", "("),
    (")", ")"),
    ("√", "sqrt("),
    ("x", "x"),
    ("y", "y"),
    ("z", "z"),
    ("=", "="),
    ("+", "+"),
    ("−", "-"),
    ("×", "*"),
    (",", ","),
    (".", "."),
    ("π", "π"),
    ("θ", "θ"),
    ("⇥", "Tab"),
)
_KEY_NAMES = frozenset({"Arrow Left", "Arrow Right", "Backspace", "Tab"})


def press(action: str) -> bool:
    """Do what a key does, to the focused editor; False if there is none."""
    editor = focused_editor()
    if editor is None:
        return False
    if action in _KEY_NAMES:
        editor.handle_key(action)
    else:
        editor.type_text(action)
    return True


class MathKeypad:
    def __init__(self) -> None:
        self._labels: list[ft.Text] = []
        self._faces: list[ft.Container] = []
        keys: list[ft.Control] = []
        for label, action in KEYS:
            text = ft.Text(label, size=17)
            face = ft.Container(
                text,
                width=KEY_SIZE[0],
                height=KEY_SIZE[1],
                border_radius=6,
                alignment=ft.Alignment.CENTER,
            )
            self._labels.append(text)
            self._faces.append(face)
            keys.append(ft.GestureDetector(content=face, on_tap=partial(self._tapped, action)))
        self.control = ft.Container(
            ft.Row(keys, spacing=4, scroll=ft.ScrollMode.HIDDEN),
            padding=ft.Padding.symmetric(horizontal=4, vertical=3),
            visible=False,
        )
        self.apply_theme()

    def _tapped(self, action: str, _event: object) -> None:
        press(action)

    def apply_theme(self) -> None:
        current = theme.current()
        self.control.bgcolor = current.panel
        self.control.border = ft.Border(top=ft.BorderSide(1, current.row_border))
        for face, text in zip(self._faces, self._labels, strict=True):
            face.bgcolor = current.row
            text.color = current.text
