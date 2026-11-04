"""The expression list: one editable row per expression or data table.

Rows only emit *intents* (text edited, toggle visibility, delete, ...) tagged with the
row id; the controller decides what to do with them. The panel never parses or
evaluates anything. Rows are reordered by dragging their grip; ``row_moved`` reports
where a row was dropped. Colours come from the current :mod:`~opengraph.ui.theme`.

Each row class owns its Flet controls (``.control``) and works without a page, so the
controller and the tests drive the same objects the window shows.
"""

from __future__ import annotations

import copy
import csv
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path

import flet as ft

from opengraph import colors, events
from opengraph.document import MAX_TABLE_ROWS, PALETTE, SliderRange, TableData
from opengraph.engine.compiler import compile_with_values, split_column_header
from opengraph.engine.errors import MathError
from opengraph.engine.formatting import format_value
from opengraph.ui import dialogs, fonts, runtime, theme
from opengraph.ui.geometry import Point
from opengraph.ui.mathedit.nodes import GREEK
from opengraph.ui.mathedit.widget import MathEdit

# Kept for compatibility; the live colours come from theme.current().
ERROR_COLOR = theme.LIGHT.error
INFO_COLOR = theme.LIGHT.info

CURVE_ORDER = ("derivative", "integral", "simplify", "solve")
CURVE_OPERATIONS = frozenset(CURVE_ORDER)
TABLE_STYLE_LABELS = {"points": "Points", "lines": "Lines", "both": "Points and Lines"}
TABLE_VISIBLE_ROWS = 8
TABLE_SHOWN_ROWS = 500  # rows a table shows for editing (the rest are kept and used)
CELL_HEIGHT = 26
GUTTER = 46  # width of the grip and the visibility dot, so sliders line up with the text


def _border(current: bool, error_tint: bool = False) -> ft.Border:
    t = theme.current()
    return ft.Border(
        left=ft.BorderSide(3, t.accent if current else "#00000000"),
        bottom=ft.BorderSide(1, t.row_border),
    )


def _is_hover(event: ft.Event[ft.Container]) -> bool:
    return event.data == "true" or event.data is True


def _style_actions(row: ExpressionRow | TableRow) -> None:
    """Show a row's menu and delete buttons only while it is hovered or being edited, or
    always on a touch device, which has no hover.

    Hidden buttons only become transparent: they stay clickable and the layout never
    shifts.
    """
    current = theme.current()
    shown = row.hover or row.current or runtime.device.touch
    for button in (row.menu_button, row.delete_button):
        button.opacity = 1.0 if shown else 0.0
        button.icon_color = current.muted


def _footer_button(label: str, icon: ft.IconData, tooltip: str) -> ft.Container:
    """An "add" button of the panel footer: accent icon and label, tinted on hover."""
    button = ft.Container(
        ft.Row(
            [ft.Icon(icon, size=16), ft.Text(label, weight=ft.FontWeight.W_600)],
            spacing=6,
            tight=True,
        ),
        padding=ft.Padding.symmetric(horizontal=12, vertical=10),
        tooltip=tooltip,
    )

    def on_hover(event: ft.Event[ft.Container]) -> None:
        current = theme.current()
        button.bgcolor = current.hover if _is_hover(event) else current.row
        runtime.push(button)

    button.on_hover = on_hover
    return button


class VisibilityButton:
    """A colour dot: filled when the graph is shown, a hollow ring when hidden."""

    def __init__(self) -> None:
        self.toggled = events.Signal()  # bool
        self._checked = True
        self._color = PALETTE[0]
        self._has_graph = True
        self._hover = False
        self.dot = ft.Container(width=18, height=18, border_radius=9)
        self.control = ft.Container(
            content=self.dot,
            width=26,
            height=26,
            border_radius=13,
            alignment=ft.Alignment.CENTER,
            on_click=lambda _: self.click(),
            on_hover=self._on_hover,
        )
        self._style()

    def is_checked(self) -> bool:
        return self._checked

    def set_checked(self, checked: bool) -> None:
        self._checked = checked
        self._style()

    def click(self) -> None:
        self._checked = not self._checked
        self._style()
        self.toggled.emit(self._checked)

    def set_color(self, color: str) -> None:
        self._color = color
        self._style()

    def set_has_graph(self, has_graph: bool) -> None:
        if has_graph != self._has_graph:
            self._has_graph = has_graph
            self._style()

    def _on_hover(self, event: ft.Event[ft.Container]) -> None:
        self._hover = _is_hover(event)
        self._style()

    def _style(self) -> None:
        color = self._color if self._has_graph else colors.with_alpha(self._color, 70)
        if self._checked:
            self.dot.bgcolor, self.dot.border = color, None
        else:
            self.dot.bgcolor = None
            self.dot.border = ft.Border.all(2, color)
        ring = colors.with_alpha(theme.current().text, 60)
        self.control.border = ft.Border.all(1, ring) if self._hover else None
        self.control.tooltip = "Hide graph" if self._checked else "Show graph"
        runtime.push(self.control)


def _drag_handle() -> ft.Control:
    return ft.ReorderableDragHandle(
        content=ft.Container(
            ft.Icon(ft.Icons.DRAG_INDICATOR, size=16, color=theme.current().muted),
            width=12,
            height=26,
            alignment=ft.Alignment.CENTER,
            tooltip="Drag to reorder",
        ),
        mouse_cursor=ft.MouseCursor.GRAB,
    )


class ExpressionEdit(MathEdit):
    """The row's native math editor: source stays in OpenGraph syntax.

    Enter submits, Up/Down move between rows (unless there is a fraction or exponent to
    move into), Backspace on an empty row deletes it. ``text()`` is the row's source.
    """

    def __init__(self, text: str = "") -> None:
        super().__init__(text)
        self.error_color = theme.current().error


class BoundEdit(MathEdit):
    """A slider's minimum or maximum, typeset like the row itself (``2π``, ``a/2``).

    Sized to its content, centred and drawn muted until focused, with an underline on
    hover and focus. ``editing_finished`` fires on Enter and when focus leaves.
    """

    MIN_WIDTH = 28
    MAX_WIDTH = 120

    def __init__(self, text: str = "") -> None:
        self.editing_finished = events.Signal()
        self._hover = False
        super().__init__(text)
        self.centered = True
        self.submitted.connect(self.editing_finished.emit)
        self.focus_changed.connect(self._on_focus_changed)
        self.detector.on_enter = lambda _: self._set_hover(True)
        self.detector.on_exit = lambda _: self._set_hover(False)
        self._fit()

    def _fit(self) -> None:
        width = int(self.math_layout().width + 0.999) + 8
        fitted = max(self.MIN_WIDTH, min(width, self.MAX_WIDTH))
        if self.control.width != fitted:
            self.control.width = fitted
            runtime.push(self.control)
        self.resize(fitted)

    def _refresh(self) -> None:
        super()._refresh()
        if hasattr(self, "control"):
            self._fit()

    def _set_hover(self, hover: bool) -> None:
        self._hover = hover
        self.update()

    def _on_focus_changed(self, focused: bool) -> None:
        if not focused:
            self.editing_finished.emit()

    def ink(self) -> str:
        current = theme.current()
        return current.text if self.has_focus else current.muted

    def render(self) -> list[ft.canvas.Shape]:
        shapes = super().render()
        current = theme.current()
        color = current.accent if self.has_focus else current.row_border if self._hover else None
        if color is not None:
            y = self.content_height() - 1
            shapes.append(
                ft.canvas.Line(0, y, self.width, y, paint=ft.Paint(color=color, stroke_width=1))
            )
        return shapes


class SuggestionBar:
    """A label and one button per name, like "Solve for: k q" under a row.

    ``chosen`` carries the clicked name, or every name for the optional "all" button.
    """

    def __init__(
        self, label: str, tooltip: Callable[[str], str], *, all_tooltip: str | None = None
    ) -> None:
        self.chosen = events.Signal()  # list of names
        self._tooltip = tooltip
        self.caption = ft.Text(label, size=12, color=theme.current().muted)
        self.all_button: ft.TextButton | None = None
        if all_tooltip is not None:
            self.all_button = self._button("all", all_tooltip, italic=False)
            self.all_button.on_click = lambda _: self.chosen.emit(list(self._names))
        self.buttons: list[ft.TextButton] = []
        self._names: tuple[str, ...] = ()
        self.row = ft.Row([self.caption], spacing=4, wrap=True, run_spacing=2)
        self.control = ft.Container(
            self.row, padding=ft.Padding.symmetric(vertical=2), visible=False
        )
        self._layout()

    @staticmethod
    def _button(text: str, tooltip: str, *, italic: bool = True) -> ft.TextButton:
        current = theme.current()
        return ft.TextButton(
            content=ft.Text(
                text, italic=italic, size=13, color=current.accent, font_family=fonts.SERIF
            ),
            tooltip=tooltip,
            style=ft.ButtonStyle(
                padding=ft.Padding.symmetric(horizontal=6),
                shape=ft.RoundedRectangleBorder(radius=4),
                side=ft.BorderSide(1, current.row_border),
                visual_density=ft.VisualDensity.COMPACT,
            ),
            height=24,
        )

    def _layout(self) -> None:
        controls: list[ft.Control] = [self.caption]
        if self.all_button is not None:
            controls.append(self.all_button)
        controls.extend(self.buttons)
        self.row.controls = controls

    @property
    def names(self) -> tuple[str, ...]:
        return self._names

    @property
    def visible(self) -> bool:
        return bool(self.control.visible)

    def labels(self) -> list[str]:
        return [
            str(b.content.value) if isinstance(b.content, ft.Text) else "" for b in self.buttons
        ]

    def choose(self, name: str) -> None:
        """Click the button of ``name``."""
        self.chosen.emit([name])

    def choose_all(self) -> None:
        self.chosen.emit(list(self._names))

    def set_names(self, names: tuple[str, ...], *, enabled: bool = True) -> None:
        """One button per name; no names hides the bar."""
        if names != self._names:
            self.buttons = []
            for name in names:
                button = self._button(GREEK.get(name, name), self._tooltip(name))
                button.on_click = lambda _, n=name: self.chosen.emit([n])
                self.buttons.append(button)
            self._names = names
            self._layout()
        if self.all_button is not None:
            self.all_button.visible = len(names) > 1
        self.control.visible = bool(names)
        for button in self.buttons:
            button.disabled = not enabled
        runtime.push(self.control)


RangeEvaluator = Callable[[Sequence[str]], SliderRange]


def constant_slider_range(texts: Sequence[str]) -> SliderRange:
    """A slider range from min/max/step texts that use no other rows (``2π``, ``1/3``)."""
    _, values = compile_with_values([], [(i, text, None) for i, text in enumerate(texts)])
    return SliderRange.from_texts(
        texts, [v.message if isinstance(v, MathError) else v for v in values.values()]
    )


class SliderSettingsDialog:
    """Edit a slider's minimum, maximum and step (numbers or expressions).

    ``evaluate`` turns the three texts into a range or raises ValueError; by default
    expressions may only use built-in names. ``on_accept`` gets the new range.
    """

    def __init__(
        self,
        current: SliderRange,
        evaluate: RangeEvaluator | None = None,
        on_accept: Callable[[SliderRange], object] | None = None,
    ) -> None:
        self.range = current
        self.accepted = False
        self._evaluate = evaluate or constant_slider_range
        self._on_accept = on_accept
        hint = "a number or an expression, like 2π"
        self.min_edit, self.max_edit, self.step_edit = (
            ft.TextField(
                value=text,
                label=label,
                hint_text=hint,
                dense=True,
                on_submit=lambda _: self.accept(),
            )
            for text, label in zip(current.texts, ("Minimum", "Maximum", "Step"), strict=True)
        )
        self.error = ft.Text("", color=theme.current().error, visible=False, size=12)
        self.dialog = ft.AlertDialog(
            title="Slider Settings",
            content=ft.Column(
                [self.min_edit, self.max_edit, self.step_edit, self.error],
                tight=True,
                spacing=10,
                width=320,
            ),
            actions=[
                ft.TextButton("Cancel", on_click=lambda _: self.reject()),
                ft.FilledButton("OK", on_click=lambda _: self.accept()),
            ],
        )

    def show(self) -> None:
        dialogs.host.show(self.dialog)

    def accept(self) -> None:
        texts = [(e.value or "").strip() for e in (self.min_edit, self.max_edit, self.step_edit)]
        try:
            self.range = self._evaluate(texts)
        except ValueError as exc:
            self.error.value = str(exc)
            self.error.visible = True
            runtime.push(self.error)
            return
        self.accepted = True
        dialogs.host.close(self.dialog)
        if self._on_accept is not None:
            self._on_accept(self.range)

    def reject(self) -> None:
        dialogs.host.close(self.dialog)


def _menu_item(
    label: str, on_click: Callable[[], object], *, disabled: bool = False
) -> ft.PopupMenuItem:
    return ft.PopupMenuItem(label, on_click=lambda _: on_click(), disabled=disabled)


class ExpressionRow:
    def __init__(self, item_id: int, text: str, color: str, visible: bool) -> None:
        self.text_edited = events.Signal()  # item id, text
        self.visibility_toggled = events.Signal()  # item id, bool
        self.color_chosen = events.Signal()  # item id, colour
        self.delete_requested = events.Signal()  # item id
        self.symbolic_requested = events.Signal()  # item id, operation
        self.solve_for_requested = events.Signal()  # item id, undefined name
        self.sliders_requested = events.Signal()  # item id, undefined names
        self.focused = events.Signal()  # item id
        self.submitted = events.Signal()  # item id
        self.navigate = events.Signal()  # item id, direction
        self.slider_moved = events.Signal()  # item id, new value
        self.slider_range_edited = events.Signal()  # item id, min, max, step texts

        self.item_id = item_id
        self._color = color
        self._error_span: tuple[int, int] | None = None
        self.error = False  # the row has an error (tinted)
        self.current = False  # the row is being edited (accent bar)
        self.hover = False  # the pointer is over the row (shows its action buttons)

        self.handle = _drag_handle()
        self.toggle = VisibilityButton()
        self.toggle.set_color(color)
        self.toggle.set_checked(visible)
        self.toggle.toggled.connect(lambda on: self.visibility_toggled.emit(self.item_id, on))

        self.edit = ExpressionEdit(text)
        self.edit.set_placeholder("Type an expression, e.g. y = x^2")
        self.edit.text_edited.connect(self._on_text_edited)
        self.edit.submitted.connect(lambda: self.submitted.emit(self.item_id))
        self.edit.navigate.connect(lambda d: self.navigate.emit(self.item_id, d))
        self.edit.delete_empty.connect(lambda: self.delete_requested.emit(self.item_id))
        self.edit.focus_changed.connect(self._on_focus_changed)

        self.feedback = ft.Text("", size=12, selectable=True, visible=False)
        self._feedback_error = False

        # Suggestions when the row uses names nothing defines: "Add slider: all k q",
        # "Solve for: k q".
        self.slider_suggestions = SuggestionBar(
            "Add slider:",
            lambda name: f"Add {name} = 1 below, with a slider",
            all_tooltip="Add a slider for every undefined name",
        )
        self.slider_suggestions.chosen.connect(
            lambda names: self.sliders_requested.emit(self.item_id, names)
        )
        self.solve_bar = SuggestionBar(
            "Solve for:", lambda name: f"Rearrange this row for {name} and add the result below"
        )
        self.solve_bar.chosen.connect(
            lambda names: self.solve_for_requested.emit(self.item_id, names[0])
        )

        self._slider_range: SliderRange | None = None
        # Checks the Slider Settings dialog's input; the panel sets one that knows the
        # other rows (so a bound can be ``a + 1``).
        self.evaluate_range: RangeEvaluator | None = None
        self.slider = ft.Slider(
            min=0,
            max=1,
            value=0,
            expand=True,
            tooltip="Drag to change the value",
            on_change=self._on_slider_change,
        )
        self.min_edit = self._bound_edit("Slider minimum: a number or an expression")
        self.max_edit = self._bound_edit("Slider maximum: a number or an expression")
        self.slider_bar = ft.Container(
            ft.Row(
                [self.min_edit.control, self.slider, self.max_edit.control],
                spacing=6,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            padding=ft.Padding.only(left=GUTTER, top=2, bottom=2),
            visible=False,
        )

        self.symbolic_actions = [
            _menu_item(label, partial(self._request_symbolic, op))
            for label, op in (
                ("Add Derivative", "derivative"),
                ("Add Integral", "integral"),
                ("Add Simplified Form", "simplify"),
                ("Solve = 0 (Mark Roots)", "solve"),
            )
        ]
        self.exact_action = _menu_item(
            "Eval Exact Value",
            lambda: self.symbolic_requested.emit(self.item_id, "exact"),
            disabled=True,
        )
        self.slider_settings_action = _menu_item(
            "Slider Settings…", self._slider_settings, disabled=True
        )
        self.menu_button = ft.PopupMenuButton(
            icon=ft.Icons.MORE_VERT,
            icon_size=18,
            tooltip="More actions",
            animate_opacity=120,
            items=[
                _menu_item("Color…", self._custom_color),
                ft.PopupMenuItem(),
                *self.symbolic_actions,
                self.exact_action,
                ft.PopupMenuItem(),
                self.slider_settings_action,
                ft.PopupMenuItem(),
                _menu_item("Delete", lambda: self.delete_requested.emit(self.item_id)),
            ],
        )
        self.delete_button = ft.IconButton(
            ft.Icons.CLOSE,
            icon_size=16,
            tooltip="Delete expression",
            animate_opacity=120,
            on_click=lambda _: self.delete_requested.emit(self.item_id),
        )

        text_column = ft.Column(
            [
                self.edit.control,
                self.feedback,
                self.slider_suggestions.control,
                self.solve_bar.control,
            ],
            spacing=2,
            expand=True,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
        )
        self.control = ft.Container(
            ft.Column(
                [
                    ft.Row(
                        [
                            self.handle,
                            self.toggle.control,
                            text_column,
                            self.menu_button,
                            self.delete_button,
                        ],
                        spacing=4,
                        vertical_alignment=ft.CrossAxisAlignment.START,
                    ),
                    self.slider_bar,
                ],
                spacing=0,
            ),
            padding=ft.Padding.only(left=1, top=6, right=4, bottom=6),
            on_hover=self._on_hover,
        )
        self.set_color(color)
        self._style()

    def _bound_edit(self, tooltip: str) -> BoundEdit:
        edit = BoundEdit()
        edit.control.tooltip = tooltip
        edit.editing_finished.connect(self._on_bound_edited)
        return edit

    # -- menu ------------------------------------------------------------------------
    def _request_symbolic(self, operation: str) -> None:
        self.symbolic_requested.emit(self.item_id, operation)

    def choose_color(self, color: str) -> None:
        self.set_color(color)
        self.color_chosen.emit(self.item_id, color)

    def _custom_color(self) -> None:
        dialogs.ColorDialog("Expression Color", self._color, self.choose_color).show()

    def _slider_settings(self) -> None:
        if self._slider_range is None:
            return
        dialog = SliderSettingsDialog(
            self._slider_range,
            self.evaluate_range,
            lambda r: self.slider_range_edited.emit(self.item_id, *r.texts),
        )
        dialog.show()
        self.settings_dialog = dialog

    # -- slider ----------------------------------------------------------------------
    def _on_slider_change(self, event: ft.Event[ft.Slider]) -> None:
        self.move_slider(round(float(self.slider.value or 0)))

    def move_slider(self, position: int) -> None:
        """The user moved the slider to ``position`` (0 to its number of steps)."""
        if self._slider_range is not None:
            position = max(0, min(position, self._slider_range.steps))
            self.slider.value = position
            self.slider_moved.emit(self.item_id, self._slider_range.value_at(position))

    @property
    def slider_position(self) -> int:
        return round(float(self.slider.value or 0))

    @property
    def slider_maximum(self) -> int:
        return round(float(self.slider.max or 0))

    def _on_bound_edited(self) -> None:
        current = self._slider_range
        if current is None:
            return
        lo, hi = self.min_edit.text().strip(), self.max_edit.text().strip()
        old_lo, old_hi, step = current.texts
        if not lo or not hi:
            self._show_bounds(current)
        elif (lo, hi) != (old_lo, old_hi):
            self.slider_range_edited.emit(self.item_id, lo, hi, step)

    def _show_bounds(self, slider_range: SliderRange) -> None:
        lo, hi, _ = slider_range.texts
        for edit, text in ((self.min_edit, lo), (self.max_edit, hi)):
            if edit.text() != text:  # keeps the caret and undo history while unchanged
                edit.set_text(text)
        # An expression bound also shows its current value.
        for edit, expr, value, tip in (
            (self.min_edit, slider_range.min_expr, slider_range.min, "minimum"),
            (self.max_edit, slider_range.max_expr, slider_range.max, "maximum"),
        ):
            shown = f" = {format_value(value)}" if expr else ""
            edit.control.tooltip = f"Slider {tip}{shown}: a number or an expression"
            runtime.push(edit.control)

    # -- state from the controller --------------------------------------------------------
    def set_color(self, color: str) -> None:
        self._color = color
        self.toggle.set_color(color)
        current = theme.current()
        self.slider.active_color = color
        self.slider.thumb_color = color
        self.slider.inactive_color = current.track
        runtime.push(self.slider)

    def _style(self) -> None:
        current = theme.current()
        if self.current:
            background = current.row_current
        elif self.error:
            background = current.row_error
        else:
            background = current.row
        self.control.bgcolor = background
        self.control.border = _border(self.current)
        _style_actions(self)
        color = current.error if self._feedback_error else current.info
        self.feedback.color = color
        self.edit.error_color = current.error
        for bar in (self.slider_suggestions, self.solve_bar):
            bar.caption.color = current.muted
        runtime.push(self.control)

    def apply_theme(self) -> None:
        """Re-read theme colours (row tint, feedback text, slider, error underline)."""
        self.set_color(self._color)
        self._style()
        for edit in (self.edit, self.min_edit, self.max_edit):
            edit.update()

    def set_slider(self, value: float | None, slider_range: SliderRange | None = None) -> None:
        """Show the slider at ``value`` within ``slider_range``, or hide it (None)."""
        shown = value is not None and slider_range is not None
        self.slider_settings_action.disabled = not shown
        self.slider_bar.visible = shown
        if value is None or slider_range is None:
            self._slider_range = None
            runtime.push(self.slider_bar, self.menu_button)
            return
        self._slider_range = slider_range
        # A programmatic move is not an edit: Flet reports only the user's changes.
        self.slider.min = 0
        self.slider.max = max(1, slider_range.steps)
        self.slider.divisions = max(1, slider_range.steps)
        self.slider.value = slider_range.position_of(value)
        self._show_bounds(slider_range)
        runtime.push(self.slider_bar, self.menu_button)

    def set_solve_for(self, names: tuple[str, ...], *, enabled: bool = True) -> None:
        """Show a "Solve for" button per undefined name (none hides the bar)."""
        self.solve_bar.set_names(names, enabled=enabled)

    def set_slider_suggestions(self, names: tuple[str, ...]) -> None:
        """Show an "Add slider" button per undefined name (none hides the bar)."""
        self.slider_suggestions.set_names(names)

    def set_symbolic_enabled(self, enabled: bool) -> None:
        """Enable or disable the curve operations (derivative, integral, ...)."""
        self.set_symbolic_operations(CURVE_OPERATIONS if enabled else frozenset())

    def set_symbolic_operations(self, operations: frozenset[str]) -> None:
        for action, operation in zip(self.symbolic_actions, CURVE_ORDER, strict=True):
            action.disabled = operation not in operations
        self.exact_action.disabled = "exact" not in operations
        runtime.push(self.menu_button)

    def set_feedback(
        self,
        message: str,
        *,
        error: bool,
        span: tuple[int, int] | None = None,
        has_graph: bool = True,
    ) -> None:
        self.toggle.set_has_graph(has_graph)
        self.feedback.visible = bool(message)
        self.feedback.value = message
        self._feedback_error = error
        self.error = error
        self._style()
        self._error_span = span if error else None
        self.edit.set_error_span(self._error_span)

    def replace_text(self, text: str) -> None:
        """Replace the text keeping it on the edit's undo stack (Ctrl+Z restores it)."""
        self.edit.replace_text(text)
        self._on_text_edited(text)

    def _on_text_edited(self, text: str) -> None:
        # Stale underline would point at the wrong characters until the next compile.
        self.edit.set_error_span(None)
        self.text_edited.emit(self.item_id, text)

    def _on_focus_changed(self, focused: bool) -> None:
        self.current = focused
        self._style()
        if focused:
            self.focused.emit(self.item_id)

    def _on_hover(self, event: ft.Event[ft.Container]) -> None:
        self.hover = _is_hover(event)
        self._style()


class TableRow:
    """A data table in the expression list: named columns of values.

    The first column holds x values, each other column y values; each named column also
    defines a list (``x_1``) usable in expressions. Every edit emits ``table_changed``
    with a fresh copy of the whole :class:`TableData`.
    """

    def __init__(
        self,
        item_id: int,
        data: TableData,
        color: str,
        visible: bool,
        *,
        taken_names: Callable[[], set[str]] | None = None,
    ) -> None:
        self.table_changed = events.Signal()  # item id, TableData
        self.visibility_toggled = events.Signal()
        self.color_chosen = events.Signal()
        self.delete_requested = events.Signal()
        self.focused = events.Signal()

        self.item_id = item_id
        self.data = copy.deepcopy(data)
        # Column names in use elsewhere (other tables), so a new column gets a free one.
        self._taken_names = taken_names or set
        self._color = color
        self._cell_errors: dict[tuple[int, int], str] = {}
        self._table_values: dict[tuple[int, int], str] = {}
        self._loading = False
        self._focus_count = 0
        self.current = False
        self.hover = False

        self.handle = _drag_handle()
        self.toggle = VisibilityButton()
        self.toggle.set_color(color)
        self.toggle.set_checked(visible)
        self.toggle.toggled.connect(lambda on: self.visibility_toggled.emit(self.item_id, on))

        self.title = ft.Text("Table", size=12, weight=ft.FontWeight.W_500)
        self.feedback = ft.Text("", size=12, visible=False, selectable=True)
        self._feedback_error = False
        self.name_edits: list[ft.TextField] = []
        self.names_bar = ft.Row(spacing=0)
        self.cells: list[list[ft.TextField]] = []
        self.grid = ft.Column(spacing=0, scroll=ft.ScrollMode.AUTO)
        self.more = ft.Text("", size=11, visible=False)

        self.style_actions = {
            style: ft.PopupMenuItem(
                f"Style: {label}",
                checked=style == self.data.style,
                on_click=lambda _, s=style: self.set_style(s),
            )
            for style, label in TABLE_STYLE_LABELS.items()
        }
        self.remove_column_action = _menu_item("Remove Last Column", self.remove_column)
        self.menu_button = ft.PopupMenuButton(
            icon=ft.Icons.MORE_VERT,
            icon_size=18,
            tooltip="More actions",
            animate_opacity=120,
            items=[
                _menu_item("Color…", self._custom_color),
                *self.style_actions.values(),
                ft.PopupMenuItem(),
                _menu_item("Add Column", self.add_column),
                self.remove_column_action,
                _menu_item("Import CSV…", self._import_csv),
                ft.PopupMenuItem(),
                _menu_item("Delete", lambda: self.delete_requested.emit(self.item_id)),
            ],
        )
        self.delete_button = ft.IconButton(
            ft.Icons.CLOSE,
            icon_size=16,
            tooltip="Delete table",
            animate_opacity=120,
            on_click=lambda _: self.delete_requested.emit(self.item_id),
        )
        column = ft.Column(
            [self.title, self.names_bar, self.grid, self.more, self.feedback],
            spacing=2,
            expand=True,
        )
        self.control = ft.Container(
            ft.Row(
                [self.handle, self.toggle.control, column, self.menu_button, self.delete_button],
                spacing=4,
                vertical_alignment=ft.CrossAxisAlignment.START,
            ),
            padding=ft.Padding.only(left=1, top=6, right=4, bottom=6),
            on_hover=self._on_hover,
        )
        self._load()
        self._style()

    # -- menu ------------------------------------------------------------------------
    def choose_color(self, color: str) -> None:
        self.set_color(color)
        self.color_chosen.emit(self.item_id, color)

    def _custom_color(self) -> None:
        dialogs.ColorDialog("Table Color", self._color, self.choose_color).show()

    def set_style(self, style: str) -> None:
        self.data.style = style
        for name, action in self.style_actions.items():
            action.checked = name == style
        runtime.push(self.menu_button)
        self._emit()

    def add_column(self) -> None:
        taken = {split_column_header(n)[0] for n in self.data.names} | self._taken_names()
        index = 2
        while f"y_{index}" in taken:
            index += 1
        self.data.names.append(f"y_{index}")
        self.data.columns.append([])
        self._load()
        self._emit()

    def remove_column(self) -> None:
        if len(self.data.names) > 2:
            self.data.names.pop()
            self.data.columns.pop()
            self._load()
            self._emit()

    def _import_csv(self) -> None:
        dialogs.ask_open_path("Import CSV", Path.home(), ["csv", "txt"], self.import_csv)

    def import_csv(self, path: Path) -> None:
        try:
            data = read_csv(path, self.data)
        except (OSError, UnicodeDecodeError, csv.Error, ValueError) as exc:
            dialogs.message("Import Failed", f"Could not import {path.name}: {exc}")
            return
        self.set_data(data)
        self._emit()

    # -- cells -----------------------------------------------------------------------
    @property
    def row_count(self) -> int:
        """Rows shown for editing (the data rows and one spare row)."""
        return len(self.cells)

    @property
    def column_count(self) -> int:
        return len(self.data.names)

    def cell(self, row: int, column: int) -> ft.TextField:
        return self.cells[row][column]

    def set_cell_text(self, row: int, column: int, text: str) -> None:
        """Type ``text`` into a cell (as the user would)."""
        field = self.cells[row][column]
        field.value = text
        self._on_cell_changed(row, column)

    def set_name(self, column: int, text: str) -> None:
        """Type a column name and leave the field (as the user would)."""
        self.name_edits[column].value = text
        self._on_name_edited(column)

    def set_data(self, data: TableData) -> None:
        self.data = copy.deepcopy(data)
        self._load()

    def _load(self) -> None:
        """Rebuild the name fields and cells from ``self.data``."""
        self._loading = True
        try:
            self.name_edits = [self._name_edit(c, name) for c, name in enumerate(self.data.names)]
            self.names_bar.controls = list(self.name_edits)
            self.cells = []
            total = self.data.row_count + 1  # a spare row to type into
            for r in range(min(total, TABLE_SHOWN_ROWS)):
                self.cells.append(self._cell_row(r))
            self.grid.controls = [ft.Row(list(row), spacing=0) for row in self.cells]
            self.more.value = (
                f"Showing {TABLE_SHOWN_ROWS} of {total - 1} rows"
                if total > TABLE_SHOWN_ROWS
                else ""
            )
            self.more.visible = total > TABLE_SHOWN_ROWS
            self.remove_column_action.disabled = len(self.data.names) <= 2
            for style, action in self.style_actions.items():
                action.checked = style == self.data.style
            self._apply_cell_errors()
            self._fit_height()
        finally:
            self._loading = False
        runtime.push(self.control)

    def _name_edit(self, column: int, name: str) -> ft.TextField:
        tooltip = (
            "Column name: x values" if column == 0 else "Column name: y values"
        ) + ". Write name = formula (like y_2 = x_1^2) to compute the column."
        return ft.TextField(
            value=name,
            text_align=ft.TextAlign.CENTER,
            text_style=ft.TextStyle(italic=True, size=13, font_family=fonts.SERIF),
            dense=True,
            expand=True,
            border=ft.UnderlineInputBorder(),
            content_padding=ft.Padding.symmetric(vertical=2),
            tooltip=tooltip,
            on_blur=lambda _, c=column: self._on_name_edited(c),
            on_submit=lambda _, c=column: self._on_name_edited(c),
            on_focus=lambda _: self._on_focus(True),
        )

    def _is_formula_column(self, column: int) -> bool:
        return split_column_header(self.data.names[column])[1] is not None

    def _cell_row(self, row: int) -> list[ft.TextField]:
        return [self._cell(row, c) for c in range(len(self.data.names))]

    def _cell(self, row: int, column: int) -> ft.TextField:
        """A cell; those of a formula column show computed values and can't be edited."""
        formula = self._is_formula_column(column)
        text = self._table_values.get((row, column), "") if formula else self.data.cell(row, column)
        current = theme.current()
        return ft.TextField(
            value=text,
            read_only=formula,
            dense=True,
            expand=True,
            height=CELL_HEIGHT,
            text_size=13,
            content_padding=ft.Padding.symmetric(horizontal=4, vertical=4),
            border=ft.OutlineInputBorder(
                side=ft.BorderSide(0.5, current.row_border), border_radius=0
            ),
            color=current.muted if formula else current.text,
            on_change=lambda _, r=row, c=column: self._on_cell_changed(r, c),
            on_focus=lambda _: self._on_focus(True),
            on_blur=lambda _: self._on_focus(False),
        )

    def set_table_values(self, values: dict[tuple[int, int], str]) -> None:
        """Show the computed cells of the formula columns."""
        self._table_values = dict(values)
        for r, row in enumerate(self.cells):
            for c, field in enumerate(row):
                if self._is_formula_column(c):
                    field.value = self._table_values.get((r, c), "")
        runtime.push(self.grid)

    def _fit_height(self) -> None:
        shown = min(len(self.cells), TABLE_VISIBLE_ROWS)
        self.grid.height = shown * CELL_HEIGHT + 2

    def _on_name_edited(self, column: int) -> None:
        self._on_focus(False)
        name = (self.name_edits[column].value or "").strip()
        if name != self.data.names[column]:
            was_formula = self._is_formula_column(column)
            self.data.names[column] = name
            if was_formula != self._is_formula_column(column):
                self._load()  # the column's cells become computed (or editable)
            self._emit()

    def _on_cell_changed(self, row: int, column: int) -> None:
        if self._loading or row >= MAX_TABLE_ROWS or self._is_formula_column(column):
            return
        text = (self.cells[row][column].value or "").strip()
        self.data.set_cell(row, column, text)
        if row == len(self.cells) - 1 and text and len(self.cells) < TABLE_SHOWN_ROWS:
            # Keep one empty row at the bottom (the cell being edited stays focused).
            new_row = self._cell_row(row + 1)
            self.cells.append(new_row)
            self.grid.controls.append(ft.Row(list(new_row), spacing=0))
            self._fit_height()
            runtime.push(self.grid)
        self._emit()

    def _emit(self) -> None:
        self.table_changed.emit(self.item_id, copy.deepcopy(self.data))

    def _on_focus(self, focused: bool) -> None:
        self._focus_count = max(0, self._focus_count + (1 if focused else -1))
        current = self._focus_count > 0
        if focused:
            runtime.focus_tracker.gained(self)
        else:
            runtime.focus_tracker.lost(self)
        if current != self.current:
            self.current = current
            self._style()
        if focused:
            self.focused.emit(self.item_id)

    def set_color(self, color: str) -> None:
        self._color = color
        self.toggle.set_color(color)

    def _on_hover(self, event: ft.Event[ft.Container]) -> None:
        self.hover = _is_hover(event)
        self._style()

    def apply_theme(self) -> None:
        self._load()
        self._style()

    def _style(self) -> None:
        current = theme.current()
        self.control.bgcolor = current.row_current if self.current else current.row
        self.control.border = _border(self.current)
        _style_actions(self)
        self.title.color = current.muted
        self.more.color = current.muted
        self.feedback.color = current.error if self._feedback_error else current.info
        runtime.push(self.control)

    def set_feedback(
        self,
        message: str,
        *,
        error: bool,
        span: tuple[int, int] | None = None,
        has_graph: bool = True,
    ) -> None:
        self.toggle.set_has_graph(has_graph)
        self.feedback.visible = bool(message)
        self.feedback.value = message
        self._feedback_error = error
        self._style()

    def set_cell_errors(self, errors: dict[tuple[int, int], str]) -> None:
        self._cell_errors = dict(errors)
        self._apply_cell_errors()
        runtime.push(self.grid)

    def _apply_cell_errors(self) -> None:
        tint = colors.with_alpha(theme.current().error, 50)
        for r, row in enumerate(self.cells):
            for c, field in enumerate(row):
                message = self._cell_errors.get((r, c))
                field.bgcolor = tint if message else None
                field.filled = bool(message)
                field.fill_color = tint if message else None
                field.tooltip = message or None

    # Rows share this interface with ExpressionRow so the controller treats them alike.
    def set_symbolic_operations(self, operations: frozenset[str]) -> None:
        pass

    def set_solve_for(self, names: tuple[str, ...], *, enabled: bool = True) -> None:
        pass

    def set_slider_suggestions(self, names: tuple[str, ...]) -> None:
        pass

    def set_symbolic_enabled(self, enabled: bool) -> None:
        pass

    def set_slider(self, value: float | None, slider_range: SliderRange | None = None) -> None:
        pass


def read_csv(path: Path, current: TableData) -> TableData:
    """Read a CSV file into a table: numeric-looking cells, with an optional header row.

    Keeps ``current``'s column names unless the file has a header of valid-looking
    names; uses as many columns as the file has (at least two).
    """
    with path.open(newline="", encoding="utf-8-sig") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        try:
            dialect: type[csv.Dialect] | csv.Dialect = csv.Sniffer().sniff(sample, ",;\t ")
        except csv.Error:
            dialect = csv.excel
        rows = [row for row in csv.reader(handle, dialect) if any(cell.strip() for cell in row)]
    if not rows:
        raise ValueError("the file is empty")

    def numeric(cell: str) -> bool:
        try:
            float(cell)
        except ValueError:
            return False
        return True

    names = list(current.names)
    if not all(numeric(cell) for cell in rows[0] if cell.strip()):
        header, rows = rows[0], rows[1:]
        cleaned = ["".join(ch for ch in cell if ch.isalnum() or ch == "_") for cell in header]
        if all(cleaned) and len(set(cleaned)) == len(cleaned):
            names = cleaned
    width = max(2, max(len(row) for row in rows) if rows else 2)
    while len(names) < width:
        names.append(f"y_{len(names)}")
    names = names[:width]
    columns = [[(row[c].strip() if c < len(row) else "") for row in rows] for c in range(width)]
    return TableData(names, [c[:MAX_TABLE_ROWS] for c in columns], current.style)


Row = ExpressionRow | TableRow


class ExpressionPanel:
    """Reorderable list of :class:`ExpressionRow` and :class:`TableRow` plus "add" buttons.

    Row signals are re-emitted by the panel so consumers connect once.
    """

    def __init__(self) -> None:
        self.add_requested = events.Signal()
        self.add_table_requested = events.Signal()
        self.text_edited = events.Signal()
        self.visibility_toggled = events.Signal()
        self.color_chosen = events.Signal()
        self.delete_requested = events.Signal()
        self.symbolic_requested = events.Signal()
        self.solve_for_requested = events.Signal()
        self.sliders_requested = events.Signal()
        self.submitted = events.Signal()
        self.navigate = events.Signal()
        self.slider_moved = events.Signal()
        self.slider_range_edited = events.Signal()
        self.table_changed = events.Signal()  # item id, TableData
        self.row_focused = events.Signal()
        self.linear_input_requested = events.Signal()  # a row's context menu: plain text
        self.row_moved = events.Signal()  # item id, new index (the panel has moved it)

        self._rows: dict[int, ExpressionRow] = {}
        self._tables: dict[int, TableRow] = {}
        self._order: list[int] = []
        self._linear = False
        # Turns a row's slider min/max/step texts into a range, using the other rows.
        self.evaluate_slider_range: Callable[[int, Sequence[str]], SliderRange] | None = None

        self.list = ft.ReorderableListView(
            controls=[],
            show_default_drag_handles=False,
            expand=True,
            padding=0,
            on_reorder=self._on_reorder,
        )
        self.add_button = _footer_button("Add expression", ft.Icons.ADD, "Add an expression")
        self.add_button.expand = True
        self.add_button.on_click = lambda _: self.add_requested.emit()
        self.add_table_button = _footer_button(
            "Table", ft.Icons.TABLE_CHART_OUTLINED, "Add a data table"
        )
        self.add_table_button.on_click = lambda _: self.add_table_requested.emit()
        self.control = ft.Container(
            ft.Column(
                [self.list, ft.Row([self.add_button, self.add_table_button], spacing=0)],
                spacing=0,
                expand=True,
            ),
            width=340,
        )
        self.apply_theme()

    def row(self, item_id: int) -> ExpressionRow:
        return self._rows[item_id]

    def table(self, item_id: int) -> TableRow:
        return self._tables[item_id]

    def widget(self, item_id: int) -> Row:
        """The row for an item, whichever kind it is."""
        return self._rows[item_id] if item_id in self._rows else self._tables[item_id]

    def row_ids(self) -> list[int]:
        return list(self._order)

    def _sync(self) -> None:
        self.list.controls = [self.widget(i).control for i in self._order]
        runtime.push(self.list)

    def insert_row(
        self, index: int, item_id: int, text: str, color: str, visible: bool
    ) -> ExpressionRow:
        row = ExpressionRow(item_id, text, color, visible)
        row.text_edited.connect(self.text_edited.emit)
        row.visibility_toggled.connect(self.visibility_toggled.emit)
        row.color_chosen.connect(self.color_chosen.emit)
        row.delete_requested.connect(self.delete_requested.emit)
        row.symbolic_requested.connect(self.symbolic_requested.emit)
        row.solve_for_requested.connect(self.solve_for_requested.emit)
        row.sliders_requested.connect(self.sliders_requested.emit)
        row.submitted.connect(self.submitted.emit)
        row.navigate.connect(self.navigate.emit)
        row.slider_moved.connect(self.slider_moved.emit)
        row.slider_range_edited.connect(self.slider_range_edited.emit)
        row.evaluate_range = lambda texts: self._evaluate_slider_range(item_id, texts)
        row.focused.connect(self.row_focused.emit)
        row.edit.linear_requested.connect(self.linear_input_requested.emit)
        row.edit.set_linear(self._linear)
        self._rows[item_id] = row
        self._order.insert(index, item_id)
        self._sync()
        return row

    def _evaluate_slider_range(self, item_id: int, texts: Sequence[str]) -> SliderRange:
        if self.evaluate_slider_range is None:
            return constant_slider_range(texts)
        return self.evaluate_slider_range(item_id, texts)

    def insert_table_row(
        self, index: int, item_id: int, data: TableData, color: str, visible: bool
    ) -> TableRow:
        row = TableRow(item_id, data, color, visible, taken_names=self._table_names)
        row.table_changed.connect(self.table_changed.emit)
        row.visibility_toggled.connect(self.visibility_toggled.emit)
        row.color_chosen.connect(self.color_chosen.emit)
        row.delete_requested.connect(self.delete_requested.emit)
        row.focused.connect(self.row_focused.emit)
        self._tables[item_id] = row
        self._order.insert(index, item_id)
        self._sync()
        return row

    # -- reordering ------------------------------------------------------------------------
    def _on_reorder(self, event: ft.OnReorderEvent) -> None:
        if event.old_index is None or event.new_index is None:
            return
        self.reorder(event.old_index, event.new_index)

    def reorder(self, old_index: int, new_index: int) -> None:
        """A row was dragged from ``old_index`` and dropped at ``new_index``."""
        if not 0 <= old_index < len(self._order):
            return
        item_id = self._order[old_index]
        new_index = max(0, min(new_index, len(self._order) - 1))
        self.move_row(item_id, new_index)
        self.row_moved.emit(item_id, new_index)

    def move_row(self, item_id: int, index: int) -> None:
        """Put a row at ``index`` in the list."""
        self._order.remove(item_id)
        self._order.insert(index, item_id)
        self._sync()

    def _table_names(self) -> set[str]:
        return {
            split_column_header(name)[0]
            for table in self._tables.values()
            for name in table.data.names
        }

    def remove_row(self, item_id: int) -> None:
        row = self._rows.pop(item_id) if item_id in self._rows else self._tables.pop(item_id)
        self._order.remove(item_id)
        if isinstance(row, ExpressionRow):
            for edit in (row.edit, row.min_edit, row.max_edit):
                edit._set_focus(False)
        self._sync()

    def clear(self) -> None:
        for item_id in [*self._rows, *self._tables]:
            self.remove_row(item_id)

    @property
    def linear_input(self) -> bool:
        return self._linear

    def set_linear_input(self, on: bool) -> None:
        """Edit every expression row as plain text (``on``) or as typeset mathematics."""
        self._linear = on
        for row in self._rows.values():
            row.edit.set_linear(on)

    def apply_theme(self) -> None:
        current = theme.current()
        self.control.bgcolor = current.panel
        border = ft.Border(top=ft.BorderSide(1, current.row_border))
        for button in (self.add_button, self.add_table_button):
            button.bgcolor = current.row
            button.border = border
            if isinstance(button.content, ft.Row):
                for part in button.content.controls:
                    if isinstance(part, ft.Icon | ft.Text):
                        part.color = current.accent
        self.add_table_button.border = ft.Border(
            top=ft.BorderSide(1, current.row_border), left=ft.BorderSide(1, current.row_border)
        )
        for row in self._rows.values():
            row.apply_theme()
        for table in self._tables.values():
            table.apply_theme()
        runtime.push(self.control)

    def focus_row(self, item_id: int, *, cursor_at_end: bool = True) -> None:
        if item_id in self._tables:
            table = self._tables[item_id]
            if table.cells:
                runtime.focus(table.cells[0][0])
            table.focused.emit(item_id)
            return
        row = self._rows[item_id]
        row.edit.focus()
        if cursor_at_end:
            row.edit.end(False)

    def click_point(self, item_id: int, point: Point) -> None:
        """Click a row's editor at ``point`` (editor coordinates)."""
        self._rows[item_id].edit.click(point)
