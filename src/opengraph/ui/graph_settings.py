"""The Graph Settings panel: axes, grid, angle unit and number mode.

Non-modal: it floats over the graph, every control applies immediately
(``settings_changed``), and the bounds fields follow the view as it is panned and
zoomed. Controls that a setting makes meaningless are disabled with an explanation
instead of silently ignored: a polar grid or complex mode needs linear axes, and a polar
grid keeps the aspect ratio.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import replace
from functools import partial

import flet as ft

from opengraph import events
from opengraph.engine.formatting import format_value
from opengraph.graph.settings import AxisSettings, GraphSettings
from opengraph.ui import runtime, theme


def _parse(text: str) -> float | None:
    try:
        value = float(text.strip().replace("−", "-"))
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _checkbox(label: str) -> ft.Checkbox:
    return ft.Checkbox(label=label, value=False)


class GraphSettingsDialog:
    def __init__(self, settings: GraphSettings) -> None:
        self.settings_changed = events.Signal()  # GraphSettings
        self.range_edited = events.Signal()  # axis, min, max
        self.closed = events.Signal()
        self._settings = settings
        self._updating = False
        self._editing: set[ft.TextField] = set()  # fields with the focus keep their text
        self.visible = False

        # -- grid ---------------------------------------------------------------
        self.grid_kind = ft.RadioGroup(
            ft.Row(
                [
                    ft.Radio(value="cartesian", label="Cartesian"),
                    ft.Radio(value="polar", label="Polar"),
                ]
            ),
            value="cartesian",
            on_change=lambda _: self._on_changed(),
        )
        self.major_grid = _checkbox("Major gridlines")
        self.minor_grid = _checkbox("Minor gridlines")
        self.axes = _checkbox("Axis lines")
        self.axis_numbers = _checkbox("Axis numbers")

        # -- axes ---------------------------------------------------------------
        self.bounds: dict[str, tuple[ft.TextField, ft.TextField]] = {}
        self.steps: dict[str, ft.TextField] = {}
        self.scales: dict[str, ft.Dropdown] = {}
        axis_rows: list[ft.Control] = [
            ft.Row(
                [
                    ft.Text(t, width=w, size=12)
                    for t, w in (
                        ("", 44),
                        ("Minimum", 90),
                        ("Maximum", 90),
                        ("Step", 70),
                        ("Scale", 130),
                    )
                ],
                spacing=6,
            )
        ]
        self._axis_rows = axis_rows
        for axis in ("x", "y"):
            edited = partial(self._on_bounds_edited, axis)
            lo, hi = self._field(90, edited), self._field(90, edited)
            step = self._field(70, self._on_changed, hint="auto")
            step.tooltip = "Major grid spacing; leave empty for automatic spacing"
            scale = ft.Dropdown(
                options=[
                    ft.DropdownOption(key="linear", text="Linear"),
                    ft.DropdownOption(key="log", text="Logarithmic"),
                ],
                value="linear",
                width=130,
                dense=True,
                on_select=lambda _: self._on_changed(),
            )
            self.bounds[axis] = (lo, hi)
            self.steps[axis] = step
            self.scales[axis] = scale
            axis_rows.append(
                ft.Row([ft.Text(f"{axis}-axis", width=44), lo, hi, step, scale], spacing=6)
            )
        # Only a 3D view has this axis; it shares the columns of the others.
        z_edited = partial(self._on_bounds_edited, "z")
        self.bounds["z"] = (self._field(90, z_edited), self._field(90, z_edited))
        self.z_row = ft.Row(
            [ft.Text("z-axis", width=44), *self.bounds["z"]], spacing=6, visible=False
        )
        axis_rows.append(self.z_row)
        self.lock_aspect = _checkbox("Lock aspect ratio (Ctrl/Shift+wheel zooms one axis when off)")
        self.note = ft.Text("", size=12, visible=False)

        # -- maths --------------------------------------------------------------
        self.angles = ft.RadioGroup(
            ft.Row(
                [
                    ft.Radio(value="radians", label="Radians"),
                    ft.Radio(value="degrees", label="Degrees"),
                ]
            ),
            value="radians",
            on_change=lambda _: self._on_changed(),
        )
        self.complex_mode = _checkbox(
            "Complex mode (i is the imaginary unit; values plot as a + bi)"
        )
        self.points_of_interest = _checkbox("Show points of interest for the selected row")
        for box in (
            self.major_grid,
            self.minor_grid,
            self.axes,
            self.axis_numbers,
            self.lock_aspect,
            self.complex_mode,
            self.points_of_interest,
        ):
            box.on_change = lambda _: self._on_changed()

        def heading(text: str) -> ft.Text:
            return ft.Text(text, weight=ft.FontWeight.W_600, size=13)

        self.control = ft.Card(
            ft.Container(
                ft.Column(
                    [
                        ft.Row(
                            [
                                ft.Text(
                                    "Graph Settings",
                                    weight=ft.FontWeight.BOLD,
                                    size=15,
                                    expand=True,
                                ),
                                ft.IconButton(
                                    ft.Icons.CLOSE,
                                    icon_size=16,
                                    tooltip="Close",
                                    on_click=lambda _: self.close(),
                                ),
                            ]
                        ),
                        heading("Grid"),
                        self.grid_kind,
                        ft.Row([self.major_grid, self.minor_grid]),
                        ft.Row([self.axes, self.axis_numbers]),
                        heading("Axes"),
                        *axis_rows,
                        self.lock_aspect,
                        self.note,
                        heading("Calculation"),
                        ft.Row([ft.Text("Angles"), self.angles]),
                        self.complex_mode,
                        self.points_of_interest,
                    ],
                    spacing=6,
                    tight=True,
                    scroll=ft.ScrollMode.AUTO,
                ),
                padding=14,
                width=520,
            ),
            elevation=6,
            visible=False,
        )
        self.set_settings(settings)

    def _field(
        self, width: int, on_done: Callable[[], object], hint: str | None = None
    ) -> ft.TextField:
        field = ft.TextField(width=width, dense=True, hint_text=hint, text_size=13)

        def done(_: object) -> None:
            self._editing.discard(field)
            runtime.focus_tracker.lost(field)
            on_done()

        def focused(_: object) -> None:
            self._editing.add(field)
            runtime.focus_tracker.gained(field)

        field.on_blur = done
        field.on_submit = done
        field.on_focus = focused
        return field

    def is_visible(self) -> bool:
        return self.visible

    def set_compact(self, compact: bool) -> None:
        """On a narrow screen the panel fills the window, and its rows wrap."""
        content = self.control.content
        assert isinstance(content, ft.Container)
        content.width = None if compact else 520
        self.control.left, self.control.bottom = (8, 8) if compact else (None, None)
        self.control.right, self.control.top = (8, 8) if compact else (12, 12)
        for row in self._axis_rows:
            if isinstance(row, ft.Row):
                row.wrap = compact
        runtime.push(self.control)

    def show(self) -> None:
        self.visible = self.control.visible = True
        runtime.push(self.control)

    def close(self) -> None:
        self.visible = self.control.visible = False
        runtime.push(self.control)
        self.closed.emit()

    # -- from the application -------------------------------------------------------------
    def set_settings(self, settings: GraphSettings) -> None:
        self._settings = settings
        self._updating = True
        try:
            self.grid_kind.value = "polar" if settings.polar_grid else "cartesian"
            self.major_grid.value = settings.major_grid
            self.minor_grid.value = settings.minor_grid
            self.axes.value = settings.axes
            self.axis_numbers.value = settings.axis_numbers
            self.lock_aspect.value = settings.lock_aspect
            self.angles.value = "degrees" if settings.degrees else "radians"
            self.complex_mode.value = settings.complex_mode
            self.points_of_interest.value = settings.points_of_interest
            for axis, axis_settings in (("x", settings.x_axis), ("y", settings.y_axis)):
                step = self.steps[axis]
                if step not in self._editing:
                    step.value = (
                        "" if axis_settings.step is None else format_value(axis_settings.step)
                    )
                self.scales[axis].value = "log" if axis_settings.log else "linear"
            linear_only = settings.polar_grid or settings.complex_mode
            for scale in self.scales.values():
                scale.disabled = linear_only
            self.lock_aspect.disabled = settings.polar_grid
            notes = []
            if settings.polar_grid:
                notes.append("The polar grid uses linear axes with equal scales.")
            elif settings.complex_mode:
                notes.append("Complex mode plots the complex plane on linear axes.")
            self.note.value = " ".join(notes)
            self.note.color = theme.current().muted
            self.note.visible = bool(notes)
        finally:
            self._updating = False
        runtime.push(self.control)

    def set_ranges(
        self,
        x: tuple[float, float],
        y: tuple[float, float],
        z: tuple[float, float] | None = None,
    ) -> None:
        """Show the visible ranges (fields being edited are left alone); a range for z
        when the graph is a 3D view."""
        self.z_row.visible = z is not None
        ranges = [("x", x), ("y", y)] + ([("z", z)] if z is not None else [])
        for axis, (lo, hi) in ranges:
            for edit, value in zip(self.bounds[axis], (lo, hi), strict=True):
                if edit not in self._editing:
                    edit.value = format_value(value, 6)
        runtime.push(self.control)

    # -- to the application -------------------------------------------------------------
    def _on_changed(self) -> None:
        if self._updating:
            return
        axes = {}
        for axis in ("x", "y"):
            text = (self.steps[axis].value or "").strip()
            step = _parse(text) if text else None
            if step is not None and step <= 0:
                step = None
            axes[axis] = AxisSettings(self.scales[axis].value == "log", step)
        settings = replace(
            self._settings,
            x_axis=axes["x"],
            y_axis=axes["y"],
            polar_grid=self.grid_kind.value == "polar",
            major_grid=bool(self.major_grid.value),
            minor_grid=bool(self.minor_grid.value),
            axes=bool(self.axes.value),
            axis_numbers=bool(self.axis_numbers.value),
            lock_aspect=bool(self.lock_aspect.value),
            degrees=self.angles.value == "degrees",
            complex_mode=bool(self.complex_mode.value),
            points_of_interest=bool(self.points_of_interest.value),
        )
        self.settings_changed.emit(settings)

    def change(self, control: ft.Control, value: object) -> None:
        """Set a control's value as the user would (tests and keyboard shortcuts)."""
        control.value = value  # type: ignore[attr-defined]
        self._on_changed()

    def _on_bounds_edited(self, axis: str) -> None:
        lo_edit, hi_edit = self.bounds[axis]
        lo, hi = _parse(lo_edit.value or ""), _parse(hi_edit.value or "")
        if lo is not None and hi is not None:
            self.range_edited.emit(axis, lo, hi)

    def edit_bounds(self, axis: str, lo: str, hi: str) -> None:
        """Type new bounds for ``axis`` and leave the field (as the user would)."""
        lo_edit, hi_edit = self.bounds[axis]
        lo_edit.value, hi_edit.value = lo, hi
        self._on_bounds_edited(axis)

    def edit_step(self, axis: str, text: str) -> None:
        self.steps[axis].value = text
        self._on_changed()
