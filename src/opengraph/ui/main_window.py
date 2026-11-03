"""Application main window: layout, menus, shortcuts, file handling and status bar."""

from __future__ import annotations

import io
import logging
import re
import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path

import flet as ft

from opengraph import __version__, events
from opengraph.controller import Controller
from opengraph.document import DocumentError
from opengraph.engine.builtins import CONSTANTS, FUNCTIONS, SPECIAL_FORMS
from opengraph.engine.formatting import format_value
from opengraph.graph.canvas import GraphCanvas
from opengraph.graph.settings import GraphSettings, GraphToggle
from opengraph.preferences import Preferences, config_dir
from opengraph.tasks import TaskState, format_duration
from opengraph.ui import dialogs, fonts, runtime, theme
from opengraph.ui.expression_panel import ExpressionPanel
from opengraph.ui.graph_settings import GraphSettingsDialog
from opengraph.ui.keypad import MathKeypad
from opengraph.ui.mathedit.widget import focused_editor
from opengraph.ui.tasks_dialog import TasksDialog

log = logging.getLogger(__name__)
FILE_EXTENSIONS = ["json"]  # .ograph.json documents
COMPACT_WIDTH = 720  # narrower than this, the graph goes above the expressions
PANEL_SHARE = 0.42  # of the window's height, for the expressions on a narrow screen
PANEL_MIN_HEIGHT = 120
GRAPH_MIN_HEIGHT = 120
AUTOSAVE_NAME = "autosave.ograph.json"
THEME_LABELS = {"light": "Light", "dark": "Dark", "system": "System"}
STATUS_MS = 6000
PANEL_MIN, PANEL_MAX = 260, 900


def _code(text: str) -> str:
    return f"`{text.replace('`', '')}`"


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


# Expressions and plots in the syntax reference: (syntax, meaning).
_PLOT_SYNTAX = (
    (
        "`y = x^2`, `x = y^2`, `x^2+y^2=1`",
        "Explicit and implicit Cartesian curves. Bare expressions in x also plot.",
    ),
    (
        "`r = 1 + cos(theta)`, `r = 2`",
        (
            "Polar equations have the same meaning on every grid; r and theta are reserved "
            "coordinates."
        ),
    ),
    (
        "`f(t) = t^2`, `a = 2`",
        (
            "Explicit definitions. Every function of one parameter is plotted, regardless "
            "of its name."
        ),
    ),
    (
        "`2a = 4`, `a^2 = 4`",
        (
            "Show equation solutions without defining a. To bind a finite root list, write "
            "`A = roots(a^2 - 4, a)`, then `y = A x`."
        ),
    ),
    (
        "`plot(t^2, t, (0, 2])`",
        "A graph restricted to an interval; open and closed endpoints are respected.",
    ),
    (
        "`parametric(cos(u), sin(u), u, [0, 2pi])`",
        "An explicit local parameter and range. Other rows defining u do not affect it.",
    ),
    (
        "`t = [0, 2pi]`, then `point[cos(t), sin(t)]`",
        (
            "A continuous curve from an interval. Arithmetic and functions apply to every "
            "value; no parametric command is needed. Try `u = [-2, 2]` and `point[u, u^2]` "
            "for a parabola segment."
        ),
    ),
    (
        "`point[1, 2]`, `p = point[1, 2]`",
        "Points and vectors. Points never introduce an implicit parameter.",
    ),
    (
        "`z = x^2 - y^2`, `x = y z`",
        (
            "A surface. Any equation that uses z draws in space, and the graph turns into a "
            "3D view while such a row is visible. z is always a coordinate, like x and y."
        ),
    ),
    (
        "`x^2 + y^2 + z^2 = 9`",
        "An implicit surface, sampled numerically; small features may be missed.",
    ),
    (
        "`point[1, 2, 3]`, `parametric(cos(u), sin(u), u for u in [0, 6pi])`",
        (
            "Points and curves in space. With `t = [0, 6pi]`, `point[cos(t), sin(t), t]` "
            "draws the same curve. Rows without z stay flat in the plane z = 0."
        ),
    ),
    (
        "`list[1, 2, 3]`, `list[1...10]`",
        (
            "Finite lists. L[2], L[2...4], L[list[3, 1]] select elements (1-based). Scalars "
            "broadcast; two lists must have equal lengths."
        ),
    ),
    (
        "`S = seq[2k + 1 for k in sets[N]]`",
        (
            "An infinite ordered sequence, with a labelled display preview. sets[N] starts "
            "at 1; sets[Z] enumerates 0, 1, -1, 2, -2, ... . A plain N or Z is an ordinary "
            "variable. S[1000] generates a term directly. Use an explicit finite slice such "
            "as S[1...100] for graphs or total/mean/length. `k in sets[N]` (or k ∈ sets[Z]) "
            "is shorthand for `k = seq[k for k in sets[N]]`. Set notation does not define "
            "sequences."
        ),
    ),
    (
        "`[0, 1)`, `(2, 2)`",
        (
            "Continuous lists, usable in arithmetic, functions, point coordinates, and plot "
            "ranges. Two intervals pair at the same fractional position; scalars broadcast. "
            "Open endpoints are excluded; equal endpoints with an open end give an empty "
            "interval. Finite indexing and total/mean/length require finite lists. `y = [1, "
            "2) x` shades a continuous linear family; use point coordinates for nonlinear "
            "interval mappings."
        ),
    ),
    (
        "`y < x^2`, `x+y >= 1`",
        (
            "Shaded inequalities; strict boundaries are dashed. Chains such as 1 < x <= 3 "
            "shade their intersection."
        ),
    ),
    (
        "`formula(a, (y-b)/x)`",
        (
            "A symbolic display that allows free names and does not define a variable. "
            "Solve for inserts these rows; copy its expression into a definition to use it."
        ),
    ),
)


def syntax_reference() -> str:
    """Reference text (Markdown) generated from the builtin registry, so it can't drift."""
    seen: set[str] = set()
    rows = []
    for builtin in FUNCTIONS.values():
        if builtin.name in seen:
            continue
        seen.add(builtin.name)
        rows.append(f"| {_code(builtin.signature)} | {_cell(builtin.summary)} |")
    special = "\n".join(
        f"| {_code(form.signature)} | {_cell(form.summary)} |" for form in SPECIAL_FORMS
    )
    constants = ", ".join(f"{_code(n)} = {format_value(v, 8)}" for n, v in CONSTANTS.items())
    functions = "\n".join(rows)
    plots = "\n".join(f"| {_cell(a)} | {_cell(b)} |" for a, b in _PLOT_SYNTAX)
    return f"""
## Expressions and plots

| Syntax | Meaning |
|---|---|
{plots}

### Stable syntax and editing

Adjacent letters form one identifier: `ab` is a name. Use `a b` or `a*b` for a product.
Numbers may touch names: `2pi x` means 2 times pi times x. Function application uses
parentheses.

Precedence: + and -, then * and / (including implicit products), unary minus, factorial,
then right-associative powers. Thus `1/2x = (1/2)x`, `x^2/3 = (x^2)/3`, `3^2! = (3^2)!`.
Ordinary typing follows the same rules as pasted source. Use brackets for a larger
exponent or denominator. Arrow keys can edit inside a structure explicitly.

To type a mixed interval such as `[0, 1)` or `(0, 1]`, type the closing bracket you
want; it replaces the automatic closer.

Greek and root shortcuts convert complete words at a delimiter; words such as spin and
sqrtvalue stay intact. Backslash commands such as `\\pi` and `\\frac` are also
available. Plain-text mode: Ctrl+Shift+L. F1 opens this reference.

### Functions

| Function | Meaning |
|---|---|
{functions}

### Calculus and list operations

| Operation | Meaning |
|---|---|
{special}

Always name the index of a sum/product: `sum(k^2, k, 1, 10)`. A three-argument definite
integral binds x; otherwise name its variable explicitly: `int(t^2, t, 0, 1)`.

Standalone algebra, derivatives, limits and Taylor polynomials accept free symbols:
`expand((a+b)^2)`, `diff(a*x^2, x)`, and `diff(diff(x^2*y^3, x), y)` display formulas
without defining a, b or y. Defining the remaining parameters lets formulas in x plot
automatically. Use `cancel` to cancel factors, `together` to combine fractions,
`apart(f, t)` for partial fractions, `collect(f, t)` to group powers of t, and
`trigsimp` for trigonometric identities.

Partial fractions use current parameter values and recompute when those values change,
including when poles coincide. If a decomposition needs conditions on free parameters,
define those parameters first.

Algebra and derivatives retain the original domain. A menu result may use
`guard(value, original)` to preserve exclusions across save/copy/paste;
`restrict(value, coordinate, interval)` similarly retains a range. Antiderivatives use
current parameter values and recompute when those values change; conditional formulas
are never silently treated as universal.

### Numerical limits

Numerical root feedback states its search interval and may miss roots. Use Solve for
exact solutions. Implicit curves use numerical contours; very small features and
unrecognized tangent zero sets may be missed. Powers and repeated identical outer factors
are normalized before contouring, with the original domain retained.

Values outside a real domain are undefined. Complex mode uses principal branches.
Standalone expressions with interval values trace their real and imaginary parts in the
complex plane: `t = [0, 2pi]`, then `e^(i t)` draws a circle in radians. Equations such
as `y = sqrt(x)` draw where their values are real. Angle units affect trigonometric
functions; grid appearance does not change expression meaning.

### Constants

{constants}; i is the imaginary unit in complex mode and inside `re(...)`, `im(...)` and
`arg(...)`. These functions evaluate complex arguments in either mode. Compact products
such as `ix` and `it` mean `i*x` and `i*t` in these contexts; explicitly defined names
and function parameters take priority. For example, `simplify(re(e^ix))` gives `cos(x)`
in radians.

### Tables and interaction

Tables define finite column lists and support formula columns, CSV import, points and
lines. Literal numeric definitions have sliders. Undefined names offer Add slider and,
where meaningful, Solve for. Drag to pan; wheel zooms at the cursor. Ctrl+J opens
cancellable background tasks. Save/open .ograph.json; export PNG/JPEG.

In a 3D view, drag to turn the view and Shift+drag to pan; the wheel zooms, Ctrl+wheel
only x and y, and Shift+wheel only z. Arrow keys turn the view and Ctrl+0 resets it.

On a touch screen, one finger pans (or turns a 3D view) and two fingers pinch to zoom and
move. Tap or press and hold to read coordinates. The row of keys above the keyboard moves
the caret and types symbols; press and hold an expression for copy and paste.
"""


Shortcut = tuple[bool, bool, str]  # command (Ctrl/⌘), Shift, key label

MACOS = sys.platform == "darwin"
_COMMAND_SHORTCUT = re.compile(r"Ctrl\+(Shift\+)?(?=[A-Z0-9=,-])")


def shortcut_label(text: str) -> str:
    """``text`` with its command shortcuts written for this platform: ``Ctrl+Shift+S``
    becomes ``⇧⌘S`` on macOS, where the Command key triggers them (Ctrl+wheel stays)."""
    if not MACOS:
        return text
    return _COMMAND_SHORTCUT.sub(lambda match: "⇧⌘" if match[1] else "⌘", text)


class _MenuItem:
    """A menu entry with a shortcut hint and an optional check mark."""

    def __init__(
        self,
        label: str,
        action: Callable[[], object],
        shortcut: str | None = None,
        *,
        checkable: bool = False,
    ) -> None:
        self.action = action
        self.checkable = checkable
        self._checked = False
        content: ft.Control = ft.Text(label)
        if shortcut and not runtime.device.touch:
            hint = ft.Text(shortcut_label(shortcut), size=12, color=ft.Colors.ON_SURFACE_VARIANT)
            content = ft.Row([content, ft.Container(width=28), hint], tight=True, spacing=0)
        self.button = ft.MenuItemButton(
            content=content,
            leading=ft.Icon(ft.Icons.CHECK, size=16, opacity=0) if checkable else None,
            on_click=lambda _: self.trigger(),
        )

    def trigger(self) -> None:
        if not self.button.disabled:
            self.action()

    def is_checked(self) -> bool:
        return self._checked

    def set_checked(self, checked: bool) -> None:
        self._checked = checked
        if isinstance(self.button.leading, ft.Icon):
            self.button.leading.opacity = 1 if checked else 0
            runtime.push(self.button)

    def is_enabled(self) -> bool:
        return not self.button.disabled

    def set_enabled(self, enabled: bool) -> None:
        self.button.disabled = not enabled
        runtime.push(self.button)


class MainWindow:
    def __init__(self, page: ft.Page | None = None, preferences: Preferences | None = None) -> None:
        self.page = page
        self.preferences = preferences or Preferences()
        self.path: Path | None = None

        self.panel = ExpressionPanel()
        self.canvas = GraphCanvas()
        self.controller = Controller(self.panel, self.canvas)

        self.settings_dialog = GraphSettingsDialog(self.controller.settings)
        self.settings_dialog.settings_changed.connect(self.controller.set_settings)
        self.settings_dialog.range_edited.connect(self._set_range)
        self.settings_dialog.control.top = 12
        self.settings_dialog.control.right = 12
        self.tasks_dialog = TasksDialog(self.controller.tasks)
        self.tasks_dialog.control.bottom = 12
        self.tasks_dialog.control.right = 12

        self.status = ft.Text("", size=12, expand=True, no_wrap=True)
        self._status_timer = events.Timer(STATUS_MS, single_shot=True)
        self._status_timer.timeout.connect(lambda: self.show_message(""))
        self.coords = ft.Text("", size=12, width=220, text_align=ft.TextAlign.RIGHT)
        self.progress = ft.ProgressBar(
            width=160,
            bar_height=6,
            visible=False,
            tooltip="Computing symbolically… (View ▸ Tasks to cancel)",
        )
        # Shows what runs in the background and for how long; a click opens the Tasks
        # panel, where tasks can be cancelled.
        self.tasks_button = ft.TextButton(
            "",
            tooltip=shortcut_label("Show running tasks (Ctrl+J)"),
            visible=False,
            on_click=lambda _: self.show_tasks(),
        )
        self._tasks_timer = events.Timer(200)
        self._tasks_timer.timeout.connect(self._update_tasks_button)
        self.canvas.cursor_moved.connect(self._show_coordinates)
        self.controller.status_message.connect(self.show_message)
        self.controller.modified_changed.connect(lambda _: self._update_title())
        self.controller.busy_changed.connect(self._on_busy_changed)
        self.controller.settings_changed.connect(self._on_settings_changed)
        self.controller.tasks.changed.connect(self._update_tasks_button)
        self.canvas.view_changed.connect(self._update_dialog_ranges)
        self.canvas.pressed.connect(self._focus_graph)

        self._shortcuts: dict[Shortcut, Callable[[], object]] = {}
        self.menu_bar = self._build_menus()
        self.panel.linear_input_requested.connect(self.set_linear_input)

        # The menu bar spans the window on a band of its own, above a hairline.
        self.menu_band = ft.Container(ft.Row([self.menu_bar], spacing=0))

        # A hairline between the panel and the graph, in a wider grab area; it takes the
        # accent colour while hovered or dragged.
        self._splitter_hover = False
        self._panel_width = float(self.panel.control.width or 340)
        self._splitter_line = ft.Container(width=1)
        self.splitter = ft.GestureDetector(
            content=ft.Container(self._splitter_line, width=7, alignment=ft.Alignment.CENTER),
            mouse_cursor=ft.MouseCursor.RESIZE_COLUMN,
            on_horizontal_drag_update=self._on_splitter_drag,
            on_enter=lambda _: self._set_splitter_hover(True),
            on_exit=lambda _: self._set_splitter_hover(False),
        )
        self._compact = False
        self._size = (0.0, 0.0)
        self._panel_share = PANEL_SHARE
        self._splitter_row_line = ft.Container(height=1)
        self.row_splitter = ft.GestureDetector(
            content=ft.Container(
                self._splitter_row_line,
                height=18 if runtime.device.touch else 7,
                alignment=ft.Alignment.CENTER,
            ),
            mouse_cursor=ft.MouseCursor.RESIZE_ROW,
            on_vertical_drag_update=self._on_row_splitter_drag,
        )
        self.keypad = MathKeypad()
        runtime.focus_tracker.changed.connect(self._update_keypad)
        self.status_bar = ft.Container(
            ft.Row(
                [self.status, self.tasks_button, self.progress, self.coords],
                spacing=8,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            padding=ft.Padding.symmetric(horizontal=8, vertical=2),
            height=28,
        )
        self.graph_area = ft.Stack(
            [self.canvas.control, self.settings_dialog.control, self.tasks_dialog.control],
            expand=True,
        )
        self.body: ft.Row | ft.Column = self._wide_body()
        self.body_host = ft.Container(self.body, expand=True)
        # Tracks the modifier keys (Ctrl/Shift+wheel, Shift+click) for the whole window.
        self.frame = ft.Column(
            [self.menu_band, self.body_host, self.keypad.control, self.status_bar],
            spacing=0,
            expand=True,
        )
        # The floating panels are over the graph, or over everything on a narrow screen.
        self.overlay = ft.Stack([self.frame], expand=True, fit=ft.StackFit.EXPAND)
        self.keyboard = ft.KeyboardListener(
            self.overlay,
            on_key_down=lambda e: runtime.modifiers.update(e.key, True),
            on_key_up=lambda e: runtime.modifiers.update(e.key, False),
        )
        self.control = ft.Container(self.keyboard, expand=True)

        self.set_linear_input(self.preferences.get_bool("linearInput", False))
        self.theme_mode = self.preferences.get_str("theme", "system")
        if self.theme_mode not in theme.MODES:
            self.theme_mode = "system"
        self.apply_theme(self.theme_mode)
        self.controller.new_document()
        self._update_title()

    # -- layout ----------------------------------------------------------------------------
    def _wide_body(self) -> ft.Row:
        return ft.Row(
            [self.panel.control, self.splitter, self.graph_area],
            spacing=0,
            expand=True,
            vertical_alignment=ft.CrossAxisAlignment.STRETCH,
        )

    def _narrow_body(self) -> ft.Column:
        return ft.Column(
            [self.graph_area, self.row_splitter, self.panel.control],
            spacing=0,
            expand=True,
            horizontal_alignment=ft.CrossAxisAlignment.STRETCH,
        )

    @property
    def compact(self) -> bool:
        return self._compact

    def resize(self, width: float, height: float) -> None:
        """The window's size: a narrow one has the graph above the expressions."""
        self._size = (width, height)
        compact = width < COMPACT_WIDTH
        changed = compact != self._compact
        self._compact = compact
        if changed:
            self.body = self._narrow_body() if compact else self._wide_body()
            self.body_host.content = self.body
            self.panel.control.width = None if compact else self._panel_width
            self.coords.visible = not compact
            self.progress.width = 60 if compact else 160
            self._move_floating_panels(compact)
        if compact:
            self._fit_panel_height()
        elif changed:
            self.panel.control.height = None
        if changed:
            runtime.push(self.control)
        else:
            runtime.push(self.panel.control)

    def _move_floating_panels(self, compact: bool) -> None:
        panels = [self.settings_dialog.control, self.tasks_dialog.control]
        old, new = (self.graph_area, self.overlay) if compact else (self.overlay, self.graph_area)
        old.controls = [control for control in old.controls if control not in panels]
        new.controls = [*new.controls, *panels]
        self.settings_dialog.set_compact(compact)
        self.tasks_dialog.set_compact(compact)

    def _fit_panel_height(self) -> None:
        height = self._size[1]
        wanted = height * self._panel_share
        most = max(PANEL_MIN_HEIGHT, height - GRAPH_MIN_HEIGHT - 110)  # bars and keys
        self.panel.control.height = round(min(max(wanted, PANEL_MIN_HEIGHT), most))

    def _on_row_splitter_drag(self, event: ft.DragUpdateEvent[ft.GestureDetector]) -> None:
        height = self._size[1]
        if not height:
            return
        current = float(self.panel.control.height or height * self._panel_share)
        self._panel_share = (current - (event.primary_delta or 0.0)) / height
        self._fit_panel_height()
        self._panel_share = float(self.panel.control.height or current) / height
        runtime.push(self.panel.control)

    def _update_keypad(self) -> None:
        shown = runtime.device.touch and focused_editor() is not None
        if shown != self.keypad.control.visible:
            self.keypad.control.visible = shown
            runtime.push(self.keypad.control)

    # -- menus -----------------------------------------------------------------------------
    def _item(
        self,
        label: str,
        action: Callable[[], object],
        shortcut: str | None = None,
        *,
        checkable: bool = False,
    ) -> _MenuItem:
        item = _MenuItem(label, action, shortcut, checkable=checkable)
        if shortcut is not None:
            parts = shortcut.split("+")
            key = parts[-1]
            self._shortcuts[("Ctrl" in parts, "Shift" in parts, key.upper())] = item.trigger
        return item

    def _build_menus(self) -> ft.MenuBar:
        file_items = [
            self._item("New", self.new_file, "Ctrl+N"),
            self._item("Open…", self.open_file, "Ctrl+O"),
            self._item("Save", self.save_file, "Ctrl+S"),
            self._item("Save As…", self.save_file_as, "Ctrl+Shift+S"),
            self._item("Export Image…", self.export_image, "Ctrl+E"),
            self._item("Quit", self.close, "Ctrl+Q"),
        ]
        # Replacing the document is locked while a symbolic job runs.
        self._locked_actions = file_items[:2]
        edit_items = [
            self._item(
                "Add Expression",
                lambda: self.controller.add_expression(focus=True),
                "Ctrl+Shift+N",
            ),
            self._item("Add Table", lambda: self.controller.add_table(focus=True), "Ctrl+Shift+T"),
        ]
        view_items = [
            self._item("Zoom In", lambda: self.canvas.zoom(1.25), "Ctrl+="),
            self._item("Zoom Out", lambda: self.canvas.zoom(0.8), "Ctrl+-"),
            self._item("Reset View", self.canvas.reset_view, "Ctrl+0"),
            self._item("Graph Settings…", self.show_graph_settings, "Ctrl+,"),
        ]
        self._shortcuts[(True, False, "+")] = view_items[0].trigger
        self._shortcuts[(True, True, "=")] = view_items[0].trigger
        # Quick toggles for the most used settings; the panel has the rest.
        self._toggles: dict[GraphToggle, _MenuItem] = {}
        quick_toggles: tuple[tuple[GraphToggle, str], ...] = (
            ("polar_grid", "Polar Grid"),
            ("degrees", "Degrees"),
            ("complex_mode", "Complex Mode"),
            ("points_of_interest", "Points of Interest"),
        )
        for field, label in quick_toggles:
            item = self._item(label, partial(self._toggle_setting, field), checkable=True)
            self._toggles[field] = item
        self._linear_action = self._item(
            "Plain-Text Input",
            lambda: self.set_linear_input(not self.panel.linear_input),
            "Ctrl+Shift+L",
            checkable=True,
        )
        tasks_item = self._item("Tasks…", self.show_tasks, "Ctrl+J")
        self._theme_actions: dict[str, _MenuItem] = {
            mode: _MenuItem(label, partial(self.set_theme_mode, mode), checkable=True)
            for mode, label in THEME_LABELS.items()
        }
        help_items = [
            _MenuItem("Syntax Reference", self.show_reference, "F1"),
            _MenuItem("About OpenGraph", self.show_about),
        ]
        self._shortcuts[(False, False, "F1")] = self.show_reference

        def menu(title: str, controls: list[ft.Control]) -> ft.SubmenuButton:
            return ft.SubmenuButton(content=ft.Text(title), controls=controls)

        def buttons(items: list[_MenuItem]) -> list[ft.Control]:
            return [item.button for item in items]

        divider: Callable[[], ft.Control] = lambda: ft.Divider(height=1)  # noqa: E731
        return ft.MenuBar(
            controls=[
                menu(
                    "File",
                    [
                        *buttons(file_items[:4]),
                        divider(),
                        file_items[4].button,
                        # Android and iOS apps are closed by the system, never by themselves.
                        *(
                            []
                            if runtime.device.native_mobile
                            else [divider(), file_items[5].button]
                        ),
                    ],
                ),
                menu("Edit", buttons(edit_items)),
                menu(
                    "View",
                    [
                        *buttons(view_items),
                        divider(),
                        *buttons(list(self._toggles.values())),
                        divider(),
                        self._linear_action.button,
                        tasks_item.button,
                        menu("Theme", buttons(list(self._theme_actions.values()))),
                    ],
                ),
                menu("Help", buttons(help_items)),
            ],
        )

    # -- keyboard ---------------------------------------------------------------------------
    def on_key(self, event: ft.KeyboardEvent) -> None:
        self.handle_key(
            event.key, shift=event.shift, ctrl=event.ctrl, alt=event.alt, meta=event.meta
        )

    def handle_key(
        self,
        key: str,
        *,
        shift: bool = False,
        ctrl: bool = False,
        alt: bool = False,
        meta: bool = False,
    ) -> bool:
        """A key pressed anywhere in the window: the focused editor, then the menu
        shortcuts, then (when no text input has the focus) the graph get it."""
        editor = focused_editor()
        if editor is not None and editor.handle_key(
            key, shift=shift, ctrl=ctrl, alt=alt, meta=meta
        ):
            return True
        command = ctrl or meta
        action = self._shortcuts.get((command, shift, key.upper() if len(key) == 1 else key))
        if action is not None and not alt:
            action()
            return True
        if runtime.focus_tracker.owner is None and not command and not alt:
            return self.canvas.handle_key(key)
        return False

    def _focus_graph(self) -> None:
        """A click on the graph moves the keyboard focus away from the text inputs, so the
        arrow and zoom keys reach the graph."""
        editor = focused_editor()
        if editor is not None:
            editor._set_focus(False)
        runtime.focus_tracker.owner = None
        runtime.focus(self.keyboard)

    # -- settings and themes ---------------------------------------------------------------
    def _toggle_setting(self, field: GraphToggle) -> None:
        on = not bool(getattr(self.controller.settings, field))
        self.controller.set_settings(self.controller.settings.with_toggle(field, on))

    def _on_settings_changed(self, settings: GraphSettings) -> None:
        for field, item in self._toggles.items():
            item.set_checked(bool(getattr(settings, field)))
        self.settings_dialog.set_settings(settings)
        self._update_dialog_ranges()

    def show_graph_settings(self) -> None:
        self.settings_dialog.show()
        self._update_dialog_ranges()

    def _set_range(self, axis: str, lo: float, hi: float) -> None:
        if not self.canvas.set_range(axis, lo, hi):
            log_axis = self.canvas.viewport.x_log if axis == "x" else self.canvas.viewport.y_log
            log_axis = log_axis and not self.canvas.view3d.active
            reason = " and positive on a logarithmic axis" if log_axis else ""
            self.show_message(f"The {axis}-axis minimum must be less than its maximum{reason}")
        self._update_dialog_ranges()

    def _update_dialog_ranges(self) -> None:
        dialog = self.settings_dialog
        if dialog.is_visible():
            canvas = self.canvas
            z = canvas.axis_range("z") if canvas.view3d.active else None
            dialog.set_ranges(canvas.axis_range("x"), canvas.axis_range("y"), z)

    def set_linear_input(self, on: bool) -> None:
        """Edit expressions as plain text instead of typeset maths; remembered."""
        self.panel.set_linear_input(on)
        self._linear_action.set_checked(on)
        self.preferences.set("linearInput", on)

    def set_theme_mode(self, mode: str) -> None:
        """Switch to ``light``, ``dark`` or ``system`` and remember the choice."""
        self.theme_mode = mode
        self.preferences.set("theme", mode)
        self.apply_theme(mode)

    def apply_theme(self, mode: str) -> None:
        current = theme.resolve(mode)
        for name, item in self._theme_actions.items():
            item.set_checked(name == mode)
        page = self.page
        if page is not None:
            page.theme = page.dark_theme = current.material()
            page.theme_mode = ft.ThemeMode.DARK if current.dark else ft.ThemeMode.LIGHT
            page.bgcolor = current.window
        hairline = ft.BorderSide(1, current.row_border)
        self.menu_band.bgcolor = current.panel
        self.menu_band.border = ft.Border(bottom=hairline)
        self.menu_bar.style = ft.MenuStyle(
            bgcolor=current.panel, elevation=0, shadow_color="#00000000"
        )
        self.keypad.apply_theme()
        self.status_bar.bgcolor = current.window
        self.status_bar.border = ft.Border(top=hairline)
        for text in (self.status, self.coords):
            text.color = current.info
        self.controller.set_theme(current)
        self._style_splitter()
        self.settings_dialog.set_settings(self.controller.settings)
        if page is not None:
            page.update()

    def on_brightness_changed(self, dark: bool) -> None:
        theme.set_system_dark(dark)
        if self.theme_mode == "system":
            self.apply_theme("system")

    def _on_busy_changed(self, busy: bool) -> None:
        self.progress.visible = busy
        for item in self._locked_actions:
            item.set_enabled(not busy)
        runtime.push(self.status_bar)

    def show_tasks(self) -> None:
        self.tasks_dialog.show()

    def _update_tasks_button(self) -> None:
        active = self.controller.tasks.active()
        running = [t for t in active if t.state is TaskState.RUNNING]
        self.tasks_button.visible = bool(active)
        if not active:
            self._tasks_timer.stop()
        else:
            if not self._tasks_timer.is_active():
                self._tasks_timer.start()
            longest = max((t.elapsed() or 0.0 for t in running), default=None)
            count = f"{len(active)} task{'s' if len(active) != 1 else ''}"
            self.tasks_button.content = (
                count if longest is None else f"{count} · {format_duration(longest)}"
            )
        runtime.push(self.tasks_button)

    # -- title & status -----------------------------------------------------------------
    def title(self) -> str:
        name = self.path.name if self.path else "Untitled"
        return f"{name}{'*' if self.controller.modified else ''} — OpenGraph"

    def _update_title(self) -> None:
        if self.page is not None:
            self.page.title = self.title()
            self.page.update()

    def show_message(self, message: str, timeout_ms: int = STATUS_MS) -> None:
        self.status.value = message
        if message:
            self._status_timer.start(timeout_ms)
        runtime.push(self.status)

    def current_message(self) -> str:
        return self.status.value or ""

    def _show_coordinates(self, x: float, y: float) -> None:
        # (x, y), r and θ on the polar grid, or a + bi in complex mode.
        self.coords.value = self.canvas.format_position(x, y)
        runtime.push(self.coords)

    def _set_splitter_hover(self, hover: bool) -> None:
        self._splitter_hover = hover
        self._style_splitter()

    def _style_splitter(self) -> None:
        current = theme.current()
        self._splitter_line.bgcolor = current.accent if self._splitter_hover else current.row_border
        self._splitter_line.width = 2 if self._splitter_hover else 1
        runtime.push(self._splitter_line)

    def _on_splitter_drag(self, event: ft.DragUpdateEvent[ft.GestureDetector]) -> None:
        delta = event.primary_delta or 0.0
        width = float(self.panel.control.width or 340) + delta
        self.panel.control.width = self._panel_width = max(PANEL_MIN, min(PANEL_MAX, width))
        runtime.push(self.panel.control)

    # -- files ---------------------------------------------------------------------------
    def _confirm_discard(self, then: Callable[[], object]) -> None:
        """Run ``then`` unless the user keeps unsaved changes (offering to save them)."""
        if not self.controller.modified:
            then()
            return

        def answer(choice: str | None) -> None:
            if choice == "Save":
                self.save_file(then)
            elif choice == "Discard":
                then()

        dialogs.ask(
            "Unsaved Changes",
            "Save changes to the current graph?",
            ["Cancel", "Discard", "Save"],
            answer,
            default="Save",
        )

    def _last_dir(self) -> Path:
        return Path(self.preferences.get_str("lastDir", str(Path.home())))

    def new_file(self) -> None:
        def start() -> None:
            self.path = None
            self.controller.new_document()
            self._update_title()

        self._confirm_discard(start)

    def open_file(self) -> None:
        self._confirm_discard(
            lambda: dialogs.ask_open_path(
                "Open Graph", self._last_dir(), FILE_EXTENSIONS, self.open_path
            )
        )

    def open_path(self, path: Path) -> bool:
        try:
            self.controller.open_document(path)
        except DocumentError as exc:
            dialogs.message("Open Failed", str(exc))
            return False
        self.path = path
        self.preferences.set("lastDir", str(path.parent))
        self._update_title()
        return True

    def save_file(self, then: Callable[[], object] | None = None) -> bool:
        """Save to the current path (asking for one first if there is none). Where the
        client's file chooser does the saving, there is always something to choose: an
        opened file's path is only a temporary copy on Android and iOS."""
        if self.path is None or dialogs.host.saves_through_client:
            self.save_file_as(then)
            return False
        try:
            self.controller.save_document(self.path)
        except OSError as exc:
            dialogs.message("Save Failed", f"Could not save {self.path.name}: {exc}")
            return False
        self.show_message(f"Saved {self.path.name}", 4000)
        self._update_title()
        if then is not None:
            then()
        return True

    def save_file_as(self, then: Callable[[], object] | None = None) -> None:
        start = self.path or self._last_dir() / "graph.ograph.json"
        if dialogs.host.saves_through_client:

            def saved(path: Path | None) -> None:
                self.path = path or self.path
                self.controller.mark_saved()
                self.show_message(f"Saved {(path or start).name}", 4000)
                self._update_title()
                if then is not None:
                    then()

            data = self.controller.document_data()
            dialogs.save_bytes("Save Graph", start.name, data, saved)
            return

        def chosen(path: Path) -> None:
            self.path = path
            self.preferences.set("lastDir", str(path.parent))
            self.save_file(then)

        dialogs.ask_save_path("Save Graph", start, FILE_EXTENSIONS, chosen)

    def export_image(self) -> None:
        if dialogs.host.saves_through_client:

            def exported(path: Path | None) -> None:
                self.show_message(f"Exported {(path or Path('graph.png')).name}", 4000)

            async def hand_over() -> None:
                dialogs.save_bytes("Export Image", "graph.png", await self.capture(), exported)

            self._when_rendered(lambda: runtime.spawn(hand_over))
            return
        dialogs.ask_save_path(
            "Export Image", self._last_dir() / "graph.png", ["png", "jpg"], self.export_to
        )

    def export_to(self, path: Path) -> None:
        """Save the graph as an image once its visible geometry has loaded."""
        self._when_rendered(lambda: runtime.spawn(lambda: self._write_image(path)))

    def _when_rendered(self, then: Callable[[], object]) -> None:
        """Run ``then`` once the graph's visible geometry has loaded."""

        def save() -> None:
            if self.canvas.is_rendering:
                return
            self.canvas.rendering_finished.disconnect(save)
            then()

        self.canvas.rendering_finished.connect(save)
        if self.canvas.is_rendering:
            self.show_message("Rendering graph for export…")
        else:
            events.call_soon(save)

    async def capture(self) -> bytes:
        """The graph as PNG data (the client draws it; tests replace this)."""
        return await self.canvas.capture_png()

    async def _write_image(self, path: Path) -> None:
        try:
            data = await self.capture()
            if path.suffix.lower() in (".jpg", ".jpeg"):
                from PIL import Image

                with Image.open(io.BytesIO(data)) as image:
                    buffer = io.BytesIO()
                    image.convert("RGB").save(buffer, "JPEG", quality=92)
                    data = buffer.getvalue()
            path.write_bytes(data)
        except (OSError, ValueError, RuntimeError) as exc:
            dialogs.message("Export Failed", f"Could not write {path}: {exc}")
            return
        self.show_message(f"Exported {path.name}", 4000)

    # -- help ----------------------------------------------------------------------------------
    def show_reference(self) -> None:
        dialog = ft.AlertDialog(
            title="Syntax Reference",
            content=ft.Container(
                ft.Column(
                    [
                        ft.Markdown(
                            shortcut_label(syntax_reference()),
                            selectable=True,
                            extension_set=ft.MarkdownExtensionSet.GITHUB_WEB,
                            code_theme=ft.MarkdownCodeTheme.GITHUB,
                        )
                    ],
                    scroll=ft.ScrollMode.AUTO,
                ),
                width=min(680, self._size[0] - 40) if self._size[0] else 680,
                height=min(720, self._size[1] - 150) if self._size[1] else 720,
            ),
        )
        dialog.actions = [ft.TextButton("Close", on_click=lambda _: dialogs.host.close(dialog))]
        dialogs.host.show(dialog)

    def show_about(self) -> None:
        dialogs.message(
            "About OpenGraph",
            f"OpenGraph {__version__}\n\nA graphing calculator built with Flet, NumPy and SymPy.",
        )

    # -- lifecycle ---------------------------------------------------------------------------
    def close(self) -> None:
        """Quit, after offering to save unsaved changes."""
        self._confirm_discard(self._quit)

    # -- the app's life on a phone ---------------------------------------------------------
    def on_lifecycle(self, state: ft.AppLifecycleState) -> None:
        """Android and iOS may end a backgrounded app without notice: keep the work."""
        if runtime.device.native_mobile and state in (
            ft.AppLifecycleState.HIDE,
            ft.AppLifecycleState.INACTIVE,
            ft.AppLifecycleState.PAUSE,
            ft.AppLifecycleState.DETACH,
        ):
            self.autosave()

    def autosave(self) -> None:
        path = config_dir() / AUTOSAVE_NAME
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            data = self.controller.document_data()
            path.with_suffix(".tmp").write_bytes(data)
            path.with_suffix(".tmp").replace(path)  # never a half-written file
        except OSError:
            log.exception("Could not keep the current graph")
            return
        self.preferences.set("autosaveModified", self.controller.modified)

    def restore_autosave(self) -> bool:
        """Continue where the app left off, if there is anything to continue."""
        path = config_dir() / AUTOSAVE_NAME
        if not path.exists():
            return False
        try:
            self.controller.open_document(path)
        except (OSError, DocumentError):
            log.exception("Could not restore the last graph")
            return False
        if self.preferences.get_bool("autosaveModified", False):
            self.controller.mark_modified()
        self._update_title()
        return True

    def _quit(self) -> None:
        self.shutdown()
        page = self.page
        if page is not None:
            runtime.spawn(page.window.destroy)

    def shutdown(self) -> None:
        page = self.page
        if page is not None and page.window.width and page.window.height:
            self.preferences.set("windowWidth", float(page.window.width))
            self.preferences.set("windowHeight", float(page.window.height))
        self.controller.tasks.shutdown()
        self.canvas.shutdown()


def fonts_for_page() -> dict[str, str]:
    """The bundled font families, by asset path (see :mod:`opengraph.ui.fonts`)."""
    return dict(fonts.FILES)
