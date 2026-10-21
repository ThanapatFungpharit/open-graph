"""``MathEdit``: a single-line control that edits an expression in its typeset form.

The maths is laid out in Python (:mod:`~opengraph.ui.mathedit.layout`) and drawn on a
Flet canvas. Keyboard input reaches the editor two ways: characters through a hidden
text field that holds the focus (so keyboard layouts, AltGr and dead keys work), and
editing keys (arrows, Backspace, Enter, Tab, Ctrl+Z…) through :meth:`MathEdit.handle_key`,
which the window calls for the focused editor. Mouse presses, drags and double clicks
on the canvas place the caret and select.
"""

from __future__ import annotations

import math
import time

import flet as ft
import flet.canvas as cv

from opengraph import colors, events
from opengraph.ui import fonts, runtime, theme
from opengraph.ui.geometry import Point, Rect
from opengraph.ui.mathedit.editing import EditorModel
from opengraph.ui.mathedit.layout import Layout, Typesetter
from opengraph.ui.mathedit.nodes import Char, MathList, Node, Position, Structure

_MARGIN_X = 3.0
_MARGIN_Y = 4.0
_SCROLL_MARGIN = 12.0  # keep this much room around the caret when scrolling
_BLINK_MS = 530
_FIELD_SUPPRESS_S = 0.25  # how long after Ctrl+Y the hidden field's own redo is ignored
_DEFAULT_WIDTH = 300.0  # until the client reports the real width

_focused: MathEdit | None = None


def focused_editor() -> MathEdit | None:
    """The editor that has the keyboard focus, if any."""
    return _focused


def _inserted(old: str, new: str) -> str:
    """The text typed into ``old`` to give ``new`` (empty for deletions)."""
    prefix = 0
    limit = min(len(old), len(new))
    while prefix < limit and old[prefix] == new[prefix]:
        prefix += 1
    suffix = 0
    while suffix < limit - prefix and old[len(old) - 1 - suffix] == new[len(new) - 1 - suffix]:
        suffix += 1
    return new[prefix : len(new) - suffix]


@ft.control("Canvas", isolated=True)  # type: ignore[arg-type]
class _EditCanvas(cv.Canvas):
    """A canvas whose shapes are sent only by its own updates, not the page's."""


class MathEdit:
    """Edits an expression as typeset maths; :meth:`text` is always valid source syntax.

    ``text_edited`` (with the new source text) fires only when the user changes the
    content — caret movement and selection never do, so the controller's debounced
    recompilation is not disturbed by them. The typeset layout is cached per content
    revision and width, so drawing the caret or a selection does no layout work.
    """

    def __init__(self, text: str = "", *, size: float = fonts.EDITOR_SIZE) -> None:
        self.text_edited = events.Signal()  # source text
        self.submitted = events.Signal()
        self.navigate = events.Signal()  # -1 up, +1 down (no fraction/exponent to move into)
        self.delete_empty = events.Signal()
        self.linear_requested = events.Signal()  # bool: the context menu's plain-text toggle
        self.focus_changed = events.Signal()  # bool

        self.model = EditorModel(text)
        self.size = size
        self.error_color = theme.current().error
        self.centered = False  # centre content narrower than the control (else left-align)
        self.has_focus = False
        self._typesetter: Typesetter | None = None
        self._layout: Layout | None = None
        self._layout_key: tuple[int, int, float] | None = None
        self._scroll = 0.0
        self._placeholder = ""
        self._error_span: tuple[int, int] | None = None
        self._dragging = False
        self._caret_visible = True
        self._caret_shape: cv.Rect | None = None
        self._blink = events.Timer(_BLINK_MS)
        self._blink.timeout.connect(self._toggle_caret)
        self._width = _DEFAULT_WIDTH
        self._height = 0
        self._typed = ""  # the hidden field's text, as last reported
        self._suppress_until = 0.0
        self._render_pending = False

        self.canvas = _EditCanvas(expand=True, resize_interval=30, on_resize=self._on_resize)
        self.detector = ft.GestureDetector(
            # A painted background makes the whole area receive pointer events.
            content=ft.Container(self.canvas, bgcolor="#00000000"),
            mouse_cursor=ft.MouseCursor.TEXT,
            on_pan_down=lambda e: self.press(_xy(e.local_position)),
            on_pan_update=lambda e: self.drag_to(_xy(e.local_position)),
            on_pan_end=lambda _: self.release(),
            on_pan_cancel=lambda _: self.release(),
            on_double_tap_down=lambda e: self.double_click(_xy(e.local_position)),
        )
        self._menu_items = {
            "undo": ft.PopupMenuItem("Undo", on_click=lambda _: self.undo()),
            "redo": ft.PopupMenuItem("Redo", on_click=lambda _: self.redo()),
            "cut": ft.PopupMenuItem("Cut", on_click=lambda _: self.cut()),
            "copy": ft.PopupMenuItem("Copy", on_click=lambda _: self.copy()),
            "paste": ft.PopupMenuItem("Paste", on_click=lambda _: self.paste()),
            "select_all": ft.PopupMenuItem("Select All", on_click=lambda _: self.select_all()),
            "latex": ft.PopupMenuItem("Copy as LaTeX", on_click=lambda _: self.copy_latex()),
            "linear": ft.PopupMenuItem(
                "Plain-text input",
                checked=False,
                on_click=lambda _: self.linear_requested.emit(not self.model.linear),
            ),
        }
        items = self._menu_items
        self.menu = ft.ContextMenu(
            content=self.detector,
            items=[
                items["undo"],
                items["redo"],
                ft.PopupMenuItem(),
                items["cut"],
                items["copy"],
                items["paste"],
                ft.PopupMenuItem(),
                items["select_all"],
                ft.PopupMenuItem(),
                items["latex"],
                ft.PopupMenuItem(),
                items["linear"],
            ],
        )
        # Receives the characters typed while the editor has the focus; drawn invisibly
        # beneath the canvas.
        self.input = ft.TextField(
            value="",
            opacity=0,
            width=40,
            height=24,
            left=0,
            top=0,
            dense=True,
            border=ft.NoInputBorder(),
            content_padding=0,
            autocorrect=False,
            enable_suggestions=False,
            smart_dashes_type=False,
            smart_quotes_type=False,
            enable_interactive_selection=False,
            ignore_up_down_keys=True,
            on_change=lambda _: self._on_input_change(),
            on_focus=lambda _: self._set_focus(True),
            on_blur=lambda _: self._set_focus(False),
        )
        self.control = ft.Stack([self.input, self.menu], height=self._content_height())
        self._refresh()

    # -- QLineEdit-like API ----------------------------------------------------------------
    def text(self) -> str:
        return self.model.source

    def set_text(self, text: str) -> None:
        """Load ``text`` (kept verbatim) and clear the undo history; no signal."""
        self.model.set_source(text, undoable=False)
        self._refresh()

    def replace_text(self, text: str) -> None:
        """Load ``text`` as one undoable step (Ctrl+Z restores the old one); no signal."""
        self.model.set_source(text, undoable=True)
        self._refresh()

    def set_placeholder(self, text: str) -> None:
        self._placeholder = text
        self.update()

    def placeholder(self) -> str:
        return self._placeholder

    def select_all(self) -> None:
        self.model.select_all()
        self._refresh()

    def end(self, mark: bool = False) -> None:
        self.model.move_to_edge(1, select=mark)
        self._refresh()

    def home(self, mark: bool = False) -> None:
        self.model.move_to_edge(-1, select=mark)
        self._refresh()

    def has_selected_text(self) -> bool:
        return self.model.has_selection()

    def selected_text(self) -> str:
        return self.model.selected_source()

    def undo(self) -> None:
        self._apply(self.model.undo())

    def redo(self) -> None:
        self._apply(self.model.redo())

    def copy(self) -> None:
        if self.model.has_selection():
            runtime.clipboard.set(self.model.selected_source())

    def cut(self) -> None:
        if self.model.has_selection():
            self.copy()
            self._apply(self.model.delete_selection())

    def paste(self, text: str | None = None) -> None:
        """Paste ``text``, or the clipboard's text (fetched from the client if shown)."""
        if text is not None:
            self._apply(self.model.paste(text))
        elif not runtime.spawn(self._paste_clipboard):
            self._apply(self.model.paste(runtime.clipboard.text))

    async def _paste_clipboard(self) -> None:
        self._apply(self.model.paste(await runtime.clipboard.get()))

    def type_text(self, text: str) -> None:
        """Insert typed text (what the hidden field delivers; tests call it directly)."""
        self._apply(self.model.type_text(text))

    @property
    def is_linear(self) -> bool:
        return self.model.linear

    def set_linear(self, linear: bool) -> None:
        """Edit the source as plain text (True) or as typeset mathematics (False); the
        content, undo history and, going to plain text, the caret position are kept."""
        if linear == self.model.linear:
            return
        self.model.set_linear(linear)
        self._typesetter = None
        self._layout = None
        self._scroll = 0.0
        self._refresh()

    def copy_latex(self) -> None:
        """Put the selection (or the whole expression) on the clipboard as LaTeX."""
        runtime.clipboard.set(self.model.selected_latex())

    def set_error_span(self, span: tuple[int, int] | None) -> None:
        """Underline the part of the drawing produced by source ``span`` (or clear)."""
        if span != self._error_span:
            self._error_span = span
            self.update()

    def focus(self) -> None:
        """Give this editor the keyboard focus."""
        self._set_focus(True)
        runtime.focus(self.input)

    # -- geometry -------------------------------------------------------------------------
    @property
    def width(self) -> float:
        return self._width

    def resize(self, width: float) -> None:
        """The editor's width on screen (reported by the client, or set by tests)."""
        if width > 0 and abs(width - self._width) > 0.5:
            self._width = width
            self._refresh()

    def typesetter(self) -> Typesetter:
        if self._typesetter is None or self._typesetter.size != self.size:
            self._typesetter = Typesetter(self.size, plain=self.model.linear)
            self._layout = None
        return self._typesetter

    def math_layout(self) -> Layout:
        """The current typeset layout (layout coordinates; see :meth:`to_widget`)."""
        typesetter = self.typesetter()
        wrap_width = self._width - 2 * _MARGIN_X
        key = (self.model.revision, id(typesetter), wrap_width)
        if self._layout is None or self._layout_key != key:
            self._layout = typesetter.layout(self.model.root, wrap_width)
            self._layout_key = key
        return self._layout

    def _base_font(self) -> fonts.Font:
        return self.typesetter().font(1.0)

    def _baseline(self) -> float:
        return _MARGIN_Y + max(self.math_layout().ascent, self._base_font().ascent())

    def _left(self) -> float:
        """Where the layout's origin sits, before scrolling."""
        if self.centered:
            free = self._width - self.math_layout().width
            if free > 2 * _MARGIN_X:
                return free / 2
        return _MARGIN_X

    def ink(self) -> str:
        """The colour the maths is drawn in."""
        return theme.current().text

    def to_widget(self, point: Point) -> Point:
        return Point(point.x + self._left() - self._scroll, point.y + self._baseline())

    def to_layout(self, point: Point) -> Point:
        return Point(point.x - self._left() + self._scroll, point.y - self._baseline())

    def caret_rect(self) -> Rect:
        x, top, bottom = self.math_layout().caret_line(self.model.cursor)
        a = self.to_widget(Point(x, top))
        return Rect(a.x - 0.5, a.y, 1.5, bottom - top)

    def _content_height(self) -> int:
        layout = self.math_layout()
        font = self._base_font()
        ascent = max(layout.ascent, font.ascent())
        descent = max(layout.descent, font.descent())
        return int(ascent + descent + 2 * _MARGIN_Y + 0.999)

    def content_height(self) -> int:
        """The height the editor needs for its content (it grows with the notation)."""
        return self._content_height()

    def content_width(self) -> float:
        return self.math_layout().width + 2 * _MARGIN_X

    def _ensure_caret_visible(self) -> None:
        layout = self.math_layout()
        view = max(1.0, self._width - 2 * _MARGIN_X)
        x = layout.caret_line(self.model.cursor)[0]
        if x - self._scroll > view - _SCROLL_MARGIN:
            self._scroll = x - view + _SCROLL_MARGIN
        if x - self._scroll < _SCROLL_MARGIN:
            self._scroll = x - _SCROLL_MARGIN
        self._scroll = max(0.0, min(self._scroll, layout.width - view + _SCROLL_MARGIN))

    def _refresh(self) -> None:
        """After any model change: relayout lazily, resize, scroll, redraw."""
        height = self._content_height()
        if height != self._height:
            self._height = height
            if hasattr(self, "control"):
                self.control.height = height
                runtime.push(self.control)
        self._ensure_caret_visible()
        self._caret_visible = True
        if self.has_focus:
            self._blink.start()
        self.update()

    def _apply(self, changed: bool) -> None:
        self._refresh()
        if changed:
            self.text_edited.emit(self.model.source)

    # -- caret blinking -----------------------------------------------------------------
    def _toggle_caret(self) -> None:
        self._caret_visible = not self._caret_visible
        caret = self._caret_shape
        if caret is not None and runtime.page_of(self.canvas) is not None:
            caret.paint = _fill(self.ink() if self._caret_visible else "#00000000")
            self.canvas.update()

    # -- focus ------------------------------------------------------------------------------
    def _set_focus(self, focused: bool) -> None:
        global _focused
        if focused == self.has_focus:
            return
        self.has_focus = focused
        if focused:
            if _focused is not None and _focused is not self:
                _focused._set_focus(False)
            _focused = self
            runtime.focus_tracker.gained(self)
            self._typed = ""
            if self.input.value:
                self.input.value = ""
                runtime.push(self.input)
            self._caret_visible = True
            self._blink.start()
        else:
            if _focused is self:
                _focused = None
            runtime.focus_tracker.lost(self)
            self._blink.stop()
            self._dragging = False
        self.update()
        self.focus_changed.emit(focused)

    # -- keyboard -----------------------------------------------------------------------
    def _on_input_change(self) -> None:
        value = self.input.value or ""
        typed = _inserted(self._typed, value)
        self._typed = value
        if time.monotonic() < self._suppress_until:
            return  # the field replaying its own redo history
        typed = "".join(ch for ch in typed if ch.isprintable())
        if typed:
            self.type_text(typed)

    def handle_key(
        self,
        key: str,
        *,
        shift: bool = False,
        ctrl: bool = False,
        alt: bool = False,
        meta: bool = False,
    ) -> bool:
        """An editing key pressed while this editor has the focus; True if handled.

        ``key`` is Flutter's key label ("Arrow Left", "Backspace", "Z"…). Characters
        arrive through the hidden text field instead, so plain letters return False.
        """
        model = self.model
        command = ctrl or meta
        letter = key.upper() if len(key) == 1 else ""
        if command and not alt:
            if letter == "Z" and not shift:
                self.undo()
            elif letter == "Y" or (letter == "Z" and shift):
                self._suppress_until = time.monotonic() + _FIELD_SUPPRESS_S
                self.redo()
            elif letter == "C":
                self.copy()
            elif letter == "X":
                self.cut()
            elif letter == "V":
                self._suppress_until = time.monotonic() + _FIELD_SUPPRESS_S
                self.paste()
            elif letter == "A":
                self.select_all()
            else:
                return False
            return True
        if ctrl or alt or meta:
            return False
        if key == "Tab":
            revision = model.revision
            if model.tab(-1 if shift else 1):
                self._apply(model.revision != revision)
                runtime.focus(self.input)  # the client moves the focus on Tab: take it back
                return True
            return False
        if key in ("Enter", "Numpad Enter"):
            self.submitted.emit()
        elif key in ("Arrow Up", "Arrow Down"):
            self._vertical(-1 if key == "Arrow Up" else 1)
        elif key in ("Arrow Left", "Arrow Right"):
            model.move(-1 if key == "Arrow Left" else 1, select=shift)
            self._refresh()
        elif key in ("Home", "End"):
            model.move_to_edge(-1 if key == "Home" else 1, select=shift)
            self._refresh()
        elif key == "Backspace":
            if model.is_empty:
                self.delete_empty.emit()
            else:
                self._apply(model.backspace())
        elif key == "Delete":
            self._apply(model.delete_forward())
        else:
            return False
        return True

    def _vertical(self, direction: int) -> None:
        target = self.model.vertical_target(direction)
        layout = self.math_layout()
        if target is None:
            line = layout.line_step(self.model.cursor, direction)  # in a wrapped list
            if line is None:
                self.navigate.emit(direction)
            else:
                self.model.set_cursor(line)
                self._refresh()
            return
        x = layout.caret_line(self.model.cursor)[0]
        self.model.set_cursor(Position(target, layout.nearest_index(target, x)))
        self._refresh()

    # -- mouse ----------------------------------------------------------------------------
    def press(self, point: Point, *, shift: bool | None = None) -> None:
        """The left button went down at ``point`` (control coordinates)."""
        shift = runtime.modifiers.shift if shift is None else shift
        self.model.set_cursor(self._position_at(point), select=shift)
        self._dragging = True
        self.focus()
        self._refresh()

    def drag_to(self, point: Point) -> None:
        if self._dragging:
            self.model.set_cursor(self._position_at(point), select=True)
            self._refresh()

    def release(self) -> None:
        self._dragging = False

    def click(self, point: Point, *, shift: bool = False) -> None:
        self.press(point, shift=shift)
        self.release()

    def double_click(self, point: Point) -> None:
        """Select the word (name or number) under the pointer, else the whole block."""
        pos = self._position_at(point)
        nodes = pos.lst.nodes

        def wordy(node: Node) -> bool:
            return isinstance(node, Char) and (node.text.isalnum() or node.text == ".")

        start = end = pos.index
        while start > 0 and wordy(nodes[start - 1]):
            start -= 1
        while end < len(nodes) and wordy(nodes[end]):
            end += 1
        if start == end:
            start, end = 0, len(nodes)
        self.model.select_range(pos.lst, start, end)
        self._dragging = False
        self._refresh()

    def _position_at(self, point: Point) -> Position:
        return self.math_layout().position_at(self.to_layout(point), self.model.root)

    def _on_resize(self, event: cv.CanvasResizeEvent) -> None:
        if math.isfinite(event.width):
            self.resize(event.width)

    # -- drawing ------------------------------------------------------------------------
    def update(self) -> None:
        """Redraw once the current callback is done (changes made together draw once)."""
        if not self._render_pending:
            self._render_pending = True
            events.call_soon(self._render_now)

    def _render_now(self) -> None:
        self._render_pending = False
        if runtime.page_of(self.canvas) is None:
            return
        self.render()
        self.canvas.update()
        self._update_menu()

    def _update_menu(self) -> None:
        items, model = self._menu_items, self.model
        selected = model.has_selection()
        states = {
            "undo": model.can_undo,
            "redo": model.can_redo,
            "cut": selected,
            "copy": selected,
            "select_all": not model.is_empty,
            "latex": not model.is_empty,
        }
        changed = False
        for name, enabled in states.items():
            if items[name].disabled == enabled:
                items[name].disabled = not enabled
                changed = True
        if items["linear"].checked != model.linear:
            items["linear"].checked = model.linear
            changed = True
        if changed:
            runtime.push(self.menu)

    def render(self) -> list[cv.Shape]:
        """Build the canvas shapes for the current state (and keep them on the canvas)."""
        layout = self.math_layout()
        current = theme.current()
        ink = self.ink()
        baseline = self._baseline()
        shapes: list[cv.Shape] = []

        if self.model.is_empty and self._placeholder:
            font = self._base_font()
            shapes.append(
                _glyph(self._left(), baseline, self._placeholder, font, current.placeholder)
            )

        ox, oy = self._left() - self._scroll, baseline
        selection = self.model.selection()
        if selection is not None:
            highlight = colors.with_alpha(current.accent, 90 if self.has_focus else 45)
            rect = layout.range_rect(*selection).adjusted(-1, -1, 1, 1).translated(ox, oy)
            shapes.append(cv.Rect(rect.x, rect.y, rect.width, rect.height, paint=_fill(highlight)))

        r, g, b, _ = colors.rgba(ink)
        for hole in layout.holes:
            rect = hole.translated(ox, oy)
            shapes.append(
                cv.Rect(
                    rect.x,
                    rect.y,
                    rect.width,
                    rect.height,
                    paint=_fill(colors.from_rgba(r, g, b, 18)),
                )
            )
            shapes.append(
                cv.Rect(
                    rect.x,
                    rect.y,
                    rect.width,
                    rect.height,
                    paint=_stroke(colors.from_rgba(r, g, b, 110), 1, dash=[1, 2]),
                )
            )

        for point, text, font in layout.glyphs:
            shapes.append(_glyph(point.x + ox, point.y + oy, text, font, ink))
        for path, width in layout.strokes:
            elements: list[cv.Path.PathElement] = []
            for step in path.translated(ox, oy).steps:
                if step[0] == "M":
                    elements.append(cv.Path.MoveTo(step[1], step[2]))
                elif step[0] == "L":
                    elements.append(cv.Path.LineTo(step[1], step[2]))
                elif len(step) == 5:
                    elements.append(cv.Path.QuadraticTo(step[1], step[2], step[3], step[4]))
            shapes.append(cv.Path(elements, paint=_stroke(ink, width)))

        if self._error_span is not None:
            for rect in self.error_rects():
                shifted = rect.translated(ox, oy)
                shapes.append(
                    _wave(shifted.left, shifted.right, shifted.bottom + 1.5, self.error_color)
                )

        self._caret_shape = None
        if self.has_focus and selection is None:
            x, top, bottom = layout.caret_line(self.model.cursor)
            color = ink if self._caret_visible else "#00000000"
            self._caret_shape = cv.Rect(
                x - 0.5 + ox, top + oy, 1.5, bottom - top, paint=_fill(color)
            )
            shapes.append(self._caret_shape)
        self.canvas.shapes = shapes
        return shapes

    def error_rects(self) -> list[Rect]:
        """Layout rectangles of the nodes the current error span covers."""
        if self._error_span is None:
            return []
        start, end = self._error_span
        length = len(self.model.source)
        end = min(end, length)
        if end <= start:  # point at the last character for end-of-input errors
            start, end = max(0, end - 1), max(1, end)
        layout = self.math_layout()
        rects = _covered(self.model.root, start, end, self.model.source_map, layout)
        if not rects and self.model.root.nodes:
            last = layout.nodes.get(id(self.model.root.nodes[-1]))
            rects = [last] if last is not None else []
        return rects


def _covered(
    lst: MathList, start: int, end: int, source_map: dict[int, tuple[int, int]], layout: Layout
) -> list[Rect]:
    rects: list[Rect] = []
    for node in lst.nodes:
        span = source_map.get(id(node))
        rect = layout.nodes.get(id(node))
        if span is None or rect is None or span[1] <= start or span[0] >= end:
            continue
        if isinstance(node, Structure) and not (start <= span[0] and span[1] <= end):
            inner = [r for b in node.blocks for r in _covered(b, start, end, source_map, layout)]
            rects.extend(inner or [rect])
        else:
            rects.append(rect)
    return rects


def _xy(offset: ft.Offset | None) -> Point:
    if offset is None:
        return Point(0.0, 0.0)
    return Point(float(offset.x), float(offset.y))


def _fill(color: str) -> ft.Paint:
    return ft.Paint(color=color, style=ft.PaintingStyle.FILL)


def _stroke(color: str, width: float, *, dash: list[float] | None = None) -> ft.Paint:
    return ft.Paint(
        color=color,
        stroke_width=width,
        style=ft.PaintingStyle.STROKE,
        stroke_cap=ft.StrokeCap.ROUND,
        stroke_join=ft.StrokeJoin.ROUND,
        stroke_dash_pattern=dash,
    )


def _glyph(x: float, baseline: float, text: str, font: fonts.Font, color: str) -> cv.Text:
    """``text`` with its baseline origin at ``(x, baseline)``."""
    return cv.Text(
        x,
        baseline - font.ascent(),
        text,
        style=ft.TextStyle(size=font.size, font_family=font.family, color=color),
    )


def _wave(left: float, right: float, y: float, color: str) -> cv.Path:
    elements: list[cv.Path.PathElement] = [cv.Path.MoveTo(left, y)]
    x, up = left, True
    while x < right:
        x = min(right, x + 2.0)
        elements.append(cv.Path.LineTo(x, y - 1.2 if up else y + 0.6))
        up = not up
    return cv.Path(elements, paint=_stroke(color, 1.2))
