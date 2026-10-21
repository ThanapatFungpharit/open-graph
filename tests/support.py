"""Helpers that drive the interface the way the Flet client would, without one.

The window's objects work without a page: these helpers deliver the same calls the
client's events produce (typed text, key presses, clicks, sizes) and run the private
event loop that timers and background results use.
"""

from __future__ import annotations

from collections.abc import Callable

import flet.canvas as cv

from opengraph import events
from opengraph.ui.geometry import Point
from opengraph.ui.main_window import MainWindow
from opengraph.ui.mathedit.widget import MathEdit

CANVAS_SIZE = (800, 600)
EDITOR_WIDTH = 280.0


def make_window() -> MainWindow:
    """A main window laid out as the client would report it."""
    window = MainWindow()
    window.canvas.resize(*CANVAS_SIZE)
    for row_id in window.panel.row_ids():
        if row_id in window.panel._rows:
            window.panel.row(row_id).edit.resize(EDITOR_WIDTH)
    return window


def wait(ms: float) -> None:
    """Run the event loop for ``ms`` milliseconds (timers fire, results arrive)."""
    events.process_events(ms / 1000)


def wait_until(condition: Callable[[], object], timeout_ms: float = 5000) -> None:
    assert events.wait_until(condition, timeout_ms / 1000), "condition not reached in time"


def key_clicks(edit: MathEdit, text: str) -> None:
    """Type ``text`` one key at a time into a focused editor."""
    if not edit.has_focus:
        edit.focus()
    for char in text:
        edit.type_text(char)


def key(
    window_or_edit: MainWindow | MathEdit,
    name: str,
    *,
    shift: bool = False,
    ctrl: bool = False,
    alt: bool = False,
) -> bool:
    """Press a key: through the window (editor, shortcuts, graph) or straight to an editor."""
    if isinstance(window_or_edit, MathEdit):
        if not window_or_edit.has_focus:
            window_or_edit.focus()
        return window_or_edit.handle_key(name, shift=shift, ctrl=ctrl, alt=alt)
    return window_or_edit.handle_key(name, shift=shift, ctrl=ctrl, alt=alt)


def layer_texts(window: MainWindow) -> list[str]:
    doc = window.controller.document
    return [doc.get(layer.key).text for layer in window.canvas._layers]  # type: ignore[arg-type]


def first_id(window: MainWindow) -> int:
    return window.controller.document.items[0].id


def to_widget(edit: MathEdit, x: float, y: float) -> Point:
    return edit.to_widget(Point(x, y))


def shapes_in_color(shapes: list[cv.Shape], color: str) -> list[cv.Shape]:
    """Shapes painted (stroked or filled) in ``color``."""
    found = []
    for shape in shapes:
        paint = getattr(shape, "paint", None)
        if paint is not None and str(paint.color).lower() == color.lower():
            found.append(shape)
    return found


def stroked_points(shapes: list[cv.Shape], color: str) -> list[tuple[float, float]]:
    """Every vertex of the lines and point sets drawn in ``color``."""
    points: list[tuple[float, float]] = []
    for shape in shapes_in_color(shapes, color):
        if isinstance(shape, cv.Points) and shape.points:
            for point in shape.points:
                x, y = point if isinstance(point, tuple) else (point.x, point.y)
                points.append((float(x), float(y)))
    return points


def near(points: list[tuple[float, float]], x: float, y: float, radius: float = 2.0) -> bool:
    return any((px - x) ** 2 + (py - y) ** 2 <= radius**2 for px, py in points)


def drawn_near(shapes: list[cv.Shape], color: str, x: float, y: float, radius: float = 2.0) -> bool:
    """Whether a line drawn in ``color`` passes within ``radius`` pixels of ``(x, y)``."""
    for shape in shapes_in_color(shapes, color):
        if not isinstance(shape, cv.Points) or not shape.points:
            continue
        pts = [p if isinstance(p, tuple) else (p.x, p.y) for p in shape.points]
        if shape.point_mode == cv.PointMode.LINES:
            pairs = list(zip(pts[::2], pts[1::2], strict=False))
        else:
            pairs = list(zip(pts, pts[1:], strict=False))
        for (x0, y0), (x1, y1) in pairs:
            dx, dy = x1 - x0, y1 - y0
            length = dx * dx + dy * dy
            t = 0.0 if not length else max(0.0, min(1.0, ((x - x0) * dx + (y - y0) * dy) / length))
            if (x0 + t * dx - x) ** 2 + (y0 + t * dy - y) ** 2 <= radius**2:
                return True
    return False
