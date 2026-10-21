"""Dragging points on the canvas and reordering rows by their grip."""

from __future__ import annotations

import pytest

from opengraph.controller import LIVE_COMPILE_MS
from opengraph.engine.compiler import compile_document
from opengraph.engine.plots import PointPlot
from opengraph.ui.geometry import Point
from opengraph.ui.main_window import MainWindow
from tests.support import first_id, wait


def screen_point(win: MainWindow, x: float, y: float) -> Point:
    sx, sy = win.canvas.viewport.to_screen(x, y)
    return Point(round(float(sx)), round(float(sy)))


def drag_canvas(win: MainWindow, start: Point, end: Point) -> None:
    canvas = win.canvas
    canvas.press(start)
    for step in range(1, 5):
        f = step / 4
        canvas.move(Point(start.x + (end.x - start.x) * f, start.y + (end.y - start.y) * f))
        wait(LIVE_COMPILE_MS * 2)
    canvas.release(end)


def test_point_reports_drag_coordinates() -> None:
    results = compile_document(
        [(1, "q = 3.8"), (2, "w = 5.5"), (3, "point[q, w]"), (4, "point[-1, 2q]")]
    )
    assert results[3].drag == ("q", "w")
    assert results[4].drag == ((6, 8), None)  # the literal's span; 2q is computed
    assert compile_document([(1, "point[1, 2], point[3, 4]")])[1].drag is None


def test_dragging_point_rewrites_its_sliders(window: MainWindow) -> None:
    c = window.controller
    q = first_id(window)
    c.set_text(q, "q = 3.8")
    w = c.add_expression("w = 5.5")
    p = c.add_expression("point[q, w]")
    c.flush()
    layer = next(layer for layer in window.canvas._layers if layer.key == p)
    assert layer.draggable
    window.canvas.viewport.reset()
    start = screen_point(window, 3.8, 5.5)
    drag_canvas(window, start, Point(start.x - 60, start.y + 40))
    c.flush()
    plot = c.results[p].plot
    assert isinstance(plot, PointPlot)
    assert plot.x < 3.8 and plot.y < 5.5
    assert window.panel.row(q).edit.text() == c.document.get(q).text
    assert float(c.document.get(q).text.removeprefix("q = ")) == pytest.approx(plot.x)
    assert float(c.document.get(w).text.removeprefix("w = ")) == pytest.approx(plot.y)
    assert c.document.get(p).text == "point[q, w]"  # the point row itself is unchanged
    slider = c.document.get(q).slider
    assert slider is not None
    assert window.panel.row(q).slider_position == slider.position_of(plot.x)


def test_dragging_literal_point_rewrites_its_numbers(window: MainWindow) -> None:
    c = window.controller
    p = first_id(window)
    c.set_text(p, "point[-1, 2]")
    c.flush()
    window.canvas.viewport.reset()
    start = screen_point(window, -1, 2)
    drag_canvas(window, start, Point(start.x + 80, start.y))
    c.flush()
    text = c.document.get(p).text
    x, y = (float(v) for v in text.removeprefix("point[").rstrip("]").split(","))
    assert x > -1 and y == 2


def test_computed_point_is_not_draggable(window: MainWindow) -> None:
    c = window.controller
    p = first_id(window)
    c.set_text(p, "point[1 + 1, 2]")  # x is computed, but y could move
    q = c.add_expression("point[sin(1), cos(1)]")
    c.flush()
    draggable = {layer.key: layer.draggable for layer in window.canvas._layers}
    assert draggable == {p: True, q: False}


def test_grip_drag_reorders_rows(window: MainWindow) -> None:
    c = window.controller
    a = first_id(window)
    c.set_text(a, "y = x")
    b = c.add_expression("y = 2x")
    d = c.add_expression("y = 3x")
    c.flush()
    panel = window.panel
    panel.reorder(0, 2)  # the client reports a row dragged from the top to the bottom
    assert [item.id for item in c.document.items] == [b, d, a]
    assert panel.row_ids() == [b, d, a]
    assert panel.list.controls == [panel.widget(i).control for i in (b, d, a)]
    assert [layer.key for layer in window.canvas._layers] == [b, d, a]
    assert c.modified


def test_move_expression_moves_the_row(window: MainWindow) -> None:
    c = window.controller
    a = first_id(window)
    b = c.add_expression("y = 2x")
    c.move_expression(b, 0)
    assert window.panel.row_ids() == [b, a]
    assert [item.id for item in c.document.items] == [b, a]


def test_vector_points_report_drag_coordinates() -> None:
    results = compile_document(
        [
            (1, "p = point[1, 2.5]"),
            (2, "q = point[3, 4]"),
            (3, "p + q"),
            (4, "2point[1, 2]"),
            (5, "r = 2p"),
        ]
    )
    assert results[1].drag == ((10, 11), (13, 16))  # spans of the literals in its own row
    assert results[3].drag is None  # computed: it follows p and q
    assert results[4].drag is None
    assert results[5].drag is None


def test_dragging_named_point_moves_vector_sums(window: MainWindow) -> None:
    c = window.controller
    p = first_id(window)
    c.set_text(p, "p = point[1, 2]")
    c.add_expression("q = point[3, 4]")
    total = c.add_expression("p + q")
    c.flush()
    layers = {layer.key: layer for layer in window.canvas._layers}
    assert layers[p].draggable and not layers[total].draggable
    window.canvas.viewport.reset()
    start = screen_point(window, 1, 2)
    drag_canvas(window, start, Point(start.x + 60, start.y))
    c.flush()
    assert c.document.items[0].text.startswith("p = point[")
    moved = c.results[p].plot
    plot = c.results[total].plot
    assert isinstance(moved, PointPlot) and isinstance(plot, PointPlot)
    assert moved.x > 1 and moved.y == pytest.approx(2)
    assert plot.x == pytest.approx(moved.x + 3) and plot.y == pytest.approx(6)
