"""End-to-end tests of the window, driven the way the Flet client drives it."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from opengraph.controller import COMPILE_DELAY_MS, LIVE_COMPILE_MS
from opengraph.document import SliderRange
from opengraph.engine.plots import ExplicitPlot, ImplicitPlot, PointSetPlot
from opengraph.ui import dialogs, fonts, runtime
from opengraph.ui.expression_panel import ExpressionEdit, SliderSettingsDialog
from opengraph.ui.geometry import Point
from opengraph.ui.main_window import MainWindow, syntax_reference
from opengraph.ui.mathedit.layout import Typesetter
from opengraph.ui.mathedit.nodes import Frac, Paren, Position, Sup
from tests.support import (
    drawn_near,
    first_id,
    key,
    key_clicks,
    layer_texts,
    wait,
    wait_until,
)


def test_typing_an_expression_plots_it(window: MainWindow) -> None:
    row = window.panel.row(first_id(window))
    key_clicks(row.edit, "y = x^2")
    wait_until(lambda: layer_texts(window) == ["y = x^2"], timeout_ms=2000)
    assert isinstance(window.canvas._layers[0].plot, ExplicitPlot)
    assert window.controller.modified
    assert window.title() == "Untitled* — OpenGraph"


def test_edits_are_debounced(window: MainWindow) -> None:
    row = window.panel.row(first_id(window))
    key_clicks(row.edit, "x")
    assert layer_texts(window) == []  # not yet: waiting for typing to pause
    wait(COMPILE_DELAY_MS * 3)
    assert layer_texts(window) == ["x"]


def test_enter_adds_row_and_focuses_it(window: MainWindow) -> None:
    first = first_id(window)
    row = window.panel.row(first)
    key_clicks(row.edit, "sin(x)")
    assert key(window, "Enter")
    ids = [i.id for i in window.controller.document.items]
    assert len(ids) == 2 and ids[0] == first
    assert window.panel.row_ids() == ids
    assert layer_texts(window) == ["sin(x)"]  # Enter flushed the pending compile
    assert window.panel.row(ids[1]).edit.has_focus and not row.edit.has_focus
    # Backspace on the new empty row removes it.
    assert key(window, "Backspace")
    assert [i.id for i in window.controller.document.items] == [first]


def test_multiple_expressions_hide_show_delete(window: MainWindow) -> None:
    c = window.controller
    c.set_text(first_id(window), "y = x")
    b = c.add_expression("x^2 + y^2 = 4")
    d = c.add_expression("point[1, 1]")
    c.flush()
    assert layer_texts(window) == ["y = x", "x^2 + y^2 = 4", "point[1, 1]"]

    window.panel.row(b).toggle.click()
    assert layer_texts(window) == ["y = x", "point[1, 1]"]
    assert not c.document.get(b).visible
    window.panel.row(b).toggle.click()
    assert layer_texts(window) == ["y = x", "x^2 + y^2 = 4", "point[1, 1]"]

    window.panel.row(d).delete_requested.emit(d)  # the ✕ button
    assert layer_texts(window) == ["y = x", "x^2 + y^2 = 4"]
    assert d not in window.panel.row_ids()

    c.set_text(b, "x^2 + y^2 = 9")
    c.flush()
    assert isinstance(window.canvas._layers[1].plot, ImplicitPlot)


def test_invalid_row_shows_feedback_without_breaking_others(window: MainWindow) -> None:
    c = window.controller
    c.set_text(first_id(window), "y = 2x")
    bad = c.add_expression("y = sin(x))")
    value = c.add_expression("2^10")
    c.flush()
    assert layer_texts(window) == ["y = 2x"]
    bad_row = window.panel.row(bad)
    assert bad_row.feedback.visible and "Unmatched" in (bad_row.feedback.value or "")
    assert bad_row.error is True
    assert window.panel.row(value).feedback.value == "= 1024"
    # Fixing the row plots it.
    c.set_text(bad, "y = sin(x)")
    c.flush()
    assert layer_texts(window) == ["y = 2x", "y = sin(x)"]
    assert not bad_row.feedback.visible


def test_dependent_rows_update_together(window: MainWindow) -> None:
    c = window.controller
    defn = first_id(window)
    c.set_text(defn, "f(x) = x^2")
    use = c.add_expression("y = f(x) + 1")
    c.flush()
    use_plot = c.results[use].plot
    assert isinstance(use_plot, ExplicitPlot) and use_plot.fn.scalar(x=2.0) == 5
    c.set_text(defn, "f(x) = x^3")
    c.flush()
    new_plot = c.results[use].plot
    assert isinstance(new_plot, ExplicitPlot) and new_plot.fn.scalar(x=2.0) == 9


def test_unchanged_rows_keep_their_plot_object(window: MainWindow) -> None:
    c = window.controller
    c.set_text(first_id(window), "y = cos(x)")
    other = c.add_expression("y = 1")
    c.flush()
    before = c.results[first_id(window)].plot
    c.set_text(other, "y = 2")
    c.flush()
    assert c.results[first_id(window)].plot is before  # canvas cache stays valid


def test_color_change_updates_layer(window: MainWindow) -> None:
    c = window.controller
    c.set_text(first_id(window), "y = x")
    c.flush()
    window.panel.row(first_id(window)).choose_color("#123456")
    assert window.canvas._layers[0].color == "#123456"
    assert c.document.get(first_id(window)).color == "#123456"


def test_color_dialog_accepts_palette_and_typed_colors(window: MainWindow) -> None:
    c = window.controller
    row = window.panel.row(first_id(window))
    dialog = dialogs.ColorDialog("Expression Color", "#000000", row.choose_color)
    dialog.field.value = "nonsense"
    dialog.accept()
    assert dialog.error.visible
    dialog.field.value = "#ABCDEF"
    dialog.accept()
    assert c.document.get(first_id(window)).color == "#abcdef"
    dialog.choose("#16a34a")
    assert c.document.get(first_id(window)).color == "#16a34a"


def test_curve_is_drawn_where_expected(window: MainWindow) -> None:
    c = window.controller
    c.set_text(first_id(window), "y = x")
    c.flush()
    window.panel.row(first_id(window)).choose_color("#ff0000")
    wait_until(lambda: not window.canvas.is_rendering)
    shapes = window.canvas.render()
    vp = window.canvas.viewport
    sx, sy = vp.to_screen(1.3, 1.3)
    assert drawn_near(shapes, "#ff0000", float(sx), float(sy), radius=1.5)
    # And not somewhere off the line.
    ox, oy = vp.to_screen(1.3, -1.3)
    assert not drawn_near(shapes, "#ff0000", float(ox), float(oy), radius=3)


def test_drag_pans_and_wheel_zooms_around_cursor(window: MainWindow) -> None:
    canvas = window.canvas
    vp = canvas.viewport
    world = vp.to_world(300.0, 300.0)
    canvas.press(Point(300, 300))
    canvas.move(Point(350, 260))
    canvas.release(Point(350, 260))
    assert vp.to_world(350.0, 260.0) == pytest.approx(world)

    anchor = Point(420, 180)
    world = vp.to_world(anchor.x, anchor.y)
    scale = vp.scale_x
    canvas.wheel(anchor, 120)
    assert vp.scale_x > scale
    assert vp.to_screen(*world) == pytest.approx((anchor.x, anchor.y))


def test_scroll_event_zooms_in_when_scrolling_up(window: MainWindow) -> None:
    import flet as ft

    canvas = window.canvas
    scale = canvas.viewport.scale_x
    event = ft.ScrollEvent(
        name="scroll",
        control=canvas.detector,
        local_position=ft.Offset(400, 300),
        global_position=ft.Offset(400, 300),
        scroll_delta=ft.Offset(0, -100),
    )
    canvas._on_scroll(event)
    assert canvas.viewport.scale_x > scale


def test_keyboard_navigation_of_canvas(window: MainWindow) -> None:
    canvas = window.canvas
    scale = canvas.viewport.scale_x
    assert window.panel.row(first_id(window)).edit.has_focus  # the first row at start
    assert not key(window, "+")
    canvas.press(Point(100, 100))  # a click on the graph gives it the keys
    canvas.release(Point(100, 100))
    assert not window.panel.row(first_id(window)).edit.has_focus
    assert key(window, "+")
    assert canvas.viewport.scale_x > scale
    assert key(window, "Arrow Right")
    assert canvas.viewport.center_x > 0
    assert key(window, "0")
    assert canvas.viewport.center_x == 0 and canvas.viewport.scale_x == scale
    # While a text input has the focus, the same keys are left to it.
    window.panel.row(first_id(window)).edit.focus()
    runtime.focus_tracker.gained(object())
    assert not key(window, "+")
    assert canvas.viewport.scale_x == scale


def test_menu_shortcuts(window: MainWindow) -> None:
    c = window.controller
    scale = window.canvas.viewport.scale_x
    assert key(window, "=", ctrl=True)
    assert window.canvas.viewport.scale_x > scale
    assert key(window, "0", ctrl=True)
    assert window.canvas.viewport.scale_x == scale
    assert key(window, "N", ctrl=True, shift=True)
    assert len(c.document) == 2
    assert key(window, "J", ctrl=True)
    assert window.tasks_dialog.visible
    assert key(window, ",", ctrl=True)
    assert window.settings_dialog.is_visible()
    assert key(window, "L", ctrl=True, shift=True)
    assert window.panel.linear_input and window._linear_action.is_checked()


def test_hover_traces_curve(window: MainWindow) -> None:
    c = window.controller
    c.set_text(first_id(window), "y = x^2")
    c.flush()
    wait_until(lambda: not window.canvas.is_rendering)
    window.canvas.render()  # populate the geometry cache
    vp = window.canvas.viewport
    sx, sy = vp.to_screen(1.0, 1.0)
    window.canvas.move(Point(round(float(sx)), round(float(sy)) + 3))
    trace = window.canvas._trace
    assert trace is not None
    # The traced point is on the curve at the cursor's x, not at a sampled vertex.
    tx, ty = (float(v) for v in trace.label.strip("()").split(", "))
    assert tx == pytest.approx(1.0, abs=0.02)
    assert ty == pytest.approx(tx**2, abs=0.002)
    assert window.current_message() == "" and window.coords.value
    # Away from every curve: no trace.
    window.canvas.move(Point(int(sx) + 200, int(sy) + 200))
    assert window.canvas._trace is None


def test_panning_reuses_cached_geometry(window: MainWindow) -> None:
    c = window.controller
    c.set_text(first_id(window), "y = sin(x)")
    c.flush()
    canvas = window.canvas
    wait_until(lambda: not canvas.is_rendering)
    canvas.render()
    canvas.viewport.pan_pixels(5, 0)
    canvas.render()
    wait_until(lambda: not canvas.is_rendering)
    entry = canvas._cache[first_id(window)][0]
    canvas.viewport.pan_pixels(40, 10)
    canvas.render()
    assert canvas._cache[first_id(window)][0] is entry
    canvas.viewport.zoom_center(1.5)
    canvas.render()
    wait_until(lambda: not canvas.is_rendering)
    canvas.render()
    assert canvas._cache[first_id(window)][0] is not entry


def test_symbolic_derivative_adds_row(window: MainWindow) -> None:
    pytest.importorskip("sympy")
    c = window.controller
    c.set_text(first_id(window), "y = x^3")
    c.run_symbolic(first_id(window), "derivative")
    wait_until(lambda: len(c.document) == 2, timeout_ms=30_000)
    assert c.document.items[1].text == "y = 3*x^2"


def test_symbolic_simplify_keeps_equation(window: MainWindow) -> None:
    c = window.controller
    c.set_text(first_id(window), "x^2 + 2*x = 0")
    c.run_symbolic(first_id(window), "simplify")
    wait_until(lambda: len(c.document) == 2, timeout_ms=30_000)
    assert c.document.items[1].text == "x*(x + 2) = 0"
    assert not c.results[c.document.items[1].id].is_error


def test_simplify_unassigned_variables_in_row(window: MainWindow) -> None:
    c = window.controller
    row_id = first_id(window)
    c.set_text(row_id, "simplify(a*x + a*x)")
    c.flush()
    wait_until(lambda: c.results[row_id].message == "= 2*a*x", timeout_ms=30_000)
    assert not c.results[row_id].is_error
    assert len(c.document) == 1


def test_symbolic_simplify_real_part_of_euler_exponential(window: MainWindow) -> None:
    c = window.controller
    c.set_text(first_id(window), "re(e^ix)")
    c.run_symbolic(first_id(window), "simplify")
    wait_until(lambda: len(c.document) == 2, timeout_ms=30_000)
    assert c.document.items[1].text == "y = cos(x)"
    assert not c.results[c.document.items[1].id].is_error


def test_symbolic_solve_adds_points(window: MainWindow) -> None:
    pytest.importorskip("sympy")
    c = window.controller
    c.set_text(first_id(window), "x^2 - 4")
    c.run_symbolic(first_id(window), "solve")
    wait_until(lambda: len(c.document) == 2, timeout_ms=30_000)
    assert [i.text for i in c.document.items[1:]] == ["point[list[-2, 2], 0]"]
    assert not c.results[c.document.items[1].id].is_error


def test_symbolic_solve_for_an_unknown_reports_exact_roots(window: MainWindow) -> None:
    pytest.importorskip("sympy")
    c = window.controller
    messages: list[str] = []
    c.status_message.connect(messages.append)
    c.set_text(first_id(window), "a^2 = 2")
    c.flush()
    wait_until(lambda: "sqrt" in c.results[first_id(window)].message, timeout_ms=30_000)
    assert (
        c.results[first_id(window)].message
        == "a = list[-sqrt(2), sqrt(2)]  (≈ list[-1.41421, 1.41421])"
    )
    c.run_symbolic(first_id(window), "derivative")  # only Solve applies
    assert messages[-1] == "That operation doesn't apply to this row"
    c.run_symbolic(first_id(window), "solve")
    wait_until(lambda: messages[-1].startswith("Exact solutions"), timeout_ms=30_000)
    assert messages[-1] == "Exact solutions: a = list[-sqrt(2), sqrt(2)]"
    assert window.current_message() == messages[-1]
    assert len(c.document) == 1  # nothing is added


def test_unknowns_are_solved_exactly_when_sympy_can(window: MainWindow) -> None:
    pytest.importorskip("sympy")
    c = window.controller
    c.set_text(first_id(window), "3a = 1")
    c.add_expression("b^3 = 2")
    c.add_expression("y = a x + b")
    c.add_expression("sin(t) = 0.5")
    c.flush()
    ids = [item.id for item in c.document.items]
    assert c.results[ids[1]].message.startswith("b ≈ 1.25992")  # numerical until SymPy answers
    wait_until(lambda: c.results[ids[1]].message.startswith("b = "), timeout_ms=30_000)
    assert c.wait_for_jobs()
    wait_until(lambda: c.results[ids[0]].message.startswith("a = 1/3"), timeout_ms=30_000)
    assert c.results[ids[0]].message == "a = 1/3  (≈ 0.333333)"
    assert c.results[ids[1]].message == "b = 2^(1/3)  (≈ 1.25992)"
    assert c.results[ids[2]].is_error  # equation feedback never binds names
    # Infinitely many: the exact general solution, and t supplies a bounded preview.
    wait_until(lambda: "ℤ" in c.results[ids[3]].message, timeout_ms=30_000)
    assert c.results[ids[3]].message == "t = pi/6 + 2*pi*n, 5*pi/6 + 2*pi*n  (n ∈ ℤ)"
    c.add_expression("t + 1")
    c.flush()
    assert c.results[c.document.items[-1].id].is_error


def test_symbolic_solve_plots_periodic_families(window: MainWindow) -> None:
    pytest.importorskip("sympy")
    c = window.controller
    messages: list[str] = []
    c.status_message.connect(messages.append)
    c.set_text(first_id(window), "(x - 1) sin(x)")
    c.run_symbolic(first_id(window), "solve")
    wait_until(lambda: not c.busy, timeout_ms=30_000)
    assert [i.text for i in c.document.items[1:]] == ["point[1, 0]", "point[pi*list[-20...20], 0]"]
    assert messages[-1] == "Infinitely many solutions: x = 1, pi*n  (n ∈ ℤ)"
    assert not any(c.results[i.id].is_error for i in c.document.items)


def test_symbolic_solve_plots_nonlinear_generators(window: MainWindow) -> None:
    c = window.controller
    c.set_text(first_id(window), "sin(sqrt(x))")
    c.run_symbolic(first_id(window), "solve")
    wait_until(lambda: not c.busy, timeout_ms=30_000)
    points = [item for item in c.document.items[1:] if "seq[" in item.text]
    assert len(points) == 2
    for item in points:
        result = c.results[item.id]
        assert not result.is_error, result.message
        assert isinstance(result.plot, PointSetPlot)
        assert len(result.plot.points) == 100
        assert all(x >= 0 and y == 0 for x, y in result.plot.points)


def test_solve_for_buttons_add_rearranged_rows(window: MainWindow) -> None:
    pytest.importorskip("sympy")
    c = window.controller
    c.set_text(first_id(window), "y = a x + b")
    c.flush()
    row = window.panel.row(first_id(window))
    assert row.solve_bar.visible
    assert row.solve_bar.labels() == ["a", "b"]
    row.solve_bar.choose("b")
    wait_until(lambda: len(c.document) == 2, timeout_ms=30_000)
    assert c.document.items[1].text == "formula(b, -a*x + y)"
    assert c.document.items[0].text == "y = a x + b"  # the original row is kept
    c.set_text(first_id(window), "y = 2x")
    c.flush()
    assert not row.solve_bar.visible
    c.set_text(first_id(window), "x^2 + y^2 = rho^2")
    c.flush()
    assert row.solve_bar.labels() == ["ρ"]  # Greek names show their letter
    row.solve_bar.choose("rho")  # several solutions: one row with a list
    wait_until(lambda: len(c.document) == 3, timeout_ms=30_000)
    assert c.document.items[1].text == "formula(rho, list[-sqrt(x^2 + y^2), sqrt(x^2 + y^2)])"


def test_add_slider_suggestions_define_the_names(window: MainWindow) -> None:
    c = window.controller
    c.set_text(first_id(window), "y = a sin(k x) + b")
    c.flush()
    row = window.panel.row(first_id(window))
    bar = row.slider_suggestions
    assert bar.visible
    assert bar.labels() == ["a", "k", "b"]
    assert bar.all_button is not None and bar.all_button.visible
    bar.choose("k")  # just k
    assert [i.text for i in c.document.items] == ["y = a sin(k x) + b", "k = 1"]
    assert bar.labels() == ["a", "b"]
    bar.choose_all()
    texts = [i.text for i in c.document.items]
    assert texts == ["y = a sin(k x) + b", "a = 1", "b = 1", "k = 1"]
    assert not bar.visible and not row.solve_bar.visible
    result = c.results[first_id(window)]
    assert not result.is_error and isinstance(result.plot, ExplicitPlot)
    slider_row = window.panel.row(c.document.items[1].id)
    assert slider_row.slider_bar.visible  # "a = 1" gets a slider
    c.set_text(first_id(window), "c + 1")
    c.flush()
    assert bar.visible and bar.all_button is not None and not bar.all_button.visible


def test_brackets_close_themselves_while_typing(window: MainWindow) -> None:
    edit = first_edit(window)
    key_clicks(edit, "y = 2(x+1")
    assert edit.text() == "y = 2(x+1)"
    key_clicks(edit, ")^2")  # ")" steps out of the bracket
    window.controller.flush()
    assert edit.text() == "y = 2(x+1)^2"
    assert value_at(window, first_id(window), 1.0) == pytest.approx(8.0)


@pytest.mark.parametrize("text", ["[)", "[0,1)", "(0,1]", "point[1,2]", "point[[0,1),2]"])
def test_point_and_mixed_brackets_can_be_typed(window: MainWindow, text: str) -> None:
    edit = first_edit(window)
    key_clicks(edit, text)
    assert edit.text() == text
    window.controller.flush()
    if text != "[)":
        assert not window.controller.results[first_id(window)].is_error


def test_save_and_reopen(tmp_path: Path, window: MainWindow) -> None:
    c = window.controller
    c.set_text(first_id(window), "f(x) = x/2")
    c.add_expression("point[3, 4]")
    window.canvas.viewport.pan_pixels(100, 0)
    path = tmp_path / "g.ograph.json"
    c.save_document(path)
    assert not c.modified
    c.new_document()
    assert len(c.document) == 1
    assert window.open_path(path)
    assert [i.text for i in c.document.items] == ["f(x) = x/2", "point[3, 4]"]
    assert window.canvas.viewport.center_x == pytest.approx(-2.0)
    assert len(window.canvas._layers) == 2
    assert window.title() == "g.ograph.json — OpenGraph"


def test_new_file_asks_before_discarding_changes(window: MainWindow) -> None:
    c = window.controller
    c.set_text(first_id(window), "y = x")
    c.flush()
    window.new_file()
    (question,) = dialogs.host.shown
    assert c.document.items[0].text == "y = x"  # nothing happens until answered
    cancel, discard, _save = question.actions
    cancel.on_click()  # type: ignore[call-arg, misc]
    assert c.document.items[0].text == "y = x" and not dialogs.host.shown
    window.new_file()
    discard = dialogs.host.shown[0].actions[1]
    discard.on_click()  # type: ignore[call-arg, misc]
    assert c.document.items[0].text == "" and not c.modified


def test_open_failure_is_reported(tmp_path: Path, window: MainWindow) -> None:
    broken = tmp_path / "broken.ograph.json"
    broken.write_text("{not json", encoding="utf-8")
    assert not window.open_path(broken)
    (message,) = dialogs.host.shown
    assert message.title == "Open Failed"


def test_syntax_reference_lists_all_functions() -> None:
    text = syntax_reference()
    for name in ("sin(x)", "log(x) or log(x, b)", "max(a, b, ...)", "mod(a, b)"):
        assert name in text
    for name in ("cancel(f)", "together(f)", "apart(f, t)", "collect(f, t)", "trigsimp(f)"):
        assert name in text


def test_reference_and_about_dialogs_open(window: MainWindow) -> None:
    window.show_reference()
    assert key(window, "F1")
    window.show_about()
    titles = [d.title for d in dialogs.host.shown]  # type: ignore[attr-defined]
    assert titles == ["Syntax Reference", "Syntax Reference", "About OpenGraph"]


# -- sliders ------------------------------------------------------------------------


def slope_at_1(win: MainWindow, item_id: int) -> float:
    plot = win.controller.results[item_id].plot
    assert isinstance(plot, ExplicitPlot)
    return plot.fn.scalar(x=1.0)


def test_only_literal_constants_get_a_slider(window: MainWindow) -> None:
    c = window.controller
    k = first_id(window)
    c.set_text(k, "k = 1.5")
    b = c.add_expression("b = 2k")
    y = c.add_expression("y = k x")
    c.flush()
    assert window.panel.row(k).slider_bar.visible
    assert not window.panel.row(k).slider_settings_action.disabled
    assert not window.panel.row(b).slider_bar.visible
    assert not window.panel.row(y).slider_bar.visible
    assert c.document.get(k).slider == SliderRange(-10, 10, 0.1)
    assert window.panel.row(k).slider_position == 115  # (1.5 - -10) / 0.1
    c.set_text(k, "k = 3b")  # no longer a plain number (and now circular)
    c.flush()
    assert not window.panel.row(k).slider_bar.visible
    assert window.panel.row(k).slider_settings_action.disabled


def test_dragging_slider_animates_dependent_curve(window: MainWindow) -> None:
    c = window.controller
    k = first_id(window)
    c.set_text(k, "k = 0")
    y = c.add_expression("y = k x + 1")
    c.flush()
    assert slope_at_1(window, y) == 1.0
    row = window.panel.row(k)
    start = row.slider_position
    seen = []
    for dx in (5, 10, 15):
        row.move_slider(start + dx)
        # Live: well before the typing debounce would have fired.
        wait(LIVE_COMPILE_MS * 3)
        seen.append(slope_at_1(window, y))
    assert seen[0] > 1.0 and seen == sorted(seen) and len(set(seen)) == 3
    # The row text holds the value the curve was drawn with.
    text = c.document.get(k).text
    assert row.edit.text() == text
    assert float(text.removeprefix("k = ")) + 1.0 == pytest.approx(seen[-1])
    assert c.modified


def test_slider_steps_rewrite_row_text(window: MainWindow) -> None:
    c = window.controller
    k = first_id(window)
    c.set_text(k, "a = 1")
    y = c.add_expression("y = x + a")
    c.flush()
    row = window.panel.row(k)
    row.move_slider(row.slider_position + 1)
    row.move_slider(row.slider_position + 1)
    assert c.document.get(k).text == "a = 1.2"
    c.flush()
    assert slope_at_1(window, y) == pytest.approx(2.2)
    row.move_slider(0)
    assert c.document.get(k).text == "a = -10"


def test_programmatic_slider_moves_are_not_edits(window: MainWindow) -> None:
    c = window.controller
    c.set_text(first_id(window), "k = 1")
    c.flush()
    row = window.panel.row(first_id(window))
    moved: list[object] = []
    row.slider_moved.connect(lambda *a: moved.append(a))
    from opengraph.document import SliderRange as Range

    row.set_slider(3.0, Range())
    assert moved == [] and c.document.get(first_id(window)).text == "k = 1"


def test_typed_value_outside_range_widens_it(window: MainWindow) -> None:
    c = window.controller
    k = first_id(window)
    c.set_text(k, "k = 1")
    c.flush()
    c.set_text(k, "k = 25")
    c.flush()
    assert c.document.get(k).slider == SliderRange(-10, 25, 0.1)
    assert window.panel.row(k).slider_position == window.panel.row(k).slider_maximum
    assert window.panel.row(k).max_edit.text() == "25"


def test_new_constant_outside_default_range_is_centred(window: MainWindow) -> None:
    c = window.controller
    c.set_text(first_id(window), "k = -40")
    c.flush()
    assert c.document.get(first_id(window)).slider == SliderRange(-50, -30, 0.1)


def edit_bound(window: MainWindow, edit: ExpressionEdit | object, text: str) -> None:
    """Select everything in a slider bound, type ``text`` and press Enter."""
    from opengraph.ui.mathedit.widget import MathEdit

    assert isinstance(edit, MathEdit)
    edit.focus()
    edit.select_all()
    key_clicks(edit, text)
    assert key(window, "Enter")


def test_inline_bounds_edit_range_and_clamp_value(window: MainWindow) -> None:
    c = window.controller
    k = first_id(window)
    c.set_text(k, "k = 1")
    y = c.add_expression("y = k x")
    c.flush()
    row = window.panel.row(k)
    edit_bound(window, row.max_edit, "5")
    assert c.document.get(k).slider == SliderRange(-10, 5, 0.1)
    assert c.document.get(k).text == "k = 1"  # still inside: unchanged
    edit_bound(window, row.min_edit, "2")
    assert c.document.get(k).slider == SliderRange(2, 5, 0.1)
    assert c.document.get(k).text == "k = 2"  # clamped into the new range
    assert slope_at_1(window, y) == 2.0
    assert row.slider_position == 0


def test_bound_edit_commits_when_it_loses_focus(window: MainWindow) -> None:
    c = window.controller
    k = first_id(window)
    c.set_text(k, "k = 1")
    c.flush()
    row = window.panel.row(k)
    row.max_edit.focus()
    row.max_edit.select_all()
    key_clicks(row.max_edit, "4")
    row.edit.focus()  # clicking elsewhere
    assert c.document.get(k).slider == SliderRange(-10, 4, 0.1)


def test_invalid_inline_bounds_are_reverted(window: MainWindow) -> None:
    c = window.controller
    c.set_text(first_id(window), "k = 1")
    c.flush()
    row = window.panel.row(first_id(window))
    messages: list[str] = []
    c.status_message.connect(messages.append)
    for text in ("20", "abc"):
        edit_bound(window, row.min_edit, text)
        assert row.min_edit.text() == "-10"
    assert messages == [
        "Slider minimum must be less than its maximum",
        "Slider minimum: Unknown name 'abc'",
    ]
    assert c.document.get(first_id(window)).slider == SliderRange()


def test_inline_bounds_take_expressions(window: MainWindow) -> None:
    c = window.controller
    k = first_id(window)
    c.set_text(k, "k = 1")
    c.flush()
    row = window.panel.row(k)
    edit_bound(window, row.max_edit, "2pi")
    slider_range = c.document.get(k).slider
    assert slider_range is not None
    assert slider_range.max == pytest.approx(2 * math.pi) and slider_range.max_expr == "2pi"
    assert row.max_edit.text() == "2pi"
    assert c.modified


def test_expression_bounds_follow_other_rows(window: MainWindow) -> None:
    c = window.controller
    k = first_id(window)
    c.set_text(k, "k = 1")
    a = c.add_expression("a = 3")
    c.flush()

    def bounds() -> tuple[float, float]:
        slider_range = c.document.get(k).slider
        assert slider_range is not None
        return slider_range.min, slider_range.max

    c.set_slider_range(k, "-a", "a + 1", "a/30")
    assert c.document.get(k).slider == SliderRange(-3, 4, 0.1, "-a", "a + 1", "a/30")
    c.set_text(a, "a = 5")
    c.flush()
    assert bounds() == (-5, 6)
    assert window.panel.row(k).min_edit.text() == "-a"
    c.set_text(k, "k = 9")  # outside: an expression bound is not widened
    c.flush()
    assert bounds() == (-5, 6)
    row = window.panel.row(k)
    assert row.slider_position == row.slider_maximum
    c.set_text(a, "a = -5")  # an empty range keeps the last good one and says why
    c.flush()
    assert bounds() == (-5, 6)
    assert "less than" in (window.panel.row(k).feedback.value or "")


def test_slider_range_cannot_use_its_own_value(window: MainWindow) -> None:
    c = window.controller
    k = first_id(window)
    c.set_text(k, "k = 1")
    c.flush()
    messages: list[str] = []
    c.status_message.connect(messages.append)
    c.set_slider_range(k, 0, "k + 1", 0.1)
    assert messages == ["Slider maximum: Can't depend on k itself"]
    assert c.document.get(k).slider == SliderRange()


def test_slider_settings_dialog_validates() -> None:
    accepted: list[SliderRange] = []
    dialog = SliderSettingsDialog(SliderRange(), on_accept=accepted.append)
    dialog.step_edit.value = "0"
    dialog.accept()
    assert not dialog.accepted
    assert "positive" in (dialog.error.value or "") and dialog.error.visible
    dialog.step_edit.value = "x"
    dialog.accept()
    assert "Slider step" in (dialog.error.value or "")
    dialog.min_edit.value = "0"
    dialog.max_edit.value = "pi"
    dialog.step_edit.value = "pi/8"
    dialog.accept()
    assert dialog.accepted
    assert accepted == [SliderRange(0, math.pi, math.pi / 8, "", "pi", "pi/8")]


def test_menu_slider_settings_sets_step(window: MainWindow) -> None:
    c = window.controller
    k = first_id(window)
    c.set_text(k, "k = 1")
    c.flush()
    row = window.panel.row(k)
    row._slider_settings()  # Slider Settings… in the row menu
    dialog = row.settings_dialog
    assert dialog.dialog in dialogs.host.shown
    dialog.step_edit.value = "0.5"
    dialog.accept()
    assert dialog.dialog not in dialogs.host.shown
    assert c.document.get(k).slider == SliderRange(-10, 10, 0.5)
    assert row.slider_maximum == 40
    row.move_slider(row.slider_position + 1)
    assert c.document.get(k).text == "k = 1.5"


def test_slider_range_is_saved_and_reopened(tmp_path: Path, window: MainWindow) -> None:
    c = window.controller
    k = first_id(window)
    c.set_text(k, "k = 1")
    c.add_expression("y = k x")
    c.flush()
    c.set_slider_range(k, 0, 4, 0.5)
    path = tmp_path / "s.ograph.json"
    c.save_document(path)
    c.new_document()
    assert window.open_path(path)
    assert not c.modified  # showing the stored slider is not an edit
    item = c.document.items[0]
    assert item.slider == SliderRange(0, 4, 0.5)
    assert window.panel.row(item.id).slider_position == 2


# -- math editor ------------------------------------------------------------------------


def first_edit(win: MainWindow) -> ExpressionEdit:
    edit = win.panel.row(first_id(win)).edit
    edit.focus()
    return edit


def value_at(win: MainWindow, item_id: int, x: float) -> float:
    plot = win.controller.results[item_id].plot
    assert isinstance(plot, ExplicitPlot)
    return plot.fn.scalar(x=x)


def test_typing_is_typeset_as_it_is_entered(window: MainWindow) -> None:
    edit = first_edit(window)
    key_clicks(edit, "y = x^2")
    assert key(window, "Arrow Right")  # only an arrow leaves the exponent
    key_clicks(edit, "+1/2")
    assert edit.text() == "y = x^2+1/2"
    assert edit.model.describe() == "y = x^{2}+\\frac{1}{2|}"
    layout = edit.math_layout()
    exponent = next(n for n in edit.model.root.nodes if isinstance(n, Sup))
    x_rect = layout.nodes[id(edit.model.root.nodes[4])]
    assert layout.nodes[id(exponent)].top < x_rect.top  # raised
    fraction = edit.model.root.nodes[-1]
    assert isinstance(fraction, Frac)
    num, den = layout.geometry(fraction.num), layout.geometry(fraction.den)
    assert num.baseline < 0 < den.baseline  # stacked around the main baseline
    wait_until(lambda: layer_texts(window) == ["y = x^2+1/2"], timeout_ms=2000)
    assert value_at(window, first_id(window), 1.0) == pytest.approx(1.5)


def test_editor_draws_glyphs_at_their_layout_positions(window: MainWindow) -> None:
    import flet.canvas as cv

    edit = first_edit(window)
    key_clicks(edit, "y = x^2")
    shapes = edit.render()
    texts = [s for s in shapes if isinstance(s, cv.Text)]
    assert [t.value for t in texts] == ["y", "=", "x", "2"]
    layout = edit.math_layout()
    for text, (point, value, font) in zip(texts, layout.glyphs, strict=True):
        origin = edit.to_widget(point)
        assert text.value == value and text.x == pytest.approx(origin.x)
        assert text.y == pytest.approx(origin.y - font.ascent())  # baseline placed exactly
        assert text.style is not None and text.style.font_family in fonts.FILES
    caret = [s for s in shapes if isinstance(s, cv.Rect)][-1]
    assert caret.width == 1.5  # the caret is drawn while the editor has the focus


def test_typed_structure_compiles_as_displayed(window: MainWindow) -> None:
    edit = first_edit(window)
    key_clicks(edit, "y = 2^3x")  # the x is typed into the exponent: 2^(3x)
    window.controller.flush()
    assert edit.text() == "y = 2^(3)x"
    assert value_at(window, first_id(window), 1.0) == 8.0
    assert key(window, "Arrow Right")  # out of the exponent
    key_clicks(edit, "+sqrt(x)")
    window.controller.flush()
    assert edit.text() == "y = 2^(3)x+sqrt(x)"
    assert value_at(window, first_id(window), 4.0) == pytest.approx(8 * 4 + 2)


def test_arrow_keys_walk_through_structures(window: MainWindow) -> None:
    edit = first_edit(window)
    key_clicks(edit, "1/23")
    fraction = edit.model.root.nodes[0]
    assert isinstance(fraction, Frac)
    key(window, "Arrow Left")
    assert edit.model.describe() == "\\frac{1}{2|3}"
    key(window, "Arrow Up")
    assert edit.model.cursor.lst is fraction.num
    key(window, "Arrow Down")
    assert edit.model.cursor.lst is fraction.den
    key(window, "Home")
    assert edit.model.describe() == "|\\frac{1}{23}"
    for _ in range(3):
        key(window, "Arrow Right")
    assert edit.model.describe() == "\\frac{1}{|23}"
    key(window, "End")
    assert edit.model.describe() == "\\frac{1}{23}|"


def test_up_down_still_move_between_rows(window: MainWindow) -> None:
    c = window.controller
    c.set_text(first_id(window), "y = 1/2")
    second = c.add_expression("y = x", focus=True)
    edit = window.panel.row(second).edit
    assert edit.has_focus
    key(window, "Arrow Up")
    first = window.panel.row(first_id(window)).edit
    assert first.has_focus and not edit.has_focus
    key(window, "Arrow Down")
    assert edit.has_focus


def test_long_lists_wrap_onto_lines(window: MainWindow) -> None:
    item = "2*atan(sqrt((r - x)/(r + x)))"
    text = f"k = [{item}, -{item}, {item}, -{item}]"
    c = window.controller
    c.set_text(first_id(window), "k = list[1, 2]")
    edit = window.panel.row(first_id(window)).edit
    short_height = edit.content_height()
    c.set_text(first_id(window), text)
    edit.focus()
    assert edit.content_height() > short_height
    assert edit.control.height == edit.content_height()
    assert edit.text() == text  # wrapping only changes the drawing
    layout = edit.math_layout()
    bracket = edit.model.root.nodes[-1]
    assert isinstance(bracket, Paren)
    geo = layout.geometry(bracket.body)
    assert len(set(geo.ys)) > 1  # stacked lines
    one_line = Typesetter(edit.size).layout(edit.model.root)
    assert layout.width < one_line.width / 2
    # Up/Down move between the lines before leaving the row.
    edit.model.set_cursor(Position(bracket.body, len(bracket.body.nodes)))
    last_line = geo.ys[-1]
    key(window, "Arrow Up")
    assert edit.model.cursor.lst is bracket.body
    assert geo.ys[edit.model.cursor.index] < last_line
    key(window, "Arrow Down")
    assert geo.ys[edit.model.cursor.index] == last_line
    # Clicking a lower line puts the caret on it.
    x, top, bottom = layout.caret_line(Position(bracket.body, 2))
    edit.click(edit.to_widget(Point(x, (top + bottom) / 2)))
    assert edit.model.cursor == Position(bracket.body, 2)


def test_selection_deletion_and_insertion(window: MainWindow) -> None:
    edit = first_edit(window)
    key_clicks(edit, "y = x+1/2")
    key(window, "Arrow Left", shift=True)
    assert edit.selected_text() == "2"
    key(window, "Arrow Left", shift=True)
    assert edit.selected_text() == "1/2"  # the selection grew to the whole fraction
    key_clicks(edit, "3")
    assert edit.text() == "y = x+3"
    key(window, "Backspace")
    key(window, "Backspace")
    assert edit.text() == "y = x"
    window.controller.flush()
    assert layer_texts(window) == ["y = x"]


def test_backspace_steps_into_and_removes_exponent(window: MainWindow) -> None:
    edit = first_edit(window)
    key_clicks(edit, "x^2")
    key(window, "Arrow Right")
    seen: list[str] = []
    edit.text_edited.connect(seen.append)
    key(window, "Backspace")  # into the exponent: no edit
    assert seen == []
    key(window, "Backspace")
    key(window, "Backspace")
    assert edit.text() == "x" and seen == ["x^()", "x"]


def test_mouse_click_places_caret_inside_fraction(window: MainWindow) -> None:
    c = window.controller
    c.set_text(first_id(window), "y = (x+1)/2")
    edit = window.panel.row(first_id(window)).edit
    fraction = edit.model.root.nodes[-1]
    assert isinstance(fraction, Frac)
    geo = edit.math_layout().geometry(fraction.num)
    # Between "x" and "+" in the numerator.
    edit.click(edit.to_widget(Point(geo.xs[1], geo.baseline - 2)))
    assert edit.has_focus
    assert edit.model.cursor == Position(fraction.num, 1)
    key_clicks(edit, "^2")
    assert edit.text() == "y = (x^2+1)/2"
    assert c.document.get(first_id(window)).text == "y = (x^2+1)/2"


def test_mouse_drag_and_double_click_select(window: MainWindow) -> None:
    c = window.controller
    c.set_text(first_id(window), "y = 25 + x")
    edit = window.panel.row(first_id(window)).edit
    geo = edit.math_layout().geometry(edit.model.root)
    start = edit.to_widget(Point(geo.xs[4], geo.baseline - 3))
    end = edit.to_widget(Point(geo.xs[6], geo.baseline - 3))
    edit.press(start, shift=False)
    edit.drag_to(end)
    edit.release()
    assert edit.selected_text() == "25"
    edit.click(end)
    assert not edit.has_selected_text()
    edit.double_click(start)
    assert edit.selected_text() == "25"


def test_clipboard_and_undo_shortcuts(window: MainWindow) -> None:
    c = window.controller
    edit = first_edit(window)
    key_clicks(edit, "y = sqrt(x)/2")
    key(window, "A", ctrl=True)
    key(window, "C", ctrl=True)
    assert runtime.clipboard.text == "y = (sqrt(x))/2"
    other = c.add_expression(focus=True)
    target = window.panel.row(other).edit
    assert target.has_focus
    key(window, "V", ctrl=True)
    assert target.text() == "y = (sqrt(x))/2"
    assert c.document.get(other).text == target.text()
    key(window, "Z", ctrl=True)
    assert target.text() == "" and c.document.get(other).text == ""
    key(window, "Y", ctrl=True)
    assert target.text() == "y = (sqrt(x))/2"
    key(window, "A", ctrl=True)
    key(window, "X", ctrl=True)
    assert target.text() == "" and runtime.clipboard.text == "y = (sqrt(x))/2"
    c.flush()
    assert layer_texts(window) == ["y = (sqrt(x))/2"]


def test_caret_movement_does_not_recompile(window: MainWindow) -> None:
    c = window.controller
    c.set_text(first_id(window), "y = 1/x")
    c.flush()
    edit = first_edit(window)
    seen: list[str] = []
    edit.text_edited.connect(seen.append)
    for name in ("Arrow Left", "Arrow Up", "Home", "End"):
        key(window, name)
    key(window, "Arrow Left", shift=True)
    assert seen == [] and not c._compile_timer.is_active()
    key(window, "End")
    key(window, "Arrow Left")  # into the denominator, after the x
    key(window, "Arrow Left")
    key_clicks(edit, "2")
    assert seen == ["y = 1/(2x)"] and c._compile_timer.is_active()


def test_loaded_text_is_kept_until_edited(window: MainWindow, tmp_path: Path) -> None:
    c = window.controller
    c.set_text(first_id(window), "y=1/2x")
    path = tmp_path / "keep.ograph.json"
    c.save_document(path)
    c.new_document()
    assert window.open_path(path)
    assert not c.modified
    item = c.document.items[0]
    edit = window.panel.row(item.id).edit
    assert edit.text() == "y=1/2x" and item.text == "y=1/2x"
    assert edit.model.describe() == "y=\\frac{1}{2}x|"  # shown as the parser reads it
    edit.focus()
    key_clicks(edit, "+1")
    assert item.text == "y=(1/2)x+1"
    c.flush()
    assert value_at(window, item.id, 4.0) == 3.0


def test_errors_are_underlined_under_the_right_part(window: MainWindow) -> None:
    import flet.canvas as cv

    c = window.controller
    c.set_text(first_id(window), "y = q + x^2")
    c.flush()
    edit = window.panel.row(first_id(window)).edit
    q = edit.model.root.nodes[4]
    assert edit.error_rects() == [edit.math_layout().nodes[id(q)]]
    wave = [
        s for s in edit.render() if isinstance(s, cv.Path) and s.paint.color == edit.error_color
    ]
    assert len(wave) == 1  # drawn under the q
    c.set_text(first_id(window), "y = (x+1)/()")
    c.flush()
    assert edit.error_rects()  # the empty denominator is flagged
    edit.render()  # drawing with an error span works


def test_row_height_follows_the_notation(window: MainWindow) -> None:
    edit = first_edit(window)
    flat = edit.content_height()
    key_clicks(edit, "1/2")
    assert edit.content_height() > flat
    assert edit.control.height == edit.content_height()
    edit.size = fonts.EDITOR_SIZE * 2
    edit.set_text(edit.text())
    assert edit.content_height() > flat * 1.5


def test_long_expressions_scroll_to_the_caret(window: MainWindow) -> None:
    edit = first_edit(window)
    key_clicks(edit, "y = " + " + ".join(f"{n}x" for n in range(1, 40)))
    assert edit.math_layout().width > edit.width
    caret = edit.caret_rect()
    assert 0 <= caret.center.x <= edit.width
    key(window, "Home")
    caret = edit.caret_rect()
    assert 0 <= caret.center.x <= edit.width


def test_incomplete_input_never_breaks_editing(window: MainWindow) -> None:
    edit = first_edit(window)
    for keys in ("y = (", "/", "^", "_", ")", "sqrt", "$", "<", "=", ",", "|"):
        key_clicks(edit, keys)
        edit.render()
    window.controller.flush()
    assert window.panel.row(first_id(window)).error is True
    key(window, "A", ctrl=True)
    key(window, "Delete")
    assert edit.text() == ""


def test_symbolic_integral_adds_row(window: MainWindow) -> None:
    pytest.importorskip("sympy")
    c = window.controller
    c.set_text(first_id(window), "y = 3x^2")
    c.run_symbolic(first_id(window), "integral")
    wait_until(lambda: len(c.document) == 2, timeout_ms=30_000)
    assert c.document.items[1].text == "y = x^3"


def test_symbolic_job_locks_until_done(window: MainWindow) -> None:
    pytest.importorskip("sympy")
    c = window.controller
    row_id = first_id(window)
    c.set_text(row_id, "x^2 - 4")
    c.run_symbolic(row_id, "solve")
    assert c.busy and window.progress.visible
    assert all(a.disabled for a in c.panel.row(row_id).symbolic_actions)
    assert not window._locked_actions[0].is_enabled()  # File ▸ New waits
    c.run_symbolic(row_id, "derivative")  # refused while busy
    wait_until(lambda: not c.busy, timeout_ms=30_000)
    assert [item.text for item in c.document.items] == ["x^2 - 4", "point[list[-2, 2], 0]"]
    assert not window.progress.visible
    assert not any(a.disabled for a in c.panel.row(row_id).symbolic_actions)
    assert window._locked_actions[0].is_enabled()


def test_computer_algebra_rows_are_computed_in_the_background(window: MainWindow) -> None:
    pytest.importorskip("sympy")
    c = window.controller
    c.set_text(first_id(window), "a = 2")
    c.add_expression("g(x) = diff(a x^3)")
    c.add_expression("g(1)")
    c.flush()
    ids = [item.id for item in c.document.items]
    assert c.results[ids[1]].message == "Computing the derivative…"
    wait_until(lambda: c.results[ids[1]].message == "= 3*a*x^2", timeout_ms=30_000)
    assert c.results[ids[2]].message == "= 6"
    assert any(t.operation == "cas" for t in c.tasks.tasks())
    c.set_text(ids[0], "a = 3")  # a slider move: the formula still holds
    c.flush()
    assert c.results[ids[2]].message == "= 9"


def test_symbolic_algebra_background_result_can_be_bound_and_reused(window: MainWindow) -> None:
    c = window.controller
    row_id = first_id(window)
    c.set_text(row_id, "collect(a*x + b*x, x)")
    c.flush()
    assert c.results[row_id].message == "Computing the collected terms…"
    wait_until(lambda: c.results[row_id].message == "= x*(a + b)", timeout_ms=30_000)
    assert c.results[row_id].plot is None and not c.results[row_id].is_error
    c.add_expression("a = 2")
    c.add_expression("b = 3")
    c.flush()
    result = c.results[row_id]
    assert not result.pending and isinstance(result.plot, ExplicitPlot)
    assert result.plot.evaluate(2.0) == pytest.approx(10)
    assert len([task for task in c.tasks.tasks() if task.operation == "cas"]) == 1
