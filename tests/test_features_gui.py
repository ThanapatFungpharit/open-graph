"""Integration tests of the features through the real interface objects."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from opengraph import colors
from opengraph.document import PALETTE, Document, TableData
from opengraph.engine.plots import ParametricPlot, PlotGroup, PointSetPlot, PolarPlot
from opengraph.graph.canvas import LIGHT_COLORS
from opengraph.graph.settings import AxisSettings, GraphSettings
from opengraph.preferences import Preferences
from opengraph.ui import theme
from opengraph.ui.expression_panel import read_csv
from opengraph.ui.geometry import Point
from opengraph.ui.main_window import MainWindow, syntax_reference
from tests.support import first_id, make_window, wait_until


def compile_now(win: MainWindow) -> None:
    win.controller.flush()
    win.controller.recompile()


# -- themes ------------------------------------------------------------------------------


def test_theme_switch_applies_everywhere_and_persists(window: MainWindow) -> None:
    window.set_theme_mode("dark")
    assert theme.current() is theme.DARK
    assert window.canvas.colors == theme.DARK.canvas
    assert window.canvas.control.bgcolor == theme.DARK.canvas.background
    assert window.panel.control.bgcolor == theme.DARK.panel
    assert window.status_bar.bgcolor == theme.DARK.window
    assert Preferences().get_str("theme", "") == "dark"
    other = make_window()  # a new window starts in the saved theme
    assert other.theme_mode == "dark" and other._theme_actions["dark"].is_checked()
    other.shutdown()
    window.set_theme_mode("light")
    assert window.canvas.colors == LIGHT_COLORS


def test_system_theme_follows_the_os(window: MainWindow) -> None:
    window.on_brightness_changed(True)
    window.set_theme_mode("system")
    assert theme.current() is theme.DARK
    window.on_brightness_changed(False)
    assert theme.current() is theme.LIGHT


@pytest.mark.parametrize("current", [theme.LIGHT, theme.DARK])
def test_curve_colors_contrast_with_the_canvas(current: theme.Theme) -> None:
    for color in (*PALETTE, "#000000", "#ffffff", "#ffff00"):
        shown = current.curve_color(color)
        assert colors.contrast(shown, current.canvas.background) >= theme.MIN_CONTRAST - 0.05


def test_layers_use_theme_adjusted_colors(window: MainWindow) -> None:
    c = window.controller
    c.set_text(first_id(window), "y = x")
    c.document.get(first_id(window)).color = "#475569"  # slate: too dark on the dark canvas
    compile_now(window)
    window.set_theme_mode("dark")
    shown = window.canvas._layers[0].color
    assert colors.contrast(shown, theme.DARK.canvas.background) >= theme.MIN_CONTRAST - 0.05
    assert c.document.get(first_id(window)).color == "#475569"  # the document is unchanged


# -- graph settings ------------------------------------------------------------------------


def test_settings_panel_applies_changes_immediately(window: MainWindow) -> None:
    c = window.controller
    c.set_text(first_id(window), "sin(90)")
    compile_now(window)
    assert c.results[first_id(window)].message != "= 1"
    window.show_graph_settings()
    dialog = window.settings_dialog
    assert dialog.is_visible()
    dialog.change(dialog.angles, "degrees")
    assert c.settings.degrees and window._toggles["degrees"].is_checked()
    assert c.results[first_id(window)].message == "= 1"
    assert c.modified
    dialog.change(dialog.scales["x"], "log")
    assert c.settings.x_axis.log and window.canvas.viewport.x_log
    dialog.change(dialog.grid_kind, "polar")  # a polar grid needs linear axes
    assert not c.settings.x_axis.log and not window.canvas.viewport.x_log
    assert dialog.scales["x"].disabled and dialog.note.visible
    dialog.change(dialog.minor_grid, False)
    assert not window.canvas.settings.minor_grid


def test_settings_panel_bounds_and_step(window: MainWindow) -> None:
    window.show_graph_settings()
    dialog = window.settings_dialog
    dialog.edit_bounds("x", "-1", "3")
    assert window.canvas.viewport.range("x") == pytest.approx((-1.0, 3.0))
    dialog.edit_step("y", "0.25")
    assert window.controller.settings.y_axis.step == 0.25
    dialog.edit_bounds("x", "5", "3")  # min > max: rejected with a message, view unchanged
    assert window.canvas.viewport.range("x") == pytest.approx((-1.0, 3.0))
    assert "minimum must be less" in window.current_message()
    window.canvas.zoom(2.0)  # the fields follow the view
    assert float(dialog.bounds["x"][0].value or "nan") == pytest.approx(
        window.canvas.viewport.x_min, rel=1e-5
    )


def test_quick_toggles_and_polar_rows(window: MainWindow) -> None:
    c = window.controller
    c.set_text(first_id(window), "r = 2 + 0theta")
    compile_now(window)
    assert isinstance(c.results[first_id(window)].plot, PolarPlot)  # a circle on every grid
    window._toggles["polar_grid"].trigger()
    assert c.settings.polar_grid
    assert isinstance(c.results[first_id(window)].plot, PolarPlot)  # a circle on the polar grid
    window.canvas.viewport.reset()
    assert window.canvas.format_position(0.0, 2.0).startswith("r = 2")
    window._toggles["polar_grid"].trigger()
    assert isinstance(c.results[first_id(window)].plot, PolarPlot)


def test_complex_mode_coordinates(window: MainWindow) -> None:
    c = window.controller
    c.set_settings(replace(c.settings, complex_mode=True))
    assert window.canvas.format_position(1.0, 2.0, exact=True) == "1 + 2i"
    assert window.canvas.format_position(3.0, -1.0, exact=True) == "3 − i"
    assert window.canvas._tick_label(2.0, 1.0, "y") == "2i"
    assert window.canvas._tick_label(2.0, 1.0, "x") == "2"


def test_complex_interval_spiral_reaches_canvas_when_mode_is_enabled(window: MainWindow) -> None:
    c = window.controller
    item = first_id(window)
    c.set_text(item, "e^(t(ln(sqrt(a^2+b^2))+i*atan(b,a)))")
    for text in ("t=[0,100]", "a=1", "b=1/2"):
        c.add_expression(text)
    compile_now(window)
    assert c.results[item].is_error
    window._toggles["complex_mode"].trigger()
    compile_now(window)
    assert not c.results[item].is_error
    assert isinstance(c.results[item].plot, ParametricPlot)
    assert any(layer.key == item for layer in window.canvas._layers)
    wait_until(lambda: not window.canvas.is_rendering, timeout_ms=30_000)
    assert any(
        tile.request.layer == item and tile.geometry.polylines
        for tile in window.canvas._geometry.tiles
    )
    window._toggles["complex_mode"].trigger()
    compile_now(window)
    assert c.results[item].is_error
    assert not any(layer.key == item for layer in window.canvas._layers)


def test_independent_axis_zoom(window: MainWindow) -> None:
    vp = window.canvas.viewport
    center = Point(vp.width / 2, vp.height / 2)
    sx, sy = vp.scale_x, vp.scale_y
    window.canvas.wheel(center, 120, ctrl=True)
    assert vp.scale_x > sx and vp.scale_y == sy  # x only
    sx = vp.scale_x
    window.canvas.wheel(center, 120, shift=True)
    assert vp.scale_y > sy and vp.scale_x == sx  # y only
    c = window.controller
    c.set_settings(replace(c.settings, lock_aspect=True))
    ratio = vp.scale_y / vp.scale_x
    window.canvas.wheel(center, 120, ctrl=True)
    assert vp.scale_y / vp.scale_x == pytest.approx(ratio)  # locked: both axes


def test_settings_survive_save_and_open(tmp_path: Path, window: MainWindow) -> None:
    c = window.controller
    settings = GraphSettings(
        x_axis=AxisSettings(log=True), y_axis=AxisSettings(step=2.0), degrees=True
    )
    c.set_settings(settings)
    window.canvas.set_range("x", 0.5, 50.0)
    path = tmp_path / "graph.ograph.json"
    c.save_document(path)
    c.new_document()
    assert c.settings == GraphSettings() and not window.canvas.viewport.x_log
    assert window.open_path(path)
    assert c.settings == settings.normalized()
    assert window.canvas.viewport.x_log
    assert window.canvas.viewport.range("x") == pytest.approx((0.5, 50.0))
    assert window._toggles["degrees"].is_checked()


# -- tables ------------------------------------------------------------------------------------


def test_table_editing_plots_points(window: MainWindow) -> None:
    c = window.controller
    table_id = c.add_table(focus=True)
    row = window.panel.table(table_id)
    assert [e.value for e in row.name_edits] == ["x_1", "y_1"]
    for r, (x, y) in enumerate([("1", "2"), ("2", "4"), ("3", "8")]):
        row.set_cell_text(r, 0, x)
        row.set_cell_text(r, 1, y)
    assert row.row_count == 4  # always one spare row
    compile_now(window)
    layer = next(layer for layer in window.canvas._layers if layer.key == table_id)
    assert layer.plot == PointSetPlot(((1.0, 2.0), (2.0, 4.0), (3.0, 8.0)), True, False)
    row.set_style("both")
    compile_now(window)
    assert c.results[table_id].plot == PointSetPlot(
        ((1.0, 2.0), (2.0, 4.0), (3.0, 8.0)), True, True
    )
    assert row.style_actions["both"].checked and not row.style_actions["points"].checked
    # Columns are lists usable in expressions.
    c.add_expression("mean(y_1)")
    compile_now(window)
    assert c.results[c.document.items[-1].id].message == "= 4.666666667"


def test_table_cell_errors_are_marked(window: MainWindow) -> None:
    c = window.controller
    table_id = c.add_table()
    row = window.panel.table(table_id)
    row.set_cell_text(0, 0, "1")
    row.set_cell_text(0, 1, "2*")
    compile_now(window)
    assert c.results[table_id].is_error
    assert row.cell(0, 1).tooltip
    assert row.feedback.visible and "Row 1, column 2" in (row.feedback.value or "")


def test_table_columns_and_names(window: MainWindow) -> None:
    c = window.controller
    first = c.add_table()
    second = c.add_table()
    assert c.document.get(second).table is not None
    assert c.document.get(second).table.names == ["x_2", "y_2"]  # type: ignore[union-attr]
    row = window.panel.table(first)
    row.add_column()
    assert row.column_count == 3 and row.data.names[-1] == "y_3"
    for r, values in enumerate([("0", "1", "2"), ("1", "3", "5")]):
        for col, text in enumerate(values):
            row.set_cell_text(r, col, text)
    compile_now(window)
    assert isinstance(c.results[first].plot, PlotGroup)  # one data set per y column
    row.set_name(1, "v")
    c.add_expression("total(v)")
    compile_now(window)
    assert c.results[c.document.items[-1].id].message == "= 4"


def test_tables_save_and_reopen(tmp_path: Path, window: MainWindow) -> None:
    c = window.controller
    c.add_table(TableData(["t", "h"], [["0", "1"], ["5", "3"]], "lines"))
    path = tmp_path / "data.ograph.json"
    c.save_document(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["version"] == 2  # older builds refuse tables rather than drop them
    assert window.open_path(path)
    tables = c.document.tables()
    assert len(tables) == 1 and tables[0][1] == TableData(
        ["t", "h"], [["0", "1"], ["5", "3"]], "lines"
    )
    plot = c.results[tables[0][0]].plot
    assert plot == PointSetPlot(((0.0, 5.0), (1.0, 3.0)), False, True)
    # Documents without tables keep the old version number.
    plain = Document()
    plain.add("y = x")
    assert plain.to_dict()["version"] == 1


def test_csv_import(tmp_path: Path) -> None:
    path = tmp_path / "data.csv"
    path.write_text("time,height\n0,1.5\n1,2.5\n2,4\n", encoding="utf-8")
    data = read_csv(path, TableData())
    assert data.names == ["time", "height"]
    assert data.columns == [["0", "1", "2"], ["1.5", "2.5", "4"]]
    headerless = tmp_path / "plain.csv"
    headerless.write_text("1;2;3\n4;5;6\n", encoding="utf-8")
    data = read_csv(headerless, TableData())
    assert data.names == ["x_1", "y_1", "y_2"] and data.columns[2] == ["3", "6"]
    empty = tmp_path / "empty.csv"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        read_csv(empty, TableData())


def test_table_imports_csv_through_the_row(tmp_path: Path, window: MainWindow) -> None:
    c = window.controller
    table_id = c.add_table()
    path = tmp_path / "data.csv"
    path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    window.panel.table(table_id).import_csv(path)
    compile_now(window)
    assert c.document.get(table_id).table == TableData(["a", "b"], [["1", "3"], ["2", "4"]])


# -- points of interest ----------------------------------------------------------------------


def test_points_of_interest_for_the_focused_row(window: MainWindow) -> None:
    c = window.controller
    curve = first_id(window)
    c.set_text(curve, "y = x^2 - 1")
    line = c.add_expression("y = 1 - x")
    compile_now(window)
    window.panel.focus_row(curve)
    assert window.canvas._focus == curve
    found = {(p.kind, round(p.x, 6), round(p.y, 6)) for p in window.canvas.markers()}
    assert ("x-intercept", -1.0, 0.0) in found and ("x-intercept", 1.0, 0.0) in found
    assert ("minimum", 0.0, -1.0) in found and ("y-intercept", 0.0, -1.0) in found
    assert ("intersection", -2.0, 3.0) in found and ("intersection", 1.0, 0.0) in found
    # Hover shows the label, a click pins it.
    vp = window.canvas.viewport
    sx, sy = vp.to_screen(0.0, -1.0)
    pos = Point(float(sx), float(sy))
    trace = window.canvas._find_trace(pos)
    assert trace is not None and trace.label.startswith(("Local minimum", "y-intercept"))
    assert window.canvas.toggle_pin_at(pos)
    assert window.canvas._pinned
    # Points of interest can be turned off; another row's focus shows its own points.
    c.set_settings(replace(c.settings, points_of_interest=False))
    assert window.canvas.markers() == []
    c.set_settings(replace(c.settings, points_of_interest=True))
    window.panel.focus_row(line)
    assert window.canvas._focus == line
    assert not window.canvas._pinned
    assert any(p.kind == "x-intercept" and round(p.x, 6) == 1 for p in window.canvas.markers())


def test_exact_value_shows_in_the_row(window: MainWindow) -> None:
    pytest.importorskip("sympy")
    c = window.controller
    row_id = first_id(window)
    c.set_text(row_id, "∫(x^2, 0, 1)")
    compile_now(window)
    row = window.panel.row(row_id)
    assert not row.exact_action.disabled
    assert all(a.disabled for a in row.symbolic_actions)  # no curve to operate on
    c.run_symbolic(row_id, "exact")
    wait_until(lambda: "exact: 1/3" in (row.feedback.value or ""), timeout_ms=30_000)
    assert (row.feedback.value or "").startswith("= 0.3333333333")
    c.set_text(row_id, "∫(x^3, 0, 1)")
    compile_now(window)
    assert "exact" not in (row.feedback.value or "")  # stale once the text changes


def test_reference_documents_the_new_syntax() -> None:
    text = syntax_reference()
    for needle in (
        "sum(k^2 for k from 1 to 10)",
        "sum(body, k, a, b)",
        "int(body, t, a, b)",
        "limit(f for t to a)",
        "Gamma",
        "list[1...10]",
        "Tables",
        "complex",
    ):
        assert needle in text


def test_markers_refresh_after_zooming(window: MainWindow) -> None:
    c = window.controller
    curve = first_id(window)
    c.set_text(curve, "y = x^2 - 1")
    compile_now(window)
    window.canvas.set_focus(curve)
    assert window.canvas.markers()  # computed on request
    window.canvas.zoom(1.5)
    # While the view changes, stale points are hidden instead of recomputed per frame...
    assert window.canvas._current_markers() == []
    # ...and they come back once the view has been still for a moment.
    wait_until(lambda: bool(window.canvas._current_markers()), timeout_ms=2000)
    window.canvas.viewport.pan_pixels(40, 0)  # panning within the analysed area is free
    assert window.canvas._current_markers()


def test_formula_column_shows_computed_read_only_cells(window: MainWindow) -> None:
    c = window.controller
    table_id = c.add_table(focus=True)
    row = window.panel.table(table_id)
    for r, x in enumerate(["1", "2", "3"]):
        row.set_cell_text(r, 0, x)
    row.set_name(1, "y_1 = x_1^2")
    compile_now(window)
    assert [row.cell(r, 1).value for r in range(3)] == ["1", "4", "9"]
    assert row.cell(0, 1).read_only
    assert c.results[table_id].plot == PointSetPlot(
        ((1.0, 1.0), (2.0, 4.0), (3.0, 9.0)), True, False
    )
    row.set_name(1, "y_1")  # back to a data column: cells editable again
    assert not row.cell(0, 1).read_only
