"""Logarithmic axes, independent axis scaling, log ticks and graph settings."""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from opengraph.document import Document
from opengraph.engine.compiler import compile_expression
from opengraph.engine.plots import ExplicitPlot, ImplicitPlot
from opengraph.engine.sampling import ViewRect, from_axis, to_axis
from opengraph.graph.settings import AxisSettings, GraphSettings
from opengraph.graph.ticks import format_log_tick, log_ticks
from opengraph.graph.viewport import Viewport


def test_axis_transform_round_trip_and_domain() -> None:
    values = np.array([0.001, 1.0, 250.0])
    np.testing.assert_allclose(from_axis(to_axis(values, True), True), values)
    assert np.isnan(to_axis(np.array([0.0, -1.0]), True)).all()  # no position on a log axis
    assert to_axis(5.0, False) == 5.0
    assert math.isnan(to_axis(-2.0, True))


def test_log_viewport_round_trip() -> None:
    vp = Viewport(800, 600)
    vp.set_log("x", True)
    for x in (0.01, 1.0, 42.0):
        assert vp.to_world_x(vp.to_screen_x(x)) == pytest.approx(x)
    # Equal ratios are equal distances on a log axis.
    assert vp.to_screen_x(10.0) - vp.to_screen_x(1.0) == pytest.approx(
        vp.to_screen_x(100.0) - vp.to_screen_x(10.0)
    )
    assert not math.isfinite(vp.to_screen_x(-1.0))


def test_set_log_keeps_the_positive_part_of_the_range() -> None:
    vp = Viewport(800, 600)
    assert vp.set_range("x", 2.0, 200.0)
    vp.set_log("x", True)
    assert vp.range("x") == pytest.approx((2.0, 200.0))
    vp.set_log("x", False)
    assert vp.range("x") == pytest.approx((2.0, 200.0))
    vp.set_range("y", -10.0, 10.0)
    vp.set_log("y", True)  # only the positive part can be shown
    lo, hi = vp.range("y")
    assert 0 < lo < hi == pytest.approx(10.0)
    vp.set_log("y", False)
    vp.set_range("y", -10.0, -1.0)
    vp.set_log("y", True)  # nothing positive: a default decade range
    assert vp.range("y") == pytest.approx((0.01, 100.0))


def test_set_range_validation_and_independent_scales() -> None:
    vp = Viewport(800, 600)
    assert vp.set_range("x", -1.0, 1.0)
    assert vp.set_range("y", 0.0, 1000.0)
    assert vp.range("x") == pytest.approx((-1.0, 1.0))
    assert vp.range("y") == pytest.approx((0.0, 1000.0))
    assert vp.scale_x != pytest.approx(vp.scale_y)  # axes scaled independently
    assert not vp.set_range("x", 1.0, 1.0)
    assert not vp.set_range("x", 2.0, 1.0)
    assert not vp.set_range("x", -math.inf, 1.0)
    vp.set_log("x", True)
    assert not vp.set_range("x", -1.0, 10.0)  # log axes need positive bounds


def test_zoom_axes_independently_keeps_point_under_cursor() -> None:
    vp = Viewport(800, 600)
    world = vp.to_world(200.0, 150.0)
    scale_y = vp.scale_y
    vp.zoom_axes_at(200.0, 150.0, 2.0, 1.0)
    assert vp.scale_y == scale_y
    assert vp.to_screen(*world) == pytest.approx((200.0, 150.0))
    vp.set_log("y", True)
    world = vp.to_world(300.0, 100.0)
    vp.zoom_at(300.0, 100.0, 1.7)
    assert vp.to_screen(*world) == pytest.approx((300.0, 100.0))


def test_log_ticks() -> None:
    majors, minors = log_ticks(-1.0, 2.0, 200.0)  # 0.1 .. 100 at 200 px per decade
    assert majors == pytest.approx([0.1, 1.0, 10.0, 100.0])
    assert 2.0 in minors and 50.0 in minors
    majors, _ = log_ticks(-10.0, 10.0, 10.0)  # crowded: every n-th decade
    assert len(majors) < 21 and 1.0 in majors
    wide, _ = log_ticks(0.0, 1.0, 600.0)  # a wide decade labels 2 and 5 too
    assert wide == pytest.approx([1.0, 2.0, 5.0, 10.0])
    assert log_ticks(1.0, 1.0, 100.0) == ([], [])
    assert format_log_tick(1e6) == "10⁶"
    assert format_log_tick(1e-5) == "10⁻⁵"
    assert format_log_tick(0.02) == "0.02"
    assert format_log_tick(500.0) == "500"


def test_explicit_sampling_on_log_axes() -> None:
    plot = compile_expression("y = x^2").plot
    assert isinstance(plot, ExplicitPlot)
    view = ViewRect(0.01, 100.0, 1e-4, 1e4, 800, 600, x_log=True, y_log=True)
    geometry = plot.sample(view)
    points = np.vstack(geometry.polylines)
    assert points[:, 0].min() < 0.02 and points[:, 0].max() > 50
    np.testing.assert_allclose(points[:, 1], points[:, 0] ** 2, rtol=1e-12)
    # y = x^2 is a straight line on log-log axes: few vertices are needed.
    assert len(points) < 2000
    # Values that are negative or zero have no place on a log axis and are skipped.
    negative = compile_expression("y = -x").plot
    assert isinstance(negative, ExplicitPlot)
    assert negative.sample(view).is_empty


def test_implicit_sampling_on_log_axes() -> None:
    plot = compile_expression("x y = 1").plot
    assert isinstance(plot, ImplicitPlot)
    view = ViewRect(0.1, 10.0, 0.1, 10.0, 400, 400, x_log=True, y_log=True)
    segments = plot.sample(view).segments
    assert segments is not None and len(segments)
    np.testing.assert_allclose(segments[:, 0] * segments[:, 1], 1.0, rtol=0.05)


def test_view_rect_padding_in_axis_units() -> None:
    view = ViewRect(1.0, 100.0, 0.0, 1.0, 400, 400, x_log=True)
    padded = view.padded(0.5)
    assert padded.x_min == pytest.approx(0.1) and padded.x_max == pytest.approx(1000.0)
    assert padded.contains(view) and not view.contains(padded)
    assert not padded.contains(ViewRect(1.0, 100.0, 0.0, 1.0, 400, 400))  # other axis type


def test_settings_normalisation() -> None:
    logs = GraphSettings(x_axis=AxisSettings(log=True), y_axis=AxisSettings(log=True))
    polar = replace_settings(logs, polar_grid=True).normalized()
    assert not polar.any_log and polar.lock_aspect
    complex_ = replace_settings(logs, complex_mode=True).normalized()
    assert not complex_.any_log
    assert logs.normalized() == logs
    options = GraphSettings(degrees=True, complex_mode=True, polar_grid=True).compile_options()
    assert options.degrees and options.complex_mode and not options.polar


def replace_settings(settings: GraphSettings, **changes: bool) -> GraphSettings:
    return replace(settings, **changes)  # type: ignore[arg-type]


def test_settings_round_trip_and_lenient_parsing() -> None:
    settings = GraphSettings(
        x_axis=AxisSettings(log=True, step=0.5),
        y_axis=AxisSettings(step=None),
        minor_grid=False,
        axis_numbers=False,
        degrees=True,
        points_of_interest=False,
    )
    assert GraphSettings.from_dict(settings.to_dict()) == settings
    assert GraphSettings.from_dict(None) == GraphSettings()
    assert GraphSettings.from_dict({"degrees": "yes", "x_axis": {"step": -3}}) == GraphSettings()
    assert GraphSettings.from_dict({"x_axis": {"step": True}}).x_axis.step is None


def test_typed_toggle_preserves_settings_and_normalizes_polar_axes() -> None:
    original = GraphSettings(x_axis=AxisSettings(log=True, step=2.0), minor_grid=False)
    degrees = original.with_toggle("degrees", True)
    assert degrees == replace(original, degrees=True)
    assert not original.degrees
    polar = original.with_toggle("polar_grid", True)
    assert polar.polar_grid and polar.lock_aspect and not polar.any_log
    assert polar.x_axis.step == 2.0 and not polar.minor_grid


def test_document_persists_settings(tmp_path: Path) -> None:
    doc = Document()
    doc.add("y = x")
    doc.settings = GraphSettings(polar_grid=True, degrees=True).normalized()
    path = tmp_path / "graph.ograph.json"
    doc.save(path)
    loaded = Document.load(path)
    assert loaded.settings == doc.settings
    # Files from before settings existed open with the defaults.
    data = doc.to_dict()
    del data["settings"]
    assert Document.from_dict(data).settings == GraphSettings()
