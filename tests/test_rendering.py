"""Lazy geometry loading through the event loop, with deterministic worker barriers."""

from __future__ import annotations

import threading
from dataclasses import replace
from pathlib import Path

import flet.canvas as cv
import numpy as np
import pytest

from opengraph import events
from opengraph.engine.cancellation import Cancelled, cancellable, checkpoint
from opengraph.engine.compiler import compile_expression
from opengraph.engine.plots import ExplicitPlot, Geometry, ImplicitPlot, Plot
from opengraph.engine.sampling import ViewRect, marching_squares
from opengraph.graph.background import LatestTask
from opengraph.graph.canvas import GraphCanvas, Layer, png_bytes
from opengraph.graph.rendering import (
    TILE_PX,
    GeometryLoader,
    _simplify,
    outline,
    tile_requests,
    visible_geometry,
)
from opengraph.graph.settings import GraphSettings
from opengraph.ui.main_window import MainWindow
from tests.support import drawn_near, shapes_in_color, wait_until

VIEW = ViewRect(-8, 8, -6, 6, 800, 600)
PLAIN = GraphSettings(major_grid=False, minor_grid=False, axes=False, axis_numbers=False)


def plot(text: str) -> Plot:
    result = compile_expression(text)
    assert result.plot is not None, result.message
    return result.plot


def load(loader: GeometryLoader, expression: Plot, view: ViewRect = VIEW) -> None:
    loader.request([(1, expression)], view)
    wait_until(lambda: not loader.loading, timeout_ms=10_000)


def shown_canvas() -> GraphCanvas:
    canvas = GraphCanvas()
    canvas.resize(800, 600)
    return canvas


def test_worker_sampling_does_not_block_drawing_or_timers(monkeypatch: pytest.MonkeyPatch) -> None:
    entered, release = threading.Event(), threading.Event()
    threads: list[int] = []
    original = ExplicitPlot.sample

    def blocked(self: ExplicitPlot, view: ViewRect) -> Geometry:
        threads.append(threading.get_ident())
        entered.set()
        assert release.wait(5)
        return original(self, view)

    monkeypatch.setattr(ExplicitPlot, "sample", blocked)
    canvas = shown_canvas()
    canvas.set_layers([Layer(1, plot("sin(x)"), "#ff0000")])
    try:
        wait_until(entered.is_set)
        ticks: list[bool] = []
        events.Timer.single_shot_call(0, lambda: ticks.append(True))
        canvas.render()
        wait_until(lambda: bool(ticks))
        assert canvas.is_rendering
        assert all(thread != threading.get_ident() for thread in threads)
    finally:
        release.set()
    wait_until(lambda: not canvas.is_rendering)
    canvas.render()
    assert canvas._cache[1]
    canvas.shutdown()


def test_pan_reuses_overlap_and_returning_uses_cached_tiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[ViewRect] = []
    original = ExplicitPlot.sample

    def counted(self: ExplicitPlot, view: ViewRect) -> Geometry:
        calls.append(view)
        return original(self, view)

    monkeypatch.setattr(ExplicitPlot, "sample", counted)
    loader = GeometryLoader()
    expression = plot("sin(x)")
    load(loader, expression)
    initial = {tile.request.key: tile for tile in loader.tiles}
    count = len(calls)
    load(loader, expression, replace(VIEW, x_min=-7.9, x_max=8.1))
    assert len(calls) == count
    assert all(tile is initial[tile.request.key] for tile in loader.tiles)

    load(loader, expression, replace(VIEW, x_min=2, x_max=18))
    overlap = set(initial) & {tile.request.key for tile in loader.tiles}
    assert overlap
    assert all(
        tile is initial[tile.request.key] for tile in loader.tiles if tile.request.key in overlap
    )
    assert len(calls) > count
    count = len(calls)
    load(loader, expression)
    assert len(calls) == count
    assert all(tile is initial[tile.request.key] for tile in loader.tiles)
    loader.stop()


def test_stale_expression_results_are_discarded(monkeypatch: pytest.MonkeyPatch) -> None:
    entered, release = threading.Event(), threading.Event()
    old, new = plot("sin(x)"), plot("cos(x)")
    original = ExplicitPlot.sample

    def blocked(self: ExplicitPlot, view: ViewRect) -> Geometry:
        if self is old:
            entered.set()
            assert release.wait(5)
        return original(self, view)

    monkeypatch.setattr(ExplicitPlot, "sample", blocked)
    loader = GeometryLoader()
    loader.request([(1, old)], VIEW)
    try:
        wait_until(entered.is_set)
        loader.request([(1, new)], VIEW)
        assert all(
            job.cancelled.is_set() for job in loader._active.values() if job.request.plot is old
        )
    finally:
        release.set()
    wait_until(lambda: not loader.loading)
    assert loader.tiles
    assert all(tile.request.plot is new for tile in loader.tiles)
    assert all(tile.request.plot is new for tile in loader._cache.values())
    loader.stop()


def test_cache_limits_do_not_cause_visible_tiles_to_reload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[ViewRect] = []
    original = ExplicitPlot.sample

    def counted(self: ExplicitPlot, view: ViewRect) -> Geometry:
        calls.append(view)
        return original(self, view)

    monkeypatch.setattr(ExplicitPlot, "sample", counted)
    loader = GeometryLoader()
    loader.cache_limit, loader.byte_limit = 1, 2048
    expression = plot("x^2")
    load(loader, expression)
    count = len(calls)
    assert loader.tiles
    assert len(loader._cache) <= 1 and loader._bytes <= 2048
    load(loader, expression)
    assert len(calls) == count
    loader.request([], VIEW)
    assert not loader.loading and not loader.tiles and not loader._cache
    loader.stop()


def test_log_tiles_cover_the_view_at_the_requested_pixel_density() -> None:
    view = ViewRect(0.01, 100, 0.1, 1000, 1200, 900, True, True)
    requests = tile_requests(1, plot("x^2 + y^2 = 16"), view)
    assert requests
    assert min(r.view.x_min for r in requests) <= view.x_min
    assert max(r.view.x_max for r in requests) >= view.x_max
    assert min(r.view.y_min for r in requests) <= view.y_min
    assert max(r.view.y_max for r in requests) >= view.y_max
    for request in requests:
        assert request.view.width_px == TILE_PX and request.view.height_px == TILE_PX
        assert request.view.px_per_x == pytest.approx(view.px_per_x)
        assert request.view.px_per_y == pytest.approx(view.px_per_y)


def test_plot_groups_are_tiled_by_member_without_losing_any_curve() -> None:
    expression = plot("y = list[1, 2, 3] x")
    loader = GeometryLoader()
    load(loader, expression)
    slopes = set()
    for tile in loader.tiles:
        for line in tile.geometry.polylines:
            away = np.abs(line[:, 0]) > 0.1
            slopes.update(np.round(line[away, 1] / line[away, 0], 6))
    assert slopes == {1, 2, 3}
    loader.stop()


def test_large_implicit_canvas_keeps_a_contour_instead_of_drawing_nothing() -> None:
    view = ViewRect(-10, 10, -8, 8, 5000, 4000)
    segments = marching_squares(lambda x, y: x**2 + y**2 - 16, view)
    assert len(segments) > 100
    assert np.isfinite(segments).all()
    np.testing.assert_allclose(np.hypot(*segments[:, :2].T), 4, atol=0.01)


def test_cancellation_scope_stops_numerical_work_and_resets_afterwards() -> None:
    expression = plot("sin(x)")
    with pytest.raises(Cancelled), cancellable(lambda: True):
        expression.sample(VIEW)
    checkpoint()
    assert expression.sample(VIEW).polylines


def test_culling_preserves_crossings_and_does_not_join_separate_runs() -> None:
    line = np.array([[-20, 0], [20, 0], [20, 20], [-20, 20], [-20, 1], [20, 1]], dtype=float)
    points = np.column_stack([np.arange(-100_000, 100_000), np.zeros(200_000)])
    geometry = visible_geometry(Geometry(polylines=(line,), points=points), VIEW)
    assert len(geometry.polylines) == 2
    np.testing.assert_array_equal(geometry.polylines[0], line[:2])
    np.testing.assert_array_equal(geometry.polylines[1], line[-2:])
    assert geometry.points is not None and len(geometry.points) == 17
    assert not np.shares_memory(geometry.polylines[0], line)


def test_resize_reuses_tiles_and_zoom_requests_new_resolution() -> None:
    loader = GeometryLoader()
    expression = plot("x^2 + y^2 = 16")
    load(loader, expression)
    original = {tile.request.key: tile for tile in loader.tiles}
    # Grow the canvas without changing the scale.
    larger = replace(VIEW, x_min=-12, x_max=12, width_px=1200)
    load(loader, expression, larger)
    assert all(
        tile is original[tile.request.key] for tile in loader.tiles if tile.request.key in original
    )
    zoomed = replace(VIEW, x_min=-4, x_max=4, y_min=-3, y_max=3)
    load(loader, expression, zoomed)
    assert not (set(original) & {tile.request.key for tile in loader.tiles})
    loader.stop()


def test_one_failed_tile_does_not_stop_other_tiles(monkeypatch: pytest.MonkeyPatch) -> None:
    original = ImplicitPlot.sample

    def partial_failure(self: ImplicitPlot, view: ViewRect) -> Geometry:
        if view.x_min < 0:
            raise ValueError("test tile failed")
        return original(self, view)

    monkeypatch.setattr(ImplicitPlot, "sample", partial_failure)
    loader = GeometryLoader()
    load(loader, plot("x^2 + y^2 = 16"))
    assert any(tile.geometry.is_empty for tile in loader.tiles)
    assert any(not tile.geometry.is_empty for tile in loader.tiles)
    loader.stop()


def test_latest_analysis_discards_cancelled_work() -> None:
    task = LatestTask()
    entered, release = threading.Event(), threading.Event()
    results: list[tuple[object, object]] = []
    task.ready.connect(lambda key, value: results.append((key, value)))

    def old_analysis() -> str:
        entered.set()
        assert release.wait(5)
        checkpoint()
        return "stale"

    task.request("old", old_analysis)
    try:
        wait_until(entered.is_set)
        task.request("new", lambda: "current")
    finally:
        release.set()
    wait_until(lambda: bool(results))
    assert results == [("new", "current")]
    task.stop()


def test_image_export_waits_for_visible_geometry(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    entered, release = threading.Event(), threading.Event()
    original = ExplicitPlot.sample

    def blocked(self: ExplicitPlot, view: ViewRect) -> Geometry:
        entered.set()
        assert release.wait(5)
        return original(self, view)

    captured: list[bool] = []

    async def capture() -> bytes:
        captured.append(window.canvas.is_rendering)
        return png_bytes(np.full((2, 2, 4), 255, dtype=np.uint8))

    monkeypatch.setattr(ExplicitPlot, "sample", blocked)
    monkeypatch.setattr(window, "capture", capture)
    path = tmp_path / "export.png"
    first = window.controller.document.items[0].id
    window.controller.set_text(first, "y = x")
    window.controller.flush()
    try:
        wait_until(entered.is_set)
        window.export_to(path)
        events.process_events(0.05)
        assert not path.exists()
        assert "Rendering graph" in window.current_message()
    finally:
        release.set()
    wait_until(path.exists)
    assert captured == [False]  # captured only once the geometry had loaded
    assert path.read_bytes().startswith(b"\x89PNG")
    jpeg = tmp_path / "export.jpg"
    window.export_to(jpeg)
    wait_until(jpeg.exists)
    assert jpeg.read_bytes()[:2] == b"\xff\xd8"


def test_completed_tiles_redraw_and_recolor_without_resampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples: list[ViewRect] = []
    original_sample = ExplicitPlot.sample

    def counted_sample(self: ExplicitPlot, view: ViewRect) -> Geometry:
        samples.append(view)
        return original_sample(self, view)

    monkeypatch.setattr(ExplicitPlot, "sample", counted_sample)
    canvas = shown_canvas()
    canvas.set_settings(PLAIN)
    expression = plot("y = x")
    canvas.set_layers([Layer(1, expression, "#ff0000")])
    wait_until(lambda: not canvas.is_rendering)
    canvas.render()
    sample_count = len(samples)
    canvas.viewport.pan_pixels(5, 0)
    shapes = canvas.render()
    assert len(samples) == sample_count
    sx, sy = canvas.viewport.to_screen(1.3, 1.3)
    assert drawn_near(shapes, "#ff0000", float(sx), float(sy))
    canvas.set_layers([Layer(1, expression, "#0000ff")])
    shapes = canvas.render()
    assert len(samples) == sample_count
    assert drawn_near(shapes, "#0000ff", float(sx), float(sy))
    assert not shapes_in_color(shapes, "#ff0000")
    canvas.shutdown()


def test_inequality_tile_borders_do_not_double_the_fill() -> None:
    canvas = shown_canvas()
    canvas.set_settings(PLAIN)
    canvas.set_layers([Layer(1, plot("x + y > -100"), "#ff0000")])
    wait_until(lambda: not canvas.is_rendering)
    images = [s for s in canvas.render() if isinstance(s, cv.Image)]
    assert len(images) > 1  # x = 0 is a tile boundary in the centre of this viewport
    rects = [(i.x, i.y, i.x + i.width, i.y + i.height) for i in images]  # type: ignore[operator]
    for a in range(len(rects)):
        for b in range(a + 1, len(rects)):
            l0, t0, r0, b0 = rects[a]
            l1, t1, r1, b1 = rects[b]
            overlap_x = min(r0, r1) - max(l0, l1)
            overlap_y = min(b0, b1) - max(t0, t1)
            assert overlap_x <= 0.5 or overlap_y <= 0.5, (rects[a], rects[b])
    # Together they cover the visible part of the canvas.
    assert min(r[0] for r in rects) <= 0 and max(r[2] for r in rects) >= 800
    canvas.shutdown()


def test_png_encoding_round_trips() -> None:
    import io

    from PIL import Image

    pixels = np.zeros((3, 5, 4), dtype=np.uint8)
    pixels[1, 2] = (10, 20, 30, 40)
    with Image.open(io.BytesIO(png_bytes(pixels))) as image:
        assert image.size == (5, 3)
        assert image.getpixel((2, 1)) == (10, 20, 30, 40)


def _distance_to_polyline(points: np.ndarray, line: np.ndarray) -> np.ndarray:
    a, b = line[:-1], line[1:]
    ab = b - a
    length2 = np.maximum((ab * ab).sum(axis=1), 1e-300)
    t = np.clip(((points[:, None] - a) * ab).sum(axis=2) / length2, 0, 1)
    nearest = a + t[..., None] * ab
    return np.sqrt(((points[:, None] - nearest) ** 2).sum(axis=2)).min(axis=1)


def test_simplified_lines_stay_within_tolerance_and_keep_reversals() -> None:
    t = np.linspace(0, 20, 2000)
    wave = np.column_stack([t * 40, 80 * np.sin(t)])
    keep = _simplify(wave, 0.2)
    assert keep[0] == 0 and keep[-1] == len(wave) - 1 and len(keep) < len(wave) / 3
    assert _distance_to_polyline(wave, wave[keep]).max() <= 0.2 + 1e-9
    # Out along a line and back: the turning point is far from the chord's *segment*.
    hairpin = np.column_stack([np.r_[np.arange(50.0), np.arange(48.0, -1, -1)], np.zeros(99)])
    assert 49 in _simplify(hairpin, 0.2)
    straight = np.column_stack([np.arange(500.0), np.arange(500.0) * 0.5])
    assert len(_simplify(straight, 0.2)) <= 500 / 32 + 2


def test_outline_joins_implicit_segments_into_accurate_polylines() -> None:
    view = ViewRect(-8, 8, -6, 6, 800, 600)
    geometry = plot("x^2 + y^2 = 16").sample(view)
    assert geometry.segments is not None
    lines = outline(geometry, view)
    assert len(lines) == 1 and np.array_equal(lines[0][0], lines[0][-1])  # one closed loop
    vertices = sum(len(line) for line in lines)
    assert vertices < len(geometry.segments) / 2
    ends = geometry.segments.reshape(-1, 2) * [view.px_per_x, view.px_per_y]
    drawn = lines[0] * [view.px_per_x, view.px_per_y]
    assert _distance_to_polyline(ends, drawn).max() <= 0.2 + 1e-6


def test_redrawing_an_unchanged_view_reuses_frozen_shapes() -> None:
    canvas = shown_canvas()
    canvas.set_layers(
        [Layer(1, plot("y = sin(x)"), "#ff0000"), Layer(2, plot("x^2 + y^2 = 9"), "#0000ff")]
    )
    wait_until(lambda: not canvas.is_rendering)
    first = canvas.render()
    second = canvas.render()
    assert len(first) == len(second) and all(a is b for a, b in zip(first, second, strict=True))
    assert all(hasattr(shape, "_frozen") for shape in first)
    sx, sy = canvas.viewport.to_screen(3.0, 0.0)
    assert drawn_near(second, "#0000ff", float(sx), float(sy))
    canvas.viewport.pan_pixels(7, 0)
    panned = canvas.render()
    curves = {id(shape) for shape in shapes_in_color(first, "#ff0000")}
    assert curves and not curves & {id(shape) for shape in panned}
    assert drawn_near(panned, "#0000ff", float(sx) + 7, float(sy))
    canvas.shutdown()


def test_dashed_implicit_boundary_is_one_continuous_path() -> None:
    canvas = shown_canvas()
    canvas.set_settings(PLAIN)
    canvas.set_layers([Layer(1, plot("x^2 + y^2 < 9"), "#ff0000")])
    wait_until(lambda: not canvas.is_rendering)
    paths = [s for s in shapes_in_color(canvas.render(), "#ff0000") if isinstance(s, cv.Path)]
    assert paths and all(s.paint.stroke_dash_pattern for s in paths)
    elements = [e for path in paths for e in path.elements]
    moves = sum(isinstance(e, cv.Path.MoveTo) for e in elements)
    assert moves <= 2 * len(paths) and len(elements) > 20 * moves  # few, long sub-paths
    canvas.shutdown()
