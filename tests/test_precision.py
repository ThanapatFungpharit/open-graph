"""High-precision evaluation (gmpy2/MPFR), the compiled kernels and deep zoom."""

from __future__ import annotations

import io
import threading

import gmpy2
import numpy as np
import pytest
from PIL import Image

from opengraph import native
from opengraph.engine import precise
from opengraph.engine.ast import Node
from opengraph.engine.compiler import compile_expression
from opengraph.engine.evaluator import NumericFunction
from opengraph.engine.plots import ExplicitPlot, Plot
from opengraph.engine.sampling import ViewRect
from opengraph.graph import rendering
from opengraph.graph.canvas import FILL_ALPHA, GraphCanvas, Layer, fill_png, place_points
from opengraph.graph.rendering import GeometryLoader, _chain_segments, _simplify, outline
from opengraph.graph.settings import GraphSettings
from opengraph.graph.viewport import MAX_DEEP_SCALE, MAX_SCALE, Viewport
from tests.support import drawn_near, stroked_points, wait_until

PLAIN = GraphSettings(major_grid=False, minor_grid=False, axes=False, axis_numbers=False)
COMPILED = pytest.mark.skipif(native.kernels is None, reason="kernels not compiled")
COMPILED_MPFR = pytest.mark.skipif(native.mpvm is None, reason="MPFR machine not compiled")


def plot(text: str) -> Plot:
    result = compile_expression(text)
    assert result.plot is not None, result.message
    return result.plot


def node_of(text: str) -> Node:
    p = plot(text)
    assert isinstance(p, ExplicitPlot)
    return p.node


def value(text: str) -> float:
    result = compile_expression(text)
    numeric = NumericFunction(node_of(f"y = {text}"))
    assert numeric.is_constant, result.message
    return float(numeric.scalar())


# -- constants are folded in high precision ---------------------------------------------------


def test_constants_are_computed_exactly_then_rounded_once() -> None:
    assert value("1e16 + 1 - 1e16") == 1.0  # float64 gives 0
    assert value("0.1 + 0.2") == 0.3  # float64 gives 0.30000000000000004
    assert value("sum(1/k^2 for k from 1 to 1000)") == pytest.approx(1.6439345666815598, abs=0)
    assert value("(1 + 1e-15)^(1e15)") == pytest.approx(2.718281828459044, rel=1e-15)


def test_degree_trigonometry_of_exact_angles_is_exact() -> None:
    from opengraph.engine.compiler import CompileOptions

    result = compile_expression("y = sin(30)", CompileOptions(degrees=True))
    assert isinstance(result.plot, ExplicitPlot)
    assert NumericFunction(result.plot.node).scalar() == 0.5  # 0.49999999999999994 in float


def test_folding_keeps_real_domains_and_odd_roots() -> None:
    assert value("(-8)^(1/3)") == -2.0
    assert np.isnan(value("sqrt(-1)"))
    assert np.isnan(value("ln(-1)"))
    assert value("1/0") == np.inf
    assert np.isnan(value("Gamma(-2)"))
    assert value("factorial(20)") == 2432902008176640000.0
    assert value("nCr(60, 30)") == 118264581564861424.0


# -- the MPFR machine, its Python fallback and the float evaluator agree ----------------------

EXPRESSIONS = [
    "x^2 + sin(x)",
    "x^(1/3) + x^(2/3)",
    "x^(-1/3)",
    "Gamma(x) + factorial(x)",
    "mod(x, 3) + mod(-x, 2.5)",
    "max(x, sin(x), 0.5) - min(x, cos(x))",
    "nCr(10, x) + nPr(6, x)",
    "log(abs(x), 2) + log(abs(x)) + ln(x)",
    "atan(x, 2) + atan(x) + asin(x/10) + acos(x/10)",
    "sinh(x) + cosh(x) + tanh(x) + asinh(x) + acosh(x) + atanh(x/10)",
    "sec(x) + csc(x) + cot(x) + tan(x)",
    "floor(x) + ceil(x) + round(x) + sign(x) + abs(x) + cbrt(x) + exp(x/5)",
    "plot(x^2 for x in (0, 2])",
    "x/x",
]
# Offsets are added to an origin, so -0.0 becomes +0.0 (it only matters at a pole).
SAMPLES = np.array([-3.5, -2.0, -1.0, 0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 7.25, np.nan])


@pytest.mark.parametrize("text", EXPRESSIONS)
def test_precise_evaluation_matches_float_semantics(text: str) -> None:
    node = node_of(f"y = {text}")
    expected = NumericFunction(node)(x=SAMPLES)
    got = precise.PreciseFunction(node).evaluate(128, {"x": (0, SAMPLES)})
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12, equal_nan=True)


@COMPILED_MPFR
@pytest.mark.parametrize("text", EXPRESSIONS)
def test_compiled_machine_matches_python_fallback(
    text: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    node = node_of(f"y = {text}")
    compiled = precise.PreciseFunction(node).evaluate(200, {"x": (0, SAMPLES)})
    monkeypatch.setattr(native, "mpvm", None)
    fallback = precise.PreciseFunction(node).evaluate(200, {"x": (0, SAMPLES)})
    np.testing.assert_array_equal(compiled, fallback)


def test_offsets_from_an_exact_origin_keep_full_precision() -> None:
    bits = 256
    with precise.context(bits):
        origin = gmpy2.mpfr("10000.000000000000000000000000000001")
        out = origin * origin
    t = np.linspace(0, 1e-28, 5)
    got = precise.PreciseFunction(node_of("y = x^2")).evaluate(bits, {"x": (origin, t)}, out)
    np.testing.assert_allclose(got, 2e4 * t, rtol=1e-12)  # float64 would give all zeros


def test_complex_values_have_no_precise_form() -> None:
    node = node_of("y = re(e^(i x))")
    with pytest.raises(precise.Unsupported):
        precise.PreciseFunction(node).evaluate(64, {"x": (0, SAMPLES)})


def test_precision_is_only_used_where_float64_cannot_resolve_a_pixel() -> None:
    assert precise.precision_bits(10.0, 1 / 50) == 0
    assert precise.precision_bits(1e4, 1e-12) > 64
    assert precise.precision_bits(1e-20, 1e-30) == 0  # tiny but near zero: float64 is fine


# -- compiled geometry kernels give the reference results -------------------------------------


@COMPILED
def test_compiled_simplify_matches_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(7)
    for _ in range(100):
        points = np.cumsum(rng.normal(size=(int(rng.integers(3, 300)), 2)), axis=0)
        starts = np.array([0, len(points) // 2, len(points)])
        for tolerance in (0.0, 0.2, 2.0):
            compiled = _simplify(points, tolerance, starts)
            monkeypatch.setattr(native, "kernels", None)
            reference = _simplify(points, tolerance, starts)
            monkeypatch.undo()
            np.testing.assert_array_equal(compiled, reference)


@COMPILED
def test_compiled_chains_match_reference() -> None:
    assert native.kernels is not None
    rng = np.random.default_rng(3)
    for _ in range(200):
        n = int(rng.integers(2, 40))
        pairs = rng.integers(0, n, size=(int(rng.integers(1, 60)), 2))
        pairs = np.ascontiguousarray(pairs[pairs[:, 0] != pairs[:, 1]], dtype=np.int64)
        if not len(pairs):
            continue
        nodes, lengths = native.kernels.chain_segments(pairs, n)
        chains = [list(c) for c in np.split(nodes, np.cumsum(lengths)[:-1])]
        assert chains == _chain_segments(pairs.tolist())


@COMPILED
def test_compiled_outline_matches_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    view = ViewRect(-6, 6, -6, 6, 600, 600)
    geometry = plot("sin(x^2 + y^2) = cos(x y)").sample(view)
    compiled = outline(geometry, view)
    monkeypatch.setattr(native, "kernels", None)
    reference = outline(geometry, view)
    assert len(compiled) == len(reference)
    for a, b in zip(compiled, reference, strict=True):
        np.testing.assert_array_equal(a, b)


@COMPILED
def test_compiled_placement_matches_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    points = np.random.default_rng(1).normal(scale=300, size=(50, 2))
    compiled = place_points(points, 1.5, 0.75, 10.125, -3.3)
    monkeypatch.setattr(native, "kernels", None)
    assert compiled == place_points(points, 1.5, 0.75, 10.125, -3.3)


# -- work happens on the geometry workers -----------------------------------------------------


def test_workers_prepare_screen_geometry_and_fill_images(monkeypatch: pytest.MonkeyPatch) -> None:
    ui_thread = threading.get_ident()
    threads: list[int] = []
    original = rendering.screen_geometry

    def recording(*args: object) -> rendering.ScreenGeometry:
        threads.append(threading.get_ident())
        return original(*args)  # type: ignore[arg-type]

    monkeypatch.setattr(rendering, "screen_geometry", recording)
    loader = GeometryLoader()
    loader.request([(1, plot("y < sin(x)"))], ViewRect(-8, 8, -6, 6, 800, 600))
    wait_until(lambda: not loader.loading, timeout_ms=10_000)
    assert loader.tiles and all(tile.screen is not None for tile in loader.tiles)
    assert threads and ui_thread not in threads
    fills = [tile.screen.fill for tile in loader.tiles if tile.screen and tile.screen.fill]
    assert fills
    loader.stop()

    with Image.open(io.BytesIO(fill_png(fills[0], "#336699"))) as image:
        rgba = np.asarray(image.convert("RGBA"))
    shaded = rgba[rgba[:, :, 3] > 0]
    assert len(shaded) and (shaded == (0x33, 0x66, 0x99, FILL_ALPHA)).all()


# -- deep zoom ---------------------------------------------------------------------------------


def deep_viewport(x: float, y: float, scale: float) -> Viewport:
    vp = Viewport(800, 600)
    vp.center_x, vp.center_y = x, y
    vp.scale_x = vp.scale_y = scale
    vp._reanchor()
    return vp


def test_viewport_anchors_deep_zoom_and_releases_it_when_zooming_out() -> None:
    vp = deep_viewport(10_000.0, 0.0, 50.0)
    assert not vp.anchored
    for _ in range(30):
        vp.zoom_center(10)
    assert vp.anchored and vp.bits >= 128
    vp.pan_pixels(-1, 0)  # one pixel is 2e-32 world units: kept exactly
    with precise.context(512):
        moved = gmpy2.mpfr(vp.absolute("x", vp.center_x)) - 10_000
    assert float(moved) == pytest.approx(1 / vp.scale_x, rel=1e-9)
    for _ in range(30):
        vp.zoom_center(0.1)
    assert not vp.anchored and vp.center_x == pytest.approx(10_000.0)


def test_log_axes_keep_the_float64_zoom_limit() -> None:
    vp = Viewport(800, 600, x_log=True)
    for _ in range(100):
        vp.zoom_center(10)
    assert vp.scale_x <= MAX_SCALE and not vp.anchored
    linear = Viewport(800, 600)
    for _ in range(100):
        linear.zoom_center(10)
    assert linear.scale_x == pytest.approx(MAX_DEEP_SCALE)


def test_deep_tiles_have_exact_origins_and_smooth_curves() -> None:
    vp = deep_viewport(3.0, 9.0, 1e25)  # y = x^2 passes through (3, 9) with slope 6
    vp.anchor_x, vp.anchor_y = precise.mpfr(3, 512), precise.mpfr(9, 512)
    vp.center_x = vp.center_y = 0.0
    view = vp.view_rect()
    assert view.bits > 53
    requests = rendering.tile_requests(1, plot("y = x^2"), view)
    assert requests and all(r.view.bits == view.bits and r.view.origin_x for r in requests)
    lines = [line for r in requests for line in r.plot.sample(r.view).polylines]
    assert lines
    for request in requests:
        for line in request.plot.sample(request.view).polylines:
            with precise.context(512):
                x = [gmpy2.mpfr(request.view.origin_x) + float(t) for t in line[:, 0]]
                y = [gmpy2.mpfr(request.view.origin_y) + float(v) for v in line[:, 1]]
                errors = [float((a * a - b) * 1e25) for a, b in zip(x, y, strict=True)]
            assert max(abs(e) for e in errors) < 1e-6  # in pixels: no staircase


def test_canvas_draws_and_labels_a_deep_zoom() -> None:
    canvas = GraphCanvas()
    canvas.resize(800, 600)
    canvas.set_settings(GraphSettings(major_grid=True, minor_grid=False))
    canvas.set_layers([Layer(1, plot("y = x^2"), "#ff0000")])
    vp = canvas.viewport
    vp.center_x, vp.center_y = 3.0, 9.0
    for _ in range(24):  # 50 * 10^24 px per unit: far beyond float64 at (3, 9)
        vp.zoom_center(10)
    canvas._view_changed()
    assert vp.anchored
    wait_until(lambda: not canvas.is_rendering, timeout_ms=20_000)
    shapes = canvas.render()
    # The curve passes through the centre with slope 6 (on screen: down-right is -y).
    assert drawn_near(shapes, "#ff0000", 400, 300, radius=2)
    assert drawn_near(shapes, "#ff0000", 400 + 30, 300 - 180, radius=3)
    assert len(stroked_points(shapes, "#ff0000")) >= 2
    # Labels and coordinates show the digits a pixel resolves.
    texts = [s.value for s in shapes if hasattr(s, "value") and isinstance(s.value, str)]
    assert any("3.0000000000000000000000" in t.replace("×", "") for t in texts), texts
    position = canvas.format_position(1 / vp.scale_x, 0.0)
    assert position.startswith("(3.00000000000000000000000")
    # Ticks sit exactly on the multiples of their (decimal) step, labelled by value.
    for axis, center in (("x", 3), ("y", 9)):
        majors, _, step = canvas._ticks(axis)
        assert majors and step is not None
        exact_step = precise.literal(step, 512)
        for tick in majors:
            with precise.context(512):
                multiple = gmpy2.mpfr(vp.absolute(axis, tick)) / exact_step
                assert abs(float(multiple - gmpy2.rint(multiple))) < 1e-6
        middle = min(majors, key=abs)
        if abs(middle) < 1e-9 / vp.scale_x:  # the tick at the centre
            assert canvas._tick_label(middle, step, axis).startswith(f"{center}.000000000")
    assert canvas.markers() == []  # no float64 analysis at this depth
    canvas.shutdown()


def test_deep_zoom_without_the_compiled_machine_still_draws(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(native, "mpvm", None)
    vp = deep_viewport(1.0, 1.0, 1e22)
    vp.anchor_x, vp.anchor_y = precise.mpfr(1, 512), precise.mpfr(1, 512)
    vp.center_x = vp.center_y = 0.0
    requests = rendering.tile_requests(1, plot("y = x^3"), vp.view_rect())
    assert any(r.plot.sample(r.view).polylines for r in requests)
