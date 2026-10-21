from __future__ import annotations

import numpy as np
import pytest

from opengraph.engine.compiler import compile_expression
from opengraph.engine.plots import ExplicitPlot, Geometry, ImplicitPlot
from opengraph.engine.sampling import ViewRect, sample_parametric

VIEW = ViewRect(-10, 10, -7.5, 7.5, 800, 600)


def sample(text: str, view: ViewRect = VIEW) -> Geometry:
    result = compile_expression(text)
    assert result.plot is not None, result.message
    return result.plot.sample(view)


def test_smooth_curve_is_one_accurate_polyline() -> None:
    geometry = sample("x^2 / 4")
    assert len(geometry.polylines) == 1
    line = geometry.polylines[0]
    assert line[0, 0] <= VIEW.x_min and line[-1, 0] >= VIEW.x_max
    np.testing.assert_allclose(line[:, 1], line[:, 0] ** 2 / 4)
    assert np.all(np.diff(line[:, 0]) > 0)


def test_samples_are_grid_aligned_so_panning_is_stable() -> None:
    a = sample("sin(x)").polylines[0][:, 0]
    shifted = ViewRect(-9.7, 10.3, -7.5, 7.5, 800, 600)
    b = sample("sin(x)", shifted).polylines[0][:, 0]
    common = np.intersect1d(a, b)
    assert len(common) > 0.8 * min(len(a), len(b))


def test_pole_breaks_the_curve() -> None:
    lines = sample("1/x").polylines
    assert len(lines) == 2
    left, right = lines
    assert np.max(left[:, 0]) < 0 < np.min(right[:, 0])
    # Each branch should run off the screen rather than stopping short of the edge.
    assert np.min(left[:, 1]) < VIEW.y_min and np.max(right[:, 1]) > VIEW.y_max


def test_tan_is_split_at_every_asymptote() -> None:
    lines = sample("tan(x)").polylines
    # Asymptotes at ±pi/2, ±3pi/2, ±5pi/2 within [-10, 10] -> 7 branches.
    assert len(lines) == 7
    for line in lines:
        # No segment joins the region above the view to the region below it.
        y0, y1 = line[:-1, 1], line[1:, 1]
        crossing = ((y0 > VIEW.y_max) & (y1 < VIEW.y_min)) | ((y0 < VIEW.y_min) & (y1 > VIEW.y_max))
        assert not crossing.any()


def test_jump_discontinuities_are_not_joined() -> None:
    lines = sample("floor(x)").polylines
    assert len(lines) == 22  # floor(-10.05) = -11 ... floor(10.05) = 10
    for line in lines:
        assert np.ptp(line[:, 1]) == 0


def test_domain_edges_are_located_precisely() -> None:
    (line,) = sample("sqrt(4 - x^2)").polylines
    assert line[0, 0] < -1.9999 and line[-1, 0] > 1.9999
    assert np.isfinite(line).all()


def test_removable_hole_does_not_split_visibly() -> None:
    lines = sample("sin(x)/x").polylines
    if len(lines) == 2:  # the sample at exactly x=0 is undefined
        gap = lines[1][0, 0] - lines[0][-1, 0]
        assert gap < 1e-5  # far below a pixel


def test_steep_continuous_curve_is_not_broken() -> None:
    assert len(sample("atan(1000x)").polylines) == 1
    assert len(sample("x^(1/3)").polylines) == 1


def test_undefined_everywhere_draws_nothing() -> None:
    assert sample("sqrt(-1 - x^2)").is_empty


def test_huge_values_are_clamped_for_rendering() -> None:
    for line in sample("exp(x^2)").polylines:
        assert np.abs(line[:, 1]).max() < 1e4


def test_x_as_function_of_y() -> None:
    (line,) = sample("x = y^2").polylines
    np.testing.assert_allclose(line[:, 0], line[:, 1] ** 2)
    assert line[0, 1] <= VIEW.y_min


def test_implicit_circle_is_accurate() -> None:
    geometry = sample("x^2 + y^2 = 16")
    assert geometry.segments is not None and len(geometry.segments) > 50
    ends = geometry.segments.reshape(-1, 2)
    radii = np.hypot(ends[:, 0], ends[:, 1])
    np.testing.assert_allclose(radii, 4, atol=0.02)


def test_implicit_rejects_pole_crossings() -> None:
    # F = y^2 - 1/x changes sign across the pole at x = 0, but has no zero there.
    geometry = sample("y^2 = 1/x")
    assert geometry.segments is not None
    xs = geometry.segments[:, [0, 2]]
    assert np.all(np.abs(xs) > 0.01)


def test_explicit_plot_evaluate_matches_sampled_curve() -> None:
    plot = compile_expression("y = 2x + 1").plot
    assert isinstance(plot, ExplicitPlot)
    assert float(plot.evaluate(3.0)) == 7.0


def test_implicit_plot_equality_by_expression() -> None:
    a = compile_expression("x^2 + y^2 = 1").plot
    b = compile_expression("x^2+y^2=1").plot
    assert isinstance(a, ImplicitPlot) and a == b


@pytest.mark.filterwarnings("error::RuntimeWarning")
@pytest.mark.parametrize(
    "expression",
    [
        "parametric(exp(t), exp(t), t, [0, 720])",
        "r = exp(20theta)",
    ],
)
def test_parametric_and_polar_overflow_keeps_finite_geometry(expression: str) -> None:
    geometry = sample(expression)
    assert geometry.polylines
    for line in geometry.polylines:
        assert np.isfinite(line).all()
        # Values with no finite screen position must not reach the renderer.
        assert np.isfinite(line * [VIEW.px_per_x, VIEW.px_per_y]).all()


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_parametric_huge_constant_coordinates_stay_connected() -> None:
    unit_view = ViewRect(-400, 400, -300, 300, 800, 600)
    (line,) = sample_parametric(
        lambda t: (np.full_like(t, 1e308), np.full_like(t, 1e308)), 0, 1, unit_view
    )
    assert len(line) >= 2
    assert np.all(line == 1e308)


@pytest.mark.filterwarnings("error::RuntimeWarning")
@pytest.mark.parametrize("same_sign", [False, True])
def test_parametric_extreme_jump_is_not_joined(same_sign: bool) -> None:
    unit_view = ViewRect(-400, 400, -300, 300, 800, 600)

    def curve(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.where(t < 0.123, -1e308, 1e308)
        return x, x if same_sign else -x

    lines = sample_parametric(curve, 0, 1, unit_view)
    assert len(lines) == 2
    assert np.all(lines[0][:, 0] < 0)
    assert np.all(lines[1][:, 0] > 0)


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_parametric_undefined_values_split_the_curve() -> None:
    def curve(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        y = np.where(np.abs(t) < 0.1, np.nan, t)
        return t, y

    lines = sample_parametric(curve, -1, 1, VIEW)
    assert len(lines) == 2
    assert np.all(lines[0][:, 0] <= -0.1)
    assert np.all(lines[1][:, 0] >= 0.1)


@pytest.mark.filterwarnings("error::RuntimeWarning")
def test_parametric_log_domain_edge_keeps_the_positive_branch() -> None:
    view = ViewRect(0.01, 10, -1, 1, 800, 600, x_log=True)
    (line,) = sample_parametric(lambda t: (t, t * 0), -1, 1, view)
    assert np.all(line[:, 0] > 0)
    assert line[0, 0] < view.x_min
    assert line[-1, 0] == 1
