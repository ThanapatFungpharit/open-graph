from __future__ import annotations

import numpy as np
import pytest

from opengraph.engine.compiler import CompileOptions, compile_document, compile_expression
from opengraph.engine.plots import ExplicitPlot, ParametricPlot, PointPlot, RegionPlot, members
from opengraph.engine.sampling import ViewRect

COMPLEX = CompileOptions(complex_mode=True)


def rows(*texts: str):
    results = compile_document(list(enumerate(texts)))
    assert all(not result.is_error for result in results.values()), results
    return list(results.values())


def test_named_interval_draws_circle_without_parametric_command() -> None:
    result = rows("u = [0, 2pi]", "point[cos(u), sin(u)]")[-1]
    assert isinstance(result.plot, ParametricPlot)
    t = np.linspace(0, 1, 1001)
    x, y = result.plot.xy(t)
    np.testing.assert_allclose(x, np.cos(2 * np.pi * t))
    np.testing.assert_allclose(y, np.sin(2 * np.pi * t))
    geometry = result.plot.sample(ViewRect(-2, 2, -2, 2, 400, 400))
    assert geometry.polylines
    for line in geometry.polylines:
        np.testing.assert_allclose(line[:, 0] ** 2 + line[:, 1] ** 2, 1, atol=1e-12)


def test_transforms_and_function_calls_preserve_correlation() -> None:
    result = rows("u = [-2, 3]", "v = 2u + 1", "f(a) = a^2", "point[v, f(u) - u u]")[-1]
    assert isinstance(result.plot, ParametricPlot)
    x, y = result.plot.xy(np.array([0, 0.5, 1]))
    np.testing.assert_allclose(x, [-3, 2, 7])
    np.testing.assert_allclose(y, 0)


def test_distinct_intervals_pair_by_fractional_position() -> None:
    result = compile_expression("point[[0, 2], [10, 20]]")
    assert isinstance(result.plot, ParametricPlot)
    x, y = result.plot.xy(np.array([0, 0.5, 1]))
    np.testing.assert_allclose(x, [0, 1, 2])
    np.testing.assert_allclose(y, [10, 15, 20])


@pytest.mark.parametrize(
    "interval, finite",
    [
        ("[0, 1]", [True, True, True]),
        ("[0, 1)", [True, True, False]),
        ("(0, 1]", [False, True, True]),
        ("(0, 1)", [False, True, False]),
    ],
)
def test_open_and_closed_endpoints(interval: str, finite: list[bool]) -> None:
    result = compile_expression(f"point[{interval}, 3]")
    assert isinstance(result.plot, ParametricPlot)
    x, y = result.plot.xy(np.array([0, 0.5, 1]))
    np.testing.assert_array_equal(np.isfinite(x) & np.isfinite(y), finite)


def test_empty_and_singleton_intervals() -> None:
    assert compile_expression("point[[2, 2), 3]").plot is None
    assert compile_expression("point[[2, 2], 3]").plot == PointPlot(2, 3)
    assert compile_expression("y = [2, 2)").plot is None


def test_lists_broadcast_into_multiple_continuous_curves() -> None:
    result = rows("u = [0, 1]", "point[u, list[1, 2] u]")[-1]
    assert result.plot is not None
    curves = list(members(result.plot))
    assert len(curves) == 2 and all(isinstance(curve, ParametricPlot) for curve in curves)
    for multiplier, curve in enumerate(curves, 1):
        assert isinstance(curve, ParametricPlot)
        x, y = curve.xy(np.array([0, 0.5, 1]))
        np.testing.assert_allclose(y, multiplier * x)


@pytest.mark.parametrize("text", ["[1, 2]+1", "-[1, 2]", "sin([1, 2])", "2[1, 2]"])
def test_arithmetic_feedback_keeps_continuous_values(text: str) -> None:
    result = compile_expression(text)
    assert not result.is_error and "continuous values" in result.message


def test_affine_curve_family_fills_both_signs_of_slope() -> None:
    result = compile_expression("y = [1, 2) x")
    assert result.plot is not None
    regions = [p for p in members(result.plot) if isinstance(p, RegionPlot) and p.shade]
    x = np.array([-1, -1, -1, 1, 1, 1])
    y = np.array([-1, -1.5, -2, 1, 1.5, 2])
    mask = np.logical_or.reduce([region.holds(x, y) for region in regions])
    np.testing.assert_array_equal(mask, [True, True, False, True, True, False])
    collapsed = [p for p in members(result.plot) if isinstance(p, ExplicitPlot)]
    assert len(collapsed) == 1 and collapsed[0].evaluate(0) == 0


def test_function_can_return_continuous_values() -> None:
    result = rows("f(a) = [1, 2] a", "y = f(x)")[-1]
    assert result.plot is not None
    regions = [p for p in members(result.plot) if isinstance(p, RegionPlot) and p.shade]
    assert any(region.holds(np.array([2]), np.array([3]))[0] for region in regions)


@pytest.mark.parametrize(
    "text",
    [
        "total([0, 1])",
        "length([0, 1])",
        "[0, sin([0, 1])]",
        "list[1...sin([0, 1])]",
        "list[[0, 1]+1]",
        "list[1, 2][sin([0, 1])]",
        "[0, 1][1]",
        "2a = [0, 1]",
    ],
)
def test_finite_number_operations_reject_continuous_collections(text: str) -> None:
    assert compile_expression(text).is_error


def test_complex_interval_expression_draws_the_reported_spiral() -> None:
    texts = [
        "e^(t(ln(sqrt(a^2+b^2))+i*atan(b,a)))",
        "t=[0,100]",
        "a=1",
        "b=1/2",
    ]
    results = compile_document(list(enumerate(texts)), COMPLEX)
    assert all(not result.is_error for result in results.values())
    result = results[0]
    assert "continuous values" in result.message
    assert isinstance(result.plot, ParametricPlot)
    fraction = np.linspace(0, 1, 1001)
    expected = np.exp(100 * fraction * (np.log(np.hypot(1, 0.5)) + 1j * np.arctan2(0.5, 1)))
    x, y = result.plot.xy(fraction)
    np.testing.assert_allclose(x + 1j * y, expected, rtol=1e-12)
    geometry = result.plot.sample(ViewRect(-10, 10, -10, 10, 600, 600))
    assert geometry.polylines
    assert all(np.isfinite(line).all() for line in geometry.polylines)
    assert not results[1].plot  # the parameter definition stays a definition
    real_result = compile_document(list(enumerate(texts)))[0]
    assert real_result.is_error and "Unknown name 'i'" in real_result.message


def test_complex_interval_circle_preserves_correlation_and_samples_geometry() -> None:
    result = compile_expression("e^(i [0,2pi])", COMPLEX)
    assert isinstance(result.plot, ParametricPlot)
    geometry = result.plot.sample(ViewRect(-2, 2, -2, 2, 400, 400))
    assert geometry.polylines
    for line in geometry.polylines:
        np.testing.assert_allclose(np.sum(line**2, axis=1), 1, atol=1e-12)
    results = compile_document([(0, "u=[0,2]"), (1, "(u-u)+i*u")], COMPLEX)
    plot = results[1].plot
    assert isinstance(plot, ParametricPlot)
    x, y = plot.xy(np.array([0, 0.5, 1]))
    np.testing.assert_allclose(x, 0)
    np.testing.assert_allclose(y, [0, 1, 2])


@pytest.mark.parametrize(
    "interval,finite",
    [
        ("[0,1]", [True, True, True]),
        ("[0,1)", [True, True, False]),
        ("(0,1]", [False, True, True]),
        ("(0,1)", [False, True, False]),
    ],
)
def test_complex_interval_endpoints(interval: str, finite: list[bool]) -> None:
    result = compile_expression(f"e^(i*{interval})", COMPLEX)
    assert isinstance(result.plot, ParametricPlot)
    x, y = result.plot.xy(np.array([0, 0.5, 1]))
    np.testing.assert_array_equal(np.isfinite(x), finite)
    np.testing.assert_array_equal(np.isfinite(y), finite)


@pytest.mark.parametrize("text", ["e^(i*[2,2))", "e^(i*(2,2])", "[2,2)", "1/([0,0])"])
def test_empty_or_undefined_complex_interval_has_no_geometry(text: str) -> None:
    result = compile_expression(text, COMPLEX)
    assert not result.is_error and result.plot is None


def test_singleton_complex_interval_is_a_point() -> None:
    assert compile_expression("e^(i [0,0])", COMPLEX).plot == PointPlot(1, 0)
    assert compile_expression("[2,2]", COMPLEX).plot == PointPlot(2, 0)


def test_complex_interval_lists_broadcast_into_curves() -> None:
    result = compile_expression("list[1,2]*e^(i [0,2pi])", COMPLEX)
    assert result.plot is not None
    curves = list(members(result.plot))
    assert len(curves) == 2
    for radius, curve in enumerate(curves, 1):
        assert isinstance(curve, ParametricPlot)
        x, y = curve.xy(np.linspace(0, 1, 101))
        np.testing.assert_allclose(np.hypot(x, y), radius)
    assert compile_expression("list[]*e^(i [0,1])", COMPLEX).plot is None


def test_real_interval_values_plot_on_the_real_axis_only_in_complex_mode() -> None:
    for text in ("[1,3]", "[0,2]+1"):
        result = compile_expression(text, COMPLEX)
        assert isinstance(result.plot, ParametricPlot)
        x, y = result.plot.xy(np.array([0, 0.5, 1]))
        np.testing.assert_allclose(x, [1, 2, 3])
        np.testing.assert_allclose(y, 0)
        assert compile_expression(text).plot is None


def test_complex_interval_undefined_values_remain_gaps() -> None:
    result = compile_expression("1/([0,2]-1)+i", COMPLEX)
    assert isinstance(result.plot, ParametricPlot)
    x, y = result.plot.xy(np.array([0, 0.5, 1]))
    np.testing.assert_allclose(x[[0, 2]], [-1, 1])
    np.testing.assert_allclose(y[[0, 2]], 1)
    assert not (np.isfinite(x[1]) and np.isfinite(y[1]))


def test_complex_interval_functions_use_principal_branches() -> None:
    result = compile_expression("sqrt([-4,-1])", COMPLEX)
    assert isinstance(result.plot, ParametricPlot)
    x, y = result.plot.xy(np.array([0, 0.5, 1]))
    np.testing.assert_allclose(x, 0)
    np.testing.assert_allclose(y, np.sqrt([4, 2.5, 1]))
