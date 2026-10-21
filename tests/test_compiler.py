from __future__ import annotations

import math

import numpy as np
import pytest

from opengraph.engine.compiler import (
    CompileOptions,
    RowResult,
    compile_document,
    compile_expression,
    compile_with_values,
)
from opengraph.engine.errors import MathError
from opengraph.engine.plots import ExplicitPlot, ImplicitPlot, ParametricPlot, PointPlot, RegionPlot
from opengraph.engine.sampling import ViewRect


def compile_rows(*texts: str) -> list[RowResult]:
    results = compile_document(list(enumerate(texts)))
    return [results[i] for i in range(len(texts))]


def explicit(result: RowResult) -> ExplicitPlot:
    assert not result.is_error, result.message
    assert isinstance(result.plot, ExplicitPlot)
    return result.plot


@pytest.mark.parametrize(
    ("text", "independent"),
    [
        ("y = x^2", "x"),
        ("x^2", "x"),
        ("x^2 = y", "x"),
        ("y = 3", "x"),
        ("x = y^2", "y"),
        ("y^2 = x", "y"),
        ("x = 2", "y"),
    ],
)
def test_explicit_forms(text: str, independent: str) -> None:
    assert explicit(compile_expression(text)).independent == independent


def test_classification_of_other_forms() -> None:
    assert isinstance(compile_expression("x^2 + y^2 = 1").plot, ImplicitPlot)
    # Linear in y with a constant coefficient: rearranged to y = x^2/2.
    assert explicit(compile_expression("2y = x^2")).independent == "x"
    # x·y = 1 would need dividing by x (losing x = 0 for x·y = 0): stays implicit.
    assert isinstance(compile_expression("x y = 1").plot, ImplicitPlot)
    assert explicit(compile_expression("2y = x")).independent == "y"  # x = 2y
    point = compile_expression("point[1, -2/4]").plot
    assert point == PointPlot(1.0, -0.5)
    value = compile_expression("2 + 3")
    assert value.plot is None and value.message == "= 5" and not value.is_error
    assert compile_expression("   ").plot is None and not compile_expression("").is_error


def test_function_definition_is_graphed_and_usable() -> None:
    defn, use, other = compile_rows("f(x) = x^2", "y = f(x - 1) + 1", "g(t) = 2t")
    assert defn.defines == "f"
    assert explicit(defn).fn.scalar(x=3.0) == 9
    assert explicit(use).fn.scalar(x=3.0) == 5
    assert other.defines == "g" and explicit(other).fn.scalar(x=3.0) == 6


def test_multi_parameter_functions_and_constants() -> None:
    _, _, row = compile_rows("h(a, b) = a^2 + b", "k = 2 * 3", "y = h(x, k)")
    assert explicit(row).fn.scalar(x=2.0) == 10


def test_constant_rows_show_computed_value() -> None:
    a, b = compile_rows("a = 4", "b = a / 8")
    assert a.message == "" and a.defines == "a"
    assert b.message == "= 0.5"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("k = 1.5", 1.5),
        ("k = -2", -2.0),
        ("k=2e3", 2000.0),
        ("k = 0", 0.0),
        ("b = 2a", None),
        ("c = 1/3", None),
        ("d = 2 + 1", None),
        ("e_1 = pi", None),
        ("f(x) = 2", None),
        ("y = 2", None),
        ("2", None),
        ("k = 1e400", None),  # infinite: no slider
    ],
)
def test_literal_constant_values(text: str, expected: float | None) -> None:
    _, row = compile_rows("a = 1", text)
    assert row.literal_value == expected
    if expected is not None:
        assert row.defines is not None and row.message == ""


def test_duplicate_or_broken_constants_have_no_literal_value() -> None:
    first, second = compile_rows("k = 1", "k = 2")
    assert first.is_error and first.literal_value is None and second.literal_value is None


def test_definition_order_does_not_matter() -> None:
    use, defn = compile_rows("y = f(x)", "f(x) = 2x")
    assert explicit(use).fn.scalar(x=1.0) == 2
    assert defn.defines == "f"


def test_function_parameters_are_local() -> None:
    _, f, use = compile_rows("a = 100", "f(a) = a + 1", "y = f(x) + a")
    assert explicit(use).fn.scalar(x=1.0) == 102
    assert not f.is_error


@pytest.mark.parametrize(
    ("text", "at_x", "expected"),
    [
        ("2pi x", 1.0, 2 * np.pi),
        ("x sin(x)", np.pi / 2, np.pi / 2),
        ("x(x + 1)", 2.0, 6.0),
        ("pi x", 1.0, np.pi),
    ],
)
def test_implicit_multiplication_of_adjacent_names(text: str, at_x: float, expected: float) -> None:
    assert explicit(compile_expression(text)).fn.scalar(x=at_x) == pytest.approx(expected)


def test_user_constants_split_too() -> None:
    _, row = compile_rows("a = 3", "y = a x")
    assert explicit(row).fn.scalar(x=2.0) == 6


def test_constant_times_group_is_multiplication() -> None:
    _, row = compile_rows("a = 3", "y = a(x + 1)")
    assert explicit(row).fn.scalar(x=1.0) == 6


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("y = q", "Unknown name 'q'"),
        ("foo(x)", "Unknown function 'foo'"),
        ("sin", "'sin' is a function"),
        ("sin(1, 2)", "wrong number of arguments"),
        ("log(1, 2, 3)", "wrong number of arguments"),
        ("y < q", "Unknown name 'q'"),
        ("y + 1", "need an equation"),
        ("point[1, 2, 3]", "exactly two"),
        ("point[x, 1]", "must be numbers"),
        ("y = point[1, 2] + x", "add a number and a point"),
        ("pi = 3", "built-in constant"),
        ("f(x, x) = 1", "must be different"),
        ("a = x + 1", "define a function instead"),
    ],
)
def test_resolution_errors(text: str, message: str) -> None:
    result = compile_expression(text)
    assert result.is_error
    assert message in result.message


def test_errors_carry_spans_for_highlighting() -> None:
    result = compile_expression("y = 2 + qq")
    assert result.error_span == (8, 10)


def test_invalid_rows_do_not_affect_unrelated_rows() -> None:
    good, bad, also_good = compile_rows("y = x", "y = sin(", "x^2 + y^2 = 4")
    assert bad.is_error
    assert isinstance(good.plot, ExplicitPlot)
    assert isinstance(also_good.plot, ImplicitPlot)


def test_errors_propagate_only_to_dependents() -> None:
    defn, user, bystander = compile_rows("f(x) = x + zz", "y = f(x)", "y = 2x")
    assert defn.is_error and "Unknown name 'zz'" in defn.message
    assert user.is_error and user.message == "'f' has an error"
    assert user.error_span == (4, 5)  # points at the reference in *this* row
    assert not bystander.is_error


def test_duplicate_definitions_are_errors() -> None:
    a1, a2, use = compile_rows("a = 1", "a = 2", "y = a x")
    assert a1.is_error and a2.is_error and "more than once" in a1.message
    assert use.is_error


def test_cycles_are_detected() -> None:
    f, g, use = compile_rows("f(x) = g(x)", "g(x) = f(x) + 1", "y = f(x)")
    assert "Circular definition: f → g → f" in f.message
    assert "Circular definition: g → f → g" in g.message
    assert use.is_error
    (self_ref,) = compile_rows("a = a + 1")
    assert "Circular" in self_ref.message


def test_wrong_arity_for_user_function() -> None:
    _, row = compile_rows("f(x) = x", "y = f(x, 2)")
    assert "takes 1 argument, got 2" in row.message


def test_function_used_without_call() -> None:
    _, row = compile_rows("f(x) = x", "y = f")
    assert "call it like f(x)" in row.message


def test_exponential_expansion_is_bounded() -> None:
    rows = ["f_0(x) = x + x"]
    rows += [f"f_{i}(x) = f_{i - 1}(x) + f_{i - 1}(x)" for i in range(1, 16)]
    rows.append("y = f_15(x)")
    results = compile_rows(*rows)
    assert "too large" in results[-2].message  # the f_15 definition itself
    assert results[-1].message == "'f_15' has an error"
    assert not results[5].is_error  # small enough definitions are fine


def test_deep_nesting_is_reported_not_crashing() -> None:
    result = compile_expression("(" * 3000 + "x" + ")" * 3000)
    assert result.is_error


def test_equal_resolved_rows_give_equal_plots() -> None:
    # Plot equality lets the UI keep cached geometry when unrelated rows change.
    a = compile_rows("f(x) = x^2", "y = f(x)")[1].plot
    b = compile_rows("f(x) = x^2", "y = f(x)", "y = 1")[1].plot
    c = compile_rows("f(x) = x^3", "y = f(x)")[1].plot
    assert a == b
    assert a != c


def test_symbolic_target_is_exposed_for_function_rows() -> None:
    target = compile_expression("x = y^2").symbolic
    assert target is not None and target.variable == "y"
    assert compile_expression("x^2 + y^2 = 1").symbolic is None


def test_values_are_evaluated_against_the_document() -> None:
    rows = [(1, "a = 3"), (2, "k = 1"), (3, "b = 2k")]
    values = [
        ("pi", "2pi", None),
        ("a", "a + 1", 2),
        ("own", "k", 2),
        ("indirect", "b", 2),
        ("x", "x", None),
        ("eq", "y = 1", None),
        ("nan", "sqrt(-1)", None),
        ("list", "list[1, 2]", None),
    ]
    results, evaluated = compile_with_values(rows, values)
    assert results[3].message == "= 2"  # the rows compile as usual
    assert evaluated["pi"] == pytest.approx(2 * math.pi)
    assert evaluated["a"] == 4
    messages = {k: v.message for k, v in evaluated.items() if isinstance(v, MathError)}
    assert messages == {
        "own": "Can't depend on k itself",
        "indirect": "'b' has an error: Unknown name 'k'",
        "x": "Must be a number, not depend on a variable",
        "eq": "Enter a value, not an equation",
        "nan": "Is not a finite real number",
        "list": "Must be a single number, not a list",
    }


@pytest.mark.parametrize(
    ("text", "point"),
    [
        ("point[1, 2] + point[3, 4]", (4, 6)),
        ("point[1, 2] - point[3, 5]", (-2, -3)),
        ("2point[1, 2]", (2, 4)),
        ("point[1, 2]2", (2, 4)),
        ("point[1, 2]/2", (0.5, 1)),
        ("-point[1, 2]", (-1, -2)),
        ("point[1, 2] * 3 - point[0, 1]", (3, 5)),
        ("(point[1, 2] + point[3, 4])/2", (2, 3)),
    ],
)
def test_points_are_vectors(text: str, point: tuple[float, float]) -> None:
    result = compile_expression(text)
    assert not result.is_error, result.message
    assert result.plot == PointPlot(*point)


def test_named_points_are_vectors() -> None:
    p, q, _, total, scaled, mid, named_scale = compile_rows(
        "p = point[1, 2]", "q = point[3, 4]", "a = 2", "p + q", "a(1, 2)", "m = (p + q)/2", "a p"
    )
    assert p.plot == PointPlot(1, 2) and p.defines == "p" and not p.is_error
    assert q.plot == PointPlot(3, 4)
    assert total.plot == PointPlot(4, 6)
    assert scaled.plot == PointPlot(2, 4)
    assert mid.plot == PointPlot(2, 3) and mid.defines == "m"
    assert named_scale.plot == PointPlot(2, 4)


def test_user_functions_take_points() -> None:
    _, row = compile_rows("f(t) = 2t", "f(point[1, 2]) + point[1, 1]")
    assert row.plot == PointPlot(3, 5)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("point[1, 2] + 1", "add a number and a point"),
        ("1 - point[1, 2]", "subtract a number and a point"),
        ("point[1, 2]point[3, 4]", "multiply two points"),
        ("point[1, 2] * point[3, 4]", "multiply two points"),
        ("2/point[1, 2]", "divide by a point"),
        ("point[1, 2]/point[1, 2]", "divide by a point"),
        ("point[1, 2]^2", "raise a point to a power"),
        ("2^point[1, 2]", "can't be an exponent"),
        ("sin(point[1, 2])", "'sin' needs a number, not a point"),
        ("|point[1, 2]|", "'abs' needs a number, not a point"),
        ("point[point[1, 2], 3]", "can't be points"),
        ("point[1, 2] + point[1, 2, 3]", "exactly two"),
        ("y = x + point[1, 2]", "add a number and a point"),
        ("y = 2point[1, 2]", "point can't be used here"),
        ("list[point[1, 2]] + point[1, 1]", "Points can't be list items"),
        ("Σ(point[k, 1], k, 1, 3)", "needs numbers, not points"),
        ("point[x, 1] + point[1, 1]", "must be numbers"),
    ],
)
def test_unsupported_point_operations(text: str, message: str) -> None:
    result = compile_expression(text)
    assert result.is_error
    assert message in result.message


def test_unsupported_point_operations_by_name() -> None:
    _, _, product, offset, power = compile_rows(
        "p = point[1, 2]", "q = point[3, 4]", "p q", "p + 1", "p^2"
    )
    assert "multiply two points" in product.message
    assert "add a number and a point" in offset.message
    assert "raise a point to a power" in power.message


def test_inequality_is_a_region_with_an_explicit_boundary() -> None:
    plot = compile_expression("y < x^2").plot
    assert isinstance(plot, RegionPlot)
    assert isinstance(plot.boundary, ExplicitPlot)
    assert plot.holds(np.array([2.0, 2.0]), np.array([3.0, 5.0])).tolist() == [True, False]


@pytest.mark.parametrize(
    ("text", "inside", "outside"),
    [
        ("y >= x", (0.0, 1.0), (1.0, 0.0)),
        ("y > x", (0.0, 1.0), (1.0, 1.0)),
        ("x <= 2", (2.0, 5.0), (3.0, 0.0)),
        ("x^2 + y^2 < 1", (0.0, 0.5), (1.0, 1.0)),
        ("1 > x", (0.0, 0.0), (2.0, 0.0)),
    ],
)
def test_inequality_membership(
    text: str, inside: tuple[float, float], outside: tuple[float, float]
) -> None:
    plot = compile_expression(text).plot
    assert isinstance(plot, RegionPlot)
    assert bool(plot.holds(np.array(inside[0]), np.array(inside[1])))
    assert not bool(plot.holds(np.array(outside[0]), np.array(outside[1])))


def test_inequality_sampling_and_dashed_boundary() -> None:
    view = ViewRect(-5, 5, -5, 5, 400, 400)
    plot = compile_expression("y < x").plot
    assert plot is not None
    strict = plot.sample(view)
    assert strict.dashed and strict.fill is not None
    assert strict.fill.mask.mean() == pytest.approx(0.5, abs=0.02)
    closed = compile_expression("y <= x").plot
    assert closed is not None and not closed.sample(view).dashed


def test_numeric_inequality_is_true_or_false() -> None:
    assert compile_expression("1 < 2").message == "true"
    assert compile_expression("3 <= 2").message == "false"
    assert compile_expression("2 >= 2").message == "true"


def test_inequality_with_a_constant_and_in_complex_mode() -> None:
    rows = compile_rows("a = 1", "y < a x")
    assert not rows[1].is_error and rows[1].plot is not None
    assert compile_expression("y < x", CompileOptions(complex_mode=True)).is_error


def test_parametric_curve() -> None:
    plot = compile_expression("parametric(cos(t), sin(t), t, [0, 2pi])").plot
    assert isinstance(plot, ParametricPlot)
    x, y = plot.xy(np.array([0.0, math.pi / 2]))
    assert np.allclose(x, [1, 0], atol=1e-12) and np.allclose(y, [0, 1])
    lines = plot.sample(ViewRect(-2, 2, -2, 2, 400, 400)).polylines
    assert len(lines) == 1 and np.allclose(np.hypot(lines[0][:, 0], lines[0][:, 1]), 1)


def test_parametric_range_and_constant_coordinate() -> None:
    rows = compile_rows("tmin = -1", "tmax = 3", "parametric(t, 2, t, [-1, 3])")
    plot = rows[2].plot
    assert isinstance(plot, ParametricPlot) and (plot.t_min, plot.t_max) == (-1, 3)
    x, y = plot.xy(np.array([0.0, 1.0]))
    assert y.tolist() == [2, 2]
    assert compile_rows("tmin = 5", "tmax = 1", "parametric(t, t, t, [5, 1])")[2].is_error


def test_t_stays_a_plain_name_when_defined_or_outside_points() -> None:
    rows = compile_rows("t = 2", "point[t, t^2]")
    assert isinstance(rows[1].plot, PointPlot)
    assert compile_expression("point[t, 1], point[2, 3]").is_error


def test_list_indexing() -> None:
    rows = compile_rows(
        "L = list[10, 20, 30, 40]", "L[2]", "L[2...3]", "L[list[4, 1]]", "list[5, 6, 7][3]"
    )
    assert rows[1].message == "= 20"
    assert rows[2].message == "= list[20, 30]"
    assert rows[3].message == "= list[40, 10]"
    assert rows[4].message == "= 7"


def test_list_indexing_in_plots_and_arithmetic() -> None:
    rows = compile_rows("k = list[1, 2, 3]", "y = k[2] x", "n = 2 k[3] + k[1]", "point[k[1], k[3]]")
    assert isinstance(rows[1].plot, ExplicitPlot)
    assert rows[1].plot.evaluate(np.array([3.0])).tolist() == [6.0]
    assert rows[3].plot is not None and not rows[3].is_error
    assert rows[2].message == "= 7"


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("L[0]", "out of range"),
        ("L[4]", "out of range"),
        ("L[1.5]", "whole number"),
        ("L[x]", "must be a number"),
        ("L[1...9]", "out of range"),
        ("a[1]", "Only a list can be indexed"),
    ],
)
def test_list_index_errors(text: str, message: str) -> None:
    rows = compile_rows("L = list[1, 2, 3]", "a = 5", text)
    assert rows[2].is_error and message in rows[2].message


def test_bracket_after_a_space_or_number_still_multiplies() -> None:
    assert compile_rows("a = 3", "a list[1, 2]")[1].message == "= list[3, 6]"
    assert compile_expression("2list[1, 2]").message == "= list[2, 4]"
