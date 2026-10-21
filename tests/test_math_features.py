"""Engine tests for the extended language: factorials and Γ, Σ/Π/∫, lists, tables,
points, equation auto-detection, angle units, complex mode and polar curves."""

from __future__ import annotations

import math

import numpy as np
import pytest

from opengraph.engine import symbolic
from opengraph.engine.analysis import (
    explicit_points,
    extrema,
    intersections,
    solve_equation,
    zeros,
)
from opengraph.engine.ast import Call, ListExpr, RangeExpr, TupleExpr
from opengraph.engine.builtins import factorial, gamma
from opengraph.engine.compiler import (
    CompileOptions,
    RowResult,
    TableColumn,
    TableInput,
    compile_document,
    compile_expression,
)
from opengraph.engine.errors import ParseError
from opengraph.engine.formatting import format_complex
from opengraph.engine.parser import parse
from opengraph.engine.plots import (
    ExplicitPlot,
    ImplicitPlot,
    PlotGroup,
    PointPlot,
    PointSetPlot,
    PolarPlot,
)
from opengraph.engine.sampling import ViewRect

DEGREES = CompileOptions(degrees=True)
COMPLEX = CompileOptions(complex_mode=True)
POLAR = CompileOptions(polar=True)


def value(text: str, options: CompileOptions | None = None) -> str:
    result = compile_expression(text, options)
    assert not result.is_error, result.message
    return result.message


def rows(*texts: str, options: CompileOptions | None = None) -> list[RowResult]:
    results = compile_document(list(enumerate(texts)), options)
    return [results[i] for i in range(len(texts))]


# -- parsing ----------------------------------------------------------------------------


def test_factorial_precedence() -> None:
    assert parse("5!") == Call("factorial", (parse("5"),))  # type: ignore[arg-type]
    assert value("3!^2") == "= 36"  # (3!)^2: ! binds tighter than the exponent's base
    assert value("2^3!") == "= 40320"  # (2^3)! — matches the editor's typeset reading
    assert value("-3!") == "= -6"
    assert value("2(3!)") == "= 12"


def test_lists_ranges_and_point_lists_parse() -> None:
    assert isinstance(parse("list[1, 2, 3]"), ListExpr)
    assert isinstance(parse("list[]"), ListExpr)
    assert isinstance(parse("list[1...5]"), RangeExpr)
    assert isinstance(parse("list[1…5]"), RangeExpr)  # typographic ellipsis
    points = parse("point[1, 2], point[3, 4]")
    assert isinstance(points, ListExpr) and all(isinstance(p, TupleExpr) for p in points.items)
    assert value("1.5 + 1") == "= 2.5"  # numbers with a decimal point still lex


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("list[1, 2", "Missing closing ']'"),
        ("point[1, 2], 3", "Separate points"),
        ("1]", "Unmatched"),
    ],
)
def test_list_parse_errors(text: str, message: str) -> None:
    with pytest.raises(ParseError, match=message):
        parse(text)


# -- special functions ------------------------------------------------------------------


def test_gamma_and_factorial_values() -> None:
    assert gamma(np.array([5.0]))[0] == pytest.approx(24)
    assert gamma(np.array([0.5]))[0] == pytest.approx(math.sqrt(math.pi), rel=1e-13)
    assert gamma(np.array([-0.5]))[0] == pytest.approx(-2 * math.sqrt(math.pi), rel=1e-13)
    assert np.isnan(gamma(np.array([0.0, -1.0, -2.0]))).all()  # poles are undefined
    assert factorial(np.array([20.0]))[0] == math.factorial(20)  # exact table
    assert factorial(np.array([170.0]))[0] == pytest.approx(float(math.factorial(170)))
    assert np.isinf(factorial(np.array([171.0]))[0])  # overflows, not a wrong number
    z = gamma(np.array([1 + 1j]))[0]
    assert z == pytest.approx(0.4980156681183560 - 0.1549498283018106j, rel=1e-12)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("5!", "= 120"),
        ("Γ(5)", "= 24"),
        ("Gamma(0.5)^2", "= 3.141592654"),
        ("nCr(5, 2)", "= 10"),
        ("nPr(5, 2)", "= 20"),
        ("0.5!", "= 0.8862269255"),
        ("(-1)!", "= undefined"),
    ],
)
def test_special_function_rows(text: str, expected: str) -> None:
    assert value(text) == expected


# -- sums, products, integrals --------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Σ(i, i, 1, 10)", "= 55"),
        ("sum(k^2, k, 1, 3)", "= 14"),
        ("Sigma(1/2^k, k, 0, 50)", "= 2"),
        ("prod(k, k, 1, 5)", "= 120"),
        ("Σ(k, k, 5, 1)", "= 0"),  # empty sum
        ("Π(k, k, 5, 1)", "= 1"),  # empty product
        ("∫(x^2, 0, 1)", "= 0.3333333333"),
        ("int(t^2, t, 0, 3)", "= 9"),
        ("integral(1/sqrt(t), t, 0, 1)", "= 2"),  # integrable endpoint singularity
        ("∫(e^(-t^2), t, -10, 10)^2", "= 3.141592654"),
        ("∫(x, 1, 0)", "= -0.5"),  # reversed bounds
    ],
)
def test_reductions(text: str, expected: str) -> None:
    assert value(text) == expected


def test_reduction_with_defined_bound_and_inference() -> None:
    _, total = rows("n = 10", "Σ(i, i, 1, n)")
    assert total.message == "= 55" and not total.is_error


def test_reductions_as_functions_of_x() -> None:
    integral = compile_expression("y = int(t^2, t, 0, x)").plot
    assert isinstance(integral, ExplicitPlot)
    xs = np.array([-1.0, 0.0, 2.0, 3.0])
    np.testing.assert_allclose(integral.evaluate(xs), xs**3 / 3, rtol=1e-10, atol=1e-14)
    series = compile_expression("y = Σ(x^k/k!, k, 0, 20)").plot
    assert isinstance(series, ExplicitPlot)
    np.testing.assert_allclose(series.evaluate(np.array([0.0, 1.0])), [1.0, math.e], rtol=1e-12)
    variable_bound = compile_expression("y = sum(1, k, 1, x)").plot
    assert isinstance(variable_bound, ExplicitPlot)
    np.testing.assert_allclose(variable_bound.evaluate(np.array([2.0, 5.0])), [2.0, 5.0])


def test_nested_reductions_and_user_functions_do_not_capture() -> None:
    _, g = rows("f(a) = Σ(a k, k, 1, 2)", "Σ(f(k), k, 1, 3)")
    assert g.message == "= 18"  # Σ_k 3k for k = 1..3, the inner k is a different variable


def test_reduction_errors() -> None:
    assert "Name the index" in compile_expression("Σ(x, 1, 3)").message
    assert "Name the index" in compile_expression("Σ(a b, 1, 3)").message
    assert "names the variable" in compile_expression("sum(k, 2, 1, 3)").message
    assert compile_expression("int(1/t, t, -1, 1)").message == "= undefined"  # divergent
    assert compile_expression("Σ(k, k, 1, 10^9)").message == "= undefined"  # too many terms
    assert "can't be redefined" in compile_expression("sum = 3").message


# -- lists ------------------------------------------------------------------------------------


def test_list_values_and_aggregates() -> None:
    assert value("list[1...5]") == "= list[1, 2, 3, 4, 5]"
    assert value("list[3...1]") == "= list[3, 2, 1]"
    assert value("list[1, 2, 3]^2") == "= list[1, 4, 9]"
    assert "equal lengths" in compile_expression("list[1, 2] + list[10, 20, 30]").message
    assert value("total(list[1, 2, 3])") == "= 6"
    assert value("mean(list[1...4])") == "= 2.5"
    assert value("length(list[])") == "= 0"
    assert value("max(list[4, 9, 2])") == "= 9"
    assert value("total(list[])") == "= 0"
    assert value("mean(list[])") == "= undefined"
    long = value("list[1...20]")
    assert long.endswith("… (20 items)]")


def test_list_definitions_broadcast_into_curves_and_points() -> None:
    k, curves, points, squares = rows("k = list[-1, 1, 2]", "y = k x", "point[k, k^2]", "k^2")
    assert k.message == "= list[-1, 1, 2]" and k.literal_value is None
    assert isinstance(curves.plot, PlotGroup) and len(curves.plot.plots) == 3
    assert curves.symbolic is None  # symbolic tools work on single expressions
    assert points.plot == PointSetPlot(((-1.0, 1.0), (1.0, 1.0), (2.0, 4.0)))
    assert squares.message == "= list[1, 1, 4]"


def test_list_edge_cases() -> None:
    empty, curve = rows("k = list[]", "y = k x")
    assert empty.message == "= list[]"
    assert curve.plot is None and "Empty list" in curve.message
    assert "contain other lists" in compile_expression("list[list[1], 2]").message
    assert "at most" in compile_expression("list[1...20000]").message
    assert "Too many curves" in compile_expression("y = list[1...200] x").message
    assert "Range bounds" in compile_expression("y = list[1...x]").message


def test_point_rows() -> None:
    assert compile_expression("point[1, 2]").plot == PointPlot(1.0, 2.0)
    several = compile_expression("point[1, 2], point[3, 4], point[5, 1/0]").plot
    assert isinstance(several, PointSetPlot) and len(several.points) == 3
    geometry = several.sample(ViewRect(-10, 10, -10, 10, 100, 100))
    assert geometry.points is not None and len(geometry.points) == 2  # undefined: skipped
    assert compile_expression("list[point[1, 2], point[3, 4]]").plot == PointSetPlot(
        ((1.0, 2.0), (3.0, 4.0))
    )


# -- tables -----------------------------------------------------------------------------------


def table(
    *columns: tuple[str, tuple[str, ...]], markers: bool = True, lines: bool = False
) -> TableInput:
    return TableInput(tuple(TableColumn(n, c) for n, c in columns), markers, lines)


def test_tables_plot_points_and_define_lists() -> None:
    data = table(("x_1", ("1", "2", "3", "")), ("y_1", ("2", "4", "k", "")), lines=True)
    results = compile_document([(0, "k = 6"), (1, "mean(y_1)"), (2, "y = x_1")], tables=[(3, data)])
    plot = results[3].plot
    assert plot == PointSetPlot(((1.0, 2.0), (2.0, 4.0), (3.0, 6.0)), True, True)
    assert results[1].message == "= 4"  # trailing blank cells are not data
    lines = results[2].plot  # y = x_1: one horizontal line per value, as in Desmos
    assert isinstance(lines, PlotGroup) and len(lines.plots) == 3


def test_table_errors_are_per_cell() -> None:
    data = table(("x_1", ("1", "2+", "3")), ("y_1", ("1", "2", "x")))
    result = compile_document([], tables=[(0, data)])[0]
    assert result.is_error and set(result.cell_errors) == {(1, 0), (2, 1)}
    assert result.message.startswith("Row 2, column 1")
    assert isinstance(result.plot, PointSetPlot)  # valid rows still plot
    assert len([p for p in result.plot.points if all(map(math.isfinite, p))]) == 1


def test_table_names_are_checked() -> None:
    bad = compile_document([], tables=[(0, table(("x 1", ("1",)), ("y_1", ("2",))))])[0]
    assert "not a valid name" in bad.message
    clash = compile_document([(0, "x_1 = 3")], tables=[(1, table(("x_1", ("1",)), ("y", ("2",))))])
    assert "more than once" in clash[0].message and "more than once" in clash[1].message


# -- equation auto-detection --------------------------------------------------------------------


def test_single_variable_equations_list_their_solutions() -> None:
    result = compile_expression("x^2 + x - 67 = 1")
    assert isinstance(result.plot, ImplicitPlot) and not result.is_error
    assert result.message.startswith("x ≈ -8.76136, 7.76136")
    assert result.symbolic is not None and result.symbolic.variable == "x"
    assert compile_expression("x^2 = 0").message.startswith("x ≈ 0")  # tangential root
    assert "No roots found in numerical search" in compile_expression("x^2 = -1").message
    many = compile_expression("sin(x) = 0.5").message
    assert "…" in many and "0.523599" in many and "numerical search" in many
    assert compile_expression("y^2 = 4").message.startswith("y ≈ -2, 2")


def test_linear_equations_are_rearranged() -> None:
    line = compile_expression("2y + 3x = 6").plot
    assert isinstance(line, ExplicitPlot) and line.independent == "x"
    np.testing.assert_allclose(line.evaluate(np.array([0.0, 2.0])), [3.0, 0.0])
    sideways = compile_expression("y^2 + 2x = 4").plot
    assert isinstance(sideways, ExplicitPlot) and sideways.independent == "y"
    vertical = compile_expression("2x = 3")
    assert vertical.message == "x = 1.5" and isinstance(vertical.plot, ExplicitPlot)
    assert isinstance(compile_expression("x y = 0").plot, ImplicitPlot)  # B depends on x
    assert compile_expression("sin(x) = x").message.startswith("x ≈ 0")


def test_equations_solve_for_an_unknown_name() -> None:
    result = compile_expression("2a + 3 = 7")
    assert result.message == "a = 2" and not result.is_error and result.plot is None
    assert result.solved == ("a", (2.0,))
    assert result.symbolic is not None and result.symbolic.variable == "a"
    assert compile_expression("k^3 = 8").message.startswith("k ≈ 2")
    assert compile_expression("t^2 = 2").message.startswith("t ≈ -1.41421, 1.41421")
    assert compile_expression("a(2 + 1) = 6").message == "a = 2"  # a times the group
    assert "numerical search" in compile_expression("b^2 = -1").message
    several = compile_expression("a + b = 3")
    assert several.is_error and "a and b" in several.message
    assert compile_expression("y = a x").message == "Unknown name 'a'"  # a curve, not solved
    assert compile_expression("a + 1").message == "Unknown name 'a'"  # not an equation


def test_equation_solutions_do_not_define_names_in_other_rows() -> None:
    results = compile_document([(0, "y = a x"), (1, "2a + 3 = 7"), (2, "b = 3a")])
    assert results[1].message == "a = 2"
    assert results[0].is_error and results[2].is_error
    defined = compile_document([(0, "a=2"), (1, "y=a x")])
    assert isinstance(defined[1].plot, ExplicitPlot)
    assert defined[1].plot.evaluate(3.0) == 6


def test_exact_solutions_update_feedback_without_binding_names() -> None:
    rows = [(0, "t^2=2"), (1, "y=t x"), (2, "2v=1")]
    initial = compile_document(rows)
    target = initial[0].symbolic
    assert target is not None
    exact = {target: (("-sqrt(2)", -math.sqrt(2)), ("sqrt(2)", math.sqrt(2)))}
    completed = compile_document(rows, exact_solutions=exact)
    assert completed[0].message == "t = list[-sqrt(2), sqrt(2)]  (≈ list[-1.41421, 1.41421])"
    assert initial[1].is_error and completed[1].is_error
    assert completed[2].message == "v = 0.5"


def test_rows_with_undefined_names_offer_to_solve_for_them() -> None:
    line = compile_expression("(rho*sin(k)-w)/(rho*cos(k)-q)*(x-q)+w")
    assert line.is_error and line.message == "Unknown name 'rho'"
    assert line.solve_for == ("rho", "k", "w", "q")  # in order of appearance
    assert line.rearrange is not None and line.rearrange.variable == "rho"
    assert compile_expression("y = a x + b").solve_for == ("a", "b")
    assert compile_expression("a + b = 3").solve_for == ("a", "b")
    assert compile_expression("x^2 + y^2 = rho^2").solve_for == ("rho",)
    # Nothing to offer: solved already, no x for a bare expression, points, parse errors.
    assert compile_expression("2a + 1 = 5").solve_for == ()
    assert compile_expression("a + 1").solve_for == ()
    assert compile_expression("point[a, 1]").solve_for == ()
    assert compile_expression("y = sin(x").solve_for == ()
    assert compile_expression("y = x^2").solve_for == ()


def test_rows_report_their_undefined_names() -> None:
    def undefined(text: str) -> tuple[str, ...]:
        return compile_expression(text).undefined

    assert undefined("(rho*sin(k)-w)/(rho*cos(k)-q)*(x-q)+w") == ("rho", "k", "w", "q")
    assert undefined("a + 1") == ("a",)  # wider than solve_for: values count too
    assert undefined("point[a, b]") == ("a", "b")
    assert undefined("point[1, 2], point[c, d]") == ("c", "d")
    assert undefined("f(x) = m x + c") == ("m", "c")  # and definitions
    assert undefined("b = 2a") == ("a",)
    assert undefined("y = 2x") == ()
    assert undefined("2a + 1 = 5") == ()  # solved, so a is defined
    assert undefined("y = sin(x") == ()


def test_adjacent_letters_remain_one_identifier() -> None:
    line = compile_expression("y = ax + b")
    assert line.message == "Unknown name 'ax'"
    assert line.undefined == ("ax", "b") and line.solve_for == ("ax", "b")
    assert compile_expression("y = abx").undefined == ("abx",)
    assert compile_expression("3ka = 6").message == "ka = 2"
    results = compile_document([(0, "y = ax"), (1, "a = 2")])
    assert results[0].is_error


def test_unknowns_solved_twice_or_ambiguously_stay_undefined() -> None:
    twice = compile_document([(0, "a + 1 = 3"), (1, "2a = 6"), (2, "a")])
    assert twice[0].message == "a = 2" and twice[1].message == "a = 3"
    assert twice[2].message == "Unknown name 'a'"
    two_roots = compile_document([(0, "a^2 = 4"), (1, "a")])
    assert two_roots[0].message.startswith("a ≈ -2, 2")
    assert two_roots[1].is_error
    defined = compile_document([(0, "a = 5"), (1, "2a = 10")])
    assert defined[1].message == "true"  # a known name is not solved for


def test_constant_equations_are_true_or_false() -> None:
    assert compile_expression("2 + 2 = 4").message == "true"
    assert compile_expression("1 = 2").message == "false"
    assert not compile_expression("1 = 2").is_error


# -- angle units, complex mode, polar ------------------------------------------------------------


def test_degree_mode_applies_to_trig_and_inverse_trig() -> None:
    assert value("sin(90)", DEGREES) == "= 1"
    assert value("cos(180)", DEGREES) == "= -1"
    assert value("asin(1)", DEGREES) == "= 90"
    assert value("atan(1, 1)", DEGREES) == "= 45"  # the angle of the point (1, 1)
    assert value("sin(pi/2)") == "= 1"  # radians by default
    plot = compile_expression("y = sin(x)", DEGREES).plot
    radians = compile_expression("y = sin(x)").plot
    assert plot != radians  # the canvas must resample when the unit changes


def test_complex_mode_values_points_and_curves() -> None:
    assert value("(1 + 2i)(3 - i)", COMPLEX) == "= 5 + 5i"
    assert value("sqrt(-4)", COMPLEX) == "= 2i"
    assert value("e^(i pi)", COMPLEX) == "= -1"
    assert value("abs(3 + 4i)", COMPLEX) == "= 5"
    assert value("re(2 - 3i) + im(2 - 3i)", COMPLEX) == "= -1"
    assert value("conj(1 + i)", COMPLEX) == "= 1 - i"
    assert compile_expression("1 + 2i", COMPLEX).plot == PointPlot(1.0, 2.0)
    roots = compile_expression("e^(2 pi i list[0...3]/4)", COMPLEX).plot
    assert isinstance(roots, PointSetPlot) and len(roots.points) == 4
    curve = compile_expression("y = sqrt(x)", COMPLEX).plot
    assert isinstance(curve, ExplicitPlot)
    values = curve.evaluate(np.array([-4.0, 4.0]))
    assert np.isnan(values[0]) and values[1] == 2  # only real values are drawn
    assert "Unknown name 'i'" in compile_expression("1 + 2i").message  # real mode
    assert "imaginary unit" in compile_expression("i = 2", COMPLEX).message
    assert value("Σ(i, i, 1, 4)", COMPLEX) == "= 10"  # a bound i shadows the unit


def test_format_complex() -> None:
    assert format_complex(3 + 0j) == "3"
    assert format_complex(-1j) == "-i"
    assert format_complex(2 - 0.5j) == "2 - 0.5i"
    assert format_complex(complex(math.nan, 0)) == "undefined"


def test_polar_curves() -> None:
    cardioid = compile_expression("r = 1 + cos(θ)").plot
    assert isinstance(cardioid, PolarPlot)
    x, y = cardioid.xy(np.array([0.0, math.pi / 2]))
    np.testing.assert_allclose([x[0], y[0], x[1], y[1]], [2, 0, 0, 1], atol=1e-12)
    assert isinstance(compile_expression("r = theta").plot, PolarPlot)
    # A constant assignment defines r on every grid.
    assert compile_expression("r = 2").defines == "r"
    assert compile_expression("r = 2").plot is None
    assert rows("theta = 1", "r = theta")[0].is_error
    assert compile_expression("r = 2", POLAR).defines == "r"
    assert isinstance(compile_expression("r^2 = cos(2θ)", POLAR).plot, ImplicitPlot)
    assert "polar coordinate" in compile_expression("θ = 2", POLAR).message
    assert "need an equation" in compile_expression("sin(θ)").message
    degrees = compile_expression("r = θ", CompileOptions(polar=True, degrees=True)).plot
    assert isinstance(degrees, PolarPlot) and degrees.radius(math.pi) == pytest.approx(180)


def test_polar_sampling_draws_a_closed_circle() -> None:
    plot = compile_expression("r = 2 + 0theta", POLAR).plot
    assert plot is not None
    geometry = plot.sample(ViewRect(-4, 4, -3, 3, 800, 600))
    points = np.vstack(geometry.polylines)
    np.testing.assert_allclose(np.hypot(points[:, 0], points[:, 1]), 2.0, rtol=1e-12)


# -- points of interest -------------------------------------------------------------------------


def test_zeros_reject_poles_and_jumps() -> None:
    ts = np.linspace(-3, 3, 601)
    assert zeros(lambda t: 1 / t, ts) == []
    assert zeros(lambda t: np.where(t < 0.123, -1.0, 1.0), ts) == []  # a jump, no zero
    roots = zeros(lambda t: t**2 - 2, ts)
    assert roots == pytest.approx([-math.sqrt(2), math.sqrt(2)], abs=1e-12)


def test_extrema_reject_poles() -> None:
    ts = np.linspace(-3, 3, 600)
    assert extrema(np.tan, ts) == []
    (peak,) = extrema(lambda t: -((t - 0.3) ** 2) + 1, ts)
    # Near a smooth peak f varies as δ², so the location is only known to ~sqrt(eps).
    assert peak.kind == "maximum" and peak.t == pytest.approx(0.3, abs=1e-7)
    assert not peak.touches_zero


def test_explicit_points_of_interest() -> None:
    plot = compile_expression("y = x^3 - 3x").plot
    assert isinstance(plot, ExplicitPlot)
    found = explicit_points(plot, np.linspace(-5, 5, 2000))
    kinds = {(p.kind, round(p.x, 6), round(p.y, 6)) for p in found}
    root = round(math.sqrt(3), 6)
    assert ("x-intercept", -root, 0.0) in kinds and ("x-intercept", root, 0.0) in kinds
    assert ("maximum", -1.0, 2.0) in kinds and ("minimum", 1.0, -2.0) in kinds
    assert ("y-intercept", 0.0, 0.0) in kinds
    double = compile_expression("y = (x - 1)^2").plot
    assert isinstance(double, ExplicitPlot)
    assert any(
        p.kind == "x-intercept" and p.x == pytest.approx(1)
        for p in explicit_points(double, np.linspace(-5, 5, 2000))
    )


def test_intersections() -> None:
    parabola, line, circle = (
        compile_expression(t).plot for t in ("y = x^2", "y = x + 2", "x^2 + y^2 = 4")
    )
    assert parabola is not None and line is not None and circle is not None
    xs = np.linspace(-5, 5, 2000)
    found = sorted((p.x, p.y) for p in intersections(parabola, line, xs, xs))
    assert found == [pytest.approx((-1, 1)), pytest.approx((2, 4))]
    on_circle = intersections(line, circle, xs, xs)
    assert {round(p.x, 6) for p in on_circle} == {-2.0, 0.0}


def test_solve_equation_scan() -> None:
    solutions, truncated = solve_equation(lambda t: t**3 - t)
    assert solutions == pytest.approx([-1, 0, 1], abs=1e-12) and not truncated
    solutions, truncated = solve_equation(np.sin)
    assert truncated and 0.0 in solutions


# -- symbolic -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "options", "expected"),
    [
        ("∫(x^2, 0, 1)", None, "1/3"),
        ("Σ(k^2, k, 1, 10)", None, "385"),
        ("5!/3!", None, "20"),
        ("(1 + 2i)^2", COMPLEX, "-3 + 4*i"),
    ],
)
def test_exact_values(text: str, options: CompileOptions | None, expected: str) -> None:
    target = compile_expression(text, options).symbolic
    assert target is not None and target.value
    assert symbolic.exact_value(target) == expected


def test_symbolic_results_keep_the_angle_unit() -> None:
    target = compile_expression("y = sin(x)", DEGREES).symbolic
    assert target is not None
    text = symbolic.derivative(target)
    assert text == "pi*cos(x)/180"  # cos of x degrees, re-read in degree mode
    check = compile_expression(f"y = {text}", DEGREES).plot
    assert isinstance(check, ExplicitPlot)
    assert check.evaluate(0.0) == pytest.approx(math.pi / 180)


def test_symbolic_handles_integrals_and_unsupported_results() -> None:
    target = compile_expression("y = int(t^2, t, 0, x)").symbolic
    assert target is not None and symbolic.simplify(target).startswith("guard(x^3/3, ")
    factorial_target = compile_expression("y = x!").symbolic
    assert factorial_target is not None
    with pytest.raises(symbolic.SymbolicError, match="polygamma"):
        symbolic.derivative(factorial_target)
    roots = compile_expression("x^2 + x - 67 = 1").symbolic
    assert roots is not None
    assert [r.exact for r in symbolic.solve(roots)] == ["-sqrt(273)/2 - 1/2", "-1/2 + sqrt(273)/2"]


@pytest.mark.parametrize(
    "text",
    [
        "k = list[-1, 1]",
        "5!",
        "x^2!",
        "1/2!",
        "Σ(i, i, 1, n)",
        "y = Σ(x^k/k!, k, 0, 6)",
        "∫(x^2, 0, 1)",
        "list[1...5]",
        "list[1/2, 3]",
        "point[1, 2], point[3, 4]",
        "Gamma(x)",
    ],
)
def test_typeset_editor_round_trips_new_syntax(text: str) -> None:
    from opengraph.ui.mathedit.source import from_source, to_source

    tree, _ = from_source(text)
    assert parse(to_source(tree)[0]) == parse(text)


def test_formula_columns_are_computed_row_by_row() -> None:
    data = table(
        ("x_1", ("1", "2", "3")),
        ("y_1 = x_1^2 + a", ("ignored", "", "")),
        ("z_1 = y_1 - x_1", ()),
    )
    results = compile_document(
        [(0, "a = 1"), (1, "sum(z_1)")] if False else [(0, "a = 1"), (1, "total(z_1)")],
        tables=[(2, data)],
    )
    result = results[2]
    assert not result.is_error, result.message
    assert isinstance(result.plot, PlotGroup)
    first, second = result.plot.plots
    assert isinstance(first, PointSetPlot) and isinstance(second, PointSetPlot)
    assert first.points == ((1.0, 2.0), (2.0, 5.0), (3.0, 10.0))
    assert second.points == ((1.0, 1.0), (2.0, 3.0), (3.0, 7.0))
    assert result.table_values[(2, 1)] == "10" and result.table_values[(0, 2)] == "1"
    assert results[1].message == "= 11"  # a formula column is a list other rows can use


def test_formula_column_with_a_constant_and_in_a_plot_only_column() -> None:
    data = table(("x_1", ("1", "2")), ("y = 5", ()))
    result = compile_document([], tables=[(0, data)])[0]
    assert isinstance(result.plot, PointSetPlot) and result.plot.points == ((1.0, 5.0), (2.0, 5.0))


@pytest.mark.parametrize(
    ("header", "message"),
    [
        ("y_1 = ", "enter a formula"),
        ("y_1 = 2+", "Column 2"),
        ("y_1 = q", "Unknown name 'q'"),
        ("y_1 = y_1 + 1", "Column 2"),
        ("y_1 = x", "other columns"),
    ],
)
def test_formula_column_errors(header: str, message: str) -> None:
    data = table(("x_1", ("1", "2")), (header, ()))
    result = compile_document([], tables=[(0, data)])[0]
    assert result.is_error and message in result.message
