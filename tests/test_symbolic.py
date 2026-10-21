from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from opengraph.engine import symbolic
from opengraph.engine.compiler import (
    CompileOptions,
    GeneralSolution,
    SymbolicTarget,
    compile_document,
    compile_expression,
)
from opengraph.engine.plots import ExplicitPlot, PointSetPlot

pytest.importorskip("sympy")


def target(text: str) -> SymbolicTarget:
    result = compile_expression(text)
    assert result.symbolic is not None, result.message
    return result.symbolic


def numeric(text: str, var: str = "x") -> ExplicitPlot:
    plot = compile_expression(text if var == "x" else f"x = {text}").plot
    assert isinstance(plot, ExplicitPlot)
    return plot


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("x^3 - 2x", "3*x^2 - 2"),
        ("sin(x)", "cos(x)"),
        ("ln(x)", "1/x"),
        ("e^(2x)", "2*exp(2*x)"),
        ("pi x", "pi"),
        ("abs(x)", "sign(x)"),
    ],
)
def test_derivative(text: str, expected: str) -> None:
    actual = symbolic.derivative(target(text))
    assert actual == expected or actual.startswith(f"guard({expected}, ")


@pytest.mark.parametrize(
    "text", ["x^3 - 2x", "sqrt(x^2 + 1)", "log(x)", "x sin(x)", "atan(x) / (1 + x^2)", "cbrt(x)"]
)
def test_derivative_agrees_with_finite_differences(text: str) -> None:
    f = numeric(text).fn
    d = numeric(symbolic.derivative(target(text))).fn
    xs = np.array([0.3, 0.7, 1.9, 3.1])
    h = 1e-6
    approx = (f(x=xs + h) - f(x=xs - h)) / (2 * h)
    np.testing.assert_allclose(d(x=xs), approx, rtol=1e-5)


@pytest.mark.parametrize("text", ["sum(1/(n^x), n, 1, 10000)", "sum(x^n/n!, n, 0, 10000)"])
def test_long_sums_stay_unevaluated_and_plot(text: str) -> None:
    # Written out term by term, 10000 terms made every operation hang.
    derivative = symbolic.derivative(target(text))
    assert derivative.startswith("guard(sum(")
    f = numeric(text).fn
    d = numeric(derivative).fn
    xs = np.array([1.5, 2.0, 3.0])
    h = 1e-6
    approx = (f(x=xs + h) - f(x=xs - h)) / (2 * h)
    np.testing.assert_allclose(d(x=xs), approx, rtol=1e-5)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("sum(n^2, n, 1, 10000)", "333383335000"),  # closed form
        ("prod(1 + 1/n, n, 1, 10000)", "10001"),
    ],
)
def test_long_sums_with_small_closed_forms_are_exact(text: str, expected: str) -> None:
    assert symbolic.exact_value(target(text)) == expected


def test_long_sum_without_small_closed_form_has_no_exact_value() -> None:
    # harmonic(10000, 2) is a fraction tens of thousands of digits long.
    with pytest.raises(symbolic.SymbolicError):
        symbolic.exact_value(target("sum(1/n^2, n, 1, 10000)"))


def test_long_sum_is_solved_numerically() -> None:
    roots = symbolic.solve(target("sum(1/(n^x), n, 1, 10000) - 2"))
    assert isinstance(roots, list) and len(roots) == 1
    assert roots[0].value == pytest.approx(1.7277205444, rel=1e-9)


def test_derivative_with_respect_to_y() -> None:
    assert symbolic.derivative(target("x = y^2")) == "2*y"


def test_simplify() -> None:
    assert symbolic.simplify(target("sin(x)^2 + cos(x)^2")) == "1"
    assert symbolic.simplify(target("(x^2 - 1)/(x - 1)")).startswith("guard(x + 1, ")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("x^2 + 2*x = 0", "x*(x + 2)"),
        ("y^2 + 2*y = 0", "y*(y + 2)"),
        ("sin(x)^2 + cos(x)^2 = 1", "0"),
    ],
)
def test_simplify_equation_without_substitution(text: str, expected: str) -> None:
    assert symbolic.simplify(target(text)) == expected


def test_simplify_equation_keeps_parameter_and_domain() -> None:
    result = compile_document([(0, "a = 2"), (1, "(x^2 - a^2)/(x - a) = 0")])[1]
    assert result.symbolic is not None
    text = symbolic.simplify(result.symbolic)
    assert text.startswith("guard(a + x, ")
    # Changing the parameter must still move both the expression and its hole.
    rows = compile_document([(0, "a = 3"), (1, f"y = {text}")])
    assert isinstance(rows[1].plot, ExplicitPlot)
    assert np.isnan(rows[1].plot.evaluate(3.0))
    assert rows[1].plot.evaluate(-3.0) == pytest.approx(0)


@pytest.mark.parametrize("complex_mode", [False, True])
@pytest.mark.parametrize("projection, expected", [("re", "cos(x)"), ("im", "sin(x)")])
def test_simplify_menu_euler_identity(projection: str, expected: str, complex_mode: bool) -> None:
    options = CompileOptions(complex_mode=complex_mode)
    original = compile_expression(f"{projection}(e^ix)", options)
    assert not original.is_error, original.message
    assert original.symbolic is not None
    text = symbolic.simplify(original.symbolic)
    assert text == expected
    result = compile_expression(text, options)
    assert isinstance(original.plot, ExplicitPlot) and isinstance(result.plot, ExplicitPlot)
    xs = np.array([-3.0, -0.7, 0.0, 0.7, 3.0])
    np.testing.assert_allclose(original.plot.evaluate(xs), result.plot.evaluate(xs), atol=1e-14)


def test_complex_projection_scope_does_not_enable_complex_mode_for_other_rows() -> None:
    results = compile_document([(0, "re(e^ix)"), (1, "i"), (2, "sqrt(-1)")])
    assert not results[0].is_error
    assert results[1].is_error and results[1].message == "Unknown name 'i'"
    assert results[2].message == "= undefined"


def test_decimals_become_exact_rationals() -> None:
    assert symbolic.simplify(target("0.1x + 0.2x")) == "3*x/10"


def test_user_definitions_are_inlined() -> None:
    from opengraph.engine.compiler import compile_document

    results = compile_document([(0, "f(x) = x^2"), (1, "y = 3f(x)")])
    assert results[1].symbolic is not None
    assert symbolic.derivative(results[1].symbolic) == "6*x"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("y = a x + a x", "2*a*x"),
        ("a x + a x", "2*a*x"),
        ("g(x) = (a x^2 - a)/(x - 1)", "a*(x + 1)"),
        ("x = a y + y", "y*(a + 1)"),
        ("y = 3f(x) + a", "a + 3*x^2"),  # functions are still expanded
        ("2y = a x", "a*x/2"),  # equations are solved without inlining a
        ("a y = x + a", "(a + x)/a"),
        ("y^2 = a x", "y^2/a"),  # solved for x
    ],
)
def test_simplify_keeps_constants_as_names(text: str, expected: str) -> None:
    results = compile_document([(0, "a = 2"), (1, "f(x) = x^2"), (2, text)])
    assert results[2].symbolic is not None, results[2].message
    actual = symbolic.simplify(results[2].symbolic)
    assert actual == expected or actual.startswith(f"guard({expected}, ")


@pytest.mark.parametrize(
    ("operation", "text", "expected"),
    [
        (symbolic.derivative, "y = a x^2 + b", "2*a*x"),
        (symbolic.derivative, "f(x) = sin(a x)", "a*cos(a*x)"),
        (symbolic.derivative, "2y = a x^2", "a*x"),  # an equation solved for y
        (symbolic.integral, "y = a x", "x^2"),
        (symbolic.integral, "y = x^b", "x^4/4"),  # general case, not b = 3
        (symbolic.integral, "y = e^(a x)", "exp(2*x)/2"),
        (symbolic.derivative, "y = x^b", "b*x^(b - 1)"),
    ],
)
def test_calculus_keeps_constants_as_names(operation, text: str, expected: str) -> None:
    results = compile_document([(0, "a = 2"), (1, "b = 3"), (2, text)])
    assert results[2].symbolic is not None, results[2].message
    actual = operation(results[2].symbolic)
    assert actual == expected or actual.startswith(f"guard({expected}, ")


def test_calculus_never_substitutes_values() -> None:
    # Without a form that keeps the constants as names there is no result, rather
    # than one computed from the constants' current values.
    results = compile_document([(0, "a = 2"), (1, "y = a x")])
    assert results[1].symbolic is not None
    unnamed = replace(results[1].symbolic, named=None)
    for operation in (symbolic.simplify, symbolic.derivative):
        with pytest.raises(symbolic.SymbolicError, match="without substituting"):
            operation(unnamed)


def test_solve_finite_roots_sorted() -> None:
    roots = symbolic.solve(target("x^3 - 2x"))
    assert [r.exact for r in roots] == ["-sqrt(2)", "0", "sqrt(2)"]
    assert roots[0].value == pytest.approx(-math.sqrt(2))


def test_solve_reports_no_real_roots_and_intervals() -> None:
    assert symbolic.solve(target("x^2 + 1")) == []
    with pytest.raises(symbolic.SymbolicError, match="interval"):
        symbolic.solve(target("abs(x) - x"))


def families(text: str, **options: bool) -> list[str]:
    result = compile_expression(text, CompileOptions(**options))
    assert result.symbolic is not None
    general = symbolic.solve(result.symbolic)
    assert isinstance(general, GeneralSolution)
    return [text for text, _ in general.roots] + [f.text() for f in general.families]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("sin(x) - 1/2", ["pi/6 + 2*pi*n", "5*pi/6 + 2*pi*n"]),
        ("sin(x)", ["pi*n"]),  # 2πn and π + 2πn merge
        ("cos(x)", ["pi/2 + pi*n"]),
        ("tan(x)", ["pi*n"]),
        ("(x - 1) sin(x)", ["1", "pi*n"]),  # an isolated root besides the family
        ("sin(x)^2 - 1/4", ["pi/6 + pi*n", "5*pi/6 + pi*n"]),
    ],
)
def test_solve_gives_periodic_families(text: str, expected: list[str]) -> None:
    assert families(text) == expected


def test_periodic_families_in_degrees() -> None:
    assert families("sin(x) - 1/2", degrees=True) == ["30 + 360*n", "150 + 360*n"]


def test_periodic_solution_values_and_text() -> None:
    general = symbolic.solve(target("sin(x) - 1/2"))
    assert isinstance(general, GeneralSolution)
    assert general.values(0, 7) == pytest.approx([math.pi / 6, 5 * math.pi / 6, 13 * math.pi / 6])
    assert general.text("x") == "x = pi/6 + 2*pi*n, 5*pi/6 + 2*pi*n  (n ∈ ℤ)"
    assert general.text("n").endswith("(k ∈ ℤ)")


def test_half_line_families_are_not_periodic() -> None:
    # sqrt(x) sin(x) has the roots πn for n >= 0 only: not a family over all integers.
    assert not isinstance(symbolic.solve(target("sqrt(x) (sin(x) + 2)")), GeneralSolution)


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("sqrt(x) sin(x)", [k * math.pi for k in range(7)]),
        ("sin(sqrt(x))", [(k * math.pi) ** 2 for k in range(4)]),
        ("sin(x)/x", [-3 * math.pi, -2 * math.pi, -math.pi, math.pi, 2 * math.pi, 3 * math.pi]),
    ],
)
def test_exact_generators_respect_domains(expression: str, expected: list[float]) -> None:
    general = symbolic.solve(target(expression))
    assert isinstance(general, GeneralSolution)
    assert general.generators and not general.families
    hi = 100.0 if expression == "sin(sqrt(x))" else 20.0
    lo = -10.0 if expression == "sin(x)/x" else 0.0
    found = general.values(lo, hi)
    if expression == "sin(x)/x":
        found = [v for v in found if abs(v) <= 10]
    assert found == pytest.approx(expected)
    for generator in general.generators:
        result = compile_expression(generator.source())
        assert not result.is_error, result.message


def test_generator_can_have_an_accumulation_point_without_including_a_pole() -> None:
    general = symbolic.solve(target("sin(1/x)"))
    assert isinstance(general, GeneralSolution) and general.generators
    found = general.values(-1, 1)
    assert found and 0 not in found
    assert max(found) == pytest.approx(1 / math.pi)
    assert min(found) == pytest.approx(-1 / math.pi)


def test_nonlinear_generators_in_degrees_and_parameter_name_collision() -> None:
    result = compile_expression("sin(sqrt(n)) = 0", CompileOptions(degrees=True))
    assert result.symbolic is not None
    general = symbolic.solve(result.symbolic)
    assert isinstance(general, GeneralSolution) and general.generators
    assert general.values(0, 130000) == pytest.approx([0, 180**2, 360**2])
    assert "for k in sets[N]" in general.text("n")


def test_domain_check_is_global_not_just_near_zero() -> None:
    general = symbolic.solve(target("sqrt(x + 100) sin(x)"))
    assert isinstance(general, GeneralSolution) and general.generators
    assert not general.values(-200, -101)
    assert general.values(-101, -90) == pytest.approx(
        [-100, -31 * math.pi, -30 * math.pi, -29 * math.pi]
    )


def test_infinite_solution_feedback_never_defines_a_preview() -> None:
    texts = [(0, "sin(t) = 1/2"), (1, "point[t, 0]"), (2, "t[3]")]
    initial = compile_document(texts)
    target = initial[0].symbolic
    assert target is not None
    general = symbolic.solve(target)
    assert isinstance(general, GeneralSolution)
    results = compile_document(texts, exact_solutions={target: general})
    assert "ℤ" in results[0].message
    assert results[1].is_error and results[2].is_error


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("sqrt(x)(x + 1)", [0.0]),  # sqrt(-1)·0 is not a point of the curve
        ("ln(x)(x + 2)", [1.0]),
        ("sqrt(x - 3)(x - 1)", [3.0]),
    ],
)
def test_solve_drops_roots_outside_the_domain(text: str, expected: list[float]) -> None:
    assert [r.value for r in symbolic.solve(target(text))] == pytest.approx(expected)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("x^(1/3) - x", [-1.0, 0.0, 1.0]),  # real cube root, as plotted
        ("x^(1/3) + 2", [-8.0]),
        ("x^(2/3) - 4", [-8.0, 8.0]),
        ("x^(5/3) + 32", [-8.0]),
    ],
)
def test_solve_uses_real_odd_roots(text: str, expected: list[float]) -> None:
    assert [r.value for r in symbolic.solve(target(text))] == pytest.approx(expected)


def test_unrepresentable_results_raise() -> None:
    with pytest.raises(symbolic.SymbolicError):
        symbolic.derivative(target("floor(x)"))
    with pytest.raises(symbolic.SymbolicError):
        symbolic.derivative(target("x!"))  # polygamma


@pytest.mark.parametrize(
    "text",
    [
        "x^(1/3) + 2",
        "ln(x)/ln(10)",
        "mod(x, 3)",
        "min(x, 2)",
        "abs(x - 1)",
        "ceil(x)",
        "exp(-x^2/2)",
        "e^x + pi",
        "tan(x)^2",
        "asin(x)",
    ],
)
def test_printer_round_trips_through_parser(text: str) -> None:
    printed = symbolic.to_source(symbolic.to_sympy(target(text).node))
    original = numeric(text).fn
    reparsed = numeric(printed).fn
    xs = np.array([0.15, 0.5, 0.9, 2.4])
    np.testing.assert_allclose(reparsed(x=xs), original(x=xs), rtol=1e-12, equal_nan=True)


@pytest.mark.parametrize(
    ("text", "expected"),
    [("3x^2", "x^3"), ("cos(x)", "sin(x)"), ("1/x", "ln(abs(x))")],
)
def test_integral(text: str, expected: str) -> None:
    result = numeric(symbolic.integral(target(text)))
    xs = np.array([0.5, 1.0, 2.0])
    np.testing.assert_allclose(result.evaluate(xs), numeric(expected).evaluate(xs))


def test_integral_without_closed_form() -> None:
    with pytest.raises(symbolic.SymbolicError):
        symbolic.integral(target("x^x"))


def rearranged(text: str, name: str) -> str:
    result = compile_expression(text)
    assert result.rearrange is not None, result.message
    return symbolic.rearrange(replace(result.rearrange, variable=name))


def test_rearrange_for_an_undefined_name() -> None:
    assert rearranged("y = a x + b", "b") == "b = -a*x + y"
    assert rearranged("x^2 + y^2 = a^2", "a") == "a = list[-sqrt(x^2 + y^2), sqrt(x^2 + y^2)]"
    line = "(a*sin(k)-w)/(a*cos(k)-q)*(x-q)+w"  # a bare expression in x means y = …
    expected = "w = (-a*q*sin(k) + a*x*sin(k) - a*y*cos(k) + q*y)/(-a*cos(k) + x)"
    assert rearranged(line, "w") == expected
    with pytest.raises(symbolic.SymbolicError, match="No solution for a"):
        rearranged("a^2 + b^2 = -1 - c^2", "a")  # only complex solutions


def test_rearrange_keeps_defined_names() -> None:
    rows = [(0, "y = m f(x) + k x + c"), (1, "m = 3"), (2, "k = 0.5"), (3, "f(t) = t^2")]
    result = compile_document(rows)[0]
    assert result.solve_for == ("c",)  # only the undefined name is offered
    assert result.rearrange is not None
    # m and k stay names (not 3 and 0.5); the function f is expanded.
    assert symbolic.rearrange(result.rearrange) == "c = -k*x - m*x^2 + y"


def solved_with(rows: list[str]) -> list[str]:
    results = compile_document(list(enumerate(rows)))
    target = results[len(rows) - 1].symbolic
    assert target is not None
    roots = symbolic.solve(target)
    assert isinstance(roots, list)
    return [root.exact for root in roots]


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        (["m = 2", "c = 3", "y = m x + c"], ["-c/m"]),
        (["a = 1", "y = a x^2 - 4"], ["-2*sqrt(1/a)", "2*sqrt(1/a)"]),
        (["k = 2", "y = x^3 - k x"], ["-sqrt(k)", "0", "sqrt(k)"]),
        (["f(x) = x^2", "b = 9", "y = f(x) - b"], ["-sqrt(b)", "sqrt(b)"]),  # f expanded
        (["a = -1", "y = a x^2 - 4"], []),  # the named roots are complex here
        (["y = x^2 - 4"], ["-2", "2"]),
    ],
)
def test_solve_keeps_constants_as_names(rows: list[str], expected: list[str]) -> None:
    assert solved_with(rows) == expected


def test_named_roots_plot_where_the_values_are() -> None:
    rows = ["a = 4", "y = a x^2 - 4"]
    roots = f"point[list[{', '.join(solved_with(rows))}], 0]"
    rows = ["a = 9", rows[1], roots]  # the roots follow a
    plot = compile_document(list(enumerate(rows)))[2].plot
    assert isinstance(plot, PointSetPlot)
    assert [x for x, _ in plot.points] == pytest.approx([-2 / 3, 2 / 3])
