"""Computer-algebra rows, symbolic composition, domain guards and cache reuse."""

from __future__ import annotations

import math

import numpy as np
import pytest

from opengraph.engine.ast import free_vars
from opengraph.engine.compiler import (
    CasFailure,
    CasRequest,
    CompileOptions,
    RowResult,
    compile_document,
)
from opengraph.engine.plots import ExplicitPlot

pytest.importorskip("sympy")

from opengraph.engine import symbolic  # noqa: E402 - needs SymPy


def compile_rows(
    rows: list[str], options: CompileOptions | None = None
) -> tuple[list[RowResult], dict[CasRequest, str | CasFailure]]:
    """Compile, computing the results rows wait for until none do (as the app does)."""
    known: dict[CasRequest, str | CasFailure] = {}
    keyed = list(enumerate(rows))
    while True:
        results = compile_document(keyed, options, cas_results=known)
        pending = {request for result in results.values() for request in result.pending}
        if not pending:
            return list(results.values()), known
        for request in pending:
            try:
                known[request] = symbolic.cas(request)
            except symbolic.SymbolicError as exc:
                known[request] = CasFailure(exc.message)


def messages(rows: list[str], options: CompileOptions | None = None) -> list[str]:
    return [result.message for result in compile_rows(rows, options)[0]]


def test_rows_wait_for_their_result() -> None:
    (result,) = compile_document([(0, "diff(x^3)")]).values()
    assert not result.is_error and result.plot is None
    assert result.message == "Computing the derivative…"
    (request,) = result.pending
    assert request.operation == "diff" and request.variable == "x"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("diff(x^3)", "= 3*x^2"),
        ("y = diff(x sin(x))", "= x*cos(x) + sin(x)"),
        ("diff(x^5, x, 3)", "= 60*x^2"),
        ("integrate(x e^x)", "= (x - 1)*exp(x)"),
        ("simplify(sin(x)^2 + cos(x)^2)", "= 1"),
        ("expand((x + 1)^3)", "= x^3 + 3*x^2 + 3*x + 1"),
        ("factor(x^2 - 1)", "= (x - 1)*(x + 1)"),
    ],
)
def test_operations(text: str, expected: str) -> None:
    assert messages([text]) == [expected]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("expand((a + b)^2)", "a^2 + 2*a*b + b^2"),
        ("factor(a^2 - b^2)", "(a - b)*(a + b)"),
        ("diff(a*x^2, x)", "2*a*x"),
        ("diff(t^2*y, t)", "2*t*y"),
        ("diff(diff(x^2*y^3, x), y)", "6*x*y^2"),
        ("limit(sin(a*x)/x, x, 0)", "a"),
        ("taylor(e^(a*x), x, 0, 2)", "a^2*x^2/2 + a*x + 1"),
        ("cancel((a^2 - b^2)/(a - b))", "a + b"),
        ("together(1/a + 1/b)", "(a + b)/(a*b)"),
        ("apart(1/(t^2 - 1), t)", "-1/(2*(t + 1)) + 1/(2*(t - 1))"),
        ("collect(a*x + b*x + c*x^2, x)", "c*x^2 + x*(a + b)"),
        ("trigsimp(sin(a)^2 + cos(a)^2)", "1"),
        ("factor(expand((a + b)^2))", "(a + b)^2"),
        ("cancel(together(1/a + 1/b))", "(a + b)/(a*b)"),
    ],
)
def test_operations_with_free_symbols(text: str, expected: str) -> None:
    (result,), _ = compile_rows([text])
    assert not result.is_error, result.message
    assert result.message.split("  (")[0] == f"= {expected}"
    assert result.plot is None


@pytest.mark.parametrize("operation", ["expand", "factor", "cancel", "together", "trigsimp"])
def test_symbolic_operations_do_not_define_free_names(operation: str) -> None:
    results, _ = compile_rows([f"{operation}(a + a)", "a + a", "y = a*x"])
    assert not results[0].is_error
    assert all(row.is_error and row.message == "Unknown name 'a'" for row in results[1:])


@pytest.mark.parametrize(
    ("operation", "expression"),
    [
        ("cancel", "(x^2 - 1)/(x - 1)"),
        ("together", "1/x - 1/x"),
        ("apart", "(x^2 - 1)/(x - 1)"),
        ("collect", "x/x + x^2/x"),
        ("trigsimp", "sin(x)/tan(x)"),
    ],
)
def test_algebra_results_keep_numeric_values_and_original_domain(
    operation: str, expression: str
) -> None:
    args = f"{expression}, x" if operation == "collect" else expression
    (result,), _ = compile_rows([f"{operation}({args})"])
    assert not result.is_error, result.message
    original = compile_document([(0, expression)])[0]
    assert isinstance(result.plot, ExplicitPlot) and isinstance(original.plot, ExplicitPlot)
    xs = np.array([-2.0, -1.0, 0.0, 0.5, 1.0, 2.0])
    np.testing.assert_allclose(
        result.plot.evaluate(xs), original.plot.evaluate(xs), equal_nan=True, atol=1e-14
    )


def test_cancellation_preserves_a_real_domain_after_nested_operations() -> None:
    (result,), _ = compile_rows(["factor(cancel(sqrt(x)^2/x))"])
    assert not result.is_error and isinstance(result.plot, ExplicitPlot)
    np.testing.assert_allclose(
        result.plot.evaluate(np.array([-1.0, 0.0, 1.0, 4.0])),
        [np.nan, np.nan, 1.0, 1.0],
        equal_nan=True,
    )


def test_rational_algebra_reuses_formula_when_parameters_change() -> None:
    text = "cancel((x^2 - a^2)/(x - a))"
    (_, initial), known = compile_rows(["a = 2", text])
    assert not initial.is_error
    for value in (-3, 0, 4):
        result = compile_document([(0, f"a = {value}"), (1, text)], cas_results=known)[1]
        assert not result.is_error and not result.pending
        assert isinstance(result.plot, ExplicitPlot)
        assert np.isnan(result.plot.evaluate(float(value)))
        assert result.plot.evaluate(value + 1.0) == pytest.approx(2 * value + 1)


def test_free_symbol_formula_becomes_a_plot_when_bound() -> None:
    text = "collect(a*x + b*x, x)"
    (initial,), known = compile_rows([text])
    assert initial.plot is None and not initial.is_error
    result = compile_document([(0, "a = 2"), (1, "b = 3"), (2, text)], cas_results=known)[2]
    assert not result.is_error and not result.pending
    assert isinstance(result.plot, ExplicitPlot)
    assert result.plot.evaluate(2.0) == pytest.approx(10)


def test_partial_fractions_bind_function_arguments_and_infer_the_variable() -> None:
    results, _ = compile_rows(["f(t) = apart(1/(t^2 - 1))", "f(2)", "f(1)"])
    assert not results[0].is_error and isinstance(results[0].plot, ExplicitPlot)
    assert results[1].message == "= 0.3333333333"
    assert results[2].message == "= undefined"


def test_partial_fractions_recompute_when_parameter_values_merge_poles() -> None:
    text = "apart(1/(x^2 - a^2), x)"
    (_, initial), known = compile_rows(["a = 2", text])
    assert isinstance(initial.plot, ExplicitPlot)
    # At a=0 the correct answer is 1/x^2, not coefficients containing 1/a.
    changed = compile_document([(0, "a = 0"), (1, text)], cas_results=known)[1]
    assert changed.pending
    for request in changed.pending:
        known[request] = symbolic.cas(request)
    result = compile_document([(0, "a = 0"), (1, text)], cas_results=known)[1]
    assert not result.is_error and isinstance(result.plot, ExplicitPlot)
    assert result.plot.evaluate(2.0) == pytest.approx(0.25)
    assert not np.isfinite(result.plot.evaluate(0.0))


@pytest.mark.parametrize("expression", ["1/(x^2 - a^2)", "1/((x-a)*(x-b))", "1/(a*x^2 + b*x)"])
def test_partial_fractions_do_not_assume_distinct_symbolic_poles(expression: str) -> None:
    (result,), _ = compile_rows([f"apart({expression}, x)"])
    assert result.is_error
    assert result.message == (
        "apart: Partial fractions need parameter conditions; define the parameters first"
    )


@pytest.mark.parametrize("expression", ["a/(x^2-1)", "1/(a*(x^2-1))", "1/(x-a)"])
def test_partial_fractions_allow_parameters_without_new_exclusions(expression: str) -> None:
    (result,), _ = compile_rows([f"apart({expression}, x)"])
    assert not result.is_error, result.message


def test_trigonometric_simplification_respects_degrees() -> None:
    (result,), _ = compile_rows(["trigsimp(2*sin(x)*cos(x))"], CompileOptions(degrees=True))
    assert not result.is_error and result.message == "= sin(2*x)"
    assert isinstance(result.plot, ExplicitPlot)
    xs = np.array([-60.0, 0.0, 30.0, 90.0])
    np.testing.assert_allclose(result.plot.evaluate(xs), np.sin(np.deg2rad(2 * xs)))


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("a + a", "2*a"),
        ("2*a + 3*a - a", "4*a"),
        ("a*x + a*x", "2*a*x"),
        ("(a + b)^2 - a^2 - b^2", "2*a*b"),
        ("sin(a)^2 + cos(a)^2", "1"),
        ("sqrt(a^2)", "abs(a)  (original domain preserved)"),
        ("(a^2 - 1)/(a - 1)", "a + 1  (original domain preserved)"),
        ("y + y", "2*y"),
    ],
)
def test_simplify_unassigned_variables(expression: str, expected: str) -> None:
    (result,), known = compile_rows([f"simplify({expression})"])
    assert not result.is_error, result.message
    assert result.message == f"= {expected}"
    assert len(known) == 1
    assert next(iter(known)).operation == "simplify"


def test_simplify_free_symbols_can_be_bound_later_without_recomputation() -> None:
    text = "simplify((a*x - a)/(x - 1))"
    (result,), known = compile_rows([text])
    assert not result.is_error and result.plot is None
    assert result.message == "= a  (original domain preserved)"
    (request,) = known
    assert free_vars(request.node) == {"a", "x"}
    assert known[request] == "a"
    for value in (0, 3, -2):
        row = compile_document([(0, f"a = {value}"), (1, text)], cas_results=known)[1]
        assert not row.is_error and not row.pending
        assert isinstance(row.plot, ExplicitPlot)
        assert np.isnan(row.plot.evaluate(1.0))
        assert row.plot.evaluate(2.0) == pytest.approx(value)


def test_simplify_does_not_bind_free_symbols_in_other_rows() -> None:
    results, _ = compile_rows(["simplify(a + a)", "a + a", "y = a*x"])
    assert not results[0].is_error
    for row in results[1:]:
        assert row.is_error and row.message == "Unknown name 'a'"


@pytest.mark.parametrize("complex_mode", [False, True])
@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("re(e^ix)", "cos(x)"),
        ("re(e^(i*x))", "cos(x)"),
        ("re(exp(ix))", "cos(x)"),
        ("im(e^ix)", "sin(x)"),
        ("re(e^(2ix))", "cos(2*x)"),
        ("im(e^(-ix))", "-sin(x)"),
        ("re(e^(a + ix))", "exp(a)*cos(x)"),
        ("im(e^(a + ix))", "exp(a)*sin(x)"),
        ("re(sqrt(-1))", "0"),
        ("im(sqrt(-1))", "1"),
    ],
)
def test_simplify_complex_projections_without_substitution(
    expression: str, expected: str, complex_mode: bool
) -> None:
    (result,), known = compile_rows(
        [f"simplify({expression})"], CompileOptions(complex_mode=complex_mode)
    )
    assert not result.is_error, result.message
    assert result.message.split("  (")[0] == f"= {expected}"
    assert expected in known.values()


def test_simplify_complex_projection_respects_degree_mode() -> None:
    (result,), _ = compile_rows(["simplify(re(e^ix))"], CompileOptions(degrees=True))
    assert not result.is_error, result.message
    assert result.message == "= cos(180*x/pi)"
    assert isinstance(result.plot, ExplicitPlot)
    xs = np.array([-3.0, -0.5, 0.0, 0.5, 3.0])
    np.testing.assert_allclose(result.plot.evaluate(xs), np.cos(xs))


def test_simplify_complex_shorthand_preserves_explicit_names() -> None:
    results, known = compile_rows(["ix = 3", "simplify(re(e^ix))"])
    assert not results[1].is_error and "exp(ix)" in known.values()
    results, known = compile_rows(["f(ix) = simplify(re(e^ix))", "f(2)"])
    assert not results[0].is_error and results[1].message == "= 7.389056099"
    assert "exp(ix)" in known.values()
    results, _ = compile_rows(["i = 2", "simplify(re(e^(i*x)))"])
    assert not results[1].is_error and results[1].message == "= exp(i*x)"


def test_simplify_complex_projection_preserves_a_pole() -> None:
    (result,), _ = compile_rows(["simplify(re(1/ix))"])
    assert not result.is_error and isinstance(result.plot, ExplicitPlot)
    xs = np.array([-2.0, 0.0, 2.0])
    np.testing.assert_allclose(result.plot.evaluate(xs), [0.0, np.nan, 0.0], equal_nan=True)


def test_simplify_complex_projection_can_retain_a_complex_argument() -> None:
    (result,), known = compile_rows(["simplify(arg(e^ix))"])
    assert not result.is_error, result.message
    assert "arg(exp(i*x))" in known.values()
    assert isinstance(result.plot, ExplicitPlot)
    xs = np.array([-4.0, -1.0, 0.0, 1.0, 4.0])
    np.testing.assert_allclose(result.plot.evaluate(xs), np.angle(np.exp(1j * xs)))


def test_simplify_complex_shorthand_in_a_function_parameter() -> None:
    results, known = compile_rows(["f(t) = simplify(re(e^it))", "f(2)"])
    assert not results[0].is_error and not results[1].is_error
    assert "cos(t)" in known.values()
    assert results[1].message == "= -0.4161468365"


def test_results_are_plotted() -> None:
    (result,), _ = compile_rows(["diff(x^3)"])
    assert isinstance(result.plot, ExplicitPlot)
    assert result.plot.evaluate(2.0) == pytest.approx(12.0)


def test_constants_stay_names_so_sliders_need_no_recomputation() -> None:
    results, known = compile_rows(["a = 2", "diff(a x^3)"])
    assert results[1].message == "= 3*a*x^2"
    # A new value for a reuses the result: nothing is pending.
    moved = compile_document([(0, "a = 5"), (1, "diff(a x^3)")], cas_results=known)
    assert not moved[1].pending
    plot = moved[1].plot
    assert isinstance(plot, ExplicitPlot)
    assert plot.evaluate(1.0) == pytest.approx(15.0)


def test_functions_defined_by_an_operation() -> None:
    rows = ["f(x) = x^3 + x", "g(x) = diff(f(x))", "g(2)", "h(t) = diff(t^3)", "h(2)"]
    assert messages(rows) == ["", "= 3*x^2 + 1", "= 13", "= 3*t^2", "= 12"]


def test_order_may_be_a_constant() -> None:
    assert messages(["k = 3", "diff(x^5, x, k)"])[1] == "= 60*x^2"


def test_degrees() -> None:
    assert messages(["diff(sin(x))"], CompileOptions(degrees=True)) == ["= pi*cos(x)/180"]


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        (["diff(floor(x))"], "diff: This expression can't be differentiated symbolically"),
        (["diff(x, x, 0)"], "diff: the order must be a whole number from 1 to 20"),
        (["diff(x, 2)"], "diff: the second argument names the variable"),
        (["factor(x, y)"], "Use factor(f)"),
        (["h(x) = diff(h(x))"], "Circular definition: h → h"),
        (["diff = 2"], "'diff' is a built-in function and can't be redefined"),
        (["collect(x + 1)"], "Use collect(a x + b x for x)"),
        (["collect(x + 1, 2)"], "collect: the second argument names the variable"),
        (["apart(x, x, 2)"], "Use apart(1/(x^2 - 1)) or apart(1/(x^2 - 1) for x)"),
        (
            ["apart(sin(x), x)"],
            "apart: Partial fractions need a rational expression in x; "
            "use apart(f, variable) to choose another variable",
        ),
    ],
)
def test_errors(rows: list[str], expected: str) -> None:
    (result,), _ = compile_rows(rows)
    assert result.is_error
    assert result.message == expected


def test_rows_using_a_pending_definition_wait_too() -> None:
    results = compile_document([(0, "g(x) = diff(x^2)"), (1, "g(3)")])
    assert results[0].pending and not results[0].is_error
    assert results[1].is_error and "still being computed" in results[1].message


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("limit(sin(x)/x, x, 0)", "= 1"),
        ("limit((x^2 - 1)/(x - 1), x, 1)", "= 2"),
        ("limit(abs(x)/x, x, 0, right)", "= 1"),
        ("limit(abs(x)/x, x, 0, left)", "= -1"),
        ("limit((e^t - 1)/t, t, 0)", "= 1"),
        ("limit(x^2, x, pi)", "= 9.869604401"),  # closed results show their value
        ("taylor(e^x, x, 0, 3)", "= x^3/6 + x^2/2 + x + 1"),
        ("taylor(sin(x), x, 0, 5)", "= x^5/120 - x^3/6 + x"),
        ("taylor(cos(x), x, 0, 0)", "= 1"),
    ],
)
def test_limit_and_taylor(text: str, expected: str) -> None:
    assert messages([text]) == [expected]


def test_limit_and_taylor_keep_constants_as_names() -> None:
    rows = [
        "a = 2",
        "limit(sin(a x)/x, x, 0)",
        "taylor(e^x, x, a, 1)",
        "n = 2",
        "taylor(e^x, x, 0, n)",
    ]
    results, known = compile_rows(rows)
    assert [r.message for r in results[1:3]] == ["= 2", "= (-a + x)*exp(a) + exp(a)"]
    assert "a" in known.values()  # the limit is the formula a, not the number 2
    assert results[4].message == "= x^2/2 + x + 1"
    moved = compile_document([(0, "a = 3"), (1, "taylor(e^x, x, a, 1)")], cas_results=known)
    assert not moved[1].pending
    plot = moved[1].plot
    assert isinstance(plot, ExplicitPlot)
    assert plot.evaluate(3.0) == pytest.approx(math.exp(3))


def test_taylor_defines_a_function() -> None:
    assert messages(["p(x) = taylor(ln(1 + x), x, 0, 2)", "p(1)"]) == ["= -x^2/2 + x", "= 0.5"]


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        (["limit(abs(x)/x, x, 0)"], "limit: The limit doesn't exist: the one-sided limits differ"),
        (["limit(sin(1/x), x, 0)"], "limit: The limit doesn't exist: the function oscillates"),
        (["limit(1/x, x, 0)"], "limit: The limit is infinite or doesn't exist"),
        (["limit(1/x, x, 0, right)"], "limit: The limit is infinite or doesn't exist"),
        (["limit(x, x, 0, up)"], "limit: the side must be left or right"),
        (["limit(x, x, x)"], "limit: the point must be a number or constants, not use x"),
        (["f(c) = limit(x, x, c)"], "limit: the point must be a number or constants, not use c"),
        (["limit(x, 0)"], "Use limit(sin(x)/x for x to 0), or … to 0 from right"),
        (["taylor(e^x, x, 0)"], "Use taylor(e^x for x at 0 degree 3)"),
        (["taylor(e^x, x, 0, 1.5)"], "taylor: the degree must be a whole number from 0 to 20"),
        (["taylor(1/x, x, 0, 2)"], "taylor: No Taylor polynomial about that point"),
        (["taylor(sqrt(x), x, 0, 2)"], "taylor: No Taylor polynomial about that point"),
        (["taylor(abs(x), x, 0, 2)"], "taylor: No Taylor polynomial about that point"),
    ],
)
def test_limit_and_taylor_errors(rows: list[str], expected: str) -> None:
    (result,), _ = compile_rows(rows)
    assert result.is_error
    assert result.message == expected
