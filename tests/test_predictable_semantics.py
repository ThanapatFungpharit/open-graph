"""Regression checks for mathematical correctness and explicit, stable semantics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from opengraph.engine import symbolic
from opengraph.engine.compiler import (
    CasFailure,
    CasRequest,
    CompileOptions,
    RowResult,
    TableColumn,
    TableInput,
    compile_document,
    compile_expression,
)
from opengraph.engine.plots import (
    ExplicitPlot,
    ImplicitPlot,
    ParametricPlot,
    PlotGroup,
    PointPlot,
    RegionPlot,
)
from opengraph.engine.sampling import ViewRect
from opengraph.graph.settings import GraphSettings
from opengraph.ui.mathedit.editing import EditorModel


def computed(*texts: str) -> list[RowResult]:
    cache: dict[CasRequest, str | CasFailure] = {}
    for _ in range(20):
        results = compile_document(list(enumerate(texts)), cas_results=cache)
        pending = {q for r in results.values() for q in r.pending}
        if not pending:
            return list(results.values())
        for request in pending:
            try:
                cache[request] = symbolic.cas(request)
            except symbolic.SymbolicError as exc:
                cache[request] = CasFailure(exc.message)
    raise AssertionError("CAS did not settle")


def test_sample_agreement_cannot_prove_a_constant_coefficient() -> None:
    row = compile_expression("(1 + floor(abs(x)/10)) y = 1")
    assert isinstance(row.plot, ImplicitPlot)
    assert row.plot.fn(x=20.0, y=1 / 3) == pytest.approx(0)
    assert row.plot.fn(x=20.0, y=1.0) == pytest.approx(2)


@pytest.mark.parametrize(
    "parameter, expression, expected",
    [
        ("a = 0", "integrate(e^(a x))", 2.0),
        ("b = -1", "integrate(x^b)", math.log(2)),
    ],
)
def test_antiderivatives_handle_exceptional_parameter_values(
    parameter: str,
    expression: str,
    expected: float,
) -> None:
    result = computed(parameter, expression)[1]
    assert not result.is_error and isinstance(result.plot, ExplicitPlot)
    assert result.plot.evaluate(2.0) == pytest.approx(expected)


def test_parameter_changes_recompute_an_antiderivative() -> None:
    cache: dict[CasRequest, str | CasFailure] = {}
    for a in (2, 0, -2):
        rows = [(0, f"a={a}"), (1, "integrate(e^(a x))")]
        initial = compile_document(rows, cas_results=cache)[1]
        assert initial.pending
        for request in initial.pending:
            cache[request] = symbolic.cas(request)
        result = compile_document(rows, cas_results=cache)[1]
        assert isinstance(result.plot, ExplicitPlot)
        expected = 1 if a == 0 else math.exp(a) / a
        assert result.plot.evaluate(1.0) == pytest.approx(expected)


@pytest.mark.parametrize("operation", ["simplify", "expand", "factor"])
def test_algebra_preserves_a_removable_hole(operation: str) -> None:
    result = computed(f"y = {operation}((x^2 - 1)/(x - 1))")[0]
    assert isinstance(result.plot, ExplicitPlot)
    assert np.isnan(result.plot.evaluate(1.0))
    assert result.plot.evaluate(2.0) == pytest.approx(3)


def test_symbolic_menu_result_preserves_domain_after_source_round_trip() -> None:
    original = compile_expression("(x^2 - 1)/(x - 1)")
    assert original.symbolic is not None
    text = symbolic.simplify(original.symbolic)
    result = compile_expression("y = " + text)
    assert isinstance(result.plot, ExplicitPlot)
    assert np.isnan(result.plot.evaluate(1.0))
    assert result.plot.evaluate(2.0) == pytest.approx(3)


def test_domain_guards_bind_function_arguments_at_call_time() -> None:
    results = computed("f(t) = simplify(t/t)", "f(0)", "f(2)")
    assert results[1].message == "= undefined"
    assert results[2].message == "= 1"


@pytest.mark.parametrize(
    "text",
    [
        "1/2x",
        "x^2/3",
        "3^2!",
        "2^3x",
        "2^-x*3",
        "x^1e3",
        "x^1e-3",
        "1/2E+3",
        "x^1e-x",
        "x^(1)e3",
    ],
)
def test_typing_pasting_and_plain_text_have_the_same_precedence(text: str) -> None:
    typed, pasted = EditorModel(), EditorModel()
    for ch in text:
        typed.type_text(ch)
    pasted.paste(text)
    left, right = compile_expression(typed.source), compile_expression(pasted.source)
    assert not left.is_error and not right.is_error
    if isinstance(left.plot, ExplicitPlot):
        assert isinstance(right.plot, ExplicitPlot)
        np.testing.assert_allclose(
            left.plot.evaluate(np.array([1.0, 2.0, 8.0])),
            right.plot.evaluate(np.array([1.0, 2.0, 8.0])),
        )
    else:
        assert left.message == right.message


def test_identifiers_do_not_depend_on_which_other_names_are_defined() -> None:
    for extra in ([], [(3, "a=2"), (4, "b=3")]):
        row = compile_document([(0, "ab"), *extra])[0]
        assert row.is_error and "'ab'" in row.message
    result = compile_document([(0, "ab"), (1, "ab=10"), (2, "a=2"), (3, "b=3")])[0]
    assert result.message == "= 10"


@pytest.mark.parametrize("word", ["spin", "pix", "sqrtvalue", "epsilonvalue"])
def test_editor_never_converts_a_substring_of_a_name(word: str) -> None:
    model = EditorModel()
    for ch in word:
        model.type_text(ch)
    assert model.source == word


def test_equations_never_introduce_bindings_even_after_exact_solving() -> None:
    rows = [(0, "a^2=4"), (1, "y=a x")]
    initial = compile_document(rows)
    target = initial[0].symbolic
    assert target is not None
    completed = compile_document(rows, exact_solutions={target: (("-2", -2.0), ("2", 2.0))})
    assert initial[1].is_error and completed[1].is_error
    assert initial[0].defines is None and completed[0].defines is None
    duplicated = compile_document([(0, "2a=4"), (1, "2a=4"), (2, "a")])
    assert duplicated[0].message == duplicated[1].message and duplicated[2].is_error


def test_explicit_root_lists_can_be_bound_and_used_for_curve_families() -> None:
    results = computed("A = roots(a^2 - 4, a)", "y = A x", "total(A)")
    assert results[0].message == "= list[-2, 2]"
    assert isinstance(results[1].plot, PlotGroup)
    assert results[2].message == "= 0"


def test_grid_changes_do_not_reinterpret_rows() -> None:
    rows = [(0, "r=2"), (1, "r=1+cos(theta)"), (2, "y=x^2")]
    cartesian = compile_document(rows, GraphSettings().compile_options())
    polar = compile_document(rows, GraphSettings(polar_grid=True).compile_options())
    assert cartesian == polar


def test_parametric_parameters_and_ranges_are_local() -> None:
    rows = [
        (0, "parametric(cos(u), sin(u), u, [0, pi])"),
        (1, "u=99"),
        (2, "tmin=-100"),
        (3, "tmax=100"),
    ]
    plot = compile_document(rows)[0].plot
    assert isinstance(plot, ParametricPlot) and plot.t_max == pytest.approx(math.pi)
    x, y = plot.xy(np.array([0.0, math.pi / 2]))
    np.testing.assert_allclose(x, [1, 0], atol=1e-12)
    np.testing.assert_allclose(y, [0, 1], atol=1e-12)
    assert compile_expression("point[cos(t),sin(t)]").is_error


def test_function_parameter_names_do_not_control_visibility_or_derivatives() -> None:
    rows = compile_document([(0, "f(t)=t^3")])
    assert isinstance(rows[0].plot, ExplicitPlot)
    assert rows[0].symbolic is not None
    assert symbolic.derivative(rows[0].symbolic) == "3*x^2"


def test_interval_restrictions_respect_open_and_closed_endpoints() -> None:
    row = compile_expression("plot(u^2, u, (0, 2])")
    assert isinstance(row.plot, ExplicitPlot)
    values = row.plot.evaluate(np.array([-1.0, 0.0, 1.0, 2.0, 3.0]))
    np.testing.assert_allclose(values, [np.nan, np.nan, 1.0, 4.0, np.nan], equal_nan=True)
    assert not compile_expression("(2,2)").is_error


def test_unequal_list_lengths_are_never_silently_truncated() -> None:
    for text in ("list[1,2]+list[10,20,30]", "point[list[1,2],list[3]]"):
        result = compile_expression(text)
        assert result.is_error and "equal lengths" in result.message


def test_numerical_search_discloses_limits_and_does_not_claim_completeness() -> None:
    row = compile_expression("x^2=1000000000000000000")
    assert "numerical search [-1e8, 1e8]" in row.message
    assert "may be missed" in row.message


def test_even_power_implicit_curves_keep_their_zero_set() -> None:
    view = ViewRect(-2.03, 2.07, -2.01, 2.11, 777, 593)
    plain = compile_expression("x^2+y^2=1").plot
    squared = compile_expression("(x^2+y^2-1)^2=0").plot
    assert isinstance(plain, ImplicitPlot) and isinstance(squared, ImplicitPlot)
    np.testing.assert_allclose(plain.sample(view).segments, squared.sample(view).segments)


def test_symbolic_formula_with_free_names_is_a_valid_display_row() -> None:
    result = compile_expression("formula(a, (y-b)/x)")
    assert not result.is_error and result.defines is None and result.plot is None
    assert result.message.startswith("a = ")


def test_reduction_binding_is_independent_of_other_definitions() -> None:
    rows = compile_document([(0, "k=99"), (1, "sum(k,k,1,3)")])
    assert rows[1].message == "= 6"
    assert compile_expression("sum(k,1,3)").is_error


def test_complex_mode_still_has_explicit_principal_branch_semantics() -> None:
    row = compile_expression("(-8)^(1/3)", CompileOptions(complex_mode=True))
    assert "1 + 1.732" in row.message


def test_chained_inequalities_intersect_masks_and_keep_individual_boundary_styles() -> None:
    result = compile_expression("1 < x <= 3")
    assert not result.is_error and isinstance(result.plot, PlotGroup)
    first, second = result.plot.plots
    assert isinstance(first, RegionPlot) and isinstance(second, RegionPlot)
    values = first.holds(np.array([0.0, 1.0, 2.0, 3.0, 4.0]), np.zeros(5))
    assert values.tolist() == [False, False, True, True, False]
    view = ViewRect(0, 4, -2, 2, 400, 400)
    a, b = first.sample(view), second.sample(view)
    assert a.dashed and not b.dashed
    assert a.fill is not None and b.fill is None  # shade the intersection just once
    assert compile_expression("1 < 2 <= 3").message == "true"
    assert compile_expression("1 < 0 < x").message == "false"


def test_singleton_and_empty_plot_ranges_have_their_mathematical_meaning() -> None:
    assert compile_expression("plot(u^2, u, [2, 2])").plot == PointPlot(2, 4)
    assert compile_expression("plot(u^2, u, (2, 2))").plot is None
    assert compile_expression("parametric(t, t^2, t, [2, 2])").plot == PointPlot(2, 4)
    assert compile_expression("parametric(t, t^2, t, [2, 2))").plot is None


def test_solving_a_guarded_expression_does_not_restore_an_excluded_root() -> None:
    original = compile_expression("guard(x, 1/x)")
    assert original.symbolic is not None
    assert symbolic.solve(original.symbolic) == []
    assert computed("roots(guard(x, 1/x), x)")[0].message == "= list[]"


def test_explicit_root_lists_use_the_real_odd_root_convention() -> None:
    row = computed("roots(x^(1/3)-x, x)")[0]
    assert not row.is_error and row.message == "= list[-1, 0, 1]"


@pytest.mark.parametrize("word", ["spin", "pix", "epsilonvalue", "sqrtvalue"])
def test_latex_round_trips_a_whole_identifier(word: str) -> None:
    from opengraph.ui.mathedit.latex import latex_to_source, to_latex
    from opengraph.ui.mathedit.source import from_source

    root, _ = from_source(word)
    assert latex_to_source(to_latex(root.nodes)) == word


def test_restricted_function_ranges_follow_the_argument_and_survive_symbolic_output() -> None:
    results = compile_document(
        [(0, "f(t) = plot(t^2, t, (0, 2])"), (1, "f(0)"), (2, "f(2)"), (3, "y=f(x+1)")]
    )
    assert results[1].message == "= undefined" and results[2].message == "= 4"
    assert isinstance(results[3].plot, ExplicitPlot)
    assert results[3].plot.evaluate(0.0) == 1
    assert np.isnan(results[3].plot.evaluate(2.0))
    target = results[3].symbolic
    assert target is not None
    copied = compile_expression("y=" + symbolic.simplify(target))
    assert isinstance(copied.plot, ExplicitPlot)
    assert copied.plot.evaluate(0.0) == 1 and np.isnan(copied.plot.evaluate(2.0))


def test_shorter_table_columns_keep_row_alignment_with_undefined_cells() -> None:
    table = TableInput(
        columns=(
            TableColumn("x_1", ("1", "2", "3")),
            TableColumn("y_1", ("10",)),
            TableColumn("z_1 = x_1 + y_1", ()),
        )
    )
    results = compile_document([(0, "length(y_1)"), (1, "y_1[3]")], tables=[(2, table)])
    assert results[0].message == "= 3" and results[1].message == "= undefined"
    assert not results[2].is_error
    assert results[2].table_values[(0, 2)] == "11"
    assert results[2].table_values[(2, 2)] == ""


@pytest.mark.parametrize(
    "options",
    [GraphSettings().compile_options(), GraphSettings(polar_grid=True).compile_options()],
)
def test_radius_assignment_is_a_reusable_variable(options) -> None:
    results = compile_document([(0, "r=1"), (1, "r+2"), (2, "x^2+y^2=r^2")], options)
    assert results[0].defines == "r"
    assert results[0].plot is None and not results[0].is_error
    assert results[1].message == "= 3"
    assert results[2].plot is not None and not results[2].is_error
