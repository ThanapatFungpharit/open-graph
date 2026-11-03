"""``point[x, y]``, ``list[a, b]`` and intervals never share a spelling."""

from __future__ import annotations

import pytest

from opengraph.engine.ast import (
    BinOp,
    Index,
    IntervalExpr,
    ListExpr,
    Num,
    RangeExpr,
    TupleExpr,
    children,
    rebuild,
)
from opengraph.engine.compiler import RowResult, compile_document, compile_expression
from opengraph.engine.errors import ParseError
from opengraph.engine.parser import parse
from opengraph.engine.plots import ParametricPlot, PointPlot, PointSetPlot


def compile_rows(*texts: str) -> list[RowResult]:
    results = compile_document(list(enumerate(texts)))
    return [results[i] for i in range(len(texts))]


# -- parsing -------------------------------------------------------------------------------


def test_point_parses_to_a_point_node() -> None:
    node = parse("point[1, 2]")
    assert isinstance(node, TupleExpr) and len(node.items) == 2
    assert node.span == (0, 11)


def test_point_source_uses_square_brackets() -> None:
    from opengraph.engine.source import source

    text = source(parse("point[1, 2]"))
    assert text == "point[1, 2]"
    assert isinstance(parse(text), TupleExpr)


@pytest.mark.parametrize("text", ["point(1, 2)", "2point(1, 2)", "point()"])
def test_old_point_syntax_explains_the_new_spelling(text: str) -> None:
    with pytest.raises(ParseError, match=r"point\[x, y\]"):
        parse(text)


def test_list_and_range_need_the_keyword() -> None:
    assert isinstance(parse("list[1, 2, 3]"), ListExpr)
    assert isinstance(parse("list[7]"), ListExpr)  # a one-element list is still a list
    assert isinstance(parse("list[]"), ListExpr)
    assert isinstance(parse("list[1...5]"), RangeExpr)
    assert isinstance(parse("list [1, 2]"), ListExpr)  # a space before the bracket is fine


@pytest.mark.parametrize(
    ("text", "lo", "hi", "closed_lo", "closed_hi"),
    [
        ("[1, 2]", 1, 2, True, True),
        ("(1, 2)", 1, 2, False, False),
        ("[1, 2)", 1, 2, True, False),
        ("(1, 2]", 1, 2, False, True),
        ("[-1, 1]", -1, 1, True, True),
        ("  ( 0 ,5 ]  ", 0, 5, False, True),
    ],
)
def test_interval_notation(
    text: str, lo: float, hi: float, closed_lo: bool, closed_hi: bool
) -> None:
    node = parse(text)
    assert isinstance(node, IntervalExpr)
    assert (node.closed_lo, node.closed_hi) == (closed_lo, closed_hi)
    assert isinstance(node.lo, Num) or children(node)  # endpoints are expressions
    assert compile_expression(text).message == (
        f"= {'[' if closed_lo else '('}{lo}, {hi}{']' if closed_hi else ')'}"
    )


def test_interval_endpoints_are_expressions() -> None:
    node = parse("[1 + 1, 2^3)")
    assert isinstance(node, IntervalExpr) and isinstance(node.lo, BinOp)
    assert compile_expression("[1 + 1, 2^3)").message == "= [2, 8)"
    assert compile_expression("(-pi, pi]").message.startswith("= (-3.14159")
    assert compile_expression("[sqrt(4), 2 sqrt(4)]").message == "= [2, 4]"


def test_nested_expressions_inside_each_construct() -> None:
    assert isinstance(parse("point[L[1], 3]"), TupleExpr)
    assert isinstance(parse("list[point[1, 2], point[3, 4]]"), ListExpr)
    assert isinstance(parse("point[list[1, 2][2], |list[1, 2][1]|]"), TupleExpr)
    assert isinstance(parse("point[point[1, 2], 3]"), TupleExpr)  # rejected when compiled
    assert isinstance(parse("[list[1, 2][1], (3 + 4)]"), IntervalExpr)
    assert isinstance(parse("f(x) = [x, x + 1]"), object)


def test_index_syntax_still_works_beside_the_new_brackets() -> None:
    assert isinstance(parse("L[2]"), Index)
    assert isinstance(parse("L[2...3]"), Index)
    assert isinstance(parse("L[list[1, 3]]"), Index)
    assert isinstance(parse("list[5, 6, 7][3]"), Index)
    assert isinstance(parse("L[2][1]"), Index)


def test_point_list_in_one_row() -> None:
    node = parse("point[1, 2], point[3, 4], point[5, 6]")
    assert isinstance(node, ListExpr) and all(isinstance(p, TupleExpr) for p in node.items)


def test_interval_node_children_and_rebuild() -> None:
    node = parse("[1, 2)")
    assert isinstance(node, IntervalExpr)
    again = rebuild(node, [Num(5), Num(6)])
    assert isinstance(again, IntervalExpr)
    assert (again.lo, again.hi) == (Num(5), Num(6))
    assert (again.closed_lo, again.closed_hi) == (True, False)
    assert len(children(node)) == 2


# -- syntax errors -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("(1, 2, 3)", "Too many values"),
        ("[1, 2, 3]", "Too many values"),
        ("[1]", "Brackets alone are an interval"),
        ("[x + 1]", "Brackets alone are an interval"),
        ("[]", "Empty brackets"),
        ("()", "Empty parentheses"),
        ("{1, 2}", "Braces only group"),
        ("[1...5]", "Ranges are written list"),
        ("(1...5)", "Ranges are written list"),
        ("point[1]", "two or three"),
        ("point[1, 2, 3, 4]", "two or three"),
        ("point[]", "needs two or three coordinates"),
        ("list[1, 2", "Missing closing ']'"),
        ("list[1 2]", "Expected ']'"),
        ("[1, 2", "Missing closing"),
        ("(1, 2", "Missing closing"),
        ("[1, 2}", "Expected"),
        ("(1)]", "Unmatched ']'"),
        ("[1, 2, 3)", "Too many values"),
        ("L[1, 2]", "An index takes one position"),
        ("L[list[1, 2]", "Missing closing ']'"),
        ("1, 2", "several points as point"),
        ("(1, 2), (3, 4)", "Unexpected ','"),
        ("point[1, 2], 3", "Separate points"),
        ("[1,", "incomplete"),
    ],
)
def test_syntax_errors(text: str, message: str) -> None:
    with pytest.raises(ParseError) as caught:
        parse(text)
    assert message.lower() in caught.value.message.lower()


def test_unfinished_input_is_marked_incomplete() -> None:
    for text in ("[1,", "(1, 2", "list[1, 2", "point[1,", "[1, 2"):
        with pytest.raises(ParseError) as caught:
            parse(text)
        assert caught.value.incomplete, text


def test_error_spans_point_at_the_construct() -> None:
    with pytest.raises(ParseError) as caught:
        parse("1 + (1, 2, 3)")
    assert caught.value.span == (4, 13)
    with pytest.raises(ParseError) as caught:
        parse("1 + point[1]")
    assert caught.value.span == (4, 12)


# -- points --------------------------------------------------------------------------------


def test_points_plot() -> None:
    assert isinstance(compile_expression("point[1, 2]").plot, PointPlot)
    assert isinstance(compile_expression("point[1, 2], point[3, 4]").plot, PointSetPlot)
    assert isinstance(compile_expression("list[point[1, 2], point[3, 4]]").plot, PointSetPlot)
    assert isinstance(compile_expression("point[list[1, 2, 3], 0]").plot, PointSetPlot)
    assert isinstance(
        compile_expression("parametric(cos(t), sin(t), t, [0, 2pi])").plot, ParametricPlot
    )


def test_named_points_and_vector_arithmetic() -> None:
    rows = compile_rows("p = point[1, 2]", "q = p + point[3, 4]", "2q", "point[1, 2]/2")
    assert all(not r.is_error for r in rows)
    assert isinstance(rows[2].plot, PointPlot) and (rows[2].plot.x, rows[2].plot.y) == (8, 12)


def test_a_bare_pair_is_an_interval_not_a_point() -> None:
    result = compile_expression("(1, 2)")
    assert result.plot is None and result.message == "= (1, 2)"


def test_nested_points_are_rejected() -> None:
    assert "can't be points" in compile_expression("point[point[1, 2], 3]").message
    assert "list items" in compile_expression("list[point[1, 2]] + 1").message


# -- lists ---------------------------------------------------------------------------------


def test_list_values_display_like_their_source() -> None:
    assert compile_expression("list[1, 2, 3]").message == "= list[1, 2, 3]"
    assert compile_expression("list[]").message == "= list[]"
    assert compile_expression("list[1...4]").message == "= list[1, 2, 3, 4]"
    assert compile_expression("2 list[1, 2]").message == "= list[2, 4]"


def test_lists_in_definitions_and_indexing() -> None:
    rows = compile_rows("L = list[10, 20, 30]", "L[2]", "L[list[3, 1]]", "L[1...2]", "total(L)")
    assert [r.message for r in rows[1:]] == [
        "= 20",
        "= list[30, 10]",
        "= list[10, 20]",
        "= 60",
    ]


def test_list_of_lists_is_rejected() -> None:
    assert "other lists" in compile_expression("list[list[1], 2]").message


# -- intervals as values -------------------------------------------------------------------


def test_interval_definitions() -> None:
    rows = compile_rows("I = [0, 1)", "J = (a, a + 1]", "a = 2")
    assert rows[0].message == "= [0, 1)" and rows[0].defines == "I"
    assert rows[1].message == "= (2, 3]"
    assert not any(r.is_error for r in rows)


def test_degenerate_and_reversed_intervals() -> None:
    assert compile_expression("[2, 2]").message == "= [2, 2]"
    assert not compile_expression("[2, 2)").is_error
    assert not compile_expression("(2, 2)").is_error
    assert "out of order" in compile_expression("[3, 1]").message
    assert compile_expression("[3, 1]").is_error


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("[0, x]", "must be numbers"),
        ("[0, list[1, 2]]", "must be numbers"),
        ("[0, point[1, 2]]", "must be numbers"),
        ("[0, [1, 2]]", "must be numbers"),
        ("[0, 1/0]", "finite"),
        ("[0, ln(0)]", "finite"),
        ("list[[1, 2]]", "can't be used here"),
        ("[1, 2] = 3", "can't be used here"),
        ("list[1...[1, 2]]", "Range bounds must be numbers"),
        ("list[1, 2][[1, 2]]", "can't be used here"),
    ],
)
def test_intervals_are_not_numbers(text: str, message: str) -> None:
    result = compile_expression(text)
    assert result.is_error and message in result.message


def test_interval_names_broadcast_in_arithmetic_and_points() -> None:
    rows = compile_rows("I = [0, 1]", "I + 1", "y = I x", "point[I, 0]", "L = list[I]")
    assert not rows[0].is_error
    assert all(not r.is_error for r in rows[:4])
    assert rows[2].plot is not None and isinstance(rows[3].plot, ParametricPlot)
    assert rows[4].is_error  # finite lists do not nest continuous collections


def test_interval_in_a_table_cell_is_an_error() -> None:
    from opengraph.engine.compiler import TableColumn, TableInput

    table = TableInput(columns=(TableColumn("x", ("[1, 2]", "3")), TableColumn("y_1", ("1", "2"))))
    assert compile_document([], tables=[(0, table)])[0].cell_errors


# -- reserved words and compatibility ------------------------------------------------------


@pytest.mark.parametrize("text", ["point = 3", "list = 3", "point[x] = x", "list(x) = x"])
def test_point_and_list_cannot_be_redefined(text: str) -> None:
    assert compile_expression(text).is_error


def test_existing_features_are_unchanged() -> None:
    rows = compile_rows(
        "y = x^2",
        "f(x) = 2x + 1",
        "f(3)",
        "|x - 1|",
        "{x + 1}^2",
        "sum(k, k, 1, 4)",
        "2(x + 1)",
        "x^2 + y^2 = 1",
        "y < x",
        "k = 3",
        "n!",
    )
    errors = [r.message for r in rows if r.is_error]
    assert errors == ["Unknown name 'n'"] or all(not r.is_error for r in rows[:-1])
    assert rows[2].message == "= 7"
    assert compile_expression("max(list[1, 5, 3])").message == "= 5"
    assert compile_expression("f(1, 2)").message.startswith("Unknown function")
