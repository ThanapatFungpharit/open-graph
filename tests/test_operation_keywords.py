"""Operations with named parts: ``sum(k^2 for k from 1 to 10)`` and friends."""

from __future__ import annotations

import pytest

from opengraph.engine.compiler import RowResult, compile_document
from opengraph.engine.parser import parse
from opengraph.ui.mathedit.editing import EditorModel
from opengraph.ui.mathedit.nodes import BigOp
from opengraph.ui.mathedit.source import from_source


def rows(*texts: str) -> list[RowResult]:
    results = compile_document(list(enumerate(texts)))
    return [results[i] for i in range(len(texts))]


@pytest.mark.parametrize(
    ("keywords", "positional"),
    [
        ("sum(k^2 for k from 1 to 10)", "sum(k^2, k, 1, 10)"),
        ("Σ(k^2 for k to 10 from 1)", "Σ(k^2, k, 1, 10)"),  # clauses in any order
        ("prod(k for k from 1 to 5)", "prod(k, k, 1, 5)"),
        ("int(sin(t) for t from 0 to x)", "int(sin(t), t, 0, x)"),
        ("limit(sin(x)/x for x to 0)", "limit(sin(x)/x, x, 0)"),
        ("limit(1/x for x to 0 from right)", "limit(1/x, x, 0, right)"),
        ("taylor(e^x for x at 0 degree 3)", "taylor(e^x, x, 0, 3)"),
        ("taylor(e^x for x degree 3 at 0)", "taylor(e^x, x, 0, 3)"),
        ("diff(x^3 for x)", "diff(x^3, x)"),
        ("diff(x^3 for x order 2)", "diff(x^3, x, 2)"),
        ("integrate(x^2 for x)", "integrate(x^2, x)"),
        ("roots(a^2 - 4 for a)", "roots(a^2 - 4, a)"),
        ("collect(a x + b x for x)", "collect(a x + b x, x)"),
        ("plot(t^2 for t in (0, 2])", "plot(t^2, t, (0, 2])"),
        ("parametric(cos(u), sin(u) for u in [0, 2pi])", "parametric(cos(u), sin(u), u, [0, 2pi])"),
        ("sum((k + 1)^2 for k from 1 to n + 1)", "sum((k + 1)^2, k, 1, n + 1)"),
    ],
)
def test_keyword_form_means_the_positional_call(keywords: str, positional: str) -> None:
    assert parse(keywords) == parse(positional)


def test_keyword_forms_compute() -> None:
    total, product, integral = rows(
        "sum(k^2 for k from 1 to 10)", "prod(k for k from 1 to 5)", "int(t for t from 0 to 2)"
    )
    assert total.message == "= 385"
    assert product.message == "= 120"
    assert integral.message.startswith("= 2")


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("sum(k^2 for k in [1, 2])", "sum has no 'in' part; write sum(k^2 for k from 1 to 10)"),
        ("limit(1/x for x to 0 from up)", "limit: write 'from left' or 'from right'"),
        ("sum(k^2 for k from 1 from 2 to 3)", "'from' is given twice"),
        ("parametric(cos(u) for u in [0, 1])", "parametric takes 2 or 3 expressions before 'for'"),
        ("sum(k^2 for k from 1 to 10) to", "Unknown name 'to'"),  # a name outside `(…)`
    ],
)
def test_wrong_clauses_say_how_to_write_them(text: str, message: str) -> None:
    result = rows(text)[0]
    assert result.is_error and not result.incomplete
    assert message in result.message


@pytest.mark.parametrize(
    "text",
    [
        "sum(k^2 for)",
        "sum(k^2 for k)",
        "sum(k^2 for k from)",
        "sum(k^2 for k from 1)",
        "sum(k^2 for k from 1 to)",
        "sum(k^2 for k from 1 to 10",
        "diff(x^3 for x ord)",
        "taylor(e^x for x a)",
    ],
)
def test_unfinished_clauses_are_hints(text: str) -> None:
    result = rows(text)[0]
    assert result.is_error and result.incomplete, result.message


def test_clause_words_are_names_outside_operations() -> None:
    _, _, _, curve, value = rows("to = 3", "at = 2", "from = 1", "y = to x", "to at from")
    assert not curve.is_error
    assert value.message == "= 6"


def typed(keys: str) -> EditorModel:
    model = EditorModel()
    for key in keys:
        model.type_text(key)
    return model


@pytest.mark.parametrize(
    ("keys", "expected"),
    [
        ("sum(k^2 for k from 1 to 10)", "sum(k^2, k, 1, 10)"),
        ("Σ(k for k to 4 from 2)", "Σ(k, k, 2, 4)"),
        ("sum(2^k for k from 1 to 2^3)", "sum(2^k, k, 1, 2^3)"),
        ("int(sin(t) for t from 0 to x)", "int(sin(t), t, 0, x)"),
        ("limit(sin(x)/x for x to 0)", "limit(sin(x)/x, x, 0)"),
        ("limit(1/x for x to 0 from right)", "limit(1/x, x, 0, right)"),
        ("taylor(e^x for x at 0 degree 3)", "taylor(e^x, x, 0, 3)"),
        ("plot(t^2 for t in (0, 2])", "plot(t^2, t, (0, 2])"),
        ("sum(k^2, k, 1, 3)", "sum(k^2, k, 1, 3)"),  # the positional form still types
    ],
)
def test_typing_the_keyword_form_in_the_editor(keys: str, expected: str) -> None:
    model = typed(keys)
    assert parse(model.source) == parse(expected), model.source


def test_typed_clauses_fill_the_template_fields() -> None:
    model = typed("sum(k^2 for k from 1 to n")
    (op,) = model.root.nodes
    assert isinstance(op, BigOp)
    assert [len(block.nodes) > 0 for block in op.arguments] == [True] * 4


def test_keyword_text_opens_as_the_template() -> None:
    root, _ = from_source("sum(k^2 for k from 1 to 10)")
    assert [type(node) for node in root.nodes] == [BigOp]
