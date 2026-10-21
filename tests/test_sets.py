"""Infinite sequences are lazy values; previews never become arithmetic operands."""

from __future__ import annotations

import pytest

from opengraph.engine.compiler import RowResult, compile_document
from opengraph.engine.errors import ParseError
from opengraph.engine.parser import parse
from opengraph.engine.plots import PlotGroup


def rows(*texts: str) -> list[RowResult]:
    results = compile_document(list(enumerate(texts)))
    return [results[i] for i in range(len(texts))]


@pytest.mark.parametrize(
    ("text", "preview"),
    [
        ("k in sets[N]", "seq[k for k in sets[N]]  (preview: 1, 2, 3, 4, 5, 6, 7, 8, …)"),
        ("k ∈ sets[N]", "seq[k for k in sets[N]]  (preview: 1, 2, 3, 4, 5, 6, 7, 8, …)"),
        ("m in sets[Z]", "seq[m for m in sets[Z]]  (preview: 0, 1, -1, 2, -2, 3, -3, 4, …)"),
    ],
)
def test_membership_is_shorthand_for_a_sequence(text: str, preview: str) -> None:
    result = rows(text)[0]
    assert not result.is_error
    assert result.defines == text[0]
    assert result.message == f"= {preview}"
    name = text[0]
    domain = "Z" if name == "m" else "N"
    assert parse(text) == parse(f"{name} = seq[{name} for {name} in sets[{domain}]]")


def test_membership_stays_lazy() -> None:
    _, term, terms, total, curves = rows(
        "k in sets[N]", "k[1000]", "k[2...4]", "total(k)", "y = k[1...3] x"
    )
    assert term.message == "= 1000"
    assert terms.message == "= list[2, 3, 4]"
    assert total.is_error and "S[1...100]" in total.message
    assert isinstance(curves.plot, PlotGroup) and len(curves.plot.plots) == 3


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("x in sets[N]", "plot axis"),
        ("2k in sets[N]", "one name before 'in'"),
        ("k in R", "Only sets[N] (natural numbers) and sets[Z] (integers)"),
        ("k in sets[Q]", "Only sets[N] (natural numbers) and sets[Z] (integers)"),
        ("k in N", "Write sets[N] for the natural numbers; a bare N is a variable"),
        ("k in Z", "Write sets[Z] for the integers; a bare Z is a variable"),
        ("k ∈ ℕ", "Write sets[N] for the natural numbers"),
        ("seq[k for k in N]", "Write sets[N] for the natural numbers; a bare N is a variable"),
        ("sets[N]", "A number set goes after 'in'"),
        ("sets = 1", "keyword"),
        ("ℕ = 1", "keyword"),
    ],
)
def test_invalid_membership_is_rejected(text: str, message: str) -> None:
    result = rows(text)[0]
    assert result.is_error and message in result.message


@pytest.mark.parametrize("text", ["k in", "k in s", "k in sets", "k in sets[]", "k ∈"])
def test_unfinished_membership_is_incomplete(text: str) -> None:
    result = rows(text)[0]
    assert result.is_error and result.incomplete, result.message


def test_plain_n_and_z_are_variables() -> None:
    """The point of sets[N]: a variable N never changes meaning next to a sequence."""
    _, _, _, curve, value, sequence = rows(
        "N = 5", "Z = 2", "k in sets[N]", "y = N x", "N Z", "seq[N k for k in sets[Z]]"
    )
    assert not curve.is_error
    assert value.message == "= 10"
    assert "preview: 0, 5, -5, 10, -10" in sequence.message


def test_generator_is_shown_exactly_with_a_labelled_preview() -> None:
    result = rows("S = seq[2k + 1 for k in sets[N]]")[0]
    assert (
        result.message == "= seq[2k + 1 for k in sets[N]]  (preview: 3, 5, 7, 9, 11, 13, 15, 17, …)"
    )
    assert result.defines == "S"


def test_sequences_retain_order_and_repeated_values() -> None:
    result = rows("seq[k^2 for k in sets[Z]]")[0]
    assert "preview: 0, 1, 1, 4, 4, 9, 9, 16, …" in result.message


@pytest.mark.parametrize(
    "operation",
    ["length(S)", "total(S)", "mean(S)", "min(S)", "max(S)", "y = S x", "point[S, 0]", "S + 1"],
)
def test_infinite_sequence_needs_an_explicit_finite_selection(operation: str) -> None:
    result = rows("S = seq[1 for k in sets[N]]", operation)[1]
    assert result.is_error and "finite sequence terms" in result.message


def test_finite_selection_can_be_aggregated_and_plotted() -> None:
    results = rows(
        "S = seq[k for k in sets[N]]", "total(S[1...3])", "length(S[1...4])", "y = S[1...3] x"
    )
    assert results[1].message == "= 6" and results[2].message == "= 4"
    assert isinstance(results[3].plot, PlotGroup) and len(results[3].plot.plots) == 3


def test_generate_a_term_beyond_the_display_preview_and_through_aliases() -> None:
    results = rows(
        "S = seq[2k + 1 for k in sets[N]]",
        "S[1000]",
        "S[999...1001]",
        "T = S",
        "T[1000]",
        "seq[k for k in sets[Z]][1000]",
    )
    assert [r.message for r in results[1:3]] == ["= 2001", "= list[1999, 2001, 2003]"]
    assert results[4].message == "= 2001" and results[5].message == "= 500"


def test_generator_calls_user_function_and_tracks_constants() -> None:
    results = rows("f(t) = t^2 + a", "a = 3", "S = seq[f(k) for k in sets[N]]", "S[1000]")
    assert not any(r.is_error for r in results)
    assert results[-1].message == "= 1000003"


@pytest.mark.parametrize("index", ["0", "-1", "1.5", "x", "point[1, 2]"])
def test_sequence_indices_must_be_positive_whole_numbers(index: str) -> None:
    assert rows("S = seq[k for k in sets[N]]", f"S[{index}]")[1].is_error


@pytest.mark.parametrize(
    "text",
    [
        "seq[k for k in R]",
        "seq[k for k]",
        "seq[x k for k in sets[N]]",
        "seq[k]",
        "seq[]",
        "seq[1 for diff in sets[N]]",
    ],
)
def test_invalid_generators_are_rejected(text: str) -> None:
    assert rows(text)[0].is_error


def test_set_notation_is_not_used_for_sequences() -> None:
    with pytest.raises(ParseError, match="ordered sequence"):
        parse("set[k for k in sets[N]]")


def test_in_and_for_stay_names_outside_their_syntax() -> None:
    from opengraph.engine.compiler import CompileOptions

    cx = CompileOptions(complex_mode=True)
    texts = ["n = 2", "3in", "2 in", "x in"]
    results = compile_document(list(enumerate(texts)), cx)
    assert results[1].message == "= 6i"  # compact imaginary product i·n
    assert results[2].message == "= 4i"
    assert not results[3].is_error
    # Real mode: a stray keyword says how to use it rather than offering a slider.
    stray = rows("2 in")[0]
    assert stray.is_error and "keyword" in stray.message and stray.undefined == ()


@pytest.mark.parametrize(
    "prefix",
    [
        "seq[]",
        "seq[2k]",
        "seq[2k + 1 f]",
        "seq[2k + 1 for]",
        "seq[2k + 1 for k]",
        "seq[2k + 1 for k i]",
        "seq[2k + 1 for k in]",
        "S = seq[2k for k in ]",
        "seq[2k + 1 for k in s]",
        "seq[2k + 1 for k in sets[]]",
        "k in",
        "k ∈",
    ],
)
def test_typing_a_sequence_shows_hints_not_errors(prefix: str) -> None:
    """The editor closes ']' ahead of the caret, so these are what typing passes through."""
    result = rows(prefix)[0]
    assert result.is_error and result.incomplete, result.message


def test_a_finished_wrong_sequence_is_still_an_error() -> None:
    for text in ("seq[2k] + 1", "seq[2k for k in R]", "seq[2k for k on N]"):
        result = rows(text)[0]
        assert result.is_error and not result.incomplete, text
