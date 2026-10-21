"""Headless tests of the math editor's tree model, source mapping and editing commands."""

from __future__ import annotations

import math
import random

import pytest

from opengraph.engine.compiler import compile_expression
from opengraph.engine.parser import parse
from opengraph.engine.plots import ExplicitPlot
from opengraph.ui.mathedit.editing import EditorModel
from opengraph.ui.mathedit.nodes import (
    BigOp,
    Char,
    Frac,
    MathList,
    Node,
    Paren,
    Position,
    Root,
    Sub,
    Sup,
    Symbol,
    path_of,
    position_at,
)
from opengraph.ui.mathedit.source import from_source, node_text, to_source


def typed(keys: str, model: EditorModel | None = None) -> EditorModel:
    """Type ``keys`` one character at a time, like a user would; → and ← are arrow keys."""
    model = model or EditorModel()
    for key in keys:
        if key in "→←":
            model.move(1 if key == "→" else -1)
        else:
            model.type_text(key)
    return model


def loaded(text: str) -> str:
    model = EditorModel(text)
    model.anchor = None
    model.cursor = Position(model.root, 0)
    return model.describe().removeprefix("|")


def value_at(source: str, x: float) -> float:
    plot = compile_expression(f"y = {source}").plot
    assert isinstance(plot, ExplicitPlot), compile_expression(f"y = {source}").message
    return plot.fn.scalar(x=x)


# -- source -> tree -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "structure"),
    [
        ("y = x^2", "y = x^{2}"),
        ("x**2", "x^{2}"),
        ("x^(2x+1)", "x^{2x+1}"),
        ("x^((a))", "x^{\\left(a\\right)}"),
        ("2^3^2", "2^{3^{2}}"),
        ("2^-x", "2^{-x}"),
        ("1/2", "\\frac{1}{2}"),
        ("(x+1)/(x-1)", "\\frac{x+1}{x-1}"),
        ("2x/3", "\\frac{2x}{3}"),  # the parser groups (2x)/3 ...
        ("1/2x", "\\frac{1}{2}x"),  # ... and (1/2)x
        ("1/2/3", "\\frac{\\frac{1}{2}}{3}"),
        ("x^2/2", "\\frac{x^{2}}{2}"),
        ("e^-x/2", "\\frac{e^{-x}}{2}"),
        ("a + b/c", "a + \\frac{b}{c}"),
        ("sqrt(x+1)", "\\sqrt{x+1}"),
        ("cbrt(8)", "\\sqrt[3]{8}"),
        ("sqrt(x)/2", "\\frac{\\sqrt{x}}{2}"),
        ("sqrt(1, 2)", "sqrt\\left(1, 2\\right)"),  # not a valid radical: brackets
        ("a_1 + b_xy", "a_{1} + b_{xy}"),
        ("theta_2", "\\theta _{2}"),
        ("2pi x", "2\\pi  x"),
        ("pix", "pix"),
        ("x <= 2", "x <= 2"),
        ("(1/2, 3)", "\\left(\\frac{1}{2}, 3\\right)"),
        ("{[(x)]}", "\\left{\\left[\\left(x\\right)\\right]\\right}"),
        ("(x+1)^2", "\\left(x+1\\right)^{2}"),
        ("(x]", "\\left(x\\right]"),  # mixed brackets retain both delimiters
        ("|x|/2", "\\frac{|x|}{2}"),
        ("f(x) = x/2", "f\\left(x\\right) = \\frac{x}{2}"),
    ],
)
def test_loading_recognises_structure(text: str, structure: str) -> None:
    assert loaded(text) == structure


@pytest.mark.parametrize(
    ("text", "structure"),
    [
        ("y = sin(x", "y = sin(x"),
        ("1/", "1/"),
        ("x^2 +", "x^{2} +"),  # unparsable, but the exponent is still recognised
        ("y = x^(2", "y = x^(2"),
        ("sqrt(x) +* 2", "\\sqrt{x} +* 2"),
        ("y = $x^2", "y = $x^{2}"),  # an unknown character is kept as text
        ("", ""),
    ],
)
def test_invalid_text_loads_with_partial_structure(text: str, structure: str) -> None:
    assert loaded(text) == structure


def test_loading_keeps_the_text_verbatim() -> None:
    for text in ("y=1/2x", "x**2", "  y = sin( x )", "2^-x"):
        model = EditorModel(text)
        assert model.source == text  # loading a document never rewrites it


def test_source_map_of_loaded_text() -> None:
    root, source_map = from_source("y = (x+1)/2 + a_1")
    frac = next(n for n in root.nodes if isinstance(n, Frac))
    sub = next(n for n in root.nodes if isinstance(n, Sub))
    assert source_map[id(frac)] == (4, 11)
    assert source_map[id(sub)] == (15, 17)
    assert source_map[id(frac.num.nodes[0])] == (5, 6)  # the x


# -- tree -> source -----------------------------------------------------------------------

ROUND_TRIP = [
    "y = x^2",
    "x^2 + y^2 = 4",
    "y = 2x^2 - 3x + 1",
    "f(x) = (x+1)/(x-1)",
    "1/2x",
    "2x/3",
    "1/2/3",
    "8 / 4 / 2",
    "x^2/2",
    "e^-x/2",
    "2^-x*3",
    "-x^2",
    "-1/2",
    "2^3^2",
    "2^3x",
    "x^(1/3)",
    "(1/2)^2",
    "sqrt(x)/2",
    "sqrt(x^2 + 1)",
    "cbrt(x + 1)^2",
    "3 sin(x)",
    "sin(x)^2",
    "|x|/2",
    "2|x - 1|",
    "a_1 x + b_2",
    "theta_1 + theta",
    "2pi x",
    "2pix",
    "x**2",
    "max(1, x/2, 3)",
    "(1/2, -3/4)",
    "y = x/(2x)",
    "1/(2x)",
    "a/b c",
    "x/y^2",
    "x <= 1/2",
    "g(a, b) = a b/2",
    "1e-3/x",
    ".5x/2",
]


@pytest.mark.parametrize("text", ROUND_TRIP)
def test_round_trip_preserves_meaning(text: str) -> None:
    root, _ = from_source(text)
    source, _ = to_source(root)
    assert parse(source) == parse(text)  # AST equality ignores spans and spelling
    again, _ = to_source(from_source(source)[0])
    assert again == source  # writing back is stable


def frac(num: str, den: str) -> Frac:
    return Frac(from_source(num)[0], from_source(den)[0])


def sup(text: str) -> Sup:
    return Sup(from_source(text)[0])


def chars(text: str) -> list[Node]:
    return [Char(c) for c in text]


@pytest.mark.parametrize(
    ("nodes", "source"),
    [
        ([frac("1", "2")], "1/2"),
        ([frac("x+1", "2")], "(x+1)/2"),
        ([frac("", "")], "()/()"),
        ([*chars("3"), frac("1", "2")], "3(1/2)"),  # not 32/2
        ([frac("1", "2"), *chars("3")], "(1/2)3"),  # not 1/23
        ([frac("1", "x"), *chars("y")], "(1/x)y"),  # not 1/(xy)
        ([frac("1", "x"), *chars("(2)")], "(1/x)(2)"),  # not 1/x(2), a call
        ([*chars("c/"), frac("a", "b")], "c/(a/b)"),
        ([*chars("a*"), frac("b", "c")], "a*(b/c)"),  # would read back as (a*b)/c
        ([*chars("a-"), frac("b", "c")], "a-b/c"),
        ([frac("a", "b"), *chars("*c")], "a/b*c"),
        ([frac("1", "2"), sup("2")], "(1/2)^2"),
        ([*chars("y = "), frac("1", "2"), *chars(" + 1")], "y = 1/2 + 1"),
        ([*chars("x"), sup("2")], "x^2"),
        ([*chars("x"), sup("2"), *chars("3")], "x^(2)3"),  # not x^23
        ([*chars("x"), sup("a"), *chars("(2)")], "x^(a)(2)"),  # not x^a(2)
        ([*chars("x"), sup("2x")], "x^(2x)"),
        ([*chars("x"), sup("-1")], "x^(-1)"),
        ([*chars("x"), sup("")], "x^()"),
        ([*chars("2"), Symbol("pi"), *chars("x")], "2pi x"),
        ([*chars("x"), Symbol("theta")], "x theta"),
        ([*chars("a"), Sub(from_source("1")[0]), *chars("b")], "a_1 b"),
        ([*chars("a"), Sub(from_source("1")[0]), *chars("2")], "a_1 2"),
        ([*chars("x"), Root("sqrt", from_source("2")[0])], "x sqrt(2)"),
        ([*chars("2"), Root("sqrt", from_source("2")[0])], "2sqrt(2)"),
        ([Root("cbrt", MathList())], "cbrt()"),
    ],
)
def test_serialisation_keeps_layout_grouping(nodes: list[Node], source: str) -> None:
    text, source_map = to_source(MathList(nodes))
    assert text == source
    for node in nodes:  # every top-level node maps to a non-empty span
        start, end = source_map[id(node)]
        assert 0 <= start < end <= len(text)


def test_source_map_of_serialised_tree() -> None:
    f = frac("1", "x")
    text, source_map = to_source(MathList([f, *chars("y")]))
    assert text == "(1/x)y"
    assert source_map[id(f)] == (0, 5)
    assert text[slice(*source_map[id(f.den.nodes[0])])] == "x"


def _random_tree(rng: random.Random, depth: int) -> tuple[MathList, object]:
    """A random well-formed expression tree and an evaluator for what it *shows*."""
    terms: list[Node] = []
    values: list[tuple[str, object]] = []
    for i in range(rng.randint(1, 3)):
        if i:
            op = rng.choice("+-*")
            terms.append(Char(op))
            values.append(("op", op))
        atom, value = _random_atom(rng, depth)
        terms.extend(atom)
        if rng.random() < 0.3:
            exponent, power = _random_atom(rng, depth - 1, small=True)
            terms.append(Sup(MathList(exponent)))
            value = ("pow", value, power)
        values.append(("term", value))
    return MathList(terms), values


def _random_atom(rng: random.Random, depth: int, small: bool = False) -> tuple[list[Node], object]:
    kind = rng.choice(["num", "x", "frac", "root", "group"] if depth > 0 else ["num", "x"])
    if small and kind in ("root", "group"):
        kind = "num"
    if kind == "num":
        n = rng.randint(1, 9)
        return [Char(str(n))], float(n)
    if kind == "x":
        return [Char("x")], "x"
    num, nv = _random_tree(rng, depth - 1)
    if kind == "root":
        return [Root("sqrt", num)], ("sqrt", nv)
    if kind == "group":
        return [Paren("(", num)], nv
    den, dv = _random_tree(rng, depth - 1)
    return [Frac(num, den)], ("div", nv, dv)


def _evaluate(value: object, x: float) -> float:
    if isinstance(value, float):
        return value
    if value == "x":
        return x
    if isinstance(value, list):  # terms joined by + - * with the usual precedence
        total, product, sign = 0.0, None, 1.0
        pending = None
        for kind, item in value:
            if kind == "op":
                pending = item
                continue
            v = _evaluate(item, x)
            if pending == "*":
                assert product is not None
                product *= v
                continue
            if product is not None:
                total += sign * product
            sign = -1.0 if pending == "-" else 1.0
            product = v
        assert product is not None
        return total + sign * product
    assert isinstance(value, tuple)
    if value[0] == "pow":
        result: float | complex = _evaluate(value[1], x) ** _evaluate(value[2], x)
        if isinstance(result, complex):
            raise ValueError("not real")
        return result
    if value[0] == "sqrt":
        return math.sqrt(_evaluate(value[1], x))
    return _evaluate(value[1], x) / _evaluate(value[2], x)


def test_random_trees_serialise_to_what_they_show() -> None:
    rng = random.Random(1234)
    x = 1.37
    checked = 0
    for _ in range(600):
        tree, meaning = _random_tree(rng, 3)
        try:
            expected = _evaluate(meaning, x)
        except (ZeroDivisionError, ValueError, OverflowError):
            continue
        if not math.isfinite(expected) or abs(expected) > 1e12:
            continue
        source, _ = to_source(tree)
        assert value_at(source, x) == pytest.approx(expected, rel=1e-9), source
        reloaded, _ = to_source(from_source(source)[0])
        assert reloaded == source
        checked += 1
    assert checked > 300  # most samples are real-valued and finite


# -- typing ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("keys", "structure", "source"),
    [
        ("y = x^2", "y = x^{2|}", "y = x^2"),
        ("y = x^2+1", "y = x^{2}+1|", "y = x^2+1"),
        ("y = x^2→+1", "y = x^{2}+1|", "y = x^2+1"),
        ("x^-1", "x^{-1|}", "x^(-1)"),
        ("x^2 y", "x^{2} y|", "x^(2) y"),
        ("x^(2+1)", "x^{\\left(2+1\\right)|}", "x^(2+1)"),
        ("(x^2)", "\\left(x^{2}\\right)|", "(x^2)"),  # ")" closes the bracket around it
        ("(", "\\left(|\\right)", "()"),  # brackets close themselves
        ("[1,2", "\\left[1,2|\\right]", "[1,2]"),
        ("{[(x)]}", "\\left{\\left[\\left(x\\right)\\right]\\right}|", "{[(x)]}"),
        ("(x]", "\\left(x\\right]|", "(x]"),
        ("[)", "\\left[\\right)|", "[)"),
        ("[0,1)", "\\left[0,1\\right)|", "[0,1)"),
        ("(0,1]", "\\left(0,1\\right]|", "(0,1]"),
        ("point[1,2]", "point\\left[1,2\\right]|", "point[1,2]"),
        ("1/2→+x", "\\frac{1}{2}+x|", "1/2+x"),
        ("1/2+x", "\\frac{1}{2}+x|", "1/2+x"),
        ("sqrt(x^2)", "\\sqrt{x^{2}}|", "sqrt(x^2)"),
        ("1/2", "\\frac{1}{2|}", "1/2"),
        ("2x/3", "\\frac{2x}{3|}", "(2x)/3"),
        ("x+1/2", "x+\\frac{1}{2|}", "x+1/2"),
        ("(x+1)/2", "\\frac{x+1}{2|}", "(x+1)/2"),  # the bar replaces the parentheses
        ("sin(x)/x", "\\frac{sin\\left(x\\right)}{x|}", "(sin(x))/x"),
        ("x^2→/2", "\\frac{x^{2}}{2|}", "(x^2)/2"),
        ("x^2/2", "\\frac{x^{2}}{2|}", "(x^2)/2"),
        ("e^-x→/2", "\\frac{e^{-x}}{2|}", "(e^(-x))/2"),
        ("/", "\\frac{|}{}", "()/()"),
        ("(1/x)", "\\left(\\frac{1}{x}\\right)|", "(1/x)"),
        ("sqrt", "sqrt|", "sqrt"),
        ("sqrt(x)+1", "\\sqrt{x}+1|", "sqrt(x)+1"),
        ("(sqrtx)", "\\left(sqrtx\\right)|", "(sqrtx)"),
        ("cbrt8", "cbrt8|", "cbrt8"),
        ("2theta", "2theta|", "2theta"),
        ("pix", "pix|", "pix"),
        ("epsilon", "epsilon|", "epsilon"),
        ("a_1b", "a_{1b|}", "a_1b"),
        ("a_1+b", "a_{1}+b|", "a_1+b"),
        ("x<=2", "x<=2|", "x<=2"),
        ("y=(1,2)", "y=\\left(1,2\\right)|", "y=(1,2)"),
    ],
)
def test_typing_builds_structure(keys: str, structure: str, source: str) -> None:
    model = typed(keys)
    assert model.describe() == structure
    assert model.source == source


@pytest.mark.parametrize("text", ["[)", "(]", "[0,1)", "(0,1]", "point[[0,1),2]"])
def test_mixed_brackets_survive_reload_paste_and_undo(text: str) -> None:
    model = typed(text)
    assert model.source == text
    reloaded, _ = from_source(model.source)
    assert to_source(reloaded)[0] == text
    model.type_text("+2")
    assert model.source == text + "+2"
    assert model.undo()
    assert model.source == text
    assert model.redo()
    assert model.source == text + "+2"
    pasted = EditorModel()
    assert pasted.paste(text)
    assert pasted.source == text
    assert pasted.root.nodes
    assert any(isinstance(node, Paren) for node in pasted.root.nodes)


def test_changing_an_auto_closer_is_undoable_and_updates_source_map() -> None:
    model = typed("[0,1")
    model.move(-1)
    model.move(1)  # start a new edit after navigating to the interval endpoint
    revision = model.revision
    assert model.type_text(")")
    assert model.revision > revision
    assert model.source == "[0,1)"
    pair = model.root.nodes[0]
    assert isinstance(pair, Paren) and pair.right == ")"
    assert model.source_map[id(pair)] == (0, 5)
    assert model.cursor == Position(model.root, 1)
    assert model.undo()
    assert model.source == "[0,1]"
    assert model.redo()
    assert model.source == "[0,1)"


@pytest.mark.parametrize("text,closer", [("[0,1", ")"), ("(0,1", "]")])
def test_loaded_unfinished_interval_accepts_a_mixed_closer(text: str, closer: str) -> None:
    model = typed(closer, EditorModel(text))
    assert model.source == text + closer
    assert len(model.root.nodes) == 1
    assert isinstance(model.root.nodes[0], Paren)
    assert model.cursor == Position(model.root, 1)


@pytest.mark.parametrize("keys", ["point[[0,1),2]", "point[(0,1],2]", "[0,1/2)", "(0,2^3]"])
def test_mixed_closers_leave_the_nearest_enclosure(keys: str) -> None:
    model = typed(keys)
    assert model.cursor == Position(model.root, len(model.root))
    assert not compile_expression(model.source).is_error
    assert to_source(from_source(model.source)[0])[0] == model.source


@pytest.mark.parametrize("keys", ["(0,1]/2", "[0,1)/2", "(0,1]^2"])
def test_fraction_and_exponent_keep_mixed_interval_brackets(keys: str) -> None:
    model = typed(keys)
    interval = "(0,1]" if keys.startswith("(") else "[0,1)"
    assert interval in model.source
    assert interval in to_source(from_source(model.source)[0])[0]


@pytest.mark.parametrize(
    ("keys", "x", "expected"),
    [
        ("2^3x", 1.0, 8.0),  # 2^(3x): what is shown, not the text reading (2^3)x
        ("1/2x", 2.0, 1.0),  # 1/(2x)
        ("x^2→+1", 3.0, 10.0),
        ("x^2+1", 3.0, 10.0),
        ("e^-x→/2", 0.0, 0.5),
        ("{x+1}list[2]", 2.0, 6.0),
        ("sqrt(x+1)*2", 3.0, 4.0),
        ("(x+1)/(x-1)^2", 3.0, 1.0),
    ],
)
def test_typed_expressions_compile_to_what_is_shown(keys: str, x: float, expected: float) -> None:
    assert value_at(typed(keys).source, x) == pytest.approx(expected)


def test_typing_replaces_and_wraps_selection() -> None:
    model = typed("a+b")
    model.select_all()
    model.type_text("/")
    assert model.describe() == "\\frac{a+b}{|}"
    model = typed("a+b")
    model.select_all()
    model.type_text("^")
    assert model.describe() == "^{a+b|}"
    model = typed("a+b")
    model.move(-1, select=True)
    model.type_text("(")
    assert model.describe() == "a+\\left(b\\right)|"
    model.select_all()
    model.type_text("z")
    assert model.describe() == "z|"


def test_caret_exponent_reuses_existing_exponent() -> None:
    model = typed("x^2")
    model.move(1)  # out of the exponent
    model.type_text("^")
    assert model.describe() == "x^{2|}"


# -- caret movement and selection ---------------------------------------------------------


def test_left_right_visit_every_block() -> None:
    model = EditorModel("1/2 + x^3")
    model.move_to_edge(-1)
    seen = [model.describe()]
    for _ in range(8):
        model.move(1)
        seen.append(model.describe())
    assert seen == [
        "|\\frac{1}{2} + x^{3}",
        "\\frac{|1}{2} + x^{3}",
        "\\frac{1|}{2} + x^{3}",
        "\\frac{1}{|2} + x^{3}",
        "\\frac{1}{2|} + x^{3}",
        "\\frac{1}{2}| + x^{3}",
        "\\frac{1}{2} |+ x^{3}",
        "\\frac{1}{2} +| x^{3}",
        "\\frac{1}{2} + |x^{3}",
    ]
    model.move_to_edge(1)
    model.move(-1)
    assert model.describe() == "\\frac{1}{2} + x^{3|}"
    model.move(-1)
    model.move(-1)
    assert model.describe() == "\\frac{1}{2} + x|^{3}"


def test_caret_stays_inside_the_expression() -> None:
    model = EditorModel("x")
    model.move(1)
    assert model.describe() == "x|"
    model.move_to_edge(-1)
    model.move(-1)
    assert model.describe() == "|x"


def test_shift_selection_grows_to_whole_structures() -> None:
    model = EditorModel("a + 1/2")
    model.move(-1)  # caret in the denominator
    model.move(-1, select=True)
    assert model.describe() == "a + \\frac{1}{[2]}"
    model.move(-1, select=True)  # leaves the block: the whole fraction
    assert model.describe() == "a + [\\frac{1}{2}]"
    assert model.selected_source() == "1/2"
    model.move(-1, select=True)
    assert model.describe() == "a +[ \\frac{1}{2}]"
    model.move(1)  # collapse to the selection's end
    assert model.describe() == "a + \\frac{1}{2}|"


def test_selection_across_blocks_is_the_structure() -> None:
    model = EditorModel("x + 1/2")
    frac_node = model.root.nodes[-1]
    assert isinstance(frac_node, Frac)
    model.set_cursor(Position(frac_node.num, 0))
    model.set_cursor(Position(frac_node.den, 1), select=True)
    assert model.selection() == (model.root, 4, 5)
    model.delete_selection()
    assert model.source == "x + "


def test_vertical_targets() -> None:
    model = EditorModel("1/2")
    f = model.root.nodes[0]
    assert isinstance(f, Frac)
    model.set_cursor(Position(f.num, 1))
    assert model.vertical_target(1) is f.den
    assert model.vertical_target(-1) is None  # leave the row
    model.set_cursor(Position(f.den, 0))
    assert model.vertical_target(-1) is f.num
    model = typed("x^2")
    assert model.vertical_target(1) is model.root
    assert model.vertical_target(-1) is None


def test_paths_round_trip() -> None:
    model = EditorModel("a + sqrt(1/x^2)")
    root_node = model.root.nodes[-1]
    assert isinstance(root_node, Root)
    inner = root_node.body.nodes[0]
    assert isinstance(inner, Frac)
    pos = Position(inner.den, 1)
    assert position_at(model.root, path_of(pos)) == pos
    assert position_at(model.root, (99, 0, 0)) == Position(model.root, len(model.root))


# -- deletion ---------------------------------------------------------------------------


def test_backspace_enters_then_removes_structures() -> None:
    model = typed("x^2")
    model.move(1)
    assert model.describe() == "x^{2}|"
    assert not model.backspace()  # steps into the exponent, changes nothing
    assert model.describe() == "x^{2|}"
    assert model.backspace()
    assert model.describe() == "x^{|}"
    assert model.backspace()  # empty exponent: removed
    assert model.describe() == "x|"
    assert model.source == "x"


def test_backspace_at_block_start_dissolves_structure() -> None:
    model = EditorModel("1/2")
    f = model.root.nodes[0]
    assert isinstance(f, Frac)
    model.set_cursor(Position(f.den, 0))
    model.backspace()
    assert model.describe() == "1|2"
    model = EditorModel("sqrt(x+1)")
    r = model.root.nodes[0]
    assert isinstance(r, Root)
    model.set_cursor(Position(r.body, 0))
    model.backspace()
    assert model.describe() == "|x+1"


def test_delete_forward_mirrors_backspace() -> None:
    model = EditorModel("a^2b")
    model.move_to_edge(-1)
    model.move(1)
    model.delete_forward()  # into the exponent
    assert model.describe() == "a^{|2}b"
    model.delete_forward()
    model.delete_forward()  # at the end of the empty exponent: dissolve it
    assert model.describe() == "a|b"


def test_deleting_a_symbol_removes_it_whole() -> None:
    model = typed(r"2\theta ")
    model.backspace()
    assert model.source == "2"


# -- undo, clipboard ------------------------------------------------------------------------


def test_typing_is_one_undo_step_per_run() -> None:
    model = typed("12+3")
    model.type_text("^")
    model.type_text("2")
    assert model.source == "12+3^2"
    model.undo()
    assert model.describe() == "12+3^{|}"
    model.undo()
    assert model.describe() == "12+3|"
    model.undo()
    assert model.source == "" and not model.can_undo
    model.redo()
    model.redo()
    model.redo()
    assert model.source == "12+3^2"
    assert not model.can_redo


def test_moving_the_caret_starts_a_new_undo_step() -> None:
    model = typed("ab")
    model.move(-1)
    model.type_text("c")
    model.undo()
    assert model.describe() == "a|b"


def test_undo_restores_letters_of_auto_command() -> None:
    model = typed("pi+")
    assert model.source == "pi+" and isinstance(model.root.nodes[0], Symbol)
    model.undo()
    assert model.describe() == "pi|" and all(isinstance(n, Char) for n in model.root.nodes)


def test_undo_returns_to_loaded_text_verbatim() -> None:
    model = EditorModel("y=1/2x")
    model.type_text("+")
    assert model.source == "y=(1/2)x+"
    model.undo()
    assert model.source == "y=1/2x"


def test_set_source_is_undoable() -> None:
    model = EditorModel("k = 1")
    model.set_source("k = 2")
    model.undo()
    assert model.source == "k = 1"


def test_copy_and_paste_round_trip() -> None:
    model = EditorModel("y = (x+1)/2 + a_1")
    model.select_all()
    copied = model.selected_source()
    assert parse(copied) == parse("y = (x+1)/2 + a_1")
    target = typed("z + ")
    assert target.paste("sqrt(x)/2")
    assert target.describe() == "z + \\frac{\\sqrt{x}}{2}|"
    assert target.source == "z + (sqrt(x))/2"
    target.select_all()
    target.paste("line one\nline two")
    assert target.source == "line one line two"
    assert node_text([Frac(MathList([Char("1")]), MathList([Char("2")]))]) == "1/2"


def test_no_change_means_no_undo_step() -> None:
    model = EditorModel("x")
    model.move_to_edge(-1)
    assert not model.backspace()
    assert not model.can_undo


# -- big operators (Σ, Π, ∫, lim) ------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "structure"),
    [
        ("sum(x^2,x,0,100)", r"\sum_{x=0}^{100}{x^{2}}"),
        ("Σ(k^2, 1, n)", r"\sum_{=1}^{n}{k^{2}}"),  # the variable is inferred
        ("Sigma(i, 1, n) + 1", r"\sum_{=1}^{n}{i} + 1"),
        ("prod(k, k, 1, 4)", r"\prod_{k=1}^{4}{k}"),
        ("int(t^2, t, 0, 1)", r"\int_{0}^{1}{t^{2}}d{t}"),
        ("∫(x^2, 0, 1)", r"\int_{0}^{1}{x^{2}}d{}"),
        ("limit(sin(x)/x, x, 0)", r"\lim_{x\to 0^{}}{\frac{sin\left(x\right)}{x}}"),
        ("limit(1/x, x, 0, right)", r"\lim_{x\to 0^{right}}{\frac{1}{x}}"),
        ("y = 2sum(k, k, 1, x)", r"y = 2\sum_{k=1}^{x}{k}"),
        ("sum(x)", r"sum\left(x\right)"),  # too few arguments: stays a call
        ("limit(x, 0)", r"limit\left(x, 0\right)"),
    ],
)
def test_loading_recognises_big_operators(text: str, structure: str) -> None:
    assert loaded(text) == structure


@pytest.mark.parametrize(
    ("keys", "structure", "source"),
    [
        ("sum(x^2,x,0,100)", r"\sum_{x=0}^{100}{x^{2}}|", "sum(x^2, x, 0, 100)"),
        ("Σ(k^2,1,n)", r"\sum_{=1}^{n}{k^{2}}|", "Σ(k^2, 1, n)"),  # the bounds move up
        ("Sigma(k,k,1,3)", r"\sum_{k=1}^{3}{k}|", "Sigma(k, k, 1, 3)"),
        ("prod(k+1,k,1,4)", r"\prod_{k=1}^{4}{k+1}|", "prod(k+1, k, 1, 4)"),
        ("y=int(t,t,0,x)→+1", r"y=\int_{0}^{x}{t}d{t}+1|", "y=int(t, t, 0, x)+1"),
        (
            "limit(sin(x)/x,x,0)",
            r"\lim_{x\to 0^{}}{\frac{sin\left(x\right)}{x}}|",
            "limit((sin(x))/x, x, 0)",
        ),
        ("limit(1/x,x,0,right)", r"\lim_{x\to 0^{right}}{\frac{1}{x}}|", "limit(1/x, x, 0, right)"),
        ("sum((1,2),k,0,1)", r"\sum_{k=0}^{1}{\left(1,2\right)}|", "sum((1,2), k, 0, 1)"),
        ("sum(", r"\sum_{=}^{}{|}", "sum(, , )"),
        ("summit(", r"summit\left(|\right)", "summit()"),  # not an operator name
    ],
)
def test_typing_builds_big_operators(keys: str, structure: str, source: str) -> None:
    model = typed(keys)
    assert model.describe() == structure
    assert model.source == source


@pytest.mark.parametrize(
    ("source", "x", "expected"),
    [
        ("sum(k^2, k, 1, 4)", 0.0, 30.0),
        ("Σ(k, k, 1, 3)", 0.0, 6.0),
        ("prod(k, k, 1, 4)", 0.0, 24.0),
        ("int(t, t, 0, x)", 2.0, 2.0),
    ],
)
def test_big_operators_round_trip(source: str, x: float, expected: float) -> None:
    root, _ = from_source(source)
    assert isinstance(root.nodes[0], BigOp)
    text, _ = to_source(root)
    assert value_at(text, x) == pytest.approx(expected)
    assert to_source(from_source(text)[0])[0] == text


def test_big_operator_slots_in_reading_order() -> None:
    model = EditorModel("sum(x^2, x, 0, 100)")
    model.move_to_edge(-1)
    seen = []
    for _ in range(14):
        model.move(1)
        seen.append(model.describe())
    assert seen[:3] == [
        r"\sum_{|x=0}^{100}{x^{2}}",
        r"\sum_{x|=0}^{100}{x^{2}}",
        r"\sum_{x=|0}^{100}{x^{2}}",
    ]
    assert seen[-1] == r"\sum_{x=0}^{100}{x^{2}}|"
    op = model.root.nodes[0]
    assert isinstance(op, BigOp)
    model.set_cursor(Position(op.upper, 0))
    assert model.vertical_target(1) is op.var
    model.set_cursor(Position(op.lower, 0))
    assert model.vertical_target(-1) is op.upper


def test_backspace_turns_a_big_operator_back_into_a_call() -> None:
    model = EditorModel("sum(x^2, x, 0, 100)")
    op = model.root.nodes[0]
    assert isinstance(op, BigOp)
    model.set_cursor(Position(op.var, 0))
    assert model.backspace()
    assert model.describe() == r"sum\left(x^{2},|x,0,100\right)"
    assert parse(model.source) == parse("sum(x^2, x, 0, 100)")
    model = typed("sum(")
    assert model.backspace()
    assert model.source == "sum()"


def test_undo_gives_back_the_operator_name() -> None:
    model = typed("sum(")
    assert model.undo()
    assert model.describe() == "sum|"


def test_bare_big_operator_name_creates_empty_fields() -> None:
    # "sum" + space: variable = start below, end above, body last; no call syntax needed.
    model = typed("sum ")
    op = model.root.nodes[0]
    assert isinstance(op, BigOp)
    assert model.cursor.lst is op.var
    # Fill the fields in reading order with the arrow keys: var, start, end, body.
    typed("k→1→n→k^2", model)
    assert model.source == "sum(k^2, k, 1, n)"


def test_bare_big_operator_without_variable_infers_it() -> None:
    model = typed("sum →1→n→k", model=None)
    assert model.source == "sum(k, 1, n)"


def test_integral_name_starts_at_lower_bound() -> None:
    model = typed("int ")
    op = model.root.nodes[0]
    assert isinstance(op, BigOp)
    assert model.cursor.lst is op.lower


def test_names_merely_containing_an_operator_name_stay_text() -> None:
    assert typed("summit x").source == "summit x"
    assert typed("a sum").source == "a sum"


def test_bare_big_operator_name_undoes_to_letters() -> None:
    model = typed("sum ")
    assert model.undo()
    assert model.describe() == "sum|"
