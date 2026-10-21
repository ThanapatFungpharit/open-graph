from __future__ import annotations

import pytest

from opengraph.engine.ast import BinOp, Call, Name, Num, Relation, TupleExpr, UnaryOp
from opengraph.engine.errors import ParseError
from opengraph.engine.lexer import Kind, tokenize
from opengraph.engine.parser import parse


def show(node: object) -> str:
    """Compact fully-parenthesised rendering, for asserting structure."""
    match node:
        case Num(value=v):
            return f"{v:g}"
        case Name(name=n):
            return n
        case UnaryOp(operand=o):
            return f"(-{show(o)})"
        case BinOp(op=op, left=a, right=b):
            return f"({show(a)}{op}{show(b)})"
        case Call(name=n, args=args):
            return f"{n}[{','.join(show(a) for a in args)}]"
        case TupleExpr(items=items):
            return f"<{','.join(show(i) for i in items)}>"
        case Relation(op=op, left=a, right=b):
            return f"{show(a)} {op} {show(b)}"
    raise AssertionError(node)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1 + 2 * 3", "(1+(2*3))"),
        ("(1 + 2) * 3", "((1+2)*3)"),
        ("8 - 3 - 2", "((8-3)-2)"),
        ("8 / 4 / 2", "((8/4)/2)"),
        ("2 ^ 3 ^ 2", "(2^(3^2))"),
        ("-x^2", "(-(x^2))"),
        ("-2^2", "(-(2^2))"),
        ("2^-x", "(2^(-x))"),
        ("2^-x*3", "((2^(-x))*3)"),
        ("+x", "x"),
        ("--x", "(-(-x))"),
        ("2x", "(2*x)"),
        ("2x^2", "(2*(x^2))"),
        ("1/2x", "((1/2)*x)"),
        ("2^3x", "((2^3)*x)"),
        ("x2", "(x*2)"),
        ("2(x+1)", "(2*(x+1))"),
        ("(x+1)(x-1)", "((x+1)*(x-1))"),
        ("3 sin(x)", "(3*sin[x])"),
        ("sin(x)^2", "(sin[x]^2)"),
        ("max(1, x, 3)", "max[1,x,3]"),
        ("|x|", "abs[x]"),
        ("|x - |x||", "abs[(x-abs[x])]"),
        ("2|x|", "(2*abs[x])"),
        ("|x||x|", "(abs[x]*abs[x])"),
        ("(|x|)", "abs[x]"),
        ("x**2", "(x^2)"),
        ("2·x − 1", "((2*x)-1)"),
        ("point[1, 2]", "<1,2>"),
        ("y = x^2", "y = (x^2)"),
        ("f(x) = x", "f[x] = x"),
        ("x ≤ 2", "x <= 2"),
        ("1e3x", "(1000*x)"),
        (".5x", "(0.5*x)"),
        ("θ", "θ"),
        ("a_1 x", "(a_1*x)"),
    ],
)
def test_structure_and_precedence(text: str, expected: str) -> None:
    assert show(parse(text)) == expected


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "Empty"),
        ("   ", "Empty"),
        ("1 +", "incomplete"),
        ("(1 + 2", "Missing closing ')'"),
        ("1 + 2)", "Unmatched ')'"),
        ("2 3", "Unexpected '3'"),
        ("sin()", "needs an argument"),
        ("()", "Empty parentheses"),
        ("|x", "Missing closing '|'"),
        ("x $ 2", "Unexpected character '$'"),
        ("1 = 2 = 3", "Only one"),
        ("* 2", "Unexpected '*'"),
        ("1, 2", "several points as point"),
    ],
)
def test_syntax_errors(text: str, message: str) -> None:
    with pytest.raises(ParseError) as info:
        parse(text)
    assert message in info.value.message


def test_error_spans_point_at_the_problem() -> None:
    with pytest.raises(ParseError) as info:
        parse("x + 2 3")
    assert info.value.span == (6, 7)
    with pytest.raises(ParseError) as info:
        parse("sin(x")
    assert info.value.span == (3, 4)  # the unclosed parenthesis


def test_lexer_kinds_and_spans() -> None:
    tokens = tokenize("2.5x+sin(y)")
    assert [t.kind for t in tokens] == [
        Kind.NUMBER,
        Kind.NAME,
        Kind.OP,
        Kind.NAME,
        Kind.OP,
        Kind.NAME,
        Kind.OP,
        Kind.END,
    ]
    assert tokens[0].span == (0, 3)
    assert tokens[3].text == "sin"


def test_node_spans_cover_source() -> None:
    node = parse("12 + sin(x)")
    assert node.span == (0, 11)


def test_braces_group_like_parentheses() -> None:
    assert parse("{x + 1}2") == parse("(x + 1)2")
    assert parse("2{3}") == parse("2(3)")
    with pytest.raises(ParseError, match="Missing closing '}'"):
        parse("{x")
    with pytest.raises(ParseError, match="Expected '}'"):
        parse("{x)")
    with pytest.raises(ParseError, match="Unmatched '}'"):
        parse("x}")
