"""Conversion between source text and the editable tree, in both directions. Toolkit-free.

``from_source`` reuses the engine's lexer and parser: structures (fractions, exponents,
radicals) are recognised from the parsed AST and matched brackets from the tokens, so
the editor shows exactly the grouping the compiler will use (``2x/3`` is one fraction
with numerator ``2x``; ``1/2x`` is ½ times x). Text the parser rejects is still loaded,
token by token, with whatever structure can be recognised locally; anything else is kept
as plain characters.

``to_source`` writes the tree back as source text. It adds parentheses wherever the
two-dimensional layout groups something the flat text would not (a fraction's numerator
and denominator, an exponent, a fraction next to a higher-precedence neighbour) and
spaces where neighbouring tokens would otherwise merge, so ``parse(to_source(tree))``
always means what the tree shows.

Both directions also return a :data:`SourceMap` from node to source span, used to put
compiler error spans under the right part of the drawing.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from opengraph.engine.ast import BinOp, Call, ComparisonChain, Relation, Statement, UnaryOp
from opengraph.engine.ast import Node as AstNode
from opengraph.engine.ast import children as ast_children
from opengraph.engine.errors import MathError, Span
from opengraph.engine.lexer import Kind, Token, tokenize
from opengraph.engine.parser import parse
from opengraph.ui.mathedit.nodes import (
    BIG_OPERATORS,
    BRACKETS,
    CLOSERS,
    GREEK,
    ROOT_FUNCTIONS,
    BigOp,
    Char,
    Frac,
    MathList,
    Node,
    Paren,
    Root,
    Sub,
    Sup,
    Symbol,
    clone_node,
)

SourceMap = dict[int, Span]
"""``id(node)`` -> the node's span in the source text."""

# A bare fraction ``a/b`` keeps its grouping only between these neighbours (``/``, ``^``,
# a digit or a name next to it would regroup it); otherwise it is parenthesised. After
# ``*`` the value would survive, but ``a*b/c`` reads back as (a·b)/c, not a·(b/c).
_SAFE_BEFORE_FRAC = frozenset("+-=<>,([{−–≤≥")
_SAFE_AFTER_FRAC = frozenset("+-*/=<>,)]}−–×·⋅÷≤≥")
# Brackets that only group (``[…]`` is a list, so it is never added or removed).
_GROUPING = frozenset("({")
# A bare exponent ``^2`` must not be followed by anything that would extend it.
_SAFE_AFTER_SUP = _SAFE_AFTER_FRAC


# -- source -> tree ------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Spec:
    """A structure recognised in the source: ``[start, end)`` becomes one node."""

    start: int
    end: int
    kind: str  # frac | sup | sub | root | paren | symbol | op | bigop
    blocks: tuple[Span, ...] = ()
    name: str = ""


def _tokens(text: str) -> list[Token]:
    """Tokenise ``text``, blanking characters the lexer rejects (offsets are kept)."""
    chars = list(text)
    for _ in range(len(text) + 1):
        try:
            return tokenize("".join(chars))
        except MathError as exc:
            if exc.span is None:  # pragma: no cover - the lexer always gives a span
                break
            chars[exc.span[0]] = " "
    return []  # pragma: no cover


class _Loader:
    def __init__(self, text: str) -> None:
        self.text = text
        self.tokens = _tokens(text)
        self.at_start = {t.span[0]: i for i, t in enumerate(self.tokens) if t.kind is not Kind.END}
        self.at_end = {t.span[1]: i for i, t in enumerate(self.tokens) if t.kind is not Kind.END}
        # Round/square pairs may mix for intervals; braces must match exactly.
        self.match: dict[int, int] = {}
        stack: list[int] = []
        for i, token in enumerate(self.tokens):
            if token.kind is not Kind.OP:
                continue
            if token.text in BRACKETS:
                stack.append(i)
            elif token.text in CLOSERS and stack:
                left = self.tokens[stack[-1]].text
                if left != CLOSERS[token.text] and not (left in "([" and token.text in ")]"):
                    continue
                opener = stack.pop()
                self.match[opener] = i
                self.match[i] = opener
        self.specs: dict[int, list[_Spec]] = {}
        self.source_map: SourceMap = {}

    # -- recognising structures ---------------------------------------------------------
    def collect(self) -> None:
        for token in self.tokens:
            if token.kind is Kind.NAME:
                self.name_specs(token)
                if token.text == "in":
                    self.add(_Spec(*token.span, "op", name="in"))
            elif token.kind is Kind.OP and token.span[1] - token.span[0] > 1:
                self.add(_Spec(*token.span, "op", name=token.text))
        for opener, closer in self.match.items():
            if opener < closer:
                start, inner_start = self.tokens[opener].span
                inner_end, end = self.tokens[closer].span
                name = self.tokens[opener].text + self.tokens[closer].text
                self.add(_Spec(start, end, "paren", ((inner_start, inner_end),), name=name))
        try:
            tree: Statement | None = parse(self.text) if self.text.strip() else None
        except (MathError, RecursionError):
            tree = None
        if tree is not None:
            self.ast_specs(tree)
        else:
            self.fallback_specs()

    def add(self, spec: _Spec) -> None:
        self.specs.setdefault(spec.start, []).append(spec)

    def name_specs(self, token: Token) -> None:
        start, end = token.span
        name = token.text
        base_end = start + name.index("_") if "_" in name else end
        if self.text[start:base_end] in GREEK:
            self.add(_Spec(start, base_end, "symbol", name=self.text[start:base_end]))
        if base_end < end:
            self.add(_Spec(base_end, end, "sub", ((base_end + 1, end),)))

    def ast_specs(self, node: Statement) -> None:
        if isinstance(node, ComparisonChain):
            for comparison in node.comparisons:
                self.ast_specs(comparison)
            return
        if isinstance(node, BinOp) and node.op in "/^":
            left, right = self.extent(node.left), self.extent(node.right)
            if node.op == "/":
                self.add(_Spec(left[0], right[1], "frac", (self.strip(left), self.strip(right))))
            else:
                caret = next(t for t in self.tokens if t.span[0] >= left[1] and t.text == "^").span[
                    0
                ]
                self.add(_Spec(caret, right[1], "sup", (self.strip(right),)))
        elif isinstance(node, Call) and node.name in ROOT_FUNCTIONS and len(node.args) == 1:
            opener = self.at_start.get(node.name_span[1])
            closer = self.at_end.get(node.span[1])
            if opener is not None and closer is not None and self.match.get(opener) == closer:
                inner = (self.tokens[opener].span[1], self.tokens[closer].span[0])
                self.add(_Spec(*node.span, "root", (inner,), name=node.name))
        elif isinstance(node, Call) and node.name in BIG_OPERATORS:
            self.big_operator_spec(node)
        if isinstance(node, Relation):
            self.ast_specs(node.left)
            self.ast_specs(node.right)
        else:
            for child in ast_children(node):
                self.ast_specs(child)

    def big_operator_spec(self, node: Call) -> None:
        """``sum(body, k, a, b)``, ``Σ(body, a, b)``, ``limit(f, t, a[, side])``."""
        opener = self.at_start.get(node.name_span[1])
        closer = self.at_end.get(node.span[1])
        if opener is None or closer is None or self.match.get(opener) != closer:
            return
        args = [self.extent(arg) for arg in node.args]
        if BIG_OPERATORS[node.name] == "limit":
            if len(args) not in (3, 4):
                return
        elif len(args) == 3:
            args.insert(
                1, (args[0][1], args[0][1])
            )  # omitted variable: show the original input, including incomplete sums
        elif len(args) != 4:
            return
        while len(args) < 4:  # a two-sided limit
            args.append((args[-1][1], args[-1][1]))
        self.add(_Spec(*node.span, "bigop", tuple(args), name=node.name))

    def fallback_specs(self) -> None:
        """Structures that can be recognised from tokens alone, for unparsable text."""
        tokens = self.tokens
        for i, token in enumerate(tokens):
            if token.kind is not Kind.OP or token.text != "^" or i + 1 >= len(tokens):
                continue
            j = i + 1
            if tokens[j].kind is Kind.OP and tokens[j].text == "-" and j + 1 < len(tokens):
                j += 1
            target = tokens[j]
            if target.kind in (Kind.NUMBER, Kind.NAME):
                span = (tokens[i + 1].span[0], target.span[1])
                self.add(_Spec(token.span[0], target.span[1], "sup", (span,)))
            elif (
                target.text in _GROUPING
                and j in self.match
                and j == i + 1
                and tokens[self.match[j]].text == BRACKETS[target.text]
            ):
                closer = tokens[self.match[j]]
                inner = (target.span[1], closer.span[0])
                self.add(_Spec(token.span[0], closer.span[1], "sup", (inner,)))
        for i, token in enumerate(tokens[:-1]):
            opener = i + 1
            if (
                token.kind is Kind.NAME
                and token.text in ROOT_FUNCTIONS
                and tokens[opener].text == "("
                and opener in self.match
                and tokens[self.match[opener]].text == ")"
            ):
                closer = tokens[self.match[opener]]
                inner = (tokens[opener].span[1], closer.span[0])
                self.add(_Spec(token.span[0], closer.span[1], "root", (inner,), name=token.text))

    # -- spans -----------------------------------------------------------------------------
    def extent(self, node: AstNode) -> Span:
        """Source range of ``node`` including parentheses that wrap exactly it.

        AST spans leave out the parentheses of a group, and a binary node's span starts
        at its left operand's span, so both are reconstructed here from the tokens.
        """
        if isinstance(node, BinOp):
            start, end = self.extent(node.left)[0], self.extent(node.right)[1]
        elif isinstance(node, UnaryOp):
            start, end = node.span[0], self.extent(node.operand)[1]
        else:
            start, end = node.span
        while True:
            first, last = self.at_start.get(start), self.at_end.get(end)
            if first is None or last is None or first == 0:
                return start, end
            opener, closer = first - 1, last + 1
            if (
                self.tokens[opener].text not in _GROUPING
                or self.match.get(opener) != closer
                or self.tokens[closer].text != BRACKETS[self.tokens[opener].text]
            ):
                return start, end
            start, end = self.tokens[opener].span[0], self.tokens[closer].span[1]

    def strip(self, span: Span) -> Span:
        """The inside of ``span`` when it is one parenthesised group (the layout groups)."""
        first, last = self.at_start.get(span[0]), self.at_end.get(span[1])
        if first is None or last is None or self.match.get(first) != last:
            return span
        if (
            self.tokens[first].text in _GROUPING
            and self.tokens[last].text == BRACKETS[self.tokens[first].text]
        ):
            return self.tokens[first].span[1], self.tokens[last].span[0]
        return span

    # -- building ---------------------------------------------------------------------------
    def build(self, start: int, end: int) -> MathList:
        nodes: list[Node] = []
        i = start
        while i < end:
            fitting = [s for s in self.specs.get(i, ()) if s.end <= end and s.end > i]
            if fitting:
                spec = max(fitting, key=lambda s: s.end)
                node = self.make(spec)
                self.source_map[id(node)] = (spec.start, spec.end)
                nodes.append(node)
                i = spec.end
                continue
            ch = self.text[i]
            char = Char(" " if ch.isspace() else ch)
            self.source_map[id(char)] = (i, i + 1)
            nodes.append(char)
            i += 1
        return MathList(nodes)

    def make(self, spec: _Spec) -> Node:
        blocks = [self.build(*b) for b in spec.blocks]
        match spec.kind:
            case "frac":
                return Frac(blocks[0], blocks[1])
            case "sup":
                return Sup(blocks[0])
            case "sub":
                return Sub(blocks[0])
            case "root":
                return Root(spec.name, blocks[0])
            case "paren":
                return Paren(spec.name[0], blocks[0], spec.name[1])
            case "bigop":
                return BigOp(spec.name, *blocks)
            case "symbol":
                return Symbol(spec.name)
        return Char(spec.name)  # a multi-character operator such as "<="


def from_source(text: str) -> tuple[MathList, SourceMap]:
    """Load source text into a tree (never fails; unknown input stays as characters)."""
    loader = _Loader(text)
    loader.collect()
    root = loader.build(0, len(text))
    return root, loader.source_map


# -- tree -> source ------------------------------------------------------------------------


@dataclass(slots=True)
class _Fragment:
    text: str = ""
    spans: list[tuple[int, int, int]] = field(default_factory=list)  # (id, start, end)

    def append(self, other: _Fragment) -> None:
        offset = len(self.text)
        self.text += other.text
        self.spans.extend((key, s + offset, e + offset) for key, s, e in other.spans)

    def append_text(self, text: str) -> None:
        self.text += text

    def wrapped(self, before: str, after: str) -> _Fragment:
        result = _Fragment(before)
        result.append(self)
        result.append_text(after)
        return result


def _is_single_token(text: str) -> bool:
    """True for one number or one name, which can be an operand without parentheses."""
    try:
        tokens = tokenize(text)
    except MathError:
        return False
    return len(tokens) == 2 and tokens[0].kind in (Kind.NUMBER, Kind.NAME)


def _ungrouped(lst: MathList) -> tuple[MathList, Paren | None]:
    """The contents of ``lst`` when it is only a grouping bracket pair, like ``(x + 1)``.

    A numerator or exponent groups already, so its own brackets are written once (and
    loading the text drops them, as for typed-in ``(x+1)/2``). Also returns the bracket.
    """
    outer: Paren | None = None
    while len(lst.nodes) == 1:
        node = lst.nodes[0]
        if not (
            isinstance(node, Paren) and node.left in _GROUPING and node.right == BRACKETS[node.left]
        ):
            break
        outer = outer or node
        lst = node.body
    return lst, outer


def _operand(lst: MathList) -> _Fragment:
    """``lst`` as a self-contained operand: parenthesised unless it is a single token."""
    lst, group = _ungrouped(lst)
    inner = _serialize(lst)
    if not _is_single_token(inner.text):
        inner = inner.wrapped("(", ")")
    if group is not None:
        inner.spans.append((id(group), 0, len(inner.text)))
    return inner


def _first_visible(fragments: Sequence[_Fragment], start: int) -> str | None:
    for fragment in fragments[start:]:
        stripped = fragment.text.lstrip()
        if stripped:
            return stripped[0]
    return None


def _node_fragment(node: Node) -> _Fragment:
    """The node's preferred ("bare") text, before adjusting for its neighbours."""
    match node:
        case Char(text=text):
            fragment = _Fragment(text)
        case Symbol(name=name):
            fragment = _Fragment(name)
        case Frac():
            fragment = _operand(node.num)
            fragment.append_text("/")
            fragment.append(_operand(node.den))
        case Sup():
            fragment = _Fragment("^")
            fragment.append(_serialize(node.body))
        case Sub():
            fragment = _Fragment("_")
            fragment.append(_serialize(node.body))
        case Root():
            fragment = _serialize(node.body).wrapped(f"{node.function}(", ")")
        case Paren():
            fragment = _serialize(node.body).wrapped(node.left, node.right)
        case BigOp():
            fragment = _Fragment(f"{node.function}(")
            for k, arg in enumerate(_big_operator_arguments(node)):
                if k:
                    fragment.append_text(", ")
                fragment.append(_serialize(arg))
            fragment.append_text(")")
    fragment.spans.append((id(node), 0, len(fragment.text)))
    return fragment


def _big_operator_arguments(node: BigOp) -> list[MathList]:
    """The arguments written for ``node``: an empty variable of Σ/Π/∫ or an empty side
    of a limit is left out (the three-argument forms)."""
    body, var, lower, upper = node.arguments
    if node.kind == "limit":
        return [body, var, lower, upper] if upper.nodes else [body, var, lower]
    return [body, var, lower, upper] if var.nodes else [body, lower, upper]


def _merges(node: Node, following: str) -> bool:
    """Would ``node``'s text run into the text after it (a name absorbing letters)?"""
    if not following:
        return False
    if isinstance(node, Symbol):
        return following[0].isalpha()
    return isinstance(node, Sub) and following[0].isalnum()


def _serialize(lst: MathList) -> _Fragment:
    bare = [_node_fragment(n) for n in lst.nodes]
    out = _Fragment()
    previous: Node | None = None
    for i, node in enumerate(lst.nodes):
        fragment = bare[i]
        following = bare[i + 1].text if i + 1 < len(bare) else ""
        before = out.text.rstrip()[-1:] or None
        after = _first_visible(bare, i + 1)
        if isinstance(node, Frac):
            safe = (before is None or before in _SAFE_BEFORE_FRAC) and (
                after is None or after in _SAFE_AFTER_FRAC
            )
            if not safe:
                fragment = _rewrap(node, fragment, "(", ")")
        elif isinstance(node, Sup):
            inner, group = _ungrouped(node.body)
            body = _serialize(inner)
            plain = _is_single_token(body.text) and (after is None or after in _SAFE_AFTER_SUP)
            if group is not None or not plain:
                if not plain:
                    body = body.wrapped("(", ")")
                if group is not None:
                    body.spans.append((id(group), 0, len(body.text)))
                fragment = _Fragment("^")
                fragment.append(body)
                fragment.spans.append((id(node), 0, len(fragment.text)))
        if isinstance(node, Symbol | Root | BigOp) and (
            isinstance(previous, Sub) or (out.text and out.text[-1].isalpha())
        ):
            out.append_text(" ")  # "x pi", "a_1 sqrt(x)": keep the names apart
        out.append(fragment)
        if _merges(node, following):
            out.append_text(" ")  # "pi x", "a_1 b": keep the names apart
        previous = node
    return out


def _rewrap(node: Node, fragment: _Fragment, before: str, after: str) -> _Fragment:
    result = fragment.wrapped(before, after)
    result.spans.append((id(node), 0, len(result.text)))
    return result


def to_source(lst: MathList) -> tuple[str, SourceMap]:
    """Source text for ``lst``, plus the span of every node in it."""
    fragment = _serialize(lst)
    source_map: SourceMap = {}
    for key, start, end in fragment.spans:
        source_map[key] = (start, end)  # outermost (last recorded) span wins
    return fragment.text, source_map


def node_text(nodes: Sequence[Node]) -> str:
    """Source text for a run of nodes taken out of context (copy to clipboard)."""
    return to_source(MathList(clone_node(n) for n in nodes))[0]
