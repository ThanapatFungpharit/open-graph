"""Immutable expression tree.

The parser produces *surface* nodes (``Name``, ``Call``, ``TupleExpr``, ``Relation``).
Three bracketed constructs are kept strictly apart: ``point[x, y]`` is a ``TupleExpr``,
``list[a, b]`` / ``list[a...b]`` a ``ListExpr`` / ``RangeExpr``, and ``[a, b]``, ``(a, b)``,
``[a, b)``, ``(a, b]`` an ``IntervalExpr``.
Name resolution (see :mod:`opengraph.engine.compiler`) turns them into a *resolved* tree
that only contains ``Num``, ``Const``, ``Var``, ``UnaryOp``, ``BinOp`` and ``FuncCall`` — a closed
expression over the plotting variables that can be evaluated, sampled or converted to
SymPy without any further context.

Nodes are frozen dataclasses so resolved trees are hashable and can be used directly as
cache keys. Source spans are excluded from equality.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field

from opengraph.engine.errors import Span

NO_SPAN: Span = (0, 0)


@dataclass(frozen=True, slots=True)
class Num:
    value: float
    span: Span = field(default=NO_SPAN, compare=False)


@dataclass(frozen=True, slots=True)
class Name:
    """An identifier as written by the user (unresolved)."""

    name: str
    span: Span = field(default=NO_SPAN, compare=False)


@dataclass(frozen=True, slots=True)
class Var:
    """A free plotting variable (``x`` or ``y``) in a resolved tree."""

    name: str
    span: Span = field(default=NO_SPAN, compare=False)


@dataclass(frozen=True, slots=True)
class Const:
    """A named builtin constant (``pi``, ``e``) in a resolved tree.

    Kept distinct from :class:`Num` so symbolic output can print ``pi`` instead of
    ``3.14159...``.
    """

    name: str
    value: complex  # real except for i in complex mode or a complex projection
    span: Span = field(default=NO_SPAN, compare=False)


@dataclass(frozen=True, slots=True)
class UnaryOp:
    op: str  # only "-" survives parsing; unary "+" is dropped
    operand: Node
    span: Span = field(default=NO_SPAN, compare=False)


@dataclass(frozen=True, slots=True)
class BinOp:
    op: str  # one of + - * / ^
    left: Node
    right: Node
    span: Span = field(default=NO_SPAN, compare=False)


@dataclass(frozen=True, slots=True)
class Call:
    """``name(args)`` as written; may turn out to be a multiplication after resolution."""

    name: str
    args: tuple[Node, ...]
    span: Span = field(default=NO_SPAN, compare=False)
    name_span: Span = field(default=NO_SPAN, compare=False)


@dataclass(frozen=True, slots=True)
class FuncCall:
    """A call to a whitelisted builtin function in a resolved tree."""

    name: str
    args: tuple[Node, ...]
    span: Span = field(default=NO_SPAN, compare=False)


@dataclass(frozen=True, slots=True)
class TupleExpr:
    """A point, written ``point[1, 2]``; resolved points are always two coordinates."""

    items: tuple[Node, ...]
    span: Span = field(default=NO_SPAN, compare=False)


@dataclass(frozen=True, slots=True)
class ListExpr:
    """``list[a, b, c]``. Also appears in resolved trees until lists are broadcast away."""

    items: tuple[Node, ...]
    span: Span = field(default=NO_SPAN, compare=False)


@dataclass(frozen=True, slots=True)
class RangeExpr:
    """``list[start...stop]``: the integers from ``start`` to ``stop`` (surface only)."""

    start: Node
    stop: Node
    span: Span = field(default=NO_SPAN, compare=False)


@dataclass(frozen=True, slots=True)
class SequenceExpr:
    """``seq[body for var in sets[N]]``: ``body`` for every ``var`` in an infinite domain.

    ``domain`` is ``"N"`` (1, 2, 3, …) or ``"Z"`` (0, 1, -1, 2, -2, …); ``text`` is the
    body as written, for showing the generator. Resolution retains a lazy sequence;
    only an explicit index or finite slice materializes its terms.
    """

    body: Node
    var: str
    domain: str
    text: str = field(compare=False)
    span: Span = field(default=NO_SPAN, compare=False)


@dataclass(frozen=True, slots=True)
class IntervalExpr:
    """An interval ``[lo, hi]``, ``(lo, hi)``, ``[lo, hi)`` or ``(lo, hi]``.

    ``closed_lo`` / ``closed_hi`` say whether the endpoint belongs to the set. After
    resolution the endpoints are plain numbers with ``lo <= hi`` (a degenerate interval
    may be empty). In expressions it represents continuous element-wise values;
    it can also supply an explicit plot range.
    """

    lo: Node
    hi: Node
    closed_lo: bool
    closed_hi: bool
    span: Span = field(default=NO_SPAN, compare=False)


@dataclass(frozen=True, slots=True)
class Index:
    """``target[index]``: one element (1-based), or several when ``index`` is a list or a
    range (surface only; resolved away into the picked element(s))."""

    target: Node
    index: Node
    span: Span = field(default=NO_SPAN, compare=False)


@dataclass(frozen=True, slots=True)
class Reduce:
    """A bound-variable operation in a resolved tree: Σ, Π or a definite integral.

    ``kind`` is ``"sum"``, ``"prod"`` or ``"int"``. ``var`` names the bound variable,
    which appears as ``Var(var)`` inside ``body`` only; ``lower``/``upper`` are evaluated
    in the enclosing scope.
    """

    kind: str
    var: str
    body: Node
    lower: Node
    upper: Node
    span: Span = field(default=NO_SPAN, compare=False)


@dataclass(frozen=True, slots=True)
class Guard:
    """A transformed value, defined only where its original expression is finite."""

    value: Node
    original: Node
    span: Span = field(default=NO_SPAN, compare=False)


@dataclass(frozen=True, slots=True)
class Bounded:
    """A value restricted to an explicit interval of a bound plotting variable."""

    value: Node
    coordinate: Node
    interval: IntervalExpr
    span: Span = field(default=NO_SPAN, compare=False)


Node = (
    Num
    | Const
    | Name
    | Var
    | UnaryOp
    | BinOp
    | Call
    | FuncCall
    | TupleExpr
    | ListExpr
    | RangeExpr
    | SequenceExpr
    | IntervalExpr
    | Index
    | Reduce
    | Guard
    | Bounded
)


@dataclass(frozen=True, slots=True)
class Relation:
    """A top-level ``left <op> right`` statement; op is one of = < > <= >=."""

    op: str
    left: Node
    right: Node
    span: Span = field(default=NO_SPAN, compare=False)


@dataclass(frozen=True, slots=True)
class ComparisonChain:
    """Conjunction of adjacent inequalities, such as 0 < x <= 2."""

    comparisons: tuple[Relation, ...]
    span: Span = field(default=NO_SPAN, compare=False)


Statement = Node | Relation | ComparisonChain


def children(node: Node) -> tuple[Node, ...]:
    match node:
        case Guard(value=value, original=original):
            return (value, original)
        case Bounded(value=value, coordinate=coordinate, interval=interval):
            return (value, coordinate, interval)
        case UnaryOp(operand=operand):
            return (operand,)
        case BinOp(left=left, right=right):
            return (left, right)
        case Call(args=args) | FuncCall(args=args):
            return args
        case TupleExpr(items=items) | ListExpr(items=items):
            return items
        case RangeExpr(start=start, stop=stop):
            return (start, stop)
        case SequenceExpr(body=body):
            return (body,)
        case IntervalExpr(lo=lo, hi=hi):
            return (lo, hi)
        case Index(target=target, index=index):
            return (target, index)
        case Reduce(body=body, lower=lower, upper=upper):
            return (body, lower, upper)
        case _:
            return ()


def rebuild(node: Node, kids: Sequence[Node]) -> Node:
    """A copy of ``node`` with its children replaced by ``kids`` (same order as
    :func:`children`)."""
    match node:
        case Guard(span=span):
            return Guard(kids[0], kids[1], span)
        case Bounded(span=span):
            assert isinstance(kids[2], IntervalExpr)
            return Bounded(kids[0], kids[1], kids[2], span)
        case UnaryOp(op=op, span=span):
            return UnaryOp(op, kids[0], span)
        case BinOp(op=op, span=span):
            return BinOp(op, kids[0], kids[1], span)
        case Call(name=name, span=span, name_span=name_span):
            return Call(name, tuple(kids), span, name_span)
        case FuncCall(name=name, span=span):
            return FuncCall(name, tuple(kids), span)
        case TupleExpr(span=span):
            return TupleExpr(tuple(kids), span)
        case ListExpr(span=span):
            return ListExpr(tuple(kids), span)
        case RangeExpr(span=span):
            return RangeExpr(kids[0], kids[1], span)
        case SequenceExpr(var=var, domain=domain, text=text, span=span):
            return SequenceExpr(kids[0], var, domain, text, span)
        case IntervalExpr(closed_lo=closed_lo, closed_hi=closed_hi, span=span):
            return IntervalExpr(kids[0], kids[1], closed_lo, closed_hi, span)
        case Index(span=span):
            return Index(kids[0], kids[1], span)
        case Reduce(kind=kind, var=var, span=span):
            return Reduce(kind, var, kids[0], kids[1], kids[2], span)
    return node


def replace(node: Node, fn: Callable[[Node], Node | None]) -> Node:
    """Rewrite ``node`` top-down: where ``fn`` returns a node, it replaces that sub-tree.

    Shared sub-trees are rewritten once (inlined definitions share structure).
    """
    memo: dict[int, Node] = {}

    def visit(current: Node) -> Node:
        key = id(current)
        if key in memo:
            return memo[key]
        replacement = fn(current)
        if replacement is None:
            old = children(current)
            new = tuple(visit(child) for child in old)
            replacement = current
            if any(a is not b for a, b in zip(old, new, strict=True)):
                replacement = rebuild(current, new)
        memo[key] = replacement
        return replacement

    return visit(node)


def walk(node: Node) -> Iterator[Node]:
    """Pre-order traversal."""
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        stack.extend(reversed(children(current)))


def free_vars(node: Node) -> frozenset[str]:
    """Variables a resolved tree depends on (bound variables of Σ/Π/∫ excluded)."""
    match node:
        case SequenceExpr(body=body, var=var):
            return free_vars(body) - {var}
        case Var(name=name):
            return frozenset((name,))
        case Reduce(var=var, body=body, lower=lower, upper=upper):
            return (free_vars(body) - {var}) | free_vars(lower) | free_vars(upper)
    result: frozenset[str] = frozenset()
    for child in children(node):
        result |= free_vars(child)
    return result


def contains(node: Node, kind: type) -> bool:
    return any(isinstance(n, kind) for n in walk(node))


def node_count(node: Node) -> int:
    return sum(1 for _ in walk(node))


def domain_sensitive(node: Node) -> bool:
    """Whether algebraic rewriting could remove a domain restriction."""
    entire = {"sin", "cos", "sinh", "cosh", "exp", "abs", "re", "im", "conj"}
    for item in walk(node):
        if isinstance(item, Guard | Bounded | Reduce):
            return True
        if isinstance(item, FuncCall) and item.name not in entire:
            return True
        if isinstance(item, BinOp):
            if item.op == "/" and not (
                isinstance(item.right, Num | Const) and item.right.value != 0
            ):
                return True
            if (
                item.op == "^"
                and not (isinstance(item.left, Num | Const) and item.left.value.real > 0)
                and not (
                    isinstance(item.right, Num)
                    and item.right.value >= 0
                    and float(item.right.value).is_integer()
                )
            ):
                return True
    return False
