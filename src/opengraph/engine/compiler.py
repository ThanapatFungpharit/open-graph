"""Whole-document compilation with explicit definitions and stable identifiers.

Rows may reference definitions in any order. Equations report solutions without
introducing bindings; roots(expression, variable) explicitly returns a finite list.
Lists pair by equal length, sequences require explicit finite selection, and plot
parameters/ranges are local. Angle units and number mode affect evaluation; display
grids do not. Every failure is reported per row without breaking unrelated rows.
"""

from __future__ import annotations

import contextlib
import dataclasses
import math
import re
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TypeVar, cast

import numpy as np

from opengraph.engine.analysis import solve_equation
from opengraph.engine.ast import (
    NO_SPAN,
    BinOp,
    Bounded,
    Call,
    ComparisonChain,
    Const,
    FuncCall,
    Guard,
    Index,
    IntervalExpr,
    ListExpr,
    Name,
    Node,
    Num,
    RangeExpr,
    Reduce,
    Relation,
    SequenceExpr,
    Statement,
    TupleExpr,
    UnaryOp,
    Var,
    children,
    contains,
    domain_sensitive,
    free_vars,
    replace,
    walk,
)
from opengraph.engine.builtins import (
    ALGEBRA_OPERATIONS,
    CAS_OPERATIONS,
    COMPLEX_PROJECTIONS,
    CONSTANTS,
    FUNCTIONS,
    IMAGINARY_UNIT,
    INVERSE_TRIG_FUNCTIONS,
    KEYWORDS,
    LIST_FUNCTIONS,
    PLOT_VARIABLES,
    REDUCERS,
    RESERVED,
    TRIG_FUNCTIONS,
)
from opengraph.engine.continuous import intervals, parameterize
from opengraph.engine.errors import MathError, ParseError, ResolveError, Span
from opengraph.engine.evaluator import NumericFunction
from opengraph.engine.formatting import format_complex, format_value
from opengraph.engine.lexer import Kind, tokenize
from opengraph.engine.parser import parse
from opengraph.engine.plots import (
    ExplicitPlot,
    ImplicitPlot,
    ParametricPlot,
    Plot,
    PlotGroup,
    PointPlot,
    PointSetPlot,
    PolarPlot,
    RegionPlot,
)

MAX_NODES = 5000
MAX_LIST = 10_000  # elements in one list
MAX_CURVES = 100  # curves drawn by one row through list broadcasting
MAX_SHOWN_VALUES = 8  # list elements shown in a row's feedback
SEQUENCE_PREVIEW_TERMS = 100  # explicit diagnostic sample of a general solution

THETA = "θ"
THETA_NAMES = frozenset({"θ", "theta"})
RADIUS = "r"
_TO_RADIANS = BinOp("/", Const("pi", math.pi), Num(180.0))
_TO_DEGREES = BinOp("/", Num(180.0), Const("pi", math.pi))

K = TypeVar("K", bound=Hashable)
V = TypeVar("V", bound=Hashable)


@dataclass(frozen=True, slots=True)
class CompileOptions:
    """Document settings that change how expressions are interpreted."""

    degrees: bool = False
    complex_mode: bool = False
    polar: bool = False  # retained for callers; display grids never change name semantics


@dataclass(frozen=True, slots=True)
class SymbolicTarget:
    """What symbolic tools operate on for a row: ``node`` as a function of ``variable``.

    ``value`` targets are closed expressions (a row showing a number): only the exact
    value can be computed for them.
    """

    node: Node
    variable: str  # "x" or "y"
    options: CompileOptions = field(default_factory=CompileOptions)
    value: bool = False
    # ``node`` with user constants left as names (``a x`` rather than ``2 x``), for
    # simplifying without substituting their values. None when it couldn't be built.
    named: Node | None = field(default=None, compare=False)
    # Set when ``named`` is an equation (as ``named = 0``) still to be solved for this
    # variable: ``2y = a x`` is only solved for y numerically, with a's value inlined.
    named_unsolved: str | None = field(default=None, compare=False)
    # The current value of each user constant in ``named``, as a closed tree: lets a
    # result in terms of the names be checked against the values (``Solve`` keeps
    # ``a x^2 = 4`` as ``±2/sqrt(a)`` only where that gives the plotted roots).
    constants: tuple[tuple[str, Node], ...] = field(default=(), compare=False)
    # ``node`` is the residual of an equation, so simplifying it yields ``result = 0``.
    equation: bool = field(default=False, compare=False)


@dataclass(frozen=True, slots=True)
class SolutionFamily:
    """The solutions ``base + n·period`` for every integer n (``pi/6 + 2*pi*n``)."""

    base: str  # source text, 0 <= base < period
    period: str  # source text, positive
    base_value: float
    period_value: float

    def text(self, index: str = "n") -> str:
        if self.base == "0":
            return f"{self.period}*{index}"
        return f"{self.base} + {self.period}*{index}"

    def values(self, lo: float, hi: float) -> list[float]:
        """The family's members in ``[lo, hi]``, ascending."""
        first = math.ceil((lo - self.base_value) / self.period_value)
        last = math.floor((hi - self.base_value) / self.period_value)
        return [self.base_value + n * self.period_value for n in range(first, last + 1)]


@dataclass(frozen=True, slots=True)
class GeneralSolution:
    """Exact countable solutions: periodic/general generators plus isolated roots."""

    families: tuple[SolutionFamily, ...]
    roots: tuple[tuple[str, float], ...] = ()
    generators: tuple[SolutionGenerator, ...] = ()

    def index(self, variable: str) -> str:
        """The name of the integer parameter: n, unless that is the variable itself."""
        return "k" if variable == "n" else "n"

    def text(self, variable: str) -> str:
        """``x = pi/6 + 2*pi*n, 5*pi/6 + 2*pi*n  (n ∈ ℤ)``."""
        index = self.index(variable)
        if self.generators:
            parts = ["{" + ", ".join(text for text, _ in self.roots) + "}"] if self.roots else []
            parts += [g.source(index) for g in self.as_generators()]
            return f"{variable} = " + " ∪ ".join(parts)
        parts = [text for text, _ in self.roots] + [f.text(index) for f in self.families]
        return f"{variable} = {', '.join(parts)}  ({index} ∈ ℤ)"

    def as_generators(self) -> tuple[SolutionGenerator, ...]:
        return tuple(SolutionGenerator(f.text(), "Z") for f in self.families) + self.generators

    def values(self, lo: float, hi: float) -> list[float]:
        """Periodic members and a bounded preview of other generators in [lo, hi]."""
        found = [v for _, v in self.roots if lo <= v <= hi]
        for family in self.families:
            found += family.values(lo, hi)
        for generator in self.generators:
            preview = Index(
                generator.sequence_node(), RangeExpr(Num(1), Num(SEQUENCE_PREVIEW_TERMS))
            )
            node = _Resolver({}, {}, generator.options).resolve_root(preview)
            values = _list_values(node, False)
            if values is not None:
                found += [float(v) for v in values if lo <= v <= hi]
        return sorted(set(found))


def _expression(text: str) -> Node:
    node = parse(text)
    assert not isinstance(node, Relation | ComparisonChain)
    return node


@dataclass(frozen=True, slots=True)
class SolutionGenerator:
    """An exact image of ℕ or ℤ; only its preview is truncated for plotting."""

    expression: str  # source text in the bound parameter n
    domain: str
    options: CompileOptions = field(default_factory=CompileOptions)

    def text(self, index: str = "n") -> str:
        return re.sub(r"\bn\b", index, self.expression)

    def source(self, index: str = "n") -> str:
        return f"seq[{self.text(index)} for {index} in sets[{self.domain}]]"

    def sequence_node(self) -> SequenceExpr:
        return SequenceExpr(_expression(self.expression), "n", self.domain, self.expression)

    def term(self, position: int) -> Node:
        value = position + 1 if self.domain == "N" else _integer_member(position)
        return replace(
            _expression(self.expression),
            lambda node: Num(float(value)) if node == Name("n") else None,
        )


ExactRoots = tuple[tuple[str, float], ...] | GeneralSolution
"""Exact real solutions of an equation, as ``(source text, value)`` pairs (from SymPy),
or a :class:`GeneralSolution` when there are infinitely many."""


@dataclass(frozen=True, slots=True)
class CasRequest:
    """A computer-algebra call to evaluate with SymPy: ``diff(a x^3)`` is
    ``CasRequest("diff", a·x^3, "x")``.

    Most algebra and derivatives keep user constants and function parameters as
    names, so moving a slider can reuse the formula. Integrals, roots and partial
    fractions include current parameter values to handle exceptional values.
    """

    operation: str  # a key of CAS_OPERATIONS
    node: Node
    variable: str
    options: CompileOptions = field(default_factory=CompileOptions)
    order: int = 1  # how many times to differentiate; the Taylor polynomial's degree
    point: Node | None = None  # where limit/taylor are taken, constants as names
    direction: str = "+-"  # limit: "+-" two-sided, "+" from the right, "-" from the left


@dataclass(frozen=True, slots=True)
class CasFailure:
    """Why SymPy couldn't compute a :class:`CasRequest`."""

    message: str


CasResults = Mapping[CasRequest, "str | CasFailure"]
"""Results of computer-algebra calls: the result's source text, or why there is none."""

MAX_DIFF_ORDER = 20


class PendingError(ResolveError):
    """A computer-algebra result the row needs hasn't been computed yet."""

    def __init__(self, request: CasRequest, span: Span | None = None) -> None:
        super().__init__(f"Computing the {CAS_OPERATIONS[request.operation]}…", span)
        self.request = request


@dataclass(frozen=True, slots=True)
class TableColumn:
    name: str  # a list name like "x_1", or "x"/"y" to only plot
    cells: tuple[str, ...]  # expression text per cell; "" is an empty cell


@dataclass(frozen=True, slots=True)
class TableInput:
    """A table row: the first column is plotted against each other column."""

    columns: tuple[TableColumn, ...]
    markers: bool = True
    lines: bool = False


def split_column_header(header: str) -> tuple[str, str | None]:
    """A column header ``name`` or ``name = formula`` as ``(name, formula)``.

    A formula column (``y_2 = x_1^2 + 1``) is computed from the other columns, row by row.
    """
    name, sep, formula = header.partition("=")
    return name.strip(), (formula.strip() if sep else None)


# A point coordinate that can follow a drag: a bare name (rewritten in the row that
# defines it, if that is a ``name = <number>`` row), the span of a number literal in
# the point's own row (rewritten in place), or None (computed: it stays put).
DragCoordinate = str | Span | None


@dataclass(frozen=True, slots=True)
class RowResult:
    plot: Plot | None = None
    message: str = ""  # error text, computed value, or a short note
    is_error: bool = False
    error_span: Span | None = None
    defines: str | None = None
    symbolic: SymbolicTarget | None = None
    # For ``name = <number>`` rows (``k = 1.5``, ``k = -2``): the number, so a slider can
    # drive it. None for computed bodies such as ``b = 2a`` or ``c = 1/3``.
    literal_value: float | None = None
    # Table rows: a message per offending cell, keyed by (row, column).
    cell_errors: Mapping[tuple[int, int], str] = field(default_factory=dict)
    # Table rows: the computed text of each cell of a formula column, keyed by (row, column).
    table_values: Mapping[tuple[int, int], str] = field(default_factory=dict)
    # Equations solved for an unknown name (``2a + 3 = 7``): the name and its solutions.
    solved: tuple[str, tuple[float, ...]] | None = None
    # Rows failing only because of undefined names: those names, in order of appearance,
    # and the equation (as ``node = 0``) to rearrange for one of them. The target's
    # ``variable`` is the first name; replace it with the name being solved for.
    solve_for: tuple[str, ...] = ()
    rearrange: SymbolicTarget | None = None
    # Rows failing because of undefined names: all of them, in order of appearance
    # (for "add slider" suggestions). Wider than ``solve_for``: definitions, points and
    # plain values count too.
    undefined: tuple[str, ...] = ()
    # A single point ``(q, w)`` or ``(3.8, 5.5)``: per coordinate, what dragging the point
    # rewrites (see DragCoordinate). None for anything else.
    drag: tuple[DragCoordinate, DragCoordinate] | None = None
    # Computer-algebra results the row waits for (its message says so; not an error).
    pending: tuple[CasRequest, ...] = ()
    # An error that only needs more typing (``x^``, ``sqrt(``): still an error, but the
    # editor shows it as a hint and keeps the row's last drawing.
    incomplete: bool = False

    @staticmethod
    def error(exc: MathError) -> RowResult:
        if isinstance(exc, PendingError):
            return RowResult(message=exc.message, pending=(exc.request,))
        return RowResult(
            message=exc.message, is_error=True, error_span=exc.span, incomplete=exc.incomplete
        )


EMPTY = RowResult()


@dataclass(frozen=True, slots=True)
class _Definition:
    name: str
    params: tuple[str, ...] | None  # None for constants
    body: Node
    span: Span


def _mentions(node: Node, names: frozenset[str]) -> bool:
    return any(isinstance(n, Name | Call) and n.name in names for n in walk(node))


def _definition_of(stmt: Statement) -> _Definition | None:
    """Recognise ``name = body`` and ``name(p, q) = body`` statements.

    Assignments to r define a variable unless the body uses polar coordinates.
    """
    if not isinstance(stmt, Relation) or stmt.op != "=":
        return None
    left = stmt.left
    if isinstance(left, Name) and left.name not in PLOT_VARIABLES:
        if left.name == RADIUS and _mentions(stmt.right, THETA_NAMES | {RADIUS}):
            return None
        return _Definition(left.name, None, stmt.right, left.span)
    if (
        isinstance(left, Call)
        and left.name not in PLOT_VARIABLES
        # "sin(x) = 0.5" and "x! = 6" are equations, not redefinitions of builtins.
        and left.name not in FUNCTIONS
        and left.name not in REDUCERS
        and left.name not in LIST_FUNCTIONS
        and all(isinstance(a, Name) for a in left.args)
    ):
        params = tuple(a.name for a in left.args if isinstance(a, Name))
        return _Definition(left.name, params, stmt.right, left.name_span)
    return None


def _check_definition(defn: _Definition, options: CompileOptions) -> None:
    if defn.name in RESERVED:
        functions = FUNCTIONS.keys() | REDUCERS.keys() | LIST_FUNCTIONS | CAS_OPERATIONS.keys()
        kind = "function" if defn.name in functions else "constant"
        if defn.name in PLOT_VARIABLES:
            kind = "variable"
        elif defn.name in KEYWORDS:
            kind = "keyword"
        raise ResolveError(f"'{defn.name}' is a built-in {kind} and can't be redefined", defn.span)
    if options.complex_mode and defn.name == IMAGINARY_UNIT:
        raise ResolveError("'i' is the imaginary unit in complex mode", defn.span)
    if defn.name in THETA_NAMES:
        raise ResolveError(f"'{defn.name}' is a reserved polar coordinate", defn.span)
    if defn.params is not None:
        if len(set(defn.params)) != len(defn.params):
            raise ResolveError("Parameter names must be different", defn.span)
        for p in defn.params:
            if p in RESERVED - set(PLOT_VARIABLES):
                raise ResolveError(f"'{p}' is a built-in name; choose another parameter", defn.span)


def _tree_size(node: Node, memo: dict[int, int]) -> int:
    """Size of the tree, counting shared sub-trees once per use (linear time).

    Plain numbers in a list are free, so data lists aren't limited by ``MAX_NODES``.
    """
    key = id(node)
    if key not in memo:
        kids = children(node)
        if isinstance(node, ListExpr):
            kids = tuple(k for k in kids if not isinstance(k, Num))
        memo[key] = 1 + sum(_tree_size(c, memo) for c in kids)
    return memo[key]


def _balanced(op: str, items: Sequence[Node], span: Span) -> Node:
    """``items[0] op items[1] op ...`` as a balanced tree (no deep recursion)."""
    if len(items) == 1:
        return items[0]
    mid = len(items) // 2
    return BinOp(op, _balanced(op, items[:mid], span), _balanced(op, items[mid:], span), span)


class _Resolver:
    def __init__(
        self,
        definitions: Mapping[str, _Definition],
        broken: Mapping[str, str],
        options: CompileOptions,
        exact_solutions: Mapping[SymbolicTarget, ExactRoots] | None = None,
        *,
        keep_constants: bool = False,
        cas_results: CasResults | None = None,
    ) -> None:
        self.definitions = definitions
        # Leave user constants as named variables instead of inlining their values (for
        # symbolic rearranging: "Solve for w" should give w in terms of k, not of 1.5).
        self.keep_constants = keep_constants
        self.broken = broken  # name -> reason, for definitions whose row has an error
        self.options = options
        # Exact roots known for equations solved for an unknown (see compile_document).
        self.exact_solutions: Mapping[SymbolicTarget, ExactRoots] = exact_solutions or {}
        # Computer-algebra results known so far (see CasRequest); the source text of each
        # one used by this resolver's own row is recorded by call span, for its feedback.
        self.cas_results: CasResults = cas_results or {}
        self.cas_used: dict[Span, str] = {}
        self.constant_cache: dict[str, Node] = {}
        self.stack: list[str] = []
        self.foreign_depth = 0  # how many other rows' definitions are being expanded
        self.depth = 0  # nesting of Σ/Π/∫ bodies, used to name bound variables uniquely
        # While resolving an equation: unknown names (not inside other rows' definitions)
        # become variables to solve for, recorded here with their first span.
        self.unknowns: dict[str, Span] | None = None

    # -- entry points --------------------------------------------------------------
    def resolve_root(
        self, node: Node, locals_: Mapping[str, Node] | None = None, *, continuous: bool = False
    ) -> Node:
        """Resolve ``node`` to a number-valued expression (points are errors)."""
        resolved = self.resolve_value(node, locals_)
        if isinstance(resolved, TupleExpr):
            raise ResolveError("A point can't be used here; this needs a number", node.span)
        if continuous:
            _reject_sequence(resolved, node.span)
        else:
            _reject_interval(resolved, node.span)
        return resolved

    def resolve_value(self, node: Node, locals_: Mapping[str, Node] | None = None) -> Node:
        """Resolve ``node`` to a number-valued expression or a point: a ``TupleExpr`` of
        two number-valued coordinates, with any vector arithmetic carried out."""
        resolved = self.resolve(node, locals_ or {})
        if _tree_size(resolved, {}) > MAX_NODES:
            raise ResolveError("Expression is too large after expanding definitions", node.span)
        return resolved

    def constant(self, name: str) -> Node:
        """Resolved body of the user constant ``name`` (must not depend on x or y)."""
        if name in self.constant_cache:
            return self.constant_cache[name]
        defn = self.definitions[name]
        body = self.resolve(defn.body, {})
        used = free_vars(body) if not isinstance(body, SequenceExpr) else frozenset()
        if used:
            var = sorted(used)[0]
            raise ResolveError(
                f"'{name}' depends on {var}; define a function instead, like {name}({var}) = …",
                defn.body.span,
            )
        self.constant_cache[name] = body
        return body

    # -- helpers -------------------------------------------------------------------
    def expanding(self, name: str, span: Span) -> _Expansion:
        """Expanding another row's definition: errors are re-labelled at ``span``."""
        return _Expansion(self, name, span, foreign=True)

    def own(self, name: str) -> _Expansion:
        """Resolving the current row's own definition (only tracked for cycles)."""
        return _Expansion(self, name, (0, 0), foreign=False)

    def is_variable(self, name: str) -> bool:
        if name in PLOT_VARIABLES:
            return True
        if name in THETA_NAMES:
            return True
        return name == RADIUS

    def is_builtin_constant(self, name: str) -> bool:
        if name in CONSTANTS:
            return True
        return self.options.complex_mode and name == IMAGINARY_UNIT

    def is_value_name(self, name: str, locals_: Mapping[str, Node]) -> bool:
        if name in locals_ or self.is_variable(name) or self.is_builtin_constant(name):
            return True
        defn = self.definitions.get(name)
        return defn is not None and defn.params is None

    def is_function_name(self, name: str, locals_: Mapping[str, Node]) -> bool:
        if name in locals_:
            return False
        if name in FUNCTIONS or name in REDUCERS or name in LIST_FUNCTIONS:
            return True
        if name in CAS_OPERATIONS:
            return True
        defn = self.definitions.get(name)
        return defn is not None and defn.params is not None

    # -- resolution ----------------------------------------------------------------
    def resolve(self, node: Node, locals_: Mapping[str, Node]) -> Node:
        match node:
            case Num() | Const():
                return node
            case Var(name=name):
                return locals_.get(name, node)
            case Name(name=name, span=span):
                return self.resolve_name(name, span, locals_)
            case UnaryOp(op=op, operand=operand, span=span):
                value = self.resolve(operand, locals_)
                _reject_sequence(value, operand.span)
                if isinstance(value, TupleExpr):
                    return TupleExpr(tuple(UnaryOp(op, c, span) for c in value.items), span)
                return UnaryOp(op, value, span)
            case BinOp(op=op, left=left, right=right, span=span):
                a, b = self.resolve(left, locals_), self.resolve(right, locals_)
                _reject_sequence(a, left.span)
                _reject_sequence(b, right.span)
                return _combine(op, a, b, span)
            case Call():
                return self.resolve_call(node, locals_)
            case TupleExpr(items=items, span=span):
                return self.resolve_point(items, span, locals_)
            case ListExpr(items=items, span=span):
                return self.resolve_list(items, span, locals_)
            case RangeExpr():
                return self.resolve_range(node, locals_)
            case SequenceExpr():
                return self.resolve_sequence(node, locals_)
            case IntervalExpr():
                return self.resolve_interval(node, locals_)
            case Index():
                return self.resolve_index(node, locals_)
            case FuncCall() | Reduce() | Guard() | Bounded():
                return node
        raise ResolveError("Unsupported expression", None)  # pragma: no cover

    def resolve_point(
        self, items: tuple[Node, ...], span: Span, locals_: Mapping[str, Node]
    ) -> TupleExpr:
        """``(a, b)`` inside an expression: a 2-D vector of number-valued coordinates."""
        if len(items) != 2:
            raise ResolveError("A point needs exactly two coordinates, like point[1, 2]", span)
        coords: list[Node] = []
        for item in items:
            value = self.resolve(item, locals_)
            if isinstance(value, TupleExpr):
                raise ResolveError("Point coordinates can't be points", item.span)
            _reject_sequence(value, item.span)
            coords.append(value)
        return TupleExpr(tuple(coords), span)

    def resolve_list(
        self, items: tuple[Node, ...], span: Span, locals_: Mapping[str, Node]
    ) -> Node:
        if len(items) > MAX_LIST:
            raise ResolveError(f"Lists can have at most {MAX_LIST} elements", span)
        resolved = []
        for item in items:
            value = self.resolve(item, locals_)
            _reject_interval(value, item.span)
            if isinstance(value, TupleExpr):
                raise ResolveError("Points can't be list items inside an expression", item.span)
            if contains(value, ListExpr):
                raise ResolveError("Lists can't contain other lists", item.span)
            resolved.append(value)
        return ListExpr(tuple(resolved), span)

    def resolve_index(self, node: Index, locals_: Mapping[str, Node]) -> Node:
        """``L[2]`` -> the element; ``L[1...3]`` or ``L[list[1, 3]]`` -> a shorter list."""
        if isinstance(node.target, SequenceExpr):
            return self.resolve_sequence_index(node.target, node.index, locals_)
        if isinstance(node.target, Name) and node.target.name not in locals_:
            defn = self.definitions.get(node.target.name)
            if (
                defn is not None
                and defn.params is None
                and defn.name not in self.broken
                and isinstance(defn.body, SequenceExpr | Name)
            ):
                with self.expanding(defn.name, node.target.span):
                    return self.resolve_index(Index(defn.body, node.index, node.span), locals_)
        target = self.resolve(node.target, locals_)
        _reject_interval(target, node.target.span)
        if isinstance(target, TupleExpr) or not contains(target, ListExpr):
            raise ResolveError("Only a list can be indexed, like L[2]", node.target.span)
        items = _expand(target)  # also lists computed from lists, like (2L)[1]
        picks: list[Node] = []
        index = self.resolve(node.index, locals_)
        _reject_interval(index, node.index.span)
        positions = index.items if isinstance(index, ListExpr) else (index,)
        for position in positions:
            if (
                free_vars(position)
                or contains(position, ListExpr)
                or isinstance(position, TupleExpr)
            ):
                raise ResolveError("A list index must be a number", node.index.span)
            value = NumericFunction(position).scalar()
            if not math.isfinite(value) or abs(value - round(value)) > 1e-9:
                raise ResolveError("A list index must be a whole number", node.index.span)
            number = round(value)
            if not 1 <= number <= len(items):
                raise ResolveError(
                    f"Index {number} is out of range for a list of {len(items)}", node.index.span
                )
            picks.append(items[number - 1])
        if isinstance(index, ListExpr):
            return ListExpr(tuple(picks), node.span)
        return picks[0]

    def resolve_sequence_index(
        self, node: SequenceExpr, index: Node, locals_: Mapping[str, Node]
    ) -> Node:
        """Generate requested terms directly, including terms beyond the plot preview."""
        sequence = self.resolve_sequence(node, locals_)
        assert isinstance(sequence, SequenceExpr)
        resolved = self.resolve_root(index, locals_)
        positions = _expand(resolved)
        terms: list[Node] = []
        for position in positions:
            if free_vars(position):
                raise ResolveError("A sequence index must be a positive whole number", index.span)
            value = NumericFunction(position).scalar()
            if not math.isfinite(value) or value < 1 or value != math.floor(value):
                raise ResolveError("A sequence index must be a positive whole number", index.span)
            k = int(value) if node.domain == "N" else _integer_member(int(value) - 1)

            def substitute(n: Node, value: float = float(k)) -> Node | None:
                return Num(value) if n == Var(sequence.var) else None

            term = replace(sequence.body, substitute)
            terms.append(term)
        return ListExpr(tuple(terms), index.span) if contains(resolved, ListExpr) else terms[0]

    def resolve_range(self, node: RangeExpr, locals_: Mapping[str, Node]) -> Node:
        bounds = []
        for bound in (node.start, node.stop):
            value = self.resolve(bound, locals_)
            if (
                free_vars(value)
                or contains(value, ListExpr)
                or isinstance(value, TupleExpr | IntervalExpr | SequenceExpr)
                or any(intervals(value))
            ):
                raise ResolveError("Range bounds must be numbers", bound.span)
            bounds.append(NumericFunction(value).scalar())
        start, stop = bounds
        if not (math.isfinite(start) and math.isfinite(stop)):
            raise ResolveError("Range bounds must be finite", node.span)
        count = math.floor(abs(stop - start) + 1e-9) + 1
        if count > MAX_LIST:
            raise ResolveError(f"Lists can have at most {MAX_LIST} elements", node.span)
        step = 1.0 if stop >= start else -1.0
        return ListExpr(tuple(Num(start + k * step) for k in range(count)), node.span)

    def resolve_sequence(self, node: SequenceExpr, locals_: Mapping[str, Node]) -> Node:
        """Resolve a lazy sequence without materializing a plotting/arithmetic prefix."""
        var = node.var
        if var in RESERVED - set(PLOT_VARIABLES):
            raise ResolveError(f"'{var}' is a built-in name; use another variable", node.span)
        bound = f"{var}@{self.depth}"
        self.depth += 1
        try:
            body = self.resolve(node.body, {**locals_, var: Var(bound, node.span)})
        finally:
            self.depth -= 1
        _reject_interval(body, node.body.span)
        if isinstance(body, TupleExpr):
            raise ResolveError("Points can't be sequence terms", node.body.span)
        if contains(body, ListExpr):
            raise ResolveError("Sequences can't contain lists", node.body.span)
        if free_vars(body) - {bound}:
            raise ResolveError(
                "Sequence terms must be numbers; they can't use x or y", node.body.span
            )

        return SequenceExpr(body, bound, node.domain, node.text, node.span)

    def resolve_interval(self, node: IntervalExpr, locals_: Mapping[str, Node]) -> IntervalExpr:
        """``[a, b)`` and friends: both endpoints must be finite numbers with ``a <= b``;
        an interval with equal endpoints is the single point ``[a, a]`` or else empty."""
        bounds = []
        for end in (node.lo, node.hi):
            value = self.resolve(end, locals_)
            if (
                free_vars(value)
                or contains(value, ListExpr)
                or isinstance(value, TupleExpr | IntervalExpr | SequenceExpr)
                or any(intervals(value))
            ):
                raise ResolveError("Interval endpoints must be numbers", end.span)
            number = NumericFunction(value).scalar()
            if not math.isfinite(number):
                raise ResolveError("Interval endpoints must be finite numbers", end.span)
            bounds.append(number)
        lo, hi = bounds
        if lo > hi:
            raise ResolveError(
                f"Interval endpoints are out of order: {format_value(lo)} is greater "
                f"than {format_value(hi)}",
                node.span,
            )
        return IntervalExpr(Num(lo), Num(hi), node.closed_lo, node.closed_hi, node.span)

    def resolve_name(self, name: str, span: Span, locals_: Mapping[str, Node]) -> Node:
        if name in locals_:
            return locals_[name]
        if name in PLOT_VARIABLES:
            return Var(name, span)
        defn = self.definitions.get(name)
        if defn is not None:
            if defn.params is not None:
                example = f"{name}({', '.join(defn.params)})"
                raise ResolveError(f"'{name}' is a function; call it like {example}", span)
            if name in self.broken:
                raise ResolveError(f"'{name}' {self.broken[name]}", span)
            if self.keep_constants:
                return Var(name, span)
            with self.expanding(name, span):
                return self.constant(name)
        if self.is_variable(name):
            return Var(THETA if name in THETA_NAMES else name, span)
        if name in CONSTANTS:
            canonical = {"π": "pi", "τ": "tau"}.get(name, name)
            return Const(canonical, CONSTANTS[name], span)
        if self.options.complex_mode and name == IMAGINARY_UNIT:
            return Const(IMAGINARY_UNIT, 1j, span)
        if self.is_function_name(name, {}) and name not in self.definitions:
            raise ResolveError(f"'{name}' is a function; use parentheses, like {name}(x)", span)
        if (
            self.options.complex_mode
            and len(name) == 2
            and name.startswith(IMAGINARY_UNIT)
            and name[1].isalpha()
        ):
            # Compact imaginary products: ix, iy, it, ... . Explicit definitions
            # and local parameters above retain priority over this shorthand.
            middle = span[0] + 1
            return BinOp(
                "*",
                self.resolve_name(IMAGINARY_UNIT, (span[0], middle), locals_),
                self.resolve_name(name[1], (middle, span[1]), locals_),
                span,
            )
        if name in ("in", "for"):  # a keyword out of place, like "k in" while typing
            raise ResolveError(
                f"'{name}' is a keyword: write k in sets[N], or seq[2k for k in sets[N]]", span
            )
        if self.unknowns is not None and not self.foreign_depth:
            self.unknowns.setdefault(name, span)
            return Var(name, span)
        raise ResolveError(f"Unknown name '{name}'", span)

    def resolve_call(self, node: Call, locals_: Mapping[str, Node]) -> Node:
        name, span = node.name, node.span
        if name == "restrict":
            if len(node.args) != 3:
                raise ResolveError("Use restrict(value, coordinate, interval)", span)
            value, coordinate = (self.resolve_root(a, locals_) for a in node.args[:2])
            interval = self.resolve_value(node.args[2], locals_)
            if not isinstance(interval, IntervalExpr):
                raise ResolveError("The restriction must be an interval", node.args[2].span)
            return Bounded(value, coordinate, interval, span)
        if name == "plot":
            if len(node.args) != 3 or not isinstance(node.args[1], Name):
                raise ResolveError("Use plot(t^2 for t in [0, 2])", span)
            parameter = node.args[1].name
            if parameter in RESERVED - set(PLOT_VARIABLES):
                raise ResolveError("Choose an ordinary parameter name", node.args[1].span)
            interval = self.resolve_value(node.args[2], locals_)
            if not isinstance(interval, IntervalExpr):
                raise ResolveError("The plot range must be an interval", node.args[2].span)
            coordinate = locals_.get(parameter, Var("x"))
            value = self.resolve_root(node.args[0], {**locals_, parameter: coordinate})
            if free_vars(value) - free_vars(coordinate):
                raise ResolveError("The plot expression must use only its parameter", span)
            return Bounded(value, coordinate, interval, span)
        if name == "guard":
            if len(node.args) != 2:
                raise ResolveError("Use guard(value, original_expression)", span)
            value, original = (self.resolve_root(a, locals_) for a in node.args)
            return Guard(value, original, span)
        if self.is_function_name(name, locals_):
            return self.call_function(name, node.args, span, node.name_span, locals_)
        if self.is_value_name(name, locals_) or name in self.definitions:
            # "a(x+1)" with a constant a, or "x(x-1)": multiplication.
            return self._multiply_by_group(
                self.resolve_name(name, node.name_span, locals_), node, locals_
            )
        if self.unknowns is not None and not self.foreign_depth and len(node.args) == 1:
            # "a(x + 1) = 4" with an unknown a: multiplication, as for a known constant.
            unknown = self.resolve_name(name, node.name_span, locals_)
            return self._multiply_by_group(unknown, node, locals_)
        raise ResolveError(f"Unknown function '{name}'", node.name_span)

    def _multiply_by_group(self, factor: Node, node: Call, locals_: Mapping[str, Node]) -> Node:
        if len(node.args) == 2:  # "a(1, 2)": a point scaled by a
            group = self.resolve_point(node.args, node.span, locals_)
            return _combine("*", factor, group, node.span)
        if len(node.args) != 1:
            raise ResolveError(f"'{node.name}' is not a function", node.name_span)
        return _combine("*", factor, self.resolve(node.args[0], locals_), node.span)

    def call_function(
        self,
        name: str,
        args: tuple[Node, ...],
        span: Span,
        name_span: Span,
        locals_: Mapping[str, Node],
    ) -> Node:
        if name in REDUCERS and name not in self.definitions:
            return self.reduce(REDUCERS[name], name, args, span, locals_)
        if name in CAS_OPERATIONS:
            return self.computer_algebra(name, args, span, locals_)
        previous_options = self.options
        if name in COMPLEX_PROJECTIONS:
            self.options = dataclasses.replace(self.options, complex_mode=True)
        try:
            resolved_args = tuple(self.resolve(a, locals_) for a in args)
        finally:
            self.options = previous_options
        for arg, value in zip(args, resolved_args, strict=True):
            _reject_sequence(value, arg.span)
        if name not in self.definitions:  # user functions may take points; built-ins can't
            for arg, value in zip(args, resolved_args, strict=True):
                if isinstance(value, TupleExpr):
                    raise ResolveError(f"'{name}' needs a number, not a point", arg.span)
        if name in LIST_FUNCTIONS or (name in ("min", "max") and len(args) == 1):
            return self.list_function(name, resolved_args, span)
        builtin = FUNCTIONS.get(name)
        if name in self.definitions or builtin is None:
            defn = self.definitions[name]
            assert defn.params is not None
            if len(args) != len(defn.params):
                raise ResolveError(
                    f"'{name}' takes {len(defn.params)} argument"
                    f"{'s' if len(defn.params) != 1 else ''}, got {len(args)}",
                    span,
                )
            if name in self.broken:
                raise ResolveError(f"'{name}' {self.broken[name]}", name_span)
            with self.expanding(name, name_span):
                return self.resolve(defn.body, dict(zip(defn.params, resolved_args, strict=True)))
        n = len(args)
        if n < builtin.min_args or (builtin.max_args is not None and n > builtin.max_args):
            raise ResolveError(f"{builtin.signature}: wrong number of arguments ({n})", span)
        if self.options.degrees and builtin.name in TRIG_FUNCTIONS:
            resolved_args = (BinOp("*", resolved_args[0], _TO_RADIANS, span),)
        call = FuncCall(builtin.name, resolved_args, span)
        if self.options.degrees and builtin.name in INVERSE_TRIG_FUNCTIONS:
            return BinOp("*", call, _TO_DEGREES, span)
        return call

    def computer_algebra(
        self, name: str, args: tuple[Node, ...], span: Span, locals_: Mapping[str, Node]
    ) -> Node:
        """``diff(f)``, ``integrate(f, t)``, ``factor(f)``…: the result SymPy computed
        for the argument, resolved like text the user typed, or PendingError until
        there is one.

        The argument is resolved with user constants and the enclosing function's
        parameters as names, so ``g(t) = diff(t^3)`` differentiates in t before g is
        applied to anything, and the result is a formula in those names.
        """
        least, most = {
            "diff": (1, 3),
            "integrate": (1, 2),
            "limit": (3, 4),
            "taylor": (4, 4),
            "roots": (2, 2),
            "apart": (1, 2),
            "collect": (2, 2),
        }.get(name, (1, 1))
        if not least <= len(args) <= most:
            form = {
                "diff": "diff(x^3), diff(x^3 for x) or diff(x^3 for x order 2)",
                "integrate": "integrate(x^2) or integrate(x^2 for x)",
                "limit": "limit(sin(x)/x for x to 0), or … to 0 from right",
                "taylor": "taylor(e^x for x at 0 degree 3)",
                "roots": "roots(a^2 - 4 for a)",
                "apart": "apart(1/(x^2 - 1)) or apart(1/(x^2 - 1) for x)",
                "collect": "collect(a x + b x for x)",
            }.get(name, f"{name}(f)")
            raise ResolveError(f"Use {form}", span)
        variable: str | None = None
        if len(args) >= 2:
            if not isinstance(args[1], Name):
                raise ResolveError(f"{name}: the second argument names the variable", args[1].span)
            variable = args[1].name
            if variable in RESERVED - set(PLOT_VARIABLES):
                raise ResolveError(f"'{variable}' is a built-in name, not a variable", args[1].span)
        order = 1
        if name == "diff" and len(args) == 3:
            order = self.whole_number(args[2], locals_, 1, "diff: the order must be a whole number")
        elif name == "taylor":
            order = self.whole_number(
                args[3], locals_, 0, "taylor: the degree must be a whole number"
            )
        direction = "+-"
        if name == "limit" and len(args) == 4:
            side = args[3].name if isinstance(args[3], Name) else None
            if side not in ("left", "right"):
                raise ResolveError("limit: the side must be left or right", args[3].span)
            direction = "-" if side == "left" else "+"

        named = _Resolver(
            self.definitions,
            self.broken,
            self.options,
            self.exact_solutions,
            keep_constants=name not in ("integrate", "roots", "apart"),
            cas_results=self.cas_results,
        )
        named.stack, named.depth = self.stack, self.depth  # cycles through definitions
        if name not in ("integrate", "roots"):
            # Symbolic algebra/calculus can keep free parameters. Integrals and
            # roots require values; apart diagnoses conditional decompositions.
            named.unknowns = {}
        parameters: dict[str, Node] = {p: Var(p.split("@")[0], span) for p in locals_}
        if variable is not None:
            parameters[variable] = Var(variable, span)  # diff(t^2, t): t is bound here
        body = named.resolve_root(args[0], parameters)
        if contains(body, ListExpr):
            raise ResolveError(f"{name} can't take a list", args[0].span)
        if variable is None:
            variable = self.cas_variable(body, locals_)
        point: Node | None = None
        if name in ("limit", "taylor"):
            # A number or user constants (kept as names, like the body's): never the
            # variable itself or a function parameter.
            point = named.resolve_root(args[2], parameters)
            if contains(point, ListExpr) or isinstance(point, TupleExpr):
                raise ResolveError(f"{name}: the point must be a number", args[2].span)
            constants = {n for n, d in self.definitions.items() if d.params is None}
            others = free_vars(point) - (constants - {variable})
            if others:
                raise ResolveError(
                    f"{name}: the point must be a number or constants, not use "
                    + ", ".join(sorted(others)),
                    args[2].span,
                )
        request = CasRequest(name, body, variable, self.options, order, point, direction)
        result = self.cas_results.get(request)
        if result is None:
            raise PendingError(request, span)
        if isinstance(result, CasFailure):
            raise ResolveError(f"{name}: {result.message}", span)
        if not self.foreign_depth:
            self.cas_used[span] = result
        parsed = parse(result)
        if isinstance(parsed, Relation | ComparisonChain):
            raise ResolveError(f"{name}: the result is an equation", span)
        resolved = self.resolve(parsed, locals_)
        if isinstance(resolved, TupleExpr):  # pragma: no cover - SymPy returns numbers
            raise ResolveError(f"{name}: the result is not a number", span)
        if name in ALGEBRA_OPERATIONS or name in ("diff", "integrate"):
            original = self.resolve_root(args[0], {**parameters, **locals_})
            if domain_sensitive(original):
                resolved = Guard(resolved, original, span)
                if not self.foreign_depth:
                    self.cas_used[span] = result + "  (original domain preserved)"
        return resolved

    def whole_number(self, arg: Node, locals_: Mapping[str, Node], least: int, message: str) -> int:
        """A count argument (``diff``'s order, ``taylor``'s degree): a whole number
        from ``least`` to MAX_DIFF_ORDER, possibly given by user constants' values."""
        count = self.resolve(arg, locals_)
        value = math.nan if free_vars(count) else NumericFunction(count).scalar()
        if not (math.isfinite(value) and value == int(value) and least <= value <= MAX_DIFF_ORDER):
            raise ResolveError(f"{message} from {least} to {MAX_DIFF_ORDER}", arg.span)
        return int(value)

    def cas_variable(self, body: Node, locals_: Mapping[str, Node]) -> str:
        """The variable ``diff``/``integrate`` act on by default: x if the argument uses
        it, else y, else its one parameter or unknown (not a user constant), else x."""
        constants = {n for n, d in self.definitions.items() if d.params is None}
        candidates = free_vars(body) - constants
        for name in PLOT_VARIABLES:
            if name in candidates:
                return name
        parameters = candidates & {p.split("@")[0] for p in locals_}
        if len(parameters) == 1:
            return next(iter(parameters))
        if len(candidates) == 1:
            return next(iter(candidates))
        return "x"

    def list_function(self, name: str, args: tuple[Node, ...], span: Span) -> Node:
        if len(args) != 1:
            raise ResolveError(f"{name}(list) takes one list", span)
        if any(intervals(args[0])):
            raise ResolveError(
                f"{name} needs a finite list; an interval has continuous values", span
            )
        items = _expand(args[0]) if contains(args[0], ListExpr) else [args[0]]
        if name == "length":
            return Num(float(len(items)), span)
        if not items:
            return Num(math.nan if name != "total" else 0.0, span)
        if name in ("total", "mean"):
            total = _balanced("+", items, span)
            return total if name == "total" else BinOp("/", total, Num(float(len(items))), span)
        return items[0] if len(items) == 1 else FuncCall(name, tuple(items), span)

    def reduce(
        self,
        kind: str,
        name: str,
        args: tuple[Node, ...],
        span: Span,
        locals_: Mapping[str, Node],
    ) -> Node:
        """Σ, Π or ∫: ``op(body, var, lower, upper)`` or ``op(body, lower, upper)``."""
        if len(args) == 4:
            if not isinstance(args[1], Name):
                raise ResolveError(f"{name}: the second argument names the variable", args[1].span)
            var = args[1].name
            body, lower, upper = args[0], args[2], args[3]
        elif len(args) == 3:
            body, lower, upper = args
            var = self.infer_variable(kind, name, body, locals_)
        else:
            example = (
                f"{name}(t^2 for t from 0 to 1)"
                if kind == "int"
                else f"{name}(k^2 for k from 1 to 10)"
            )
            raise ResolveError(
                f"{name} takes (body, lower, upper) or (body, variable, lower, upper), "
                f"like {example}",
                span,
            )
        if var in FUNCTIONS or var in CONSTANTS or var in REDUCERS:
            raise ResolveError(f"'{var}' is a built-in name; use another variable", span)
        resolved_lower = self.resolve(lower, locals_)
        resolved_upper = self.resolve(upper, locals_)
        bound = f"{var}@{self.depth}"
        self.depth += 1
        try:
            resolved_body = self.resolve(body, {**locals_, var: Var(bound, span)})
        finally:
            self.depth -= 1
        for original, value in (
            (body, resolved_body),
            (lower, resolved_lower),
            (upper, resolved_upper),
        ):
            _reject_interval(value, original.span)
            if isinstance(value, TupleExpr):
                raise ResolveError(f"{name} needs numbers, not points", original.span)
        return Reduce(kind, bound, resolved_body, resolved_lower, resolved_upper, span)

    def infer_variable(self, kind: str, name: str, body: Node, locals_: Mapping[str, Node]) -> str:
        """Three-argument integrals bind x; sums/products require an explicit index."""
        if kind == "int":
            return "x"
        raise ResolveError(f"Name the index explicitly, like {name}(k^2, k, 1, 10)", body.span)

    @staticmethod
    def _product(factors: list[Node], span: Span) -> Node:
        result = factors[0]
        for factor in factors[1:]:
            result = _combine("*", result, factor, span)
        return result


def _reject_sequence(value: Node, span: Span) -> None:
    if isinstance(value, SequenceExpr):
        raise ResolveError("Select finite sequence terms explicitly, like S[1...100]", span)


def _reject_interval(value: Node, span: Span) -> None:
    """Operations requiring one number cannot consume continuous collections."""
    _reject_sequence(value, span)
    if any(intervals(value)):
        raise ResolveError("An interval can't be used here; this needs a single number", span)


def _format_interval(node: IntervalExpr) -> str:
    assert isinstance(node.lo, Num) and isinstance(node.hi, Num)
    left = "[" if node.closed_lo else "("
    right = "]" if node.closed_hi else ")"
    return f"{left}{format_value(node.lo.value)}, {format_value(node.hi.value)}{right}"


# -- points as vectors ----------------------------------------------------------------------


def _combine(op: str, left: Node, right: Node, span: Span) -> Node:
    """``left op right``, with points (``TupleExpr``) as 2-D vectors: points add and
    subtract component-wise and scale by numbers; anything else with a point is an error."""
    if not isinstance(left, TupleExpr) and not isinstance(right, TupleExpr):
        return BinOp(op, left, right, span)
    if op in ("+", "-"):
        if not (isinstance(left, TupleExpr) and isinstance(right, TupleExpr)):
            verb = "add" if op == "+" else "subtract"
            raise ResolveError(f"Can't {verb} a number and a point", span)
        pairs = zip(left.items, right.items, strict=True)
        return TupleExpr(tuple(BinOp(op, a, b, span) for a, b in pairs), span)
    if op == "*":
        if isinstance(left, TupleExpr) and isinstance(right, TupleExpr):
            raise ResolveError("Can't multiply two points; scale a point by a number", span)
        if isinstance(left, TupleExpr):
            return TupleExpr(tuple(BinOp("*", c, right, span) for c in left.items), span)
        assert isinstance(right, TupleExpr)
        return TupleExpr(tuple(BinOp("*", left, c, span) for c in right.items), span)
    if op == "/":
        if isinstance(right, TupleExpr):
            raise ResolveError("Can't divide by a point", span)
        assert isinstance(left, TupleExpr)
        return TupleExpr(tuple(BinOp("/", c, right, span) for c in left.items), span)
    if isinstance(left, TupleExpr):
        raise ResolveError("Can't raise a point to a power", span)
    raise ResolveError("A point can't be an exponent", span)


def _point_plot(point: TupleExpr, span: Span) -> Plot | None:
    """The plot of a resolved point; its coordinates may hold lists (several points)."""
    x, y = point.items
    if free_vars(x) or free_vars(y):
        raise ResolveError("Point coordinates must be numbers", span)
    if any(intervals(point)):
        plots = [
            plot
            for variant in _curves(_expand(point), span)
            if (plot := _continuous_point(cast(TupleExpr, variant))) is not None
        ]
        return _group(plots) if plots else None
    joint = _joint_values([x, y], False)  # lists in x and y zip together
    if joint is None:
        return PointPlot(NumericFunction(x).scalar(), NumericFunction(y).scalar())
    pairs = tuple(zip(joint[0].tolist(), joint[1].tolist(), strict=True))
    if len(pairs) > MAX_LIST:
        raise ResolveError(f"Too many points; the limit is {MAX_LIST}", span)
    return PointSetPlot(pairs) if pairs else None


def _continuous_point(point: TupleExpr, *, complex_mode: bool = False) -> Plot | None:
    coordinates, domain = parameterize(point.items)
    if domain.lo == domain.hi:
        if not (domain.closed_lo and domain.closed_hi):
            return None
        return PointPlot(*(NumericFunction(c, complex_mode).scalar(t=0) for c in coordinates))
    x, y = (Bounded(c, Var("t"), domain) for c in coordinates)
    return ParametricPlot(x, y, 0, 1, complex_mode)


def _continuous_complex_plot(node: Node) -> Plot | None:
    """Trace continuous complex values as (real, imaginary) coordinates."""
    plots: list[Plot] = []
    for variant in _curves(_expand(node), node.span):
        coordinates = TupleExpr(
            (FuncCall("re", (variant,)), FuncCall("im", (variant,))), variant.span
        )
        plot = _continuous_point(coordinates, complex_mode=True)
        if plot is None:
            continue
        if isinstance(plot, PointPlot) and not (np.isfinite(plot.x) and np.isfinite(plot.y)):
            continue
        plots.append(plot)
    return _group(plots) if plots else None


def _continuous_message(node: Node, options: CompileOptions) -> str:
    from opengraph.engine.source import source

    return f"= {source(node, degrees=options.degrees)}  (continuous values)"


class _Expansion:
    """Context manager tracking definitions being expanded.

    Detects cycles, and re-labels errors raised while expanding another row's
    definition so they read "'f' has an error: ..." and point at the reference in the
    current row (spans from the other row's text would be meaningless here).
    """

    def __init__(self, resolver: _Resolver, name: str, span: Span, *, foreign: bool) -> None:
        self.resolver, self.name, self.span, self.foreign = resolver, name, span, foreign

    def __enter__(self) -> None:
        stack = self.resolver.stack
        if self.name in stack:
            cycle = " → ".join([*stack[stack.index(self.name) :], self.name])
            raise _CycleError(f"Circular definition: {cycle}", self.span)
        stack.append(self.name)
        self.resolver.foreign_depth += self.foreign

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.resolver.stack.pop()
        self.resolver.foreign_depth -= self.foreign
        if not (self.foreign and isinstance(exc, ResolveError)):
            return
        exc.span = self.span  # the outermost foreign expansion wins
        if not isinstance(exc, _CycleError | _ForeignError | PendingError):
            exc.__class__ = _ForeignError
            exc.message = f"'{self.name}' has an error: {exc.message}"
            exc.args = (exc.message,)


class _CycleError(ResolveError):
    pass


class _ForeignError(ResolveError):
    """An error that originated in another row's definition (already re-labelled)."""


# -- lists ----------------------------------------------------------------------------------


def _list_length(nodes: Sequence[Node]) -> int | None:
    lengths = [len(n.items) for node in nodes for n in walk(node) if isinstance(n, ListExpr)]
    if lengths and any(length != lengths[0] for length in lengths):
        raise ResolveError("Lists must have equal lengths; use explicit slices to pair elements")
    return lengths[0] if lengths else None


def _pick(node: Node, index: int) -> Node:
    return replace(node, lambda n: n.items[index] if isinstance(n, ListExpr) else None)


def _broadcast(nodes: Sequence[Node]) -> list[tuple[Node, ...]] | None:
    """Element-wise copies of ``nodes`` (list lengths must agree), or
    None when no list is involved."""
    length = _list_length(nodes)
    if length is None:
        return None
    return [tuple(_pick(node, j) for node in nodes) for j in range(length)]


def _expand(node: Node) -> list[Node]:
    picked = _broadcast([node])
    return [node] if picked is None else [p[0] for p in picked]


def _joint_values(nodes: Sequence[Node], cx: bool) -> list[np.ndarray] | None:
    """Values of closed list-valued expressions, broadcast together and vectorised.

    Each list is evaluated once into an array and replaced by a variable, so a data
    list of thousands of numbers costs one NumPy evaluation. None when no list is used.
    """
    lists: dict[ListExpr, str] = {}
    for node in nodes:
        for n in walk(node):
            if isinstance(n, ListExpr) and n not in lists:
                lists[n] = f"#list{len(lists)}"
    if not lists:
        return None
    dtype = np.complex128 if cx else np.float64
    length = _list_length(nodes)
    assert length is not None
    if length == 0:
        return [np.empty(0, dtype=dtype) for _ in nodes]
    arrays: dict[str, np.ndarray] = {}
    for lst, key in lists.items():
        values: list[complex] = []
        for item in lst.items[:length]:
            if isinstance(item, Num):
                values.append(item.value)
            else:
                fn = NumericFunction(item, cx)
                values.append(fn.complex_scalar() if cx else fn.scalar())
        arrays[key] = np.array(values, dtype=dtype)
    results = []
    for node in nodes:
        substituted = replace(node, lambda n: Var(lists[n]) if isinstance(n, ListExpr) else None)
        fn = NumericFunction(substituted, cx)
        values_array = fn.complex_values(**arrays) if cx else fn(**arrays)
        results.append(np.broadcast_to(values_array, (length,)).copy())
    return results


def _integer_member(position: int) -> int:
    """The zero-based enumeration of ℤ: 0, 1, -1, 2, -2, …, without building a list."""
    return (position + 1) // 2 if position % 2 else -(position // 2)


def _format_sequence(node: SequenceExpr, resolver: _Resolver) -> str:
    preview = resolver.resolve_sequence_index(
        node,
        RangeExpr(Num(1), Num(8)),
        {},
    )
    values = _list_values(preview, resolver.options.complex_mode)
    assert values is not None
    variable = node.var.split("@")[0]
    terms = ", ".join(_format_number(v) for v in values)
    return f"= seq[{node.text} for {variable} in sets[{node.domain}]]  (preview: {terms}, …)"


def _list_values(node: Node, cx: bool) -> np.ndarray | None:
    values = _joint_values([node], cx)
    return None if values is None else values[0]


def _format_values(values: np.ndarray) -> str:
    shown = [_format_number(v) for v in values[:MAX_SHOWN_VALUES]]
    if values.size > MAX_SHOWN_VALUES:
        shown.append(f"… ({values.size} items)")
    return "list[" + ", ".join(shown) + "]"


def _format_number(value: complex | float) -> str:
    if isinstance(value, complex | np.complexfloating):
        return format_complex(complex(value))
    return format_value(float(value))


# -- classification ----------------------------------------------------------------------


def _literal_number(node: Node) -> float | None:
    """The value of a plain (optionally negated) finite number literal, else None."""
    if isinstance(node, UnaryOp) and node.op == "-" and isinstance(node.operand, Num):
        value = -node.operand.value
    elif isinstance(node, Num):
        value = node.value
    else:
        return None
    return value if math.isfinite(value) else None


def _group(plots: list[Plot]) -> Plot:
    return plots[0] if len(plots) == 1 else PlotGroup(tuple(plots))


def _curves(nodes: list[Node], span: Span) -> list[Node]:
    if len(nodes) > MAX_CURVES:
        raise ResolveError(f"Too many curves ({len(nodes)}); the limit is {MAX_CURVES}", span)
    return nodes


def _explicit(node: Node, independent: str, options: CompileOptions, span: Span) -> RowResult:
    if any(intervals(node)):
        continuous_plots = [
            plot
            for variant in _curves(_expand(node), span)
            if (plot := _continuous_explicit(variant, independent, span)) is not None
        ]
        return RowResult(plot=_group(continuous_plots) if continuous_plots else None)
    if (
        isinstance(node, Bounded)
        and node.coordinate == Var(independent)
        and node.interval.lo == node.interval.hi
    ):
        if not (node.interval.closed_lo and node.interval.closed_hi):
            return RowResult(message="Empty plot range: nothing to plot")
        assert isinstance(node.interval.lo, Num)
        position = node.interval.lo.value
        value = NumericFunction(node, options.complex_mode).scalar(**{independent: position})
        x, y = (position, value) if independent == "x" else (value, position)
        return RowResult(plot=PointPlot(x, y))
    variants = _expand(node)
    plots: list[Plot] = [
        ExplicitPlot(v, independent, options.complex_mode) for v in _curves(variants, span)
    ]
    symbolic = None
    if len(variants) == 1 and not contains(node, ListExpr):
        symbolic = SymbolicTarget(node, independent, options)
    if not plots:
        return RowResult(message="Empty list: nothing to plot")
    return RowResult(plot=_group(plots), symbolic=symbolic)


def _continuous_explicit(node: Node, independent: str, span: Span) -> Plot | None:
    """An affine continuous family fills its exact band, including reversed slopes."""
    (body,), domain = parameterize([node], "#interval")
    if domain.lo == domain.hi and not (domain.closed_lo and domain.closed_hi):
        return None
    parts = _linear_parts(body, "#interval")
    if parts is None:
        raise ResolveError(
            "This continuous curve family is nonlinear; use point[interval, expression] "
            "to draw a curve through its values",
            span,
        )
    start, step = parts
    end = BinOp("+", start, step)
    dependent = Var("y" if independent == "x" else "x")
    first, last = BinOp("-", dependent, start), BinOp("-", dependent, end)
    plots: list[Plot] = []
    for increasing in (True, False):
        first_op = ">" if increasing else "<"
        last_op = "<" if increasing else ">"
        if domain.closed_lo:
            first_op += "="
        if domain.closed_hi:
            last_op += "="
        direction = (step, ">" if increasing else "<")
        plots.append(
            RegionPlot(
                first, first_op, ExplicitPlot(start, independent), ((last, last_op), direction)
            )
        )
        plots.append(
            RegionPlot(
                last,
                last_op,
                ExplicitPlot(end, independent),
                ((first, first_op), direction),
                shade=False,
            )
        )
    # When all values coincide the result is still a curve, even for open intervals.
    plots.append(
        ExplicitPlot(Bounded(start, step, IntervalExpr(Num(0), Num(0), True, True)), independent)
    )
    return _group(plots)


def _value_row(node: Node, options: CompileOptions) -> RowResult:
    cx = options.complex_mode
    if any(intervals(node)):
        return RowResult(
            plot=_continuous_complex_plot(node) if cx else None,
            message=_continuous_message(node, options),
        )
    values = _list_values(node, cx)
    if values is not None:
        plot: Plot | None = None
        if cx and values.size:
            plot = PointSetPlot(tuple((float(v.real), float(v.imag)) for v in values))
        return RowResult(plot=plot, message=f"= {_format_values(values)}")
    fn = NumericFunction(node, cx)
    target = SymbolicTarget(node, "x", options, value=True)
    if cx:
        z = fn.complex_scalar()
        point = PointPlot(z.real, z.imag) if np.isfinite(z) else None
        return RowResult(plot=point, message=f"= {format_complex(z)}", symbolic=target)
    return RowResult(message=f"= {format_value(fn.scalar())}", symbolic=target)


def _point_coordinates(items: Sequence[Node], resolver: _Resolver, span: Span) -> list[Node]:
    """Point coordinates are numeric; parameters belong to parametric(...)."""
    if len(items) != 2:
        raise ResolveError("A point needs exactly two coordinates, like point[1, 2]", span)
    coords: list[Node] = []
    for item in items:
        node = resolver.resolve_value(item)
        if isinstance(node, TupleExpr):
            raise ResolveError("Point coordinates can't be points", item.span)
        _reject_sequence(node, item.span)
        if free_vars(node):
            raise ResolveError(
                "Point coordinates must be numbers; use parametric for curves", item.span
            )
        coords.append(node)
    return coords


def _drag_coordinate(node: Node) -> DragCoordinate:
    if isinstance(node, Name):
        return node.name
    return node.span if _literal_number(node) is not None else None


def _points(stmt: TupleExpr | ListExpr, resolver: _Resolver) -> RowResult:
    """``(a, b)``, ``(a, b), (c, d)``, ``[(a, b), …]`` or ``(k, k^2)`` with a list k."""
    tuples = [stmt] if isinstance(stmt, TupleExpr) else list(stmt.items)
    resolved_points = []
    for item in tuples:
        if not isinstance(item, TupleExpr):
            raise ResolveError("A list of points must contain only points", item.span)
        resolved_points.append(
            TupleExpr(tuple(_point_coordinates(item.items, resolver, item.span)))
        )
    if any(any(intervals(point)) for point in resolved_points):
        plots = [
            plot for point in resolved_points if (plot := _point_plot(point, stmt.span)) is not None
        ]
        return RowResult(plot=_group(plots) if plots else None)
    pairs: list[tuple[float, float]] = []
    for point in resolved_points:
        x, y = point.items
        joint = _joint_values([x, y], False)  # lists in x and y zip together
        if joint is None:
            pairs.append((NumericFunction(x).scalar(), NumericFunction(y).scalar()))
        else:
            pairs.extend(zip(joint[0].tolist(), joint[1].tolist(), strict=True))
    if isinstance(stmt, TupleExpr) and len(pairs) == 1:
        drag = (_drag_coordinate(stmt.items[0]), _drag_coordinate(stmt.items[1]))
        return RowResult(plot=PointPlot(*pairs[0]), drag=drag)
    if len(pairs) > MAX_LIST:
        raise ResolveError(f"Too many points; the limit is {MAX_LIST}", stmt.span)
    return RowResult(plot=PointSetPlot(tuple(pairs)) if pairs else None)


def _classify(
    stmt: Statement, resolver: _Resolver, defn: _Definition | None, options: CompileOptions
) -> RowResult:
    resolver.cas_used.clear()
    result = _classify_row(stmt, resolver, defn, options)
    shown = _cas_result_text(stmt, defn, resolver)
    if shown is not None and not result.message:
        result = dataclasses.replace(result, message=f"= {shown}")
    target = result.symbolic
    if target is None or target.value or target.variable not in PLOT_VARIABLES:
        return result
    found = _named_node(stmt, defn, resolver, options, target.variable, target.equation)
    if found is None:
        return result
    named, unsolved = found
    target = dataclasses.replace(
        target,
        named=named,
        named_unsolved=unsolved,
        constants=_constant_values(named, resolver),
    )
    return dataclasses.replace(result, symbolic=target)


def _cas_result_text(stmt: Statement, defn: _Definition | None, resolver: _Resolver) -> str | None:
    """The result of the computer-algebra call a row consists of (``factor(x^2 - 1)``,
    ``y = diff(x^3)``, ``g(x) = integrate(f(x))``), to show in its feedback."""
    top: Statement = stmt
    if defn is not None:
        top = defn.body
    elif isinstance(stmt, Relation) and stmt.op == "=":
        top = stmt.right
    if isinstance(top, Call) and top.name in CAS_OPERATIONS:
        return resolver.cas_used.get(top.span)
    return None


def _constant_values(named: Node, resolver: _Resolver) -> tuple[tuple[str, Node], ...]:
    """The value of each user constant ``named`` uses, as a closed tree; empty when one
    has no single number for a value (a list, a point, an error)."""
    values = []
    for name in sorted(free_vars(named) - set(PLOT_VARIABLES)):
        try:
            value = resolver.resolve_root(Name(name))
        except (MathError, RecursionError):
            return ()
        if free_vars(value) or contains(value, ListExpr):
            return ()
        values.append((name, value))
    return tuple(values)


def _named_node(
    stmt: Statement,
    defn: _Definition | None,
    resolver: _Resolver,
    options: CompileOptions,
    variable: str,
    equation: bool = False,
) -> tuple[Node, str | None] | None:
    """The row's curve with user constants kept as names (``y = a x`` stays ``a x``).

    For rows written as a curve in ``variable`` — ``f(x) = …``, a plain expression in
    x, or ``y = …`` / ``x = …`` — that curve and None. For other equations (``2y = a x``)
    the equation as ``node = 0`` and the name it is to be solved for. Residual targets
    retain the equation directly, without solving for or assigning any variable.
    """
    if isinstance(stmt, ComparisonChain):
        return None
    named = _Resolver(
        resolver.definitions,
        resolver.broken,
        options,
        resolver.exact_solutions,
        keep_constants=True,
        cas_results=resolver.cas_results,
    )
    dependent = Var("y" if variable == "x" else "x")
    try:
        if defn is not None:
            if defn.params is None:
                return None
            locals_ = {p: Var(p) for p in defn.params}
            with named.own(defn.name):
                body = named.resolve_root(defn.body, locals_)
                if len(defn.params) == 1:
                    parameter = defn.params[0]
                    body = replace(body, lambda n: Var(variable) if n == Var(parameter) else None)
                return body, None
        if not isinstance(stmt, Relation):
            return named.resolve_root(stmt), None
        left, right = named.resolve_root(stmt.left), named.resolve_root(stmt.right)
    except (MathError, RecursionError):
        return None
    if equation:
        return BinOp("-", left, right, stmt.span), None
    for side, other in ((left, right), (right, left)):
        if side == dependent and dependent.name not in free_vars(other):
            return other, None
    if dependent.name not in free_vars(left) | free_vars(right):
        return None  # polar rows, rewritten in x and y
    return BinOp("-", left, right, stmt.span), dependent.name


def _classify_row(
    stmt: Statement, resolver: _Resolver, defn: _Definition | None, options: CompileOptions
) -> RowResult:
    if isinstance(stmt, ComparisonChain):
        regions: list[RegionPlot] = []
        for comparison in stmt.comparisons:
            result = _classify_inequality(comparison, resolver, options)
            if result.plot is None:
                if result.message == "false":
                    return result
            elif isinstance(result.plot, RegionPlot):
                regions.append(result.plot)
            else:
                raise ResolveError("Chained inequalities require scalar expressions", stmt.span)
        if not regions:
            return RowResult(message="true")
        return RowResult(
            plot=_group(
                [
                    dataclasses.replace(
                        region,
                        shade=index == 0,
                        constraints=tuple(
                            (other.node, other.op) for j, other in enumerate(regions) if j != index
                        ),
                    )
                    for index, region in enumerate(regions)
                ]
            )
        )
    if isinstance(stmt, Call) and stmt.name == "formula":
        if len(stmt.args) != 2 or not isinstance(stmt.args[0], Name):
            raise ResolveError("Use formula(name, expression)", stmt.span)
        from opengraph.engine.source import source

        named = _Resolver(
            resolver.definitions,
            resolver.broken,
            options,
            keep_constants=True,
            cas_results=resolver.cas_results,
        )
        named.unknowns = {}
        body = named.resolve_root(stmt.args[1])
        return RowResult(message=f"{stmt.args[0].name} = {source(body, degrees=options.degrees)}")

    if isinstance(stmt, Call) and stmt.name == "parametric":
        if len(stmt.args) != 4 or not isinstance(stmt.args[2], Name):
            raise ResolveError("Use parametric(cos(u), sin(u) for u in [0, 2pi])", stmt.span)
        parameter = stmt.args[2].name
        if parameter in RESERVED - set(PLOT_VARIABLES):
            raise ResolveError("Choose an ordinary parameter name", stmt.args[2].span)
        interval = resolver.resolve_value(stmt.args[3])
        if not isinstance(interval, IntervalExpr):
            raise ResolveError("The parameter range must be an interval", stmt.args[3].span)
        assert isinstance(interval.lo, Num) and isinstance(interval.hi, Num)
        if interval.lo.value == interval.hi.value and not (
            interval.closed_lo and interval.closed_hi
        ):
            return RowResult(message="Empty parameter range: nothing to plot")
        coordinates = [resolver.resolve_root(a, {parameter: Var("t")}) for a in stmt.args[:2]]
        if any(free_vars(c) - {"t"} or contains(c, ListExpr) for c in coordinates):
            raise ResolveError("Parametric coordinates must use only their parameter", stmt.span)
        if interval.lo.value == interval.hi.value:
            a, b = (
                NumericFunction(c, options.complex_mode).scalar(t=interval.lo.value)
                for c in coordinates
            )
            return RowResult(plot=PointPlot(a, b))
        x, y = (Bounded(c, Var("t"), interval) for c in coordinates)
        return RowResult(
            plot=ParametricPlot(x, y, interval.lo.value, interval.hi.value, options.complex_mode)
        )

    if isinstance(stmt, Relation) and stmt.op != "=":
        return _classify_inequality(stmt, resolver, options)

    if defn is not None:
        if defn.params is None:
            with resolver.own(defn.name):
                value_node = resolver.constant(defn.name)
            if isinstance(value_node, IntervalExpr):  # "I = [0, 1)"
                return RowResult(message=f"= {_format_interval(value_node)}", defines=defn.name)
            if isinstance(value_node, TupleExpr):  # "p = point[1, 2]" or "r = p + q"
                plot = _point_plot(value_node, defn.body.span)
                drag = None
                if isinstance(defn.body, TupleExpr) and isinstance(plot, PointPlot):
                    drag = (
                        _drag_coordinate(defn.body.items[0]),
                        _drag_coordinate(defn.body.items[1]),
                    )
                return RowResult(plot=plot, defines=defn.name, drag=drag)
            literal = _literal_number(defn.body)
            if isinstance(value_node, SequenceExpr):
                return RowResult(message=_format_sequence(value_node, resolver), defines=defn.name)
            if any(intervals(value_node)):
                return RowResult(
                    message=_continuous_message(value_node, options), defines=defn.name
                )
            values = _list_values(value_node, options.complex_mode)
            if values is not None:
                return RowResult(message=f"= {_format_values(values)}", defines=defn.name)
            if options.complex_mode:
                z = NumericFunction(value_node, True).complex_scalar()
                message = "" if literal is not None else f"= {format_complex(z)}"
            else:
                value = NumericFunction(value_node).scalar()
                message = "" if literal is not None else f"= {format_value(value)}"
            return RowResult(message=message, defines=defn.name, literal_value=literal)
        locals_ = {p: Var(p) for p in defn.params}
        with resolver.own(defn.name):
            body = resolver.resolve_root(defn.body, locals_, continuous=True)
        if len(defn.params) == 1 and free_vars(body) <= {defn.params[0]}:
            parameter = defn.params[0]
            body = replace(body, lambda n: Var("x") if n == Var(parameter) else None)
            result = _explicit(body, "x", options, defn.body.span)
            return RowResult(plot=result.plot, defines=defn.name, symbolic=result.symbolic)
        return RowResult(defines=defn.name)

    if isinstance(stmt, Relation):
        return _classify_equation(stmt, resolver, options)

    if isinstance(stmt, TupleExpr) or (
        isinstance(stmt, ListExpr) and stmt.items and isinstance(stmt.items[0], TupleExpr)
    ):
        return _points(stmt, resolver)

    if isinstance(stmt, Call) and stmt.name in CAS_OPERATIONS.keys() - {"integrate", "roots"}:
        # A standalone symbolic request can display a formula even when its free
        # symbols have no values or cannot be plotted as a function of x.
        previous_unknowns = resolver.unknowns
        resolver.unknowns = {}
        try:
            node = resolver.resolve_value(stmt)
        finally:
            resolver.unknowns = previous_unknowns
        if free_vars(node) - {"x"}:
            return RowResult(message=f"= {resolver.cas_used[stmt.span]}")
    else:
        node = resolver.resolve_value(stmt)
    if isinstance(node, SequenceExpr):
        return RowResult(message=_format_sequence(node, resolver))
    if isinstance(node, IntervalExpr):
        return RowResult(
            plot=_continuous_complex_plot(node) if options.complex_mode else None,
            message=f"= {_format_interval(node)}",
        )
    if isinstance(node, TupleExpr):  # vector arithmetic: "point[1, 2] + point[3, 4]", "2p"
        return RowResult(plot=_point_plot(node, stmt.span))
    used = free_vars(node)
    if not used:
        return _value_row(node, options)
    if used == {"x"}:
        return _explicit(node, "x", options, stmt.span)
    if used <= {THETA, RADIUS}:
        raise ResolveError("Polar expressions need an equation, like r = sin(θ)", stmt.span)
    raise ResolveError("Expressions using y need an equation, like x = y^2", stmt.span)


def _linear_parts(node: Node, variable: str) -> tuple[Node, Node] | None:
    """Prove node = a + b*variable without evaluating sample points."""
    if variable not in free_vars(node):
        return node, Num(0)
    if node == Var(variable):
        return Num(0), Num(1)
    if isinstance(node, UnaryOp):
        parts = _linear_parts(node.operand, variable)
        return (UnaryOp("-", parts[0]), UnaryOp("-", parts[1])) if parts else None
    if not isinstance(node, BinOp):
        return None
    left, right = _linear_parts(node.left, variable), _linear_parts(node.right, variable)
    if left is None or right is None:
        return None
    a, b = left
    c, d = right
    if node.op in ("+", "-"):
        return BinOp(node.op, a, c), BinOp(node.op, b, d)
    if node.op == "*":
        if variable not in free_vars(node.left):
            return BinOp("*", a, c), BinOp("*", a, d)
        if variable not in free_vars(node.right):
            return BinOp("*", a, c), BinOp("*", b, c)
    if node.op == "/" and not free_vars(node.right):
        divisor = NumericFunction(node.right).scalar()
        if math.isfinite(divisor) and divisor != 0:
            return BinOp("/", a, c), BinOp("/", b, c)
    return None


def _constant_fold(node: Node, options: CompileOptions) -> Node:
    """Fold constant arithmetic, including zero coefficients, without erasing domains."""
    if not free_vars(node):
        return Num(NumericFunction(node, options.complex_mode).scalar())
    if isinstance(node, BinOp):
        a, b = _constant_fold(node.left, options), _constant_fold(node.right, options)
        if node.op in ("+", "-") and b == Num(0):
            return a
        if node.op == "+" and a == Num(0):
            return b
        if node.op == "*" and (a == Num(0) or b == Num(0)):
            # Used only to prove coefficients; the original expression guards the result.
            return Num(0)
        return BinOp(node.op, a, b)
    if isinstance(node, UnaryOp):
        inner = _constant_fold(node.operand, options)
        return Num(-inner.value) if isinstance(inner, Num) else UnaryOp("-", inner)
    return node


def _solve_linear(difference: Node, var: str, options: CompileOptions) -> Node | None:
    parts = _linear_parts(difference, var)
    if parts is None:
        return None
    at_zero, coefficient = parts
    coefficient = _constant_fold(coefficient, options)
    if free_vars(coefficient):
        return None
    value = NumericFunction(coefficient, options.complex_mode).scalar()
    if not math.isfinite(value) or value == 0:
        return None
    return BinOp("/", UnaryOp("-", at_zero), coefficient)


def _polar_to_cartesian(node: Node, options: CompileOptions) -> Node:
    """Substitute r = √(x² + y²) and θ = atan(y, x) (in the angle unit)."""
    x, y = Var("x"), Var("y")
    radius = FuncCall("sqrt", (BinOp("+", BinOp("^", x, Num(2.0)), BinOp("^", y, Num(2.0))),))
    angle: Node = FuncCall("atan", (y, x))
    if options.degrees:
        angle = BinOp("*", angle, _TO_DEGREES)

    def swap(n: Node) -> Node | None:
        if n == Var(RADIUS):
            return radius
        if n == Var(THETA):
            return angle
        return None

    return replace(node, swap)


def _equation_value(difference: Node, options: CompileOptions) -> RowResult:
    values = _list_values(difference, options.complex_mode)
    if values is None:
        values = np.array([NumericFunction(difference, options.complex_mode).complex_scalar()])
    truth = np.abs(values) <= 1e-9 * (1 + np.abs(values))
    words = ["true" if t else "false" for t in truth]
    return RowResult(message=words[0] if len(words) == 1 else "list[" + ", ".join(words) + "]")


def _equation_sides(stmt: Relation, resolver: _Resolver) -> tuple[Node, Node, dict[str, Span]]:
    """Both sides resolved, with unknown names as variables (returned with their spans)."""

    def side(node: Node) -> Node:
        # "r = f(θ)" on the Cartesian grid: r is the polar radius when θ is a variable.
        if isinstance(node, Name) and node.name == RADIUS and RADIUS not in resolver.definitions:
            return Var(RADIUS, node.span)
        return resolver.resolve_root(node, continuous=True)

    unknowns: dict[str, Span] = {}
    resolver.unknowns = unknowns
    try:
        left, right = side(stmt.left), side(stmt.right)
    finally:
        resolver.unknowns = None
    return left, right, unknowns


@dataclass(frozen=True, slots=True)
class _Solutions:
    """Real solutions of an equation for its unknown ``name``."""

    name: str
    values: tuple[float, ...]
    texts: tuple[str, ...] | None = None  # exact forms, when solved symbolically
    exact: bool = False  # the values are exact (shown with "=", not "≈")
    truncated: bool = False
    general: GeneralSolution | None = None  # infinitely many; ``values`` are a sample

    def message(self) -> str:
        if self.general is not None:
            return self.general.text(self.name)
        if self.texts is None:
            return _solutions_text(self.name, self.values, self.exact, self.truncated)
        if not self.texts:
            return "No real solutions"
        message = f"{self.name} = {_listed(self.texts)}"
        if not all(_is_number_text(t) for t in self.texts):  # "a = sqrt(2)  (≈ 1.41421)"
            message += f"  (≈ {_listed([format_value(v, 6) for v in self.values])})"
        return message


def _listed(texts: Sequence[str]) -> str:
    """``sqrt(2)`` for one solution, ``list[-sqrt(2), sqrt(2)]`` for several."""
    return texts[0] if len(texts) == 1 else "list[" + ", ".join(texts) + "]"


def _is_number_text(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def _unknown_solutions(
    difference: Node,
    unknowns: Mapping[str, Span],
    options: CompileOptions,
    span: Span,
    exact_solutions: Mapping[SymbolicTarget, ExactRoots],
) -> _Solutions:
    """Solve ``difference = 0`` for its one unknown name.

    Exact roots in ``exact_solutions`` (keyed by the row's symbolic target) are used when
    present; otherwise a linear equation is solved directly and anything else by the
    numerical scan. Raises when the equation also uses the plotting variables, several
    unknowns, or lists.
    """
    names = sorted(unknowns, key=lambda n: unknowns[n])
    if free_vars(difference) - set(names):
        raise ResolveError(f"Unknown name '{names[0]}'", unknowns[names[0]])
    if len(names) > 1:
        listed = ", ".join(names[:-1]) + f" and {names[-1]}"
        raise ResolveError(
            f"Can't solve for {listed} at once; define all but one of them", unknowns[names[1]]
        )
    name = names[0]
    if contains(difference, ListExpr):
        raise ResolveError("Equations with lists can't be solved for an unknown", span)
    roots = exact_solutions.get(SymbolicTarget(difference, name, options))
    if isinstance(roots, GeneralSolution):
        # General solutions affect feedback only; no generated values become bindings.
        near_zero = sorted(roots.values(-100, 100), key=abs)[:MAX_SHOWN_VALUES]
        return _Solutions(name, tuple(sorted(near_zero)), exact=True, general=roots)
    if roots is not None:
        texts = tuple(text for text, _ in roots)
        return _Solutions(name, tuple(value for _, value in roots), texts, exact=True)
    solved = _solve_linear(difference, name, options)
    if solved is not None and not free_vars(solved):
        value = NumericFunction(solved, options.complex_mode).scalar()
        if math.isfinite(value):
            return _Solutions(name, (value,), exact=True)
    solutions, exact, truncated = _numeric_solutions(difference, name, options.complex_mode)
    return _Solutions(name, tuple(solutions), exact=exact, truncated=truncated)


@lru_cache(maxsize=256)
def _numeric_solutions(node: Node, var: str, complex_mode: bool) -> tuple[list[float], bool, bool]:
    """Real solutions of ``node = 0`` in ``var`` (any other variable at zero).

    Returns ``(solutions, exact, truncated)``. Cached: a row solving for an unknown is
    solved while collecting definitions and again when it is classified.
    """
    fn = NumericFunction(node, complex_mode)
    others = free_vars(node) - {var}

    def at(t: np.ndarray) -> np.ndarray:
        return fn(**{var: t}, **{name: np.zeros_like(t) for name in others})

    solutions, truncated = solve_equation(at)
    return solutions, False, truncated


def _solutions_text(var: str, solutions: Sequence[float], exact: bool, truncated: bool) -> str:
    """``x ≈ -8.69, 7.69`` for an equation in one variable."""
    if not solutions:
        return "No roots found in numerical search [-1e8, 1e8]; roots may be missed"
    nearest = sorted(sorted(solutions, key=abs)[:MAX_SHOWN_VALUES])  # periodic: near 0
    shown = [format_value(s, 6) for s in nearest]
    if truncated or len(solutions) > MAX_SHOWN_VALUES:
        shown.append("…")
    text = f"{var} {'=' if exact else '≈'} {', '.join(shown)}"
    return text if exact else text + "  (numerical search [-1e8, 1e8]; may miss roots)"


def _inequality_holds(op: str, v: float) -> bool:
    """``v <op> 0`` for the difference of two constants (with rounding slack for ≤, ≥)."""
    slack = 1e-9 * (1 + abs(v))
    return {"<": v < 0, "<=": v <= slack, ">": v > 0}.get(op, v >= -slack)


def _boundary_plot(left: Node, right: Node, difference: Node, options: CompileOptions) -> Plot:
    """The curve where ``difference`` (= left - right) is zero, as an equation draws it."""
    for dep, other in ((left, right), (right, left)):
        if isinstance(dep, Var) and dep.name in PLOT_VARIABLES:
            indep = "x" if dep.name == "y" else "y"
            if free_vars(other) <= {indep}:
                return ExplicitPlot(other, indep, options.complex_mode)
    used = free_vars(difference)
    for var, free in (("y", "x"), ("x", "y")):
        if var in used:
            solved = _solve_linear(difference, var, options)
            if solved is not None:
                return ExplicitPlot(solved, free, options.complex_mode)
    return ImplicitPlot(difference, options.complex_mode)


def _classify_inequality(stmt: Relation, resolver: _Resolver, options: CompileOptions) -> RowResult:
    """``y < x^2``, ``x + y >= 1``, ``x > 2``: a shaded region; with no variables, true/false."""
    left, right, unknowns = _equation_sides(stmt, resolver)
    _reject_interval(left, stmt.left.span)
    _reject_interval(right, stmt.right.span)
    span = stmt.span
    if unknowns:
        name = next(iter(unknowns))
        raise ResolveError(f"Unknown name '{name}'", unknowns[name])
    if options.complex_mode:
        raise ResolveError("Inequalities are not available in complex mode", span)
    difference: Node = BinOp("-", left, right, span)
    used = free_vars(difference)
    if not used:
        values = _list_values(difference, False)
        if values is None:
            values = np.array([NumericFunction(difference).scalar()])
        words = ["true" if _inequality_holds(stmt.op, float(v)) else "false" for v in values]
        return RowResult(message=words[0] if len(words) == 1 else "list[" + ", ".join(words) + "]")
    polar = bool(used & {THETA, RADIUS})
    if polar:
        if used & set(PLOT_VARIABLES):
            raise ResolveError("Use either x and y or r and θ in one inequality", span)
        difference = _polar_to_cartesian(difference, options)
    variants = _curves(_expand(difference), span)
    if not variants:
        return RowResult(message="Empty list: nothing to plot")
    single = not polar and len(variants) == 1 and variants[0] == difference
    regions = tuple(
        RegionPlot(
            v,
            stmt.op,
            _boundary_plot(left, right, v, options)
            if single
            else _boundary_plot(v, Num(0.0), v, options),
        )
        for v in variants
    )
    return RowResult(plot=regions[0] if len(regions) == 1 else PlotGroup(regions))


def _classify_equation(stmt: Relation, resolver: _Resolver, options: CompileOptions) -> RowResult:
    left, right, unknowns = _equation_sides(stmt, resolver)
    span = stmt.span
    if unknowns:
        _reject_interval(left, stmt.left.span)
        _reject_interval(right, stmt.right.span)
        equation: Node = BinOp("-", left, right, span)
        found = _unknown_solutions(equation, unknowns, options, span, resolver.exact_solutions)
        return RowResult(
            message=found.message(),
            symbolic=SymbolicTarget(equation, found.name, options),
            solved=(found.name, found.values),
        )
    lvars, rvars = free_vars(left), free_vars(right)
    for dep, other, other_vars in ((left, right, rvars), (right, left, lvars)):
        if isinstance(dep, Var) and dep.name in PLOT_VARIABLES:
            indep = "x" if dep.name == "y" else "y"
            if other_vars <= {indep}:
                return _explicit(other, indep, options, span)
        if dep == Var(RADIUS) and other_vars <= {THETA}:
            _reject_interval(other, span)
            plots: list[Plot] = [
                PolarPlot(v, options.degrees, options.complex_mode)
                for v in _curves(_expand(other), span)
            ]
            return RowResult(plot=_group(plots) if plots else None)
    _reject_interval(left, stmt.left.span)
    _reject_interval(right, stmt.right.span)
    used = lvars | rvars
    difference: Node = BinOp("-", left, right, span)
    if not used:
        return _equation_value(difference, options)
    if used & {THETA, RADIUS}:
        if used & set(PLOT_VARIABLES):
            raise ResolveError("Use either x and y or r and θ in one equation", span)
        difference = _polar_to_cartesian(difference, options)
        used = free_vars(difference)
    variants = _curves(_expand(difference), span)
    if not variants:
        return RowResult(message="Empty list: nothing to plot")
    if len(variants) > 1:
        return RowResult(
            plot=PlotGroup(tuple(ImplicitPlot(v, options.complex_mode) for v in variants))
        )
    difference = variants[0]
    for var, free in (("y", "x"), ("x", "y")):
        if var in used:
            solved = _solve_linear(difference, var, options)
            if solved is not None:
                message = ""
                if not free_vars(solved):  # "2x = 3": one solution
                    value = NumericFunction(solved, options.complex_mode).scalar()
                    message = f"{var} = {format_value(value, 6)}"
                return RowResult(
                    plot=ExplicitPlot(solved, free, options.complex_mode),
                    message=message,
                    symbolic=SymbolicTarget(solved, free, options),
                )
    plot = ImplicitPlot(difference, options.complex_mode)
    if len(used) == 1:
        (var,) = used
        solutions, exact, truncated = _numeric_solutions(difference, var, options.complex_mode)
        return RowResult(
            plot=plot,
            message=_solutions_text(var, solutions, exact, truncated),
            symbolic=SymbolicTarget(difference, var, options, equation=True),
        )
    return RowResult(
        plot=plot,
        message="Numerical contour: small features and unrecognized tangent zeros may be missed",
    )


def _offer_solving(
    failed: RowResult, stmt: Statement, resolver: _Resolver, options: CompileOptions
) -> RowResult:
    """``failed`` plus the undefined names it could be solved for, if those are its only
    problem: ``y = (r sin(k) - w)/(r cos(k) - q)(x - q) + w`` can be rearranged for k,
    q, r or w. An equation is rearranged as written; an expression in x as ``y = …``.
    Defined constants stay names in the equation (``k`` with a slider is not replaced
    by its value), so the rearranged formula keeps them; functions are still expanded.
    """
    resolver = _Resolver(
        resolver.definitions,
        resolver.broken,
        options,
        resolver.exact_solutions,
        keep_constants=True,
        cas_results=resolver.cas_results,
    )
    try:
        if isinstance(stmt, Relation) and stmt.op == "=":
            left, right, unknowns = _equation_sides(stmt, resolver)
            difference: Node = BinOp("-", left, right, stmt.span)
        elif isinstance(stmt, Relation | ComparisonChain | TupleExpr | ListExpr | IntervalExpr):
            return failed
        else:
            unknowns = {}
            resolver.unknowns = unknowns
            try:
                body = resolver.resolve_root(stmt)
            finally:
                resolver.unknowns = None
            if "x" not in free_vars(body):
                return failed
            difference = BinOp("-", Var("y"), body, stmt.span)
    except (MathError, RecursionError):
        return failed
    if not unknowns or contains(difference, ListExpr):
        return failed
    names = tuple(sorted(unknowns, key=lambda n: unknowns[n]))
    return RowResult(
        message=failed.message,
        is_error=True,
        error_span=failed.error_span,
        solve_for=names,
        rearrange=SymbolicTarget(difference, names[0], options),
    )


def _undefined_names(
    stmt: Statement, defn: _Definition | None, resolver: _Resolver
) -> tuple[str, ...]:
    """Names the row uses that nothing defines, in order of appearance.

    ``resolver`` should be a fresh one: resolving with unknowns as variables would leave
    those variables in its cache of constants.
    """
    unknowns: dict[str, Span] = {}

    def visit(node: Node | Statement) -> None:
        if isinstance(node, ComparisonChain):
            for comparison in node.comparisons:
                visit(comparison)
        elif isinstance(node, Relation):
            visit(node.left)
            visit(node.right)
        elif isinstance(node, TupleExpr | ListExpr):
            for item in node.items:
                visit(item)
        elif isinstance(node, IntervalExpr):
            visit(node.lo)
            visit(node.hi)
        else:
            with contextlib.suppress(MathError, RecursionError):
                resolver.resolve(node, {})

    resolver.unknowns = unknowns
    try:
        if defn is not None:
            locals_ = {p: Var(p) for p in defn.params or ()}
            with resolver.own(defn.name):
                resolver.resolve(defn.body, locals_)
        else:
            visit(stmt)
    except (MathError, RecursionError):
        pass
    finally:
        resolver.unknowns = None
    return tuple(sorted(unknowns, key=lambda n: unknowns[n]))


# -- tables ------------------------------------------------------------------------------


def _is_name(text: str) -> bool:
    try:
        tokens = tokenize(text)
    except MathError:
        return False
    return len(tokens) == 2 and tokens[0].kind is Kind.NAME and tokens[0].text == text


def _table_definitions(
    table: TableInput, options: CompileOptions
) -> tuple[list[_Definition], dict[tuple[int, int], str], list[Node]]:
    """Parse a table: a list definition per named column, cell errors, column lists."""
    definitions = []
    errors: dict[tuple[int, int], str] = {}
    columns: list[Node] = []
    data_cells = [list(c.cells) for c in table.columns if split_column_header(c.name)[1] is None]
    for texts in data_cells:
        while texts and not texts[-1].strip():
            texts.pop()
    row_count = max((len(texts) for texts in data_cells), default=0)
    for c, column in enumerate(table.columns):
        name, formula = split_column_header(column.name)
        if formula is not None:
            try:
                body = parse(formula) if formula else None
            except ParseError as exc:
                raise ResolveError(f"Column {c + 1}: {exc.message}") from None
            if body is None or isinstance(
                body, Relation | ComparisonChain | TupleExpr | IntervalExpr
            ):
                raise ResolveError(f"Column {c + 1}: enter a formula after '='")
            columns.append(body)
            if name not in PLOT_VARIABLES:
                definitions.append(_Definition(name, None, body, NO_SPAN))
            continue
        cells: list[Node] = []
        texts = list(column.cells[:row_count])
        texts.extend("" for _ in range(row_count - len(texts)))
        for r, text in enumerate(texts):
            if not text.strip():
                cells.append(Num(math.nan))
                continue
            try:
                stmt = parse(text)
            except ParseError as exc:
                errors[(r, c)] = exc.message
                cells.append(Num(math.nan))
                continue
            if isinstance(stmt, Relation | ComparisonChain | TupleExpr | IntervalExpr):
                errors[(r, c)] = "A cell holds a number or an expression"
                cells.append(Num(math.nan))
                continue
            cells.append(stmt)
        lst = ListExpr(tuple(cells))
        columns.append(lst)
        if name and name not in PLOT_VARIABLES:
            definitions.append(_Definition(name, None, lst, NO_SPAN))
    return definitions, errors, columns


def _classify_table(
    table: TableInput, columns: list[Node], resolver: _Resolver, errors: dict[tuple[int, int], str]
) -> RowResult:
    formulas = {}
    for c, column in enumerate(table.columns):
        name, formula = split_column_header(column.name)
        if name and not _is_name(name):
            raise ResolveError(f"Column {c + 1}: '{name}' is not a valid name, like x_1")
        if formula is not None:
            formulas[c] = name
    if len(columns) < 2:
        return RowResult(message="A table needs an x column and a y column")
    values: list[np.ndarray] = []
    for c, cells in enumerate(columns):
        if c in formulas:
            values.append(np.empty(0))  # computed below, once the data columns are known
            continue
        assert isinstance(cells, ListExpr)
        row_values = []
        for r, cell in enumerate(cells.items):
            if (r, c) in errors:
                row_values.append(math.nan)
                continue
            try:
                node = resolver.resolve_root(cell)
                if free_vars(node) or contains(node, ListExpr):
                    raise ResolveError("A cell must be a single number")
                row_values.append(NumericFunction(node).scalar())
            except MathError as exc:
                errors[(r, c)] = exc.message
                row_values.append(math.nan)
        values.append(np.array(row_values, dtype=np.float64))
    computed: dict[tuple[int, int], str] = {}
    data_rows = max((len(v) for c, v in enumerate(values) if c not in formulas), default=0)
    for c, name in formulas.items():
        column_values = _formula_column(name, columns[c], data_rows, resolver, c)
        values[c] = column_values
        for r, v in enumerate(column_values.tolist()):
            computed[(r, c)] = "" if math.isnan(v) else format_value(v)
    plots: list[Plot] = []
    xs = values[0]
    for ys in values[1:]:
        pairs = tuple(zip(xs.tolist(), ys.tolist(), strict=True))
        plots.append(PointSetPlot(pairs, table.markers, table.lines))
    message = ""
    if errors:
        (r, c), text = min(errors.items())
        message = f"Row {r + 1}, column {c + 1}: {text}"
    return RowResult(
        plot=_group(plots),
        message=message,
        is_error=bool(errors),
        cell_errors=dict(errors),
        table_values=computed,
    )


def _formula_column(
    name: str, formula: Node, rows: int, resolver: _Resolver, column: int
) -> np.ndarray:
    """The values of a formula column: the formula over the other columns' lists, so it
    is evaluated row by row (a formula without columns repeats for every row)."""
    try:
        with resolver.own(name) if name in resolver.definitions else contextlib.nullcontext():
            node = resolver.resolve_value(formula)
        if isinstance(node, TupleExpr | IntervalExpr) or free_vars(node):
            raise ResolveError("A formula can only use other columns and defined names")
        found = _list_values(node, False)
        if found is None:
            return np.full(rows, NumericFunction(node).scalar(), dtype=np.float64)
        return np.asarray(found, dtype=np.float64)
    except MathError as exc:
        raise ResolveError(f"Column {column + 1}: {exc.message}") from None


# -- documents ---------------------------------------------------------------------------


def compile_document(
    rows: Sequence[tuple[K, str]],
    options: CompileOptions | None = None,
    tables: Sequence[tuple[K, TableInput]] = (),
    exact_solutions: Mapping[SymbolicTarget, ExactRoots] | None = None,
    cas_results: CasResults | None = None,
) -> dict[K, RowResult]:
    """Compile every row and table; the result maps each key to its :class:`RowResult`.

    ``exact_solutions`` holds exact roots (from SymPy) for equations solved for an
    unknown name, keyed by the row's ``symbolic`` target. These update feedback only;
    equations missing from it are solved numerically. Computing them is
    left to the caller because SymPy is slow and can't be interrupted.

    ``cas_results`` likewise holds computer-algebra results (``diff``, ``factor``…); a
    row needing one that is missing lists it in :attr:`RowResult.pending`.
    """
    return compile_with_values(rows, (), options, tables, exact_solutions, cas_results)[0]


def compile_with_values(
    rows: Sequence[tuple[K, str]],
    values: Sequence[tuple[V, str, K | None]],
    options: CompileOptions | None = None,
    tables: Sequence[tuple[K, TableInput]] = (),
    exact_solutions: Mapping[SymbolicTarget, ExactRoots] | None = None,
    cas_results: CasResults | None = None,
) -> tuple[dict[K, RowResult], dict[V, float | MathError]]:
    """:func:`compile_document`, plus the real value of each extra expression text.

    ``values`` are ``(key, text, row)`` triples for expressions that are not rows of the
    document (a slider's bounds, ``2π`` or ``a + 1``): each may use the document's
    definitions except the name defined by ``row`` (so a slider's range can't depend on
    its own value). Each maps to its value, or the error that prevented one.
    """
    options = options or CompileOptions()
    parsed: dict[K, Statement | MathError | None] = {}
    for key, text in rows:
        if not text.strip():
            parsed[key] = None
            continue
        try:
            parsed[key] = parse(text)
        except MathError as exc:
            parsed[key] = exc
        except RecursionError:
            parsed[key] = MathError("Expression is nested too deeply")

    # Collect definitions; duplicates and invalid names become row errors.
    definitions: dict[str, _Definition] = {}
    owners: dict[str, list[K]] = {}
    row_defs: dict[K, _Definition] = {}
    results: dict[K, RowResult] = {}
    for key, stmt in parsed.items():
        if isinstance(stmt, MathError) or stmt is None:
            continue
        defn = _definition_of(stmt)
        if defn is None:
            continue
        try:
            _check_definition(defn, options)
        except ResolveError as exc:
            results[key] = RowResult.error(exc)
            continue
        owners.setdefault(defn.name, []).append(key)
        definitions[defn.name] = defn
        row_defs[key] = defn
    row_names = {key: defn.name for key, defn in row_defs.items()}

    table_data: dict[K, tuple[TableInput, list[Node], dict[tuple[int, int], str]]] = {}
    table_defs: dict[K, list[_Definition]] = {}
    for key, table in tables:
        try:
            defs, errors, columns = _table_definitions(table, options)
        except MathError as exc:  # a malformed column formula
            results[key] = RowResult.error(exc)
            continue
        table_data[key] = (table, columns, errors)
        valid = []
        for defn in defs:
            try:
                _check_definition(defn, options)
            except ResolveError as exc:
                results[key] = RowResult.error(exc)
                break
            valid.append(defn)
        else:
            table_defs[key] = valid
            for defn in valid:
                owners.setdefault(defn.name, []).append(key)
                definitions[defn.name] = defn

    broken: dict[str, str] = {}
    for name, keys in owners.items():
        if len(keys) > 1:
            broken[name] = "is defined more than once"
            for key in keys:
                if key in row_defs:
                    defn = row_defs.pop(key)
                    span: Span | None = defn.span
                else:
                    span = None
                results[key] = RowResult.error(
                    ResolveError(f"'{name}' is defined more than once", span)
                )

    # Equations report solutions; only explicit definitions introduce names.
    exact = exact_solutions or {}
    cas = cas_results or {}

    # Classify definition rows first so that every definition whose row fails is known
    # as broken; other rows then report "'f' has an error" instead of re-expanding it.
    # The snapshot keeps each definition row's diagnosis independent of row order.
    resolver = _Resolver(definitions, dict(broken), options, exact, cas_results=cas)
    for key, defn in row_defs.items():
        try:
            results[key] = _classify(cast(Statement, parsed[key]), resolver, defn, options)
        except PendingError as exc:
            results[key] = RowResult.error(exc)
            broken.setdefault(defn.name, "is still being computed")
        except ResolveError as exc:
            fresh = _Resolver(definitions, broken, options, exact, cas_results=cas)
            undefined = _undefined_names(cast(Statement, parsed[key]), defn, fresh)
            results[key] = dataclasses.replace(RowResult.error(exc), undefined=undefined)
            broken.setdefault(defn.name, "has an error")
        except RecursionError:
            results[key] = RowResult.error(MathError("Expression is nested too deeply"))
            broken.setdefault(defn.name, "has an error")
    resolver = _Resolver(definitions, broken, options, exact, cas_results=cas)

    for key, (table, columns, errors) in table_data.items():
        if key in results:
            continue
        try:
            results[key] = _classify_table(table, columns, resolver, dict(errors))
        except MathError as exc:
            results[key] = RowResult.error(exc)

    for key, _text in rows:
        if key in results:
            continue
        stmt = parsed[key]
        if stmt is None:
            results[key] = EMPTY
        elif isinstance(stmt, MathError):
            results[key] = RowResult.error(stmt)
        else:
            row_resolver = resolver
            try:
                results[key] = _classify(stmt, row_resolver, None, options)
            except PendingError as exc:
                results[key] = RowResult.error(exc)
            except MathError as exc:
                failed = _offer_solving(RowResult.error(exc), stmt, row_resolver, options)
                fresh = _Resolver(row_resolver.definitions, broken, options, exact, cas_results=cas)
                undefined = _undefined_names(stmt, None, fresh)
                results[key] = dataclasses.replace(failed, undefined=undefined)
            except RecursionError:
                results[key] = RowResult.error(MathError("Expression is nested too deeply"))
    order = [key for key, _ in rows] + [key for key, _ in tables]
    evaluated: dict[V, float | MathError] = {}
    for value_key, text, row in values:
        excluded = row_names.get(row) if row is not None else None
        try:
            value = _real_value(text, excluded, definitions, broken, options, exact, cas)
            evaluated[value_key] = value
        except MathError as exc:
            evaluated[value_key] = exc
        except RecursionError:
            evaluated[value_key] = MathError("Expression is nested too deeply")
    return {key: results[key] for key in order}, evaluated


def _real_value(
    text: str,
    own: str | None,
    definitions: Mapping[str, _Definition],
    broken: Mapping[str, str],
    options: CompileOptions,
    exact: Mapping[SymbolicTarget, ExactRoots],
    cas: CasResults,
) -> float:
    """The real number an expression text evaluates to, using the given definitions."""
    if not text.strip():
        raise MathError("Enter a number or an expression")
    stmt = parse(text)
    if isinstance(stmt, Relation | ComparisonChain):
        raise ResolveError("Enter a value, not an equation", stmt.span)
    if own is not None:
        if _mentions(stmt, frozenset({own})):
            raise ResolveError(f"Can't depend on {own} itself", stmt.span)
        definitions = {name: d for name, d in definitions.items() if name != own}
    node = _Resolver(definitions, dict(broken), options, exact, cas_results=cas).resolve_root(stmt)
    if free_vars(node):
        raise ResolveError("Must be a number, not depend on a variable", stmt.span)
    if _list_values(node, False) is not None:
        raise ResolveError("Must be a single number, not a list", stmt.span)
    value = NumericFunction(node).scalar()
    if not math.isfinite(value):
        raise ResolveError("Is not a finite real number", stmt.span)
    return value


def compile_expression(text: str, options: CompileOptions | None = None) -> RowResult:
    """Compile a single standalone row (convenience for tests and tools)."""
    return compile_document([(0, text)], options)[0]
