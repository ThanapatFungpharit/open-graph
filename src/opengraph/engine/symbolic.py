"""Symbolic operations backed by SymPy.

The resolved AST is converted to SymPy node by node through the builtin whitelist —
user text is never passed to ``sympify``. Results are printed back into this
application's own syntax and verified by re-parsing, so whatever is inserted into the
document is guaranteed to be valid input.

The printed text means the same thing under the document's :class:`CompileOptions`:
in degree mode the compiler turns ``sin(a)`` into ``sin(a·π/180)``, so the printer
undoes that conversion (``sin(u)`` prints as ``sin(u·180/π)``, simplified), and in
complex mode the imaginary unit prints as ``i``.

SymPy is a required dependency. It is slow to import, so import this module only where
symbolic operations are offered, not on the plotting path. These functions can be slow
(``simplify``/``solveset``); callers run them off the UI thread.
"""

from __future__ import annotations

import math
import operator
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from functools import cache
from itertools import islice
from typing import Any, NoReturn

import numpy as np
import sympy as sp
from sympy.calculus.util import continuous_domain
from sympy.core.relational import Relational
from sympy.functions.elementary.piecewise import ExprCondPair
from sympy.printing.str import StrPrinter

from opengraph.engine.ast import (
    BinOp,
    Bounded,
    Const,
    FuncCall,
    Guard,
    Node,
    Num,
    Reduce,
    UnaryOp,
    Var,
    children,
    contains,
    domain_sensitive,
    walk,
)
from opengraph.engine.builtins import (
    COMPLEX_PROJECTIONS,
    FUNCTIONS,
    PLOT_VARIABLES,
    SYMBOLIC_CONSTANTS,
)
from opengraph.engine.compiler import (
    CasRequest,
    CompileOptions,
    GeneralSolution,
    SolutionFamily,
    SolutionGenerator,
    SymbolicTarget,
    compile_expression,
)
from opengraph.engine.errors import MathError
from opengraph.engine.evaluator import MAX_TERM_WORK, NumericFunction
from opengraph.engine.parser import parse
from opengraph.engine.source import source as original_source

MAX_SOLUTIONS = 64

# Σ/Π with at least _EXPAND_TERMS terms are kept unevaluated unless a closed form exists
# whose exact numbers fit in _MAX_EXACT_BITS and that has at most _MAX_CLOSED_FORM_OPS
# operations (sum(1/n^2, n, 1, 10000) has an exact value thousands of digits long).
_EXPAND_TERMS = 100
_MAX_EXACT_BITS = 256
_MAX_CLOSED_FORM_OPS = 100

# Numeric fallback for equations with no closed form: roots are searched for in
# [-SCAN_LIMIT, SCAN_LIMIT]. A root that only touches zero (``(x - 1)^2``) is accepted
# when ``|f|`` bottoms out below _TANGENT_TOLERANCE, looking only at dips under
# _TANGENT_LIMIT.
SCAN_LIMIT = 1_000.0
_TANGENT_LIMIT = 1e-3
_SCAN_CHUNK = 4096  # scan points per evaluation when they are evaluated together
_SCAN_WORK = 20_000_000  # most Σ/Π terms × points one numeric scan evaluates
_TANGENT_TOLERANCE = 1e-9

_PLOT_NAMES = frozenset(PLOT_VARIABLES)

_TRIG = ("sin", "cos", "tan", "sec", "csc", "cot")
_INVERSE_TRIG = {"asin": "asin", "acos": "acos", "atan": "atan", "atan2": "atan"}

# SymPy function class -> OpenGraph function name, for plain ``name(args...)`` calls.
_FUNCTION_NAMES = {
    "Abs": "abs",
    "log": "ln",
    "ceiling": "ceil",
    "Min": "min",
    "Max": "max",
    "Mod": "mod",
    "gamma": "Gamma",
    "factorial": "factorial",
    "binomial": "nCr",
    "re": "re",
    "im": "im",
    "arg": "arg",
    "conjugate": "conj",
    "atan2": "atan",  # OpenGraph spells atan2 as atan(y, x)
}

# SymPy functions that OpenGraph spells as the reciprocal of one of its own:
# sech(u) -> 1/cosh(u), and asec(u) -> acos(1/u) (the angle unit applies to the result).
_RECIPROCALS = {"sech": "cosh", "csch": "sinh", "coth": "tanh"}
_INVERSE_RECIPROCALS = {"asech": "acosh", "acsch": "asinh", "acoth": "atanh"}
_INVERSE_RECIPROCAL_TRIG = {"asec": "acos", "acsc": "asin", "acot": "atan"}

# SymPy nodes OpenGraph has no way to write. (A function it has no name for, such as
# ``LambertW``, is rejected by ``_print_Function`` without being listed here.)
_UNSUPPORTED = (
    "Piecewise", "Derivative", "ComplexInfinity", "NaN", "Infinity", "NegativeInfinity",
    "Subs",
)  # fmt: skip

_BINARY_OPERATORS = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
}


class SymbolicError(MathError):
    """A symbolic operation could not produce a usable result."""


# ---------------------------------------------------------------------------
# Printing: SymPy -> OpenGraph source
# ---------------------------------------------------------------------------


def _renamed(name: str) -> Any:
    def print_call(self: Any, expr: Any) -> str:
        return str(self._call(name, *expr.args))

    return print_call


def _reciprocal(name: str) -> Any:
    """Prints ``f(u)`` as ``1/name(u)``."""

    def print_reciprocal(self: Any, expr: Any) -> str:
        return f"(1/{self._call(name, *expr.args)})"

    return print_reciprocal


def _of_reciprocal(name: str, *, degrees: bool) -> Any:
    """Prints ``f(u)`` as ``name(1/u)``; in degree mode the result is converted back
    from OpenGraph's degrees to the radians SymPy means."""

    def print_of_reciprocal(self: Any, expr: Any) -> str:
        text = self._call(name, sp.S.One / expr.args[0])
        return f"({text}*pi/180)" if degrees else text

    return print_of_reciprocal


@cache
def _printer(degrees: bool, complex_mode: bool) -> Any:
    class Printer(StrPrinter):  # type: ignore[misc]
        """Prints SymPy expressions in OpenGraph syntax."""

        _complex_depth: int = 0

        def _call(self, name: str, *args: Any) -> str:
            previous_depth = self._complex_depth
            if name in COMPLEX_PROJECTIONS:
                self._complex_depth += 1
            try:
                return f"{name}({', '.join(self._print(a) for a in args)})"
            finally:
                self._complex_depth = previous_depth

        def _print_Function(self, expr: Any) -> str:
            # Only functions OpenGraph has a name for; anything else (``LambertW``,
            # ``polygamma``, ``besselj``…) would print as a call that can't be compiled.
            if type(expr).__name__ not in FUNCTIONS:
                return self._unsupported(expr)
            return str(super()._print_Function(expr))

        def _print_Exp1(self, expr: Any) -> str:
            return "e"

        def _print_Float(self, expr: Any) -> str:
            return f"{float(expr):.12g}".replace("e+", "e")

        def _print_ImaginaryUnit(self, expr: Any) -> str:
            if complex_mode or self._complex_depth:
                return "i"
            return self._unsupported(expr)

        def _bounded(self, name: str, expr: Any) -> str:
            (var, lower, upper), *rest = expr.limits
            if rest or lower is None:
                return self._unsupported(expr)
            return self._call(name, expr.function, var, lower, upper)

        def _print_Sum(self, expr: Any) -> str:
            return self._bounded("sum", expr)

        def _print_Product(self, expr: Any) -> str:
            return self._bounded("prod", expr)

        def _print_Integral(self, expr: Any) -> str:
            if any(len(limit) != 3 for limit in expr.limits):
                return self._unsupported(expr)  # an antiderivative that wasn't found
            return self._bounded("int", expr)

        def _print_Heaviside(self, expr: Any) -> str:
            # sign(u) is -1, 0 or 1, so (sign(u) + 1)/2 steps 0 -> 1/2 -> 1: SymPy's
            # default Heaviside, including its value 1/2 at 0.
            value_at_zero = expr.args[1] if len(expr.args) > 1 else sp.S.Half
            if value_at_zero != sp.S.Half:
                return self._unsupported(expr)
            return f"((sign({self._print(expr.args[0])})+1)/2)"

        def _print_frac(self, expr: Any) -> str:
            (arg,) = expr.args
            text = self._print(arg)
            return f"(({text})-floor({text}))"

        # erf, erfc, erfi and the Fresnel integrals have no OpenGraph name, but each is a
        # definite integral from 0, which OpenGraph does have.
        def _integral_from_zero(self, integrand: str, variable: str, upper: Any) -> str:
            return f"int({integrand}, {variable}, 0, {self._print(upper)})"

        def _fresh_variable(self, expr: Any) -> str:
            used = {symbol.name for symbol in expr.free_symbols}
            return next(name for name in "tuvwsk" if name not in used)

        def _print_erf(self, expr: Any) -> str:
            (arg,) = expr.args
            t = self._fresh_variable(arg)
            return f"(2/sqrt(pi)*{self._integral_from_zero(f'exp(-{t}^2)', t, arg)})"

        def _print_erfc(self, expr: Any) -> str:
            return f"(1-{self._print_erf(expr)})"

        def _print_erfi(self, expr: Any) -> str:
            (arg,) = expr.args
            t = self._fresh_variable(arg)
            return f"(2/sqrt(pi)*{self._integral_from_zero(f'exp({t}^2)', t, arg)})"

        def _fresnel(self, trig: str, expr: Any) -> str:
            (arg,) = expr.args
            t = self._fresh_variable(arg)
            angle = f"(pi*{t}^2/2)" + ("*180/pi" if degrees else "")
            return self._integral_from_zero(f"{trig}({angle})", t, arg)

        def _print_fresnels(self, expr: Any) -> str:
            return self._fresnel("sin", expr)

        def _print_fresnelc(self, expr: Any) -> str:
            return self._fresnel("cos", expr)

        def _unsupported(self, expr: Any) -> str:
            raise SymbolicError(
                f"The result uses {type(expr).__name__}, which OpenGraph can't express"
            )

    for sympy_name, name in _FUNCTION_NAMES.items():
        setattr(Printer, f"_print_{sympy_name}", _renamed(name))
    for sympy_name, name in _RECIPROCALS.items():
        setattr(Printer, f"_print_{sympy_name}", _reciprocal(name))
    for sympy_name, name in _INVERSE_RECIPROCALS.items():
        setattr(Printer, f"_print_{sympy_name}", _of_reciprocal(name, degrees=False))
    for sympy_name, name in _INVERSE_RECIPROCAL_TRIG.items():
        setattr(Printer, f"_print_{sympy_name}", _of_reciprocal(name, degrees=degrees))
    for sympy_name in _UNSUPPORTED:
        setattr(Printer, f"_print_{sympy_name}", Printer._unsupported)

    if degrees:
        # The compiler converts degrees to radians around trig calls; undo it in print.
        def forward(name: str) -> Any:
            def print_trig(self: Any, expr: Any) -> str:
                return str(self._call(name, _simplify(expr.args[0] * 180 / sp.pi)))

            return print_trig

        def inverse(name: str) -> Any:
            def print_inverse(self: Any, expr: Any) -> str:
                return f"({self._call(name, *expr.args)}*pi/180)"

            return print_inverse

        for name in _TRIG:
            setattr(Printer, f"_print_{name}", forward(name))
        for sympy_name, name in _INVERSE_TRIG.items():
            setattr(Printer, f"_print_{sympy_name}", inverse(name))
    return Printer()


def _print(expr: Any, options: CompileOptions) -> str:
    """``expr`` in OpenGraph syntax (not yet verified)."""
    return str(_printer(options.degrees, options.complex_mode).doprint(expr)).replace("**", "^")


def _check_parses(source: str) -> None:
    try:
        parse(source)
    except MathError as exc:
        raise SymbolicError(f"Could not express the result ({exc.message})") from None


def to_source(expr: Any, options: CompileOptions | None = None) -> str:
    """Print a SymPy expression as OpenGraph input text, verified by re-parsing."""
    options = options or CompileOptions()
    text = _print(expr, options)
    names = {s.name for s in expr.free_symbols}
    source = f"x = {text}" if "y" in names else text
    if names - _PLOT_NAMES:
        # User constants kept as names compile only beside their definitions: check the
        # syntax alone.
        _check_parses(source)
        return text
    result = compile_expression(source, options)
    if result.is_error:
        raise SymbolicError(f"Could not express the result ({result.message})")
    return text


# ---------------------------------------------------------------------------
# Conversion: AST -> SymPy
# ---------------------------------------------------------------------------


def _symbol(name: str, **assumptions: bool) -> Any:
    # Bound variables are named "k@0" in resolved trees; SymPy sees plain "k".
    return sp.Symbol(name.split("@")[0], **assumptions)


def _variable(target: SymbolicTarget) -> Any:
    return _symbol(target.variable, real=True)


def _odd_root(base: Any, exponent: Any) -> Any | None:
    """``base ** exponent`` with the evaluator's real odd roots (``(-8)^(1/3) = -2``),
    or None when the exponent isn't a non-integer rational with an odd denominator."""
    if not isinstance(exponent, sp.Rational) or exponent.is_integer or exponent.q % 2 == 0:
        return None
    magnitude = sp.Abs(base) ** exponent
    return sp.sign(base) * magnitude if exponent.p % 2 else magnitude


def _evaluate_bounded(operation: Any, body: Any, symbol: Any, lower: Any, upper: Any) -> Any:
    """``operation`` (∫, Σ or Π) over ``lower..upper``, evaluated where that stays small.

    A Σ or Π over many terms is never written out term by term (``Σ 1/n^x`` to 10000
    would be 10000 terms that every later step crawls through). It takes a closed form
    found with a symbolic upper bound, when that form is small once the bound is put
    back; otherwise it stays an unevaluated sum, which SymPy still differentiates term
    by term and the plotter evaluates numerically.
    """
    expr = operation(body, (symbol, lower, upper))
    count = upper - lower
    if operation is sp.Integral or not (count.is_number and count >= _EXPAND_TERMS):
        return expr.doit()
    end = sp.Dummy("end", integer=True)
    closed = sp.gammasimp(operation(body, (symbol, lower, end)).doit())
    if closed.has(sp.Sum, sp.Product) or not _algebraic_in(closed, end):
        return expr
    value = closed.subs(end, upper)
    return value if _manageable(value) else expr


def _algebraic_in(expr: Any, end: Any) -> bool:
    """Whether ``expr`` uses ``end`` only in sums, products and powers with a symbolic
    base: putting a large number in for it then can't make an enormous exact number,
    as ``harmonic(end, 2)``, ``factorial(end)`` or ``2^end`` would."""
    for sub in sp.preorder_traversal(expr):
        if not sub.has(end) or sub == end:
            continue
        if isinstance(sub, sp.Pow):
            if sub.exp.has(end) and (sub.base.is_number or sub.base.has(end)):
                return False
        elif not isinstance(sub, (sp.Add, sp.Mul, sp.Piecewise, ExprCondPair, Relational)):
            return False
    return True


def _simplify(expr: Any) -> Any:
    """``sp.simplify`` that leaves the terms of unevaluated Σ/Π as they are.

    Simplified on its own, a term can turn into a form SymPy takes as 0 at some index
    but the plotter can't evaluate there: ``n x^(n-1)/n!`` becomes ``x^(n-1)/(n-1)!``,
    which is ``x^(-1)/(-1)!`` at n = 0.
    """
    bounded = {s: sp.Dummy() for s in expr.atoms(sp.Sum, sp.Product)}
    if not bounded:
        return sp.simplify(expr)
    simplified = sp.simplify(expr.xreplace(bounded))
    return simplified.xreplace({dummy: s for s, dummy in bounded.items()})


def _manageable(expr: Any) -> bool:
    """Small enough to be worth writing out: no huge exact numbers or sprawling terms."""
    if any(
        max(abs(r.p).bit_length(), r.q.bit_length()) > _MAX_EXACT_BITS
        for r in expr.atoms(sp.Rational)
    ):
        return False
    return bool(sp.count_ops(expr) <= _MAX_CLOSED_FORM_OPS)


def to_sympy(node: Node, real_roots: bool = False) -> Any:
    """``node`` as a SymPy expression; ``real_roots`` evaluates ``a^(p/q)`` with odd
    ``q`` as the plotter does (real root of a negative base) instead of SymPy's
    principal complex root."""

    def convert(current: Node) -> Any:
        match current:
            case Guard(value=value) | Bounded(value=value):
                return convert(value)
            case Num(value=value):
                if not math.isfinite(value):
                    raise SymbolicError("Can't convert a non-finite number")
                if value == int(value) and abs(value) < 1e15:
                    return sp.Integer(int(value))
                return sp.Rational(repr(value))  # 0.1 -> 1/10, exactly as typed
            case Const(name=name):
                return SYMBOLIC_CONSTANTS[name](sp)
            case Var(name=name):
                return _symbol(name, real=True)
            case UnaryOp(operand=operand):
                return -convert(operand)
            case BinOp(op=op, left=left, right=right):
                a, b = convert(left), convert(right)
                apply = _BINARY_OPERATORS.get(op)
                if apply is not None:
                    return apply(a, b)
                special = _odd_root(a, b) if real_roots else None
                return a**b if special is None else special
            case FuncCall(name=name, args=args):
                return FUNCTIONS[name].symbolic(sp, *(convert(a) for a in args))
            case Reduce(kind=kind, var=var, body=body, lower=lower, upper=upper):
                real_symbol = _symbol(var, real=True)
                if kind == "int":
                    symbol, operation = real_symbol, sp.Integral
                else:
                    symbol = _symbol(var, integer=True)
                    operation = sp.Sum if kind == "sum" else sp.Product
                inner = convert(body).subs(real_symbol, symbol)
                return _evaluate_bounded(operation, inner, symbol, convert(lower), convert(upper))
        raise SymbolicError(f"Can't convert {type(current).__name__} to a symbolic expression")

    return convert(node)


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------


def _named(target: SymbolicTarget) -> Any:
    """The curve with user constants kept as symbols.

    ``target.node`` has the constants' values inlined, so it is never used instead:
    without a named form there is no result rather than one for particular values.
    """
    if target.named is None:
        raise SymbolicError("This row can't be transformed without substituting values")
    expr = to_sympy(target.named)
    if target.named_unsolved is None:
        return expr
    solutions = sp.solve(expr, _symbol(target.named_unsolved, real=True))
    # Not exactly one solution: e.g. the coefficient is zero for some constant values.
    if len(solutions) != 1:
        raise SymbolicError(f"This equation has no single solution for {target.named_unsolved}")
    return solutions[0]


def _without_substitution(target: SymbolicTarget, operation: Callable[[Any], Any]) -> str:
    """``operation`` applied to the curve with user constants left as names.

    Purely algebraic: variables are never given values, and when the symbolic result
    can't be written the operation fails instead of falling back to the constants'
    current values.
    """
    text = to_source(_simplify(operation(_named(target))), target.options)
    if target.named is not None and domain_sensitive(target.named):
        text = f"guard({text}, {original_source(target.named, degrees=target.options.degrees)})"
    return text


def derivative(target: SymbolicTarget) -> str:
    """The derivative, keeping user constants as names: ``y = a x^2`` -> ``2*a*x``."""
    return _without_substitution(target, lambda expr: _derivative(expr, _variable(target)))


def integral(target: SymbolicTarget) -> str:
    """An antiderivative at current parameter values, with the original domain retained."""
    text = to_source(
        _antiderivative(to_sympy(target.node), _variable(target), target.options.complex_mode),
        target.options,
    )
    if domain_sensitive(target.node):
        text = f"guard({text}, {original_source(target.node, degrees=target.options.degrees)})"
    return text


def _antiderivative(expr: Any, variable: Any, complex_mode: bool = False) -> Any:
    result = sp.integrate(expr, variable)
    if result.has(sp.Piecewise):
        raise SymbolicError(
            "The antiderivative needs parameter conditions; define the parameters first"
        )
    if result.has(sp.Integral) or result.has(sp.Piecewise):
        raise SymbolicError("No closed-form antiderivative found")
    if not complex_mode:
        result = result.replace(
            lambda item: (
                item.func == sp.log
                and item.args[0].is_real is True
                and variable in item.args[0].free_symbols
            ),
            lambda item: sp.log(sp.Abs(item.args[0])),
        )
    return result


def _derivative(expr: Any, variable: Any, order: int = 1) -> Any:
    result = sp.diff(expr, variable, order)
    if result.has(sp.Derivative):
        raise SymbolicError("This expression can't be differentiated symbolically")
    return result


_UNWRITABLE = (sp.Limit, sp.AccumBounds, sp.Order, sp.Subs, sp.Derivative)
_INFINITIES = (sp.oo, -sp.oo, sp.zoo, sp.nan)


def _limit(expr: Any, variable: Any, point: Any, direction: str) -> Any:
    """The limit of ``expr`` as ``variable`` -> ``point``; ``direction`` is SymPy's
    ``"+-"`` (two-sided), ``"+"`` or ``"-"``. Raises unless it is a finite expression."""
    try:
        result = sp.limit(expr, variable, point, dir=direction)
    except ValueError:  # SymPy: the left- and right-hand limits differ
        raise SymbolicError("The limit doesn't exist: the one-sided limits differ") from None
    except (NotImplementedError, sp.PoleError):
        raise SymbolicError("No limit found") from None
    if result.has(sp.AccumBounds):
        raise SymbolicError("The limit doesn't exist: the function oscillates")
    if result.has(*_INFINITIES):
        raise SymbolicError("The limit is infinite or doesn't exist")
    if result.has(*_UNWRITABLE) or variable in result.free_symbols:
        raise SymbolicError("No limit found")
    return result


def _taylor(expr: Any, variable: Any, point: Any, degree: int) -> Any:
    """The Taylor polynomial of ``expr`` about ``point`` up to degree ``degree``,
    without the O() term. Raises when ``expr`` has none there (a pole, a fractional
    power, or a corner like ``abs(x)`` at 0, where the two sides' series differ)."""
    try:
        right = sp.series(expr, variable, point, degree + 1, dir="+").removeO()
        left = sp.series(expr, variable, point, degree + 1, dir="-").removeO()
    except (NotImplementedError, ValueError, TypeError, sp.PoleError):
        raise SymbolicError("No Taylor polynomial found") from None
    if right.has(*_UNWRITABLE) or right.has(*_INFINITIES):
        raise SymbolicError("No Taylor polynomial found")
    if not right.is_polynomial(variable) or _simplify(right - left) != 0:
        raise SymbolicError("No Taylor polynomial about that point")
    return right


def _coefficient_poles(expr: Any, variable: Any) -> set[Any]:
    """Parameter factors whose zero would make an entire rational term undefined.

    A partial fraction expansion may introduce these while separating poles:
    1/(x^2-a^2) needs coefficients 1/(2*a), invalid when the poles coincide at a=0.
    Factors already present in the input are covered by its original domain guard.
    """
    poles = set()
    for term in sp.Add.make_args(expr):
        for factor in sp.Mul.make_args(sp.factor(sp.denom(term))):
            if variable not in factor.free_symbols and factor.is_zero is not False:
                poles.add(factor.as_base_exp()[0])
    return poles


def _apart(expr: Any, variable: Any) -> Any:
    try:
        result = sp.apart(expr, variable)
    except (sp.PolynomialError, NotImplementedError):
        raise SymbolicError(
            f"Partial fractions need a rational expression in {variable}; "
            "use apart(f, variable) to choose another variable"
        ) from None
    if _coefficient_poles(result, variable) - _coefficient_poles(expr, variable):
        raise SymbolicError(
            "Partial fractions need parameter conditions; define the parameters first"
        )
    return result


def cas(request: CasRequest) -> str:
    """A computer-algebra call as OpenGraph source text; names other than the
    chosen variable are constants.
    """
    expr = to_sympy(request.node)
    variable = _symbol(request.variable, real=True)
    match request.operation:
        case "diff":
            result = _simplify(_derivative(expr, variable, request.order))
        case "integrate":
            result = _simplify(_antiderivative(expr, variable, request.options.complex_mode))
        case "simplify":
            result = _simplify(expr)
        case "expand":
            result = sp.expand(expr)
        case "factor":
            result = sp.factor(expr)
        case "cancel":
            result = sp.cancel(expr)
        case "together":
            result = sp.together(expr)
        case "apart":
            result = _apart(expr, variable)
        case "collect":
            result = sp.collect(expr, variable)
        case "trigsimp":
            result = sp.trigsimp(expr)
        case "limit":
            assert request.point is not None
            result = _simplify(_limit(expr, variable, to_sympy(request.point), request.direction))
        case "taylor":
            assert request.point is not None
            result = _taylor(expr, variable, to_sympy(request.point), request.order)
        case "roots":
            real = expr if request.options.complex_mode else to_sympy(request.node, real_roots=True)
            solutions = _exact_solutions(expr, real, variable)
            if solutions is sp.S.EmptySet:
                return "list[]"
            if not isinstance(solutions, sp.FiniteSet):
                raise SymbolicError(
                    "roots needs a finite exact solution set; use Solve for general solutions"
                )
            values = sorted(
                (
                    s
                    for s in solutions
                    if _is_root(real, variable, s, request.options.complex_mode)
                    and _defined_at(request.node, request.variable, s, request.options.complex_mode)
                ),
                key=lambda value: float(sp.N(value)),
            )
            if len(values) > MAX_SOLUTIONS:
                raise SymbolicError("Too many roots for a finite list")
            return "list[" + ", ".join(to_source(v, request.options) for v in values) + "]"
        case _:
            raise SymbolicError(f"Unknown operation '{request.operation}'")
    return to_source(result, request.options)


def simplify(target: SymbolicTarget) -> str:
    """Simplified without substituting user constants: ``y = a x + a x`` -> ``2*a*x``."""
    return _without_substitution(target, lambda expr: expr)


def _decimal(value: Any) -> Any:
    """``value`` as a 17-digit SymPy number (complex in complex mode's ``i``)."""
    try:
        number = complex(sp.N(value, 20))
    except (TypeError, ValueError):
        raise SymbolicError("No exact form found") from None
    if not (math.isfinite(number.real) and math.isfinite(number.imag)):
        raise SymbolicError("No exact form found")
    real = sp.Float(number.real, 17)
    if abs(number.imag) <= 1e-12 * (1 + abs(number.real)):  # rounding noise, not a part
        return real
    return real + sp.Float(number.imag, 17) * sp.I


def exact_value(target: SymbolicTarget) -> str:
    """The exact value of a closed expression, e.g. ``∫(x^2, 0, 1)`` -> ``1/3``."""
    result = _simplify(to_sympy(target.node))
    if result.free_symbols:
        raise SymbolicError("This expression has no single exact value")
    if result.has(sp.Integral, sp.Sum, sp.Product):
        raise SymbolicError("No exact form found")
    try:
        return to_source(result, target.options)
    except SymbolicError:
        # No exact spelling (a function OpenGraph lacks, say): the decimal still is
        # the value.
        return to_source(_decimal(result), target.options)


@dataclass(frozen=True, slots=True)
class Root:
    exact: str
    value: float


def _is_root(expr: Any, var: Any, value: Any, complex_mode: bool) -> bool:
    """Whether ``expr`` is defined and zero at ``var = value`` as the plotter sees it.

    Every subexpression is checked, not just the whole: SymPy reduces
    ``sqrt(-1)·0`` to 0, but the plotted curve has no point there.
    """
    for part in sp.preorder_traversal(expr):
        if part.is_Atom:
            continue
        try:
            number = complex(sp.N(part.subs(var, value)))
        except (TypeError, ValueError):  # zoo, nan, an unevaluated integral…
            return False
        if not (math.isfinite(number.real) and math.isfinite(number.imag)):
            return False
        if not complex_mode and abs(number.imag) > 1e-12 * (1 + abs(number.real)):
            return False
    try:
        return abs(complex(sp.N(expr.subs(var, value), 30))) < 1e-9
    except (TypeError, ValueError):
        return False


def _defined_at(node: Node, variable: str, value: Any, complex_mode: bool) -> bool:
    """Check the original tree, before SymPy can cancel its excluded points."""
    try:
        result = NumericFunction(node, complex_mode).scalar(**{variable: float(sp.N(value))})
        return math.isfinite(result)
    except (MathError, ValueError, OverflowError):
        return False


def _original_domain(node: Node, variable: Any) -> Any:
    """Intersect domains of original subexpressions, including cancelled denominators."""
    if isinstance(node, Guard):
        return sp.Intersection(
            _original_domain(node.value, variable), _original_domain(node.original, variable)
        )
    if isinstance(node, Bounded):
        interval = node.interval
        coordinate = to_sympy(node.coordinate)
        lo, hi = to_sympy(interval.lo), to_sympy(interval.hi)
        lower = coordinate >= lo if interval.closed_lo else coordinate > lo
        upper = coordinate <= hi if interval.closed_hi else coordinate < hi
        domain = sp.Intersection(
            sp.solveset(lower, variable, domain=sp.S.Reals),
            sp.solveset(upper, variable, domain=sp.S.Reals),
        )
        return sp.Intersection(domain, _original_domain(node.value, variable))
    try:
        domain = continuous_domain(to_sympy(node), variable, sp.S.Reals)
        if isinstance(node, BinOp) and node.op == "/":
            zeros = sp.solveset(to_sympy(node.right), variable, domain=sp.S.Reals)
            domain = domain - zeros
        return sp.Intersection(domain, *(_original_domain(c, variable) for c in children(node)))
    except NotImplementedError:
        raise SymbolicError("Cannot prove the original domain for this solution set") from None


def _is_finite(solutions: Any) -> bool:
    """Whether SymPy returned a fully enumerable finite solution set."""
    return solutions is sp.S.EmptySet or isinstance(solutions, sp.FiniteSet)


def _solve_reals(expr: Any, var: Any) -> Any:
    """``solveset(expr, var)`` over the reals; a ConditionSet when SymPy gives up."""
    try:
        solutions = sp.solveset(expr, var, domain=sp.S.Reals)
    except NotImplementedError:
        return sp.ConditionSet(var, sp.Eq(expr, 0), sp.S.Reals)
    if _is_finite(solutions):
        return solutions
    simplified = _simplify(solutions)  # may collapse a Union/Complement to a finite set
    return simplified if _is_finite(simplified) else solutions


def _raise_unsolved(solutions: Any) -> NoReturn:
    """Raise a useful error describing why the solution set cannot be returned."""
    if isinstance(solutions, sp.ConditionSet):
        raise SymbolicError("No exact solution found")
    if isinstance(solutions, sp.Interval):
        raise SymbolicError(f"The solutions form an interval: {solutions}")
    if isinstance(solutions, sp.Union) and not solutions.is_iterable:
        raise SymbolicError(f"The solutions form an infinite set: {solutions}")
    raise SymbolicError("There are infinitely many solutions")


def _root(solution: Any, options: CompileOptions) -> Root:
    value = float(sp.N(solution))
    try:
        if isinstance(solution, sp.CRootOf):
            raise SymbolicError("A root of a polynomial has no closed form")
        exact = to_source(solution, options)
    except SymbolicError:
        # No exact spelling (e.g. a LambertW root): keep the root as a decimal rather
        # than failing the whole solve.
        exact = to_source(sp.Float(value, 15), options)
    return Root(exact, value)


def _exact_solutions(expr: Any, real: Any, var: Any) -> Any:
    """SymPy's solution set for ``expr`` (and its odd-root variant ``real``), which is
    finite only when SymPy could enumerate it."""
    solutions = _solve_reals(expr, var)
    if real != expr:
        alternative = _solve_reals(real, var)
        if _is_finite(alternative):
            solutions = sp.Union(solutions, alternative) if _is_finite(solutions) else alternative
        elif solutions is sp.S.EmptySet:
            # "No roots" would be a claim about the plotter's expression that we could
            # not establish: SymPy couldn't solve that one.
            solutions = alternative
    return solutions


def _polynomial_roots(expr: Any, var: Any) -> list[Any] | None:
    """Every real root of a univariate polynomial, exactly (radicals or ``CRootOf``),
    or None when ``expr`` isn't one SymPy can isolate roots of.

    This covers what ``solveset`` can't spell out, such as a cubic with three real
    roots, whose radical form needs complex numbers.
    """
    if expr.free_symbols != {var} or not expr.is_polynomial(var):
        return None
    try:
        poly = sp.Poly(expr, var)
        if poly.degree() < 1:
            return None
        return list(dict.fromkeys(sp.real_roots(poly)))
    except Exception:  # noqa: BLE001 - best effort; the numeric scan still follows
        return None


def _scan_grid() -> Iterator[float]:
    """Sample points in ascending order: fine near zero, geometrically sparser out to
    ``SCAN_LIMIT``."""
    ratio = (SCAN_LIMIT / 10) ** (1 / 4000)
    for i in range(4000, 0, -1):
        yield -10 * ratio**i
    for i in range(-10_000, 10_001):
        yield i / 1000
    for i in range(1, 4001):
        yield 10 * ratio**i


def _evaluator(expr: Any, var: Any) -> Callable[[float], float]:
    """``expr`` as a plain float function; NaN wherever it is undefined or non-real."""
    try:
        function = sp.lambdify(var, expr, modules=["math", "mpmath"])
    except (TypeError, ValueError, NotImplementedError, SyntaxError):
        raise SymbolicError("No exact solution found") from None

    def evaluate(x: float) -> float:
        try:
            y = complex(function(x))
        except (TypeError, ValueError, ZeroDivisionError, OverflowError):
            return math.nan
        if not (math.isfinite(y.real) and math.isfinite(y.imag)):
            return math.nan
        if abs(y.imag) > 1e-12 * (1 + abs(y.real)):
            return math.nan
        return y.real

    return evaluate


def _bisect(evaluate: Callable[[float], float], lo: float, hi: float, f_lo: float) -> float:
    """The point in ``[lo, hi]`` where ``evaluate`` changes sign (``f_lo`` = value at lo)."""
    for _ in range(200):
        mid = (lo + hi) / 2
        f_mid = evaluate(mid)
        if f_mid == 0 or mid in (lo, hi):
            return mid
        if math.isnan(f_mid):
            break
        if (f_mid < 0) == (f_lo < 0):
            lo, f_lo = mid, f_mid
        else:
            hi = mid
    return (lo + hi) / 2


def _minimize_abs(evaluate: Callable[[float], float], lo: float, hi: float) -> float:
    """The point in ``[lo, hi]`` where ``|evaluate|`` is smallest (golden-section search)."""
    golden = (math.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - golden * (b - a), a + golden * (b - a)
    f_c, f_d = abs(evaluate(c)), abs(evaluate(d))
    for _ in range(100):
        if math.isnan(f_c) or math.isnan(f_d):
            break
        if f_c < f_d:
            b, d, f_d = d, c, f_c
            c = b - golden * (b - a)
            f_c = abs(evaluate(c))
        else:
            a, c, f_c = c, d, f_d
            d = a + golden * (b - a)
            f_d = abs(evaluate(d))
    return (a + b) / 2


def _root_at(
    evaluate: Callable[[float], float],
    previous: tuple[float, float],
    current: tuple[float, float],
    following: tuple[float, float],
) -> float | None:
    """The root that three consecutive samples ``(x, f(x))`` point to, if any: one at
    ``current`` or one between ``current`` and ``following``."""
    (x_before, before), (x, a), (x_after, b) = previous, current, following
    if math.isnan(a) or math.isnan(b):
        return None
    if a == 0:
        # Only an isolated zero counts: a run of exact zeros is the function
        # underflowing (``x e^x`` for very negative x), not a root.
        return x if before != 0 and b != 0 else None
    if b != 0 and (a < 0) != (b < 0):
        # Compare signs, not ``a * b``, which underflows for tiny values.
        root = _bisect(evaluate, x, x_after, a)
        return root if abs(evaluate(root)) <= min(abs(a), abs(b)) else None  # a pole grows
    if (
        not math.isnan(before)
        and before != 0
        and (before < 0) == (a < 0)  # no crossing on either side: a possible touch
        and abs(a) < abs(b)
        and abs(a) <= abs(before)
        and abs(a) < _TANGENT_LIMIT
    ):
        root = _minimize_abs(evaluate, x_before, x_after)
        return root if abs(evaluate(root)) < _TANGENT_TOLERANCE else None
    return None


def _scan_roots(
    evaluate: Callable[[float], float], samples: Iterable[tuple[float, float]] | None = None
) -> Iterator[float]:
    """Approximate real roots in ``[-SCAN_LIMIT, SCAN_LIMIT]``, in ascending order.

    Sign changes are bisected (a pole is told apart from a root because the function
    grows there instead of shrinking), and points where ``|f|`` bottoms out without
    crossing zero are searched for tangential roots such as ``(x - 1)^2``. Samples are
    evaluated lazily, so a caller that stops early skips the rest of the scan.
    ``samples`` gives ``(x, f(x))`` over :func:`_scan_grid` when they are cheaper to
    compute together than one ``evaluate`` at a time.
    """
    window: deque[tuple[float, float]] = deque([(math.nan, math.nan)], maxlen=3)
    last = -math.inf
    if samples is None:
        samples = ((x, evaluate(x)) for x in _scan_grid())
    for sample in samples:
        window.append(sample)
        if len(window) < 3:
            continue
        root = _root_at(evaluate, *window)
        if root is not None and not math.isclose(root, last, rel_tol=1e-6, abs_tol=1e-6):
            last = root
            yield root


def _refine(expr: Any, var: Any, approximate: float) -> Any:
    """``approximate`` polished to 30 digits when Newton's method confirms it."""
    try:
        polished = sp.nsolve(expr, var, approximate, prec=30)
    except (ValueError, ZeroDivisionError, TypeError, NotImplementedError):
        return sp.Float(approximate, 17)
    if abs(float(polished) - approximate) > 1e-6 * (1 + abs(approximate)):
        return sp.Float(approximate, 17)  # converged to a different root
    return polished


def _fallback_candidates(real: Any, var: Any, solutions: Any) -> list[Any]:
    """Candidate roots when SymPy can't list the solutions itself (``solutions`` is what
    it returned): exact polynomial roots, else a numeric scan of the real line."""
    if isinstance(solutions, sp.Interval):
        _raise_unsolved(solutions)

    exact = _polynomial_roots(real, var)
    if exact is not None:
        if len(exact) > MAX_SOLUTIONS:
            raise SymbolicError(f"Too many solutions ({len(exact)})")
        return exact

    approximate = sorted(islice(_scan_roots(_evaluator(real, var)), MAX_SOLUTIONS + 1))
    if not approximate:
        if isinstance(solutions, sp.ConditionSet):
            raise SymbolicError(f"No solution found in the range ±{SCAN_LIMIT:g}")
        _raise_unsolved(solutions)
    if len(approximate) > MAX_SOLUTIONS:
        raise SymbolicError(
            f"Too many solutions (more than {MAX_SOLUTIONS} in the range ±{SCAN_LIMIT:g})"
        )
    return [_refine(real, var, x) for x in approximate]


def _numeric_roots(target: SymbolicTarget) -> list[Root]:
    """Roots found by scanning the curve as the plotter evaluates it, as decimals."""
    fn = NumericFunction(target.node, target.options.complex_mode)
    name = target.variable

    def values(xs: Any) -> Any:
        try:
            ys = np.asarray(fn(**{name: xs}), dtype=float)
        except (MathError, ValueError, OverflowError):
            return np.full(np.shape(xs), math.nan)
        return np.where(np.isfinite(ys), ys, math.nan)

    def samples() -> Iterator[tuple[float, float]]:
        grid = np.fromiter(_scan_grid(), dtype=float)
        terms = _term_count(target.node)
        # Thinned so the scan stays within _SCAN_WORK term evaluations: roots closer
        # together than the wider spacing can be missed.
        stride = -(-len(grid) * terms // _SCAN_WORK)
        if stride > 1:
            grid = np.concatenate([grid[::stride], grid[-1:]])
        chunk = max(1, min(_SCAN_CHUNK, MAX_TERM_WORK // terms))
        for start in range(0, len(grid), chunk):
            xs = grid[start : start + chunk]
            yield from zip(xs.tolist(), values(xs).tolist(), strict=True)

    found = sorted(islice(_scan_roots(lambda x: float(values(x)), samples()), MAX_SOLUTIONS + 1))
    if not found:
        raise SymbolicError(f"No solution found in the range ±{SCAN_LIMIT:g}")
    if len(found) > MAX_SOLUTIONS:
        raise SymbolicError(
            f"Too many solutions (more than {MAX_SOLUTIONS} in the range ±{SCAN_LIMIT:g})"
        )
    return list(_distinct(_root(sp.Float(x, 17), target.options) for x in found))


def _term_count(node: Node) -> int:
    """The most terms of any Σ/Π in ``node`` with constant bounds (1 when none)."""
    terms = 1
    for sub in walk(node):
        if isinstance(sub, Reduce) and sub.kind != "int":
            lower, upper = NumericFunction(sub.lower), NumericFunction(sub.upper)
            if lower.is_constant and upper.is_constant:
                count = upper.scalar() - lower.scalar() + 1
                if math.isfinite(count):
                    terms = max(terms, int(count))
    return terms


def _distinct(roots: Iterable[Root]) -> Iterator[Root]:
    """``roots`` (sorted by value) without repeats of the same root."""
    last: Root | None = None
    for root in roots:
        if last is None or not math.isclose(root.value, last.value, rel_tol=1e-12, abs_tol=1e-12):
            last = root
            yield root


def _family(image: Any, real: Any, var: Any, options: CompileOptions) -> tuple[Any, Any] | None:
    """``(base, period)`` of ``ImageSet(n -> base + period·n, Integers)``, with
    ``0 <= base < period``; None unless it is one with constant base and period whose
    members are roots of ``real`` as plotted."""
    if not isinstance(image, sp.ImageSet) or image.base_sets != (sp.S.Integers,):
        return None
    (n,), body = image.lamda.signature, sp.expand(image.lamda.expr)
    if not body.is_polynomial(n) or sp.degree(body, n) != 1:
        return None
    period, base = sp.Abs(body.coeff(n, 1)), body.coeff(n, 0)
    if period.free_symbols or base.free_symbols or not period.is_positive:
        return None
    base = sp.nsimplify(_simplify(base - sp.floor(base / period) * period))
    complex_mode = options.complex_mode
    if not all(_is_root(real, var, base + k * period, complex_mode) for k in (-2, -1, 0, 1, 2)):
        return None
    try:
        parameter = sp.Symbol("parameter", real=True)
        domain = continuous_domain(real, var, sp.S.Reals)
        indices = _parameter_domain(base + parameter * period, domain, parameter)
        if sp.Intersection(indices, sp.S.Integers) != sp.S.Integers:
            return None
    except (NotImplementedError, ValueError, TypeError):
        # Some domains (tan's infinitely many poles) are not interval unions.
        # A direct symbolic proof of every substituted subexpression also suffices.
        integer = sp.Dummy("index", integer=True)
        member = base + integer * period
        for part in sp.preorder_traversal(real):
            value = _simplify(part.subs(var, member))
            if value.is_finite is not True or (not complex_mode and value.is_real is not True):
                return None
    return base, period


def _merged(families: list[tuple[Any, Any]]) -> list[tuple[Any, Any]]:
    """Families of one period with the shortest period that describes them: ``2*pi*n``
    and ``pi + 2*pi*n`` are ``pi*n``; ``pi/6, 5*pi/6, 7*pi/6, 11*pi/6 (+ 2*pi*n)`` are
    ``pi/6 + pi*n, 5*pi/6 + pi*n``."""
    by_period: dict[Any, list[Any]] = {}
    for base, period in families:
        by_period.setdefault(period, []).append(base)
    merged = []
    for period, bases in by_period.items():
        bases = sorted(set(bases), key=float)
        for parts in range(len(bases), 1, -1):
            if len(bases) % parts:
                continue
            step = period / parts

            def shifted(b: Any, step: Any = step, period: Any = period) -> Any:
                return b + step if b + step < period else b + step - period

            if all(any(_simplify(shifted(b) - c) == 0 for c in bases) for b in bases):
                bases, period = [b for b in bases if b < step], step
                break
        merged += [(b, period) for b in bases]
    return sorted(merged, key=lambda f: (float(f[1]), float(f[0])))


def _general_solution(
    solutions: Any, real: Any, var: Any, options: CompileOptions
) -> GeneralSolution | None:
    """A periodic solution set (``sin(x) = 1/2``) in closed form, or None when
    ``solutions`` isn't a union of integer-indexed progressions and finite sets."""
    parts = solutions.args if isinstance(solutions, sp.Union) else (solutions,)
    families, isolated = [], []
    for part in parts:
        if isinstance(part, sp.FiniteSet):
            isolated += list(part)
            continue
        family = _family(part, real, var, options)
        if family is None:
            return None
        families.append(family)
    if not families:
        return None
    families = _merged(families)
    if len(families) > MAX_SOLUTIONS:
        raise SymbolicError(f"Too many families of solutions ({len(families)})")

    def in_family(value: Any) -> bool:
        return any(_simplify((value - b) / p).is_integer for b, p in families)

    roots = sorted(
        (
            _root(s, options)
            for s in isolated
            if not in_family(s) and _is_root(real, var, s, options.complex_mode)
        ),
        key=lambda r: r.value,
    )
    return GeneralSolution(
        tuple(
            SolutionFamily(
                to_source(base, options), to_source(period, options), float(base), float(period)
            )
            for base, period in families
        ),
        tuple((r.exact, r.value) for r in _distinct(roots)),
    )


def _parameter_domain(body: Any, domain: Any, n: Any) -> Any:
    """The real parameter values whose image belongs to ``domain``.

    Unresolved conditions stay symbolic and are rejected by the generator converter;
    no numerical samples are used to infer a parameter's domain.
    """
    if domain is sp.S.Reals:
        return domain
    if isinstance(domain, sp.Union):
        return sp.Union(*(_parameter_domain(body, part, n) for part in domain.args))
    if isinstance(domain, sp.Intersection):
        return sp.Intersection(*(_parameter_domain(body, part, n) for part in domain.args))
    if isinstance(domain, sp.Complement):
        a, b = domain.args
        return _parameter_domain(body, a, n) - _parameter_domain(body, b, n)
    if isinstance(domain, sp.FiniteSet):
        return sp.Union(*(sp.solveset(body - value, n, domain=sp.S.Reals) for value in domain))
    if domain is sp.S.EmptySet:
        return domain
    if isinstance(domain, sp.Interval):
        conditions = []
        if domain.start is not sp.S.NegativeInfinity:
            conditions.append(body > domain.start if domain.left_open else body >= domain.start)
        if domain.end is not sp.S.Infinity:
            conditions.append(body < domain.end if domain.right_open else body <= domain.end)
        return sp.Intersection(*(sp.solveset(c, n, domain=sp.S.Reals) for c in conditions))
    raise NotImplementedError("Unsupported generator domain")


def _image_parts(image: Any, n: Any) -> list[tuple[Any, Any]]:
    """Flatten nested images into ``(formula(n), real parameter domain)`` pairs."""
    if image is sp.S.Integers:
        return [(n, sp.S.Reals)]
    if image in (sp.S.Naturals, sp.S.Naturals0):
        return [(n, sp.Interval(1 if image is sp.S.Naturals else 0, sp.oo))]
    if isinstance(image, sp.Range) and image.step == 1:
        return [(n, sp.Interval(image.start, image.stop, right_open=True))]
    if isinstance(image, sp.Union):
        return [part for item in image.args for part in _image_parts(item, n)]
    if isinstance(image, sp.ImageSet) and len(image.lamda.signature) == 1:
        (variable,) = image.lamda.signature
        return [
            (image.lamda.expr.subs(variable, body), domain)
            for body, domain in _image_parts(image.base_sets[0], n)
        ]
    if isinstance(image, sp.Intersection):
        # Pull back interval restrictions on an image, such as the nonnegative
        # arguments of sqrt in sin(sqrt(x)) = 0.
        images = [part for part in image.args if isinstance(part, sp.ImageSet)]
        if len(images) == 1:
            domain = sp.Intersection(*(part for part in image.args if part != images[0]))
            return [
                (body, indices & _parameter_domain(body, domain, n))
                for body, indices in _image_parts(images[0], n)
            ]
    if isinstance(image, sp.Complement):
        base, excluded = image.args
        return [
            (body, indices - _parameter_domain(body, excluded, n))
            for body, indices in _image_parts(base, n)
        ]
    raise NotImplementedError("No integer-indexed generator")


def _generator_solution(
    solutions: Any, real: Any, var: Any, options: CompileOptions
) -> GeneralSolution | None:
    """Countable exact solutions over ℕ or ℤ, including restricted/nonlinear images."""
    n = sp.Symbol("n", real=True)
    generators: list[SolutionGenerator] = []
    isolated: list[Any] = []
    try:
        domain = continuous_domain(real, var, sp.S.Reals)
        parts = solutions.args if isinstance(solutions, sp.Union) else (solutions,)
        for part in parts:
            if isinstance(part, sp.FiniteSet):
                isolated.extend(part)
                continue
            for body, indices in _image_parts(part, n):
                indices &= _parameter_domain(body, domain, n)
                indices = sp.Intersection(indices, sp.S.Integers)
                ranges = indices.args if isinstance(indices, sp.Union) else (indices,)
                for members in ranges:
                    if members is sp.S.EmptySet:
                        continue
                    if isinstance(members, sp.FiniteSet):
                        isolated.extend(body.subs(n, k) for k in members)
                        continue
                    if members is sp.S.Integers:
                        formula, label = body, "Z"
                    elif isinstance(members, sp.Range) and members.step == 1:
                        if members.size.is_finite:
                            if members.size > MAX_SOLUTIONS:
                                return None
                            isolated.extend(body.subs(n, k) for k in members)
                            continue
                        if members.start is sp.S.NegativeInfinity:
                            formula = body.subs(n, members.stop - n)
                        else:
                            formula = body.subs(n, members.start + n - 1)
                        label = "N"
                    else:
                        return None
                    formula = _simplify(formula)
                    if formula.free_symbols - {n}:
                        return None
                    # Verify against the unsimplified expression too (poles, real
                    # roots, and undefined factors must agree with the graph).
                    probes = (-2, -1, 0, 1, 2) if label == "Z" else (1, 2, 3, 4, 5)
                    if not all(
                        _is_root(real, var, formula.subs(n, k), options.complex_mode)
                        for k in probes
                    ):
                        return None
                    generator = SolutionGenerator(to_source(formula, options), label, options)
                    if generator not in generators:
                        generators.append(generator)
        if not generators or len(generators) > MAX_SOLUTIONS:
            return None
        roots = sorted(
            (_root(s, options) for s in isolated if _is_root(real, var, s, options.complex_mode)),
            key=lambda root: root.value,
        )
        return GeneralSolution(
            (), tuple((r.exact, r.value) for r in _distinct(roots)), tuple(generators)
        )
    except (NotImplementedError, ValueError, TypeError, SymbolicError):
        return None


def solve(target: SymbolicTarget) -> list[Root] | GeneralSolution:
    """Real roots of the expression, sorted, with equal roots merged.

    Tried in order: SymPy's exact solutions, exact roots of a polynomial, and a
    numerical scan of ``[-SCAN_LIMIT, SCAN_LIMIT]`` for anything else (``cos(x) = x``,
    ``x e^x = 2``). Roots are verified against the expression as the plotter evaluates
    it. A root with no exact spelling in OpenGraph syntax is returned as a decimal.
    Infinitely many solutions that SymPy writes as periodic families
    (``sin(x) = 1/2``: ``pi/6 + 2*pi*n, 5*pi/6 + 2*pi*n``) are a
    :class:`GeneralSolution`, as are nonlinear and restricted images that can be
    reduced to exact generators over ℕ or ℤ. Raises when no root can be found or
    there are too many to list without a generator.
    """
    var = _variable(target)
    complex_mode = target.options.complex_mode

    expr = to_sympy(target.node)
    if expr.has(sp.Sum, sp.Product):
        # A Σ/Π with too many terms to write out: SymPy can't solve it, and evaluating
        # it term by term through SymPy takes forever, so only the scan is left.
        return _numeric_roots(target)
    real = expr if complex_mode else to_sympy(target.node, real_roots=True)

    solutions = _exact_solutions(expr, real, var)
    if contains(target.node, Guard) or contains(target.node, Bounded):
        solutions = sp.Intersection(solutions, _original_domain(target.node, var))
    if not _is_finite(solutions):
        general = _general_solution(solutions, real, var, target.options)
        if general is None:
            general = _generator_solution(solutions, real, var, target.options)
        if general is not None:
            return general
    if _is_finite(solutions):
        assert isinstance(solutions, (sp.FiniteSet, type(sp.S.EmptySet)))
        if len(solutions) > MAX_SOLUTIONS:
            raise SymbolicError(f"Too many solutions ({len(solutions)})")
        found: Iterable[Any] = solutions
    else:
        found = _fallback_candidates(real, var, solutions)

    candidates = (
        s
        for s in found
        if _is_root(real, var, s, complex_mode)
        and _defined_at(target.node, target.variable, s, complex_mode)
    )
    roots = sorted((_root(s, target.options) for s in candidates), key=lambda r: r.value)
    return _with_names(target, list(_distinct(roots)))


def _with_names(target: SymbolicTarget, roots: list[Root]) -> list[Root]:
    """``roots`` written in terms of the user constants the row uses, when SymPy can
    solve the row with them kept as names: ``a x^2 - 4`` (a = 1) has the roots
    ``-2/sqrt(a)`` and ``2/sqrt(a)``, which follow a slider for a.

    Only used when the named solutions, at the constants' current values, are exactly
    the roots found (a solution that is complex or undefined there is dropped);
    otherwise the roots stay as they are.
    """
    if not roots or target.named is None or target.named_unsolved or not target.constants:
        return roots
    var = _variable(target)
    try:
        values = {_symbol(name, real=True): to_sympy(node) for name, node in target.constants}
        solutions = sp.solve(to_sympy(target.named), var, dict=False)
    except Exception:  # noqa: BLE001 - best effort; the roots found stand
        return roots
    if not isinstance(solutions, list) or len(solutions) > MAX_SOLUTIONS:
        return roots
    named: list[tuple[float, Any]] = []
    for solution in solutions:
        try:
            number = complex(sp.N(solution.subs(values)))
        except (TypeError, ValueError):
            continue
        if math.isfinite(number.real) and abs(number.imag) <= 1e-9 * (1 + abs(number.real)):
            named.append((number.real, solution))
    named.sort(key=lambda pair: pair[0])
    if len(named) != len(roots) or not all(
        math.isclose(value, root.value, rel_tol=1e-9, abs_tol=1e-9)
        for (value, _), root in zip(named, roots, strict=True)
    ):
        return roots
    try:
        texts = [to_source(_simplify(solution), target.options) for _, solution in named]
    except SymbolicError:
        return roots
    return [Root(text, root.value) for text, root in zip(texts, roots, strict=True)]


def rearrange(target: SymbolicTarget) -> str:
    """The row ``name = …`` solving ``target.node = 0`` for ``target.variable``; several
    solutions are a list, ``r = [-sqrt(x^2 + y^2), sqrt(x^2 + y^2)]``.

    The other names stay symbols, so the result may use names that nothing defines
    yet (``y = a x + b`` solved for a gives ``a = (y - b)/x``). It is checked to parse,
    not to compile.
    """
    name = target.variable
    try:
        solutions = sp.solve(to_sympy(target.node), _variable(target))
    except NotImplementedError:
        raise SymbolicError(f"No symbolic solution for {name} found") from None
    if not target.options.complex_mode:
        solutions = [s for s in solutions if not s.has(sp.I)]
    if not solutions:
        raise SymbolicError(f"No solution for {name} found")
    if len(solutions) > MAX_SOLUTIONS:
        raise SymbolicError(f"Too many solutions ({len(solutions)})")

    printed = list(dict.fromkeys(_print(s, target.options) for s in solutions))
    row = f"{name} = {printed[0]}" if len(printed) == 1 else f"{name} = list[{', '.join(printed)}]"
    _check_parses(row)
    return row
