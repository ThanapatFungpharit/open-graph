"""Compile a resolved AST into a vectorised NumPy function.

The tree is translated once into nested closures; evaluating it then costs one NumPy
operation per node regardless of how many sample points are passed in. Sub-trees that
do not depend on any variable are folded to constants at compile time.

Semantics are real-valued by default: results outside a function's domain are NaN
(never complex), division by zero yields +/-inf or NaN, and callers treat every
non-finite value as "undefined here". In *complex mode* the same tree is evaluated in
complex128 arithmetic (principal branches); :meth:`NumericFunction.complex_values` returns
those values and the ordinary call projects them to the reals (non-real values become
NaN), so curves still plot where the function is real.

The explicit complex projections re, im and arg evaluate their arguments using complex
arithmetic even in real mode, then return a real value.

Variable-free sub-trees are folded once at compile time, in 256-bit MPFR arithmetic
(:mod:`~opengraph.engine.precise`) rounded once to float64, so constants such as
``1e16 + 1 - 1e16`` or ``sin(30)`` in degrees are exact where float64 would round.

Σ, Π and ∫ (:class:`~opengraph.engine.ast.Reduce`) evaluate their body with the bound
variable added to the environment: sums and products term by term, integrals with
tanh-sinh quadrature (accurate for smooth integrands and for integrable singularities
at the endpoints). An integral whose estimate does not converge is NaN rather than a
wrong number.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from fractions import Fraction
from functools import cache
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from opengraph.engine import precise
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
    free_vars,
)
from opengraph.engine.builtins import COMPLEX_PROJECTIONS, FUNCTIONS, Array
from opengraph.engine.cancellation import checkpoint

Env = Mapping[str, Any]
_Fn = Callable[[Env], Any]

MAX_TERMS = 100_000  # Σ/Π: most terms in one sum
MAX_TERM_WORK = 4_000_000  # Σ/Π: most terms × samples per evaluation (keeps the UI live)
TERM_BLOCK = 1 << 16  # Σ/Π: most terms × samples evaluated in one array operation
QUADRATURE_LEVELS = (4, 5, 6, 7)  # tanh-sinh step h = 2^-level
QUADRATURE_TOLERANCE = 1e-8  # relative to the integral of |f|
_TANH_SINH_RANGE = 4.0  # |t| range: nodes reach 1e-37 from the ends (for 1/sqrt(t)-like ends)


def _constant(value: Any, cx: bool) -> tuple[_Fn, bool]:
    folded = np.complex128(value) if cx else np.float64(np.real(value))
    return (lambda env: folded), True


def _fold(node: Node, fn: _Fn, cx: bool, exact: bool) -> tuple[_Fn, bool]:
    """The constant value of a variable-free ``node``: computed in high precision and
    rounded once when it has a real precise form, else by its float function ``fn``."""
    if exact and not cx and not isinstance(node, Num | Const):
        value = precise.fold(node)
        if value is not None:
            return _constant(value, cx)
    return _constant(fn({}), cx)


def _real_power(base: Array, exponent: float) -> Array:
    """``base ** exponent`` with real odd roots for negative bases.

    NumPy returns NaN for ``(-8) ** (1/3)``; mathematically (and for plotting) users
    expect -2. Exponents that are (close to) rationals with an odd denominator are
    therefore evaluated as ``sign(b)^num * |b|^p``.
    """
    if exponent == round(exponent):
        return np.power(base, exponent)
    frac = Fraction(exponent).limit_denominator(999)
    if frac.denominator % 2 == 1 and abs(float(frac) - exponent) < 1e-12:
        magnitude = np.power(np.abs(base), exponent)
        return magnitude * np.sign(base) if frac.numerator % 2 else magnitude
    return np.power(base, exponent)


def _build(node: Node, cx: bool, exact: bool = True) -> tuple[_Fn, bool]:
    """Return ``(function, is_constant)`` for ``node``; ``exact`` folds constants in
    high precision (see :mod:`opengraph.engine.precise`)."""
    match node:
        case Guard(value=value, original=original):
            result, result_constant = _build(value, cx, exact)
            domain, domain_constant = _build(original, cx, exact)
            fn: _Fn = lambda env: np.where(np.isfinite(domain(env)), result(env), np.nan)  # noqa: E731
            if result_constant and domain_constant:
                return _fold(node, fn, cx, exact)
            return fn, False
        case Bounded(value=value, coordinate=coordinate, interval=interval):
            inner, value_constant = _build(value, cx, exact)
            at, coordinate_constant = _build(coordinate, cx, exact)
            assert isinstance(interval.lo, Num) and isinstance(interval.hi, Num)
            lo, hi = interval.lo.value, interval.hi.value

            def bounded(env: Env) -> Any:
                t = np.asarray(at(env))
                inside = (t >= lo if interval.closed_lo else t > lo) & (
                    t <= hi if interval.closed_hi else t < hi
                )
                return np.where(inside, inner(env), np.nan)

            if value_constant and coordinate_constant:
                return _fold(node, bounded, cx, exact)
            return bounded, False
        case Num(value=value) | Const(value=value):
            return _constant(value, cx)
        case Var(name=name):
            return (lambda env: env[name]), False
        case UnaryOp(operand=operand):
            inner, const = _build(operand, cx, exact)
            if cx:
                # 0 - z rather than -z: negating 4+0j gives -4-0j, and the signed zero
                # puts sqrt(-4) on the wrong side of the branch cut (-2i, not 2i).
                fn = lambda env: np.subtract(0, inner(env))  # noqa: E731
            else:
                fn = lambda env: np.negative(inner(env))  # noqa: E731
            return _fold(node, fn, cx, exact) if const else (fn, False)
        case BinOp(op=op, left=left, right=right):
            return _build_binary(node, op, left, right, cx, exact)
        case FuncCall(name=name, args=args):
            builtin = FUNCTIONS[name]
            impl = builtin.complex_numeric if cx else builtin.numeric
            built = [_build(arg, cx or name in COMPLEX_PROJECTIONS, exact) for arg in args]
            arg_fns = [fn for fn, _ in built]
            fn = lambda env: impl(*(f(env) for f in arg_fns))  # noqa: E731
            if all(const for _, const in built):
                return _fold(node, fn, cx, exact)
            return fn, False
        case Reduce():
            fn = _build_reduce(node, cx, exact)
            if not free_vars(node):
                return _fold(node, fn, cx, exact)
            return fn, False
    raise TypeError(f"Cannot evaluate unresolved node {type(node).__name__}")


def _build_binary(
    node: Node, op: str, left: Node, right: Node, cx: bool, exact: bool
) -> tuple[_Fn, bool]:
    lf, lconst = _build(left, cx, exact)
    rf, rconst = _build(right, cx, exact)
    fn: _Fn
    if op == "+":
        fn = lambda env: np.add(lf(env), rf(env))  # noqa: E731
    elif op == "-":
        fn = lambda env: np.subtract(lf(env), rf(env))  # noqa: E731
    elif op == "*":
        fn = lambda env: np.multiply(lf(env), rf(env))  # noqa: E731
    elif op == "/":
        fn = lambda env: np.divide(lf(env), rf(env))  # noqa: E731
    elif op == "^":
        if rconst and not cx:
            exponent = float(rf({}))
            fn = lambda env: _real_power(lf(env), exponent)  # noqa: E731
        else:
            fn = lambda env: np.power(lf(env), rf(env))  # noqa: E731
    else:  # pragma: no cover - parser only produces the operators above
        raise ValueError(f"Unknown operator {op!r}")
    if lconst and rconst:
        return _fold(node, fn, cx, exact)
    return fn, False


# -- Σ, Π and ∫ ----------------------------------------------------------------------------


def _build_reduce(node: Reduce, cx: bool, exact: bool) -> _Fn:
    body, _ = _build(node.body, cx, exact)
    lower, _ = _build(node.lower, cx, exact)
    upper, _ = _build(node.upper, cx, exact)
    if node.kind == "int":
        return lambda env: _integrate(body, node.var, lower(env), upper(env), env, cx)
    return lambda env: _accumulate(
        node.kind == "prod", body, node.var, lower(env), upper(env), env, cx
    )


def _accumulate(
    product: bool, body: _Fn, var: str, lower: Any, upper: Any, env: Env, cx: bool
) -> Any:
    """Σ or Π of ``body`` over the integers between the (rounded) bounds, per sample."""
    lo = np.round(np.real(np.asarray(lower)))
    hi = np.round(np.real(np.asarray(upper)))
    shape = np.broadcast_shapes(lo.shape, hi.shape)
    dtype = np.complex128 if cx else np.float64
    bounded = np.isfinite(lo) & np.isfinite(hi)
    if not bounded.any():
        return np.full(shape, np.nan, dtype=dtype)
    start = int(np.min(np.where(bounded, lo, np.inf)))
    stop = int(np.max(np.where(bounded, hi, -np.inf)))
    terms = stop - start + 1
    samples = max((int(np.size(v)) for v in env.values()), default=1)
    if terms > MAX_TERMS or terms * samples > MAX_TERM_WORK:
        return np.full(shape, np.nan, dtype=dtype)  # too many terms to evaluate
    result: Any = np.full(shape, 1.0 if product else 0.0, dtype=dtype)
    neutral = 1.0 if product else 0.0
    # Terms are evaluated a block at a time along a new leading axis: one array
    # operation covers many terms, instead of a Python loop step for each.
    shape = np.broadcast_shapes(shape, *(np.shape(v) for v in env.values()))
    result = np.broadcast_to(result, shape)
    block = max(1, TERM_BLOCK // max(samples, 1))
    for first in range(start, stop + 1, block):
        checkpoint()
        ks = np.arange(first, min(first + block, stop + 1), dtype=dtype)
        ks = ks.reshape((-1,) + (1,) * len(shape))
        term = np.where((lo <= ks) & (ks <= hi), body({**env, var: ks}), neutral)
        term = np.broadcast_to(term, (len(ks), *shape))
        result = result * np.prod(term, axis=0) if product else result + np.sum(term, axis=0)
    return np.where(bounded, result, np.nan)


@cache
def _tanh_sinh(level: int) -> tuple[Array, Array, Array]:
    """Tanh-sinh quadrature with step 2^-level on (-1, 1).

    Returns ``(distance, side, weight)``: each node's distance from the nearer end of
    the interval (computed directly, so nodes 1e-17 from an endpoint keep their
    precision), which end that is (-1 or +1), and its weight.
    """
    h = 2.0**-level
    count = math.ceil(_TANH_SINH_RANGE / h)
    t = np.arange(-count, count + 1) * h
    s = 0.5 * math.pi * np.sinh(t)
    distance = 2.0 / (1.0 + np.exp(2.0 * np.abs(s)))  # 1 - |tanh(s)|
    weights = h * 0.5 * math.pi * np.cosh(t) / np.cosh(s) ** 2
    keep = (distance > 0) & (weights > 1e-300)
    return distance[keep], np.where(t < 0, -1.0, 1.0)[keep], weights[keep]


def _integrate(body: _Fn, var: str, lower: Any, upper: Any, env: Env, cx: bool) -> Any:
    a = np.asarray(lower)
    b = np.asarray(upper)
    if not cx:
        a, b = np.real(a), np.real(b)
    half = 0.5 * (b - a)
    expanded = {k: np.asarray(v)[..., None] for k, v in env.items()}
    shape = np.broadcast_shapes(half.shape, *(np.shape(v) for v in env.values()))
    dtype = np.complex128 if cx else np.float64
    result = np.full(shape, np.nan, dtype=dtype)
    pending = np.ones(shape, dtype=bool)
    previous: Any = None
    for level in QUADRATURE_LEVELS:
        checkpoint()
        distance, side, weights = _tanh_sinh(level)
        offset = half[..., None] * distance  # from the nearer end of (a, b)
        points = np.where(side < 0, a[..., None] + offset, b[..., None] - offset)
        values = np.broadcast_to(body({**expanded, var: points}), (*shape, weights.size))
        finite = np.isfinite(values).all(axis=-1)
        estimate = half * np.sum(weights * np.where(np.isfinite(values), values, 0), axis=-1)
        scale = np.abs(half) * np.sum(weights * np.abs(np.nan_to_num(values)), axis=-1)
        if previous is not None:
            converged = np.abs(estimate - previous) <= QUADRATURE_TOLERANCE * scale + 1e-300
            done = pending & converged & finite
            result = np.where(done, estimate, result)
            pending &= ~done & finite
            if not pending.any():
                break
        previous = estimate
    return result


class NumericFunction:
    """A resolved expression compiled for fast vectorised evaluation.

    Call it with keyword arrays for the variables it uses, e.g. ``f(x=xs)`` or
    ``f(x=xx, y=yy)``. The result always has the broadcast shape of the inputs and
    dtype float64, even when the expression is constant. With ``complex_mode`` the
    expression is evaluated in complex arithmetic; calling it gives the real values
    (NaN where the value is not real) and :meth:`complex_values` the full complex values.
    """

    def __init__(
        self, node: Node, complex_mode: bool = False, *, precise_constants: bool = True
    ) -> None:
        self.node = node
        self.complex_mode = complex_mode
        with np.errstate(all="ignore"):
            self._fn, self.is_constant = _build(node, complex_mode, precise_constants)

    def _evaluate(self, variables: Mapping[str, ArrayLike]) -> NDArray[Any]:
        checkpoint()
        dtype = np.complex128 if self.complex_mode else np.float64
        env = {name: np.asarray(value, dtype=dtype) for name, value in variables.items()}
        shape = np.broadcast_shapes(*(v.shape for v in env.values())) if env else ()
        with np.errstate(all="ignore"):
            raw = self._fn(env)
            result = np.asarray(raw if self.complex_mode else np.real(raw), dtype=dtype)
        if result.shape != shape:
            result = np.broadcast_to(result, shape).copy()
        return result

    def __call__(self, **variables: ArrayLike) -> Array:
        result = self._evaluate(variables)
        if not self.complex_mode:
            return result
        return real_part(result)

    def complex_values(self, **variables: ArrayLike) -> NDArray[np.complex128]:
        return np.asarray(self._evaluate(variables), dtype=np.complex128)

    def scalar(self, **variables: float) -> float:
        return float(self(**variables))

    def complex_scalar(self, **variables: float) -> complex:
        return complex(self.complex_values(**variables))


def real_part(values: NDArray[Any]) -> Array:
    """Real values of a complex array; NaN where the imaginary part is not negligible."""
    real = np.real(values)
    with np.errstate(invalid="ignore"):
        is_real = np.abs(np.imag(values)) <= 1e-9 * (1.0 + np.abs(real))
    return np.where(is_real, real, np.nan).astype(np.float64)
