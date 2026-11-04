"""Arbitrary-precision evaluation with gmpy2 (MPFR), used only where float64 is not enough.

Plotting normally evaluates float64 NumPy closures (:mod:`opengraph.engine.evaluator`).
Two situations need more than float64's 53 bits:

* **Deep zoom.** When a pixel is narrower than a few thousand float64 steps at the
  view's coordinates (``x`` near 10^4 at 10^12 pixels per unit, say), samples collapse
  onto the same floats and curves turn into staircases. The viewport then keeps its
  position in high precision and every tile carries an exact *origin*; sample
  coordinates are small float64 offsets from it, the expression is evaluated in MPFR at
  :func:`bits_for_view` bits, and only the (small) offset of the result from the output
  origin is rounded back to float64. :func:`precision_bits` decides when this happens,
  so ordinary views keep the fast path.
* **Constants.** Sub-expressions without variables are folded once at compile time
  (``1e16 + 1 - 1e16``, ``sum(1/k^2 for k from 1 to 1000)``, ``sin(30)`` in degrees),
  where a 256-bit evaluation rounded once to float64 costs nothing per sample.

Two interchangeable executors implement the same semantics as the float evaluator (NaN
outside a function's real domain, real odd roots, ``Guard`` and ``Bounded`` domains):

* :class:`Program` — a flat register bytecode executed by the compiled MPFR machine in
  :mod:`opengraph.native` without the GIL. It covers every real builtin.
* :func:`scalar_closure` — gmpy2 Python closures, which also cover Σ/Π (and ∫ through the
  float evaluator). Used for folding and whenever the native machine is unavailable or a
  program needs Σ/Π/∫ with variable bounds.

Complex mode always uses float64 (complex results of a precise view fall back to it).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum
from fractions import Fraction
from functools import lru_cache
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

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
from opengraph.engine.cancellation import checkpoint
from opengraph.engine.multiprecision import gmpy2

Mpfr = Any  # gmpy2.mpfr (gmpy2 ships no type information)

FOLD_BITS = 256  # constant folding
FLOAT_SAFE_BITS = 40  # float64 resolves 2^-40 of a pixel's span: 2^13 steps per pixel
GUARD_BITS = 64  # extra bits for cancellation inside an expression
MAX_BITS = 1024
MAX_TERMS = 100_000  # Σ/Π: most terms, as in the float evaluator
CHUNK = 2048  # elements evaluated between cancellation checks


class Unsupported(Exception):
    """The expression has no real-valued precise form (complex values, unknown nodes)."""


def context(bits: int) -> Any:
    """A gmpy2 context for ``bits`` of precision that never raises (NaN/inf instead)."""
    return gmpy2.context(
        precision=bits,
        allow_complex=False,
        trap_underflow=False,
        trap_overflow=False,
        trap_inexact=False,
        trap_invalid=False,
        trap_erange=False,
        trap_divzero=False,
    )


def mpfr(value: Any, bits: int) -> Mpfr:
    """``value`` (a float, int, string or mpfr) rounded to ``bits``."""
    with context(bits):
        return gmpy2.mpfr(value)


def literal(value: float, bits: int) -> Mpfr:
    """A number as the user wrote it: floats come from decimal literals, and their
    shortest repr is that literal (``0.1``, not 0.1000000000000000055…)."""
    if math.isfinite(value):
        return mpfr(repr(float(value)), bits)
    return mpfr(value, bits)


# -- when is float64 not enough? ----------------------------------------------------------


def precision_bits(magnitude: float, units_per_px: float) -> int:
    """Bits to evaluate with at coordinates up to ``magnitude`` when one pixel is
    ``units_per_px`` wide; 0 while float64 resolves a pixel finely enough."""
    if not (magnitude > 0 and units_per_px > 0) or not math.isfinite(magnitude):
        return 0
    ratio = math.log2(magnitude) - math.log2(units_per_px)
    if ratio <= FLOAT_SAFE_BITS:
        return 0
    return min(MAX_BITS, max(64, math.ceil(ratio) + GUARD_BITS))


# -- tree evaluation with gmpy2 scalars (folding, Σ/Π, fallback) --------------------------

ScalarFn = Callable[[Mapping[str, Mpfr]], Mpfr]
_NAN = gmpy2.nan()


def _is_integer(value: Mpfr) -> bool:
    return bool(gmpy2.is_finite(value)) and bool(gmpy2.is_integer(value))


def _gamma(value: Mpfr) -> Mpfr:
    if _is_integer(value) and value <= 0:
        return _NAN
    return gmpy2.gamma(value)


def _sign(value: Mpfr) -> Mpfr:
    if gmpy2.is_nan(value):
        return value
    return gmpy2.mpfr(gmpy2.sign(value))


def _mod(a: Mpfr, b: Mpfr) -> Mpfr:
    """``a mod b`` with the sign of ``b`` (NumPy's ``mod``)."""
    r = gmpy2.fmod(a, b)
    if r != 0 and (r < 0) != (b < 0):
        r = r + b
    return r


def _minmax(pick_first: Callable[[Mpfr, Mpfr], bool]) -> Callable[..., Mpfr]:
    def apply(*args: Mpfr) -> Mpfr:
        best = args[0]
        for arg in args[1:]:
            if gmpy2.is_nan(best) or gmpy2.is_nan(arg):
                return _NAN
            if not pick_first(best, arg):
                best = arg
        return best

    return apply


def _oddpow(base: Mpfr, exponent: Mpfr, odd_numerator: bool) -> Mpfr:
    magnitude = gmpy2.mpfr(abs(base)) ** exponent
    return magnitude * gmpy2.sign(base) if odd_numerator else magnitude


_UNARY: dict[str, Callable[[Mpfr], Mpfr]] = {
    "sin": gmpy2.sin,
    "cos": gmpy2.cos,
    "tan": gmpy2.tan,
    "sec": gmpy2.sec,
    "csc": gmpy2.csc,
    "cot": gmpy2.cot,
    "asin": gmpy2.asin,
    "acos": gmpy2.acos,
    "atan": gmpy2.atan,
    "sinh": gmpy2.sinh,
    "cosh": gmpy2.cosh,
    "tanh": gmpy2.tanh,
    "asinh": gmpy2.asinh,
    "acosh": gmpy2.acosh,
    "atanh": gmpy2.atanh,
    "sqrt": gmpy2.sqrt,
    "cbrt": gmpy2.cbrt,
    "abs": lambda a: gmpy2.mpfr(abs(a)),
    "exp": gmpy2.exp,
    "ln": gmpy2.log,
    "Gamma": _gamma,
    "factorial": lambda a: _gamma(a + 1),
    "floor": gmpy2.rint_floor,
    "ceil": gmpy2.rint_ceil,
    "round": lambda a: gmpy2.rint_floor(a + gmpy2.mpfr("0.5")),
    "sign": _sign,
}


def _builtin(name: str, nargs: int) -> Callable[..., Mpfr]:
    if nargs == 1 and name in _UNARY:
        return _UNARY[name]
    if name == "atan" and nargs == 2:
        return cast(Callable[..., Mpfr], gmpy2.atan2)
    if name == "log":
        if nargs == 1:
            return cast(Callable[..., Mpfr], gmpy2.log10)
        return lambda a, b: gmpy2.log(a) / gmpy2.log(b)
    if name == "nCr":
        return lambda n, k: _gamma(n + 1) / (_gamma(k + 1) * _gamma(n - k + 1))
    if name == "nPr":
        return lambda n, k: _gamma(n + 1) / _gamma(n - k + 1)
    if name == "mod":
        return _mod
    if name == "min":
        return _minmax(lambda a, b: a <= b)
    if name == "max":
        return _minmax(lambda a, b: a >= b)
    raise Unsupported(name)


def _constant_value(node: Num | Const, bits: int) -> Mpfr:
    if isinstance(node, Num):
        return literal(node.value, bits)
    if isinstance(node.value, complex) and node.value.imag:
        raise Unsupported(node.name)
    with context(bits):
        if node.name in ("pi", "π"):
            return gmpy2.const_pi()
        if node.name in ("tau", "τ"):
            return 2 * gmpy2.const_pi()
        if node.name == "e":
            return gmpy2.exp(1)
    return literal(float(node.value.real), bits)


def _exponent_form(value: float) -> tuple[bool, bool] | None:
    """How the float evaluator raises to a constant ``value``: ``None`` for an ordinary
    power, else ``(True, odd_numerator)`` for a real odd root."""
    if not math.isfinite(value) or value == round(value):
        return None
    frac = Fraction(value).limit_denominator(999)
    if frac.denominator % 2 == 1 and abs(float(frac) - value) < 1e-12:
        return True, bool(frac.numerator % 2)
    return None


_VALUES_LIMIT = 4096
_values: dict[tuple[Node, int], Mpfr] = {}


def constant_value(node: Node, bits: int) -> Mpfr:
    """The value of a variable-free ``node`` at ``bits`` (cached by structure, so the
    nested constants of a tree are evaluated once)."""
    key = (node, bits)
    value = _values.get(key)
    if value is None:
        fn = _compile(node, bits)
        with context(bits):
            value = fn({})
        if len(_values) >= _VALUES_LIMIT:
            _values.clear()
        _values[key] = value
    return value


def scalar_closure(node: Node, bits: int) -> ScalarFn:
    """Compile ``node`` into a function of an environment of mpfr values (call it
    inside ``context(bits)``). Raises :class:`Unsupported` for complex values."""
    return _compile(node, bits)


@lru_cache(maxsize=1024)
def _free(node: Node) -> bool:
    return bool(free_vars(node))


@lru_cache(maxsize=1024)
def _compile(node: Node, bits: int) -> ScalarFn:
    def build(child: Node) -> ScalarFn:
        if not isinstance(child, Num | Const) and not _free(child):
            value = constant_value(child, bits)
            return lambda env: value
        return _compile(child, bits)

    match node:
        case Num() | Const():
            value = _constant_value(node, bits)
            return lambda env: value
        case Var(name=name):
            return lambda env: env[name]
        case UnaryOp(operand=operand):
            inner = build(operand)
            return lambda env: -inner(env)
        case BinOp(op="^", left=left, right=right) if not _free(right):
            base = build(left)
            exponent = constant_value(right, bits)
            form = _exponent_form(float(exponent))
            if form is not None:
                odd = form[1]
                return lambda env: _oddpow(base(env), exponent, odd)
            integral = float(exponent)
            if math.isfinite(integral) and integral == round(integral):
                exponent = mpfr(int(integral), bits)
            return lambda env: base(env) ** exponent
        case BinOp(op=op, left=left, right=right):
            lf, rf = build(left), build(right)
            if op == "+":
                return lambda env: lf(env) + rf(env)
            if op == "-":
                return lambda env: lf(env) - rf(env)
            if op == "*":
                return lambda env: lf(env) * rf(env)
            if op == "/":
                return lambda env: lf(env) / rf(env)
            if op == "^":
                return lambda env: lf(env) ** rf(env)
            raise Unsupported(op)
        case FuncCall(name=name, args=args):
            fn = _builtin(name, len(args))
            arg_fns = [build(arg) for arg in args]
            if len(arg_fns) == 1:
                only = arg_fns[0]
                return lambda env: fn(only(env))
            return lambda env: fn(*(f(env) for f in arg_fns))
        case Guard(value=value, original=original):
            vf, of = build(value), build(original)
            return lambda env: vf(env) if gmpy2.is_finite(of(env)) else _NAN
        case Bounded(value=value, coordinate=coordinate, interval=interval):
            vf, cf = build(value), build(coordinate)
            assert isinstance(interval.lo, Num) and isinstance(interval.hi, Num)
            lo = literal(interval.lo.value, bits)
            hi = literal(interval.hi.value, bits)
            closed_lo, closed_hi = interval.closed_lo, interval.closed_hi

            def bounded(env: Mapping[str, Mpfr]) -> Mpfr:
                t = cf(env)
                inside = (t >= lo if closed_lo else t > lo) and (t <= hi if closed_hi else t < hi)
                return vf(env) if inside else _NAN

            return bounded
        case Reduce(kind="int"):
            return _float_fallback(node)
        case Reduce(kind=kind, var=var, body=body, lower=lower, upper=upper):
            return _reduction(kind == "prod", var, build(body), build(lower), build(upper))
    raise Unsupported(type(node).__name__)


def _reduction(
    product: bool, var: str, body: ScalarFn, lower: ScalarFn, upper: ScalarFn
) -> ScalarFn:
    def reduce(env: Mapping[str, Mpfr]) -> Mpfr:
        lo, hi = gmpy2.rint(lower(env)), gmpy2.rint(upper(env))
        if not (gmpy2.is_finite(lo) and gmpy2.is_finite(hi)):
            return _NAN
        start, stop = int(lo), int(hi)
        if stop - start + 1 > MAX_TERMS:
            return _NAN
        total = gmpy2.mpfr(1 if product else 0)
        scope = dict(env)
        for k in range(start, stop + 1):
            if k % 1024 == 0:
                checkpoint()
            scope[var] = gmpy2.mpfr(k)
            term = body(scope)
            total = total * term if product else total + term
        return total

    return reduce


def _float_fallback(node: Node) -> ScalarFn:
    """∫ has no precise quadrature: the float evaluator's value at float arguments."""
    from opengraph.engine.evaluator import NumericFunction

    numeric = NumericFunction(node, precise_constants=False)

    def evaluate(env: Mapping[str, Mpfr]) -> Mpfr:
        return gmpy2.mpfr(numeric.scalar(**{k: float(v) for k, v in env.items()}))

    return evaluate


def fold(node: Node, bits: int = FOLD_BITS) -> float | None:
    """The float64 value of a variable-free ``node``, computed in ``bits`` and rounded
    once; None if it has no precise real form (then the float evaluator decides)."""
    try:
        return float(constant_value(node, bits))
    except (Unsupported, ValueError, TypeError, ZeroDivisionError, OverflowError):
        return None


# -- the register program for the native machine --------------------------------------------


class Op(IntEnum):
    """Opcodes of :mod:`opengraph.native._mpvm` (keep the numbering in sync)."""

    ADD = 1
    SUB = 2
    MUL = 3
    DIV = 4
    NEG = 5
    POW = 6
    ODDPOW = 7  # c: 1 if the root's numerator is odd (the result keeps the sign)
    SIN = 10
    COS = 11
    TAN = 12
    SEC = 13
    CSC = 14
    COT = 15
    ASIN = 16
    ACOS = 17
    ATAN = 18
    ATAN2 = 19
    SINH = 20
    COSH = 21
    TANH = 22
    ASINH = 23
    ACOSH = 24
    ATANH = 25
    SQRT = 26
    CBRT = 27
    ABS = 28
    EXP = 29
    LN = 30
    LOG10 = 31
    GAMMA = 32
    FLOOR = 33
    CEIL = 34
    SIGN = 35
    MOD = 36
    MIN = 37
    MAX = 38
    GUARD = 40  # a: value, b: original
    BOUND = 41  # a: value, b: coordinate, c: lo, d: hi, e: closed flags (1 lo, 2 hi)


_UNARY_OPS = {
    "sin": Op.SIN,
    "cos": Op.COS,
    "tan": Op.TAN,
    "sec": Op.SEC,
    "csc": Op.CSC,
    "cot": Op.COT,
    "asin": Op.ASIN,
    "acos": Op.ACOS,
    "atan": Op.ATAN,
    "sinh": Op.SINH,
    "cosh": Op.COSH,
    "tanh": Op.TANH,
    "asinh": Op.ASINH,
    "acosh": Op.ACOSH,
    "atanh": Op.ATANH,
    "sqrt": Op.SQRT,
    "cbrt": Op.CBRT,
    "abs": Op.ABS,
    "exp": Op.EXP,
    "ln": Op.LN,
    "Gamma": Op.GAMMA,
    "floor": Op.FLOOR,
    "ceil": Op.CEIL,
    "sign": Op.SIGN,
}


@dataclass(frozen=True)
class Program:
    """Straight-line register code. Registers ``0 .. len(constants)-1`` hold constants,
    the next ``len(variables)`` the inputs; ``code`` rows are ``op, dst, a, b, c, d, e``."""

    variables: tuple[str, ...]
    constants: tuple[Mpfr, ...]
    code: NDArray[np.int32]
    registers: int
    result: int
    bits: int


Ref = tuple[str, int]  # ("c", constant), ("i", input) or ("t", temporary)
_UNUSED: Ref = ("c", -1)


@lru_cache(maxsize=256)
def compile_program(node: Node, variables: tuple[str, ...], bits: int) -> Program:
    """Translate ``node`` into a :class:`Program`; raises :class:`Unsupported` for nodes
    the machine lacks (Σ/Π/∫ with variable bounds, complex values).

    Variable-free sub-trees are folded into constant registers (with
    :func:`scalar_closure`, which also handles their Σ/Π), so only the parts that vary
    per sample run per sample.
    """
    constants: list[Mpfr] = []
    rows: list[tuple[int, Ref, Ref, Ref, Ref, Ref, int]] = []
    seen: dict[Node, Ref] = {}

    def constant(value: Mpfr) -> Ref:
        constants.append(value)
        return ("c", len(constants) - 1)

    def emit(
        op: Op,
        a: Ref = _UNUSED,
        b: Ref = _UNUSED,
        c: Ref = _UNUSED,
        d: Ref = _UNUSED,
        flags: int = 0,
    ) -> Ref:
        dst: Ref = ("t", len(rows))
        rows.append((int(op), dst, a, b, c, d, flags))
        return dst

    def folded(node: Node) -> Mpfr:
        return constant_value(node, bits)

    def build(node: Node) -> Ref:
        ref = seen.get(node)
        if ref is None:
            ref = seen[node] = constant(folded(node)) if not _free(node) else build_varying(node)
        return ref

    def build_varying(node: Node) -> Ref:
        match node:
            case Var(name=name):
                if name not in variables:
                    raise Unsupported(f"unbound {name}")
                return ("i", variables.index(name))
            case UnaryOp(operand=operand):
                return emit(Op.NEG, build(operand))
            case BinOp(op="^", left=left, right=right) if not _free(right):
                exponent = folded(right)
                form = _exponent_form(float(exponent))
                if form is not None:
                    return emit(Op.ODDPOW, build(left), constant(exponent), flags=int(form[1]))
                integral = float(exponent)
                if math.isfinite(integral) and integral == round(integral):
                    exponent = mpfr(int(integral), bits)
                return emit(Op.POW, build(left), constant(exponent))
            case BinOp(op=op, left=left, right=right):
                ops = {"+": Op.ADD, "-": Op.SUB, "*": Op.MUL, "/": Op.DIV, "^": Op.POW}
                if op not in ops:
                    raise Unsupported(op)
                return emit(ops[op], build(left), build(right))
            case FuncCall(name=name, args=args):
                return call(name, [build(arg) for arg in args])
            case Guard(value=value, original=original):
                return emit(Op.GUARD, build(value), build(original))
            case Bounded(value=value, coordinate=coordinate, interval=interval):
                assert isinstance(interval.lo, Num) and isinstance(interval.hi, Num)
                lo = constant(literal(interval.lo.value, bits))
                hi = constant(literal(interval.hi.value, bits))
                flags = (1 if interval.closed_lo else 0) | (2 if interval.closed_hi else 0)
                return emit(Op.BOUND, build(value), build(coordinate), lo, hi, flags)
        raise Unsupported(type(node).__name__)

    def gamma_of_successor(arg: Ref) -> Ref:
        return emit(Op.GAMMA, emit(Op.ADD, arg, constant(mpfr(1, bits))))

    def call(name: str, args: list[Ref]) -> Ref:
        if len(args) == 1:
            (a,) = args
            if name in _UNARY_OPS:
                return emit(_UNARY_OPS[name], a)
            if name == "factorial":
                return gamma_of_successor(a)
            if name == "round":
                return emit(Op.FLOOR, emit(Op.ADD, a, constant(mpfr("0.5", bits))))
            if name == "log":
                return emit(Op.LOG10, a)
        if len(args) == 2:
            a, b = args
            if name == "atan":
                return emit(Op.ATAN2, a, b)
            if name == "log":
                return emit(Op.DIV, emit(Op.LN, a), emit(Op.LN, b))
            if name == "mod":
                return emit(Op.MOD, a, b)
            if name == "nPr":
                return emit(Op.DIV, gamma_of_successor(a), gamma_of_successor(emit(Op.SUB, a, b)))
            if name == "nCr":
                rest = emit(Op.MUL, gamma_of_successor(b), gamma_of_successor(emit(Op.SUB, a, b)))
                return emit(Op.DIV, gamma_of_successor(a), rest)
        if name in ("min", "max") and len(args) >= 2:
            op = Op.MIN if name == "min" else Op.MAX
            result = args[0]
            for arg in args[1:]:
                result = emit(op, result, arg)
            return result
        raise Unsupported(name)

    result = build(node)
    base = {"c": 0, "i": len(constants), "t": len(constants) + len(variables)}

    def index(ref: Ref) -> int:
        kind, i = ref
        return 0 if i < 0 else base[kind] + i

    code = np.array(
        [
            (op, index(dst), index(a), index(b), index(c), index(d), flags)
            for op, dst, a, b, c, d, flags in rows
        ],
        dtype=np.int32,
    ).reshape(-1, 7)
    return Program(
        variables=variables,
        constants=tuple(constants),
        code=code,
        registers=base["t"] + len(rows),
        result=index(result),
        bits=bits,
    )


# -- evaluating arrays ------------------------------------------------------------------------


class PreciseFunction:
    """``node`` evaluated at ``origin + offset`` per variable, in MPFR.

    :meth:`evaluate` returns ``float64(f(origin + offset) - out_origin)`` for arrays of
    float64 offsets: everything large stays in the exact origins, so the float64 results
    keep their full relative precision however deep the zoom.
    """

    def __init__(self, node: Node) -> None:
        self.node = node

    def evaluate(
        self,
        bits: int,
        inputs: Mapping[str, tuple[Mpfr, NDArray[np.float64] | float]],
        out_origin: Mpfr = 0,
    ) -> NDArray[np.float64]:
        names = tuple(sorted(inputs))
        offsets = [np.asarray(inputs[name][1], dtype=np.float64) for name in names]
        shape = np.broadcast_shapes(*(o.shape for o in offsets)) if offsets else ()
        flat = [np.ascontiguousarray(np.broadcast_to(o, shape)).ravel() for o in offsets]
        origins = [mpfr(inputs[name][0], bits) for name in names]
        out = np.empty(int(np.prod(shape)), dtype=np.float64)
        out_ref = mpfr(out_origin, bits)
        runner = _native_runner(self.node, names, bits)
        if runner is not None:
            locals_ = np.ascontiguousarray(np.stack(flat) if flat else np.empty((0, out.size)))
            for start in range(0, out.size, CHUNK):
                checkpoint()
                runner(origins, locals_, out_ref, out, start, min(out.size, start + CHUNK))
            return out.reshape(shape)
        fn = scalar_closure(self.node, bits)
        with context(bits):
            for i in range(out.size):
                if i % CHUNK == 0:
                    checkpoint()
                env = {
                    name: origin + float(f[i])
                    for name, origin, f in zip(names, origins, flat, strict=True)
                }
                out[i] = float(fn(env) - out_ref)
        return out.reshape(shape)


def _native_runner(node: Node, names: tuple[str, ...], bits: int) -> Callable[..., None] | None:
    from opengraph import native

    if native.mpvm is None:
        return None
    try:
        program = compile_program(node, names, bits)
    except Unsupported:
        return None

    def run(
        origins: Sequence[Mpfr],
        locals_: NDArray[np.float64],
        out_origin: Mpfr,
        out: NDArray[np.float64],
        start: int,
        stop: int,
    ) -> None:
        assert native.mpvm is not None
        native.mpvm.run(
            program.code,
            list(program.constants),
            list(origins),
            locals_,
            program.registers,
            program.result,
            out_origin,
            program.bits,
            out,
            start,
            stop,
        )

    return run


def supports(node: Node) -> bool:
    """Whether ``node`` has a real-valued precise form."""
    try:
        scalar_closure(node, 64)
    except Unsupported:
        return False
    return True
