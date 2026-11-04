"""The part of gmpy2's interface OpenGraph uses, in pure Python on mpmath's ``libmp``.

gmpy2 has no wheels for Android and iOS. mpmath comes with SymPy, so it is always there,
and its ``libmp`` layer has correctly rounded arithmetic that takes the precision as an
argument. Unlike ``mpmath.mp``, that keeps worker threads with different precisions
apart. Results agree with MPFR to within the last few bits (tests compare the two); it
is only slower, which deep zoom notices and little else does.

Semantics follow gmpy2 under :func:`opengraph.engine.precise.context`: nothing raises;
what is undefined is NaN, and a pole is an infinity. There is no negative zero.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from decimal import Decimal, localcontext
from fractions import Fraction
from typing import Any

from mpmath import libmp

Raw = Any  # libmp's tuple (sign, mantissa, exponent, bit count); libmp has no types

_ROUND = libmp.round_nearest
_DEFAULT_BITS = 53
_bits: ContextVar[int] = ContextVar("mpfr_bits", default=_DEFAULT_BITS)


@contextmanager
def context(precision: int = _DEFAULT_BITS, **_flags: object) -> Iterator[None]:
    """Compute with ``precision`` bits, in this thread (gmpy2's contexts never raise here)."""
    token = _bits.set(precision)
    try:
        yield
    finally:
        _bits.reset(token)


def _raw(value: Any) -> Raw | None:
    """``value`` as an exact raw number; None if it isn't a kind of number we know."""
    if isinstance(value, mpfr):
        return value.raw
    if isinstance(value, bool | int):
        return libmp.from_int(int(value))
    if isinstance(value, float):  # numpy.float64 too
        return libmp.from_float(float(value))
    if isinstance(value, Fraction):
        return libmp.from_rational(value.numerator, value.denominator, _bits.get(), _ROUND)
    if hasattr(value, "__float__") and not isinstance(value, str):  # other numpy scalars
        return libmp.from_float(float(value))
    return None


def _wrap(raw: Raw) -> mpfr:
    result = object.__new__(mpfr)
    result.raw = raw
    return result


def _guarded(compute: Callable[[], Raw], at_pole: Callable[[], Raw] = lambda: libmp.fnan) -> mpfr:
    """The result of ``compute``, with what libmp raises turned into what MPFR returns."""
    try:
        return _wrap(compute())
    except libmp.ComplexResult:
        return _wrap(libmp.fnan)
    except ZeroDivisionError:
        return _wrap(at_pole())
    except (ValueError, OverflowError, NotImplementedError):
        return _wrap(libmp.fnan)


def _is_nan(raw: Raw) -> bool:
    return bool(raw == libmp.fnan)


def _is_infinite(raw: Raw) -> bool:
    return bool(raw in (libmp.finf, libmp.fninf))


def _negative(raw: Raw) -> bool:
    return bool(raw != libmp.fnan and raw[0] == 1)


class mpfr:  # noqa: N801 - named like the gmpy2 type it stands in for
    __slots__ = ("raw",)
    raw: Raw

    def __init__(self, value: Any = 0) -> None:
        bits = _bits.get()
        if isinstance(value, str):
            self.raw = _from_text(value, bits)
            return
        raw = _raw(value)
        if raw is None:
            raise TypeError(f"mpfr() can't convert {type(value).__name__}")
        self.raw = libmp.mpf_pos(raw, bits, _ROUND)

    # -- arithmetic ----------------------------------------------------------------------
    def _binary(self, other: Any, operation: Callable[[Raw, Raw], mpfr]) -> Any:
        raw = _raw(other)
        return NotImplemented if raw is None else operation(self.raw, raw)

    def _reflected(self, other: Any, operation: Callable[[Raw, Raw], mpfr]) -> Any:
        raw = _raw(other)
        return NotImplemented if raw is None else operation(raw, self.raw)

    def __add__(self, other: Any) -> Any:
        return self._binary(other, _add)

    def __radd__(self, other: Any) -> Any:
        return self._reflected(other, _add)

    def __sub__(self, other: Any) -> Any:
        return self._binary(other, _sub)

    def __rsub__(self, other: Any) -> Any:
        return self._reflected(other, _sub)

    def __mul__(self, other: Any) -> Any:
        return self._binary(other, _mul)

    def __rmul__(self, other: Any) -> Any:
        return self._reflected(other, _mul)

    def __truediv__(self, other: Any) -> Any:
        return self._binary(other, _div)

    def __rtruediv__(self, other: Any) -> Any:
        return self._reflected(other, _div)

    def __pow__(self, other: Any) -> Any:
        return self._binary(other, _pow)

    def __rpow__(self, other: Any) -> Any:
        return self._reflected(other, _pow)

    def __neg__(self) -> mpfr:
        return _wrap(libmp.mpf_neg(self.raw, _bits.get(), _ROUND))

    def __pos__(self) -> mpfr:
        return _wrap(libmp.mpf_pos(self.raw, _bits.get(), _ROUND))

    def __abs__(self) -> mpfr:
        return _wrap(libmp.mpf_abs(self.raw, _bits.get(), _ROUND))

    # -- comparison ----------------------------------------------------------------------
    def _compare(self, other: Any, test: Callable[[int], bool]) -> Any:
        raw = _raw(other)
        if raw is None:
            return NotImplemented
        if _is_nan(self.raw) or _is_nan(raw):
            return False
        return test(libmp.mpf_cmp(self.raw, raw))

    def __eq__(self, other: object) -> Any:
        return self._compare(other, lambda order: order == 0)

    def __lt__(self, other: Any) -> Any:
        return self._compare(other, lambda order: order < 0)

    def __le__(self, other: Any) -> Any:
        return self._compare(other, lambda order: order <= 0)

    def __gt__(self, other: Any) -> Any:
        return self._compare(other, lambda order: order > 0)

    def __ge__(self, other: Any) -> Any:
        return self._compare(other, lambda order: order >= 0)

    def __hash__(self) -> int:
        return int(libmp.mpf_hash(self.raw))

    def __bool__(self) -> bool:
        return bool(self.raw != libmp.fzero)

    # -- conversion ----------------------------------------------------------------------
    def __float__(self) -> float:
        return float(libmp.to_float(self.raw, rnd=_ROUND))

    def __int__(self) -> int:
        if _is_nan(self.raw) or _is_infinite(self.raw):
            raise ValueError("can't convert an infinity or NaN to an integer")
        return int(libmp.to_int(self.raw))

    def __format__(self, spec: str) -> str:
        if _is_nan(self.raw) or _is_infinite(self.raw):
            return format(float(self), spec)
        sign, mantissa, exponent, _ = self.raw
        with localcontext() as exact:
            # Enough digits to hold any binary fraction of this size exactly.
            mantissa = int(mantissa)  # libmp's integers are gmpy2's where that is installed
            exact.prec = max(64, mantissa.bit_length() + abs(exponent) + 8)
            number = Decimal(mantissa) * Decimal(2) ** exponent
            return format(-number if sign else number, spec)

    def __str__(self) -> str:
        return str(libmp.to_str(self.raw, max(1, int(_bits.get() * 0.30103))))

    def __repr__(self) -> str:
        return f"mpfr('{self}')"


def _from_text(text: str, bits: int) -> Raw:
    cleaned = text.strip().lower()
    named = {"nan": libmp.fnan, "inf": libmp.finf, "+inf": libmp.finf, "-inf": libmp.fninf}
    if cleaned in named:
        return named[cleaned]
    return libmp.from_str(cleaned, bits, _ROUND)


def _add(a: Raw, b: Raw) -> mpfr:
    return _guarded(lambda: libmp.mpf_add(a, b, _bits.get(), _ROUND))


def _sub(a: Raw, b: Raw) -> mpfr:
    return _guarded(lambda: libmp.mpf_sub(a, b, _bits.get(), _ROUND))


def _mul(a: Raw, b: Raw) -> mpfr:
    return _guarded(lambda: libmp.mpf_mul(a, b, _bits.get(), _ROUND))


def _div(a: Raw, b: Raw) -> mpfr:
    if b == libmp.fzero:
        if a == libmp.fzero or _is_nan(a):
            return _wrap(libmp.fnan)
        return _wrap(libmp.fninf if _negative(a) else libmp.finf)
    return _guarded(lambda: libmp.mpf_div(a, b, _bits.get(), _ROUND))


def _pow(base: Raw, exponent: Raw) -> mpfr:
    if exponent == libmp.fzero:
        return _wrap(libmp.fone)
    if base == libmp.fzero:
        if _is_nan(exponent):
            return _wrap(libmp.fnan)
        return _wrap(libmp.finf if _negative(exponent) else libmp.fzero)
    return _guarded(lambda: libmp.mpf_pow(base, exponent, _bits.get(), _ROUND))


def _as_raw(value: Any) -> Raw:
    raw = value.raw if isinstance(value, mpfr) else mpfr(value).raw
    return raw


def _function(
    compute: Callable[[Raw, int, str], Raw], at_pole: Callable[[], Raw] = lambda: libmp.fnan
) -> Callable[[Any], mpfr]:
    def call(value: Any) -> mpfr:
        raw = _as_raw(value)
        if _is_nan(raw):
            return _wrap(libmp.fnan)
        return _guarded(lambda: compute(raw, _bits.get(), _ROUND), at_pole)

    return call


def _reciprocal(compute: Callable[[Raw, int, str], Raw]) -> Callable[[Any], mpfr]:
    def call(value: Any) -> mpfr:
        raw = _as_raw(value)
        bits = _bits.get()
        with context(bits + 16):
            denominator = _guarded(lambda: compute(raw, bits + 16, _ROUND))
        return _div(libmp.fone, denominator.raw) if bits else denominator

    return call


def _finite_only(compute: Callable[[Raw, int, str], Raw]) -> Callable[[Raw, int, str], Raw]:
    """Trigonometric functions of an infinity are NaN, not an exception."""

    def call(raw: Raw, bits: int, rounding: str) -> Raw:
        return libmp.fnan if _is_infinite(raw) else compute(raw, bits, rounding)

    return call


sin = _function(_finite_only(libmp.mpf_sin))
cos = _function(_finite_only(libmp.mpf_cos))
tan = _function(_finite_only(libmp.mpf_tan))
sec = _reciprocal(libmp.mpf_cos)
csc = _reciprocal(libmp.mpf_sin)
cot = _reciprocal(libmp.mpf_tan)
asin = _function(libmp.mpf_asin)
acos = _function(libmp.mpf_acos)
atan = _function(libmp.mpf_atan)
sinh = _function(libmp.mpf_sinh)
cosh = _function(libmp.mpf_cosh)
tanh = _function(libmp.mpf_tanh)
asinh = _function(libmp.mpf_asinh)
acosh = _function(libmp.mpf_acosh)
exp = _function(libmp.mpf_exp)


def _atanh(raw: Raw, bits: int, rounding: str) -> Raw:
    if raw == libmp.fone:
        return libmp.finf
    if raw == libmp.fnone:
        return libmp.fninf
    return libmp.mpf_atanh(raw, bits, rounding)


atanh = _function(_atanh)


def _sqrt(raw: Raw, bits: int, rounding: str) -> Raw:
    return libmp.fnan if _negative(raw) else libmp.mpf_sqrt(raw, bits, rounding)


sqrt = _function(_sqrt)


def _cbrt(raw: Raw, bits: int, rounding: str) -> Raw:
    if _is_infinite(raw) or raw == libmp.fzero:
        return raw
    if _negative(raw):  # MPFR's cube root is real for negative numbers too
        return libmp.mpf_neg(libmp.mpf_cbrt(libmp.mpf_neg(raw), bits, rounding))
    return libmp.mpf_cbrt(raw, bits, rounding)


cbrt = _function(_cbrt)


def _log(raw: Raw, bits: int, rounding: str) -> Raw:
    if raw == libmp.fzero:
        return libmp.fninf
    if _negative(raw):
        return libmp.fnan
    return libmp.mpf_log(raw, bits, rounding)


log = _function(_log)


def log10(value: Any) -> mpfr:
    raw = _as_raw(value)
    bits = _bits.get()
    with context(bits + 16):
        ratio = _div(_log(raw, bits + 16, _ROUND), libmp.mpf_log(libmp.from_int(10), bits + 16))
    return _wrap(libmp.mpf_pos(ratio.raw, bits, _ROUND))


def _gamma(raw: Raw, bits: int, rounding: str) -> Raw:
    if raw == libmp.finf:
        return raw
    if raw == libmp.fninf:
        return libmp.fnan
    if raw == libmp.fzero:
        return libmp.finf
    return libmp.mpf_gamma(raw, bits, rounding)


gamma = _function(_gamma)  # a pole at a negative integer is NaN, as in MPFR


def atan2(y: Any, x: Any) -> mpfr:
    a, b = _as_raw(y), _as_raw(x)
    if _is_nan(a) or _is_nan(b):
        return _wrap(libmp.fnan)
    if _is_infinite(a) or _is_infinite(b):
        return mpfr(math.atan2(float(_wrap(a)), float(_wrap(b))))
    if a == libmp.fzero and b == libmp.fzero:
        return _wrap(libmp.fzero)
    return _guarded(lambda: libmp.mpf_atan2(a, b, _bits.get(), _ROUND))


def nan() -> mpfr:
    return _wrap(libmp.fnan)


def const_pi() -> mpfr:
    return _wrap(libmp.mpf_pi(_bits.get(), _ROUND))


def is_nan(value: Any) -> bool:
    return _is_nan(_as_raw(value))


def is_finite(value: Any) -> bool:
    raw = _as_raw(value)
    return not (_is_nan(raw) or _is_infinite(raw))


def is_integer(value: Any) -> bool:
    raw = _as_raw(value)
    return is_finite(value) and (raw == libmp.fzero or raw[2] >= 0)


def sign(value: Any) -> int:
    raw = _as_raw(value)
    if _is_nan(raw) or raw == libmp.fzero:
        return 0
    return -1 if _negative(raw) else 1


def _whole(compute: Callable[[Raw, int, str], Raw]) -> Callable[[Any], mpfr]:
    def call(value: Any) -> mpfr:
        raw = _as_raw(value)
        if not is_finite(value):
            return _wrap(raw)
        # All of a whole number's bits matter, whatever the working precision is.
        exact = max(_bits.get(), raw[3] + abs(raw[2]) + 2)
        whole = compute(raw, exact, libmp.round_floor)
        return _wrap(libmp.mpf_pos(whole, _bits.get(), _ROUND))

    return call


floor = _whole(libmp.mpf_floor)
ceil = _whole(libmp.mpf_ceil)
rint_floor = floor
rint_ceil = ceil
rint = _whole(libmp.mpf_nint)  # ties to even, as in MPFR's default rounding


def fmod(a: Any, b: Any) -> mpfr:
    """The remainder with the sign of ``a``, like C's fmod."""
    x, y = _as_raw(a), _as_raw(b)
    if _is_nan(x) or _is_nan(y) or _is_infinite(x) or y == libmp.fzero:
        return _wrap(libmp.fnan)
    if _is_infinite(y):
        return _wrap(libmp.mpf_pos(x, _bits.get(), _ROUND))
    magnitude = libmp.mpf_abs(y)
    remainder = libmp.mpf_mod(libmp.mpf_abs(x), magnitude, _bits.get(), _ROUND)
    return _wrap(libmp.mpf_neg(remainder) if _negative(x) else remainder)
