"""The pure-Python stand-in for gmpy2, against the real library.

The rest of the suite covers it through the engine: CI runs it with OPENGRAPH_NO_GMPY2=1.
"""

from __future__ import annotations

import math
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from opengraph.engine import mpfr_fallback as fallback

gmpy2: Any = pytest.importorskip("gmpy2")

BITS = 256
UNARY = (
    "sin cos tan sec csc cot asin acos atan sinh cosh tanh asinh acosh atanh "
    "sqrt cbrt exp log log10 gamma floor ceil rint rint_floor rint_ceil"
).split()
VALUES = [
    0.0,
    1.0,
    -1.0,
    0.5,
    -0.5,
    2.0,
    -2.0,
    2.5,
    3.5,
    -3.5,
    10.0,
    1e-20,
    -1e-20,
    1e20,
    -3.0,
    0.1,
    123.456,
    math.inf,
    -math.inf,
    math.nan,
]


def real_context(bits: int) -> Any:
    """gmpy2's own context, whichever library the engine itself is running on."""
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


def real(function: str, *args: float) -> Any:
    with real_context(BITS):
        return getattr(gmpy2, function)(*(gmpy2.mpfr(a) for a in args))


def stand_in(function: str, *args: float) -> Any:
    with fallback.context(BITS):
        return getattr(fallback, function)(*(fallback.mpfr(a) for a in args))


def assert_same(a: Any, b: Any, what: str) -> None:
    """Equal to within a few of the last bits, as a difference at full precision."""
    x, y = float(a), float(b)
    if math.isnan(x) or math.isnan(y):
        assert math.isnan(x) and math.isnan(y), f"{what}: {a} vs {b}"
        return
    if math.isinf(x) or math.isinf(y):
        assert x == y, f"{what}: {a} vs {b}"
        return
    with real_context(BITS + 64):
        difference = abs(gmpy2.mpfr(format(b, ".100e")) - a)
        limit = max(abs(a), gmpy2.mpfr(1e-300)) * gmpy2.mpfr(2) ** -(BITS - 8)
        assert difference <= limit, f"{what}: {a} vs {b}"


@pytest.mark.parametrize("function", UNARY)
@pytest.mark.parametrize("value", VALUES)
def test_functions_agree_with_mpfr(function: str, value: float) -> None:
    if function in ("floor", "ceil") and not math.isfinite(value):
        pytest.skip("gmpy2 gives an integer here, which has no infinity or NaN")
    assert_same(real(function, value), stand_in(function, value), f"{function}({value})")


@pytest.mark.parametrize("function", ["atan2", "fmod"])
def test_two_argument_functions_agree_with_mpfr(function: str) -> None:
    for a in (0.5, -0.5, 7.25, -7.25, 1e10, 0.0):
        for b in (0.5, -0.5, 3.0, -3.0, 0.1):
            assert_same(real(function, a, b), stand_in(function, a, b), f"{function}({a}, {b})")


@pytest.mark.parametrize(
    ("a", "b"),
    [(0.0, 0.0), (1.0, 0.0), (-1.0, 0.0), (0.0, -1.0), (0.0, 2.0), (2.0, -1.0), (-8.0, 0.5)],
)
def test_edge_cases_of_arithmetic_agree_with_mpfr(a: float, b: float) -> None:
    with real_context(BITS):
        x, y = gmpy2.mpfr(a), gmpy2.mpfr(b)
        real_results = {"/": x / y, "**": x**y, "+": x + y, "-": x - y, "*": x * y}
    with fallback.context(BITS):
        p, q = fallback.mpfr(a), fallback.mpfr(b)
        ours = {"/": p / q, "**": p**q, "+": p + q, "-": p - q, "*": p * q}
    for op, value in real_results.items():
        assert_same(value, ours[op], f"{a} {op} {b}")


def test_random_arithmetic_agrees_with_mpfr() -> None:
    rng = random.Random(5)
    for _ in range(300):
        a, b = rng.uniform(-1e6, 1e6), rng.uniform(-1e3, 1e3)
        with real_context(BITS):
            expected = (gmpy2.mpfr(a) + b) * gmpy2.mpfr(b) / (1 + abs(gmpy2.mpfr(a)))
        with fallback.context(BITS):
            ours = (fallback.mpfr(a) + b) * fallback.mpfr(b) / (1 + abs(fallback.mpfr(a)))
        assert_same(expected, ours, f"{a}, {b}")


def test_numbers_are_exact_past_float64() -> None:
    with fallback.context(BITS):
        big = fallback.mpfr("10000.000000000000000000000000000001")
        assert float(big - 10000) == pytest.approx(1e-30)
        assert int(fallback.floor(big)) == 10000
        assert int(fallback.ceil(big)) == 10001


def test_formatting_matches_mpfr() -> None:
    text = "1234.56789012345678901234567890123"
    with real_context(BITS):
        real_value = gmpy2.mpfr(text)
    with fallback.context(BITS):
        ours = fallback.mpfr(text)
    assert f"{ours:.20f}" == f"{real_value:.20f}"
    assert float(f"{ours:.25e}") == float(f"{real_value:.25e}")


def test_comparisons_hashes_and_truth() -> None:
    with fallback.context(BITS):
        half, nan = fallback.mpfr(0.5), fallback.nan()
        assert half == 0.5 and half < 1 and half > 0 and 1 > half
        assert hash(half) == hash(0.5)  # usable in the same keys as a float
        assert not (nan == nan) and not (nan < 1) and not (nan > 1)  # noqa: PLR0124
        assert not fallback.mpfr(0) and half and nan
    assert fallback.is_nan(nan) and not fallback.is_finite(nan)
    assert fallback.is_integer(fallback.mpfr(3)) and not fallback.is_integer(half)
    assert fallback.sign(fallback.mpfr(-2)) == -1


def test_precision_belongs_to_each_thread() -> None:
    """Deep-zoom tiles are computed in parallel, with their own precision."""

    def third(bits: int) -> int:
        with fallback.context(bits):
            for _ in range(200):
                value = fallback.mpfr(1) / 3
            return int(value.raw[3])

    with ThreadPoolExecutor(4) as pool:
        for bits, used in zip([64, 512] * 8, pool.map(third, [64, 512] * 8), strict=True):
            assert used <= bits and used > bits - 8
