"""Grid spacing and axis label formatting.

Major grid steps are "nice" numbers (1, 2 or 5 times a power of ten) chosen so major
lines are at least ``min_px`` apart on screen. Minor lines subdivide each major step
into 5 (or 4 when the step starts with 2, giving 0.5-multiples).

Tick positions are generated as integer multiples ``k * step`` rather than by repeated
addition, so labels never accumulate floating-point drift.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from opengraph.engine.formatting import format_value

MAX_TICKS = 2000
_SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


@dataclass(frozen=True, slots=True)
class TickSpacing:
    major: float
    minor: float


def choose_spacing(px_per_unit: float, min_px: float = 90.0) -> TickSpacing:
    raw = min_px / px_per_unit
    exponent = math.floor(math.log10(raw))
    base = 10.0**exponent
    mantissa = next((m for m in (1, 2, 5) if m * base >= raw * (1 - 1e-9)), 10)
    major = mantissa * base
    return TickSpacing(major, major / (4 if mantissa == 2 else 5))


def tick_indices(lo: float, hi: float, step: float) -> NDArray[np.int64]:
    """Integers ``k`` with ``lo <= k*step <= hi`` (capped to a sane count)."""
    first, last = math.ceil(lo / step), math.floor(hi / step)
    if last - first > MAX_TICKS:
        return np.empty(0, dtype=np.int64)
    return np.arange(first, last + 1, dtype=np.int64)


def format_tick(value: Any, step: float) -> str:
    """Shortest label that distinguishes ticks spaced ``step`` apart.

    ``value`` may be an exact gmpy2 value (a deep zoom), whose digits go beyond float64.
    """
    if abs(value) < step * 1e-6:
        return "0"
    step_exp = math.floor(math.log10(step) + 1e-9)
    value_exp = math.floor(math.log10(abs(value)) + 1e-9)
    exact = not isinstance(value, float | int)
    if exact and -1 <= value_exp < 6:  # a deep zoom near an ordinary number: 4.000…0002
        return f"{value:.{max(0, -step_exp)}f}".replace("-", "−")
    if value_exp >= 6 or step_exp <= -5:
        digits = max(0, value_exp - step_exp)
        mantissa, exp = f"{value:.{digits}e}".split("e")
        return f"{mantissa.replace('-', '−')}×10{str(int(exp)).translate(_SUPERSCRIPT)}"
    decimals = max(0, -step_exp)
    return f"{value:.{decimals}f}".replace("-", "−")


MIN_DECADE_PX = 40.0  # below this, decades are skipped for major ticks
LOG_LINEAR_SPAN = 0.3  # decades: narrower log views use ordinary linear ticks


def log_ticks(u_min: float, u_max: float, px_per_decade: float) -> tuple[list[float], list[float]]:
    """``(major, minor)`` tick values (world units) for a log axis showing ``10^u_min ..
    10^u_max``. Majors are powers of ten (every n-th decade when they are crowded, plus
    2 and 5 times them when a decade is wide); minors are the other integer multiples.
    """
    if not (math.isfinite(u_min) and math.isfinite(u_max)) or u_max <= u_min:
        return [], []
    first, last = math.floor(u_min), math.ceil(u_max)
    if last - first > MAX_TICKS:
        return [], []
    every = 1
    if px_per_decade < MIN_DECADE_PX:
        needed = MIN_DECADE_PX / px_per_decade
        every = next(n for n in (2, 5, 10, 20, 50, 100, 200, 500, 1000) if n >= needed)
    lo, hi = 10.0**u_min, 10.0**u_max
    majors: list[float] = []
    minors: list[float] = []
    wide = px_per_decade >= 6 * MIN_DECADE_PX
    for k in range(first, last + 1):
        base = 10.0**k
        if k % every == 0 and lo <= base <= hi:
            majors.append(base)
        if every > 1:
            if k % every != 0 and lo <= base <= hi and px_per_decade >= 6:
                minors.append(base)
            continue
        for m in range(2, 10):
            value = m * base
            if lo <= value <= hi:
                (majors if wide and m in (2, 5) else minors).append(value)
    return sorted(majors), minors


def format_log_tick(value: float) -> str:
    """``0.01``, ``2``, ``500``; ``10⁶`` / ``10⁻⁵`` for powers of ten far from 1."""
    exponent = round(math.log10(value))
    if math.isclose(value, 10.0**exponent, rel_tol=1e-9) and (exponent >= 5 or exponent <= -4):
        return "10" + str(exponent).translate(_SUPERSCRIPT)
    return format_value(value, 6).replace("-", "−")


def format_coordinate(value: Any, units_per_px: float) -> str:
    """Format a traced coordinate to the precision one screen pixel represents.

    ``units_per_px <= 0`` means the value is exact (e.g. a user-entered point). An exact
    gmpy2 ``value`` (a deep zoom) shows as many digits as a pixel resolves.
    """
    if not isinstance(value, float | int) and units_per_px > 0 and math.isfinite(value):
        return _format_exact(value, units_per_px)
    value = float(value)
    if units_per_px <= 0 or not math.isfinite(value) or abs(value) >= 1e7:
        return format_value(value)
    decimals = min(max(math.ceil(-math.log10(units_per_px)) + 1, 0), 12)
    text = f"{value:.{decimals}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def _format_exact(value: Any, units_per_px: float) -> str:
    if value == 0:
        return "0"
    magnitude = math.floor(math.log10(abs(value)))
    resolution = math.ceil(-math.log10(units_per_px)) + 1  # decimals a pixel resolves
    if magnitude >= 7:
        mantissa, exponent = f"{value:.{max(1, magnitude + resolution)}e}".split("e")
        mantissa = mantissa.rstrip("0").rstrip(".")
        return f"{mantissa}e{int(exponent)}"
    text = f"{value:.{max(0, resolution)}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text
