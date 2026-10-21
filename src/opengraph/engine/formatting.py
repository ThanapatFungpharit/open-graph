"""Human-readable number formatting shared by the engine and the UI."""

from __future__ import annotations

import math


def format_value(value: float, digits: int = 10) -> str:
    """Format a computed value compactly: ``3``, ``0.333333333``, ``1.5e-12``."""
    if math.isnan(value):
        return "undefined"
    if math.isinf(value):
        return "∞" if value > 0 else "-∞"
    if value == 0:
        return "0"  # also normalises -0.0
    text = f"{value:.{digits}g}"
    if "e" in text:
        mantissa, exponent = text.split("e")
        return f"{mantissa}e{int(exponent)}"
    return text


def format_complex(value: complex, digits: int = 10) -> str:
    """``3``, ``2i``, ``1 + 2i``, ``-i``; parts below rounding noise are dropped."""
    re, im = value.real, value.imag
    if math.isnan(re) or math.isnan(im):
        return "undefined"
    size = max(abs(re), abs(im))
    if abs(im) <= 1e-12 * size:
        return format_value(re, digits)
    imag = format_value(abs(im), digits)
    imag = "i" if imag == "1" else f"{imag}i"
    if abs(re) <= 1e-12 * size:
        return f"-{imag}" if im < 0 else imag
    return f"{format_value(re, digits)} {'-' if im < 0 else '+'} {imag}"
