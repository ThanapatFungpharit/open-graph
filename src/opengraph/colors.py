"""Colours as hex strings: ``#rrggbb``, or ``#aarrggbb`` when they are translucent.

That is the notation of the document (expression colours) and of Flet. The helpers
here parse, blend and compare colours without any GUI toolkit.
"""

from __future__ import annotations

import colorsys
import re

_HEX = re.compile(r"#?([0-9a-fA-F]{3}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})")

RGBA = tuple[int, int, int, int]


def is_valid(text: str) -> bool:
    return _HEX.fullmatch(text.strip()) is not None


def rgba(color: str) -> RGBA:
    """``(red, green, blue, alpha)`` of ``#rgb``, ``#rrggbb`` or ``#aarrggbb``."""
    match = _HEX.fullmatch(color.strip())
    if match is None:
        raise ValueError(f"Not a colour: {color!r}")
    digits = match.group(1)
    if len(digits) == 3:
        digits = "".join(d * 2 for d in digits)
    if len(digits) == 6:
        digits = "ff" + digits
    a, r, g, b = (int(digits[i : i + 2], 16) for i in range(0, 8, 2))
    return r, g, b, a


def from_rgba(r: int, g: int, b: int, a: int = 255) -> str:
    clamp = [max(0, min(255, round(v))) for v in (a, r, g, b)]
    if clamp[0] == 255:
        return "#{:02x}{:02x}{:02x}".format(*clamp[1:])
    return "#{:02x}{:02x}{:02x}{:02x}".format(*clamp)


def name(color: str) -> str:
    """The opaque ``#rrggbb`` form of ``color``."""
    r, g, b, _ = rgba(color)
    return from_rgba(r, g, b)


def with_alpha(color: str, alpha: int) -> str:
    """``color`` with opacity ``alpha`` (0-255)."""
    r, g, b, _ = rgba(color)
    return from_rgba(r, g, b, alpha)


def alpha(color: str) -> int:
    return rgba(color)[3]


def hsl(color: str) -> tuple[float, float, float, float]:
    """Hue, saturation, lightness and alpha, each 0-1."""
    r, g, b, a = rgba(color)
    h, lightness, s = colorsys.rgb_to_hls(r / 255, g / 255, b / 255)
    return h, s, lightness, a / 255


def from_hsl(h: float, s: float, lightness: float, a: float = 1.0) -> str:
    r, g, b = colorsys.hls_to_rgb(h, lightness, s)
    return from_rgba(round(r * 255), round(g * 255), round(b * 255), round(a * 255))


def _luminance(color: str) -> float:
    def channel(value: int) -> float:
        v = value / 255
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4

    r, g, b, _ = rgba(color)
    return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)


def contrast(a: str, b: str) -> float:
    """WCAG contrast ratio between two colours (1 to 21)."""
    la, lb = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)
