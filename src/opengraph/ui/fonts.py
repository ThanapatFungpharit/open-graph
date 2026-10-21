"""The bundled fonts, and their metrics measured in Python.

Flet draws text on the client and cannot report how wide it is, so everything that is
laid out by hand (the typeset editor, canvas labels and tooltips) uses fonts shipped in
``assets/fonts`` (DejaVu, see ``LICENSE-DejaVu.txt``): the client draws them, and
fontTools reads the same files here for advances, ascents and ink bounds, so positions
computed in Python match what is drawn.

Sizes are in logical pixels. Kerning is ignored: the editor places every glyph itself.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from functools import cache
from pathlib import Path

from fontTools.ttLib import TTFont

from opengraph.ui.geometry import Rect

ASSETS_DIR = Path(__file__).resolve().parent / "assets"

SERIF = "OpenGraph Serif"
SERIF_ITALIC = "OpenGraph Serif Italic"
SANS = "OpenGraph Sans"
MONO = "OpenGraph Mono"

FILES = {
    SERIF: "fonts/DejaVuSerif.ttf",
    SERIF_ITALIC: "fonts/DejaVuSerif-Italic.ttf",
    SANS: "fonts/DejaVuSans.ttf",
    MONO: "fonts/DejaVuSansMono.ttf",
}

# Pixel sizes used across the interface (12 pt and 8.5 pt at 96 dpi).
EDITOR_SIZE = 16.0
LABEL_SIZE = 11.5


class Face:
    """Metrics of one font file, in ems."""

    def __init__(self, path: Path) -> None:
        font = TTFont(path, lazy=True)
        units = float(font["head"].unitsPerEm)
        hhea = font["hhea"]
        self.ascent = float(hhea.ascent) / units
        self.descent = -float(hhea.descent) / units
        self.line_gap = float(hhea.lineGap) / units
        self._units = units
        self._font = font
        self._cmap: dict[int, str] = font.getBestCmap()
        self._metrics: dict[str, tuple[int, int]] = font["hmtx"].metrics
        self.x_height = self._top("x", 0.5)
        self.cap_height = self._top("H", 0.7)

    def _glyph_bounds(self, char: str) -> tuple[float, float, float, float] | None:
        name = self._cmap.get(ord(char))
        if name is None:
            return None
        glyph = self._font["glyf"][name]
        if not hasattr(glyph, "xMin"):  # no outline (a space)
            return None
        u = self._units
        return glyph.xMin / u, glyph.yMin / u, glyph.xMax / u, glyph.yMax / u

    def _top(self, char: str, default: float) -> float:
        bounds = self._glyph_bounds(char)
        return default if bounds is None else bounds[3]

    @cache  # noqa: B019 - faces live for the whole process
    def advance(self, char: str) -> float:
        name = self._cmap.get(ord(char))
        if name is None:
            # Drawn with a fallback font on the client: a typical width is the best guess.
            if unicodedata.combining(char):
                return 0.0
            return 1.0 if unicodedata.east_asian_width(char) in ("W", "F") else 0.6
        return self._metrics[name][0] / self._units

    def ink(self, text: str) -> tuple[float, float, float, float]:
        """``(left, bottom, right, top)`` of the ink of ``text`` (y up, in ems)."""
        x = 0.0
        left = bottom = float("inf")
        right = top = float("-inf")
        for char in text:
            bounds = self._glyph_bounds(char)
            if bounds is not None:
                left, bottom = min(left, x + bounds[0]), min(bottom, bounds[1])
                right, top = max(right, x + bounds[2]), max(top, bounds[3])
            x += self.advance(char)
        if left == float("inf"):
            return 0.0, 0.0, 0.0, 0.0
        return left, bottom, right, top


@cache
def face(family: str) -> Face:
    return Face(ASSETS_DIR / FILES[family])


@dataclass(frozen=True, slots=True)
class Font:
    """A bundled family at a pixel size."""

    family: str
    size: float

    @property
    def italic(self) -> bool:
        return self.family == SERIF_ITALIC

    def ascent(self) -> float:
        return face(self.family).ascent * self.size

    def descent(self) -> float:
        return face(self.family).descent * self.size

    def height(self) -> float:
        return self.ascent() + self.descent()

    def x_height(self) -> float:
        return float(face(self.family).x_height * self.size)

    def cap_height(self) -> float:
        return float(face(self.family).cap_height * self.size)

    def advance(self, text: str) -> float:
        f = face(self.family)
        return sum(f.advance(char) for char in text) * self.size

    def tight_rect(self, text: str) -> Rect:
        """The ink of ``text`` relative to its baseline origin (y pointing down)."""
        left, bottom, right, top = face(self.family).ink(text)
        s = self.size
        return Rect(left * s, -top * s, (right - left) * s, (top - bottom) * s)
