"""World <-> screen coordinate mapping, independent of the interface toolkit.

Screen coordinates follow canvas conventions: origin at the top-left, y pointing down,
measured in device-independent pixels. World coordinates are mathematical: y up.

The viewport is described by the world point at the centre of the widget and a scale
in pixels per world unit for each axis. Keeping the centre (rather than a corner) fixed
makes resizing feel natural: the graph stays centred and simply reveals more or less.

Deep zoom: on linear axes the scale may grow far beyond what float64 coordinates can
resolve (``MAX_DEEP_SCALE``). Once a pixel is narrower than a few thousand float64 steps
at the view's coordinates, the axis gets an exact *anchor* (a gmpy2 value at
``VIEW_BITS``); ``center_x``/``center_y`` and every world coordinate the canvas handles
are then float64 offsets from the anchor, which stay small and precise. Zooming back
out folds the anchor into the centre again. Without an anchor (the usual case) it is 0
and coordinates are plain world values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, TypeVar

import gmpy2
import numpy as np
from numpy.typing import NDArray

from opengraph.engine import precise
from opengraph.engine.sampling import ViewRect, from_axis, to_axis

T = TypeVar("T", float, NDArray[np.float64])

DEFAULT_SCALE = 50.0  # pixels per unit: about -10..10 across a typical canvas
MIN_SCALE = 1e-7
MAX_SCALE = 1e12  # logarithmic axes
MAX_DEEP_SCALE = 1e40  # linear axes, using an exact anchor
VIEW_BITS = 512  # precision of the anchors
BITS_STEP = 32  # evaluation precision is rounded up to this, so pans rarely change it
REANCHOR_PX = 1e6  # move the anchor when the centre drifts this far from it


@dataclass(slots=True)
class Viewport:
    width: int = 800
    height: int = 600
    center_x: float = 0.0
    center_y: float = 0.0
    scale_x: float = DEFAULT_SCALE
    scale_y: float = DEFAULT_SCALE
    # On a logarithmic axis the centre is in log10 units and the scale in px per decade.
    x_log: bool = False
    y_log: bool = False
    # Exact world position that the centre (and canvas coordinates) are offsets from.
    anchor_x: Any = 0
    anchor_y: Any = 0

    # -- transforms ------------------------------------------------------------------
    def to_screen_x(self, x: T) -> T:
        u: T = to_axis(x, self.x_log)
        with np.errstate(over="ignore", invalid="ignore"):
            return (u - self.center_x) * self.scale_x + self.width / 2

    def to_screen_y(self, y: T) -> T:
        v: T = to_axis(y, self.y_log)
        with np.errstate(over="ignore", invalid="ignore"):
            return self.height / 2 - (v - self.center_y) * self.scale_y

    def to_world_x(self, sx: T) -> T:
        x: T = from_axis((sx - self.width / 2) / self.scale_x + self.center_x, self.x_log)
        return x

    def to_world_y(self, sy: T) -> T:
        y: T = from_axis((self.height / 2 - sy) / self.scale_y + self.center_y, self.y_log)
        return y

    def to_screen(self, x: T, y: T) -> tuple[T, T]:
        return self.to_screen_x(x), self.to_screen_y(y)

    def to_world(self, sx: T, sy: T) -> tuple[T, T]:
        return self.to_world_x(sx), self.to_world_y(sy)

    # -- visible region ------------------------------------------------------------------
    @property
    def x_min(self) -> float:
        return self.to_world_x(0.0)

    @property
    def x_max(self) -> float:
        return self.to_world_x(float(self.width))

    @property
    def y_min(self) -> float:
        return self.to_world_y(float(self.height))

    @property
    def y_max(self) -> float:
        return self.to_world_y(0.0)

    def view_rect(self) -> ViewRect:
        """The visible area; offsets from the anchors when zoomed deeply."""
        return ViewRect(
            self.x_min,
            self.x_max,
            self.y_min,
            self.y_max,
            max(1, self.width),
            max(1, self.height),
            self.x_log,
            self.y_log,
            self.anchor_x,
            self.anchor_y,
            self.bits,
        )

    # -- precision -------------------------------------------------------------------------
    @property
    def anchored(self) -> bool:
        """Whether coordinates are offsets from an exact anchor (a deep zoom)."""
        return bool(self.anchor_x) or bool(self.anchor_y)

    @property
    def bits(self) -> int:
        """Precision plots need to be evaluated with here; 0 means float64 suffices."""
        if not self.anchored:
            return 0
        needed = max(self._axis_bits("x"), self._axis_bits("y"), 64)
        return min(precise.MAX_BITS, -(-needed // BITS_STEP) * BITS_STEP)

    def _axis_bits(self, axis: str) -> int:
        anchor, center, scale, pixels = self._axis(axis)
        magnitude = abs(float(anchor) + center) + pixels / (2 * scale)
        return precise.precision_bits(magnitude, 1 / scale)

    def _axis(self, axis: str) -> tuple[Any, float, float, int]:
        if axis == "x":
            return self.anchor_x, self.center_x, self.scale_x, self.width
        return self.anchor_y, self.center_y, self.scale_y, self.height

    def max_scale(self, axis: str) -> float:
        return MAX_SCALE if (self.x_log if axis == "x" else self.y_log) else MAX_DEEP_SCALE

    def absolute(self, axis: str, offset: float) -> Any:
        """The exact world coordinate at ``offset`` from the anchor."""
        anchor = self.anchor_x if axis == "x" else self.anchor_y
        if not anchor:
            return offset
        with precise.context(VIEW_BITS):
            return anchor + gmpy2.mpfr(offset)

    def relative(self, axis: str, value: Any) -> float:
        """A world coordinate (float or exact) as an offset from the anchor."""
        anchor = self.anchor_x if axis == "x" else self.anchor_y
        if not anchor:
            return float(value)
        with precise.context(VIEW_BITS):
            return float(gmpy2.mpfr(value) - anchor)

    def _reanchor(self) -> None:
        """Give an axis an exact anchor when float64 can't resolve its pixels, move the
        anchor along when the centre drifts away from it, and drop it when not needed."""
        for axis in ("x", "y"):
            anchor, center, scale, _ = self._axis(axis)
            log = self.x_log if axis == "x" else self.y_log
            deep = not log and self._axis_bits(axis) > 0
            if deep and (not anchor or abs(center) * scale > REANCHOR_PX):
                with precise.context(VIEW_BITS):
                    anchor = gmpy2.mpfr(anchor) + gmpy2.mpfr(center)
                center = 0.0
            elif not deep and anchor:
                with precise.context(VIEW_BITS):
                    center, anchor = float(gmpy2.mpfr(anchor) + center), 0
            else:
                continue
            if axis == "x":
                self.anchor_x, self.center_x = anchor, center
            else:
                self.anchor_y, self.center_y = anchor, center

    # -- interaction ---------------------------------------------------------------------
    def resize(self, width: int, height: int) -> None:
        self.width, self.height = max(1, width), max(1, height)

    def pan_pixels(self, dx: float, dy: float) -> None:
        """Move the content by ``(dx, dy)`` screen pixels (as when dragging it)."""
        self.center_x -= dx / self.scale_x
        self.center_y += dy / self.scale_y
        self._reanchor()

    def zoom_at(self, sx: float, sy: float, factor: float) -> None:
        """Zoom by ``factor`` (>1 zooms in) keeping the world point under ``(sx, sy)``
        fixed on screen."""
        # One ratio for both axes so clamping never distorts the aspect ratio.
        ratio = min(factor, self.max_scale("x") / self.scale_x, self.max_scale("y") / self.scale_y)
        ratio = max(ratio, MIN_SCALE / min(self.scale_x, self.scale_y))
        self._zoom(sx, sy, ratio, ratio)

    def zoom_axes_at(self, sx: float, sy: float, factor_x: float, factor_y: float) -> None:
        """Zoom each axis by its own factor (independent axis scaling)."""
        fx = min(max(factor_x, MIN_SCALE / self.scale_x), self.max_scale("x") / self.scale_x)
        fy = min(max(factor_y, MIN_SCALE / self.scale_y), self.max_scale("y") / self.scale_y)
        self._zoom(sx, sy, fx, fy)

    def _zoom(self, sx: float, sy: float, fx: float, fy: float) -> None:
        # Axis coordinates under the cursor, kept fixed on screen.
        ux = self.center_x + (sx - self.width / 2) / self.scale_x
        uy = self.center_y + (self.height / 2 - sy) / self.scale_y
        self.scale_x *= fx
        self.scale_y *= fy
        self.center_x = ux - (sx - self.width / 2) / self.scale_x
        self.center_y = uy - (self.height / 2 - sy) / self.scale_y
        self._reanchor()

    def zoom_center(self, factor: float) -> None:
        self.zoom_at(self.width / 2, self.height / 2, factor)

    def reset(self) -> None:
        """The default view: about -10..10 on linear axes, 0.01..100 on log axes."""
        self.center_x = self.center_y = 0.0
        self.anchor_x = self.anchor_y = 0
        self.scale_x = self.width / 4 if self.x_log else DEFAULT_SCALE
        self.scale_y = self.height / 4 if self.y_log else DEFAULT_SCALE

    def set_range(self, axis: str, lo: float, hi: float) -> bool:
        """Show exactly ``[lo, hi]`` along ``axis`` ("x" or "y"); False if invalid."""
        log = self.x_log if axis == "x" else self.y_log
        if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi) or (log and lo <= 0):
            return False
        a, b = to_axis(lo, log), to_axis(hi, log)
        pixels = self.width if axis == "x" else self.height
        scale = min(max(pixels / (b - a), MIN_SCALE), self.max_scale(axis))
        if axis == "x":
            self.anchor_x, self.center_x, self.scale_x = 0, (a + b) / 2, scale
        else:
            self.anchor_y, self.center_y, self.scale_y = 0, (a + b) / 2, scale
        self._reanchor()
        return True

    def range(self, axis: str) -> tuple[float, float]:
        """The visible range: offsets from the anchor (world values without one)."""
        return (self.x_min, self.x_max) if axis == "x" else (self.y_min, self.y_max)

    def set_log(self, axis: str, log: bool) -> None:
        """Switch an axis between linear and log, keeping its visible range if possible."""
        if log == (self.x_log if axis == "x" else self.y_log):
            return
        lo, hi = (float(self.absolute(axis, value)) for value in self.range(axis))
        if axis == "x":
            self.x_log = log
        else:
            self.y_log = log
        if log and hi > 0:
            lo = lo if lo > 0 else hi / 1000  # keep the positive part
        elif log:
            lo, hi = 0.01, 100.0
        self.set_range(axis, lo, hi)

    def fit(self, x_min: float, x_max: float, y_min: float, y_max: float) -> None:
        """Show at least the given world rectangle with equal axis scales."""
        if not all(map(math.isfinite, (x_min, x_max, y_min, y_max))):
            return
        span_x = max(x_max - x_min, 1e-9)
        span_y = max(y_max - y_min, 1e-9)
        scale = min(self.width / span_x, self.height / span_y)
        self.scale_x = self.scale_y = min(max(scale, MIN_SCALE), MAX_SCALE)
        self.anchor_x = self.anchor_y = 0
        self.center_x = (x_min + x_max) / 2
        self.center_y = (y_min + y_max) / 2
        self._reanchor()

    # -- persistence ---------------------------------------------------------------------
    def state(self) -> dict[str, float]:
        """The view to save (a deep zoom's centre is rounded to float64)."""
        return {
            "center_x": float(self.absolute("x", self.center_x)),
            "center_y": float(self.absolute("y", self.center_y)),
            "scale_x": self.scale_x,
            "scale_y": self.scale_y,
        }

    def restore(self, state: dict[str, float]) -> None:
        values = {
            k: float(state[k]) for k in ("center_x", "center_y", "scale_x", "scale_y") if k in state
        }
        if not all(math.isfinite(v) for v in values.values()):
            return
        for key, axis in (("scale_x", "x"), ("scale_y", "y")):
            if key in values:
                values[key] = min(max(values[key], MIN_SCALE), self.max_scale(axis))
        self.anchor_x = self.anchor_y = 0
        for key, value in values.items():
            setattr(self, key, value)
        self._reanchor()
