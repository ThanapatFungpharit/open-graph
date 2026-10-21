"""Turn numeric functions into drawable geometry.

Two algorithms, both fully vectorised:

* :func:`sample_curve` — adaptive sampling of a 1-D function ``v = f(t)``. Starts from
  grid-aligned uniform samples (so geometry is stable while panning), then repeatedly
  bisects intervals that are steep in *screen* space or straddle the edge of the
  function's domain. Intervals still steep after the final pass are jumps or poles and
  break the polyline, so ``1/x`` or ``floor(x)`` are not joined by false vertical lines.

* :func:`marching_squares` — zero contour of ``F(x, y)`` on a pixel-aligned grid, with
  saddle disambiguation and a filter that rejects sign changes caused by poles.

All work is expressed in world coordinates plus pixel densities, so this module knows
nothing about the interface toolkit or the viewport implementation.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from opengraph.engine.builtins import Array
from opengraph.engine.cancellation import checkpoint

# Refinement parameters (screen pixels).
BASE_PX = 2.0  # initial sample spacing
CHORD_TOL_PX = 0.3  # refine where the curve deviates from a straight segment by more
BREAK_PX = 12.0  # after refinement, intervals taller than this may be discontinuities
MAX_PASSES = 16
MAX_POINTS = 40_000
IMPLICIT_CELL_PX = 4.0
MAX_GRID_POINTS = 400_000


def to_axis(values: Any, log: bool) -> Any:
    """World value -> axis coordinate: identity, or log10 on a logarithmic axis.

    Non-positive values have no position on a log axis and map to NaN.
    """
    if not log:
        return values
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.log10(np.where(np.asarray(values) > 0, values, np.nan))
    return result if isinstance(values, np.ndarray) else float(result)


def from_axis(values: Any, log: bool) -> Any:
    """Axis coordinate -> world value (inverse of :func:`to_axis`)."""
    if not log:
        return values
    with np.errstate(over="ignore"):
        result = np.power(10.0, values)
    return result if isinstance(values, np.ndarray) else float(result)


@dataclass(frozen=True, slots=True)
class ViewRect:
    """The visible world rectangle, its size in pixels and each axis's scale type.

    On a logarithmic axis the screen is linear in log10 of the value; ``px_per_x`` and
    ``px_per_y`` are pixels per *axis* unit (per decade on a log axis), and sampling
    happens in axis coordinates so curves are equally detailed across the decades.

    In a deep zoom (``bits > 0``, linear axes only) the bounds are offsets from an exact
    ``origin`` (gmpy2 values; see :mod:`opengraph.engine.precise`): plots evaluate at
    ``origin + offset`` with ``bits`` of precision and return geometry as offsets from
    the same origin. Otherwise the origin is 0 and the bounds are world coordinates.
    """

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    width_px: int
    height_px: int
    x_log: bool = False
    y_log: bool = False
    origin_x: Any = 0.0
    origin_y: Any = 0.0
    bits: int = 0

    @property
    def u_min(self) -> float:
        return float(to_axis(self.x_min, self.x_log))

    @property
    def u_max(self) -> float:
        return float(to_axis(self.x_max, self.x_log))

    @property
    def v_min(self) -> float:
        return float(to_axis(self.y_min, self.y_log))

    @property
    def v_max(self) -> float:
        return float(to_axis(self.y_max, self.y_log))

    @property
    def px_per_x(self) -> float:
        return self.width_px / (self.u_max - self.u_min)

    @property
    def px_per_y(self) -> float:
        return self.height_px / (self.v_max - self.v_min)

    def padded(self, fraction: float) -> ViewRect:
        """This view grown by ``fraction`` of its size on every side (in axis units)."""
        du = (self.u_max - self.u_min) * fraction
        dv = (self.v_max - self.v_min) * fraction
        scale = 1 + 2 * fraction
        return ViewRect(
            float(from_axis(self.u_min - du, self.x_log)),
            float(from_axis(self.u_max + du, self.x_log)),
            float(from_axis(self.v_min - dv, self.y_log)),
            float(from_axis(self.v_max + dv, self.y_log)),
            round(self.width_px * scale),
            round(self.height_px * scale),
            self.x_log,
            self.y_log,
            self.origin_x,
            self.origin_y,
            self.bits,
        )

    def contains(self, inner: ViewRect) -> bool:
        return (
            (self.x_log, self.y_log) == (inner.x_log, inner.y_log)
            and (self.origin_x, self.origin_y) == (inner.origin_x, inner.origin_y)
            and self.x_min <= inner.x_min
            and inner.x_max <= self.x_max
            and self.y_min <= inner.y_min
            and inner.y_max <= self.y_max
        )


Polyline = tuple[Array, Array]  # (t, v) arrays of equal length >= 2


def _aligned_grid(lo: float, hi: float, step: float) -> Array:
    """Points ``k*step`` covering ``[lo, hi]`` with one step of margin either side."""
    checkpoint()
    start = math.floor(lo / step) - 1
    stop = math.ceil(hi / step) + 1
    return np.arange(start, stop + 1, dtype=np.float64) * step


def sample_curve(
    fn: Callable[[Array], Array],
    t_min: float,
    t_max: float,
    t_px: float,
    v_min: float,
    v_max: float,
    v_px: float,
) -> list[Polyline]:
    """Sample ``v = fn(t)`` over ``[t_min, t_max]`` for display.

    ``t_px`` / ``v_px`` are pixels per world unit along each axis; ``v_min``/``v_max``
    is the visible value range (used to skip refining off-screen parts and to clamp
    huge values to just outside the view).
    """
    if not (t_max > t_min and t_px > 0 and v_px > 0):
        return []
    t = _aligned_grid(t_min, t_max, BASE_PX / t_px)
    v = fn(t)
    min_width = (t_max - t_min) * 1e-12
    # Values beyond this band are off-screen; refining there is wasted work.
    band = (v_max - v_min) * 0.5
    lo_band, hi_band = v_min - band, v_max + band

    check = np.arange(t.size - 1)  # intervals still to be tested
    for _ in range(MAX_PASSES):
        checkpoint()
        if check.size == 0:
            break
        ta, tb = t[check], t[check + 1]
        va, vb = v[check], v[check + 1]
        tm = 0.5 * (ta + tb)
        vm = fn(tm)
        fa, fb, fm = np.isfinite(va), np.isfinite(vb), np.isfinite(vm)
        # Domain edge (or a hole) inside the interval: bisect towards it.
        edge = (fa != fb) | (fm != fa)
        # Visible deviation of the curve from the chord: bisect to follow it.
        with np.errstate(invalid="ignore"):
            bent = fa & fb & fm & (np.abs(vm - 0.5 * (va + vb)) * v_px > CHORD_TOL_PX)
            off = ((va > hi_band) & (vb > hi_band) & (vm > hi_band)) | (
                (va < lo_band) & (vb < lo_band) & (vm < lo_band)
            )
        refine = np.nonzero((edge | bent) & ~off & (tb - ta > min_width))[0]
        if refine.size == 0 or t.size + refine.size > MAX_POINTS:
            break
        idx = check[refine]
        t = np.insert(t, idx + 1, tm[refine])
        v = np.insert(v, idx + 1, vm[refine])
        # Interval idx[k] is now split into intervals idx[k]+k and idx[k]+k+1.
        left = idx + np.arange(idx.size)
        check = np.sort(np.concatenate([left, left + 1]))  # shifts above need order

    return _split(fn, t, v, v_px, (lo_band, hi_band), (lo_band - band * 8, hi_band + band * 8))


def _split(
    fn: Callable[[Array], Array],
    t: Array,
    v: Array,
    v_px: float,
    band: tuple[float, float],
    clamp: tuple[float, float],
) -> list[Polyline]:
    finite = np.isfinite(v)
    both = finite[:-1] & finite[1:]
    with np.errstate(invalid="ignore"):
        # Segments with both ends beyond the same edge are invisible whichever way they
        # are drawn; never break those (refinement skipped them, so they may look tall).
        hidden = ((v[:-1] > band[1]) & (v[1:] > band[1])) | ((v[:-1] < band[0]) & (v[1:] < band[0]))
        tall = np.nonzero(both & ~hidden & (np.abs(np.diff(v)) * v_px > BREAK_PX))[0]
    if tall.size:
        # A tall interval whose midpoint lies well inside its endpoints' range is just
        # steep (refinement ran out of budget); a jump puts the midpoint on one side and
        # a pole puts it outside the range. Only the latter two break the curve.
        mid = fn(0.5 * (t[tall] + t[tall + 1]))
        lo = np.minimum(v[tall], v[tall + 1])
        hi = np.maximum(v[tall], v[tall + 1])
        margin = 0.1 * (hi - lo)
        continuous = (mid > lo + margin) & (mid < hi - margin)
        tall = tall[~continuous]
    breaks = ~both
    breaks[tall] = True

    vc = np.clip(v, *clamp)
    cut = np.nonzero(breaks)[0] + 1
    polylines: list[Polyline] = []
    for ts, vs, ok in zip(np.split(t, cut), np.split(vc, cut), np.split(finite, cut), strict=True):
        # Each piece is finite except possibly for its first sample (the one after an
        # undefined break); drop undefined samples and keep pieces with a segment.
        if ok.sum() >= 2:
            polylines.append((ts[ok], vs[ok]))
    return polylines


def marching_squares(
    fn: Callable[[Array, Array], Array], view: ViewRect, cell_px: float = IMPLICIT_CELL_PX
) -> Array:
    """Line segments approximating ``fn(x, y) = 0`` inside ``view``.

    Returns an ``(M, 4)`` array of ``x0, y0, x1, y1`` rows.
    """
    checkpoint()
    dx = cell_px / view.px_per_x
    dy = cell_px / view.px_per_y
    # The grid is regular in axis coordinates (log10 on a log axis); fn sees world values.
    xs = _aligned_grid(view.u_min, view.u_max, dx)
    ys = _aligned_grid(view.v_min, view.v_max, dy)

    def evaluate(u: Array, v: Array) -> Array:
        return fn(from_axis(u, view.x_log), from_axis(v, view.y_log))

    while xs.size * ys.size > MAX_GRID_POINTS:
        # Large exports/canvases keep a coarser contour instead of losing the plot.
        cell_px *= 1.5
        dx, dy = cell_px / view.px_per_x, cell_px / view.px_per_y
        xs = _aligned_grid(view.u_min, view.u_max, dx)
        ys = _aligned_grid(view.v_min, view.v_max, dy)
    gx, gy = np.meshgrid(xs, ys)  # shape (ny, nx); row j is y = ys[j]
    f = np.broadcast_to(evaluate(gx, gy), gx.shape)
    checkpoint()

    # Corner signs of every cell: bottom-left, bottom-right, top-right, top-left. Only
    # cells with a sign change between finite corners can hold the contour, so the rest
    # of the work (centre values, interpolation) is done for those cells alone.
    with np.errstate(invalid="ignore"):
        pos = f > 0
    finite = np.isfinite(f)
    bl_p, br_p, tr_p, tl_p = pos[:-1, :-1], pos[:-1, 1:], pos[1:, 1:], pos[1:, :-1]
    candidate = (
        ((bl_p != br_p) | (br_p != tr_p) | (tr_p != tl_p))
        & finite[:-1, :-1]
        & finite[:-1, 1:]
        & finite[1:, 1:]
        & finite[1:, :-1]
    )
    j, i = np.nonzero(candidate)  # row (y) and column (x) of each candidate cell
    if j.size == 0:
        return np.empty((0, 4))
    corners = np.stack([f[j, i], f[j, i + 1], f[j + 1, i + 1], f[j + 1, i]], axis=1)
    x0, y0 = xs[i], ys[j]
    cx, cy = 0.5 * (x0 + xs[i + 1]), 0.5 * (y0 + ys[j + 1])
    center = np.broadcast_to(evaluate(cx, cy), cx.shape)
    checkpoint()

    # Pole filter: across a genuine zero crossing the centre value stays comparable to
    # the corner values; next to a pole it explodes (or is undefined).
    scale = np.abs(corners).max(axis=1)
    with np.errstate(invalid="ignore"):
        valid = np.isfinite(center) & (np.abs(center) <= 4.0 * scale + 1e-300)

    pos = corners > 0
    # Edges: 0 bottom (bl-br), 1 right (br-tr), 2 top (tl-tr), 3 left (bl-tl).
    ends = [(0, 1), (1, 2), (3, 2), (0, 3)]
    crosses = np.stack([pos[:, a] != pos[:, b] for a, b in ends], axis=1) & valid[:, None]

    px = np.empty((j.size, 4))
    py = np.empty((j.size, 4))
    with np.errstate(all="ignore"):
        for k, (a, b) in enumerate(ends):
            fa, fb = corners[:, a], corners[:, b]
            s = fa / (fa - fb)  # interpolation parameter along the edge
            if k in (0, 2):  # horizontal edges
                px[:, k] = x0 + s * dx
                py[:, k] = y0 + (dy if k == 2 else 0.0)
            else:  # vertical edges
                px[:, k] = x0 + (dx if k == 1 else 0.0)
                py[:, k] = y0 + s * dy

    count = crosses.sum(axis=1)
    segments: list[Array] = []

    two = np.nonzero(count == 2)[0]
    if two.size:
        # Indices of the two crossed edges for each cell.
        order = np.argsort(~crosses[two], axis=1, kind="stable")[:, :2]
        rows = two[:, None]
        segments.append(
            np.column_stack(
                [
                    px[rows, order[:, :1]],
                    py[rows, order[:, :1]],
                    px[rows, order[:, 1:]],
                    py[rows, order[:, 1:]],
                ]
            )
        )

    four = np.nonzero(count == 4)[0]
    if four.size:
        # Saddle: if the centre shares bottom-left's sign, the other two corners are
        # isolated (pair bottom-right and top-left); otherwise pair bottom-left/top-right.
        same = (center[four] > 0) == pos[four, 0]
        pairs_a = np.where(same[:, None], [[0, 1]], [[0, 3]])
        pairs_b = np.where(same[:, None], [[2, 3]], [[1, 2]])
        for pairs in (pairs_a, pairs_b):
            rows = four[:, None]
            segments.append(
                np.column_stack(
                    [
                        px[rows, pairs[:, :1]],
                        py[rows, pairs[:, :1]],
                        px[rows, pairs[:, 1:]],
                        py[rows, pairs[:, 1:]],
                    ]
                )
            )

    if not segments:
        return np.empty((0, 4))
    result = np.vstack(segments)
    if view.x_log:
        result[:, [0, 2]] = from_axis(result[:, [0, 2]], True)
    if view.y_log:
        result[:, [1, 3]] = from_axis(result[:, [1, 3]], True)
    return result


PARAMETRIC_SAMPLES = 1024
PARAMETRIC_TOL_PX = 0.4


def sample_parametric(
    fn: Callable[[Array], tuple[Array, Array]],
    t_min: float,
    t_max: float,
    view: ViewRect,
) -> list[Array]:
    """Sample the plane curve ``(x(t), y(t))`` for ``t`` in ``[t_min, t_max]``.

    Intervals are bisected where the midpoint departs visibly from the chord (in screen
    pixels) or where the curve becomes undefined; segments that stay long and bent after
    refinement are treated as jumps and break the polyline. Returns ``(N, 2)`` arrays in
    world coordinates.
    """
    if not (t_max > t_min):
        return []

    def screen(t: Array) -> tuple[Array, Array, Array, Array]:
        checkpoint()
        x, y = fn(t)
        # Finite world values can overflow when scaled to pixels. Keep the resulting
        # infinities: the finite masks below exclude those unrenderable samples.
        with np.errstate(over="ignore", invalid="ignore"):
            sx = to_axis(np.asarray(x, dtype=np.float64), view.x_log) * view.px_per_x
            sy = to_axis(np.asarray(y, dtype=np.float64), view.y_log) * view.px_per_y
        return x, y, sx, sy

    t = np.linspace(t_min, t_max, PARAMETRIC_SAMPLES + 1)
    x, y, sx, sy = screen(t)
    lo_x, hi_x = view.u_min * view.px_per_x, view.u_max * view.px_per_x
    lo_y, hi_y = view.v_min * view.px_per_y, view.v_max * view.px_per_y
    margin = max(view.width_px, view.height_px)
    check = np.arange(t.size - 1)
    # Distances between extreme off-screen samples can exceed float64 as well.
    # They still require refinement; non-finite final distances must break the line.
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(MAX_PASSES):
            if check.size == 0:
                break
            tm = 0.5 * t[check] + 0.5 * t[check + 1]
            mx, my, msx, msy = screen(tm)
            ax, ay, bx, by = sx[check], sy[check], sx[check + 1], sy[check + 1]
            fa = np.isfinite(ax) & np.isfinite(ay)
            fb = np.isfinite(bx) & np.isfinite(by)
            fm = np.isfinite(msx) & np.isfinite(msy)
            edge = (fa != fb) | (fm != fa)
            deviation = np.hypot(msx - (0.5 * ax + 0.5 * bx), msy - (0.5 * ay + 0.5 * by))
            bent = fa & fb & fm & (deviation > PARAMETRIC_TOL_PX)
            # Skip intervals entirely far outside the view.
            off = (
                (np.maximum(np.maximum(ax, bx), msx) < lo_x - margin)
                | (np.minimum(np.minimum(ax, bx), msx) > hi_x + margin)
                | (np.maximum(np.maximum(ay, by), msy) < lo_y - margin)
                | (np.minimum(np.minimum(ay, by), msy) > hi_y + margin)
            )
            refine = np.nonzero((edge | bent) & ~off)[0]
            if refine.size == 0 or t.size + refine.size > MAX_POINTS:
                break
            idx = check[refine]
            t = np.insert(t, idx + 1, tm[refine])
            x = np.insert(x, idx + 1, mx[refine])
            y = np.insert(y, idx + 1, my[refine])
            sx = np.insert(sx, idx + 1, msx[refine])
            sy = np.insert(sy, idx + 1, msy[refine])
            left = idx + np.arange(idx.size)
            check = np.sort(np.concatenate([left, left + 1]))

        finite = np.isfinite(sx) & np.isfinite(sy)
        both = finite[:-1] & finite[1:]
        length = np.hypot(np.diff(sx), np.diff(sy))
        breaks = ~both
        long = np.nonzero(both & (length > BREAK_PX))[0]
        if long.size:
            # Still long after refinement: a jump if the midpoint is far off the chord.
            _, _, msx, msy = screen(0.5 * t[long] + 0.5 * t[long + 1])
            deviation = np.hypot(
                msx - (0.5 * sx[long] + 0.5 * sx[long + 1]),
                msy - (0.5 * sy[long] + 0.5 * sy[long + 1]),
            )
            # inf <= inf is true, but overflow is not evidence of continuity.
            continuous = (
                np.isfinite(length[long])
                & np.isfinite(deviation)
                & (deviation <= 0.25 * length[long])
            )
            breaks[long[~continuous]] = True
    cut = np.nonzero(breaks)[0] + 1
    lines: list[Array] = []
    for xs, ys, ok in zip(np.split(x, cut), np.split(y, cut), np.split(finite, cut), strict=True):
        if ok.sum() >= 2:
            lines.append(np.column_stack([xs[ok], ys[ok]]))
    return lines


@dataclass(frozen=True, slots=True)
class FillMask:
    """A boolean raster of the cells satisfying an inequality.

    ``mask[j, i]`` is the cell ``i`` columns right and ``j`` rows up from the corner
    ``(x_min, y_min)``; the grid is regular in axis coordinates (log10 on a log axis),
    ``x_min``..``y_max`` are world values of its outer edges.
    """

    mask: Array
    x_min: float
    x_max: float
    y_min: float
    y_max: float


def region_mask(
    test: Callable[[Array, Array], NDArray[np.bool_]], view: ViewRect, cell_px: float = 2.0
) -> FillMask:
    """Evaluate ``test(x, y) -> bool array`` at the centre of each cell covering ``view``."""
    while True:
        dx = cell_px / view.px_per_x
        dy = cell_px / view.px_per_y
        xs = _aligned_grid(view.u_min, view.u_max, dx)
        ys = _aligned_grid(view.v_min, view.v_max, dy)
        if xs.size * ys.size <= MAX_GRID_POINTS:
            break
        cell_px *= 1.5
    cx, cy = np.meshgrid(0.5 * (xs[:-1] + xs[1:]), 0.5 * (ys[:-1] + ys[1:]))
    with np.errstate(all="ignore"):
        mask = np.asarray(test(from_axis(cx, view.x_log), from_axis(cy, view.y_log)), dtype=bool)
    mask = np.broadcast_to(mask, cx.shape)
    return FillMask(
        mask,
        float(from_axis(xs[0], view.x_log)),
        float(from_axis(xs[-1], view.x_log)),
        float(from_axis(ys[0], view.y_log)),
        float(from_axis(ys[-1], view.y_log)),
    )
