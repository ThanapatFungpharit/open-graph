"""Numerical analysis of plots: points of interest and equation solving. Interface-free.

Everything here works on sampled values and refines them numerically:

* zeros: sign changes between samples, refined by bisection. A sign change across a
  pole or a jump is rejected because the refined value does not go to zero.
  Tangential zeros (``x^2``) don't change sign; they are found as extrema whose value
  is zero.
* extrema: local maxima/minima of the samples, refined by golden-section search.
  Extrema that converge to the edge of their bracket are jumps or poles, not extrema.

Intersections are zeros of a difference: ``f(x) - g(x)`` for two graphs of x,
``F(x, f(x))`` for a graph against an implicit curve.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from opengraph.engine.builtins import Array
from opengraph.engine.plots import ExplicitPlot, ImplicitPlot, Plot, members

ANALYSIS_SAMPLES = 2000
BISECTIONS = 64
GOLDEN_STEPS = 60
MAX_SOLUTIONS = 64

_LABELS = {
    "x-intercept": "x-intercept",
    "y-intercept": "y-intercept",
    "maximum": "Local maximum",
    "minimum": "Local minimum",
    "intersection": "Intersection",
}

Fn = Callable[[Array], Array]


@dataclass(frozen=True, slots=True)
class PointOfInterest:
    x: float
    y: float
    kind: str  # x-intercept | y-intercept | maximum | minimum | intersection

    @property
    def label(self) -> str:
        return _LABELS[self.kind]


def _evaluate(fn: Fn, t: Array) -> Array:
    with np.errstate(all="ignore"):
        return np.asarray(fn(t), dtype=np.float64)


def zeros(fn: Fn, ts: Array) -> list[float]:
    """Zeros of ``fn`` between consecutive samples ``ts`` (sorted), refined."""
    v = _evaluate(fn, ts)
    found = [float(t) for t in ts[v == 0]]
    finite = np.isfinite(v)
    idx = np.nonzero(finite[:-1] & finite[1:] & (np.sign(v[:-1]) * np.sign(v[1:]) < 0))[0]
    if idx.size:
        a, b = ts[idx].copy(), ts[idx + 1].copy()
        fa = v[idx].copy()
        for _ in range(BISECTIONS):
            m = 0.5 * (a + b)
            fm = _evaluate(fn, m)
            same = np.sign(fm) == np.sign(fa)
            a = np.where(same, m, a)
            fa = np.where(same, fm, fa)
            b = np.where(same, b, m)
        roots = 0.5 * (a + b)
        at_root = np.abs(_evaluate(fn, roots))
        ends = np.maximum(np.abs(v[idx]), np.abs(v[idx + 1]))
        # A genuine zero: |f| shrinks to (almost) nothing. A pole or jump keeps it large.
        genuine = np.isfinite(at_root) & (at_root <= 1e-6 * ends + 1e-12)
        found.extend(float(r) for r in roots[genuine])
    return sorted(found)


@dataclass(frozen=True, slots=True)
class Extremum:
    t: float
    value: float
    kind: str  # maximum | minimum
    touches_zero: bool  # the value is zero relative to its neighbours: a double root


def extrema(fn: Fn, ts: Array) -> list[Extremum]:
    """Local extrema strictly inside ``ts``, refined."""
    v = _evaluate(fn, ts)
    if v.size < 3:
        return []
    a, m, b = v[:-2], v[1:-1], v[2:]
    ok = np.isfinite(a) & np.isfinite(m) & np.isfinite(b)
    with np.errstate(invalid="ignore"):
        is_max = ok & (m > a) & (m >= b)
        is_min = ok & (m < a) & (m <= b)
    found: list[Extremum] = []
    for mask, kind, sign in ((is_max, "maximum", -1.0), (is_min, "minimum", 1.0)):
        idx = np.nonzero(mask)[0] + 1
        if idx.size == 0:
            continue
        lo, hi = ts[idx - 1].copy(), ts[idx + 1].copy()
        width = hi - lo
        # Golden-section search for the minimum of sign * f.
        ratio = (math.sqrt(5) - 1) / 2
        c = hi - ratio * (hi - lo)
        d = lo + ratio * (hi - lo)
        fc, fd = sign * _evaluate(fn, c), sign * _evaluate(fn, d)
        for _ in range(GOLDEN_STEPS):
            # f(c) <= f(d): the minimum is in [lo, d]; otherwise in [c, hi]. NaN-safe.
            left = ~(fc > fd)
            new_lo = np.where(left, lo, c)
            new_hi = np.where(left, d, hi)
            new_c = np.where(left, new_hi - ratio * (new_hi - new_lo), d)
            new_d = np.where(left, c, new_lo + ratio * (new_hi - new_lo))
            probe = sign * _evaluate(fn, np.where(left, new_c, new_d))
            fc, fd = np.where(left, probe, fd), np.where(left, fc, probe)
            lo, hi, c, d = new_lo, new_hi, new_c, new_d
        t = 0.5 * (lo + hi)
        value = _evaluate(fn, t)
        start, end = ts[idx - 1], ts[idx + 1]
        interior = (t - start > 1e-6 * width) & (end - t > 1e-6 * width)
        # Smoothness: a true extremum improves on the sampled peak by less than the
        # rise to it; across a pole the search lands on huge values (or the wrong side).
        peak, before, after = v[idx], v[idx - 1], v[idx + 1]
        rise = np.abs(peak - before) + np.abs(peak - after)
        gain = sign * (peak - value)  # >= 0 when the refined point is at least as extreme
        smooth = (gain >= -1e-9 * (1 + np.abs(peak))) & (gain <= 2 * rise + 1e-12)
        keep = interior & np.isfinite(value) & smooth
        scale = np.maximum(np.abs(v[idx - 1]), np.abs(v[idx + 1]))
        flat = np.abs(value) <= 1e-6 * scale
        found.extend(
            Extremum(float(ti), float(vi), kind, bool(zi))
            for ti, vi, zi in zip(t[keep], value[keep], flat[keep], strict=True)
        )
    found.sort(key=lambda e: e.t)
    return found


def _dedupe(points: list[PointOfInterest], tol_x: float, tol_y: float) -> list[PointOfInterest]:
    kept: list[PointOfInterest] = []
    for p in points:
        if not (math.isfinite(p.x) and math.isfinite(p.y)):
            continue
        if any(
            abs(p.x - q.x) <= tol_x and abs(p.y - q.y) <= tol_y and p.kind == q.kind for q in kept
        ):
            continue
        kept.append(p)
    return kept


def sample_grid(lo: float, hi: float, log: bool = False, count: int = ANALYSIS_SAMPLES) -> Array:
    """Analysis samples over ``[lo, hi]``, log-spaced on a logarithmic axis."""
    if log:
        if hi <= 0:
            return np.empty(0)
        lo = max(lo, hi * 1e-300)
        return np.logspace(math.log10(lo), math.log10(hi), count)
    return np.linspace(lo, hi, count)


def explicit_points(plot: ExplicitPlot, ts: Array) -> list[PointOfInterest]:
    """Intercepts and extrema of ``y = f(x)`` (or ``x = f(y)``) over the samples ``ts``."""
    fn = plot.evaluate
    by_x = plot.independent == "x"

    def point(t: float, value: float, kind: str) -> PointOfInterest:
        return PointOfInterest(t, value, kind) if by_x else PointOfInterest(value, t, kind)

    root_kind, cross_kind = (
        ("x-intercept", "y-intercept") if by_x else ("y-intercept", "x-intercept")
    )
    found = [point(t, 0.0, root_kind) for t in zeros(fn, ts)]
    for e in extrema(fn, ts):
        found.append(point(e.t, e.value, e.kind))
        if e.touches_zero:  # a tangential zero (double root)
            found.append(point(e.t, 0.0, root_kind))
    if ts.size and ts[0] <= 0 <= ts[-1]:
        at_zero = float(_evaluate(fn, np.array([0.0]))[0])
        if math.isfinite(at_zero):
            found.append(point(0.0, at_zero, cross_kind))
    step = float(np.max(np.diff(ts))) if ts.size > 1 else 1.0
    # Duplicates are the same kind of point found twice within one sample step.
    return _dedupe(found, step if by_x else math.inf, math.inf if by_x else step)


def implicit_points(plot: ImplicitPlot, xs: Array, ys: Array) -> list[PointOfInterest]:
    """Where an implicit curve crosses the axes."""
    found = [
        PointOfInterest(t, 0.0, "x-intercept")
        for t in zeros(lambda t: plot.fn(x=t, y=np.zeros_like(t)), xs)
    ]
    found += [
        PointOfInterest(0.0, t, "y-intercept")
        for t in zeros(lambda t: plot.fn(x=np.zeros_like(t), y=t), ys)
    ]
    step = float(np.max(np.diff(xs))) if xs.size > 1 else 1.0
    return _dedupe(found, step, step)


def points_of_interest(plot: Plot, xs: Array, ys: Array) -> list[PointOfInterest]:
    """Points of interest of one row's plot over the sample grids ``xs``/``ys``."""
    found: list[PointOfInterest] = []
    for member in members(plot):
        if isinstance(member, ExplicitPlot):
            found += explicit_points(member, xs if member.independent == "x" else ys)
        elif isinstance(member, ImplicitPlot):
            found += implicit_points(member, xs, ys)
    return found


def _intersect_pair(a: Plot, b: Plot, xs: Array, ys: Array) -> list[PointOfInterest]:
    if isinstance(b, ExplicitPlot) and not isinstance(a, ExplicitPlot):
        a, b = b, a
    if not isinstance(a, ExplicitPlot):
        return []  # implicit/implicit and polar curves: not supported
    f = a.evaluate
    ts = xs if a.independent == "x" else ys
    difference: Fn
    if isinstance(b, ExplicitPlot):
        g = b.evaluate
        if b.independent == a.independent:

            def difference(t: Array) -> Array:
                return f(t) - g(t)

        else:  # y = f(x) against x = g(y): t - g(f(t)) = 0

            def difference(t: Array) -> Array:
                return t - g(f(t))

    elif isinstance(b, ImplicitPlot):
        implicit = b.fn
        if a.independent == "x":

            def difference(t: Array) -> Array:
                return implicit(x=t, y=f(t))

        else:

            def difference(t: Array) -> Array:
                return implicit(x=f(t), y=t)

    else:
        return []
    found = []
    for t in zeros(difference, ts):
        value = float(_evaluate(f, np.array([t]))[0])
        x, y = (t, value) if a.independent == "x" else (value, t)
        found.append(PointOfInterest(x, y, "intersection"))
    return found


def intersections(a: Plot, b: Plot, xs: Array, ys: Array) -> list[PointOfInterest]:
    """Intersections of two rows' plots (curve pairs that can be reduced to 1-D)."""
    found: list[PointOfInterest] = []
    for ma in members(a):
        for mb in members(b):
            found += _intersect_pair(ma, mb, xs, ys)
    step = float(np.max(np.diff(xs))) if xs.size > 1 else 1.0
    return _dedupe(found, step, step)


def _global_grid() -> Array:
    near = np.linspace(-100.0, 100.0, 40_001)
    far = np.logspace(2, 8, 6_001)
    return np.unique(np.concatenate([-far[::-1], near, far]))


def solve_equation(fn: Fn) -> tuple[list[float], bool]:
    """Real solutions of ``fn(t) = 0`` found by scanning |t| <= 1e8.

    Returns ``(solutions, truncated)``; ``truncated`` means there were more than
    :data:`MAX_SOLUTIONS` (periodic equations) and only the ones nearest zero are kept.
    """
    ts = _global_grid()
    found = zeros(fn, ts)
    found += [e.t for e in extrema(fn, ts) if e.touches_zero]
    found.sort()
    unique: list[float] = []
    for t in found:
        if not unique or abs(t - unique[-1]) > 1e-9 * (1 + abs(t)):
            unique.append(t)
    truncated = len(unique) > MAX_SOLUTIONS
    if truncated:
        unique = sorted(sorted(unique, key=abs)[:MAX_SOLUTIONS])
    return unique, truncated
