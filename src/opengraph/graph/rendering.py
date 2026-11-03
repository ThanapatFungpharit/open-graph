"""Load only visible graph geometry, outside the interface's event loop.

Spatial plots use axis-aligned tiles (also on logarithmic axes); explicit functions
use strips, and parametric/polar curves use one padded adaptive sample. Requests are
replaced on navigation, rather than accumulating in a worker queue. The reusable LRU
has both byte and entry limits; the current visible working set stays pinned until
the view changes. No interface objects are touched by workers.

Workers do everything up to the canvas shapes: sampling, culling, simplifying the
outline, converting it to screen pixels (relative to the tile's own corner, so the
result stays valid while panning) and compressing the fill raster of an inequality.
The interface thread only places the prepared pixels. The heavy loops are compiled
(:mod:`opengraph.native`) and release the GIL, so tiles are computed in parallel.

In a deep zoom each tile has an exact origin (a multiple of the tile size) and its
geometry is offsets from it; see :class:`~opengraph.engine.sampling.ViewRect`.
"""

from __future__ import annotations

import logging
import math
import os
import zlib
from collections import OrderedDict
from collections.abc import Callable, Hashable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from functools import partial
from sys import float_info
from threading import Event

import numpy as np
from numpy.typing import NDArray

from opengraph import events, native
from opengraph.engine import precise
from opengraph.engine.builtins import Array
from opengraph.engine.cancellation import Cancelled, cancellable, checkpoint
from opengraph.engine.multiprecision import gmpy2
from opengraph.engine.plots import ExplicitPlot, Geometry, ImplicitPlot, Plot, RegionPlot, members
from opengraph.engine.sampling import FillMask, ViewRect, from_axis, to_axis

TILE_PX = 512
OUTLINE_TOL_PX = 0.2  # a stroked line may move this far (screen pixels) when simplified
SIMPLIFY_SPAN = 32  # vertices between those always kept
SIMPLIFY_PASSES = 24
# The compiled kernels release the GIL, so more workers sample tiles in parallel.
WORKERS = max(2, min(6, (os.cpu_count() or 2) - 1)) if native.kernels is not None else 2
CACHE_TILES = 256
CACHE_BYTES = 64 * 1024 * 1024
log = logging.getLogger(__name__)
_pool: ThreadPoolExecutor | None = None


def worker_pool() -> ThreadPoolExecutor:
    global _pool
    if _pool is None:
        _pool = ThreadPoolExecutor(WORKERS, thread_name_prefix="opengraph-geometry")
    return _pool


@dataclass(frozen=True, slots=True)
class TileRequest:
    layer: Hashable
    plot: Plot
    view: ViewRect
    # Pixels per axis unit the tile is drawn at (its screen geometry is prepared for it).
    scale: tuple[float, float] = (1.0, 1.0)

    @property
    def key(self) -> tuple[Hashable, int, ViewRect]:
        return self.layer, id(self.plot), self.view


TileKey = tuple[Hashable, int, ViewRect]


@dataclass(frozen=True, slots=True)
class FillImage:
    """An inequality's shading as a 1-bit PNG image body, without its colours.

    ``idat`` is the compressed scanline data; :func:`fill_png` adds a palette with the
    layer's colour, so a colour change never re-encodes the raster. ``rect`` is where
    the image goes, in pixels from the tile's corner.
    """

    idat: bytes
    width: int
    height: int
    rect: tuple[float, float, float, float]  # x, y, width, height


@dataclass(frozen=True, slots=True)
class ScreenGeometry:
    """A tile's geometry in screen pixels, measured from the tile's top-left corner at
    the tile's scale (y down). The canvas maps it to the current view with a scale and
    a translation, so it stays valid while panning and serves as a preview while
    zooming. Every line is split where a vertex has no position (log axes)."""

    scale: tuple[float, float]  # pixels per axis unit
    corner: tuple[float, float]  # (u_min, v_max) of the tile, axis units from its origin
    outline: tuple[Array, ...]  # the lines to stroke (simplified)
    lines: tuple[Array, ...]  # every polyline vertex, for tracing
    segments: Array | None  # (M, 4)
    points: Array | None  # (K, 2)
    fill: FillImage | None = None

    @property
    def nbytes(self) -> int:
        arrays = [*self.outline, *self.lines]
        arrays += [a for a in (self.segments, self.points) if a is not None]
        size = sum(a.nbytes for a in arrays)
        return size + (len(self.fill.idat) if self.fill is not None else 0)


@dataclass(frozen=True, slots=True)
class GeometryTile:
    request: TileRequest
    geometry: Geometry
    # The lines to stroke (world coordinates): the polylines and joined segments of the
    # geometry, simplified for the tile's scale. The geometry keeps every vertex.
    outline: tuple[Array, ...] | None = None
    screen: ScreenGeometry | None = None

    @property
    def cost(self) -> int:
        arrays = list(self.geometry.polylines) + list(self.outline or ())
        arrays += [a for a in (self.geometry.segments, self.geometry.points) if a is not None]
        size = sum(a.nbytes for a in arrays)
        if self.geometry.fill is not None:
            size += self.geometry.fill.mask.nbytes
        if self.screen is not None:
            size += self.screen.nbytes
        return 1024 + size


def _tile_grid(view: ViewRect, du: float, dv: float) -> tuple[int, int, int, int]:
    """Indices of the tiles covering ``view``: ``left, right, bottom, top``."""
    if not view.bits:
        return (
            math.floor(view.u_min / du),
            math.ceil(view.u_max / du),
            math.floor(view.v_min / dv),
            math.ceil(view.v_max / dv),
        )
    with precise.context(view.bits + 64):
        ox, oy = gmpy2.mpfr(view.origin_x), gmpy2.mpfr(view.origin_y)
        return (
            int(gmpy2.floor((ox + view.u_min) / du)),
            int(gmpy2.ceil((ox + view.u_max) / du)),
            int(gmpy2.floor((oy + view.v_min) / dv)),
            int(gmpy2.ceil((oy + view.v_max) / dv)),
        )


def tile_requests(layer: Hashable, plot: Plot, view: ViewRect) -> list[TileRequest]:
    """Stable tile boundaries make panning and resizing reuse the same samples."""
    # Inverting a panned viewport can introduce a few ulps of scale noise.
    sx, sy = float(f"{view.px_per_x:.12g}"), float(f"{view.px_per_y:.12g}")
    du, dv = TILE_PX / sx, TILE_PX / sy
    left, right, bottom, top = _tile_grid(view, du, dv)
    requests: list[TileRequest] = []
    for part in members(plot):
        if isinstance(part, ExplicitPlot):
            if part.independent == "x":
                bounds = [(i, i + 1, bottom, top) for i in range(left, right)]
            else:
                bounds = [(left, right, j, j + 1) for j in range(bottom, top)]
        elif isinstance(part, (ImplicitPlot, RegionPlot)):
            bounds = [(i, i + 1, j, j + 1) for j in range(bottom, top) for i in range(left, right)]
        else:
            bounds = [(left, right, bottom, top)]
        for x0, x1, y0, y1 in bounds:
            if view.bits:
                # Exact origins: offsets within a tile are small and precise.
                with precise.context(view.bits + 64):
                    ox, oy = gmpy2.mpfr(x0) * du, gmpy2.mpfr(y0) * dv
                area = ViewRect(
                    0.0,
                    (x1 - x0) * du,
                    0.0,
                    (y1 - y0) * dv,
                    (x1 - x0) * TILE_PX,
                    (y1 - y0) * TILE_PX,
                    origin_x=ox,
                    origin_y=oy,
                    bits=view.bits,
                )
                requests.append(TileRequest(layer, part, area, (sx, sy)))
                continue
            u0, u1, v0, v1 = x0 * du, x1 * du, y0 * dv, y1 * dv
            # Keep tile margins inside the representable log-domain.
            if view.x_log:
                u0 = max(math.log10(math.ulp(0.0)), u0)
                u1 = min(math.nextafter(math.log10(float_info.max), 0.0), u1)
            if view.y_log:
                v0 = max(math.log10(math.ulp(0.0)), v0)
                v1 = min(math.nextafter(math.log10(float_info.max), 0.0), v1)
            if not (u0 < u1 and v0 < v1):
                continue
            area = ViewRect(
                float(from_axis(u0, view.x_log)),
                float(from_axis(u1, view.x_log)),
                float(from_axis(v0, view.y_log)),
                float(from_axis(v1, view.y_log)),
                max(1, round((u1 - u0) * sx)),
                max(1, round((v1 - v0) * sy)),
                view.x_log,
                view.y_log,
            )
            requests.append(TileRequest(layer, part, area, (sx, sy)))
    return requests


def origin_offset(view: ViewRect, reference: ViewRect) -> tuple[float, float]:
    """How far ``view``'s origin is from ``reference``'s, in axis units (0 unless both
    are deep-zoom views with different origins)."""
    if not view.bits or (view.origin_x, view.origin_y) == (reference.origin_x, reference.origin_y):
        return 0.0, 0.0
    with precise.context(view.bits + 64):
        return (
            float(gmpy2.mpfr(view.origin_x) - reference.origin_x),
            float(gmpy2.mpfr(view.origin_y) - reference.origin_y),
        )


def screen_geometry(
    geometry: Geometry, view: ViewRect, scale: tuple[float, float], drawn: Sequence[Array]
) -> ScreenGeometry:
    """``geometry`` (and its simplified outline ``drawn``) in pixels from the corner of
    ``view`` at ``scale``; see :class:`ScreenGeometry`."""
    sx, sy = scale
    u0, v0 = view.u_min, view.v_max

    def pixels(x: Array, y: Array) -> Array:
        with np.errstate(over="ignore", invalid="ignore"):
            return np.column_stack(
                [(to_axis(x, view.x_log) - u0) * sx, (v0 - to_axis(y, view.y_log)) * sy]
            )

    def runs(lines: Sequence[Array]) -> tuple[Array, ...]:
        return tuple(
            piece for line in lines for piece in finite_runs(pixels(line[:, 0], line[:, 1]))
        )

    segments = None
    if geometry.segments is not None and len(geometry.segments):
        s = geometry.segments
        segments = np.column_stack([pixels(s[:, 0], s[:, 1]), pixels(s[:, 2], s[:, 3])])
        segments = segments[np.isfinite(segments).all(axis=1)]
    points = None
    if geometry.points is not None and len(geometry.points):
        points = pixels(geometry.points[:, 0], geometry.points[:, 1])
    fill = None
    if geometry.fill is not None:
        fill = _fill_image(geometry.fill, view, pixels)
    return ScreenGeometry(
        scale=scale,
        corner=(u0, v0),
        outline=runs(drawn),
        lines=runs(geometry.polylines),
        segments=segments,
        points=points,
        fill=fill,
    )


def _fill_image(
    fill: FillMask, view: ViewRect, pixels: Callable[[Array, Array], Array]
) -> FillImage | None:
    """The part of a shaded region inside its tile, as a 1-bit image.

    Neighbouring tiles sample a small border beyond their own area; cropping each fill
    to its tile keeps translucent shading from doubling along the seams.
    """
    corners = pixels(np.array([fill.x_min, fill.x_max]), np.array([fill.y_max, fill.y_min]))
    if not np.isfinite(corners).all():
        return None
    (left, top), (right, bottom) = corners
    rows, cols = fill.mask.shape
    col_w, row_h = (right - left) / cols, (bottom - top) / rows
    if col_w <= 0 or row_h <= 0:
        return None
    centers_x = left + (np.arange(cols) + 0.5) * col_w
    centers_y = top + (np.arange(rows) + 0.5) * row_h  # top row first
    keep_x = np.flatnonzero((centers_x >= 0) & (centers_x < view.width_px))
    keep_y = np.flatnonzero((centers_y >= 0) & (centers_y < view.height_px))
    if not len(keep_x) or not len(keep_y):
        return None
    c0, c1, r0, r1 = keep_x[0], keep_x[-1] + 1, keep_y[0], keep_y[-1] + 1
    top_down = fill.mask[::-1][r0:r1, c0:c1]  # row 0 of the mask is the bottom
    if not top_down.any():
        return None
    packed = np.packbits(top_down.astype(bool), axis=1)  # 1 bit per pixel, MSB first
    rows_data = np.zeros((packed.shape[0], packed.shape[1] + 1), dtype=np.uint8)
    rows_data[:, 1:] = packed  # filter byte 0 per row
    rect = (float(left + c0 * col_w), float(top + r0 * row_h), (c1 - c0) * col_w, (r1 - r0) * row_h)
    return FillImage(zlib.compress(rows_data.tobytes(), 6), int(c1 - c0), int(r1 - r0), rect)


def finite_runs(points: Array) -> list[Array]:
    """Split a screen polyline at non-finite vertices (e.g. x <= 0 on a log axis)."""
    finite = np.isfinite(points).all(axis=1)
    if finite.all():
        return [points]
    cut = np.nonzero(~finite)[0]
    return [
        piece[1:] if i else piece
        for i, piece in enumerate(np.split(points, cut))
        if len(piece) - (1 if i else 0) >= 2
    ]


def visible_geometry(geometry: Geometry, view: ViewRect) -> Geometry:
    """Cull invisible data before building canvas shapes or iterating point markers.

    Keep both endpoints of crossing segments, even if neither is inside the tile.
    Consecutive visible runs stay separate so culling cannot introduce a new chord.
    """

    def crosses(a: Array, b: Array) -> NDArray[np.bool_]:
        return (
            (np.minimum(a[:, 0], b[:, 0]) <= view.x_max)
            & (np.maximum(a[:, 0], b[:, 0]) >= view.x_min)
            & (np.minimum(a[:, 1], b[:, 1]) <= view.y_max)
            & (np.maximum(a[:, 1], b[:, 1]) >= view.y_min)
        )

    lines: list[Array] = []
    for line in geometry.polylines:
        mask = crosses(line[:-1], line[1:])
        edges = np.flatnonzero(np.diff(np.r_[False, mask, False]))
        lines.extend(
            line[start : end + 1].copy() for start, end in zip(edges[::2], edges[1::2], strict=True)
        )
    segments = geometry.segments
    if segments is not None:
        segments = segments[crosses(segments[:, :2], segments[:, 2:])]
    points = geometry.points
    if points is not None:
        points = points[
            (points[:, 0] >= view.x_min)
            & (points[:, 0] <= view.x_max)
            & (points[:, 1] >= view.y_min)
            & (points[:, 1] <= view.y_max)
        ]
    return replace(geometry, polylines=tuple(lines), segments=segments, points=points)


def outline(
    geometry: Geometry, view: ViewRect, tolerance_px: float = OUTLINE_TOL_PX
) -> list[Array]:
    """The lines of ``geometry`` to stroke, with as few vertices as look the same.

    Segments that share end points (an implicit curve's marching-squares cells) are
    joined into polylines, which halves the vertices sent and lets a dash pattern run
    along the curve. Every line is then simplified so that it stays within
    ``tolerance_px`` screen pixels of the original (in axis coordinates, so this also
    holds on logarithmic axes).
    """
    scale: Array = np.array([view.px_per_x, view.px_per_y])

    def pixels(points: Array) -> Array:
        with np.errstate(over="ignore", invalid="ignore"):
            axis: Array = np.column_stack(
                [to_axis(points[:, 0], view.x_log), to_axis(points[:, 1], view.y_log)]
            )
            return axis * scale

    lines = list(geometry.polylines)
    if geometry.segments is not None and len(geometry.segments):
        lines += _join_segments(geometry.segments, pixels(geometry.segments.reshape(-1, 2)))
    runs: list[Array] = []
    for line in lines:
        finite = np.isfinite(pixels(line)).all(axis=1)
        if finite.all():
            runs.append(line)
            continue
        # Without a screen position (x <= 0 on a log axis) a vertex splits the line.
        edges = np.flatnonzero(np.diff(np.r_[False, finite, False]))
        runs += [line[a:b] for a, b in zip(edges[::2], edges[1::2], strict=True) if b - a >= 2]
    if not runs:
        return []
    checkpoint()
    # All lines are simplified together (one vectorised pass instead of one per line).
    world = np.vstack(runs)
    starts = np.cumsum([0] + [len(run) for run in runs])
    kept = _simplify(pixels(world), tolerance_px, starts)
    bounds = np.searchsorted(kept, starts[1:-1])
    return np.split(world[kept], bounds)


def _join_segments(segments: Array, ends_px: Array) -> list[Array]:
    """Chain ``(M, 4)`` segments into polylines where their end points coincide.

    Neighbouring cells compute a shared crossing with slightly different rounding, so
    end points are matched to 1/1024 of a pixel (``ends_px`` are their screen positions).
    A point where other than two segments meet ends the chains through it.
    """
    with np.errstate(invalid="ignore"):
        quantized = np.round(ends_px * 1024)
    # One complex key per point: much faster than unique rows, and ordered the same way.
    keys = quantized[:, 0] + 1j * quantized[:, 1]
    _, first, node = np.unique(keys, return_index=True, return_inverse=True)
    coords = segments.reshape(-1, 2)[first]
    pairs = node.reshape(-1, 2)
    # A contour through a grid point leaves zero-length segments there; they draw
    # nothing, but would look like a junction.
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    if not len(pairs):
        return []
    if native.kernels is not None:
        chained, lengths = native.kernels.chain_segments(
            np.ascontiguousarray(pairs, dtype=np.int64), len(coords)
        )
        return np.split(coords[chained], np.cumsum(lengths)[:-1])
    return [coords[chain] for chain in _chain_segments(pairs.tolist())]


def _chain_segments(seg_nodes: list[list[int]]) -> list[list[int]]:
    """Reference implementation of ``native.kernels.chain_segments``: the node
    sequences of the chains through ``seg_nodes`` (pairs of node indices)."""
    incident: dict[int, list[int]] = {}
    for index, (a, b) in enumerate(seg_nodes):
        incident.setdefault(a, []).append(index)
        incident.setdefault(b, []).append(index)
    used = [False] * len(seg_nodes)

    def walk(index: int, at: int) -> list[int]:
        path = [at]
        while not used[index]:
            used[index] = True
            a, b = seg_nodes[index]
            at = b if a == at else a
            path.append(at)
            following = incident[at]
            if len(following) != 2:
                break
            index = following[0] if following[1] == index else following[1]
        return path

    chains: list[list[int]] = []
    for at, indices in incident.items():
        if len(indices) != 2:  # the end of an open chain, or a junction
            chains += [walk(index, at) for index in indices if not used[index]]
    for index, (start, _) in enumerate(seg_nodes):  # closed loops remain
        if not used[index]:
            chains.append(walk(index, start))
    return chains


def _simplify(
    points: Array, tolerance: float, starts: NDArray[np.intp] | None = None
) -> NDArray[np.intp]:
    """Indices of the vertices to keep so the polyline stays within ``tolerance``.

    Douglas–Peucker, splitting every unresolved span in the same vectorised pass. The
    distance is to the chord *segment*, so reversals (hairpins) are kept. After
    ``SIMPLIFY_PASSES`` passes the remaining spans keep all their vertices.

    ``points`` may be several polylines one after another: ``starts`` are the indices
    where each begins, followed by ``len(points)``. Their end points are always kept.
    """
    if native.kernels is not None:
        bounds = np.empty(0, dtype=np.int64) if starts is None else starts.astype(np.int64)
        kept: NDArray[np.intp] = native.kernels.simplify(
            np.ascontiguousarray(points, dtype=np.float64),
            tolerance,
            bounds,
            SIMPLIFY_SPAN,
            SIMPLIFY_PASSES,
        )
        return kept
    return _simplify_numpy(points, tolerance, starts)


def _simplify_numpy(
    points: Array, tolerance: float, starts: NDArray[np.intp] | None = None
) -> NDArray[np.intp]:
    """Reference implementation of :func:`_simplify` (one vectorised pass per level)."""
    n = len(points)
    if n <= 2:
        return np.arange(n)
    keep = np.zeros(n, dtype=bool)
    # Short starting spans bound the passes needed, even for curves that retrace
    # themselves (which split very unevenly).
    keep[::SIMPLIFY_SPAN] = True
    keep[-1] = True
    if starts is not None:
        keep[starts[:-1]] = True
        keep[starts[1:] - 1] = True
    positions = np.arange(n)
    for _ in range(SIMPLIFY_PASSES):
        kept = np.flatnonzero(keep)
        span = np.minimum(np.searchsorted(kept, positions, side="right") - 1, kept.size - 2)
        a, b = points[kept[span]], points[kept[span + 1]]
        chord = b - a
        length2 = np.einsum("ij,ij->i", chord, chord)
        with np.errstate(invalid="ignore", divide="ignore"):
            t = np.clip(np.einsum("ij,ij->i", points - a, chord) / length2, 0.0, 1.0)
        t[length2 == 0] = 0.0
        offset = points - (a + t[:, None] * chord)
        distance = np.einsum("ij,ij->i", offset, offset)
        distance[keep] = 0.0
        worst = np.maximum.reduceat(distance, kept[:-1])
        unresolved = worst > tolerance * tolerance
        if not unresolved.any():
            return kept
        # The farthest vertex of every unresolved span (the first, if tied) is kept.
        farthest = np.flatnonzero(unresolved[span] & (distance == worst[span]))
        keep[farthest[np.unique(span[farthest], return_index=True)[1]]] = True
    keep[unresolved[span]] = True  # spans still unresolved after the last pass
    return np.flatnonzero(keep)


@dataclass(slots=True)
class _Result:
    request: TileRequest
    geometry: Geometry | None = None
    outline: tuple[Array, ...] | None = None
    screen: ScreenGeometry | None = None
    error: Exception | None = None


class _SampleJob:
    """Samples one tile on a worker thread; the result is posted to the event loop."""

    def __init__(
        self, request: TileRequest, lifetime: Event, finished: Callable[[_Result], None]
    ) -> None:
        self.request = request
        self.cancelled = Event()
        self.lifetime = lifetime
        self.finished = finished

    def obsolete(self) -> bool:
        return self.cancelled.is_set() or self.lifetime.is_set()

    def run(self) -> None:
        result = _Result(self.request)
        try:
            with cancellable(self.obsolete):
                geometry = self.request.plot.sample(self.request.view)
                checkpoint()
                result.geometry = visible_geometry(geometry, self.request.view)
                result.outline = tuple(outline(result.geometry, self.request.view))
                checkpoint()
                result.screen = screen_geometry(
                    result.geometry, self.request.view, self.request.scale, result.outline
                )
        except Cancelled:
            pass
        except Exception as error:  # noqa: BLE001 - isolate failures to the offending tile
            result.error = error
        events.post(partial(self.finished, result))


class GeometryLoader:
    def __init__(self) -> None:
        self.changed = events.Signal()
        self.cache_limit = CACHE_TILES
        self.byte_limit = CACHE_BYTES
        self._cache: OrderedDict[TileKey, GeometryTile] = OrderedDict()
        self._bytes = 0
        self._desired: dict[TileKey, TileRequest] = {}
        self._visible: dict[TileKey, GeometryTile] = {}
        self._active: dict[TileKey, _SampleJob] = {}
        self._lifetime = Event()

    @property
    def loading(self) -> bool:
        return any(key not in self._visible for key in self._desired)

    @property
    def tiles(self) -> list[GeometryTile]:
        return [self._visible[key] for key in self._desired if key in self._visible]

    def request(self, layers: Sequence[tuple[Hashable, Plot]], view: ViewRect) -> None:
        if self._lifetime.is_set():
            return
        specs = [spec for key, plot in layers for spec in tile_requests(key, plot, view)]
        # Nearby tiles arrive first, interleaving layers at each distance.
        cx, cy = 0.5 * view.u_min + 0.5 * view.u_max, 0.5 * view.v_min + 0.5 * view.v_max
        shifts = {spec.view: origin_offset(spec.view, view) for spec in specs}

        def distance(spec: TileRequest) -> float:
            ox, oy = shifts[spec.view]
            du = ox + 0.5 * spec.view.u_min + 0.5 * spec.view.u_max - cx
            dv = oy + 0.5 * spec.view.v_min + 0.5 * spec.view.v_max - cy
            return (du * view.px_per_x) ** 2 + (dv * view.px_per_y) ** 2

        specs.sort(key=distance)
        desired = {spec.key: spec for spec in specs}
        if desired.keys() == self._desired.keys():
            return
        self._desired = desired
        self._visible = {key: tile for key, tile in self._visible.items() if key in desired}
        identities = {(key, id(part)) for key, plot in layers for part in members(plot)}
        for key in list(self._cache):
            if key[:2] not in identities:
                self._bytes -= self._cache.pop(key).cost
        for key in desired:
            if key in self._cache:
                self._visible[key] = self._cache[key]
                self._cache.move_to_end(key)
        for key, job in self._active.items():
            if key not in desired:
                job.cancelled.set()
        self._pump()

    def _pump(self) -> None:
        if self._lifetime.is_set():
            return
        for key, spec in self._desired.items():
            if len(self._active) >= WORKERS:
                break
            if key in self._visible or key in self._active:
                continue
            job = _SampleJob(spec, self._lifetime, self._completed)
            self._active[key] = job
            worker_pool().submit(job.run)

    def _completed(self, result: _Result) -> None:
        if self._lifetime.is_set():
            return
        key = result.request.key
        job = self._active.pop(key)
        if not job.obsolete() and key in self._desired:
            if result.error is not None:
                log.error("Sampling failed for layer %r: %s", result.request.layer, result.error)
            tile = GeometryTile(
                result.request, result.geometry or Geometry(), result.outline, result.screen
            )
            self._visible[key] = tile
            self._cache[key] = tile
            self._bytes += tile.cost
            while self._cache and (
                len(self._cache) > self.cache_limit or self._bytes > self.byte_limit
            ):
                _, removed = self._cache.popitem(last=False)
                self._bytes -= removed.cost
        self._pump()
        self.changed.emit()

    def stop(self) -> None:
        self._lifetime.set()
        self._desired.clear()
        self._visible.clear()
        self._cache.clear()
        self._bytes = 0
