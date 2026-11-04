"""Turn functions of two and three variables into triangles and line segments.

Three algorithms, all vectorised, in the style of :mod:`~opengraph.engine.sampling`:

* :func:`height_field` — a surface ``w = f(u, v)`` on a regular grid. It follows the
  edge of the function's domain by bisection, removes the false walls that a jump or a
  pole would otherwise leave between grid points, and clips to the box exactly.

* :func:`implicit_surface` — the zero set of ``F(x, y, z)`` by marching tetrahedra, with
  a filter that rejects sign changes caused by poles (as the implicit curves do).

* :func:`curve_segments` — a parametric curve, refined where its chords are long and
  broken at discontinuities.

Everything outside the :class:`Box` is cut away here, once, when a plot is sampled, so
turning the view only has to redraw the same triangles.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from opengraph.engine.builtins import Array
from opengraph.engine.cancellation import checkpoint

Indices = NDArray[np.int64]

_DOMAIN_BISECTIONS = 10  # steps locating the edge of a function's domain
_JUMP_HEIGHT = 0.05  # of the box: a shorter step between grid points is never a jump
_JUMP_SHARE = 0.9  # of the step: this much of it in half an edge means a jump
_WELD_BITS = 20  # vertices closer than 2^-19 of the box's half-width are merged
_CURVE_SAMPLES = 1025
_CURVE_PASSES = 6
_CURVE_MAX_POINTS = 40_000
_CURVE_CHORD = 0.02  # longest chord kept without refining, as a fraction of the box

# The six tetrahedra around a cube's diagonal (Kuhn's subdivision), as cube corners
# numbered dx + 2 dy + 4 dz. Neighbouring cubes agree on their shared face diagonals.
_TETRAHEDRA = np.array(
    [[0, 1, 3, 7], [0, 1, 5, 7], [0, 2, 3, 7], [0, 2, 6, 7], [0, 4, 5, 7], [0, 4, 6, 7]]
)
_CORNERS = np.array([[(c >> axis) & 1 for axis in range(3)] for c in range(8)])


@dataclass(frozen=True, slots=True)
class Box:
    """The region of space shown by a 3D view."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    @property
    def lows(self) -> Array:
        return np.array([self.x_min, self.y_min, self.z_min])

    @property
    def highs(self) -> Array:
        return np.array([self.x_max, self.y_max, self.z_max])

    def range(self, axis: int) -> tuple[float, float]:
        return (
            (self.x_min, self.x_max),
            (self.y_min, self.y_max),
            (self.z_min, self.z_max),
        )[axis]

    def normalized(self, points: Array) -> Array:
        """``points`` with the box mapped to the cube [-1, 1]³, the shape it is drawn in."""
        lows, highs = self.lows, self.highs
        return (points - (lows + highs) / 2) / ((highs - lows) / 2)


@dataclass(frozen=True, slots=True)
class Mesh:
    """Triangles in world coordinates. ``normals`` are unit vectors per vertex, in the
    box's normalized space (where the lighting happens)."""

    vertices: Array  # (V, 3)
    triangles: Indices  # (T, 3)
    normals: NDArray[np.float32]  # (V, 3)

    @property
    def is_empty(self) -> bool:
        return len(self.triangles) == 0


def _rotate_rows(triangles: Indices, shift: Indices) -> Indices:
    """Each row turned left by ``shift`` places, which keeps its orientation."""
    order = (shift[:, None] + np.arange(3)) % 3
    return np.take_along_axis(triangles, order, axis=1)


def _compact(vertices: Array, triangles: Indices) -> tuple[Array, Indices]:
    """Only the vertices the triangles use, renumbered."""
    used, inverse = np.unique(triangles, return_inverse=True)
    return vertices[used], inverse.reshape(triangles.shape).astype(np.int64)


def _clip_side(
    vertices: Array, triangles: Indices, axis: int, limit: float, sign: float
) -> tuple[Array, Indices]:
    """Keep the part of the triangles where ``sign * (coordinate - limit) >= 0``."""
    distance = sign * (vertices[:, axis] - limit)
    inside = distance[triangles] >= 0
    count = inside.sum(axis=1)
    kept = [triangles[count == 3]]
    new_points: list[Array] = []
    next_index = len(vertices)

    def crossing(a: Indices, b: Indices) -> Indices:
        """New vertices where the edges from inside ``a`` to outside ``b`` cross."""
        nonlocal next_index
        share = distance[a] / (distance[a] - distance[b])
        new_points.append(vertices[a] + share[:, None] * (vertices[b] - vertices[a]))
        indices = np.arange(next_index, next_index + len(a), dtype=np.int64)
        next_index += len(a)
        return indices

    one = count == 1
    if one.any():  # turn each so its inside vertex comes first: one smaller triangle
        turned = _rotate_rows(triangles[one], np.argmax(inside[one], axis=1))
        a, b, c = turned.T
        kept.append(np.column_stack([a, crossing(a, b), crossing(a, c)]))
    two = count == 2
    if two.any():  # turn each so its outside vertex comes last: a quadrilateral remains
        shift = (np.argmin(inside[two], axis=1) + 1) % 3
        turned = _rotate_rows(triangles[two], shift)
        a, b, c = turned.T
        bc, ac = crossing(b, c), crossing(a, c)
        kept += [np.column_stack([a, b, bc]), np.column_stack([a, bc, ac])]
    if new_points:
        vertices = np.vstack([vertices, *new_points])
    return vertices, np.vstack(kept)


def clip_to_range(
    vertices: Array, triangles: Indices, axis: int, lo: float, hi: float
) -> tuple[Array, Indices]:
    """Cut triangles to ``lo <= coordinate <= hi`` along ``axis``."""
    vertices, triangles = _clip_side(vertices, triangles, axis, lo, 1.0)
    return _clip_side(vertices, triangles, axis, hi, -1.0)


def vertex_normals(vertices: Array, triangles: Indices) -> NDArray[np.float32]:
    """Area-weighted unit normals; triangles must be consistently oriented."""
    a, b, c = (vertices[triangles[:, i]] for i in range(3))
    faces = np.cross(b - a, c - a)
    normals = np.zeros_like(vertices)
    for i in range(3):
        np.add.at(normals, triangles[:, i], faces)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths[lengths == 0] = 1.0
    return np.asarray(normals / lengths, dtype=np.float32)


def _weld(vertices: Array, triangles: Indices, box: Box) -> tuple[Array, Indices]:
    """Merge vertices at the same place, and drop the triangles that collapse.

    A surface through a grid point meets every grid edge there at that one point. Left
    apart, the vertices would share only triangles without area, so no normal, and show
    as a dark spot on the surface.
    """
    levels = 1 << _WELD_BITS
    scaled = np.round((box.normalized(vertices) + 1) * (levels / 2)).astype(np.int64)
    a, b, c = np.clip(scaled, 0, levels - 1).T
    keys = (a << (2 * _WELD_BITS)) | (b << _WELD_BITS) | c
    _, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
    welded = inverse.reshape(-1)[triangles]
    whole = (
        (welded[:, 0] != welded[:, 1])
        & (welded[:, 1] != welded[:, 2])
        & (welded[:, 2] != welded[:, 0])
    )
    return vertices[first], welded[whole].astype(np.int64)


def mesh_from(vertices: Array, triangles: Indices, box: Box) -> Mesh:
    """A mesh of ``triangles``, without loose or duplicate vertices."""
    empty = Mesh(np.empty((0, 3)), np.empty((0, 3), dtype=np.int64), np.empty((0, 3), "f4"))
    if not len(triangles):
        return empty
    vertices, triangles = _compact(vertices, triangles)
    vertices, triangles = _weld(vertices, triangles, box)
    if not len(triangles):
        return empty
    vertices, triangles = _compact(vertices, triangles)
    return Mesh(vertices, triangles, vertex_normals(box.normalized(vertices), triangles))


def _jumps(start: Array, middle: Array, end: Array, height: float) -> NDArray[np.bool_]:
    """Grid edges a function jumps across, from its values at the ends and the middle.

    A continuous function changes by about half as much over half the edge, however
    steep it is; across a jump one half holds nearly all of the change.
    """
    step = np.abs(end - start)
    half = np.maximum(np.abs(middle - start), np.abs(end - middle))
    with np.errstate(invalid="ignore"):
        big = step > _JUMP_HEIGHT * height
        return big & (~(half < _JUMP_SHARE * step) | ~np.isfinite(middle))


def _domain_edge(
    evaluate: Callable[[Array, Array], Array],
    inside: Array,
    outside: Array,
) -> Array:
    """Points ``(u, v, w)`` near the end of each segment that leaves the domain, found
    by bisection from the ``(u, v)`` points ``inside`` and ``outside`` it."""
    inside, outside = inside.copy(), outside.copy()
    for _ in range(_DOMAIN_BISECTIONS):
        middle = (inside + outside) / 2
        defined = np.isfinite(evaluate(middle[:, 0], middle[:, 1]))
        inside[defined] = middle[defined]
        outside[~defined] = middle[~defined]
    return np.column_stack([inside, evaluate(inside[:, 0], inside[:, 1])])


def height_field(
    evaluate: Callable[[Array, Array], Array],
    u_range: tuple[float, float],
    v_range: tuple[float, float],
    w_range: tuple[float, float],
    cells: int,
) -> tuple[Array, Indices]:
    """Triangles of ``w = evaluate(u, v)``, as points ``(u, v, w)`` cut to ``w_range``.

    The function is evaluated at twice the resolution of the triangles: the extra points
    are the middles of their edges, which tell a jump from a steep slope.
    """
    samples = 2 * cells + 1
    us = np.linspace(*u_range, samples)
    vs = np.linspace(*v_range, samples)
    grid_u, grid_v = np.meshgrid(us, vs, indexing="ij")
    fine = np.array(evaluate(grid_u, grid_v), dtype=np.float64)
    fine[~np.isfinite(fine)] = np.nan
    checkpoint()
    height = w_range[1] - w_range[0]

    # Whether the function jumps along each kind of edge of each cell.
    along_u = _jumps(fine[:-2:2, ::2], fine[1::2, ::2], fine[2::2, ::2], height)
    along_v = _jumps(fine[::2, :-2:2], fine[::2, 1::2], fine[::2, 2::2], height)
    diagonal = _jumps(fine[:-2:2, :-2:2], fine[1::2, 1::2], fine[2::2, 2::2], height)
    # Each cell splits along its diagonal into (i, j)(i+1, j)(i+1, j+1) and
    # (i, j)(i+1, j+1)(i, j+1).
    lower_ok = ~(along_u[:, :-1] | along_v[1:, :] | diagonal)
    upper_ok = ~(along_v[:-1, :] | along_u[:, 1:] | diagonal)

    nodes = cells + 1
    number = np.arange(nodes * nodes, dtype=np.int64).reshape(nodes, nodes)
    a, b, c, d = number[:-1, :-1], number[1:, :-1], number[1:, 1:], number[:-1, 1:]
    triangles = np.vstack(
        [
            np.column_stack([a[lower_ok], b[lower_ok], c[lower_ok]]),
            np.column_stack([a[upper_ok], c[upper_ok], d[upper_ok]]),
        ]
    )
    vertices = np.column_stack(
        [grid_u[::2, ::2].ravel(), grid_v[::2, ::2].ravel(), fine[::2, ::2].ravel()]
    )

    # Triangles partly outside the domain end where the domain does.
    defined = np.isfinite(vertices[:, 2])[triangles]
    count = defined.sum(axis=1)
    pieces = [triangles[count == 3]]
    new_points: list[Array] = []
    total = len(vertices)

    def edge(from_: Indices, to: Indices) -> Indices:
        nonlocal total
        new_points.append(_domain_edge(evaluate, vertices[from_, :2], vertices[to, :2]))
        indices = np.arange(total, total + len(from_), dtype=np.int64)
        total += len(from_)
        return indices

    one = count == 1
    if one.any():
        p, q, r = _rotate_rows(triangles[one], np.argmax(defined[one], axis=1)).T
        pieces.append(np.column_stack([p, edge(p, q), edge(p, r)]))
    two = count == 2
    if two.any():
        shift = (np.argmin(defined[two], axis=1) + 1) % 3
        p, q, r = _rotate_rows(triangles[two], shift).T
        qr, pr = edge(q, r), edge(p, r)
        pieces += [np.column_stack([p, q, qr]), np.column_stack([p, qr, pr])]
    if new_points:
        vertices = np.vstack([vertices, *new_points])
    triangles = np.vstack(pieces)
    usable = np.isfinite(vertices[:, 2])[triangles].all(axis=1)
    return clip_to_range(vertices, triangles[usable], 2, *w_range)


def implicit_surface(
    evaluate: Callable[[Array, Array, Array], Array], box: Box, cells: int
) -> tuple[Array, Indices]:
    """Triangles of the zero set of ``evaluate(x, y, z)`` in ``box``, by marching
    tetrahedra. Triangles are oriented with their normals towards positive values."""
    nodes = cells + 1
    axes = [np.linspace(*box.range(axis), nodes) for axis in range(3)]
    grid = np.meshgrid(*axes, indexing="ij")
    values = np.array(evaluate(*grid), dtype=np.float64)
    checkpoint()
    positive = values > 0
    usable = np.isfinite(values)

    # Only the cubes the surface can pass through.
    def corners(array: NDArray[np.bool_]) -> NDArray[np.int8]:
        total = np.zeros((cells, cells, cells), dtype=np.int8)
        for dx, dy, dz in _CORNERS:
            total += array[dx : cells + dx, dy : cells + dy, dz : cells + dz]
        return total

    above = corners(positive)
    active = np.argwhere((above > 0) & (above < 8) & (corners(usable) == 8))
    empty = np.empty((0, 3)), np.empty((0, 3), dtype=np.int64)
    if not len(active):
        return empty

    def node_ids(position: NDArray[np.int64]) -> Indices:
        return (position[..., 0] * nodes + position[..., 1]) * nodes + position[..., 2]

    cube_nodes = node_ids(active[:, None, :] + _CORNERS[None, :, :])  # (cubes, 8)
    tetrahedra = cube_nodes[:, _TETRAHEDRA].reshape(-1, 4)
    flat_positive = positive.ravel()
    signs = flat_positive[tetrahedra]
    above_count = signs.sum(axis=1)
    keep = (above_count > 0) & (above_count < 4)
    tetrahedra, signs, above_count = tetrahedra[keep], signs[keep], above_count[keep]
    # Corners above zero first, whatever order the tetrahedron lists them in.
    order = np.argsort(~signs, axis=1, kind="stable")
    tetrahedra = np.take_along_axis(tetrahedra, order, axis=1)

    # The edges each triangle's corners sit on, as (corner above zero, corner below).
    edges = []
    owners = []  # the tetrahedron of each triangle, to orient it
    for count, corner_pairs in (
        (1, [[(0, 1), (0, 2), (0, 3)]]),
        (3, [[(0, 3), (1, 3), (2, 3)]]),
        (2, [[(0, 2), (0, 3), (1, 3)], [(0, 2), (1, 3), (1, 2)]]),
    ):
        selected = np.nonzero(above_count == count)[0]
        for pairs in corner_pairs:
            edges.append(tetrahedra[selected][:, np.array(pairs)])  # (N, 3, 2)
            owners.append(selected)
    edge_nodes = np.concatenate(edges)  # (T, 3, 2)
    owner = np.concatenate(owners)

    # One vertex per grid edge, so neighbouring triangles share theirs.
    keys = edge_nodes[..., 0] * nodes**3 + edge_nodes[..., 1]
    unique_keys, triangles = np.unique(keys, return_inverse=True)
    triangles = triangles.reshape(keys.shape).astype(np.int64)
    high, low = unique_keys // nodes**3, unique_keys % nodes**3
    flat = np.column_stack([g.ravel() for g in grid])
    flat_values = values.ravel()

    def bracketed(share: Array) -> Array:
        moved: Array = flat[high] + share[:, None] * (flat[low] - flat[high])
        return moved

    # One step of false position sharpens the linear estimate, and tells poles apart:
    # near a real zero the function is small, near a pole it is larger than at the ends.
    high_values, low_values = flat_values[high], flat_values[low]
    share = high_values / (high_values - low_values)
    points = bracketed(share)
    at_points = np.asarray(evaluate(*points.T), dtype=np.float64)
    checkpoint()
    pole = ~(np.abs(at_points) <= np.maximum(np.abs(high_values), np.abs(low_values)))
    with np.errstate(invalid="ignore", divide="ignore"):
        same_side = at_points > 0
        # Narrow to the half with the sign change, and interpolate again inside it.
        refined = np.where(
            same_side,
            share + (1 - share) * at_points / (at_points - low_values),
            share * high_values / (high_values - at_points),
        )
    refined = np.where(np.isfinite(refined) & (refined >= 0) & (refined <= 1), refined, share)
    points = bracketed(refined)
    real = ~pole[triangles].any(axis=1)
    triangles, owner = triangles[real], owner[real]
    if not len(triangles):
        return empty

    # Orient every triangle towards the tetrahedron's corners above zero.
    corners_at = flat[tetrahedra[owner]]  # (T, 4, 3)
    weights = (np.arange(4)[None, :] < above_count[owner][:, None]).astype(np.float64)
    above_mean = (corners_at * weights[..., None]).sum(axis=1) / weights.sum(axis=1, keepdims=True)
    below_mean = (corners_at * (1 - weights[..., None])).sum(axis=1) / (1 - weights).sum(
        axis=1, keepdims=True
    )
    a, b, c = (points[triangles[:, i]] for i in range(3))
    wrong = (np.cross(b - a, c - a) * (above_mean - below_mean)).sum(axis=1) < 0
    triangles[wrong] = triangles[wrong][:, [0, 2, 1]]
    return points, triangles


def clip_segments(segments: Array, box: Box) -> Array:
    """The parts of ``segments`` (M, 2, 3) inside ``box`` (Liang-Barsky, vectorised)."""
    if not len(segments):
        return segments
    start, delta = segments[:, 0], segments[:, 1] - segments[:, 0]
    enter = np.zeros(len(segments))
    leave = np.ones(len(segments))
    with np.errstate(divide="ignore", invalid="ignore"):
        low_share = (box.lows - start) / delta
        high_share = (box.highs - start) / delta
    first, last = np.minimum(low_share, high_share), np.maximum(low_share, high_share)
    # A segment parallel to a side is entirely inside or outside its slab.
    parallel = delta == 0
    outside = parallel & ((start < box.lows) | (start > box.highs))
    first = np.where(parallel, -np.inf, first)
    last = np.where(parallel, np.inf, last)
    enter = np.maximum(enter, first.max(axis=1))
    leave = np.minimum(leave, last.min(axis=1))
    keep = (enter <= leave) & ~outside.any(axis=1)
    start, delta = start[keep], delta[keep]
    clipped: Array = np.stack(
        [start + enter[keep, None] * delta, start + leave[keep, None] * delta], axis=1
    )
    return clipped


def curve_segments(
    evaluate: Callable[[Array], Array], t_min: float, t_max: float, box: Box
) -> Array:
    """Segments (M, 2, 3) of the curve ``evaluate(t)`` (N, 3) that are inside ``box``.

    Chords are measured in the box's normalized space. Chords outside the box need no
    detail, and a chord still long at the end is kept only if its middle is on the way
    between its ends (otherwise it crosses a discontinuity).
    """

    def needs_detail(points: Array) -> NDArray[np.bool_]:
        q = box.normalized(points)
        a, b = q[:-1], q[1:]
        with np.errstate(invalid="ignore"):
            long = np.linalg.norm(b - a, axis=1) > 2 * _CURVE_CHORD
            beyond = ((a > 1) & (b > 1)) | ((a < -1) & (b < -1))
        return np.asarray(long & ~beyond.any(axis=1))

    ts = np.linspace(t_min, t_max, _CURVE_SAMPLES)
    points = evaluate(ts)
    for _ in range(_CURVE_PASSES):
        checkpoint()
        split = needs_detail(points)
        if not split.any() or len(ts) + split.sum() > _CURVE_MAX_POINTS:
            break
        middle = (ts[:-1][split] + ts[1:][split]) / 2
        ts_merged = np.insert(ts, np.nonzero(split)[0] + 1, middle)
        points = np.insert(points, np.nonzero(split)[0] + 1, evaluate(middle), axis=0)
        ts = ts_merged
    starts, ends = points[:-1], points[1:]
    usable = np.isfinite(starts).all(axis=1) & np.isfinite(ends).all(axis=1)
    long = needs_detail(points) & usable
    if long.any():
        middle = evaluate((ts[:-1][long] + ts[1:][long]) / 2)
        q_start, q_middle, q_end = (
            box.normalized(starts[long]),
            box.normalized(middle),
            box.normalized(ends[long]),
        )
        whole = np.linalg.norm(q_end - q_start, axis=1)
        with np.errstate(invalid="ignore"):
            half = np.maximum(
                np.linalg.norm(q_middle - q_start, axis=1),
                np.linalg.norm(q_end - q_middle, axis=1),
            )
            broken = ~(half < _JUMP_SHARE * whole)
        usable[np.nonzero(long)[0][broken]] = False
    return clip_segments(np.stack([starts[usable], ends[usable]], axis=1), box)
