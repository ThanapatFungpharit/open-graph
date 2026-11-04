"""Draw a 3D scene into an image: a software z-buffer in NumPy. Toolkit-free.

Flet's canvas has no way to fill thousands of triangles quickly, and surfaces that
intersect each other can't be drawn correctly by sorting whole triangles anyway. So a
frame is computed here, per pixel, and shown as one image.

Everything is batched rather than looped per triangle. All candidate pixels of a batch
of triangles are generated at once (the triangles' bounding boxes), :func:`numpy.maximum.at`
resolves the depth test, and only the fragments that won are shaded. That costs about
as much as a few array operations per covered pixel, so a typical surface takes tens of
milliseconds. Lines and points are stamped along their length into the same depth buffer,
so a curve passing behind a surface is hidden by it.

Normals are in the box's normalized space, and the box is always drawn as a cube, so
the lighting matches the shape on the screen even when the axes have different ranges.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray

from opengraph.engine.builtins import Array
from opengraph.engine.cancellation import checkpoint
from opengraph.engine.meshing import Box, Indices, Mesh
from opengraph.engine.plots import SpaceGeometry
from opengraph.graph.ticks import choose_spacing, tick_indices

RGB = tuple[int, int, int]

EYE_DISTANCE = 6.0  # from the centre of the box, in half-widths of the box
SCREEN_FILL = 0.25  # half the box's width on screen, as a fraction of the shorter side
MAX_BATCH_PIXELS = 1_500_000  # candidate pixels handled at once, which bounds memory
AMBIENT = 0.38  # the rest of the brightness comes from the light
LIGHT = (0.35, 0.55, 0.76)  # towards the light: right, up, towards the viewer
GRID_ON_SURFACE = 0.74  # brightness of the lines drawn on surfaces
GRID_MIN_PX = 40.0
CURVE_RADIUS = 1.25  # pixels
POINT_RADIUS = 5.0
GRID_RADIUS = 0.5
AXIS_RADIUS = 0.75
SURFACE_LINE_PX = 0.7  # half the width of the lines drawn on surfaces
SURFACE_LINE_LIMIT = 0.2  # at most this share of a grid spacing, on surfaces seen edge-on
OVERDRAW = 1.02  # how far past the box (normalized) an outdated mesh may still be drawn
_LINE_BIAS = 0.004  # lines win over surfaces they lie on
_WALL_BIAS = -0.008  # the walls lose to anything lying on them


@dataclass(frozen=True, slots=True)
class Camera:
    """How the box is seen: from a direction (radians), always centred on the screen."""

    box: Box = Box(-5.0, 5.0, -5.0, 5.0, -5.0, 5.0)
    azimuth: float = math.radians(35.0)  # around z, from the x-axis towards the y-axis
    elevation: float = math.radians(24.0)  # above the xy-plane

    def basis(self) -> Array:
        """The rows are the world directions of screen right, screen up and the viewer."""
        ca, sa = math.cos(self.azimuth), math.sin(self.azimuth)
        ce, se = math.cos(self.elevation), math.sin(self.elevation)
        return np.array(
            [
                [-sa, ca, 0.0],
                [-se * ca, -se * sa, ce],
                [ce * ca, ce * sa, se],
            ]
        )

    def project(self, points: Array, width: int, height: int) -> tuple[Array, Array, Array]:
        """Screen x and y (y down) and depth (larger is nearer) of world ``points``."""
        return self.project_normalized(self.box.normalized(points), width, height)

    def project_normalized(
        self, normalized: Array, width: int, height: int
    ) -> tuple[Array, Array, Array]:
        view = normalized @ self.basis().T
        scale = pixels_per_half_box(width, height)
        with np.errstate(divide="ignore", invalid="ignore"):
            perspective = EYE_DISTANCE / (EYE_DISTANCE - view[..., 2])
        return (
            width / 2 + scale * perspective * view[..., 0],
            height / 2 - scale * perspective * view[..., 1],
            view[..., 2],
        )


def pixels_per_half_box(width: int, height: int) -> float:
    return SCREEN_FILL * min(width, height)


def tick_step(lo: float, hi: float, width: int, height: int, custom: float | None = None) -> float:
    """The major grid spacing of an axis showing ``[lo, hi]``."""
    px_per_unit = 2 * pixels_per_half_box(width, height) / (hi - lo)
    if custom is not None and custom * px_per_unit >= GRID_MIN_PX / 4:
        return custom
    return choose_spacing(px_per_unit, GRID_MIN_PX).major


@dataclass(frozen=True, slots=True)
class Style:
    background: RGB = (255, 255, 255)
    grid: RGB = (217, 220, 225)
    axis: RGB = (47, 50, 55)
    show_grid: bool = True
    show_axes: bool = True
    steps: tuple[float | None, float | None, float | None] = (None, None, None)  # custom spacing


@dataclass(frozen=True, slots=True)
class Part:
    """Geometry to draw and its colour."""

    geometry: SpaceGeometry
    color: RGB


class _Target:
    """The pixels being drawn, and how near the surface at each pixel is."""

    def __init__(self, width: int, height: int, sampling: int, camera: Camera, background: RGB):
        self.logical = (width, height)
        self.sampling = sampling
        self.width, self.height = width * sampling, height * sampling
        self.camera = camera
        self.depth = np.full(self.width * self.height, -np.inf, dtype=np.float32)
        self.color = np.empty((self.width * self.height, 3), dtype=np.uint8)
        self.color[:] = background

    def project(self, points: Array) -> tuple[Array, Array, Array]:
        return self.camera.project(points, self.width, self.height)

    def winners(self, indices: NDArray[np.int64], depths: NDArray[np.float32]) -> NDArray[np.bool_]:
        """Run fragments through the depth test: which ones are nearest at their pixel."""
        np.maximum.at(self.depth, indices, depths)
        return np.asarray(depths >= self.depth[indices])

    def image(self) -> NDArray[np.uint8]:
        pixels = self.color.reshape(self.height, self.width, 3)
        if self.sampling == 1:
            return pixels
        s = self.sampling
        h, w = self.logical[1], self.logical[0]
        blocks = pixels.reshape(h, s, w, s, 3).astype(np.uint16)
        return np.asarray((blocks.sum(axis=(1, 3)) + s * s // 2) // (s * s), dtype=np.uint8)


def _shade(mesh: Mesh, camera: Camera, color: RGB) -> NDArray[np.float32]:
    """Colours per vertex, lit from both sides so the underside of a surface shows too."""
    basis = camera.basis()
    light = np.array(LIGHT, dtype=np.float32) @ basis.astype(np.float32)
    light /= np.linalg.norm(light)
    facing = np.abs(mesh.normals @ light)
    # A fainter light from the viewer keeps surfaces edge-on to the main light readable.
    towards = np.abs(mesh.normals @ basis[2].astype(np.float32))
    brightness = AMBIENT + (1 - AMBIENT) * (0.78 * facing + 0.22 * towards)
    return np.asarray(brightness[:, None] * np.array(color, dtype=np.float32), dtype=np.float32)


def _line_widths(
    coordinates: Array,
    triangles: Indices,
    ax: Array,
    ay: Array,
    bx: Array,
    by: Array,
    cx: Array,
    cy: Array,
    area: Array,
) -> Array:
    """How fast each grid coordinate changes per screen pixel, on each triangle (T, 3).

    A line is where a coordinate is near a multiple of the spacing, so this is what keeps
    lines one width on screen. A fixed margin in world units would smear into a blob where
    a surface is nearly level with the line, as at the top of a hill.
    """
    a, b, c = (coordinates[triangles[:, i]] for i in range(3))
    per_x = (a.T * (by - cy) + b.T * (cy - ay) + c.T * (ay - by)) / area
    per_y = (a.T * (cx - bx) + b.T * (ax - cx) + c.T * (bx - ax)) / area
    widths: Array = np.hypot(per_x, per_y).T
    return widths


def _draw_mesh(
    target: _Target, mesh: Mesh, color: RGB, steps: tuple[float, float, float] | None
) -> None:
    box = target.camera.box
    normalized = box.normalized(mesh.vertices)
    triangles = mesh.triangles
    # A mesh sampled for a larger box stays drawn while the new one loads.
    near = (np.abs(normalized) <= OVERDRAW).all(axis=1)
    triangles = triangles[near[triangles].all(axis=1)]
    if not len(triangles):
        return
    sx, sy, sd = target.camera.project_normalized(normalized, target.width, target.height)
    ax, ay, bx, by, cx, cy = (
        sx[triangles[:, 0]],
        sy[triangles[:, 0]],
        sx[triangles[:, 1]],
        sy[triangles[:, 1]],
        sx[triangles[:, 2]],
        sy[triangles[:, 2]],
    )
    area = (bx - ax) * (cy - ay) - (cx - ax) * (by - ay)
    left = np.floor(np.minimum(np.minimum(ax, bx), cx)).astype(np.int64)
    right = np.ceil(np.maximum(np.maximum(ax, bx), cx)).astype(np.int64)
    top = np.floor(np.minimum(np.minimum(ay, by), cy)).astype(np.int64)
    bottom = np.ceil(np.maximum(np.maximum(ay, by), cy)).astype(np.int64)
    left, top = np.maximum(left, 0), np.maximum(top, 0)
    right, bottom = np.minimum(right, target.width - 1), np.minimum(bottom, target.height - 1)
    widths, heights = right - left + 1, bottom - top + 1
    keep = (np.abs(area) > 1e-9) & (widths > 0) & (heights > 0)
    triangles, area = triangles[keep], area[keep]
    if not len(triangles):
        return
    ax, ay, bx, by, cx, cy = ax[keep], ay[keep], bx[keep], by[keep], cx[keep], cy[keep]
    left, top, widths, heights = left[keep], top[keep], widths[keep], heights[keep]
    counts = widths * heights
    shades = _shade(mesh, target.camera, color)
    vertex_depths = sd.astype(np.float32)
    grid = None
    if steps is not None:
        coordinates = mesh.vertices / np.array(steps)
        grid = coordinates, _line_widths(coordinates, triangles, ax, ay, bx, by, cx, cy, area)
        grid = grid[0], grid[1] * SURFACE_LINE_PX * target.sampling

    # Split into batches of a bounded number of candidate pixels.
    ends = np.cumsum(counts)
    start_at = 0
    first = 0
    while first < len(triangles):
        checkpoint()
        last = int(np.searchsorted(ends, start_at + MAX_BATCH_PIXELS, side="right"))
        last = max(last, first + 1)
        batch = np.arange(first, last)
        n = counts[batch]
        total = int(n.sum())
        owner = np.repeat(batch, n)
        offset = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(n) - n, n)
        row_width = widths[owner]
        px = left[owner] + offset % row_width
        py = top[owner] + offset // row_width
        x, y = px + 0.5, py + 0.5
        share = area[owner]
        la = ((bx[owner] - x) * (cy[owner] - y) - (cx[owner] - x) * (by[owner] - y)) / share
        lb = ((cx[owner] - x) * (ay[owner] - y) - (ax[owner] - x) * (cy[owner] - y)) / share
        lc = 1.0 - la - lb
        inside = (la >= -1e-3) & (lb >= -1e-3) & (lc >= -1e-3)
        owner, px, py = owner[inside], px[inside], py[inside]
        la, lb, lc = (
            la[inside].astype(np.float32),
            lb[inside].astype(np.float32),
            lc[inside].astype(np.float32),
        )
        ia, ib, ic = triangles[owner, 0], triangles[owner, 1], triangles[owner, 2]
        depths = la * vertex_depths[ia] + lb * vertex_depths[ib] + lc * vertex_depths[ic]
        indices = py * target.width + px
        won = target.winners(indices, depths)
        la, lb, lc = la[won, None], lb[won, None], lc[won, None]
        ia, ib, ic = ia[won], ib[won], ic[won]
        shaded = la * shades[ia] + lb * shades[ib] + lc * shades[ic]
        if grid is not None:
            coordinates, half_width = grid
            position = la * coordinates[ia] + lb * coordinates[ib] + lc * coordinates[ic]
            limit = np.minimum(half_width[owner[won]], SURFACE_LINE_LIMIT)
            on_line = (np.abs(position - np.round(position)) < limit).any(axis=1)
            shaded[on_line] *= GRID_ON_SURFACE
        target.color[indices[won]] = np.clip(shaded, 0, 255).astype(np.uint8)
        first = last
        start_at = int(ends[last - 1])


def _disc(radius: float) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Pixel offsets of a filled circle."""
    reach = max(0, math.ceil(radius - 0.5))
    span = np.arange(-reach, reach + 1, dtype=np.int64)
    dx, dy = np.meshgrid(span, span)
    keep = dx**2 + dy**2 <= max(radius, 0.5) ** 2
    return dx[keep], dy[keep]


def _stamp(
    target: _Target,
    x: Array,
    y: Array,
    depth: Array,
    color: RGB,
    radius: float,
) -> None:
    """Draw a filled circle of ``radius`` pixels at each of the given positions."""
    dx, dy = _disc(radius * target.sampling)
    px = np.floor(x).astype(np.int64)[:, None] + dx
    py = np.floor(y).astype(np.int64)[:, None] + dy
    depths = np.repeat(depth.astype(np.float32), len(dx))
    px, py = px.ravel(), py.ravel()
    on_screen = (px >= 0) & (px < target.width) & (py >= 0) & (py < target.height)
    indices = py[on_screen] * target.width + px[on_screen]
    depths = depths[on_screen]
    won = target.winners(indices, depths)
    target.color[indices[won]] = color


def _draw_segments(
    target: _Target, segments: Array, color: RGB, radius: float, bias: float = _LINE_BIAS
) -> None:
    """Draw segments (M, 2, 3) as round-capped lines, stamped about once a pixel."""
    if not len(segments):
        return
    box = target.camera.box
    normalized = box.normalized(segments)
    near = (np.abs(normalized) <= OVERDRAW).all(axis=(1, 2))
    if not near.any():
        return
    sx, sy, sd = target.camera.project_normalized(normalized[near], target.width, target.height)
    length = np.hypot(sx[:, 1] - sx[:, 0], sy[:, 1] - sy[:, 0])
    n = np.ceil(length).astype(np.int64) + 1
    owner = np.repeat(np.arange(len(n)), n)
    offset = np.arange(int(n.sum()), dtype=np.int64) - np.repeat(np.cumsum(n) - n, n)
    along = offset / np.maximum(n[owner] - 1, 1)
    x = sx[owner, 0] + along * (sx[owner, 1] - sx[owner, 0])
    y = sy[owner, 0] + along * (sy[owner, 1] - sy[owner, 0])
    depth = sd[owner, 0] + along * (sd[owner, 1] - sd[owner, 0]) + bias
    _stamp(target, x, y, depth, color, radius)


def _draw_points(target: _Target, points: Array, color: RGB, background: RGB) -> None:
    x, y, depth = target.project(points)
    # A ring in the background colour, then the dot, as on the flat canvas.
    _stamp(target, x, y, depth + 2 * _LINE_BIAS, background, POINT_RADIUS + 1.5)
    _stamp(target, x, y, depth + 3 * _LINE_BIAS, color, POINT_RADIUS)


def _ticks(lo: float, hi: float, step: float) -> Array:
    return np.asarray(tick_indices(lo, hi, step), dtype=np.float64) * step


def _steps(camera: Camera, style: Style, width: int, height: int) -> tuple[float, float, float]:
    box = camera.box
    x, y, z = (tick_step(*box.range(axis), width, height, style.steps[axis]) for axis in range(3))
    return x, y, z


def _draw_backdrop(target: _Target, style: Style, steps: tuple[float, float, float]) -> None:
    """The grid on the three walls behind the scene, the box's edges and the axes."""
    camera = target.camera
    box = camera.box
    lows, highs = box.lows, box.highs
    toward = camera.basis()[2]
    far = np.where(toward > 0, lows, highs)
    lines: list[Array] = []
    if style.show_grid:
        for wall in range(3):
            others = [axis for axis in range(3) if axis != wall]
            for along, across in (others, others[::-1]):
                ticks = _ticks(lows[across], highs[across], steps[across])
                if not len(ticks):
                    continue
                start = np.tile(far, (len(ticks), 1))
                start[:, across] = ticks
                start[:, along] = lows[along]
                end = start.copy()
                end[:, along] = highs[along]
                lines.append(np.stack([start, end], axis=1))
    if lines:
        _draw_segments(target, np.vstack(lines), style.grid, GRID_RADIUS, _WALL_BIAS)
    if not style.show_axes:
        return
    # The three axes where they pass through the box, and the edges of the walls.
    origin = np.zeros(3)
    axes = []
    for axis in range(3):
        others = [a for a in range(3) if a != axis]
        if all(lows[a] <= 0 <= highs[a] for a in others):
            start, end = origin.copy(), origin.copy()
            start[axis], end[axis] = lows[axis], highs[axis]
            axes.append(np.stack([start, end]))
    if axes:
        _draw_segments(target, np.stack(axes), style.axis, AXIS_RADIUS)
    edges = []
    for axis in range(3):
        for wall in range(3):
            if wall == axis:
                continue
            other = 3 - axis - wall
            for value in (lows[other], highs[other]):
                start = far.copy()
                start[other] = value
                start[axis] = lows[axis]
                end = start.copy()
                end[axis] = highs[axis]
                edges.append(np.stack([start, end]))
    _draw_segments(target, np.stack(edges), style.grid, AXIS_RADIUS, _WALL_BIAS)


def render(
    parts: tuple[Part, ...],
    camera: Camera,
    width: int,
    height: int,
    style: Style,
    sampling: int = 1,
) -> NDArray[np.uint8]:
    """The scene as an ``(height, width, 3)`` image; ``sampling`` is the number of samples
    per pixel along each side (2 smooths the edges, at about four times the work)."""
    target = _Target(width, height, sampling, camera, style.background)
    steps = _steps(camera, style, width, height)
    _draw_backdrop(target, style, steps)
    surface_steps = steps if style.show_grid else None
    for part in parts:
        checkpoint()
        mesh = part.geometry.mesh
        if mesh is not None and not mesh.is_empty:
            _draw_mesh(target, mesh, part.color, surface_steps)
        segments = part.geometry.segments
        if segments is not None and len(segments):
            _draw_segments(target, segments, part.color, CURVE_RADIUS)
        points = part.geometry.points
        if points is not None and len(points):
            _draw_points(target, points, part.color, style.background)
    return target.image()


def encode(pixels: NDArray[np.uint8], *, still: bool) -> bytes:
    """A frame as image data: quick JPEG while the view moves, exact PNG once it rests."""
    from PIL import Image

    buffer = io.BytesIO()
    image = Image.fromarray(pixels)
    if still:
        image.save(buffer, "PNG", compress_level=3)
    else:
        image.save(buffer, "JPEG", quality=88, subsampling=0)
    return buffer.getvalue()


def rotated(camera: Camera, azimuth: float, elevation: float) -> Camera:
    """``camera`` turned by the given angles; it can't go over the top of the scene."""
    limit = math.radians(89.0)
    return replace(
        camera,
        azimuth=math.remainder(camera.azimuth + azimuth, math.tau),
        elevation=min(max(camera.elevation + elevation, -limit), limit),
    )
