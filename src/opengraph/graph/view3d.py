"""The state of the 3D view: what it shows, from where, and the frames in progress.

A row that draws in space turns the graph into a 3D view (see :func:`is_spatial`). Two
kinds of background work keep the interface responsive, each replaced as soon as it is
out of date:

* *sampling* turns plots into meshes for the current box. Turning the view doesn't
  change the box, so the meshes survive it; zooming and panning do, and the old meshes
  stay on screen until the new ones arrive.
* *drawing* turns meshes into an image (:mod:`~opengraph.graph.raster`). While the view
  moves, frames are drawn quickly at screen resolution; once it has been still for a
  moment, one finer frame replaces them.

The canvas owns the Flet shapes and the input events; this class has no interface
objects, so the tests can drive it directly.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import partial

import numpy as np

from opengraph import colors, events
from opengraph.engine.builtins import Array
from opengraph.engine.cancellation import Cancelled
from opengraph.engine.meshing import Box
from opengraph.engine.plots import (
    Plot,
    SpaceGeometry,
    is_spatial,
    members,
    space_geometry,
)
from opengraph.graph.background import LatestTask
from opengraph.graph.raster import (
    Camera,
    Part,
    Style,
    encode,
    pixels_per_half_box,
    render,
    rotated,
    tick_step,
)
from opengraph.graph.settings import GraphSettings
from opengraph.graph.ticks import format_coordinate, format_tick, tick_indices

log = logging.getLogger(__name__)

DEFAULT_CAMERA = Camera()
STILL_DELAY_MS = 140  # how long the view must rest before the finer frame is drawn
MAX_MOVING_PIXELS = 520_000  # frames drawn while the view moves are scaled down to this
TURN_PER_PIXEL = math.radians(0.5)
KEY_TURN = math.radians(10.0)
MIN_HALF = 1e-6  # half the width of the box along an axis
MAX_HALF = 1e9
AXES = "xyz"
_CAMERA_KEYS = ("azimuth", "elevation")
_BOX_KEYS = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")
_STATE_PREFIX = "view3d_"


@dataclass(frozen=True, slots=True)
class SpaceLayer:
    key: Hashable
    plot: Plot
    color: str


@dataclass(frozen=True, slots=True)
class Frame:
    """An image of the scene, and what it shows."""

    request: tuple[object, ...]
    camera: Camera
    width: int  # on the screen
    height: int
    data: bytes
    still: bool


@dataclass(frozen=True, slots=True)
class Label:
    """Text on the 3D view, centred at a position in pixels."""

    text: str
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class Picked:
    """A point of a drawn plot near the pointer."""

    position: tuple[float, float, float]
    screen: tuple[float, float]
    color: str
    text: str


def _sample_parts(
    parts: Sequence[tuple[int, Plot]], box: Box
) -> tuple[Box, dict[int, SpaceGeometry]]:
    """Runs on a worker: the geometry of each plot. A plot that fails draws nothing."""
    results: dict[int, SpaceGeometry] = {}
    for identity, plot in parts:
        try:
            results[identity] = space_geometry(plot, box)
        except Cancelled:
            raise
        except Exception:
            log.exception("Sampling a surface failed")
            results[identity] = SpaceGeometry()
    return box, results


def _draw_frame(
    request: tuple[object, ...],
    parts: tuple[Part, ...],
    camera: Camera,
    width: int,
    height: int,
    style: Style,
    still: bool,
) -> Frame:
    """Runs on a worker: one image of the scene."""
    if still:
        pixels = render(parts, camera, width, height, style, sampling=2)
    else:
        scale = min(1.0, math.sqrt(MAX_MOVING_PIXELS / (width * height)))
        moving_width = max(1, round(width * scale))
        moving_height = max(1, round(height * scale))
        pixels = render(parts, camera, moving_width, moving_height, style)
    return Frame(request, camera, width, height, encode(pixels, still=still), still)


class View3D:
    def __init__(self) -> None:
        self.changed = events.Signal()  # something on the view changed: redraw
        self.finished = events.Signal()  # the finest frame of the current view is shown
        self.camera = DEFAULT_CAMERA
        self.frame: Frame | None = None
        self._layers: list[SpaceLayer] = []
        self._active = False
        self._size = (0, 0)
        self._style = Style()
        self._settings = GraphSettings()
        # Each plot's latest geometry, and the box it was sampled for.
        self._geometry: dict[int, tuple[Box, SpaceGeometry]] = {}
        self._sampler = LatestTask()
        self._sampler.ready.connect(self._sampled)
        self._drawer = LatestTask()
        self._drawer.ready.connect(self._drawn)
        self._still_timer = events.Timer(STILL_DELAY_MS, single_shot=True)
        self._still_timer.timeout.connect(self._draw_still)
        self._moving = False  # a drag is in progress: stay on quick frames
        self._drawing_still = False  # the frame requested last is a finer one
        self._sampling_plots: tuple[int, ...] = ()  # the plots requested from the sampler last
        self._revision = 0  # bumped when what the frames show changes, other than the view
        self._picks: tuple[object, list[tuple[Array, Array, Array, Array, str]]] | None = None

    # -- what is shown ---------------------------------------------------------------
    @property
    def active(self) -> bool:
        return self._active

    @property
    def loading(self) -> bool:
        """Whether the current view still has something to sample or draw."""
        if not self._active:
            return False
        if self.sampling:
            return True
        frame = self.frame
        return frame is None or not frame.still or frame.request != self._request(True)

    @property
    def sampling(self) -> bool:
        """Whether meshes for the current box are still on their way."""
        return self._active and bool(self._missing())

    def set_layers(self, layers: Sequence[SpaceLayer]) -> None:
        """Rows to show; the view is 3D whenever one of them draws in space."""
        self._layers = list(layers)
        self._active = any(is_spatial(layer.plot) for layer in self._layers)
        if not self._active:
            self._sampler.cancel()
            self._drawer.cancel()
            self._still_timer.stop()
            self._geometry.clear()
            self.frame = None
            self._picks = None
            return
        wanted = {id(part) for layer in self._layers for part in members(layer.plot)}
        self._geometry = {k: v for k, v in self._geometry.items() if k in wanted}
        self._revision += 1
        self._refresh()

    def set_appearance(self, settings: GraphSettings, style: Style) -> None:
        self._settings = settings
        style = replace(style, steps=(settings.x_axis.step, settings.y_axis.step, None))
        if style != self._style:
            self._style = style
            if self._active:
                self._refresh()

    def resize(self, width: int, height: int) -> None:
        self._size = (width, height)
        if self._active:
            self._refresh()

    def shutdown(self) -> None:
        self._sampler.stop()
        self._drawer.stop()
        self._still_timer.stop()

    # -- the view --------------------------------------------------------------------
    def set_moving(self, moving: bool) -> None:
        """A drag began or ended: while it lasts, no finer frame is drawn."""
        self._moving = moving
        if not moving and self._active:
            self._still_timer.start()

    def turn(self, dx_px: float, dy_px: float) -> None:
        """Follow a drag: the scene's front moves with the pointer."""
        self._set_camera(rotated(self.camera, -dx_px * TURN_PER_PIXEL, dy_px * TURN_PER_PIXEL))

    def turn_by(self, azimuth: float, elevation: float) -> None:
        self._set_camera(rotated(self.camera, azimuth, elevation))

    def pan(self, dx_px: float, dy_px: float) -> None:
        """Move the box's contents with the pointer, in the plane of the screen."""
        width, height = self._size
        if not width or not height:
            return
        basis = self.camera.basis()
        scale = pixels_per_half_box(width, height)
        normalized = (-dx_px * basis[0] + dy_px * basis[1]) / scale
        box = self.camera.box
        shift = normalized * (box.highs - box.lows) / 2
        lows, highs = box.lows + shift, box.highs + shift
        self._set_box(lows, highs)

    def zoom(self, factor: float, axes: str = AXES) -> None:
        """Show ``factor`` times less of ``axes`` around the centre of the box."""
        box = self.camera.box
        lows, highs = box.lows.copy(), box.highs.copy()
        for axis, name in enumerate(AXES):
            if name in axes:
                centre, half = (lows[axis] + highs[axis]) / 2, (highs[axis] - lows[axis]) / 2
                half = min(max(half / factor, MIN_HALF), MAX_HALF)
                lows[axis], highs[axis] = centre - half, centre + half
        self._set_box(lows, highs)

    def set_range(self, axis: str, lo: float, hi: float) -> bool:
        if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
            return False
        index = AXES.index(axis)
        box = self.camera.box
        lows, highs = box.lows.copy(), box.highs.copy()
        lows[index], highs[index] = lo, hi
        self._set_box(lows, highs)
        return True

    def axis_range(self, axis: str) -> tuple[float, float]:
        return self.camera.box.range(AXES.index(axis))

    def reset(self) -> None:
        self._set_camera(DEFAULT_CAMERA)

    def _set_box(self, lows: Array, highs: Array) -> None:
        values = (lows[0], highs[0], lows[1], highs[1], lows[2], highs[2])
        self._set_camera(replace(self.camera, box=Box(*(float(v) for v in values))))

    def _set_camera(self, camera: Camera) -> None:
        if camera != self.camera:
            self.camera = camera
            self._refresh()

    def state(self) -> dict[str, float]:
        """The view to save with the document."""
        box = self.camera.box
        values = {
            "azimuth": self.camera.azimuth,
            "elevation": self.camera.elevation,
            "x_min": box.x_min,
            "x_max": box.x_max,
            "y_min": box.y_min,
            "y_max": box.y_max,
            "z_min": box.z_min,
            "z_max": box.z_max,
        }
        return {f"{_STATE_PREFIX}{key}": value for key, value in values.items()}

    def restore(self, state: Mapping[str, float]) -> None:
        """Show a saved view; a missing or unusable one gives the default view."""
        camera = DEFAULT_CAMERA
        saved = {
            key: float(state[f"{_STATE_PREFIX}{key}"])
            for key in (*_CAMERA_KEYS, *_BOX_KEYS)
            if f"{_STATE_PREFIX}{key}" in state
        }
        if len(saved) == len(_CAMERA_KEYS) + len(_BOX_KEYS) and all(
            math.isfinite(value) for value in saved.values()
        ):
            box = Box(*(saved[key] for key in _BOX_KEYS))
            if (box.lows < box.highs).all():
                turned = Camera(box, saved["azimuth"], saved["elevation"])
                camera = rotated(turned, 0.0, 0.0)
        self._set_camera(camera)

    # -- background work -------------------------------------------------------------
    def _missing(self) -> list[tuple[int, Plot]]:
        box = self.camera.box
        missing = []
        for layer in self._layers:
            for part in members(layer.plot):
                done = self._geometry.get(id(part))
                if done is None or done[0] != box:
                    missing.append((id(part), part))
        return missing

    def _parts(self) -> tuple[Part, ...]:
        parts = []
        for layer in self._layers:
            r, g, b, _ = colors.rgba(layer.color)
            for part in members(layer.plot):
                done = self._geometry.get(id(part))
                if done is not None:
                    parts.append(Part(done[1], (r, g, b)))
        return tuple(parts)

    def _request(self, still: bool) -> tuple[object, ...]:
        geometry = tuple((key, id(done[1])) for key, done in self._geometry.items())
        colours = tuple(layer.color for layer in self._layers)
        return (geometry, colours, self.camera, self._size, self._style, still)

    def _refresh(self) -> None:
        """Bring sampling and drawing up to date with the current state."""
        width, height = self._size
        if not width or not height:
            return
        missing = self._missing()
        if missing:
            plots = tuple(identity for identity, _ in missing)
            # The same plots for another box: finish, so that a drag shows progress.
            finish = plots == self._sampling_plots
            self._sampling_plots = plots
            self._sampler.request(
                (self.camera.box, plots),
                partial(_sample_parts, missing, self.camera.box),
                finish_active=finish,
            )
        else:
            self._sampler.cancel()
        self._draw(still=False)
        if not self._moving:
            self._still_timer.start()

    def _draw(self, *, still: bool) -> None:
        width, height = self._size
        if not self._active or not width or not height:
            return
        request = self._request(still)
        if self.frame is not None and self.frame.request == request:
            return
        # A quick frame in progress is still worth showing; a finer one is not.
        finish = not still and not self._drawing_still
        self._drawing_still = still
        self._drawer.request(
            request,
            partial(
                _draw_frame,
                request,
                self._parts(),
                self.camera,
                width,
                height,
                self._style,
                still,
            ),
            finish_active=finish,
        )

    def _sampled(self, key: object, result: object) -> None:
        if not isinstance(result, tuple):
            return
        box, geometry = result
        for identity, sampled in geometry.items():
            self._geometry[identity] = (box, sampled)
        self._picks = None
        self._refresh()
        self.changed.emit()

    def _draw_still(self) -> None:
        if self._moving:
            return
        if self._missing():  # the finer frame waits for the meshes it needs
            self._still_timer.start()
            return
        self._draw(still=True)

    def _drawn(self, key: object, result: object) -> None:
        if not isinstance(result, Frame):
            return
        if result.still and result.request != self._request(True):
            return  # the view has moved on: this one took too long to be of use
        self.frame = result
        self.changed.emit()
        if result.still:
            self.finished.emit()

    # -- labels and picking ----------------------------------------------------------
    def labels(self, *, numbers: bool) -> list[Label]:
        """Axis names, and tick values along the outer edges of the box, for the frame on
        the screen."""
        frame = self.frame
        if frame is None:
            return []
        camera = frame.camera
        width, height = frame.width, frame.height
        box = camera.box
        lows, highs = box.lows, box.highs
        centre = (lows + highs) / 2
        toward = camera.basis()[2]
        near = np.where(toward > 0, highs, lows)
        floor = lows[2] if camera.elevation >= 0 else highs[2]
        result: list[Label] = []

        def screen(point: Array) -> tuple[float, float]:
            x, y, _ = camera.project(point[None, :], width, height)
            return float(x[0]), float(y[0])

        cx, cy = screen(centre)

        def outward(x: float, y: float, distance: float) -> tuple[float, float]:
            dx, dy = x - cx, y - cy
            length = math.hypot(dx, dy) or 1.0
            return x + dx / length * distance, y + dy / length * distance

        # Floor edges for x and y: the ones nearer the viewer.
        edges: list[tuple[int, Array, Array]] = []
        for axis, other in ((0, 1), (1, 0)):
            start = np.zeros(3)
            start[other], start[2] = near[other], floor
            start[axis] = lows[axis]
            end = start.copy()
            end[axis] = highs[axis]
            edges.append((axis, start, end))
        # The vertical edge furthest to the left.
        corners = [np.array([x, y, 0.0]) for x in (lows[0], highs[0]) for y in (lows[1], highs[1])]
        left = min(corners, key=lambda corner: screen(corner)[0])
        start, end = left.copy(), left.copy()
        start[2], end[2] = lows[2], highs[2]
        edges.append((2, start, end))

        for axis, start, end in edges:
            middle = screen((start + end) / 2)
            name_x, name_y = outward(*middle, 34.0)
            result.append(Label(AXES[axis], name_x, name_y))
            if not numbers:
                continue
            lo, hi = float(lows[axis]), float(highs[axis])
            step = tick_step(lo, hi, width, height, self._style.steps[axis])
            for k in tick_indices(lo, hi, step).tolist():
                value = k * step
                point = start.copy()
                point[axis] = value
                x, y = outward(*screen(point), 14.0)
                result.append(Label(format_tick(float(value), step), x, y))
        return result

    def _pick_data(self) -> list[tuple[Array, Array, Array, Array, str]]:
        """Per drawn plot: world positions, screen x and y, depth, and colour."""
        frame = self.frame
        key: tuple[object, ...] = (
            None if frame is None else frame.camera,
            tuple(id(done[1]) for done in self._geometry.values()),
            self._size,
        )
        if self._picks is not None and self._picks[0] == key:
            return self._picks[1]
        data: list[tuple[Array, Array, Array, Array, str]] = []
        if frame is not None:
            for layer in self._layers:
                for part in members(layer.plot):
                    done = self._geometry.get(id(part))
                    if done is None:
                        continue
                    geometry = done[1]
                    groups = []
                    if geometry.mesh is not None and not geometry.mesh.is_empty:
                        groups.append(geometry.mesh.vertices)
                    if geometry.segments is not None and len(geometry.segments):
                        groups.append(geometry.segments.reshape(-1, 3))
                    if geometry.points is not None and len(geometry.points):
                        groups.append(geometry.points)
                    if not groups:
                        continue
                    world = np.vstack(groups)
                    x, y, depth = frame.camera.project(world, frame.width, frame.height)
                    data.append((world, x, y, depth, layer.color))
        self._picks = (key, data)
        return data

    def pick(self, x: float, y: float, radius: float = 12.0) -> Picked | None:
        """The nearest point towards the viewer of what is drawn within ``radius`` pixels
        of ``(x, y)``."""
        best: tuple[float, Array, float, float, str] | None = None
        for world, sx, sy, depth, color in self._pick_data():
            close = (sx - x) ** 2 + (sy - y) ** 2 <= radius**2
            if not close.any():
                continue
            indices = np.nonzero(close)[0]
            nearest = indices[int(np.argmax(depth[indices]))]
            if best is None or depth[nearest] > best[0]:
                best = (
                    float(depth[nearest]),
                    world[nearest],
                    float(sx[nearest]),
                    float(sy[nearest]),
                    color,
                )
        if best is None:
            return None
        _, point, sx_best, sy_best, color = best
        position = (float(point[0]), float(point[1]), float(point[2]))
        return Picked(position, (sx_best, sy_best), color, self.format_point(position))

    def format_point(self, position: tuple[float, float, float]) -> str:
        width, height = self._size
        texts = []
        for axis, value in enumerate(position):
            lo, hi = self.camera.box.range(axis)
            px_per_unit = 2 * pixels_per_half_box(max(width, 1), max(height, 1)) / (hi - lo)
            texts.append(format_coordinate(value, 1.0 / px_per_unit))
        return f"({', '.join(texts)})"
