"""Interactive graph canvas drawn with Flet canvas shapes.

The canvas draws *layers* — a plot, a colour and a key — and knows nothing about
expression text or the editor. Visible geometry is loaded in background tiles and
reused while navigating; drawing never evaluates a plot.

How the plane is shown comes from :class:`~opengraph.graph.settings.GraphSettings`
(linear or logarithmic axes, Cartesian or polar grid, which grid parts are visible,
complex-plane labels); the colours come from :class:`CanvasColors` (the theme).
Points of interest (intercepts, extrema, intersections) are computed for the *focused*
layer — the row being edited — and can be hovered to inspect or clicked to pin.
Points of a *draggable* layer can be dragged: the canvas reports where the point should
go (rounded to about a pixel) and the owner of the layer moves it.

A layer that draws in space turns the whole graph into a 3D view
(:mod:`~opengraph.graph.view3d`): an image computed in the background, with labels drawn
over it. Dragging then turns the view. Points of interest and point dragging are for the
plane, and are not offered there.

On a touch device one finger does what the mouse's drag does, two fingers pinch and pan,
and a long press reads coordinates along the curves, since a finger can't hover.

Pointer, wheel and key handling live in methods that take plain coordinates
(:meth:`GraphCanvas.press`, :meth:`GraphCanvas.wheel`, ...); the Flet gesture detector
only forwards to them. Drawing builds a list of canvas shapes (:meth:`GraphCanvas.render`)
that is sent to the client at most once per pass of the event loop.

The tiles arrive already converted to screen pixels by the geometry workers (see
:mod:`~opengraph.graph.rendering`); drawing places them with a scale and translation.
In a deep zoom the viewport has exact anchors and the world coordinates handled here
are offsets from them (:mod:`~opengraph.graph.viewport`); labels and coordinates are
formatted from the exact values. Points of interest and point dragging are float64
analyses and are not offered there.
"""

from __future__ import annotations

import logging
import math
import struct
import time
import zlib
from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass, field
from functools import partial

import flet as ft
import flet.canvas as cv
import numpy as np

from opengraph import colors, events, native
from opengraph.engine import precise
from opengraph.engine.analysis import (
    ANALYSIS_SAMPLES,
    PointOfInterest,
    intersections,
    points_of_interest,
    sample_grid,
)
from opengraph.engine.builtins import Array
from opengraph.engine.multiprecision import gmpy2
from opengraph.engine.plots import ExplicitPlot, Geometry, Plot, PointPlot, PointSetPlot, members
from opengraph.engine.sampling import ViewRect, to_axis
from opengraph.graph.background import LatestTask
from opengraph.graph.raster import Style
from opengraph.graph.rendering import (
    FillImage,
    GeometryLoader,
    GeometryTile,
    ScreenGeometry,
    TileKey,
    outline,
    screen_geometry,
)
from opengraph.graph.settings import GraphSettings
from opengraph.graph.ticks import (
    LOG_LINEAR_SPAN,
    MAX_TICKS,
    choose_spacing,
    format_coordinate,
    format_log_tick,
    format_tick,
    log_ticks,
    tick_indices,
)
from opengraph.graph.view3d import KEY_TURN, Frame, SpaceLayer, View3D
from opengraph.graph.viewport import VIEW_BITS, Viewport
from opengraph.ui import fonts, runtime
from opengraph.ui.geometry import Point, Rect

FILL_ALPHA = 64  # opacity (0-255) of a shaded inequality region
log = logging.getLogger(__name__)

CURVE_WIDTH = 2.5
POINT_RADIUS = 5.0
POI_RADIUS = 4.0
HOVER_RADIUS_PX = 12.0
TOUCH_RADIUS_PX = 26.0  # how near a finger has to come; it covers what is under it
WHEEL_ZOOM_BASE = 1.0015  # per wheel angle unit: one notch (120) ~ 1.2x
WHEEL_UNITS_PER_PX = 1.2  # Flutter reports about 100 px per notch; Qt reported 120 units
KEY_PAN_PX = 60.0
PAN_MARGIN = 0.5  # extra fraction of the view sampled on each side while panning
CLICK_SLOP_PX = 4.0
MAX_INTERSECTION_LAYERS = 12
MARKER_DELAY_MS = 90  # points of interest are recomputed once the view is still
MIN_GRID_PX = 4.0  # custom tick steps denser than this fall back to automatic spacing
MAX_CIRCLE_PX = 2e5  # polar grid circles larger than this are not drawn
LABEL_FONT = fonts.Font(fonts.SANS, fonts.LABEL_SIZE)


@dataclass(frozen=True, slots=True)
class CanvasColors:
    background: str
    minor_grid: str
    major_grid: str
    axis: str
    label: str
    tooltip_background: str
    tooltip_text: str


LIGHT_COLORS = CanvasColors(
    "#ffffff",
    "#f1f2f4",
    "#d9dce1",
    "#2f3237",
    "#4a4f57",
    colors.from_rgba(255, 255, 255, 235),
    "#1f2328",
)

# Kept for code that still refers to the light theme's background directly.
BACKGROUND = LIGHT_COLORS.background


@dataclass(frozen=True, slots=True)
class Layer:
    key: Hashable
    plot: Plot
    color: str
    draggable: bool = False  # its point can be dragged (see GraphCanvas.point_dragged)


@dataclass(slots=True)
class _CacheEntry:
    """One tile of a layer, with the shapes last drawn for it.

    ``screen`` holds the tile's geometry in pixels from its own corner (prepared by the
    geometry workers); :meth:`GraphCanvas._transform` maps it to the current view, so
    the entry stays usable while panning, and as a preview while zooming.
    """

    plot: Plot
    sampled: ViewRect
    geometry: Geometry
    screen: ScreenGeometry
    fill_image: tuple[str, bytes] | None = None  # (colour, PNG)
    # The shape showing it: kept across redraws, so the client keeps the decoded image
    # and a pan only moves it (a new shape would blink while the image decodes).
    fill_shape: cv.Image | None = None
    # The entry's shapes as last drawn, and the colours and transform they were built
    # for. Flet diffs shapes by identity, so a redraw that reuses them sends nothing.
    shapes: list[cv.Shape] = field(default_factory=list)
    shapes_key: tuple[object, ...] | None = None


@dataclass(frozen=True, slots=True)
class _Trace:
    position: Point
    label: str
    color: str


@dataclass(slots=True)
class _PointDrag:
    """A point being dragged: its layer, and where it sits relative to the cursor."""

    key: Hashable
    color: str
    offset: Point  # point minus cursor, in pixels, so the point doesn't jump
    position: tuple[float, float]  # world coordinates last reported


@dataclass(frozen=True, slots=True)
class _Marker:
    """A point of interest on screen, with the layer colour it belongs to."""

    poi: PointOfInterest
    color: str


@dataclass(frozen=True, slots=True)
class _MarkerAnalysis:
    area: ViewRect
    points: list[PointOfInterest]


def _analyse_plots(focused: Plot, others: Sequence[Plot], area: ViewRect) -> _MarkerAnalysis:
    xs = sample_grid(area.x_min, area.x_max, area.x_log, ANALYSIS_SAMPLES)
    ys = sample_grid(area.y_min, area.y_max, area.y_log, ANALYSIS_SAMPLES)
    found = points_of_interest(focused, xs, ys)
    for other in others[:MAX_INTERSECTION_LAYERS]:
        found.extend(intersections(focused, other, xs, ys))
    return _MarkerAnalysis(area, found)


@ft.control("Canvas", isolated=True)  # type: ignore[arg-type]
class _Canvas(cv.Canvas):
    """A canvas whose shapes are sent only by its own updates, not the page's."""


class GraphCanvas:
    def __init__(self) -> None:
        self.cursor_moved = events.Signal()  # world coordinates under the mouse
        self.view_changed = events.Signal()
        self.point_dragged = events.Signal()  # layer key, new world x and y
        self.point_drag_finished = events.Signal()  # layer key
        self.rendering_finished = events.Signal()
        self.pressed = events.Signal()  # the graph was clicked: it takes the keys

        self.viewport = Viewport()
        self.settings = GraphSettings()
        self.colors = LIGHT_COLORS
        self.view3d = View3D()
        self.view3d.changed.connect(self.update)
        self.view3d.finished.connect(self.rendering_finished.emit)
        self._space_frame: Frame | None = None
        # The frame on screen over the one before it: the client decodes an image
        # asynchronously, and the old one covers the gap until it is ready.
        self._space_images: list[cv.Shape] = []
        self._space_labels: list[cv.Shape] = []
        self._space_labels_key: tuple[object, ...] | None = None
        self._layers: list[Layer] = []
        self._cache: dict[Hashable, list[_CacheEntry]] = {}
        self._prepared: dict[TileKey, _CacheEntry] = {}
        self._geometry = GeometryLoader()
        self._geometry.changed.connect(self._geometry_changed)
        self._analysis = LatestTask()
        self._analysis.ready.connect(self._markers_ready)
        self._drag_last: Point | None = None
        self._press_pos: Point | None = None
        self._point_drag: _PointDrag | None = None
        self._trace: _Trace | None = None
        self._focus: Hashable | None = None
        self._markers: list[_Marker] = []
        self._markers_key: tuple[object, ...] | None = None
        self._markers_area = ViewRect(0, 0, 0, 0, 1, 1)
        self._markers_timer = events.Timer(MARKER_DELAY_MS, single_shot=True)
        self._markers_timer.timeout.connect(self._refresh_markers)
        self._pinned: set[tuple[float, float]] = set()
        self._shown = False  # the client has reported a size
        self._render_pending = False
        self._backdrop: tuple[list[cv.Shape], list[cv.Shape]] = ([], [])
        self._backdrop_key: tuple[object, ...] | None = None
        self.last_paint_ms = 0.0

        self.canvas = _Canvas(expand=True, resize_interval=30, on_resize=self._on_resize)
        self.detector = ft.GestureDetector(
            # A painted background makes the whole area receive pointer events.
            content=ft.Container(self.canvas, bgcolor="#00000000"),
            mouse_cursor=ft.MouseCursor.GRAB,
            drag_interval=8,
            on_double_tap_down=lambda e: self.double_click(_xy(e.local_position)),
        )
        if runtime.device.touch:
            # Flutter's scale gesture also covers one finger, and can't share it with pan.
            detector = self.detector
            detector.on_scale_start = lambda e: self.touch_start(
                _xy(e.local_focal_point), e.pointer_count
            )
            detector.on_scale_update = lambda e: self.touch_move(
                _xy(e.local_focal_point), e.scale, e.pointer_count
            )
            detector.on_scale_end = lambda _: self.touch_end()
            detector.on_tap_up = lambda e: self.tap(_xy(e.local_position))
            detector.on_long_press_start = lambda e: self.read_at(_xy(e.local_position))
            detector.on_long_press_move_update = lambda e: self.read_at(_xy(e.local_position))
        else:
            detector = self.detector
            detector.on_pan_down = lambda e: self.press(_xy(e.local_position))
            detector.on_pan_update = lambda e: self.move(_xy(e.local_position))
            detector.on_pan_end = lambda e: self.release(_xy(e.local_position))
            detector.on_pan_cancel = lambda _: self.cancel_press()
            detector.on_tap_up = lambda e: self.release(_xy(e.local_position))
            detector.on_hover = lambda e: self.move(_xy(e.local_position))
            detector.on_exit = lambda _: self.leave()
            detector.on_scroll = self._on_scroll
        self._pinch_scale = 1.0
        self.menu = ft.ContextMenu(
            content=self.detector,
            items=[
                ft.PopupMenuItem("Zoom In", on_click=lambda _: self.zoom(1.25)),
                ft.PopupMenuItem("Zoom Out", on_click=lambda _: self.zoom(0.8)),
                ft.PopupMenuItem("Reset View", on_click=lambda _: self.reset_view()),
            ],
        )
        self.control = ft.Container(
            content=self.menu,
            expand=True,
            left=0,
            top=0,
            right=0,
            bottom=0,
            clip_behavior=ft.ClipBehavior.HARD_EDGE,
            bgcolor=self.colors.background,
        )

    # -- public API ------------------------------------------------------------------
    def set_layers(self, layers: Sequence[Layer]) -> None:
        self._layers = list(layers)
        keys = {layer.key for layer in self._layers}
        for key in [k for k in self._cache if k not in keys]:
            del self._cache[key]
        identities = {
            (layer.key, id(part)) for layer in self._layers for part in members(layer.plot)
        }
        self._prepared = {
            key: entry for key, entry in self._prepared.items() if key[:2] in identities
        }
        self._analysis.cancel()
        self.view3d.set_layers([SpaceLayer(la.key, la.plot, la.color) for la in self._layers])
        # Replace obsolete requests immediately, including when all layers are hidden.
        if self._shown or self._geometry.loading:
            self._request_geometry()
        self._trace = None
        self._markers_key = None
        self.update()

    def set_settings(self, settings: GraphSettings) -> None:
        """Apply axis/grid settings; switching an axis to log keeps its visible range."""
        settings = settings.normalized()
        self.settings = settings
        self.viewport.set_log("x", settings.x_axis.log)
        self.viewport.set_log("y", settings.y_axis.log)
        if settings.polar_grid and not math.isclose(self.viewport.scale_x, self.viewport.scale_y):
            self.viewport.scale_y = self.viewport.scale_x  # circles must look round
        self._markers_key = None
        self._style_view3d()
        self._view_changed()

    def set_colors(self, colors_: CanvasColors) -> None:
        self.colors = colors_
        self.control.bgcolor = colors_.background
        runtime.push(self.control)
        self._style_view3d()
        self.update()

    def _style_view3d(self) -> None:
        def rgb(color: str) -> tuple[int, int, int]:
            return colors.rgba(color)[:3]

        self.view3d.set_appearance(
            self.settings,
            Style(
                background=rgb(self.colors.background),
                grid=rgb(self.colors.major_grid),
                axis=rgb(self.colors.axis),
                show_grid=self.settings.major_grid,
                show_axes=self.settings.axes,
            ),
        )

    def set_focus(self, key: Hashable | None) -> None:
        """The layer whose points of interest are shown (the row being edited)."""
        if key != self._focus:
            self._focus = key
            self._pinned.clear()
            self._markers_key = None
            self.update()

    def reset_view(self) -> None:
        if self.view3d.active:
            self.view3d.reset()
        else:
            self.viewport.reset()
        self._view_changed()

    def zoom(self, factor: float) -> None:
        if self.view3d.active:
            self.view3d.zoom(factor)
        else:
            self.viewport.zoom_center(factor)
        self._view_changed()

    def view_state(self) -> dict[str, float]:
        """The views to save with a document: the plane's and the 3D view's."""
        return self.viewport.state() | self.view3d.state()

    def axis_range(self, axis: str) -> tuple[float, float]:
        """The range an axis shows, in the kind of view on screen."""
        if self.view3d.active:
            return self.view3d.axis_range(axis)
        lo, hi = self.viewport.range(axis)
        return float(lo), float(hi)

    def set_range(self, axis: str, lo: float, hi: float) -> bool:
        """Show ``[lo, hi]`` on one axis; with a locked aspect the other axis follows."""
        if self.view3d.active:
            if not self.view3d.set_range(axis, lo, hi):
                return False
            self._view_changed()
            return True
        vp = self.viewport
        ratio = vp.scale_y / vp.scale_x
        if not vp.set_range(axis, lo, hi):
            return False
        if self.settings.lock_aspect and not self.settings.any_log:
            if axis == "x":
                vp.scale_y = vp.scale_x * ratio
            else:
                vp.scale_x = vp.scale_y / ratio
        self._view_changed()
        return True

    def resize(self, width: int, height: int) -> None:
        """The canvas's size on screen (reported by the client, or set by tests)."""
        self.viewport.resize(width, height)
        self.view3d.resize(width, height)
        self._trace = None
        self._shown = width > 0 and height > 0
        if self._shown:
            self._request_geometry()
        self.update()

    @property
    def shown(self) -> bool:
        return self._shown

    def markers(self) -> list[PointOfInterest]:
        """The points of interest currently shown (computed if needed)."""
        return [m.poi for m in self._current_markers(force=True)]

    def _refresh_markers(self) -> None:
        focused = next((layer for layer in self._layers if layer.key == self._focus), None)
        if not self.settings.points_of_interest or focused is None or self.viewport.anchored:
            return
        if self.view3d.active:
            return
        if self._geometry.loading:
            self._markers_timer.start()
            return
        area = self.viewport.view_rect().padded(PAN_MARGIN)
        key = (self._focus, tuple(id(layer.plot) for layer in self._layers), self._frame())
        others = [layer.plot for layer in self._layers if layer is not focused]
        self._analysis.request(key, partial(_analyse_plots, focused.plot, others, area))

    def _markers_ready(self, key: object, result: object) -> None:
        current = (self._focus, tuple(id(layer.plot) for layer in self._layers), self._frame())
        if key != current or not isinstance(result, _MarkerAnalysis):
            return
        if not result.area.contains(self.viewport.view_rect()):
            return
        focused = next((layer for layer in self._layers if layer.key == self._focus), None)
        if focused is None:
            return
        self._markers_key = current
        self._markers_area = result.area
        self._markers = [_Marker(point, focused.color) for point in result.points]
        self.update()

    def format_position(self, wx: float, wy: float, exact: bool = False) -> str:
        """Coordinates as this graph's mode reads them: (x, y), r and θ, or a + bi.

        ``wx``/``wy`` are canvas world coordinates (offsets from the anchors in a deep
        zoom, which are formatted from their exact values). A 3D view reports the point
        under the pointer instead, if there is one."""
        if self.view3d.active:
            return self._trace.label if self._trace is not None else ""
        upp_x = 0.0 if exact else self._units_per_px("x", wx)
        upp_y = 0.0 if exact else self._units_per_px("y", wy)
        x_value, y_value = self.viewport.absolute("x", wx), self.viewport.absolute("y", wy)
        x_text, y_text = format_coordinate(x_value, upp_x), format_coordinate(y_value, upp_y)
        wx, wy = float(x_value), float(y_value)
        if self.settings.complex_mode:
            if wy == 0 or y_text in ("0", "-0"):
                return x_text
            sign = "−" if wy < 0 else "+"
            magnitude = y_text.lstrip("-")
            return f"{x_text} {sign} {'' if magnitude == '1' else magnitude}i"
        if self.settings.polar_grid:
            r = math.hypot(wx, wy)
            theta = math.atan2(wy, wx)
            upp = max(upp_x, upp_y)
            if self.settings.degrees:
                angle = format_coordinate(math.degrees(theta), 0.0 if exact else 0.01) + "°"
            else:
                angle = format_coordinate(theta, 0.0 if exact else 0.0001)
            return f"r = {format_coordinate(r, upp)}, θ = {angle}"
        return f"({x_text}, {y_text})"

    def _units_per_px(self, axis: str, value: float) -> float:
        vp = self.viewport
        log_axis = vp.x_log if axis == "x" else vp.y_log
        scale = vp.scale_x if axis == "x" else vp.scale_y
        if log_axis:
            return abs(value) * math.log(10) / scale if math.isfinite(value) else 0.0
        return 1.0 / scale

    def _view_changed(self) -> None:
        self._trace = None
        self._analysis.cancel()
        if self._shown:
            self._request_geometry()
        if self._markers_timer.is_active():
            self._markers_timer.start()  # debounce: wait until the view is still
        self.update()
        self.view_changed.emit()

    # -- geometry ----------------------------------------------------------------------
    def _request_geometry(self) -> None:
        layers = [] if self.view3d.active else [(la.key, la.plot) for la in self._layers]
        self._geometry.request(layers, self.viewport.view_rect())

    @property
    def is_rendering(self) -> bool:
        """Whether the current view still has visible geometry to load."""
        if self.view3d.active:
            return self.view3d.loading
        self._request_geometry()
        return self._geometry.loading

    def _geometry_changed(self) -> None:
        self.update()
        if not self._geometry.loading:
            self.rendering_finished.emit()

    def shutdown(self) -> None:
        self.view3d.shutdown()
        self._geometry.stop()
        self._analysis.stop()
        self._markers_timer.stop()

    def _frame(self) -> tuple[float, float, int, int, bool, bool]:
        vp = self.viewport
        return (vp.scale_x, vp.scale_y, vp.width, vp.height, vp.x_log, vp.y_log)

    def _origin(self, view: ViewRect) -> tuple[float, float]:
        """Where a sampled view's origin is, in canvas world coordinates."""
        if not view.bits and not self.viewport.anchored:
            return 0.0, 0.0
        vp = self.viewport
        return vp.relative("x", view.origin_x), vp.relative("y", view.origin_y)

    def _transform(self, entry: _CacheEntry) -> tuple[float, float, float, float]:
        """``(kx, ky, dx, dy)``: entry screen coordinates times k plus d are current ones."""
        vp = self.viewport
        screen = entry.screen
        kx, ky = vp.scale_x / screen.scale[0], vp.scale_y / screen.scale[1]
        ox, oy = self._origin(entry.sampled)
        u0, v0 = screen.corner
        dx = vp.width / 2 + (ox + u0 - vp.center_x) * vp.scale_x
        dy = vp.height / 2 - (oy + v0 - vp.center_y) * vp.scale_y
        return kx, ky, dx, dy

    def _entry(self, tile: GeometryTile) -> _CacheEntry:
        entry = self._prepared.get(tile.request.key)
        if entry is not None:
            return entry
        screen = tile.screen
        if screen is None:  # prepared by the worker unless the tile failed
            sampled = tile.request.view
            drawn = tile.outline if tile.outline is not None else outline(tile.geometry, sampled)
            screen = screen_geometry(tile.geometry, sampled, tile.request.scale, drawn)
        entry = _CacheEntry(tile.request.plot, tile.request.view, tile.geometry, screen)
        self._prepared[tile.request.key] = entry
        return entry

    def _same_scale(self, entry: _CacheEntry) -> bool:
        """Whether ``entry`` was prepared for the current scale (not a zoom preview)."""
        kx, ky, _, _ = self._transform(entry)
        vp = self.viewport
        return (
            abs(kx - 1) < 1e-9
            and abs(ky - 1) < 1e-9
            and (entry.sampled.x_log, entry.sampled.y_log) == (vp.x_log, vp.y_log)
        )

    # -- ticks -------------------------------------------------------------------------------
    def _ticks(self, axis: str) -> tuple[list[float], list[float], float | None]:
        """``(major, minor, step)`` tick values in world units along ``axis``.

        ``step`` is the linear major spacing (None on a log axis, whose labels are
        formatted individually). In a deep zoom the values are offsets from the anchor,
        and the ticks are the multiples of the step at their exact positions.
        """
        vp = self.viewport
        log_axis = vp.x_log if axis == "x" else vp.y_log
        scale = vp.scale_x if axis == "x" else vp.scale_y
        lo, hi = vp.range(axis)
        if log_axis:
            u_lo, u_hi = to_axis(lo, True), to_axis(hi, True)
            if u_hi - u_lo >= LOG_LINEAR_SPAN:
                majors, minors = log_ticks(u_lo, u_hi, scale)
                return majors, minors, None
            # A fraction of a decade: ordinary ticks, spaced for the local density.
            middle = 10.0 ** ((u_lo + u_hi) / 2)
            scale = scale / (middle * math.log(10))
        anchor = vp.anchor_x if axis == "x" else vp.anchor_y
        step = choose_spacing(scale)
        if anchor and axis == "x":
            # Deep-zoom labels carry every digit a pixel resolves: space them to fit.
            widest = max(abs(float(vp.absolute("x", v))) for v in (lo, hi))
            digits = math.log10(max(widest, 1.0)) - math.log10(step.major) + 3
            step = choose_spacing(scale, max(90.0, digits * LABEL_FONT.advance("0") + 16))
        major, minor = step.major, step.minor
        custom = (self.settings.x_axis if axis == "x" else self.settings.y_axis).step
        if custom is not None and custom * scale >= MIN_GRID_PX:
            major, minor = custom, custom / 5 if custom * scale / 5 >= MIN_GRID_PX else custom
        if anchor:
            return (
                self._anchored_ticks(axis, major, lo, hi),
                self._anchored_ticks(axis, minor, lo, hi),
                major,
            )
        majors = [float(k) * major for k in tick_indices(lo, hi, major)]
        minors = [float(k) * minor for k in tick_indices(lo, hi, minor)]
        return majors, minors, major

    def _anchored_ticks(self, axis: str, step: float, lo: float, hi: float) -> list[float]:
        """Offsets from the anchor of the multiples of ``step`` between offsets ``lo``
        and ``hi``. The step is taken as the decimal it stands for (float64's ``2e-20``
        is off by far more than a step once multiplied up to the anchor)."""
        vp = self.viewport
        exact_step = precise.literal(step, VIEW_BITS)
        with precise.context(VIEW_BITS):
            first = int(gmpy2.ceil(gmpy2.mpfr(vp.absolute(axis, lo)) / exact_step))
            last = int(gmpy2.floor(gmpy2.mpfr(vp.absolute(axis, hi)) / exact_step))
            if last - first > MAX_TICKS:
                return []
            return [vp.relative(axis, k * exact_step) for k in range(first, last + 1)]

    def _tick_label(self, value: float, step: float | None, axis: str) -> str:
        if step is None:
            text = format_log_tick(value)
        else:
            exact = self.viewport.absolute(axis, value)
            if not isinstance(exact, float):  # round to the tick: k * step exactly
                with precise.context(VIEW_BITS):
                    exact_step = precise.literal(step, VIEW_BITS)
                    exact = gmpy2.rint(exact / exact_step) * exact_step
            text = format_tick(exact, step)
        if axis == "y" and self.settings.complex_mode:
            return {"1": "i", "−1": "−i"}.get(text, f"{text}i")
        return text

    # -- drawing ----------------------------------------------------------------------------
    def update(self) -> None:
        """Redraw once the current callback is done (changes made together draw once)."""
        if not self._render_pending:
            self._render_pending = True
            events.call_soon(self._render_now)

    def _render_now(self) -> None:
        self._render_pending = False
        if runtime.page_of(self.canvas) is None:
            return  # drawn when shown; tests call render() themselves
        self.render()
        self.canvas.update()

    def render(self) -> list[cv.Shape]:
        """Build the canvas shapes for the current state (and keep them on the canvas)."""
        started = time.perf_counter()
        if self.view3d.active:
            scene = self._space_shapes()
            self._paint_trace(scene)
            self.canvas.shapes = scene
            self.last_paint_ms = (time.perf_counter() - started) * 1000
            return scene
        self._space_frame, self._space_images = None, []
        vp = self.viewport
        # The grid, axes and labels below and above the curves depend only on the view;
        # reusing their shapes keeps redraws for hovering or arriving tiles small.
        backdrop_key = (
            self._frame(),
            vp.center_x,
            vp.center_y,
            vp.anchor_x,
            vp.anchor_y,
            self.settings,
            self.colors,
        )
        if backdrop_key != self._backdrop_key:
            below: list[cv.Shape] = []
            above: list[cv.Shape] = []
            if self.settings.polar_grid:
                self._paint_polar_grid(below)
            else:
                self._paint_grid(below)
            self._paint_axes(below)
            if self.settings.axis_numbers:
                self._paint_labels(above)
            self._backdrop, self._backdrop_key = (_freeze(below), _freeze(above)), backdrop_key
        shapes: list[cv.Shape] = list(self._backdrop[0])
        self._paint_layers(shapes)
        shapes += self._backdrop[1]
        self._paint_markers(shapes)
        self._paint_trace(shapes)
        self._paint_point_drag(shapes)
        self.canvas.shapes = shapes
        self.last_paint_ms = (time.perf_counter() - started) * 1000
        return shapes

    def _space_shapes(self) -> list[cv.Shape]:
        """The 3D view: its latest image, with the labels that go with that image."""
        frame = self.view3d.frame
        if frame is not self._space_frame:
            self._space_frame = frame
            if frame is None:
                self._space_images = []
            else:
                image = cv.Image(src=frame.data, x=0, y=0, width=frame.width, height=frame.height)
                self._space_images = [*self._space_images[-1:], *_freeze([image])]
        key = (frame, self.settings, self.colors)
        if key != self._space_labels_key:
            self._space_labels_key = key
            labels = self.view3d.labels(numbers=self.settings.axis_numbers)
            self._space_labels = _freeze(
                [_text(la.text, Point(la.x, la.y), self.colors.label) for la in labels]
            )
        shapes = [*self._space_images, *self._space_labels]
        if self.view3d.sampling or frame is None:
            shapes.append(
                cv.Text(
                    12,
                    self.viewport.height - 12 - LABEL_FONT.ascent(),
                    "Rendering…",
                    style=_label_style(self.colors.label),
                )
            )
        return shapes

    def _grid_lines(self, values: list[float], horizontal: bool) -> list[tuple[float, float]]:
        """End points of grid lines, in pairs (for a LINES point list)."""
        vp = self.viewport
        points: list[tuple[float, float]] = []
        for value in values:
            if horizontal:
                y = round(float(vp.to_screen_y(value))) + 0.5
                points += [(0.0, y), (float(vp.width), y)]
            else:
                x = round(float(vp.to_screen_x(value))) + 0.5
                points += [(x, 0.0), (x, float(vp.height))]
        return points

    def _paint_grid(self, shapes: list[cv.Shape]) -> None:
        x_major, x_minor, _ = self._ticks("x")
        y_major, y_minor, _ = self._ticks("y")
        for show, color, xs, ys in (
            (self.settings.minor_grid, self.colors.minor_grid, x_minor, y_minor),
            (self.settings.major_grid, self.colors.major_grid, x_major, y_major),
        ):
            if show:
                lines = self._grid_lines(xs, False) + self._grid_lines(ys, True)
                if lines:
                    shapes.append(_lines(lines, _stroke(color, 1, anti_alias=False)))

    def _zero(self) -> tuple[float, float]:
        """Where the world origin is, in canvas coordinates."""
        return self.viewport.relative("x", 0.0), self.viewport.relative("y", 0.0)

    def _paint_polar_grid(self, shapes: list[cv.Shape]) -> None:
        vp = self.viewport
        ox, oy = (float(v) for v in vp.to_screen(*self._zero()))
        corners = [(0.0, 0.0), (vp.width, 0.0), (0.0, vp.height), (vp.width, vp.height)]
        r_max_px = max(math.hypot(cx - ox, cy - oy) for cx, cy in corners)
        inside = 0 <= ox <= vp.width and 0 <= oy <= vp.height
        r_min_px = (
            0.0
            if inside
            else math.hypot(max(0.0, -ox, ox - vp.width), max(0.0, -oy, oy - vp.height))
        )
        spacing = choose_spacing(vp.scale_x)
        custom = self.settings.x_axis.step
        if custom is not None and custom * vp.scale_x >= MIN_GRID_PX:
            spacing = type(spacing)(custom, custom / 5)
        for show, step, color in (
            (self.settings.minor_grid, spacing.minor, self.colors.minor_grid),
            (self.settings.major_grid, spacing.major, self.colors.major_grid),
        ):
            if not show:
                continue
            step_px = step * vp.scale_x
            first = max(1, math.ceil(r_min_px / step_px))
            last = math.floor(r_max_px / step_px)
            if last - first <= MAX_TICKS:
                for k in range(first, last + 1):
                    radius = k * step_px
                    if radius <= MAX_CIRCLE_PX:
                        shapes.append(cv.Circle(ox, oy, radius, paint=_stroke(color, 1)))
        # Rays every 15° (minor) and 30° (major).
        for show, every, color in (
            (self.settings.minor_grid, 15, self.colors.minor_grid),
            (self.settings.major_grid, 30, self.colors.major_grid),
        ):
            if not show:
                continue
            rays: list[tuple[float, float]] = []
            for degrees in range(0, 360, every):
                angle = math.radians(degrees)
                rays += [
                    (ox, oy),
                    (ox + r_max_px * math.cos(angle), oy - r_max_px * math.sin(angle)),
                ]
            shapes.append(_lines(rays, _stroke(color, 1)))
        if self.settings.axis_numbers and inside:
            self._paint_angle_labels(shapes, ox, oy)

    def _paint_angle_labels(self, shapes: list[cv.Shape], ox: float, oy: float) -> None:
        vp = self.viewport
        radius = 0.88 * min(ox, vp.width - ox, oy, vp.height - oy)
        if radius < 40:
            return
        names = {0: "0", 30: "π/6", 60: "π/3", 90: "π/2", 120: "2π/3", 150: "5π/6", 180: "π"}
        names |= {210: "7π/6", 240: "4π/3", 270: "3π/2", 300: "5π/3", 330: "11π/6"}
        height = LABEL_FONT.height()
        for degrees in range(0, 360, 30):
            text = f"{degrees}°" if self.settings.degrees else names[degrees]
            angle = math.radians(degrees)
            x = ox + radius * math.cos(angle)
            y = oy - radius * math.sin(angle)
            w = LABEL_FONT.advance(text)
            self._label(shapes, text, Rect(x - w / 2, y - height / 2, w, height))

    def _paint_axes(self, shapes: list[cv.Shape]) -> None:
        if not self.settings.axes:
            return
        vp = self.viewport
        paint = _stroke(self.colors.axis, 1.5, anti_alias=False)
        zero_x, zero_y = self._zero()
        if not vp.y_log:  # no y = 0 on a log y axis
            ay = float(vp.to_screen_y(zero_y))
            if 0 <= ay <= vp.height:
                shapes.append(cv.Line(0, ay, vp.width, ay, paint=paint))
        if not vp.x_log:
            ax = float(vp.to_screen_x(zero_x))
            if 0 <= ax <= vp.width:
                shapes.append(cv.Line(ax, 0, ax, vp.height, paint=paint))

    def _label(self, shapes: list[cv.Shape], text: str, rect: Rect) -> None:
        halo = colors.with_alpha(self.colors.background, 210)
        box = rect.adjusted(-2, 0, 2, 0)
        shapes.append(cv.Rect(box.x, box.y, box.width, box.height, paint=_fill(halo)))
        shapes.append(_text(text, rect.center, self.colors.label))

    def _paint_labels(self, shapes: list[cv.Shape]) -> None:
        vp = self.viewport
        text_h = LABEL_FONT.height()
        # On a log axis there is no zero: labels sit at the bottom/left edge instead.
        zero_x, zero_y = self._zero()
        with np.errstate(over="ignore", invalid="ignore"):
            ax = 0.0 if vp.x_log else float(vp.to_screen_x(zero_x))
            ay = float(vp.height) if vp.y_log else float(vp.to_screen_y(zero_y))

        # X labels sit below the x-axis, pinned to the top/bottom edge when it's off-screen.
        majors, _, step = self._ticks("x")
        label_y = min(max(ay + 4, 2.0), vp.height - text_h - 2)
        for value in majors:
            if step is not None and abs(value - zero_x) < step * 1e-6:
                continue
            text = self._tick_label(value, step, "x")
            w = LABEL_FONT.advance(text)
            x = float(vp.to_screen_x(value))
            self._label(shapes, text, Rect(x - w / 2, label_y, w, text_h))

        # Y labels sit left of the y-axis, pinned to the left/right edge.
        majors, _, step = self._ticks("y")
        for value in majors:
            if step is not None and abs(value - zero_y) < step * 1e-6:
                continue
            text = self._tick_label(value, step, "y")
            w = LABEL_FONT.advance(text)
            label_x = min(max(ax - w - 6, 2.0), vp.width - w - 2)
            y = float(vp.to_screen_y(value))
            self._label(shapes, text, Rect(label_x, y - text_h / 2, w, text_h))

        if not (vp.x_log or vp.y_log) and 0 <= ax <= vp.width and 0 <= ay <= vp.height:
            w = LABEL_FONT.advance("0")
            self._label(shapes, "0", Rect(ax - w - 6, ay + 4, w, text_h))

    def _paint_layers(self, shapes: list[cv.Shape]) -> None:
        self._request_geometry()
        tiles = self._geometry.tiles
        visible_keys = {tile.request.key for tile in tiles}
        self._prepared = {
            key: entry for key, entry in self._prepared.items() if key in visible_keys
        }
        for layer in self._layers:
            entries = [self._entry(tile) for tile in tiles if tile.request.layer == layer.key]
            if entries:
                self._cache[layer.key] = entries
            else:
                # Retain a transformed preview while the first new tile is loading.
                identities = {id(part) for part in members(layer.plot)}
                entries = [
                    entry
                    for entry in self._cache.get(layer.key, [])
                    if id(entry.plot) in identities
                    and (entry.sampled.x_log, entry.sampled.y_log)
                    == (self.viewport.x_log, self.viewport.y_log)
                ]
                self._cache[layer.key] = entries
            for entry in entries:
                self._draw_entry(shapes, layer, entry)
        if self._geometry.loading:
            shapes.append(
                cv.Text(
                    12,
                    self.viewport.height - 12 - LABEL_FONT.ascent(),
                    "Rendering…",
                    style=_label_style(self.colors.label),
                )
            )

    def _draw_entry(self, shapes: list[cv.Shape], layer: Layer, entry: _CacheEntry) -> None:
        transform = self._transform(entry)
        key = (layer.color, self.colors.background, *transform)
        if entry.shapes_key != key:
            entry.shapes = self._entry_shapes(layer, entry, transform)
            entry.shapes_key = key
            # The fill image is moved in place (see _paint_fill); the rest never change.
            _freeze([shape for shape in entry.shapes if shape is not entry.fill_shape])
        shapes.extend(entry.shapes)

    def _entry_shapes(
        self, layer: Layer, entry: _CacheEntry, transform: tuple[float, float, float, float]
    ) -> list[cv.Shape]:
        kx, ky, dx, dy = transform
        shapes: list[cv.Shape] = []

        def place(points: Array) -> list[tuple[float, float]]:
            return place_points(points, kx, ky, dx, dy)

        self._paint_fill(shapes, layer, entry, transform)
        dashed = entry.geometry.dashed
        paint = _stroke(layer.color, CURVE_WIDTH, dashed=dashed)
        drawn = entry.screen.outline
        if dashed:
            # One path for all lines: each line starts its own dash pattern.
            placed = [place(line) for line in drawn]
            if placed:
                shapes.append(_polyline_path(placed, paint))
        else:
            pairs = [line for line in drawn if len(line) == 2]
            for line in drawn:
                if len(line) > 2:
                    shapes.append(
                        cv.Points(_offsets(place(line)), cv.PointMode.POLYGON, paint=paint)
                    )
            if pairs:  # lone segments share one shape
                shapes.append(_lines(place(np.vstack(pairs)), paint))
        if entry.screen.points is not None:
            finite = entry.screen.points[np.isfinite(entry.screen.points).all(axis=1)]
            if len(finite):
                spots = place(finite)
                # A ring in the background colour, then the filled dot.
                shapes.append(_dots(spots, 2 * POINT_RADIUS + 3, self.colors.background))
                shapes.append(_dots(spots, 2 * POINT_RADIUS, layer.color))
        return shapes

    def _paint_fill(
        self,
        shapes: list[cv.Shape],
        layer: Layer,
        entry: _CacheEntry,
        transform: tuple[float, float, float, float],
    ) -> None:
        """The shaded region of an inequality, as a translucent raster."""
        fill = entry.screen.fill
        if fill is None:
            return
        if entry.fill_image is None or entry.fill_image[0] != layer.color:
            entry.fill_image = (layer.color, fill_png(fill, layer.color))
            entry.fill_shape = None
        _, data = entry.fill_image
        x, y, width, height = fill.rect
        kx, ky, dx, dy = transform
        place = (x * kx + dx, y * ky + dy, width * kx, height * ky)
        shape = entry.fill_shape
        if shape is None:
            shape = entry.fill_shape = cv.Image(src=data)
        if (shape.x, shape.y, shape.width, shape.height) != place:
            shape.x, shape.y, shape.width, shape.height = place
        shapes.append(shape)

    # -- points of interest -----------------------------------------------------------------
    def _current_markers(self, *, force: bool = False) -> list[_Marker]:
        """The markers in view. Like layer geometry, a padded area is analysed once per
        scale, so panning only filters cached points. A stale cache is recomputed after
        the view has been still for a moment (``force`` recomputes at once), so zooming
        doesn't pay for the analysis on every frame."""
        if not self.settings.points_of_interest or self._focus is None:
            return []
        if self.viewport.anchored:  # float64 analysis can't resolve a deep zoom
            return []
        if self.view3d.active:
            return []
        focused = next((layer for layer in self._layers if layer.key == self._focus), None)
        if focused is None:
            return []
        view = self.viewport.view_rect()
        key = (self._focus, tuple(id(layer.plot) for layer in self._layers), self._frame())
        if key != self._markers_key or not self._markers_area.contains(view):
            if not force:
                if not self._markers_timer.is_active():
                    self._markers_timer.start()
                return []
            self._markers_key = key
            self._markers_area = view.padded(PAN_MARGIN)
            self._analysis.cancel()
            self._markers = self._analyse(focused, self._markers_area)
        return [
            m
            for m in self._markers
            if view.x_min <= m.poi.x <= view.x_max and view.y_min <= m.poi.y <= view.y_max
        ]

    def _analyse(self, focused: Layer, area: ViewRect) -> list[_Marker]:
        try:
            others = [layer.plot for layer in self._layers if layer is not focused]
            result = _analyse_plots(focused.plot, others, area)
            return [_Marker(point, focused.color) for point in result.points]
        except Exception:  # analysis must never break drawing
            log.exception("Point-of-interest analysis failed")
            return []

    def _paint_markers(self, shapes: list[cv.Shape]) -> None:
        markers = self._current_markers()
        if not markers:
            return
        vp = self.viewport
        for marker in markers:
            center = Point(*(float(v) for v in vp.to_screen(marker.poi.x, marker.poi.y)))
            shapes.append(
                cv.Circle(center.x, center.y, POI_RADIUS, paint=_fill(self.colors.background))
            )
            shapes.append(cv.Circle(center.x, center.y, POI_RADIUS, paint=_stroke(marker.color, 2)))
            if (marker.poi.x, marker.poi.y) in self._pinned:
                self._paint_tooltip(shapes, center, self._marker_label(marker), marker.color)

    def _marker_label(self, marker: _Marker) -> str:
        return f"{marker.poi.label} {self.format_position(marker.poi.x, marker.poi.y, True)}"

    def _marker_at(self, pos: Point) -> _Marker | None:
        vp = self.viewport
        best: tuple[float, _Marker] | None = None
        for marker in self._current_markers():
            sx, sy = vp.to_screen(marker.poi.x, marker.poi.y)
            dist = math.hypot(sx - pos.x, sy - pos.y)
            if dist <= _reach() and (best is None or dist < best[0]):
                best = (dist, marker)
        return None if best is None else best[1]

    # -- trace -------------------------------------------------------------------------------
    def _paint_trace(self, shapes: list[cv.Shape]) -> None:
        if self._trace is None:
            return
        trace = self._trace
        x, y = trace.position.x, trace.position.y
        shapes.append(cv.Circle(x, y, POINT_RADIUS + 2, paint=_fill(self.colors.background)))
        shapes.append(cv.Circle(x, y, POINT_RADIUS + 1, paint=_fill(trace.color)))
        self._paint_tooltip(shapes, trace.position, trace.label, trace.color)

    def _paint_tooltip(self, shapes: list[cv.Shape], anchor: Point, text: str, color: str) -> None:
        w = LABEL_FONT.advance(text) + 12
        h = LABEL_FONT.height() + 6
        x = anchor.x + 10
        y = anchor.y - h - 8
        x = min(max(x, 2.0), self.viewport.width - w - 2)
        if y < 2:
            y = anchor.y + 10
        box = Rect(x, y, w, h)
        for paint in (_fill(self.colors.tooltip_background), _stroke(color, 1)):
            shapes.append(
                cv.Rect(box.x, box.y, box.width, box.height, border_radius=4, paint=paint)
            )
        shapes.append(_text(text, box.center, self.colors.tooltip_text))

    def _paint_point_drag(self, shapes: list[cv.Shape]) -> None:
        drag = self._point_drag
        if drag is None:
            return
        anchor = Point(*(float(v) for v in self.viewport.to_screen(*drag.position)))
        ring = colors.with_alpha(drag.color, 70)
        shapes.append(cv.Circle(anchor.x, anchor.y, POINT_RADIUS + 5, paint=_fill(ring)))
        self._paint_tooltip(shapes, anchor, self.format_position(*drag.position), drag.color)

    # -- hit testing ------------------------------------------------------------------------
    def _draggable_at(self, pos: Point) -> tuple[Layer, Point] | None:
        """The draggable layer whose point is under ``pos``, and that point on screen."""
        if self.viewport.anchored or self.view3d.active:  # dragging works in the plane
            return None
        best: tuple[float, Layer, Point] | None = None
        for layer in reversed(self._layers):  # topmost first wins ties
            if not layer.draggable:
                continue
            # Point coordinates are already known; dragging must not wait for a tile.
            for part in members(layer.plot):
                if not isinstance(part, PointPlot):
                    continue
                point = Point(*(float(v) for v in self.viewport.to_screen(part.x, part.y)))
                dist = math.hypot(point.x - pos.x, point.y - pos.y)
                if dist <= _reach() and (best is None or dist < best[0]):
                    best = (dist, layer, point)
        return None if best is None else (best[1], best[2])

    def _snap(self, wx: float, wy: float) -> tuple[float, float]:
        """World coordinates rounded to the fewest decimals that keep them within a pixel."""
        snapped = []
        for axis, value in (("x", wx), ("y", wy)):
            upp = self._units_per_px(axis, value)
            if upp <= 0 or not math.isfinite(upp) or not math.isfinite(value):
                snapped.append(value)
                continue
            decimals = math.ceil(-math.log10(upp))
            snapped.append(round(value, decimals) + 0.0)  # + 0.0 turns -0.0 into 0.0
        return snapped[0], snapped[1]

    def _find_trace(self, pos: Point) -> _Trace | None:
        """The point to highlight for a mouse position, if one is close enough."""
        if self.view3d.active:
            picked = self.view3d.pick(pos.x, pos.y, _reach())
            if picked is None:
                return None
            return _Trace(Point(*picked.screen), picked.text, picked.color)
        marker = self._marker_at(pos)
        if marker is not None:
            position = Point(
                *(float(v) for v in self.viewport.to_screen(marker.poi.x, marker.poi.y))
            )
            return _Trace(position, self._marker_label(marker), marker.color)
        mx, my = pos.x, pos.y
        vp = self.viewport
        # (screen distance, world x, world y, is an exact user point, layer)
        best: tuple[float, float, float, bool, Layer] | None = None

        def consider(wx: float, wy: float, exact: bool, layer: Layer) -> bool:
            nonlocal best
            if not (math.isfinite(wx) and math.isfinite(wy)):
                return False
            sx, sy = vp.to_screen(wx, wy)
            if not (math.isfinite(sx) and math.isfinite(sy)):
                return False
            dist = math.hypot(sx - mx, sy - my)
            if dist > _reach():
                return False
            if best is None or dist < best[0]:
                best = (dist, wx, wy, exact, layer)
            return True

        for layer in reversed(self._layers):  # topmost first wins ties
            explicit = [m for m in members(layer.plot) if isinstance(m, ExplicitPlot)]
            hit = False
            # Primary: each function evaluated exactly under the cursor.
            for plot in explicit:
                if plot.independent == "x":
                    wx = float(vp.to_world_x(mx))
                    hit |= consider(wx, self._explicit_value(plot, wx), False, layer)
                else:
                    wy = float(vp.to_world_y(my))
                    hit |= consider(self._explicit_value(plot, wy), wy, False, layer)
            if hit:
                continue  # the point under the cursor beats any nearby vertex
            for entry in self._cache.get(layer.key, []):
                if not self._same_scale(entry):
                    continue
                self._trace_entry(entry, layer, explicit, pos, consider)

        if best is None:
            return None
        _, wx, wy, exact, layer = best
        label = self.format_position(wx, wy, exact)
        return _Trace(Point(*(float(v) for v in vp.to_screen(wx, wy))), label, layer.color)

    def _trace_entry(
        self,
        entry: _CacheEntry,
        layer: Layer,
        explicit: list[ExplicitPlot],
        pos: Point,
        consider: Callable[[float, float, bool, Layer], bool],
    ) -> None:
        vp = self.viewport
        mx, my = pos.x, pos.y
        # Fallback: nearest drawn vertex (steep parts, implicit curves, points).
        _, _, dx, dy = self._transform(entry)
        screen = entry.screen
        data = isinstance(layer.plot, PointSetPlot)  # vertices are exact data points
        candidates: list[tuple[Array, bool]] = [(line, data) for line in screen.lines]
        if screen.segments is not None and len(screen.segments):
            candidates.append((screen.segments[:, :2], False))
        if screen.points is not None:
            candidates.append((screen.points, True))
        for pts, exact in candidates:
            d2 = (pts[:, 0] + dx - mx) ** 2 + (pts[:, 1] + dy - my) ** 2
            if not np.isfinite(d2).any():
                continue
            i = int(np.nanargmin(d2))
            wx, wy = (float(v) for v in vp.to_world(float(pts[i, 0]) + dx, float(pts[i, 1]) + dy))
            if exact and entry.geometry.points is not None and pts is screen.points:
                ox, oy = self._origin(entry.sampled)
                px, py = entry.geometry.points[i]
                wx, wy = float(px) + ox, float(py) + oy
            if len(explicit) == 1 and not exact:  # exact value, not the vertex
                plot = explicit[0]
                if plot.independent == "x":
                    wy = self._explicit_value(plot, wx)
                else:
                    wx = self._explicit_value(plot, wy)
            consider(wx, wy, exact, layer)

    def _explicit_value(self, plot: ExplicitPlot, at: float) -> float:
        """The function's value at canvas coordinate ``at`` (exact in a deep zoom)."""
        vp = self.viewport
        if not vp.anchored:
            return float(plot.evaluate(at))
        axes = ("x", "y") if plot.independent == "x" else ("y", "x")
        anchors = [vp.anchor_x if axis == "x" else vp.anchor_y for axis in axes]
        return float(plot.evaluate_offset(np.array([at]), vp.bits, *anchors)[0])

    # -- input ------------------------------------------------------------------------------
    def _set_cursor(self, cursor: ft.MouseCursor) -> None:
        if self.detector.mouse_cursor != cursor:
            self.detector.mouse_cursor = cursor
            runtime.push(self.detector)

    def _on_resize(self, event: cv.CanvasResizeEvent) -> None:
        if math.isfinite(event.width) and math.isfinite(event.height):
            self.resize(round(event.width), round(event.height))

    def _on_scroll(self, event: ft.ScrollEvent[ft.GestureDetector]) -> None:
        delta = event.scroll_delta
        pixels = delta.y if delta.y else delta.x
        self.wheel(_xy(event.local_position), -pixels * WHEEL_UNITS_PER_PX)

    def wheel(
        self, pos: Point, delta: float, *, ctrl: bool | None = None, shift: bool | None = None
    ) -> None:
        """Zoom at ``pos`` by a wheel turn (120 units a notch, positive away from the user).

        With an unlocked aspect ratio, Ctrl zooms only x and Shift only y; by default the
        keys held now are used."""
        if not delta:
            return
        ctrl = runtime.modifiers.ctrl if ctrl is None else ctrl
        shift = runtime.modifiers.shift if shift is None else shift
        factor = WHEEL_ZOOM_BASE**delta
        if self.view3d.active:
            # Ctrl zooms the floor of the box and Shift its height, unless locked.
            axes = "xyz"
            if not self.settings.lock_aspect:
                axes = "xy" if ctrl else "z" if shift else "xyz"
            self.view3d.zoom(factor, axes)
            self._view_changed()
            return
        independent = not self.settings.lock_aspect and not self.settings.polar_grid
        if independent and ctrl:
            self.viewport.zoom_axes_at(pos.x, pos.y, factor, 1.0)  # x only
        elif independent and shift:
            self.viewport.zoom_axes_at(pos.x, pos.y, 1.0, factor)  # y only
        else:
            self.viewport.zoom_at(pos.x, pos.y, factor)
        self._view_changed()

    def press(self, pos: Point) -> None:
        """The left button went down at ``pos``: grab a point or start panning."""
        self.pressed.emit()
        grabbed = self._draggable_at(pos)
        if grabbed is not None:
            layer, point = grabbed
            position = self.viewport.to_world(point.x, point.y)
            self._point_drag = _PointDrag(
                layer.key, layer.color, point - pos, (float(position[0]), float(position[1]))
            )
            self._trace = None
            self._set_cursor(ft.MouseCursor.GRABBING)
            self.update()
            return
        self._drag_last = pos
        self._press_pos = pos
        self._trace = None
        self.view3d.set_moving(True)
        self._set_cursor(ft.MouseCursor.GRABBING)
        self.update()

    def move(self, pos: Point) -> None:
        """The pointer moved to ``pos`` (with the button down while dragging)."""
        if self._point_drag is not None:
            drag = self._point_drag
            target = pos + drag.offset
            world = self.viewport.to_world(target.x, target.y)
            position = self._snap(float(world[0]), float(world[1]))
            if position != drag.position:
                drag.position = position
                self.point_dragged.emit(drag.key, *position)
                self.update()
        elif self._drag_last is not None:
            delta = pos - self._drag_last
            self._drag_last = pos
            if not self.view3d.active:
                self.viewport.pan_pixels(delta.x, delta.y)
            elif runtime.modifiers.shift:
                self.view3d.pan(delta.x, delta.y)
            else:
                self.view3d.turn(delta.x, delta.y)
            self._view_changed()
        else:
            trace = self._find_trace(pos)
            if trace != self._trace:
                self._trace = trace
                self.update()
            over_point = self._draggable_at(pos) is not None
            self._set_cursor(ft.MouseCursor.CLICK if over_point else ft.MouseCursor.GRAB)
        world = self.viewport.to_world(pos.x, pos.y)
        self.cursor_moved.emit(float(world[0]), float(world[1]))

    def release(self, pos: Point | None) -> None:
        """The left button went up (at ``pos`` when known)."""
        if self._point_drag is not None:
            key = self._point_drag.key
            self._point_drag = None
            self._set_cursor(ft.MouseCursor.CLICK)
            self.point_drag_finished.emit(key)
            self.update()
            return
        press = self._press_pos
        self._drag_last = self._press_pos = None
        self.view3d.set_moving(False)
        self._set_cursor(ft.MouseCursor.GRAB)
        if pos is None:
            return
        if press is not None and math.hypot(pos.x - press.x, pos.y - press.y) <= CLICK_SLOP_PX:
            self.toggle_pin_at(pos)
        self._trace = self._find_trace(pos)
        self.update()

    def cancel_press(self) -> None:
        if self._point_drag is not None:
            self.release(None)
        self._drag_last = self._press_pos = None
        self.view3d.set_moving(False)

    # -- touch ------------------------------------------------------------------------------
    def touch_start(self, pos: Point, pointers: int) -> None:
        """One or more fingers started moving: as a press, with room for a pinch."""
        self._pinch_scale = 1.0
        self._trace = None
        if pointers == 1:
            self.press(pos)
            return
        self.pressed.emit()
        self._drag_last = pos
        self.view3d.set_moving(True)
        self.update()

    def touch_move(self, pos: Point, scale: float, pointers: int) -> None:
        """The fingers moved: one drags as the mouse does; two also pinch, about their
        centre, and move the view without turning it."""
        if pointers == 1:
            self.move(pos)
            return
        factor = scale / self._pinch_scale if self._pinch_scale > 0 and scale > 0 else 1.0
        self._pinch_scale = scale
        last = self._drag_last or pos
        self._drag_last = pos
        delta = pos - last
        if self.view3d.active:
            self.view3d.pan(delta.x, delta.y)
            self.view3d.zoom(factor)
        else:
            self.viewport.pan_pixels(delta.x, delta.y)
            self.viewport.zoom_at(pos.x, pos.y, factor)
        self._view_changed()

    def touch_end(self) -> None:
        if self._point_drag is not None:
            self.release(None)
        self._drag_last = self._press_pos = None
        self.view3d.set_moving(False)
        self.update()

    def tap(self, pos: Point) -> None:
        """A finger touched and left: pin a label, and read what is under it."""
        self.toggle_pin_at(pos)
        self.read_at(pos)

    def read_at(self, pos: Point) -> None:
        """Show the coordinates of what is under a finger, as hovering does for a mouse."""
        trace = self._find_trace(pos)
        if trace != self._trace:
            self._trace = trace
            self.update()
        self.cursor_moved.emit(*(float(v) for v in self.viewport.to_world(pos.x, pos.y)))

    def toggle_pin_at(self, pos: Point) -> bool:
        """Click on a point of interest: pin (or unpin) its label."""
        marker = self._marker_at(pos)
        if marker is None:
            return False
        key = (marker.poi.x, marker.poi.y)
        self._pinned.symmetric_difference_update({key})
        self.update()
        return True

    def double_click(self, pos: Point, *, shift: bool | None = None) -> None:
        shift = runtime.modifiers.shift if shift is None else shift
        if self.view3d.active:
            self.view3d.zoom(0.5 if shift else 2.0)
        else:
            self.viewport.zoom_at(pos.x, pos.y, 0.5 if shift else 2.0)
        self._view_changed()

    def leave(self) -> None:
        if self._trace is not None:
            self._trace = None
            self.update()

    def handle_key(self, key: str) -> bool:
        """Keyboard navigation (when no text field has the focus); True if handled."""
        if self.view3d.active:
            return self._handle_key_3d(key)
        vp = self.viewport
        if key in ("+", "=", "Numpad Add", "Numpad Equal"):
            vp.zoom_center(1.25)
        elif key in ("-", "_", "Numpad Subtract"):
            vp.zoom_center(0.8)
        elif key == "Arrow Left":
            vp.pan_pixels(KEY_PAN_PX, 0)
        elif key == "Arrow Right":
            vp.pan_pixels(-KEY_PAN_PX, 0)
        elif key == "Arrow Up":
            vp.pan_pixels(0, KEY_PAN_PX)
        elif key == "Arrow Down":
            vp.pan_pixels(0, -KEY_PAN_PX)
        elif key in ("Home", "0", "Numpad 0"):
            vp.reset()
        else:
            return False
        self._view_changed()
        return True

    def _handle_key_3d(self, key: str) -> bool:
        view = self.view3d
        if key in ("+", "=", "Numpad Add", "Numpad Equal"):
            view.zoom(1.25)
        elif key in ("-", "_", "Numpad Subtract"):
            view.zoom(0.8)
        elif key == "Arrow Left":
            view.turn_by(-KEY_TURN, 0.0)
        elif key == "Arrow Right":
            view.turn_by(KEY_TURN, 0.0)
        elif key == "Arrow Up":
            view.turn_by(0.0, KEY_TURN)
        elif key == "Arrow Down":
            view.turn_by(0.0, -KEY_TURN)
        elif key in ("Home", "0", "Numpad 0"):
            view.reset()
        else:
            return False
        self._view_changed()
        return True

    async def capture_png(self) -> bytes:
        """The canvas as drawn on the client, as PNG data."""
        await self.canvas.capture()
        try:
            return await self.canvas.get_capture()
        finally:
            await self.canvas.clear_capture()  # type: ignore[no-untyped-call]


def _reach() -> float:
    """How far from something the pointer can be and still be on it."""
    return TOUCH_RADIUS_PX if runtime.device.touch else HOVER_RADIUS_PX


def _xy(offset: ft.Offset | None) -> Point:
    if offset is None:
        return Point(0.0, 0.0)
    return Point(float(offset.x), float(offset.y))


def _freeze(shapes: list[cv.Shape]) -> list[cv.Shape]:
    """Mark shapes as immutable, the way Flet marks a component's rendered controls.

    Flet then compares a retained shape by identity alone and keeps no snapshot of its
    point list, so redrawing an unchanged curve costs nothing to diff or encode. The
    shapes must not be modified afterwards (Flet raises if they are).
    """
    for shape in shapes:
        object.__setattr__(shape, "_frozen", True)
    return shapes


def _stroke(color: str, width: float, *, dashed: bool = False, anti_alias: bool = True) -> ft.Paint:
    return ft.Paint(
        color=color,
        stroke_width=width,
        style=ft.PaintingStyle.STROKE,
        stroke_cap=ft.StrokeCap.ROUND,
        stroke_join=ft.StrokeJoin.ROUND,
        stroke_dash_pattern=[3 * width, 2 * width] if dashed else None,
        anti_alias=anti_alias,
    )


def _fill(color: str) -> ft.Paint:
    return ft.Paint(color=color, style=ft.PaintingStyle.FILL)


def _label_style(color: str) -> ft.TextStyle:
    return ft.TextStyle(size=LABEL_FONT.size, font_family=LABEL_FONT.family, color=color)


def _text(text: str, center: Point, color: str) -> cv.Text:
    return cv.Text(
        center.x, center.y, text, style=_label_style(color), alignment=ft.Alignment.CENTER
    )


def _offsets(points: list[tuple[float, float]]) -> list[ft.OffsetValue]:
    return list(points)


def _lines(points: list[tuple[float, float]], paint: ft.Paint) -> cv.Shape:
    """Separate segments from consecutive pairs of points (dashed ones as a path)."""
    if paint.stroke_dash_pattern:
        elements: list[cv.Path.PathElement] = []
        for (x0, y0), (x1, y1) in zip(points[::2], points[1::2], strict=False):
            elements += [cv.Path.MoveTo(x0, y0), cv.Path.LineTo(x1, y1)]
        return cv.Path(elements, paint=paint)
    return cv.Points(_offsets(points), cv.PointMode.LINES, paint=paint)


def _polyline_path(lines: list[list[tuple[float, float]]], paint: ft.Paint) -> cv.Path:
    elements: list[cv.Path.PathElement] = []
    for points in lines:
        elements.append(cv.Path.MoveTo(*points[0]))
        elements += [cv.Path.LineTo(x, y) for x, y in points[1:]]
    return cv.Path(elements, paint=paint)


def _dots(points: list[tuple[float, float]], diameter: float, color: str) -> cv.Points:
    paint = ft.Paint(
        color=color,
        stroke_width=diameter,
        stroke_cap=ft.StrokeCap.ROUND,
        style=ft.PaintingStyle.STROKE,
    )
    return cv.Points(_offsets(points), cv.PointMode.POINTS, paint=paint)


def place_points(
    points: Array, kx: float, ky: float, dx: float, dy: float
) -> list[tuple[float, float]]:
    """Screen positions ``points * k + d``, rounded to 0.01 px, as tuples."""
    if native.kernels is not None:
        placed: list[tuple[float, float]] = native.kernels.place(
            np.ascontiguousarray(points, dtype=np.float64), kx, ky, dx, dy
        )
        return placed
    moved = np.column_stack([points[:, 0] * kx + dx, points[:, 1] * ky + dy])
    return [(x, y) for x, y in np.round(moved, 2).tolist()]


def _chunk(kind: bytes, data: bytes) -> bytes:
    body = kind + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))


def fill_png(fill: FillImage, color: str) -> bytes:
    """A shading raster as a PNG: 1 bit per pixel, with the colour (at ``FILL_ALPHA``)
    and transparency in a two-entry palette."""
    r, g, b, _ = colors.rgba(color)
    header = struct.pack(">IIBBBBB", fill.width, fill.height, 1, 3, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", header)
        + _chunk(b"PLTE", bytes((0, 0, 0, r, g, b)))
        + _chunk(b"tRNS", bytes((0, FILL_ALPHA)))
        + _chunk(b"IDAT", fill.idat)
        + _chunk(b"IEND", b"")
    )


def png_bytes(pixels: Array) -> bytes:
    """Encode an ``(height, width, 4)`` uint8 RGBA array as a PNG."""
    height, width = pixels.shape[:2]
    rows = np.zeros((height, width * 4 + 1), dtype=np.uint8)  # filter byte 0 per row
    rows[:, 1:] = pixels.reshape(height, width * 4)
    header = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", zlib.compress(rows.tobytes(), 6))
        + _chunk(b"IEND", b"")
    )
