"""Drawable mathematical objects.

A plot knows how to turn itself into :class:`Geometry` for a visible rectangle. Plots
compare equal when their resolved expressions are equal, which lets the controller keep
the previous object (and the canvas keep its cached geometry) when an edit elsewhere in
the document did not change what a row draws.

New plot kinds are added by implementing another class with the same ``sample``
method; the canvas only consumes ``Geometry``. Sampling happens in *axis* coordinates
(log10 on a logarithmic axis, see :class:`~opengraph.engine.sampling.ViewRect`) and the
geometry is returned in world coordinates.

A deep-zoom view (``view.bits > 0``) has bounds that are offsets from an exact origin.
Plots then evaluate ``f(origin + offset)`` in MPFR (:mod:`~opengraph.engine.precise`)
and return geometry as offsets from the same origin; complex-mode expressions, which
have no precise form, fall back to float64 there.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

import gmpy2
import numpy as np
from numpy.typing import NDArray

from opengraph.engine import precise
from opengraph.engine.ast import BinOp, Const, FuncCall, Guard, Node, Num, UnaryOp, Var
from opengraph.engine.builtins import Array
from opengraph.engine.evaluator import NumericFunction
from opengraph.engine.sampling import (
    FillMask,
    ViewRect,
    from_axis,
    marching_squares,
    region_mask,
    sample_curve,
    sample_parametric,
    to_axis,
)

POLAR_THETA_MAX = 12 * math.pi  # like Desmos: long enough for spirals and rose curves


def shifted(
    fn: NumericFunction,
    bits: int,
    inputs: Mapping[str, tuple[Any, Array | float]],
    out_origin: Any = 0,
) -> Array:
    """``fn(origin + offset) - out_origin`` for each input ``name: (origin, offsets)``,
    evaluated with ``bits`` of precision (float64 for complex-mode expressions)."""
    if not fn.complex_mode:
        try:
            return precise.PreciseFunction(fn.node).evaluate(bits, inputs, out_origin)
        except precise.Unsupported:
            pass
    values = fn(**{name: float(origin) + np.asarray(off) for name, (origin, off) in inputs.items()})
    return np.asarray(values - float(out_origin), dtype=np.float64)


def offsets(values: Array, origin: Any, bits: int) -> Array:
    """World values as offsets from an exact ``origin``."""
    if not origin:
        return values
    with precise.context(bits):
        reference = gmpy2.mpfr(origin)
        return np.array([float(gmpy2.mpfr(float(v)) - reference) for v in values.ravel()]).reshape(
            values.shape
        )


def _zero_form(node: Node) -> Node:
    """Strip outer operations that preserve a zero set, including repeated factors."""
    match node:
        case BinOp(op="-", left=left, right=Num(value=0)):
            return _zero_form(left)
        case BinOp(op="-", left=Num(value=0), right=right):
            return _zero_form(right)
        case UnaryOp(operand=operand):
            return _zero_form(operand)
        case BinOp(op="^", left=base, right=Num(value=exponent)) if exponent > 0:
            return _zero_form(base)
        case BinOp(op="*", left=left, right=right) if left == right:
            return _zero_form(left)
        case FuncCall(name=name, args=(arg,)) if name in ("abs", "sqrt", "cbrt"):
            return _zero_form(arg)
    return node


@dataclass(frozen=True, slots=True)
class Geometry:
    """World-coordinate primitives to draw."""

    polylines: tuple[Array, ...] = ()  # each (N, 2)
    segments: Array | None = None  # (M, 4) independent line segments
    points: Array | None = None  # (K, 2)
    fill: FillMask | None = None  # shaded region of an inequality
    dashed: bool = False  # the lines are a strict inequality's open boundary

    @property
    def is_empty(self) -> bool:
        return (
            not self.polylines
            and (self.segments is None or len(self.segments) == 0)
            and (self.points is None or len(self.points) == 0)
            and self.fill is None
        )

    @staticmethod
    def merge(parts: list[Geometry]) -> Geometry:
        segments = [p.segments for p in parts if p.segments is not None and len(p.segments)]
        points = [p.points for p in parts if p.points is not None and len(p.points)]
        return Geometry(
            polylines=tuple(line for p in parts for line in p.polylines),
            segments=np.vstack(segments) if segments else None,
            points=np.vstack(points) if points else None,
            fill=next((p.fill for p in parts if p.fill is not None), None),
            dashed=any(p.dashed for p in parts),
        )


@dataclass(frozen=True, eq=True)
class ExplicitPlot:
    """``y = f(x)`` (``independent == "x"``) or ``x = f(y)`` (``independent == "y"``)."""

    node: Node
    independent: str = "x"
    complex_mode: bool = False
    fn: NumericFunction = field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "fn", NumericFunction(self.node, self.complex_mode))

    def evaluate(self, t: Array | float) -> Array:
        return self.fn(**{self.independent: t})

    def evaluate_offset(self, t: Array | float, bits: int, t_origin: Any, v_origin: Any) -> Array:
        """``f(t_origin + t) - v_origin``, in ``bits`` of precision (a deep zoom)."""
        return shifted(self.fn, bits, {self.independent: (t_origin, t)}, v_origin)

    def sample(self, view: ViewRect) -> Geometry:
        if view.bits:
            return self._sample_deep(view)
        if self.independent == "x":
            t_log, v_log = view.x_log, view.y_log
            t_range = (view.u_min, view.u_max, view.px_per_x)
            v_range = (view.v_min, view.v_max, view.px_per_y)
        else:
            t_log, v_log = view.y_log, view.x_log
            t_range = (view.v_min, view.v_max, view.px_per_y)
            v_range = (view.u_min, view.u_max, view.px_per_x)

        def evaluate(u: Array) -> Array:
            return np.asarray(to_axis(self.evaluate(from_axis(u, t_log)), v_log))

        pieces = sample_curve(evaluate, *t_range, *v_range)
        if self.independent == "x":
            lines = tuple(
                np.column_stack([from_axis(t, t_log), from_axis(v, v_log)]) for t, v in pieces
            )
        else:
            lines = tuple(
                np.column_stack([from_axis(v, v_log), from_axis(t, t_log)]) for t, v in pieces
            )
        return Geometry(polylines=lines)

    def _sample_deep(self, view: ViewRect) -> Geometry:
        x_ranges = (view.x_min, view.x_max, view.px_per_x)
        y_ranges = (view.y_min, view.y_max, view.px_per_y)
        if self.independent == "x":
            origins, ranges = (view.origin_x, view.origin_y), (x_ranges, y_ranges)
        else:
            origins, ranges = (view.origin_y, view.origin_x), (y_ranges, x_ranges)

        def evaluate(t: Array) -> Array:
            return self.evaluate_offset(t, view.bits, *origins)

        pieces = sample_curve(evaluate, *ranges[0], *ranges[1])
        if self.independent == "x":
            return Geometry(polylines=tuple(np.column_stack([t, v]) for t, v in pieces))
        return Geometry(polylines=tuple(np.column_stack([v, t]) for t, v in pieces))


@dataclass(frozen=True, eq=True)
class ImplicitPlot:
    """The zero set of ``F(x, y)``; built from an equation ``lhs = rhs`` as lhs - rhs."""

    node: Node
    complex_mode: bool = False
    fn: NumericFunction = field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        normalized = _zero_form(self.node)
        if normalized != self.node:
            normalized = Guard(normalized, self.node)
        object.__setattr__(self, "fn", NumericFunction(normalized, self.complex_mode))

    def sample(self, view: ViewRect) -> Geometry:
        if view.bits:
            ox, oy = view.origin_x, view.origin_y
            return Geometry(
                segments=marching_squares(
                    lambda x, y: shifted(self.fn, view.bits, {"x": (ox, x), "y": (oy, y)}), view
                )
            )
        return Geometry(segments=marching_squares(lambda x, y: self.fn(x=x, y=y), view))


@dataclass(frozen=True, eq=True)
class RegionPlot:
    """The inequality ``F(x, y) <op> 0`` shaded, with its boundary (dashed if strict).

    ``boundary`` is the plot of the equation ``F = 0`` when it can be drawn.
    """

    node: Node
    op: str  # "<", "<=", ">" or ">="
    boundary: Plot | None = None
    constraints: tuple[tuple[Node, str], ...] = ()
    shade: bool = True
    fn: NumericFunction = field(init=False, compare=False, repr=False)
    constraint_functions: tuple[tuple[NumericFunction, str], ...] = field(
        init=False, compare=False, repr=False
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "fn", NumericFunction(self.node))
        object.__setattr__(
            self,
            "constraint_functions",
            tuple((NumericFunction(node), op) for node, op in self.constraints),
        )

    def holds(self, x: Array, y: Array, view: ViewRect | None = None) -> NDArray[np.bool_]:
        """Where the inequality holds at ``(x, y)``: world values, or offsets from the
        origin of a deep-zoom ``view``."""

        def value(function: NumericFunction) -> Array:
            if view is not None and view.bits:
                inputs = {"x": (view.origin_x, x), "y": (view.origin_y, y)}
                return shifted(function, view.bits, inputs)
            return np.asarray(function(x=x, y=y), dtype=np.float64)

        result = self.compare(value(self.fn), self.op)
        for function, op in self.constraint_functions:
            result = result & self.compare(value(function), op)
        return result

    @staticmethod
    def compare(value: Array, op: str) -> NDArray[np.bool_]:
        match op:
            case "<":
                return value < 0
            case "<=":
                return value <= 0
            case ">":
                return value > 0
            case _:
                return value >= 0

    def sample(self, view: ViewRect) -> Geometry:
        edge = self.boundary.sample(view) if self.boundary is not None else Geometry()
        if self.constraints:
            parts = [] if edge.segments is None else [edge.segments]
            parts += [np.column_stack((line[:-1], line[1:])) for line in edge.polylines]
            segments = np.vstack(parts) if parts else np.empty((0, 4))
            mask = np.ones(len(segments), dtype=bool)
            for function, op in self.constraint_functions:
                x = (segments[:, 0] + segments[:, 2]) / 2
                y = (segments[:, 1] + segments[:, 3]) / 2
                if view.bits:
                    inputs = {"x": (view.origin_x, x), "y": (view.origin_y, y)}
                    mask &= self.compare(shifted(function, view.bits, inputs), op)
                else:
                    mask &= self.compare(function(x=x, y=y), op)
            edge = Geometry(segments=segments[mask])
        return Geometry(
            polylines=edge.polylines,
            segments=edge.segments,
            points=edge.points,
            fill=region_mask(lambda x, y: self.holds(x, y, view), view) if self.shade else None,
            dashed=len(self.op) == 1,
        )


@dataclass(frozen=True, eq=True)
class PolarPlot:
    """``r = f(θ)`` for θ in ``[theta_min, theta_max]`` (radians, or degrees if set)."""

    node: Node
    degrees: bool = False
    complex_mode: bool = False
    theta_min: float = 0.0
    theta_max: float = POLAR_THETA_MAX
    fn: NumericFunction = field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "fn", NumericFunction(self.node, self.complex_mode))

    def radius(self, angle: Array | float) -> Array:
        """r at ``angle`` radians (converted to the expression's angle unit)."""
        theta = np.asarray(angle) * (180 / math.pi) if self.degrees else angle
        return self.fn(θ=theta)

    def xy(self, angle: Array) -> tuple[Array, Array]:
        r = self.radius(angle)
        return r * np.cos(angle), r * np.sin(angle)

    def sample(self, view: ViewRect) -> Geometry:
        xy = self.xy
        if view.bits:
            # θ in the expression's unit; the angle of cos/sin in radians, both exact.
            theta: Node = Var("θ")
            radians = (
                BinOp("*", theta, BinOp("/", Const("pi", math.pi), Num(180.0)))
                if self.degrees
                else theta
            )
            fx = NumericFunction(BinOp("*", self.node, FuncCall("cos", (radians,))))
            fy = NumericFunction(BinOp("*", self.node, FuncCall("sin", (radians,))))
            scale = 180 / math.pi if self.degrees else 1.0

            def xy(angle: Array) -> tuple[Array, Array]:
                if self.complex_mode:
                    x, y = self.xy(angle)
                    return x - float(view.origin_x), y - float(view.origin_y)
                inputs = {"θ": (0, np.asarray(angle) * scale)}
                return (
                    shifted(fx, view.bits, inputs, view.origin_x),
                    shifted(fy, view.bits, inputs, view.origin_y),
                )

        lines = sample_parametric(xy, self.theta_min, self.theta_max, view)
        return Geometry(polylines=tuple(lines))


@dataclass(frozen=True, eq=True)
class ParametricPlot:
    """``(x(t), y(t))`` for ``t`` in ``[t_min, t_max]``."""

    x_node: Node
    y_node: Node
    t_min: float = 0.0
    t_max: float = 2 * math.pi
    complex_mode: bool = False
    fx: NumericFunction = field(init=False, compare=False, repr=False)
    fy: NumericFunction = field(init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "fx", NumericFunction(self.x_node, self.complex_mode))
        object.__setattr__(self, "fy", NumericFunction(self.y_node, self.complex_mode))

    def xy(self, t: Array) -> tuple[Array, Array]:
        x, y = self.fx(t=t), self.fy(t=t)  # a constant coordinate evaluates to a scalar
        return np.broadcast_to(x, t.shape), np.broadcast_to(y, t.shape)

    def sample(self, view: ViewRect) -> Geometry:
        xy = self.xy
        if view.bits:

            def xy(t: Array) -> tuple[Array, Array]:
                inputs = {"t": (0, t)}
                return (
                    np.broadcast_to(shifted(self.fx, view.bits, inputs, view.origin_x), t.shape),
                    np.broadcast_to(shifted(self.fy, view.bits, inputs, view.origin_y), t.shape),
                )

        return Geometry(polylines=tuple(sample_parametric(xy, self.t_min, self.t_max, view)))


@dataclass(frozen=True, eq=True)
class PointPlot:
    x: float
    y: float

    def sample(self, view: ViewRect) -> Geometry:
        point = np.array([[self.x, self.y]])
        if view.bits:
            point = np.column_stack(
                [
                    offsets(point[:, 0], view.origin_x, view.bits),
                    offsets(point[:, 1], view.origin_y, view.bits),
                ]
            )
        return Geometry(points=point)


@dataclass(frozen=True, eq=True)
class PointSetPlot:
    """Several points (a list of points, a table or a dataset).

    ``markers`` draws a dot at each point and ``lines`` joins consecutive points; an
    undefined point (NaN) breaks the line.
    """

    points: tuple[tuple[float, float], ...]
    markers: bool = True
    lines: bool = False

    def array(self) -> Array:
        return np.array(self.points, dtype=np.float64).reshape(-1, 2)

    def sample(self, view: ViewRect) -> Geometry:
        data = self.array()
        if view.bits:
            data = np.column_stack(
                [
                    offsets(data[:, 0], view.origin_x, view.bits),
                    offsets(data[:, 1], view.origin_y, view.bits),
                ]
            ).reshape(-1, 2)
        finite = np.isfinite(data).all(axis=1)
        polylines: tuple[Array, ...] = ()
        if self.lines and len(data) > 1:
            cut = np.nonzero(~finite)[0]
            pieces = np.split(data, cut)
            polylines = tuple(
                piece[np.isfinite(piece).all(axis=1)]
                for piece in pieces
                if np.isfinite(piece).all(axis=1).sum() >= 2
            )
        return Geometry(polylines=polylines, points=data[finite] if self.markers else None)


@dataclass(frozen=True, eq=True)
class PlotGroup:
    """Several plots drawn as one row, e.g. ``y = kx`` with ``k = [1, 2, 3]``."""

    plots: tuple[Plot, ...]

    def sample(self, view: ViewRect) -> Geometry:
        return Geometry.merge([plot.sample(view) for plot in self.plots])


Plot = (
    ExplicitPlot
    | ImplicitPlot
    | RegionPlot
    | PolarPlot
    | ParametricPlot
    | PointPlot
    | PointSetPlot
    | PlotGroup
)


def members(plot: Plot) -> Iterator[Plot]:
    """The plot itself, or every plot of a group (flattened)."""
    if isinstance(plot, PlotGroup):
        for member in plot.plots:
            yield from members(member)
    else:
        yield plot
