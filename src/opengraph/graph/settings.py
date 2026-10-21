"""Per-document graph settings: axes, grid, angle unit and number mode. Toolkit-free.

Settings are stored in the document (an optional ``"settings"`` key, so older files
open with the defaults). Some combinations don't make sense together and are resolved
by :meth:`GraphSettings.normalized` rather than rejected:

* a polar grid needs equal, linear axes (circles must look like circles), so it turns
  logarithmic axes off and locks the aspect ratio;
* complex mode plots the complex plane, which is linear in both directions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Literal, assert_never

from opengraph.engine.compiler import CompileOptions
from opengraph.json_types import JSONObject, is_string_keyed_dict

GraphToggle = Literal[
    "polar_grid",
    "major_grid",
    "minor_grid",
    "axes",
    "axis_numbers",
    "lock_aspect",
    "degrees",
    "complex_mode",
    "points_of_interest",
]


@dataclass(frozen=True, slots=True)
class AxisSettings:
    log: bool = False
    step: float | None = None  # major grid spacing in world units; None = automatic

    def to_dict(self) -> JSONObject:
        return {"log": self.log, "step": self.step}

    @staticmethod
    def from_dict(data: object) -> AxisSettings:
        if not is_string_keyed_dict(data):
            return AxisSettings()
        raw = data.get("step")
        step: float | None = None
        if (
            isinstance(raw, int | float)
            and not isinstance(raw, bool)
            and math.isfinite(raw)
            and raw > 0
        ):
            step = float(raw)
        return AxisSettings(bool(data.get("log", False)), step)


@dataclass(frozen=True, slots=True)
class GraphSettings:
    x_axis: AxisSettings = field(default_factory=AxisSettings)
    y_axis: AxisSettings = field(default_factory=AxisSettings)
    polar_grid: bool = False
    major_grid: bool = True
    minor_grid: bool = True
    axes: bool = True
    axis_numbers: bool = True
    lock_aspect: bool = False  # keep the x:y scale ratio (no Ctrl/Shift+wheel axis zoom)
    degrees: bool = False
    complex_mode: bool = False
    points_of_interest: bool = True

    def normalized(self) -> GraphSettings:
        """These settings with incompatible combinations resolved (see module doc)."""
        result = self
        if result.polar_grid or result.complex_mode:
            result = replace(
                result,
                x_axis=replace(result.x_axis, log=False),
                y_axis=replace(result.y_axis, log=False),
            )
        if result.polar_grid:
            result = replace(result, lock_aspect=True)
        return result

    @property
    def any_log(self) -> bool:
        return self.x_axis.log or self.y_axis.log

    def compile_options(self) -> CompileOptions:
        return CompileOptions(degrees=self.degrees, complex_mode=self.complex_mode)

    def with_toggle(self, field: GraphToggle, enabled: bool) -> GraphSettings:
        """Return these settings with one menu-controlled boolean changed."""
        match field:
            case "polar_grid":
                return replace(self, polar_grid=enabled).normalized()
            case "major_grid":
                return replace(self, major_grid=enabled)
            case "minor_grid":
                return replace(self, minor_grid=enabled)
            case "axes":
                return replace(self, axes=enabled)
            case "axis_numbers":
                return replace(self, axis_numbers=enabled)
            case "lock_aspect":
                return replace(self, lock_aspect=enabled)
            case "degrees":
                return replace(self, degrees=enabled)
            case "complex_mode":
                return replace(self, complex_mode=enabled).normalized()
            case "points_of_interest":
                return replace(self, points_of_interest=enabled)
        assert_never(field)

    def to_dict(self) -> JSONObject:
        return {
            "x_axis": self.x_axis.to_dict(),
            "y_axis": self.y_axis.to_dict(),
            "polar_grid": self.polar_grid,
            "major_grid": self.major_grid,
            "minor_grid": self.minor_grid,
            "axes": self.axes,
            "axis_numbers": self.axis_numbers,
            "lock_aspect": self.lock_aspect,
            "degrees": self.degrees,
            "complex_mode": self.complex_mode,
            "points_of_interest": self.points_of_interest,
        }

    @staticmethod
    def from_dict(data: object) -> GraphSettings:
        """Parse stored settings leniently: unknown or malformed values use defaults."""
        if not is_string_keyed_dict(data):
            return GraphSettings()
        defaults = GraphSettings()
        return GraphSettings(
            x_axis=AxisSettings.from_dict(data.get("x_axis")),
            y_axis=AxisSettings.from_dict(data.get("y_axis")),
            polar_grid=_bool(data, "polar_grid", defaults.polar_grid),
            major_grid=_bool(data, "major_grid", defaults.major_grid),
            minor_grid=_bool(data, "minor_grid", defaults.minor_grid),
            axes=_bool(data, "axes", defaults.axes),
            axis_numbers=_bool(data, "axis_numbers", defaults.axis_numbers),
            lock_aspect=_bool(data, "lock_aspect", defaults.lock_aspect),
            degrees=_bool(data, "degrees", defaults.degrees),
            complex_mode=_bool(data, "complex_mode", defaults.complex_mode),
            points_of_interest=_bool(data, "points_of_interest", defaults.points_of_interest),
        ).normalized()


def _bool(data: dict[str, object], key: str, default: bool) -> bool:
    value = data.get(key)
    return value if isinstance(value, bool) else default
