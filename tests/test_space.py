"""3D graphing: the language, meshing, drawing and the 3D view in the window."""

from __future__ import annotations

import math
from pathlib import Path

import flet.canvas as cv
import numpy as np
import pytest

from opengraph import events
from opengraph.document import Document
from opengraph.engine.compiler import compile_document, compile_expression
from opengraph.engine.meshing import (
    Box,
    clip_segments,
    curve_segments,
    height_field,
    implicit_surface,
    mesh_from,
)
from opengraph.engine.plots import (
    ExplicitPlot,
    ImplicitPlot,
    ImplicitSurfacePlot,
    ParametricPlot,
    Plot,
    PlotGroup,
    PointPlot,
    SpaceCurvePlot,
    SpaceGeometry,
    SpacePointPlot,
    SpacePointSetPlot,
    SurfacePlot,
    is_spatial,
    members,
    space_geometry,
)
from opengraph.graph.background import LatestTask
from opengraph.graph.raster import Camera, Part, Style, encode, render
from opengraph.ui.geometry import Point
from opengraph.ui.main_window import MainWindow
from tests.support import key, wait, wait_until

BOX = Box(-5.0, 5.0, -5.0, 5.0, -5.0, 5.0)
RED = (200, 0, 0)
BLUE = (0, 0, 200)
PLAIN = Style(show_grid=False, show_axes=False)


def plot_of(text: str) -> Plot:
    result = compile_expression(text)
    assert not result.is_error, result.message
    assert result.plot is not None, text
    return result.plot


def geometry_of(text: str, box: Box = BOX) -> SpaceGeometry:
    plot = plot_of(text)
    assert not isinstance(plot, PlotGroup)
    return space_geometry(plot, box)


# -- the language --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "kind", "dependent"),
    [
        ("z = x^2 + y^2", SurfacePlot, "z"),
        ("x^2 + y^2 = z", SurfacePlot, "z"),
        ("z = 3", SurfacePlot, "z"),
        ("x = y z", SurfacePlot, "x"),
        ("y = z", SurfacePlot, "y"),
        ("y = sin(x) + z", SurfacePlot, "y"),
        ("x + y + z = 1", SurfacePlot, "z"),  # linear in z: solved rather than contoured
        ("2z = x", SurfacePlot, "x"),  # the lone coordinate is the simplest way to draw it
        ("2z = 3x", SurfacePlot, "z"),
        ("x z + y = 1", SurfacePlot, "y"),  # z and x multiply, but y is linear
    ],
)
def test_explicit_surfaces(text: str, kind: type, dependent: str) -> None:
    plot = plot_of(text)
    assert isinstance(plot, kind)
    assert isinstance(plot, SurfacePlot) and plot.dependent == dependent


@pytest.mark.parametrize("text", ["x^2 + y^2 + z^2 = 9", "x y z = 1", "z^2 = 4", "sin(z) = x y"])
def test_implicit_surfaces(text: str) -> None:
    result = compile_expression(text)
    assert isinstance(result.plot, ImplicitSurfacePlot)
    assert not result.is_error
    assert "may be missed" in result.message  # like implicit curves, numerical


def test_a_list_draws_a_family_of_surfaces() -> None:
    plot = plot_of("z = list[1, 2, 3] x")
    assert isinstance(plot, PlotGroup) and len(plot.plots) == 3
    assert all(isinstance(part, SurfacePlot) for part in plot.plots)
    assert is_spatial(plot)


@pytest.mark.parametrize(
    "text", ["y = x^2", "x = 3", "x^2 + y^2 = 1", "r = sin(θ)", "point[1, 2]", "x^2"]
)
def test_rows_without_z_stay_flat(text: str) -> None:
    assert not is_spatial(plot_of(text))


def test_points_in_space() -> None:
    assert plot_of("point[1, 2, 3]") == SpacePointPlot(1.0, 2.0, 3.0)
    assert plot_of("point[1, 2, 3] + point[1, 1, 1]") == SpacePointPlot(2.0, 3.0, 4.0)
    assert plot_of("2 point[1, 2, 3]") == SpacePointPlot(2.0, 4.0, 6.0)
    several = plot_of("point[list[1, 2], list[3, 4], 0]")
    assert several == SpacePointSetPlot(((1.0, 3.0, 0.0), (2.0, 4.0, 0.0)))
    assert plot_of("point[1, 2, 3], point[4, 5, 6]") == SpacePointSetPlot(
        ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))
    )


def test_a_defined_point_in_space_plots_but_does_not_drag() -> None:
    defined = compile_document([(0, "p = point[1, 2, 3]")])[0]
    assert defined.plot == SpacePointPlot(1.0, 2.0, 3.0)
    assert defined.drag is None
    assert compile_expression("point[1, 2, 3]").drag is None
    assert compile_expression("point[1, 2]").drag is not None


@pytest.mark.parametrize(
    "text",
    [
        "parametric(cos(u), sin(u), u for u in [0, 6pi])",
        "parametric(cos(u), sin(u), u, u, [0, 6pi])",
        "point[[0, 1], [0, 2], [0, 3]]",
    ],
)
def test_curves_in_space(text: str) -> None:
    assert isinstance(plot_of(text), SpaceCurvePlot)


def test_flat_parametric_forms_keep_their_meaning() -> None:
    assert isinstance(plot_of("parametric(cos(u), sin(u) for u in [0, 1])"), ParametricPlot)
    assert isinstance(plot_of("parametric(cos(u), sin(u), u, [0, 1])"), ParametricPlot)
    assert isinstance(plot_of("point[[0, 1], [0, 2]]"), ParametricPlot)


def test_a_degenerate_curve_in_space_is_a_point() -> None:
    assert plot_of("parametric(u, u, u for u in [2, 2])") == SpacePointPlot(2.0, 2.0, 2.0)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("z", "need an equation"),
        ("x + z", "need an equation"),
        ("z < x", "Inequalities in space"),
        ("x + y + z >= 1", "Inequalities in space"),
        ("0 < z < 1", "Inequalities in space"),
        ("z = [0, 1) x", "interval"),
        ("r + z = 1", "either x, y and z or r and θ"),
        ("r^2 = z", "either x, y and z or r and θ"),
        ("point[1, 2] + point[1, 2, 3]", "a 2D point and a 3D point"),
        ("point[1, 2], point[1, 2, 3]", "mix 2D and 3D"),
        ("point[1, x, 3]", "must be numbers"),
        ("a = z", "define a function instead"),
        ("z = 2", None),
    ],
)
def test_what_is_not_supported_says_so(text: str, message: str | None) -> None:
    result = compile_expression(text)
    if message is None:
        assert not result.is_error
    else:
        assert result.is_error
        assert message in result.message


def test_z_is_a_coordinate_and_not_a_definition() -> None:
    results = compile_document([(0, "z = 5"), (1, "y = z")])
    assert results[0].defines is None and results[0].literal_value is None
    assert isinstance(results[0].plot, SurfacePlot)
    assert isinstance(results[1].plot, SurfacePlot)  # the plane y = z, not a line y = 5


def test_z_still_works_as_a_local_name() -> None:
    assert isinstance(plot_of("f(z) = z^2"), ExplicitPlot)
    assert compile_expression("sum(z^2 for z from 1 to 3)").message == "= 14"


def test_definitions_feed_surfaces() -> None:
    results = compile_document(
        [(0, "a = 2"), (1, "g(u, v) = u v"), (2, "z = a g(x, y)"), (3, "k = 1")]
    )
    assert results[1].plot is None  # defining a function of two names doesn't draw it
    assert isinstance(results[2].plot, SurfacePlot)
    assert results[3].literal_value == 1


def test_undefined_names_offer_sliders_in_surfaces() -> None:
    result = compile_document([(0, "z = a x")])[0]
    assert result.is_error and result.undefined == ("a",)


def test_complex_mode_explains_why_there_are_no_surfaces() -> None:
    from opengraph.engine.compiler import CompileOptions

    result = compile_expression("z = x y", CompileOptions(complex_mode=True))
    assert result.is_error and "complex mode" in result.message


def test_symbolic_operations_understand_z() -> None:
    result = compile_expression("diff(x^2 z for z)")
    assert not result.is_error and len(result.pending) == 1
    assert result.pending[0].variable == "z"


# -- meshing -------------------------------------------------------------------------------


def test_a_plane_covers_the_box_with_a_regular_grid() -> None:
    mesh = geometry_of("z = 1").mesh
    assert mesh is not None
    assert len(mesh.triangles) == 2 * 72 * 72
    assert mesh.vertices[:, 2] == pytest.approx(1.0)
    assert mesh.vertices[:, 0].min() == -5 and mesh.vertices[:, 1].max() == 5


@pytest.mark.parametrize(
    ("text", "axis", "value"), [("x = 2", 0, 2.0), ("y = -1", 1, -1.0), ("z = 3", 2, 3.0)]
)
def test_any_coordinate_can_be_the_dependent_one(text: str, axis: int, value: float) -> None:
    # x = 2 alone would be a flat line, so the plane is written with z.
    source = text if axis == 2 else f"{text} + 0 z"
    mesh = geometry_of(source).mesh
    assert mesh is not None and not mesh.is_empty
    assert mesh.vertices[:, axis] == pytest.approx(value)
    others = [a for a in range(3) if a != axis]
    assert np.ptp(mesh.vertices[:, others[0]]) == pytest.approx(10.0)
    assert np.ptp(mesh.vertices[:, others[1]]) == pytest.approx(10.0)


def test_surfaces_are_cut_exactly_at_the_box() -> None:
    mesh = geometry_of("z = x^2 + y^2").mesh
    assert mesh is not None
    assert mesh.vertices[:, 2].max() == pytest.approx(5.0)
    assert mesh.vertices[:, 2].min() >= -5.0
    # The points on the cut really are on the surface.
    x, y, z = mesh.vertices.T
    top = np.isclose(z, 5.0)
    assert top.sum() > 20
    assert (np.abs(x[top] ** 2 + y[top] ** 2 - 5.0) < 0.05).all()


def test_a_surface_follows_the_edge_of_its_domain() -> None:
    mesh = geometry_of("z = sqrt(9 - x^2 - y^2)").mesh
    assert mesh is not None
    x, y, z = mesh.vertices.T
    assert np.allclose(x**2 + y**2 + z**2, 9.0)
    radius = np.hypot(x, y)
    # Reaches around the rim rather than stopping at the last whole grid cell.
    rim = radius > 2.99
    angles = np.arctan2(y[rim], x[rim])
    assert len(angles) > 100
    assert np.diff(np.sort(angles)).max() < 0.2


def test_jumps_leave_no_walls() -> None:
    mesh = geometry_of("z = floor(x)").mesh
    assert mesh is not None and not mesh.is_empty
    heights = mesh.vertices[:, 2][mesh.triangles]
    assert np.ptp(heights, axis=1).max() == pytest.approx(0.0)  # every triangle is level
    assert set(np.unique(mesh.vertices[:, 2])) <= {float(n) for n in range(-5, 6)}


def test_poles_leave_no_walls() -> None:
    mesh = geometry_of("z = 1/x").mesh
    assert mesh is not None
    x = mesh.vertices[:, 0][mesh.triangles]
    assert not ((x.min(axis=1) < 0) & (x.max(axis=1) > 0)).any()  # nothing crosses x = 0


def test_steep_continuous_surfaces_keep_their_slope() -> None:
    mesh = geometry_of("z = 100 x").mesh
    assert mesh is not None and not mesh.is_empty
    x = mesh.vertices[:, 0]
    assert x.min() == pytest.approx(-0.05) and x.max() == pytest.approx(0.05)


def test_smooth_hills_are_not_mistaken_for_jumps() -> None:
    mesh = geometry_of("z = sin(x) cos(y)").mesh
    assert mesh is not None
    assert len(mesh.triangles) == 2 * 72 * 72  # smooth: nothing is dropped


def test_the_implicit_sphere_is_accurate_and_consistently_oriented() -> None:
    mesh = geometry_of("x^2 + y^2 + z^2 = 9").mesh
    assert mesh is not None and len(mesh.triangles) > 1000
    radii = np.linalg.norm(mesh.vertices, axis=1)
    assert np.abs(radii - 3.0).max() < 0.01
    # The normals point outwards everywhere: F grows outwards, and they follow F.
    outward = (mesh.normals * mesh.vertices).sum(axis=1)
    assert (outward > 0).all()
    assert np.linalg.norm(mesh.normals, axis=1) == pytest.approx(1.0, abs=1e-5)


def test_the_implicit_mesh_is_closed_inside_the_box() -> None:
    mesh = geometry_of("x^2 + y^2 + z^2 = 9").mesh
    assert mesh is not None
    edges = np.sort(mesh.triangles[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert (counts == 2).all()  # every edge joins exactly two triangles: no cracks


def test_a_surface_through_grid_points_has_no_dark_spots() -> None:
    """The radius-3 sphere passes exactly through grid points such as (0, 0, 3)."""
    mesh = geometry_of("x^2 + y^2 + z^2 = 9").mesh
    assert mesh is not None
    top = np.linalg.norm(mesh.vertices - np.array([0.0, 0.0, 3.0]), axis=1) < 1e-6
    assert top.sum() == 1  # one vertex, not several at the same place
    assert mesh.normals[top][0] == pytest.approx([0.0, 0.0, 1.0], abs=0.01)
    a, b, c = (mesh.vertices[mesh.triangles[:, i]] for i in range(3))
    assert (np.linalg.norm(np.cross(b - a, c - a), axis=1) > 0).all()


def test_an_implicit_surface_is_cut_by_the_box() -> None:
    mesh = geometry_of("x^2 + y^2 + z^2 = 49").mesh  # wider than the box, not its corners
    assert mesh is not None and not mesh.is_empty
    assert (np.abs(mesh.vertices) <= 5.0 + 1e-9).all()


def test_squared_zero_sets_are_found() -> None:
    mesh = geometry_of("(x^2 + y^2 + z^2 - 4)^2 = 0").mesh  # never changes sign
    assert mesh is not None and not mesh.is_empty
    assert np.abs(np.linalg.norm(mesh.vertices, axis=1) - 2.0).max() < 0.01


def test_a_pole_is_not_a_surface() -> None:
    # 1/x changes sign across x = 0 but has no zero there.
    assert geometry_of("1/(x - 0.1) + 0 z = 0").mesh.is_empty  # type: ignore[union-attr]


def test_nothing_to_find_gives_an_empty_mesh() -> None:
    mesh = geometry_of("x^2 + y^2 + z^2 = -1").mesh
    assert mesh is not None and mesh.is_empty
    points, triangles = implicit_surface(lambda x, y, z: x + 100.0, BOX, 8)
    assert not len(points) and not len(triangles)
    assert mesh_from(points, triangles, BOX).is_empty


def test_height_field_returns_nothing_for_an_undefined_function() -> None:
    def nowhere(u: np.ndarray, v: np.ndarray) -> np.ndarray:
        return np.full_like(u, np.nan)

    _, triangles = height_field(nowhere, (-1, 1), (-1, 1), (-1, 1), 8)
    assert not len(triangles)


def test_clip_segments() -> None:
    box = Box(-1, 1, -1, 1, -1, 1)
    segments = np.array(
        [
            [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],  # inside
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],  # leaves
            [[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]],  # passes through
            [[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],  # outside
            [[2.0, -3.0, 0.0], [2.0, 3.0, 0.0]],  # parallel to a side, outside
        ]
    )
    clipped = clip_segments(segments, box)
    assert len(clipped) == 3
    assert clipped[0] == pytest.approx(segments[0])
    assert clipped[1, 1] == pytest.approx([1.0, 0.0, 0.0])
    assert clipped[2] == pytest.approx(np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))


def test_a_space_curve_stays_on_the_curve_and_in_the_box() -> None:
    segments = geometry_of("parametric(cos(u), sin(u), u for u in [0, 6pi])").segments
    assert segments is not None and len(segments) > 100
    points = segments.reshape(-1, 3)
    assert points[:, 2].max() == pytest.approx(5.0, abs=1e-6)  # cut at the top of the box
    assert np.hypot(points[:, 0], points[:, 1]) == pytest.approx(1.0, abs=1e-3)


def test_a_curve_breaks_at_a_jump_but_not_on_a_long_straight_line() -> None:
    def jumping(t: np.ndarray) -> np.ndarray:
        return np.column_stack([t, np.where(t < 0, -3.0, 3.0), np.zeros_like(t)])

    segments = curve_segments(jumping, -1.0, 1.0, BOX)
    assert np.abs(segments[:, 0, 1] - segments[:, 1, 1]).max() < 1e-9

    def fast(t: np.ndarray) -> np.ndarray:
        return np.column_stack([1e6 * t, np.zeros_like(t), np.zeros_like(t)])

    segments = curve_segments(fast, -1.0, 1.0, BOX)
    assert segments[:, :, 0].min() == pytest.approx(-5.0)
    assert segments[:, :, 0].max() == pytest.approx(5.0)


def test_flat_rows_lie_in_the_plane_z_0() -> None:
    plot = plot_of("y = x^2")
    assert isinstance(plot, ExplicitPlot)
    geometry = space_geometry(plot, BOX)
    assert geometry.segments is not None and len(geometry.segments)
    assert (geometry.segments[:, :, 2] == 0).all()
    assert (np.abs(geometry.segments) <= 5.0 + 1e-9).all()
    x, y = geometry.segments[:, 0, 0], geometry.segments[:, 0, 1]
    assert y == pytest.approx(x**2, abs=1e-2)  # chords of the curve


def test_other_flat_kinds_lie_in_the_plane_too() -> None:
    circle = plot_of("x^2 + y^2 = 4")
    assert isinstance(circle, ImplicitPlot)
    segments = space_geometry(circle, BOX).segments
    assert segments is not None and len(segments)
    assert np.hypot(segments[:, 0, 0], segments[:, 0, 1]) == pytest.approx(2.0, abs=0.05)
    point = plot_of("point[1, 2]")
    assert isinstance(point, PointPlot)
    assert space_geometry(point, BOX).points == pytest.approx(np.array([[1.0, 2.0, 0.0]]))


def test_flat_rows_are_gone_when_the_box_leaves_the_plane() -> None:
    plot = plot_of("y = x^2")
    assert isinstance(plot, ExplicitPlot)
    assert space_geometry(plot, Box(-5, 5, -5, 5, 1, 2)).is_empty


def test_an_inequality_draws_its_boundary_without_a_fill() -> None:
    plot = plot_of("y < x")
    assert not isinstance(plot, PlotGroup)
    geometry = space_geometry(plot, BOX)
    assert geometry.segments is not None and len(geometry.segments)
    assert geometry.mesh is None


def test_points_outside_the_box_are_not_drawn() -> None:
    assert geometry_of("point[1, 2, 3]").points is not None
    assert len(geometry_of("point[1, 2, 30]").points) == 0  # type: ignore[arg-type]


def test_space_plots_have_nothing_to_draw_in_the_plane() -> None:
    from opengraph.engine.sampling import ViewRect

    plot = plot_of("z = x")
    assert plot.sample(ViewRect(-1, 1, -1, 1, 100, 100)).is_empty


def test_plots_compare_by_their_expressions() -> None:
    assert plot_of("z = x + 1") == plot_of("z = x + 1")
    assert plot_of("z = x + 1") != plot_of("z = x + 2")
    assert plot_of("x^2 + y^2 + z^2 = 1") == plot_of("x^2 + y^2 + z^2 = 1")


# -- drawing -------------------------------------------------------------------------------


def surface(text: str, color: tuple[int, int, int] = RED) -> Part:
    return Part(geometry_of(text), color)


def color_at(image: np.ndarray, x: int, y: int) -> tuple[int, int, int]:
    return tuple(int(v) for v in image[y, x])  # type: ignore[return-value]


def is_reddish(color: tuple[int, int, int]) -> bool:
    return color[0] > 40 and color[1] == 0 and color[2] == 0


def is_blueish(color: tuple[int, int, int]) -> bool:
    return color[2] > 40 and color[0] == 0 and color[1] == 0


def test_the_camera_shows_z_up_and_the_y_axis_to_the_right() -> None:
    camera = Camera()
    points = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 5.0], [0.0, 5.0, 0.0], [5.0, 0.0, 0.0]])
    x, y, depth = camera.project(points, 800, 600)
    assert x[0] == pytest.approx(400) and y[0] == pytest.approx(300)
    assert y[1] < y[0]  # z goes up the screen
    assert x[2] > x[0]  # y goes right
    assert x[3] < x[0] and y[3] > y[0]  # x comes towards the viewer, on the left
    assert depth[3] > depth[0]


def test_perspective_makes_nearer_things_bigger() -> None:
    camera = Camera(azimuth=0.0, elevation=0.0)  # looking along -x
    points = np.array([[5.0, 2.0, 0.0], [-5.0, 2.0, 0.0]])
    x, _, _ = camera.project(points, 800, 600)
    assert x[0] - 400 > x[1] - 400 > 0


def test_a_frame_has_the_requested_size_and_background() -> None:
    image = render((), Camera(), 120, 90, PLAIN)
    assert image.shape == (90, 120, 3)
    assert (image == 255).all()
    finer = render((), Camera(), 120, 90, PLAIN, sampling=2)
    assert finer.shape == (90, 120, 3)


def test_a_surface_covers_the_pixels_it_projects_to() -> None:
    camera = Camera(elevation=math.radians(40.0))
    image = render((surface("z = 0"),), camera, 200, 150, PLAIN)
    assert is_reddish(color_at(image, 100, 75))
    assert color_at(image, 2, 2) == (255, 255, 255)


def test_the_nearer_surface_wins_whatever_the_order() -> None:
    camera = Camera(elevation=math.radians(80.0))  # from nearly straight above
    low, high = surface("z = -2", RED), surface("z = 2", BLUE)
    for parts in ((low, high), (high, low)):
        image = render(parts, camera, 200, 150, PLAIN)
        assert is_blueish(color_at(image, 100, 75))
    from_below = Camera(elevation=math.radians(-80.0))
    image = render((low, high), from_below, 200, 150, PLAIN)
    assert is_reddish(color_at(image, 100, 75))


def test_intersecting_surfaces_are_resolved_per_pixel() -> None:
    """No order of whole surfaces could draw this; each is on top on its own side."""
    camera = Camera(azimuth=0.0, elevation=math.radians(80.0))
    image = render((surface("z = y", RED), surface("z = -y", BLUE)), camera, 300, 300, PLAIN)
    # Screen right is +y at this azimuth: z = y is the higher one there.
    assert is_reddish(color_at(image, 200, 150))
    assert is_blueish(color_at(image, 100, 150))


def test_a_surface_hides_a_curve_behind_it_and_not_one_in_front() -> None:
    camera = Camera(elevation=math.radians(80.0))
    sheet = surface("z = 0", RED)
    above = surface("parametric(u, 0, 2 for u in [-3, 3])", BLUE)
    below = surface("parametric(u, 0, -2 for u in [-3, 3])", BLUE)

    def blue_pixels(*parts: Part) -> int:
        image = render(parts, camera, 300, 300, PLAIN)
        return int(((image[..., 2] > 100) & (image[..., 0] == 0)).sum())

    assert blue_pixels(sheet, above) > 50
    assert blue_pixels(sheet, below) == 0
    assert blue_pixels(below) > 50  # it is there, only hidden


def test_lighting_shows_both_sides_of_a_surface() -> None:
    above = render((surface("z = 0"),), Camera(elevation=math.radians(50.0)), 100, 100, PLAIN)
    below = render((surface("z = 0"),), Camera(elevation=math.radians(-50.0)), 100, 100, PLAIN)
    assert is_reddish(color_at(above, 50, 50)) and is_reddish(color_at(below, 50, 50))


def test_grid_lines_on_surfaces_stay_thin_on_a_hilltop() -> None:
    """A fixed margin around z = 1 would darken the whole top of this hill."""
    camera = Camera(elevation=math.radians(85.0))
    part = surface("z = 1 - (x^2 + y^2)/100")
    style = Style(show_axes=False)
    lined = render((part,), camera, 400, 400, style).astype(np.int64)
    plain = render((part,), camera, 400, 400, PLAIN).astype(np.int64)
    window = (lined - plain)[170:230, 170:230, 0]
    assert (window < 0).mean() < 0.25


def test_axes_and_walls_can_be_turned_off() -> None:
    plain = render((), Camera(), 200, 200, PLAIN)
    assert (plain == 255).all()
    grid = render((), Camera(), 200, 200, Style(show_axes=False))
    axes = render((), Camera(), 200, 200, Style(show_grid=False))
    assert (grid != 255).any() and (axes != 255).any()
    assert (axes < 100).any()  # the axis colour is dark; the grid's isn't
    assert not (grid < 100).any()


def test_frames_encode_as_images() -> None:
    image = render((), Camera(), 64, 48, Style())
    assert encode(image, still=True).startswith(b"\x89PNG")
    assert encode(image, still=False).startswith(b"\xff\xd8")


def test_a_custom_step_sets_the_grid() -> None:
    wide = render((), Camera(), 300, 300, Style(show_axes=False, steps=(5.0, 5.0, 5.0)))
    dense = render((), Camera(), 300, 300, Style(show_axes=False, steps=(1.0, 1.0, 1.0)))
    assert (dense != 255).sum() > (wide != 255).sum()


# -- the window ----------------------------------------------------------------------------


def show(window: MainWindow, *texts: str) -> list[int]:
    ids = [window.controller.document.items[0].id]
    window.controller.set_text(ids[0], texts[0])
    ids += [window.controller.add_expression(text) for text in texts[1:]]
    window.controller.flush()
    window.controller.recompile()
    return ids


def settle(window: MainWindow) -> None:
    """Wait for the finest frame of the current view."""
    wait_until(lambda: not window.canvas.view3d.loading, 10_000)
    window.canvas.render()


def images(window: MainWindow) -> list[cv.Image]:
    return [shape for shape in window.canvas.render() if isinstance(shape, cv.Image)]


def test_a_flat_document_stays_flat(window: MainWindow) -> None:
    show(window, "y = x^2")
    assert not window.canvas.view3d.active
    wait_until(lambda: not window.canvas.is_rendering)
    assert not images(window)
    assert window.canvas.view3d.frame is None


def test_a_row_using_z_turns_the_graph_into_a_3d_view(window: MainWindow) -> None:
    show(window, "z = x^2 - y^2")
    canvas = window.canvas
    assert canvas.view3d.active
    settle(window)
    shown = images(window)
    assert len(shown) >= 1
    frame = canvas.view3d.frame
    assert frame is not None and frame.still
    assert frame.width == 800 and frame.height == 600
    assert frame.data.startswith(b"\x89PNG")
    assert shown[-1].src == frame.data
    assert (shown[-1].width, shown[-1].height) == (800, 600)


def test_the_labels_name_the_axes(window: MainWindow) -> None:
    show(window, "z = x")
    settle(window)
    texts = [s.value for s in window.canvas.render() if isinstance(s, cv.Text)]
    assert {"x", "y", "z"} <= set(texts)
    assert "0" in texts


def test_axis_numbers_can_be_turned_off(window: MainWindow) -> None:
    from dataclasses import replace

    show(window, "z = x")
    settle(window)
    window.controller.set_settings(replace(window.controller.settings, axis_numbers=False))
    settle(window)
    texts = [s.value for s in window.canvas.render() if isinstance(s, cv.Text)]
    assert sorted(texts) == ["x", "y", "z"]


def test_removing_the_last_space_row_returns_to_the_plane(window: MainWindow) -> None:
    ids = show(window, "z = x", "y = x")
    canvas = window.canvas
    settle(window)
    assert canvas.view3d.active
    window.controller.delete_expression(ids[0])
    assert not canvas.view3d.active
    wait_until(lambda: not canvas.is_rendering)
    assert not images(window)
    assert canvas.render()  # the plane's grid is back


def test_hiding_the_space_row_returns_to_the_plane(window: MainWindow) -> None:
    ids = show(window, "z = x", "y = x")
    window.controller._on_visibility_toggled(ids[0], False)
    assert not window.canvas.view3d.active
    window.controller._on_visibility_toggled(ids[0], True)
    assert window.canvas.view3d.active


def test_unfinished_typing_keeps_the_3d_drawing(window: MainWindow) -> None:
    ids = show(window, "z = x")
    window.controller.set_text(ids[0], "z = x^")
    window.controller.flush()
    window.controller.recompile()
    assert window.canvas.view3d.active


def test_flat_rows_are_drawn_in_the_3d_view(window: MainWindow) -> None:
    show(window, "z = x", "y = x^2")
    settle(window)
    identities = window.canvas.view3d._geometry
    assert len(identities) == 2
    assert any(g.segments is not None and len(g.segments) for _, g in identities.values())


def test_an_edit_elsewhere_keeps_the_meshes(window: MainWindow) -> None:
    ids = show(window, "z = x", "y = 1")
    settle(window)
    view = window.canvas.view3d
    before = {key: done[1] for key, done in view._geometry.items()}
    surface_key = id(window.controller.results[ids[0]].plot)
    window.controller.set_text(ids[1], "y = 2")
    window.controller.flush()
    window.controller.recompile()
    settle(window)
    assert view._geometry[surface_key][1] is before[surface_key]


def test_recolouring_draws_a_new_frame(window: MainWindow) -> None:
    ids = show(window, "z = x")
    settle(window)
    first = window.canvas.view3d.frame
    assert first is not None
    window.controller._on_color_chosen(ids[0], "#00ff00")
    settle(window)
    second = window.canvas.view3d.frame
    assert second is not None and second is not first and second.data != first.data


def test_the_theme_recolours_the_background(window: MainWindow) -> None:
    from opengraph.ui import theme

    show(window, "z = x")
    settle(window)
    first = window.canvas.view3d.frame
    window.controller.set_theme(theme.DARK)
    settle(window)
    second = window.canvas.view3d.frame
    assert first is not None and second is not None and first.data != second.data


def test_dragging_turns_the_view(window: MainWindow) -> None:
    show(window, "z = x")
    canvas = window.canvas
    settle(window)
    camera = canvas.view3d.camera
    canvas.press(Point(400, 300))
    canvas.move(Point(460, 340))
    canvas.release(Point(460, 340))
    moved = canvas.view3d.camera
    assert moved.azimuth < camera.azimuth  # the front of the scene follows the pointer
    assert moved.elevation > camera.elevation
    assert moved.box == camera.box  # turning needs no new meshes
    assert window.controller.modified
    settle(window)
    assert canvas.view3d.frame is not None and canvas.view3d.frame.camera == moved


def test_elevation_stops_at_the_top(window: MainWindow) -> None:
    show(window, "z = x")
    canvas = window.canvas
    canvas.press(Point(0, 0))
    canvas.move(Point(0, 5000))
    canvas.release(Point(0, 5000))
    assert canvas.view3d.camera.elevation == pytest.approx(math.radians(89.0))


def test_a_drag_shows_frames_along_the_way(window: MainWindow) -> None:
    """A stream of moves faster than a frame must not cancel every frame."""
    show(window, "x^2 + y^2 + z^2 = 9")
    canvas = window.canvas
    settle(window)
    canvas.press(Point(400, 300))
    first = canvas.view3d.frame
    seen = set()
    for step in range(1, 120):
        canvas.move(Point(400 + step, 300))
        wait(3)
        frame = canvas.view3d.frame
        assert frame is not None
        seen.add(id(frame))
    # Rotated frames arrived while the pointer was still moving.
    assert canvas.view3d.frame is not first
    assert len(seen) >= 2
    assert canvas.view3d.frame is not None and not canvas.view3d.frame.still
    canvas.release(Point(520, 300))
    settle(window)
    assert canvas.view3d.frame.still


def test_no_fine_frame_during_a_pause_in_a_drag(window: MainWindow) -> None:
    show(window, "z = x")
    canvas = window.canvas
    settle(window)
    canvas.press(Point(400, 300))
    canvas.move(Point(420, 300))
    wait(500)
    assert canvas.view3d.frame is not None and not canvas.view3d.frame.still
    canvas.release(Point(420, 300))
    settle(window)
    assert canvas.view3d.frame.still


def test_shift_drag_pans(window: MainWindow) -> None:
    from opengraph.ui import runtime

    show(window, "z = x")
    canvas = window.canvas
    camera = canvas.view3d.camera
    runtime.modifiers.shift = True
    canvas.press(Point(400, 300))
    canvas.move(Point(460, 300))
    canvas.release(Point(460, 300))
    box = canvas.view3d.camera.box
    assert canvas.view3d.camera.azimuth == camera.azimuth
    assert box != camera.box
    assert box.x_max - box.x_min == pytest.approx(10.0)  # moved, not zoomed
    # The origin follows the pointer to the right.
    x, _, _ = canvas.view3d.camera.project(np.zeros((1, 3)), 800, 600)
    assert x[0] == pytest.approx(460.0, abs=8.0)


def test_panning_samples_the_new_box(window: MainWindow) -> None:
    from opengraph.ui import runtime

    show(window, "z = x")
    canvas = window.canvas
    settle(window)
    runtime.modifiers.shift = True
    canvas.press(Point(400, 300))
    canvas.move(Point(460, 300))
    canvas.release(Point(460, 300))
    settle(window)
    box = canvas.view3d.camera.box
    assert all(done[0] == box for done in canvas.view3d._geometry.values())


@pytest.mark.parametrize(
    ("ctrl", "shift", "changed"),
    [(False, False, "xyz"), (True, False, "xy"), (False, True, "z")],
)
def test_the_wheel_zooms(window: MainWindow, ctrl: bool, shift: bool, changed: str) -> None:
    show(window, "z = x")
    canvas = window.canvas
    canvas.wheel(Point(400, 300), 120, ctrl=ctrl, shift=shift)
    box = canvas.view3d.camera.box
    for index, axis in enumerate("xyz"):
        lo, hi = box.range(index)
        assert (hi - lo < 10.0) == (axis in changed)
        assert (lo + hi) / 2 == pytest.approx(0.0)  # about the centre of the box


def test_a_locked_aspect_zooms_every_axis(window: MainWindow) -> None:
    from dataclasses import replace

    show(window, "z = x")
    canvas = window.canvas
    window.controller.set_settings(replace(window.controller.settings, lock_aspect=True))
    canvas.wheel(Point(400, 300), 120, shift=True)
    box = canvas.view3d.camera.box
    assert box.x_max < 5.0 and box.z_max < 5.0


def test_zooming_has_limits(window: MainWindow) -> None:
    show(window, "z = x")
    canvas = window.canvas
    for _ in range(60):
        canvas.zoom(10.0)
    box = canvas.view3d.camera.box
    assert box.x_max - box.x_min == pytest.approx(2e-6)
    for _ in range(60):
        canvas.zoom(0.1)
    box = canvas.view3d.camera.box
    assert box.x_max - box.x_min == pytest.approx(2e9)


def test_double_click_zooms(window: MainWindow) -> None:
    show(window, "z = x")
    canvas = window.canvas
    canvas.double_click(Point(400, 300), shift=False)
    assert canvas.view3d.camera.box.x_max == pytest.approx(2.5)
    canvas.double_click(Point(400, 300), shift=True)
    assert canvas.view3d.camera.box.x_max == pytest.approx(5.0)


def test_the_keyboard_turns_zooms_and_resets(window: MainWindow) -> None:
    show(window, "z = x")
    canvas = window.canvas
    start = canvas.view3d.camera
    assert canvas.handle_key("Arrow Left")
    assert canvas.view3d.camera.azimuth < start.azimuth
    assert canvas.handle_key("Arrow Up")
    assert canvas.view3d.camera.elevation > start.elevation
    assert canvas.handle_key("+")
    assert canvas.view3d.camera.box.x_max < 5.0
    assert not canvas.handle_key("q")
    assert canvas.handle_key("Home")
    assert canvas.view3d.camera == start


def test_menu_actions_act_on_the_3d_view(window: MainWindow) -> None:
    show(window, "z = x")
    canvas = window.canvas
    viewport_state = canvas.viewport.state()
    canvas.zoom(2.0)
    assert canvas.view3d.camera.box.x_max == pytest.approx(2.5)
    assert canvas.viewport.state() == viewport_state  # the plane's view is left alone
    canvas.reset_view()
    assert canvas.view3d.camera.box.x_max == pytest.approx(5.0)


def test_the_plane_keeps_its_own_view_across_a_3d_view(window: MainWindow) -> None:
    ids = show(window, "y = x")
    canvas = window.canvas
    canvas.zoom(2.0)
    state = canvas.viewport.state()
    window.controller.set_text(ids[0], "z = x")
    window.controller.flush()
    window.controller.recompile()
    canvas.zoom(3.0)
    window.controller.set_text(ids[0], "y = x")
    window.controller.flush()
    window.controller.recompile()
    assert not canvas.view3d.active
    assert canvas.viewport.state() == state


def test_hovering_reports_the_point_nearest_the_viewer(window: MainWindow) -> None:
    show(window, "z = 0", "z = 2")
    canvas = window.canvas
    settle(window)
    canvas.move(Point(400, 300))
    trace = canvas._trace
    assert trace is not None
    assert trace.label.startswith("(") and trace.label.endswith(", 2)")  # the upper sheet
    assert canvas.format_position(0, 0) == trace.label
    assert any(isinstance(shape, cv.Rect) for shape in canvas.render())  # its tooltip


def test_hovering_empty_space_reports_nothing(window: MainWindow) -> None:
    show(window, "point[1, 2, 3]")
    canvas = window.canvas
    settle(window)
    canvas.move(Point(5, 5))
    assert canvas._trace is None
    assert canvas.format_position(0, 0) == ""


def test_a_point_in_space_can_be_inspected(window: MainWindow) -> None:
    show(window, "point[1, 2, 3]")
    canvas = window.canvas
    settle(window)
    frame = canvas.view3d.frame
    assert frame is not None
    x, y, _ = frame.camera.project(np.array([[1.0, 2.0, 3.0]]), frame.width, frame.height)
    canvas.move(Point(float(x[0]), float(y[0])))
    assert canvas._trace is not None and canvas._trace.label == "(1, 2, 3)"


def test_points_of_interest_are_for_the_plane(window: MainWindow) -> None:
    ids = show(window, "z = x", "y = x^2 - 1")
    canvas = window.canvas
    canvas.set_focus(ids[1])
    assert canvas.markers() == []


def test_flat_points_do_not_drag_in_a_3d_view(window: MainWindow) -> None:
    show(window, "z = x", "point[1, 2]")
    canvas = window.canvas
    settle(window)
    assert canvas._draggable_at(Point(400, 300)) is None


def test_the_view_is_saved_with_the_document(window: MainWindow, tmp_path: Path) -> None:
    show(window, "z = x")
    canvas = window.canvas
    canvas.view3d.turn(40, 20)
    canvas.view3d.zoom(2.0)
    canvas.view3d.set_range("z", -1.0, 3.0)
    camera = canvas.view3d.camera
    path = tmp_path / "view.ograph.json"
    window.controller.save_document(path)

    canvas.view3d.reset()
    window.controller.new_document()
    assert canvas.view3d.camera == Camera()
    window.controller.open_document(path)
    assert canvas.view3d.active
    assert canvas.view3d.camera == camera


def test_old_documents_open_with_the_default_view(window: MainWindow, tmp_path: Path) -> None:
    path = tmp_path / "old.ograph.json"
    document = Document()
    document.add("z = x")
    document.view = {"center_x": 1.0, "center_y": 2.0, "scale_x": 50.0, "scale_y": 50.0}
    document.save(path)
    window.canvas.view3d.zoom(2.0)
    window.controller.open_document(path)
    assert window.canvas.view3d.camera == Camera()


@pytest.mark.parametrize(
    "state",
    [
        {"view3d_azimuth": 1.0},  # partial
        {
            "view3d_azimuth": 1.0,
            "view3d_elevation": 1.0,
            "view3d_x_min": 2.0,  # empty range
            "view3d_x_max": 2.0,
            "view3d_y_min": -1.0,
            "view3d_y_max": 1.0,
            "view3d_z_min": -1.0,
            "view3d_z_max": 1.0,
        },
        {
            "view3d_azimuth": math.nan,
            "view3d_elevation": 1.0,
            "view3d_x_min": -1.0,
            "view3d_x_max": 1.0,
            "view3d_y_min": -1.0,
            "view3d_y_max": 1.0,
            "view3d_z_min": -1.0,
            "view3d_z_max": 1.0,
        },
    ],
)
def test_unusable_saved_views_are_ignored(window: MainWindow, state: dict[str, float]) -> None:
    window.canvas.view3d.restore(state)
    assert window.canvas.view3d.camera == Camera()


def test_the_settings_dialog_edits_all_three_ranges(window: MainWindow) -> None:
    dialog = window.settings_dialog
    window.show_graph_settings()
    assert not dialog.z_row.visible
    show(window, "z = x")
    window._update_dialog_ranges()
    assert dialog.z_row.visible
    assert dialog.bounds["z"][1].value == "5"
    dialog.edit_bounds("z", "-1", "9")
    assert window.canvas.view3d.axis_range("z") == (-1.0, 9.0)
    dialog.edit_bounds("x", "0", "2")
    assert window.canvas.view3d.axis_range("x") == (0.0, 2.0)
    dialog.edit_bounds("y", "3", "1")  # backwards: refused, with an explanation
    assert window.canvas.view3d.axis_range("y") == (-5.0, 5.0)
    assert "minimum must be less" in window.current_message()


def test_export_waits_for_the_finest_frame(window: MainWindow, tmp_path: Path) -> None:
    show(window, "x^2 + y^2 + z^2 = 9")
    canvas = window.canvas
    assert canvas.is_rendering
    finished: list[bool] = []
    canvas.rendering_finished.connect(lambda: finished.append(True))
    wait_until(lambda: finished, 10_000)
    assert not canvas.is_rendering
    frame = canvas.view3d.frame
    assert frame is not None and frame.still


def test_a_failing_plot_draws_nothing_and_the_others_still_draw(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    good, bad = show(window, "z = x", "z = y")
    plot = window.controller.results[bad].plot
    original = SurfacePlot.sample_space

    def failing(self: SurfacePlot, box: Box) -> SpaceGeometry:
        if self is plot:
            raise ValueError("broken")
        return original(self, box)

    monkeypatch.setattr(SurfacePlot, "sample_space", failing)
    window.canvas.view3d.zoom(2.0)
    settle(window)
    geometry = window.canvas.view3d._geometry
    assert geometry[id(plot)][1].is_empty
    assert not geometry[id(window.controller.results[good].plot)][1].is_empty
    assert "Sampling a surface failed" in caplog.text


def test_groups_sample_each_member(window: MainWindow) -> None:
    ids = show(window, "z = list[1, 2] x")
    settle(window)
    plot = window.controller.results[ids[0]].plot
    assert plot is not None
    parts = list(members(plot))
    assert len(parts) == 2
    assert all(id(part) in window.canvas.view3d._geometry for part in parts)


def test_the_reference_documents_3d() -> None:
    from opengraph.ui.main_window import syntax_reference

    text = syntax_reference()
    assert "z = x^2 - y^2" in text
    assert "point[1, 2, 3]" in text
    assert "Shift+drag" in text


def test_typing_a_surface_end_to_end(window: MainWindow) -> None:
    edit = window.panel.row(window.controller.document.items[0].id).edit
    edit.focus()
    for char in "z = x y":
        edit.type_text(char)
    key(edit, "Enter")
    wait_until(lambda: window.canvas.view3d.active)
    settle(window)
    assert images(window)


# -- finishing what is running --------------------------------------------------------------


def _collect(task: LatestTask) -> list[object]:
    results: list[object] = []
    task.ready.connect(lambda _key, value: results.append(value))
    return results


def test_latest_task_replaces_a_running_job_by_default() -> None:
    task = LatestTask()
    results = _collect(task)
    task.request("a", lambda: "a")
    task.request("b", lambda: "b")
    assert events.wait_until(lambda: results, 5.0)
    wait(50)
    assert results == ["b"]
    task.stop()


def test_latest_task_can_let_a_running_job_finish() -> None:
    task = LatestTask()
    results = _collect(task)
    task.request("a", lambda: "a")
    task.request("b", lambda: "b", finish_active=True)
    task.request("c", lambda: "c", finish_active=True)  # replaces only the waiting one
    assert events.wait_until(lambda: len(results) == 2, 5.0)
    wait(50)
    assert results == ["a", "c"]
    task.stop()
