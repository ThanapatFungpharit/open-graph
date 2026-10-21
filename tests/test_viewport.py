from __future__ import annotations

import numpy as np
import pytest

from opengraph.graph.ticks import choose_spacing, format_coordinate, format_tick, tick_indices
from opengraph.graph.viewport import MAX_DEEP_SCALE, MIN_SCALE, Viewport


def make() -> Viewport:
    return Viewport(width=800, height=600, center_x=1.5, center_y=-2.0, scale_x=40, scale_y=40)


def test_center_maps_to_middle_of_widget() -> None:
    vp = make()
    assert vp.to_screen(1.5, -2.0) == (400, 300)


def test_y_axis_points_up_on_screen() -> None:
    vp = make()
    assert vp.to_screen_y(0.0) < vp.to_screen_y(-1.0)


def test_round_trip_scalars_and_arrays() -> None:
    vp = make()
    xs = np.linspace(-100, 100, 7)
    ys = np.linspace(-50, 80, 7)
    wx, wy = vp.to_world(*vp.to_screen(xs, ys))
    np.testing.assert_allclose(wx, xs)
    np.testing.assert_allclose(wy, ys)
    assert vp.to_world(*vp.to_screen(3.25, -7.0)) == pytest.approx((3.25, -7.0))


def test_visible_bounds() -> None:
    vp = make()
    assert vp.x_min == pytest.approx(1.5 - 10) and vp.x_max == pytest.approx(1.5 + 10)
    assert vp.y_min == pytest.approx(-2 - 7.5) and vp.y_max == pytest.approx(-2 + 7.5)
    rect = vp.view_rect()
    assert rect.px_per_x == pytest.approx(40) and rect.px_per_y == pytest.approx(40)


def test_pan_moves_content_with_the_mouse() -> None:
    vp = make()
    before = vp.to_world(100.0, 100.0)
    vp.pan_pixels(30, -20)
    # The world point that was under (100, 100) is now under (130, 80).
    assert vp.to_world(130.0, 80.0) == pytest.approx(before)


@pytest.mark.parametrize("factor", [2.0, 0.5, 1.2, 1 / 1.2])
@pytest.mark.parametrize("anchor", [(0.0, 0.0), (400.0, 300.0), (731.0, 59.0)])
def test_zoom_keeps_point_under_cursor_fixed(factor: float, anchor: tuple[float, float]) -> None:
    vp = make()
    world = vp.to_world(*anchor)
    vp.zoom_at(*anchor, factor)
    assert vp.scale_x == pytest.approx(40 * factor)
    assert vp.to_screen(*world) == pytest.approx(anchor)


def test_repeated_zoom_round_trip() -> None:
    vp = make()
    for _ in range(25):
        vp.zoom_at(123.0, 456.0, 1.3)
    for _ in range(25):
        vp.zoom_at(123.0, 456.0, 1 / 1.3)
    assert vp.scale_x == pytest.approx(40)
    assert vp.center_x == pytest.approx(1.5) and vp.center_y == pytest.approx(-2.0)


def test_zoom_is_clamped_without_distorting_aspect() -> None:
    vp = make()
    vp.scale_y = 80
    for _ in range(200):
        vp.zoom_center(10)
    # Linear axes zoom on far past float64 (with exact anchors), then stop.
    assert vp.scale_y == pytest.approx(MAX_DEEP_SCALE)
    assert vp.scale_y / vp.scale_x == pytest.approx(2)
    for _ in range(400):
        vp.zoom_center(0.1)
    assert vp.scale_x >= MIN_SCALE and vp.scale_y / vp.scale_x == pytest.approx(2)


def test_resize_keeps_center() -> None:
    vp = make()
    vp.resize(1000, 200)
    assert vp.to_screen(1.5, -2.0) == (500, 100)
    vp.resize(0, -5)
    assert vp.width == 1 and vp.height == 1


def test_state_round_trip_and_rejects_garbage() -> None:
    vp = make()
    other = Viewport()
    other.restore(vp.state())
    assert other.state() == vp.state()
    other.restore({"center_x": float("nan"), "scale_x": 5})
    assert other.state() == vp.state()


def test_fit() -> None:
    vp = make()
    vp.fit(-1, 3, 0, 2)
    assert vp.x_min <= -1 and vp.x_max >= 3 and vp.y_min <= 0 and vp.y_max >= 2
    assert vp.scale_x == vp.scale_y


@pytest.mark.parametrize("scale", [1e-6, 0.37, 1, 40, 55, 1234.5, 1e9])
def test_major_spacing_is_nice_and_readable(scale: float) -> None:
    spacing = choose_spacing(scale, min_px=90)
    assert spacing.major * scale >= 90 * (1 - 1e-9)
    assert spacing.major * scale < 90 * 5.01  # never absurdly sparse
    mantissa = spacing.major / 10 ** np.floor(np.log10(spacing.major) + 1e-9)
    assert round(mantissa, 6) in (1, 2, 5)
    assert spacing.major / spacing.minor in (4, 5)


def test_tick_indices_are_integer_multiples() -> None:
    np.testing.assert_array_equal(tick_indices(-1.1, 2.3, 0.5), [-2, -1, 0, 1, 2, 3, 4])
    assert tick_indices(0, 1e9, 1e-9).size == 0  # refuses absurd counts


@pytest.mark.parametrize(
    ("value", "step", "expected"),
    [
        (0.0, 1, "0"),
        (3 * 0.1, 0.1, "0.3"),  # no 0.30000000000000004
        (-4, 2, "−4"),
        (2.5, 0.5, "2.5"),
        (2e6, 1e6, "2×10⁶"),
        (3e-6, 1e-6, "3×10⁻⁶"),
        (1.5e7, 5e6, "1.5×10⁷"),
    ],
)
def test_tick_labels(value: float, step: float, expected: str) -> None:
    assert format_tick(value, step) == expected


def test_coordinate_formatting_matches_pixel_precision() -> None:
    assert format_coordinate(1.23456789, 0.02) == "1.235"
    assert format_coordinate(-0.00001, 0.02) == "0"
    assert format_coordinate(2.5, 0) == "2.5"
