"""Application themes: Light, Dark, and System (follow the OS appearance).

A :class:`Theme` is the single source of colours for the whole interface: the Flet
Material theme (menus, dialogs, inputs), the expression panel, the math editor's ink,
the canvas (:class:`~opengraph.graph.canvas.CanvasColors`) and feedback text.

Expression colours stay as the user chose them in the document; for display they are
passed through :meth:`Theme.curve_color`, which lightens (dark theme) or darkens
(light theme) a colour until it has at least :data:`MIN_CONTRAST` against the canvas,
so every palette entry and custom colour stays visible in both themes.
"""

from __future__ import annotations

from dataclasses import dataclass

import flet as ft

from opengraph import colors
from opengraph.graph.canvas import LIGHT_COLORS, CanvasColors
from opengraph.ui import fonts

MODES = ("light", "dark", "system")
MIN_CONTRAST = 3.0  # WCAG minimum for graphical objects


@dataclass(frozen=True, slots=True)
class Theme:
    name: str
    dark: bool
    window: str
    panel: str
    row: str
    row_border: str
    row_current: str
    row_error: str
    text: str
    muted: str
    placeholder: str
    accent: str
    accent_soft: str
    hover: str
    error: str
    info: str
    base: str
    track: str
    canvas: CanvasColors

    def material(self) -> ft.Theme:
        """The Flet theme for menus, dialogs, buttons and text fields."""
        scheme = ft.ColorScheme(
            primary=self.accent,
            on_primary="#ffffff",
            secondary=self.accent,
            surface=self.base,
            on_surface=self.text,
            on_surface_variant=self.muted,
            surface_container=self.panel,
            surface_container_high=self.row,
            surface_container_highest=self.hover,
            outline=self.row_border,
            outline_variant=self.row_border,
            error=self.error,
        )
        return ft.Theme(
            color_scheme=scheme,
            font_family=fonts.SANS,
            visual_density=ft.VisualDensity.COMPACT,
            use_material3=True,
        )

    def curve_color(self, color: str) -> str:
        """``color`` adjusted until it contrasts with the canvas (see module doc)."""
        result = color
        background = self.canvas.background
        hue, saturation, lightness, alpha = colors.hsl(color)
        for _ in range(40):
            if colors.contrast(result, background) >= MIN_CONTRAST:
                break
            lightness = min(1.0, lightness + 0.025) if self.dark else max(0.0, lightness - 0.025)
            result = colors.from_hsl(hue, saturation, lightness, alpha)
        return result


contrast = colors.contrast


LIGHT = Theme(
    name="light",
    dark=False,
    window="#f3f4f6",
    panel="#f7f8fa",
    row="#ffffff",
    row_border="#e5e7eb",
    row_current="#f6f8ff",
    row_error="#fff8f8",
    text="#1f2328",
    muted="#6b7280",
    placeholder="#9aa0a6",
    accent="#2563eb",
    accent_soft="#eef3ff",
    hover="#eef0f3",
    error="#c62828",
    info="#4a4f57",
    base="#ffffff",
    track="#e5e7eb",
    canvas=LIGHT_COLORS,
)

DARK = Theme(
    name="dark",
    dark=True,
    window="#1c1f24",
    panel="#202329",
    row="#25282e",
    row_border="#363a42",
    row_current="#2a3140",
    row_error="#3a2526",
    text="#e6e8eb",
    muted="#9aa1ab",
    placeholder="#6f7680",
    accent="#5b8def",
    accent_soft="#2c3650",
    hover="#30343b",
    error="#ff7b72",
    info="#b9bec6",
    base="#1a1d22",
    track="#3a3f47",
    canvas=CanvasColors(
        "#16181c",
        "#202329",
        "#33373f",
        "#c9cdd3",
        "#aab0b8",
        colors.from_rgba(37, 40, 46, 240),
        "#e6e8eb",
    ),
)

_current = LIGHT
_system_dark = False


def system_is_dark() -> bool:
    """Whether the OS prefers a dark appearance (False when unknown)."""
    return _system_dark


def set_system_dark(dark: bool) -> None:
    """Record the OS appearance (the application reports the page's brightness)."""
    global _system_dark
    _system_dark = dark


def resolve(mode: str) -> Theme:
    """The theme for a mode name; ``system`` follows the OS, unknown names are light."""
    if mode == "dark" or (mode == "system" and system_is_dark()):
        return DARK
    return LIGHT


def current() -> Theme:
    """The theme in use (controls read colours from here when they are rebuilt)."""
    return _current


def set_current(theme: Theme) -> None:
    global _current
    _current = theme
