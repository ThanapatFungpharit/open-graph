"""Points, rectangles and stroke paths in screen units (y pointing down)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Point:
    x: float
    y: float

    def __add__(self, other: Point) -> Point:
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Point) -> Point:
        return Point(self.x - other.x, self.y - other.y)


@dataclass(frozen=True, slots=True)
class Rect:
    x: float
    y: float
    width: float
    height: float

    @staticmethod
    def from_points(a: Point, b: Point) -> Rect:
        return Rect(min(a.x, b.x), min(a.y, b.y), abs(b.x - a.x), abs(b.y - a.y))

    @property
    def left(self) -> float:
        return self.x

    @property
    def top(self) -> float:
        return self.y

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.height

    @property
    def center(self) -> Point:
        return Point(self.x + self.width / 2, self.y + self.height / 2)

    def contains(self, point: Point) -> bool:
        return self.left <= point.x <= self.right and self.top <= point.y <= self.bottom

    def translated(self, dx: float, dy: float) -> Rect:
        return Rect(self.x + dx, self.y + dy, self.width, self.height)

    def adjusted(self, left: float, top: float, right: float, bottom: float) -> Rect:
        """Edges moved by the given amounts (as Qt's ``QRectF.adjusted``)."""
        return Rect(
            self.x + left, self.y + top, self.width - left + right, self.height - top + bottom
        )

    def united(self, other: Rect) -> Rect:
        left, top = min(self.left, other.left), min(self.top, other.top)
        right, bottom = max(self.right, other.right), max(self.bottom, other.bottom)
        return Rect(left, top, right - left, bottom - top)


@dataclass(slots=True)
class Path:
    """A stroke: ``("M", x, y)``, ``("L", x, y)`` and ``("Q", cx, cy, x, y)`` steps."""

    steps: list[tuple[str, float, float] | tuple[str, float, float, float, float]] = field(
        default_factory=list
    )

    @staticmethod
    def start(point: Point) -> Path:
        return Path([("M", point.x, point.y)])

    def line_to(self, point: Point) -> None:
        self.steps.append(("L", point.x, point.y))

    def quad_to(self, control: Point, end: Point) -> None:
        self.steps.append(("Q", control.x, control.y, end.x, end.y))

    def translated(self, dx: float, dy: float) -> Path:
        moved: list[tuple[str, float, float] | tuple[str, float, float, float, float]] = []
        for step in self.steps:
            if len(step) == 3:
                moved.append((step[0], step[1] + dx, step[2] + dy))
            else:
                moved.append((step[0], step[1] + dx, step[2] + dy, step[3] + dx, step[4] + dy))
        return Path(moved)
