"""The graph document: an ordered list of expression items plus view state.

Pure Python (no interface toolkit) so it can be tested and serialised independently of
the UI.
Colours are stored as ``#rrggbb`` strings.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TypeGuard

from opengraph.engine.formatting import format_value
from opengraph.graph.settings import GraphSettings
from opengraph.json_types import JSONObject, is_string_keyed_dict

FORMAT = "opengraph"
VERSION = 2  # newest format this build reads
# Documents without tables are still written as version 1, so older builds open them;
# tables need version 2 (an older build would silently drop them).
TABLE_VERSION = 2

# Distinct, colour-blind-aware hues that read well on a white background.
PALETTE = (
    "#2563eb",  # blue
    "#dc2626",  # red
    "#16a34a",  # green
    "#9333ea",  # purple
    "#ea580c",  # orange
    "#0891b2",  # teal
    "#db2777",  # pink
    "#475569",  # slate
)

_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")


def _strings(value: object) -> TypeGuard[list[str]]:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _string_columns(value: object) -> TypeGuard[list[list[str]]]:
    return isinstance(value, list) and all(_strings(column) for column in value)


def _optional_string(data: dict[str, object], key: str) -> str:
    value = data.get(key)
    return value.strip() if isinstance(value, str) else ""


class DocumentError(Exception):
    """A file could not be read as a graph document."""


MAX_SLIDER_STEPS = 100_000
SLIDER_LABELS = ("Slider minimum", "Slider maximum", "Slider step")


@dataclass(frozen=True, slots=True)
class SliderRange:
    """Bounds and step of the slider shown for a ``k = <number>`` row.

    Slider positions are integers ``0..steps``; position ``i`` means ``min + i*step``,
    clamped to ``max`` (so ``max`` is reachable even when the span isn't a whole number
    of steps).

    A bound or the step may be an expression (``2π``, ``a + 1``): its text is kept in
    ``min_expr``/``max_expr``/``step_expr`` ("" for a plain number) and the number
    beside it is its latest value, which the controller refreshes on every recompile.
    """

    min: float = -10.0
    max: float = 10.0
    step: float = 0.1
    min_expr: str = ""
    max_expr: str = ""
    step_expr: str = ""

    @staticmethod
    def validated(lo: float, hi: float, step: float) -> SliderRange:
        """Build a range from user input; raises ValueError with a readable message."""
        if not all(math.isfinite(v) for v in (lo, hi, step)):
            raise ValueError("Slider bounds and step must be finite numbers")
        if lo >= hi:
            raise ValueError("Slider minimum must be less than its maximum")
        if step <= 0:
            raise ValueError("Slider step must be positive")
        if (hi - lo) / step > MAX_SLIDER_STEPS:
            raise ValueError("Slider step is too small for this range")
        return SliderRange(lo, hi, step)

    @staticmethod
    def from_texts(texts: Sequence[str], values: Sequence[float | str]) -> SliderRange:
        """Build a range from the texts typed for min, max and step and their values.

        A value is a number, or the message explaining why its text has none. Texts
        that aren't plain numbers are kept as expressions. Raises ValueError with a
        readable message.
        """
        for label, value in zip(SLIDER_LABELS, values, strict=True):
            if isinstance(value, str):
                raise ValueError(f"{label}: {value}")
        lo, hi, step = (float(v) for v in values)
        lo_expr, hi_expr, step_expr = ("" if _plain_number(t) else t.strip() for t in texts)
        return replace(
            SliderRange.validated(lo, hi, step),
            min_expr=lo_expr,
            max_expr=hi_expr,
            step_expr=step_expr,
        )

    @property
    def exprs(self) -> tuple[str, str, str]:
        return (self.min_expr, self.max_expr, self.step_expr)

    @property
    def texts(self) -> tuple[str, str, str]:
        """What to show for min, max and step: the expression, or the number."""
        return (
            self.min_expr or format_value(self.min),
            self.max_expr or format_value(self.max),
            self.step_expr or format_value(self.step),
        )

    def with_values(self, values: Sequence[float | str]) -> SliderRange:
        """This range with its expressions re-evaluated to ``values`` (see from_texts)."""
        return SliderRange.from_texts(self.texts, values)

    @staticmethod
    def default_for(value: float) -> SliderRange:
        """[-10, 10], or a window of the same width around a value outside it."""
        if -10 <= value <= 10:
            return SliderRange()
        if abs(value) < 1e9:
            return SliderRange(value - 10, value + 10, 0.1)
        width = abs(value)  # +-10 would vanish in float rounding
        return SliderRange(value - width, value + width, width / 100)

    def including(self, value: float) -> SliderRange:
        """This range, with the nearer bound moved out to ``value`` if it lies outside."""
        if self.min <= value <= self.max:
            return self
        if (self.min_expr and value < self.min) or (self.max_expr and value > self.max):
            return self  # an expression bound isn't overwritten; the handle sits at the end
        lo, hi = min(self.min, value), max(self.max, value)
        if (hi - lo) / self.step > MAX_SLIDER_STEPS:
            return self if any(self.exprs) else SliderRange.default_for(value)
        return replace(self, min=lo, max=hi)

    @property
    def steps(self) -> int:
        return max(1, math.ceil((self.max - self.min) / self.step - 1e-9))

    def position_of(self, value: float) -> int:
        return max(0, min(self.steps, round((value - self.min) / self.step)))

    def value_at(self, position: int) -> float:
        """The value at a slider position, rounded to the precision of min and step."""
        value = min(self.max, self.min + position * self.step)
        return round(value, max(_decimals(self.min), _decimals(self.step)))

    def to_dict(self) -> JSONObject:
        data: JSONObject = {"min": self.min, "max": self.max, "step": self.step}
        for key, expr in zip(_EXPR_FIELDS, self.exprs, strict=True):
            if expr:  # optional: older readers use the numbers
                data[key] = expr
        return data

    @staticmethod
    def from_dict(data: object) -> SliderRange | None:
        """Parse a stored range; anything malformed is dropped (a default is used)."""
        if not is_string_keyed_dict(data):
            return None
        values = []
        for key in ("min", "max", "step"):
            value = data.get(key)
            if isinstance(value, bool) or not isinstance(value, int | float):
                return None
            values.append(float(value))
        try:
            slider_range = SliderRange.validated(*values)
        except ValueError:
            return None
        lo_expr, hi_expr, step_expr = (_optional_string(data, key) for key in _EXPR_FIELDS)
        return replace(slider_range, min_expr=lo_expr, max_expr=hi_expr, step_expr=step_expr)


_EXPR_FIELDS = ("min_expr", "max_expr", "step_expr")


def _plain_number(text: str) -> bool:
    try:
        float(text.strip().replace("−", "-"))
    except ValueError:
        return False
    return True


def _decimals(value: float) -> int:
    """Digits after the decimal point in the compact form of ``value`` (``0.25`` -> 2)."""
    text = format_value(value)
    if "e" in text:
        mantissa, exponent = text.split("e")
        frac = len(mantissa.partition(".")[2])
        return max(0, frac - int(exponent))
    return len(text.partition(".")[2])


TABLE_STYLES = ("points", "lines", "both")
MAX_TABLE_ROWS = 10_000


@dataclass(slots=True)
class TableData:
    """A data table: named columns of cell texts. The first column is the x values.

    ``style`` is how the data is drawn: ``points``, ``lines`` (connected) or ``both``.
    """

    names: list[str] = field(default_factory=lambda: ["x_1", "y_1"])
    columns: list[list[str]] = field(default_factory=lambda: [[], []])
    style: str = "points"

    @property
    def row_count(self) -> int:
        return max((len(c) for c in self.columns), default=0)

    def cell(self, row: int, column: int) -> str:
        cells = self.columns[column]
        return cells[row] if row < len(cells) else ""

    def set_cell(self, row: int, column: int, text: str) -> None:
        cells = self.columns[column]
        while len(cells) <= row:
            cells.append("")
        cells[row] = text
        for cells in self.columns:  # drop trailing empty rows
            while cells and not cells[-1].strip():
                cells.pop()

    def to_dict(self) -> JSONObject:
        return {
            "names": list(self.names),
            "columns": [list(c) for c in self.columns],
            "style": self.style,
        }

    @staticmethod
    def from_dict(data: object) -> TableData:
        if not is_string_keyed_dict(data):
            raise DocumentError("Malformed table")
        names, columns = data.get("names"), data.get("columns")
        if not (_strings(names) and _string_columns(columns) and len(names) == len(columns) >= 2):
            raise DocumentError("Malformed table")
        style = data.get("style")
        return TableData(
            list(names),
            [list(c[:MAX_TABLE_ROWS]) for c in columns],
            style if style in TABLE_STYLES else "points",
        )


@dataclass(slots=True)
class ExpressionItem:
    id: int
    text: str = ""
    color: str = PALETTE[0]
    visible: bool = True
    slider: SliderRange | None = None  # set once the row shows a slider
    table: TableData | None = None  # a table row instead of an expression row


class Document:
    def __init__(self) -> None:
        self.items: list[ExpressionItem] = []
        self.view: dict[str, float] = {}
        self.settings = GraphSettings()
        self._next_id = 1

    # -- queries -----------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.items)

    def get(self, item_id: int) -> ExpressionItem:
        for item in self.items:
            if item.id == item_id:
                return item
        raise KeyError(item_id)

    def index_of(self, item_id: int) -> int:
        return self.items.index(self.get(item_id))

    def rows(self) -> list[tuple[int, str]]:
        """Expression rows (tables excluded) as (id, text)."""
        return [(item.id, item.text) for item in self.items if item.table is None]

    def tables(self) -> list[tuple[int, TableData]]:
        return [(item.id, item.table) for item in self.items if item.table is not None]

    def next_color(self) -> str:
        """The palette colour used least so far (earliest in the palette on ties)."""
        used = Counter(item.color.lower() for item in self.items)
        return min(PALETTE, key=lambda c: (used[c], PALETTE.index(c)))

    # -- mutation --------------------------------------------------------------------
    def add(
        self,
        text: str = "",
        index: int | None = None,
        color: str | None = None,
        visible: bool = True,
        slider: SliderRange | None = None,
        table: TableData | None = None,
    ) -> ExpressionItem:
        color = color or self.next_color()
        item = ExpressionItem(self._next_id, text, color, visible, slider, table)
        self._next_id += 1
        if index is None:
            self.items.append(item)
        else:
            self.items.insert(max(0, min(index, len(self.items))), item)
        return item

    def remove(self, item_id: int) -> int:
        """Remove an item and return the index it occupied."""
        index = self.index_of(item_id)
        del self.items[index]
        return index

    def move(self, item_id: int, index: int) -> None:
        item = self.get(item_id)
        self.items.remove(item)
        self.items.insert(max(0, min(index, len(self.items))), item)

    def clear(self) -> None:
        self.items.clear()
        self.view = {}
        self.settings = GraphSettings()

    # -- serialisation -------------------------------------------------------------------
    def to_dict(self) -> JSONObject:
        return {
            "format": FORMAT,
            "version": TABLE_VERSION if any(i.table is not None for i in self.items) else 1,
            "expressions": [_item_dict(item) for item in self.items],
            "view": dict(self.view),
            "settings": self.settings.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: object) -> Document:
        if not is_string_keyed_dict(data) or data.get("format") != FORMAT:
            raise DocumentError("Not an OpenGraph document")
        version = data.get("version")
        if not isinstance(version, int) or isinstance(version, bool) or version > VERSION:
            raise DocumentError("This document was made by a newer version of OpenGraph")
        expressions = data.get("expressions", [])
        if not isinstance(expressions, list):
            raise DocumentError("Malformed document: 'expressions' must be a list")
        doc = cls()
        for entry in expressions:
            if not is_string_keyed_dict(entry):
                raise DocumentError("Malformed expression entry")
            text = entry.get("text", "")
            if not isinstance(text, str):
                raise DocumentError("Malformed expression entry")
            color = entry.get("color")
            doc.add(
                text,
                color=color if isinstance(color, str) and _HEX.match(color) else None,
                visible=bool(entry.get("visible", True)),
                slider=SliderRange.from_dict(entry.get("slider")),
                table=TableData.from_dict(entry["table"]) if "table" in entry else None,
            )
        view = data.get("view", {})
        if isinstance(view, dict):
            doc.view = {
                k: float(v)
                for k, v in view.items()
                if isinstance(k, str) and isinstance(v, int | float)
            }
        doc.settings = GraphSettings.from_dict(data.get("settings"))
        return doc

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Document:
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DocumentError(f"Could not read {Path(path).name}: {exc}") from exc
        return cls.from_dict(data)


def _item_dict(item: ExpressionItem) -> JSONObject:
    entry: JSONObject = {"text": item.text, "color": item.color, "visible": item.visible}
    if item.slider is not None:  # optional key: older readers ignore it, so no version bump
        entry["slider"] = item.slider.to_dict()
    if item.table is not None:
        entry["table"] = item.table.to_dict()
    return entry
