"""Typesetting the math tree into glyphs and strokes.

:class:`Typesetter` turns a :class:`MathList` into a tree of :class:`Box` es (TeX style:
width, ascent above the baseline, descent below it), and :func:`flatten` turns that into
a :class:`Layout` with absolute coordinates: the glyphs and strokes to paint, plus the
geometry the editor needs — caret positions of every list, node rectangles for
selections and error underlines, and hit testing for mouse clicks.

Everything is measured with the metrics of the bundled fonts (see
:mod:`opengraph.ui.fonts`), the same files the client draws with, so the caret and the
selection line up with the glyphs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from opengraph.engine.builtins import CAS_OPERATIONS, FUNCTIONS
from opengraph.ui.fonts import EDITOR_SIZE, MONO, SERIF, SERIF_ITALIC, Font
from opengraph.ui.geometry import Path, Point, Rect
from opengraph.ui.mathedit.nodes import (
    BIG_OPERATORS,
    BigOp,
    Char,
    Frac,
    MathList,
    Node,
    Paren,
    Position,
    Root,
    Sub,
    Sup,
    Symbol,
    is_letter,
)

_DISPLAY = {"-": "−", "*": "·", "<=": "≤", ">=": "≥", "–": "−", "⋅": "·", "in": "∈"}
_BINARY = frozenset("+−·×÷")
_RELATIONS = frozenset("=<>≤≥∈")
_SETS = "sets"


def _membership_in(lst: MathList, index: int) -> bool:
    """Whether the ``in`` at ``index`` is the keyword of ``k in sets[N]``, as in the
    parser: ``sets[`` follows it (spaces aside), or ``sets`` is still being typed."""
    i = index + 1
    while i < len(lst.nodes) and _is_space(lst.nodes[i]):
        i += 1
    j = i
    while j < len(lst.nodes) and is_letter(lst.nodes[j]):
        j += 1
    word = "".join(n.text for n in lst.nodes[i:j] if isinstance(n, Char))
    after = lst.nodes[j] if j < len(lst.nodes) else None
    if after is None:
        return _SETS.startswith(word)
    return word == _SETS and isinstance(after, Paren) and after.left == "["


# Words that name an operation's parts: sum(k^2 for k from 1 to 10). Drawn upright only
# inside such an operation's parentheses; elsewhere they are ordinary names.
_CLAUSE_WORDS = frozenset({"from", "to", "at", "degree", "order"})
_OPERATIONS = frozenset({"limit", "taylor", "diff", "integrate", "roots", "collect", "apart"}) | {
    "plot",
    "parametric",
}


def _in_operation(lst: MathList) -> bool:
    """Whether ``lst`` is the inside of ``name(…)`` for an operation with clauses."""
    paren = lst.parent
    if not isinstance(paren, Paren) or paren.left != "(" or paren.parent is None:
        return False
    outer = paren.parent
    at = outer.index_of(paren)
    start = at
    while start > 0 and is_letter(outer.nodes[start - 1]):
        start -= 1
    return "".join(n.text for n in outer.nodes[start:at] if isinstance(n, Char)) in _OPERATIONS


def _set_letter(lst: MathList, index: int) -> bool:
    """Whether the N or Z at ``index`` is the whole inside of ``sets[…]``: drawn ℕ / ℤ,
    so the set never looks like a variable N."""
    paren = lst.parent
    if not isinstance(paren, Paren) or paren.left != "[" or paren.parent is None:
        return False
    if any(not _is_space(n) for i, n in enumerate(lst.nodes) if i != index):
        return False
    outer = paren.parent
    at = outer.index_of(paren)
    word = "".join(n.text for n in outer.nodes[max(0, at - len(_SETS)) : at] if isinstance(n, Char))
    return word == _SETS and (at == len(_SETS) or not is_letter(outer.nodes[at - len(_SETS) - 1]))


_PREFIX_CONTEXT = frozenset(["+", "-", "*", "/", "=", "<", ">", ",", "(", "|", "<=", ">="])
_OPENING = {")": "(", "]": "[", "}": "{"}
_FUNCTION_NAMES = sorted(
    {n for n in [*FUNCTIONS, *CAS_OPERATIONS, *BIG_OPERATORS] if len(n) > 1 and n.isascii()},
    key=len,
    reverse=True,
)

SCRIPT_SCALE = 0.7
FRACTION_SCALE = 0.9
MIN_SCALE = 0.5
WRAP_ALLOWANCE = 4.0  # ems of a row kept free beside a wrapped list ("k = [", "]")


@dataclass(slots=True)
class Box:
    """A typeset item; children and drawing ops are relative to its baseline origin."""

    width: float = 0.0
    ascent: float = 0.0
    descent: float = 0.0
    glyphs: list[tuple[float, float, str, Font]] = field(default_factory=list)
    strokes: list[tuple[Path, float]] = field(default_factory=list)
    holes: list[Rect] = field(default_factory=list)
    kids: list[tuple[float, float, Box]] = field(default_factory=list)
    node: Node | None = None
    lst: MathList | None = None
    carets: list[float] = field(default_factory=list)
    # Baseline of each caret: all 0 unless the list is wrapped onto several lines.
    caret_ys: list[float] = field(default_factory=list)
    caret_extent: tuple[float, float] = (0.0, 0.0)  # ascent, descent of a caret


@dataclass(slots=True)
class ListGeometry:
    lst: MathList
    xs: list[float]
    ys: list[float]  # the baseline at each caret position (lines of a wrapped list)
    caret_ascent: float
    caret_descent: float
    rect: Rect

    @property
    def baseline(self) -> float:
        """The baseline of the first line."""
        return self.ys[0]

    def line_indices(self, y: float) -> list[int]:
        """Caret indices on the line whose baseline is closest to ``y``."""
        nearest = min(self.ys, key=lambda b: abs(b - y))
        return [i for i, b in enumerate(self.ys) if b == nearest]


@dataclass(slots=True)
class Layout:
    width: float = 0.0
    ascent: float = 0.0
    descent: float = 0.0
    glyphs: list[tuple[Point, str, Font]] = field(default_factory=list)
    strokes: list[tuple[Path, float]] = field(default_factory=list)
    holes: list[Rect] = field(default_factory=list)
    lists: dict[int, ListGeometry] = field(default_factory=dict)
    nodes: dict[int, Rect] = field(default_factory=dict)

    def geometry(self, lst: MathList) -> ListGeometry:
        return self.lists[id(lst)]

    def caret_line(self, pos: Position) -> tuple[float, float, float]:
        """``(x, top, bottom)`` of the caret at ``pos``."""
        geo = self.geometry(pos.lst)
        i = min(pos.index, len(geo.xs) - 1)
        return geo.xs[i], geo.ys[i] - geo.caret_ascent, geo.ys[i] + geo.caret_descent

    def nearest_index(self, lst: MathList, x: float, y: float | None = None) -> int:
        """The caret index closest to ``x``, on the line closest to ``y`` if given."""
        geo = self.geometry(lst)
        indices = range(len(geo.xs)) if y is None else geo.line_indices(y)
        return min(indices, key=lambda i: abs(geo.xs[i] - x))

    def line_step(self, pos: Position, direction: int) -> Position | None:
        """The caret one line up (-1) or down (+1) in a wrapped list, if there is one."""
        geo = self.geometry(pos.lst)
        y = geo.ys[min(pos.index, len(geo.ys) - 1)]
        lines = sorted(set(geo.ys))
        k = lines.index(y) + direction
        if not 0 <= k < len(lines):
            return None
        x = self.caret_line(pos)[0]
        return Position(pos.lst, self.nearest_index(pos.lst, x, lines[k]))

    def position_at(self, point: Point, root: MathList) -> Position:
        """The caret position closest to ``point`` (in layout coordinates)."""
        best = self.lists[id(root)]
        best_area = float("inf")
        for geo in self.lists.values():
            if geo.lst is root or not geo.rect.contains(point):
                continue
            area = geo.rect.width * geo.rect.height
            if area < best_area:
                best, best_area = geo, area
        return Position(best.lst, self.nearest_index(best.lst, point.x, point.y))

    def range_rect(self, lst: MathList, start: int, end: int) -> Rect:
        """Bounding rectangle of ``lst.nodes[start:end]`` (or a caret-high sliver)."""
        geo = self.geometry(lst)
        height = geo.caret_ascent + geo.caret_descent
        x0, y0 = geo.xs[start], geo.ys[start] - geo.caret_ascent
        x1, y1 = geo.xs[end], geo.ys[end] - geo.caret_ascent
        rect = Rect(min(x0, x1), min(y0, y1), abs(x1 - x0), abs(y1 - y0) + height)
        for node in lst.nodes[start:end]:
            node_rect = self.nodes.get(id(node))
            if node_rect is not None:
                rect = rect.united(node_rect)
        return rect


class Typesetter:
    """Lays out math lists at one base size (logical pixels)."""

    def __init__(self, size: float = EDITOR_SIZE, *, plain: bool = False) -> None:
        self.size = size
        # Plain mode shows the source text as typed (monospace, no math styling); it is
        # used for linear input, where every node is a character.
        self.plain = plain
        self.wrap_width: float | None = None  # see layout()

    def font(self, scale: float, italic: bool = False) -> Font:
        family = MONO if self.plain else SERIF_ITALIC if italic else SERIF
        return Font(family, round(self.size * scale, 3))

    def em(self, scale: float) -> float:
        return self.font(scale).height() / 1.15

    # -- lists ---------------------------------------------------------------------------
    def layout(self, root: MathList, wrap_width: float | None = None) -> Layout:
        """``root`` typeset; ``[…]`` lists wider than ``wrap_width`` break after commas
        onto stacked lines."""
        self.wrap_width = wrap_width
        try:
            return flatten(self.list_box(root, 1.0, placeholder=False))
        finally:
            self.wrap_width = None

    def list_box(self, lst: MathList, scale: float, *, placeholder: bool = True) -> Box:
        fm = self.font(scale)
        upright = set() if self.plain else self._upright_letters(lst)
        boxes: list[Box | None] = [None] * len(lst.nodes)
        for i, node in enumerate(lst.nodes):
            if not isinstance(node, Sup | Sub):
                boxes[i] = self.node_box(lst, i, scale, i in upright)
        if not self.plain:
            self._stretch_parens(lst, boxes, scale)
        for i, node in enumerate(lst.nodes):
            if isinstance(node, Sup | Sub):
                boxes[i] = self.script_box(node, scale, boxes[i - 1] if i else None)

        box = Box(lst=lst, caret_extent=(fm.ascent() * 0.92, fm.descent()))
        x = 0.0
        box.ascent, box.descent = fm.ascent(), fm.descent()
        for child in boxes:
            assert child is not None
            box.carets.append(x)
            box.kids.append((x, 0.0, child))
            x += child.width
            box.ascent = max(box.ascent, child.ascent)
            box.descent = max(box.descent, child.descent)
        box.carets.append(x)
        if not lst.nodes and placeholder:
            em = self.em(scale)
            width = 0.6 * em
            box.holes.append(
                Rect(0.08 * em, -fm.ascent() * 0.72, width - 0.16 * em, fm.ascent() * 0.8)
            )
            box.carets = [width / 2]
            x = width
        box.width = x
        return box

    def _upright_letters(self, lst: MathList) -> set[int]:
        """Indices of letters spelling a builtin function name (drawn upright: sin, ln)."""
        upright: set[int] = set()
        i, nodes = 0, lst.nodes
        while i < len(nodes):
            if not is_letter(nodes[i]):
                i += 1
                continue
            j = i
            while j < len(nodes) and is_letter(nodes[j]):
                j += 1
            word = "".join(n.text for n in nodes[i:j] if isinstance(n, Char))
            if word in ("seq", "sets", "for", "in", "plot", "parametric") or (
                word in _CLAUSE_WORDS and _in_operation(lst)
            ):
                upright.update(range(i, j))
            k = 0
            while k < len(word):
                name = next((n for n in _FUNCTION_NAMES if word.startswith(n, k)), None)
                if name is None:
                    k += 1
                else:
                    upright.update(range(i + k, i + k + len(name)))
                    k += len(name)
            i = j
        return upright

    # -- leaves ------------------------------------------------------------------------------
    def text_box(
        self, text: str, scale: float, *, italic: bool = False, left: float = 0, right: float = 0
    ) -> Box:
        font = self.font(scale, italic)
        em = self.em(scale)
        pad_left, pad_right = left * em, right * em
        if italic:
            pad_right += 0.04 * em  # italic correction
        width = pad_left + font.advance(text) + pad_right
        return Box(width, font.ascent(), font.descent(), glyphs=[(pad_left, 0.0, text, font)])

    def node_box(self, lst: MathList, index: int, scale: float, upright: bool) -> Box:
        node = lst.nodes[index]
        if isinstance(node, Symbol):
            glyph = node.glyph
            box = self.text_box(glyph, scale, italic=glyph.islower())
        elif isinstance(node, Char):
            box = self.char_box(lst, index, node.text, scale, upright)
        elif isinstance(node, Frac):
            box = self.frac_box(node, scale)
        elif isinstance(node, Root):
            box = self.root_box(node, scale)
        elif isinstance(node, Paren):
            box = self.bracket_box(node, scale)
        elif isinstance(node, BigOp):
            box = self.big_op_box(node, scale)
        else:  # pragma: no cover - scripts are laid out in a second pass
            raise TypeError(node)
        box.node = node
        return box

    def char_box(self, lst: MathList, index: int, text: str, scale: float, upright: bool) -> Box:
        if self.plain:
            return self.text_box(text, scale)
        shown = _DISPLAY.get(text, text)
        if text == "in" and not _membership_in(lst, index):
            # Complex-mode "in" is i·n, not ∈: only "in sets[…]" (or "in" still being
            # typed at the end) is the membership keyword, as in the parser. In an
            # operation (plot(t^2 for t in (0, 2])) it is a clause word, drawn upright.
            return self.text_box(text, scale, italic=not _in_operation(lst))
        if text in ("N", "Z") and _set_letter(lst, index):
            shown = {"N": "ℕ", "Z": "ℤ"}[text]
        spacing = 1.0 if scale >= 0.99 else 0.55  # tighter operators in scripts
        if text == " ":
            # Operators bring their own spacing; a typed space next to one stays thin
            # (but non-zero, so the caret visibly moves over it).
            spaced = _RELATIONS | _BINARY | {","}
            neighbours = [i for i in (index - 1, index + 1) if 0 <= i < len(lst.nodes)]
            thin = any(
                isinstance(n := lst.nodes[i], Char)
                and _DISPLAY.get(n.text, n.text) in spaced
                and (n.text != "in" or _membership_in(lst, i))
                for i in neighbours
            )
            fm = self.font(scale)
            return Box((0.06 if thin else 0.25) * self.em(scale), fm.ascent(), fm.descent())
        if shown in _RELATIONS:
            return self.text_box(shown, scale, left=0.28 * spacing, right=0.28 * spacing)
        if len(text) == 1 and text.isalpha():
            return self.text_box(shown, scale, italic=not upright and text.isascii())
        if shown in _BINARY and not self._is_prefix(lst, index):
            return self.text_box(shown, scale, left=0.22 * spacing, right=0.22 * spacing)
        if shown == ",":
            return self.text_box(shown, scale, right=0.25 * spacing)
        return self.text_box(shown, scale)

    @staticmethod
    def _is_prefix(lst: MathList, index: int) -> bool:
        """Is the operator at ``index`` a sign (``-x``, ``2^-x``) rather than binary?"""
        for node in reversed(lst.nodes[:index]):
            if isinstance(node, Char) and node.text == " ":
                continue
            return isinstance(node, Char) and node.text in _PREFIX_CONTEXT
        return True

    # -- parentheses ------------------------------------------------------------------------
    def _stretch_parens(self, lst: MathList, boxes: list[Box | None], scale: float) -> None:
        fm = self.font(scale)
        stack: list[int] = []
        for i, node in enumerate(lst.nodes):
            if not isinstance(node, Char):
                continue
            if node.text == "(":
                stack.append(i)
            elif node.text == ")" and stack:
                opener = stack.pop()
                inner = [b for b in boxes[opener + 1 : i] if b is not None]
                ascent = max((b.ascent for b in inner), default=0.0)
                descent = max((b.descent for b in inner), default=0.0)
                if ascent > fm.ascent() * 1.02 or descent > fm.descent() * 1.3:
                    boxes[opener] = self.paren_box("(", ascent, descent, scale, lst.nodes[opener])
                    boxes[i] = self.paren_box(")", ascent, descent, scale, node)

    def paren_box(self, side: str, ascent: float, descent: float, scale: float, node: Node) -> Box:
        """A bracket drawn as a stroke spanning ``ascent`` + ``descent`` (any of ()[]{})."""
        em = self.em(scale)
        width = {"(": 0.42, "[": 0.36, "{": 0.5}[_OPENING.get(side, side)] * em
        top, bottom = -ascent - 0.04 * em, descent + 0.04 * em
        mid = (top + bottom) / 2
        opening = side in "([{"

        def at(x: float, y: float) -> Point:
            # Shapes are drawn as opening brackets and mirrored for closing ones.
            return Point(x if opening else width - x, y)

        if side in "()":
            # A quadratic curve bulging outwards; its control point lies beyond the box edge.
            end = width - 0.08 * em
            path = Path.start(at(end, top))
            path.quad_to(at(-0.1 * em, mid), at(end, bottom))
        elif side in "[]":
            stem, end = 0.1 * em, width - 0.06 * em
            path = Path.start(at(end, top))
            path.line_to(at(stem, top))
            path.line_to(at(stem, bottom))
            path.line_to(at(end, bottom))
        else:
            # A brace: two arms meeting in a point at mid-height.
            tip, stem, end = 0.06 * em, width / 2, width - 0.06 * em
            r = min(0.18 * em, (bottom - top) / 8)
            path = Path.start(at(end, top))
            path.quad_to(at(stem, top), at(stem, top + r))
            path.line_to(at(stem, mid - r))
            path.quad_to(at(stem, mid), at(tip, mid))
            path.quad_to(at(stem, mid), at(stem, mid + r))
            path.line_to(at(stem, bottom - r))
            path.quad_to(at(stem, bottom), at(end, bottom))
        stroke = max(1.0, 0.065 * em)
        return Box(width, -top, bottom, strokes=[(path, stroke)], node=node)

    def bracket_box(self, node: Paren, scale: float) -> Box:
        """A bracket pair around its contents; the brackets grow with tall contents."""
        content = self.list_box(node.body, scale)
        fm = self.font(scale)
        if node.left == "[" and scale >= 0.99 and self.wrap_width is not None:
            # Leave room for what usually precedes a list ("k = [") and the brackets.
            available = self.wrap_width - WRAP_ALLOWANCE * self.em(scale)
            if content.width > available:
                self._wrap(content, available, scale)
        if content.ascent > fm.ascent() * 1.02 or content.descent > fm.descent() * 1.3:
            left = self.paren_box(node.left, content.ascent, content.descent, scale, node)
            right = self.paren_box(node.right, content.ascent, content.descent, scale, node)
            left.node = right.node = None
        else:
            left, right = self.text_box(node.left, scale), self.text_box(node.right, scale)
        return Box(
            left.width + content.width + right.width,
            max(left.ascent, content.ascent, right.ascent),
            max(left.descent, content.descent, right.descent),
            kids=[
                (0.0, 0.0, left),
                (left.width, 0.0, content),
                (left.width + content.width, 0.0, right),
            ],
        )

    def _wrap(self, box: Box, width: float, scale: float) -> None:
        """Re-place a list's items on lines no wider than ``width`` where possible,
        breaking after top-level commas; the block is centred on the maths axis."""
        lst = box.lst
        assert lst is not None
        nodes = lst.nodes
        # Pieces of the list that end after a comma (and the spaces following it).
        pieces: list[tuple[int, int]] = []
        start = depth = 0
        after_comma = False
        for i, node in enumerate(nodes):
            text = node.text if isinstance(node, Char) else ""
            depth += {"(": 1, ")": -1}.get(text, 0)
            after_comma = after_comma or (depth <= 0 and text == ",")
            last = i + 1 == len(nodes)
            if last or (after_comma and not _is_space(nodes[i + 1])):
                pieces.append((start, i + 1))
                start, after_comma = i + 1, False
        if len(pieces) < 2:
            return
        kids = [kid for _, _, kid in box.kids]

        def span(a: int, b: int) -> float:
            return sum(k.width for k in kids[a:b])

        lines: list[tuple[int, int]] = []
        for a, b in pieces:
            if lines and span(lines[-1][0], b) <= width:
                lines[-1] = (lines[-1][0], b)
            else:
                lines.append((a, b))
        if len(lines) < 2:
            return

        fm = self.font(scale)
        gap = 0.3 * self.em(scale)
        baselines: list[float] = []
        y = 0.0
        for k, (a, b) in enumerate(lines):
            ascent = max([fm.ascent(), *(kid.ascent for kid in kids[a:b])])
            descent = max([fm.descent(), *(kid.descent for kid in kids[a:b])])
            if k:
                y += gap + ascent
            baselines.append(y)
            y += descent
        first_ascent = max([fm.ascent(), *(kid.ascent for kid in kids[lines[0][0] : lines[0][1]])])
        top, bottom = -first_ascent, y
        shift = -fm.x_height() * 0.5 - (top + bottom) / 2  # centre on the axis
        box.kids, box.carets, box.caret_ys = [], [], []
        box.width = box.ascent = box.descent = 0.0
        for (a, b), baseline in zip(lines, baselines, strict=True):
            x, y = 0.0, baseline + shift
            for kid in kids[a:b]:
                box.carets.append(x)
                box.caret_ys.append(y)
                box.kids.append((x, y, kid))
                box.ascent = max(box.ascent, kid.ascent - y)
                box.descent = max(box.descent, kid.descent + y)
                x += kid.width
            box.width = max(box.width, x)
        box.carets.append(x)  # after the last node, at the end of the last line
        box.caret_ys.append(y)

    # -- structures -------------------------------------------------------------------------
    def script_box(self, node: Sup | Sub, scale: float, base: Box | None) -> Box:
        inner_scale = max(MIN_SCALE, scale * SCRIPT_SCALE)
        content = self.list_box(node.body, inner_scale)
        fm = self.font(scale)
        small = self.font(inner_scale)
        em = self.em(scale)
        if isinstance(node, Sup):
            base_ascent = base.ascent if base is not None else fm.ascent()
            shift = max(base_ascent - 0.62 * small.ascent(), 0.36 * em, content.descent + 0.18 * em)
            ascent, descent = shift + content.ascent, max(0.0, content.descent - shift)
            dy = -shift
        else:
            shift = 0.22 * em + max(0.0, content.ascent - small.ascent())
            ascent, descent = max(0.0, content.ascent - shift), shift + content.descent
            dy = shift
        lead = 0.02 * em
        return Box(
            lead + content.width + 0.05 * em,
            ascent,
            descent,
            kids=[(lead, dy, content)],
            node=node,
        )

    def frac_box(self, node: Frac, scale: float) -> Box:
        inner_scale = max(MIN_SCALE, scale * FRACTION_SCALE)
        num = self.list_box(node.num, inner_scale)
        den = self.list_box(node.den, inner_scale)
        fm = self.font(scale)
        em = self.em(scale)
        axis = fm.x_height() * 0.5
        thick = max(1.0, 0.05 * em)
        gap = 0.1 * em
        pad = 0.1 * em
        margin = 0.06 * em
        width = max(num.width, den.width) + 2 * pad
        num_y = -(axis + thick / 2 + gap + num.descent)
        den_y = -axis + thick / 2 + gap + den.ascent
        bar = Path.start(Point(margin, -axis))
        bar.line_to(Point(margin + width, -axis))
        return Box(
            width + 2 * margin,
            -num_y + num.ascent,
            den_y + den.descent,
            strokes=[(bar, thick)],
            kids=[
                (margin + (width - num.width) / 2, num_y, num),
                (margin + (width - den.width) / 2, den_y, den),
            ],
        )

    def root_box(self, node: Root, scale: float) -> Box:
        radicand = self.list_box(node.body, scale)
        em = self.em(scale)
        thick = max(1.0, 0.055 * em)
        gap = 0.1 * em
        sign = 0.62 * em
        top = -(radicand.ascent + gap + thick / 2)
        bottom = radicand.descent + 0.04 * em
        height = bottom - top
        kids: list[tuple[float, float, Box]] = []
        x0 = 0.04 * em
        ascent = -top + thick / 2
        if node.index is not None:
            index_box = self.text_box(str(node.index), max(MIN_SCALE * 0.9, scale * 0.55))
            x0 = max(x0, index_box.width - 0.3 * sign)
            index_y = top + 0.42 * height
            kids.append((x0 + 0.38 * sign - index_box.width, index_y, index_box))
            ascent = max(ascent, -index_y + index_box.ascent)
        end = x0 + sign + radicand.width + 0.08 * em
        path = Path.start(Point(x0, top + 0.6 * height))
        path.line_to(Point(x0 + 0.18 * sign, top + 0.52 * height))
        path.line_to(Point(x0 + 0.45 * sign, bottom))
        path.line_to(Point(x0 + 0.92 * sign, top))
        path.line_to(Point(end, top))
        kids.append((x0 + sign, 0.0, radicand))
        return Box(end + 0.06 * em, ascent, bottom, strokes=[(path, thick)], kids=kids)

    # -- big operators ----------------------------------------------------------------------
    def big_op_box(self, node: BigOp, scale: float) -> Box:
        """Σ/Π with the bounds above and below, ∫ with them at its side, lim with
        "t → a" below; then the body (in brackets when it has several terms)."""
        em = self.em(scale)
        small = max(MIN_SCALE, scale * SCRIPT_SCALE)
        body = self._grouped_body(node.body, scale)
        # A fresh Σ/Π shows every field (variable = start); once the start is filled in
        # without a variable, its empty slot disappears; the compiler validates arity.
        fresh = node.kind in ("sum", "prod") and not node.lower.nodes
        var = self.list_box(node.var, small, placeholder=node.kind == "limit" or fresh)
        lower = self.list_box(node.lower, small)
        kids: list[tuple[float, float, Box]] = []
        if node.kind == "limit":
            op = self.text_box("lim", scale)
            arrow = self.text_box("→", small, left=0.12, right=0.12)
            side = self._side_box(node.upper, small)
            under = _row([(var, 0.0), (arrow, 0.0), (lower, 0.0), (side, -0.32 * em)])
            column = max(op.width, under.width)
            below = op.descent * 0.5 + self._ink_ascent(under, small)
            kids += [
                ((column - op.width) / 2, 0.0, op),
                ((column - under.width) / 2, below, under),
            ]
            x = column + 0.2 * em
        elif node.kind in ("sum", "prod"):
            op = self._big_glyph("∑" if node.kind == "sum" else "∏", scale, 1.9)
            upper = self.list_box(node.upper, small)
            parts = [(var, 0.0), (lower, 0.0)]
            if node.var.nodes or fresh:
                parts.insert(1, (self.text_box("=", small), 0.0))
            under = _row(parts)
            gap = 0.12 * em
            column = max(op.width, under.width, upper.width)
            above = -op.ascent - gap - self._ink_descent(upper, small)
            below = op.descent + gap + self._ink_ascent(under, small)
            kids += [
                ((column - op.width) / 2, 0.0, op),
                ((column - upper.width) / 2, above, upper),
                ((column - under.width) / 2, below, under),
            ]
            x = column + 0.12 * em
        else:
            op = self._big_glyph("∫", scale, 2.2)
            upper = self.list_box(node.upper, small)
            kids += [
                (0.0, 0.0, op),
                (op.width + 0.08 * em, -op.ascent + upper.ascent * 0.85, upper),
                (op.width - 0.12 * em, op.descent - lower.descent * 0.6, lower),
            ]
            x = max(op.width + 0.08 * em + upper.width, op.width - 0.12 * em + lower.width)
            x += 0.12 * em
        kids.append((x, 0.0, body))
        x += body.width
        if node.kind == "int":
            if node.var.nodes:
                d = self.text_box("d", scale, left=0.17)
                kids.append((x, 0.0, d))
                x += d.width
            var_box = self.list_box(node.var, scale, placeholder=False)
            kids.append((x, 0.0, var_box))
            x += var_box.width
        return Box(
            x + 0.06 * em,
            max(kid.ascent - dy for _, dy, kid in kids),
            max(dy + kid.descent for _, dy, kid in kids),
            kids=kids,
        )

    def _ink_ascent(self, box: Box, scale: float) -> float:
        """How far ``box`` reaches above its baseline, ignoring the font's leading."""
        fm = self.font(scale)
        return box.ascent - max(0.0, fm.ascent() - fm.cap_height() * 1.1) * (
            box.ascent <= fm.ascent() * 1.02
        )

    def _ink_descent(self, box: Box, scale: float) -> float:
        """How far ``box`` reaches below its baseline, ignoring the font's leading."""
        fm = self.font(scale)
        return box.descent * (0.35 if box.descent <= fm.descent() * 1.02 else 1.0)

    def _big_glyph(self, glyph: str, scale: float, size: float) -> Box:
        """A large operator sign centred on the maths axis."""
        font = self.font(scale * size)
        axis = self.font(scale).x_height() * 0.5
        ink = font.tight_rect(glyph)
        dy = -axis - (ink.top + ink.bottom) / 2
        em = self.em(scale)
        return Box(
            font.advance(glyph) + 0.04 * em,
            -(ink.top + dy) + 0.04 * em,
            ink.bottom + dy + 0.04 * em,
            glyphs=[(0.02 * em, dy, glyph, font)],
        )

    def _grouped_body(self, lst: MathList, scale: float) -> Box:
        """The body, between brackets when it is a sum of terms or a relation, so it is
        clear how far the operator reaches."""
        box = self.list_box(lst, scale)
        if not any(_splits_terms(lst, i) for i in range(1, len(lst.nodes))):
            return box
        left = self.paren_box("(", box.ascent, box.descent, scale, lst.nodes[0])
        right = self.paren_box(")", box.ascent, box.descent, scale, lst.nodes[0])
        left.node = right.node = None
        return _row([(left, 0.0), (box, 0.0), (right, 0.0)])

    def _side_box(self, lst: MathList, scale: float) -> Box:
        """The side of a limit: ``right``/``left`` show as ⁺/⁻; anything else as typed."""
        box = self.list_box(lst, scale, placeholder=False)
        word = "".join(n.text for n in lst.nodes if isinstance(n, Char))
        sign = {"right": "+", "left": "−"}.get(word) if len(word) == len(lst.nodes) else None
        if sign is None:
            return box
        font = self.font(scale)
        width = font.advance(sign) + 0.04 * self.em(scale)
        box.kids, box.glyphs = [], [(0.0, 0.0, sign, font)]
        box.carets = [0.0] * len(lst.nodes) + [width]
        box.width = width
        return box


def _is_space(node: Node) -> bool:
    return isinstance(node, Char) and node.text == " "


def _splits_terms(lst: MathList, index: int) -> bool:
    """Is ``lst.nodes[index]`` a top-level ``+``/``−``/relation (not a leading sign)?"""
    node = lst.nodes[index]
    if not isinstance(node, Char) or node.text not in ("+", "-", "–", "=", "<", ">", "<=", ">="):
        return False
    return not Typesetter._is_prefix(lst, index)


def _row(items: list[tuple[Box, float]]) -> Box:
    """Boxes side by side, each raised by its offset (negative is up)."""
    row = Box()
    for box, dy in items:
        row.kids.append((row.width, dy, box))
        row.width += box.width
        row.ascent = max(row.ascent, box.ascent - dy)
        row.descent = max(row.descent, box.descent + dy)
    return row


def flatten(box: Box) -> Layout:
    layout = Layout(box.width, box.ascent, box.descent)
    _place(box, 0.0, 0.0, layout)
    return layout


def _place(box: Box, x: float, y: float, layout: Layout) -> None:
    for gx, gy, text, font in box.glyphs:
        layout.glyphs.append((Point(x + gx, y + gy), text, font))
    for path, width in box.strokes:
        layout.strokes.append((path.translated(x, y), width))
    for hole in box.holes:
        layout.holes.append(hole.translated(x, y))
    rect = Rect(x, y - box.ascent, box.width, box.ascent + box.descent)
    if box.node is not None:
        layout.nodes[id(box.node)] = rect
    if box.lst is not None:
        caret_ascent, caret_descent = box.caret_extent
        ys = box.caret_ys or [0.0] * len(box.carets)
        layout.lists[id(box.lst)] = ListGeometry(
            box.lst,
            [x + c for c in box.carets],
            [y + cy for cy in ys],
            caret_ascent,
            caret_descent,
            rect.adjusted(-2, 0, 2, 0),
        )
    for dx, dy, kid in box.kids:
        _place(kid, x + dx, y + dy, layout)
