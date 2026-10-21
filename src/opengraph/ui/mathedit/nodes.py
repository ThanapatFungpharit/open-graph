"""Editable tree for an expression typeset as mathematics. Toolkit-free.

The tree is the editor's model; the expression *source text* (the language accepted by
:mod:`opengraph.engine.parser`) stays authoritative and is derived from it by
:func:`opengraph.ui.mathedit.source.to_source`.

* A :class:`MathList` is a horizontal run of nodes (the whole row, a numerator, an
  exponent, ...). The caret always sits *between* two nodes of one list.
* Leaves are :class:`Char` (one source token character, or ``<=``/``>=``) and
  :class:`Symbol` (a Greek letter written by name, e.g. ``theta`` shown as θ).
* Structures own one or more lists: :class:`Frac`, :class:`Sup` (exponent of the node
  before it), :class:`Sub` (subscript of a name), :class:`Root` (``sqrt``/``cbrt``) and
  :class:`Paren` (a matched ``()``/``[]``/``{}`` pair) and :class:`BigOp` (Σ, Π, ∫ and
  lim with their bounds). Unmatched brackets are Chars.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

# Greek letters that may be typed by name. "xi" is left out on purpose: it would turn
# "x" followed by "i" into ξ, and the engine reads "xi" as a name, not a letter.
GREEK: dict[str, str] = {
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
    "delta": "δ",
    "epsilon": "ε",
    "zeta": "ζ",
    "eta": "η",
    "theta": "θ",
    "iota": "ι",
    "kappa": "κ",
    "lambda": "λ",
    "mu": "μ",
    "nu": "ν",
    "pi": "π",
    "rho": "ρ",
    "sigma": "σ",
    "tau": "τ",
    "upsilon": "υ",
    "phi": "φ",
    "chi": "χ",
    "psi": "ψ",
    "omega": "ω",
    "Gamma": "Γ",
    "Delta": "Δ",
    "Theta": "Θ",
    "Lambda": "Λ",
    "Pi": "Π",
    "Sigma": "Σ",
    "Phi": "Φ",
    "Psi": "Ψ",
    "Omega": "Ω",
}

ROOT_FUNCTIONS: dict[str, int | None] = {"sqrt": None, "cbrt": 3}
"""Builtin functions drawn with a radical sign, and the index shown on it."""


class Char:
    """One source token character such as ``x``, ``2``, ``+`` or ``(``.

    ``<=`` and ``>=`` are single nodes so the caret never splits them.
    """

    __slots__ = ("text",)

    def __init__(self, text: str) -> None:
        self.text = text

    def __repr__(self) -> str:
        return f"Char({self.text!r})"


class Symbol:
    """A Greek letter written by name in the source (``theta``); drawn as θ."""

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name

    @property
    def glyph(self) -> str:
        return GREEK.get(self.name, self.name)

    def __repr__(self) -> str:
        return f"Symbol({self.name!r})"


class MathList:
    """A horizontal sequence of nodes; the parent structure (if any) owns it."""

    __slots__ = ("nodes", "parent")

    def __init__(self, nodes: Iterable[Node] = ()) -> None:
        self.nodes: list[Node] = []
        self.parent: Structure | None = None
        self.insert(0, list(nodes))

    def __len__(self) -> int:
        return len(self.nodes)

    def __repr__(self) -> str:
        return f"MathList({self.nodes!r})"

    def insert(self, index: int, nodes: list[Node]) -> None:
        for node in nodes:
            if isinstance(node, Structure):
                node.parent = self
        self.nodes[index:index] = nodes

    def remove(self, start: int, end: int) -> list[Node]:
        removed = self.nodes[start:end]
        del self.nodes[start:end]
        for node in removed:
            if isinstance(node, Structure):
                node.parent = None
        return removed

    def index_of(self, node: Node | Structure) -> int:
        for i, candidate in enumerate(self.nodes):
            if candidate is node:
                return i
        raise ValueError(f"{node!r} is not in this list")


class Structure:
    """A node made of one or more editable lists ("blocks")."""

    __slots__ = ("blocks", "parent")

    def __init__(self, *blocks: MathList) -> None:
        self.blocks: tuple[MathList, ...] = blocks
        self.parent: MathList | None = None
        for block in blocks:
            block.parent = self

    def __repr__(self) -> str:
        return f"{type(self).__name__}{self.blocks!r}"


class Frac(Structure):
    __slots__ = ()

    def __init__(self, num: MathList, den: MathList) -> None:
        super().__init__(num, den)

    @property
    def num(self) -> MathList:
        return self.blocks[0]

    @property
    def den(self) -> MathList:
        return self.blocks[1]


class Sup(Structure):
    """Exponent applied to the node just before it (as ``^`` in the source)."""

    __slots__ = ()

    def __init__(self, exp: MathList) -> None:
        super().__init__(exp)

    @property
    def body(self) -> MathList:
        return self.blocks[0]


class Sub(Structure):
    """Subscript of the name just before it (``a_1`` in the source)."""

    __slots__ = ()

    def __init__(self, sub: MathList) -> None:
        super().__init__(sub)

    @property
    def body(self) -> MathList:
        return self.blocks[0]


class Root(Structure):
    """``sqrt(...)`` or ``cbrt(...)`` drawn as a radical."""

    __slots__ = ("function", "swallowed_paren")

    def __init__(self, function: str, radicand: MathList) -> None:
        super().__init__(radicand)
        self.function = function
        # Typing "sqrt(" swallows the "(" (the radical already groups); remember it so a
        # matching ")" just leaves the radical. Editing state only, never serialised.
        self.swallowed_paren = False

    @property
    def body(self) -> MathList:
        return self.blocks[0]

    @property
    def index(self) -> int | None:
        return ROOT_FUNCTIONS.get(self.function)


BRACKETS: dict[str, str] = {"(": ")", "[": "]", "{": "}"}
"""Opening bracket -> its closing bracket."""
CLOSERS: dict[str, str] = {close: open_ for open_, close in BRACKETS.items()}


class Paren(Structure):
    """A bracket pair, including mixed interval brackets, sized to its contents.

    Typing an opening bracket inserts both sides at once (the closing one is never typed
    separately); typing the closing bracket inside steps out of it.
    """

    __slots__ = ("left", "right")

    def __init__(self, left: str, body: MathList, right: str | None = None) -> None:
        super().__init__(body)
        self.left = left
        self.right = BRACKETS[left] if right is None else right

    @property
    def body(self) -> MathList:
        return self.blocks[0]


BIG_OPERATORS: dict[str, str] = {
    "sum": "sum",
    "Σ": "sum",
    "Sigma": "sum",
    "prod": "prod",
    "Π": "prod",
    "Pi": "prod",
    "int": "int",
    "integral": "int",
    "∫": "int",
    "limit": "limit",
}
"""Call names drawn as a big operator (Σ, Π, ∫, lim) -> the operator's kind."""


class BigOp(Structure):
    """``sum(body, k, a, b)``, ``prod(…)``, ``int(…)`` or ``limit(body, t, a[, side])``
    drawn as Σ/Π/∫/lim with its bounds.

    The source order of the arguments is body, variable, lower, upper (for a limit:
    body, variable, point, side), but the blocks are kept in reading order, which is
    what the arrow keys visit: Σ and Π read variable, lower, upper, body; ∫ reads lower,
    upper, body, variable; lim reads variable, point, side, body. An empty variable of
    Σ/Π/∫ means the three-argument form (integrals default to x; sums/products need
    an index); an empty side of a
    limit means the two-sided limit.
    """

    __slots__ = ("function",)

    def __init__(
        self, function: str, body: MathList, var: MathList, lower: MathList, upper: MathList
    ) -> None:
        if BIG_OPERATORS[function] == "int":
            super().__init__(lower, upper, body, var)
        else:
            super().__init__(var, lower, upper, body)
        self.function = function

    @property
    def kind(self) -> str:
        return BIG_OPERATORS[self.function]

    def _block(self, source_index: int) -> MathList:
        order = (2, 3, 0, 1) if self.kind == "int" else (3, 0, 1, 2)
        return self.blocks[order[source_index]]

    @property
    def body(self) -> MathList:
        return self._block(0)

    @property
    def var(self) -> MathList:
        return self._block(1)

    @property
    def lower(self) -> MathList:
        """The lower bound (the point a limit is taken at)."""
        return self._block(2)

    @property
    def upper(self) -> MathList:
        """The upper bound (the side of a limit: ``left``, ``right`` or empty)."""
        return self._block(3)

    @property
    def arguments(self) -> tuple[MathList, MathList, MathList, MathList]:
        """The blocks in source order: body, variable, lower, upper."""
        return self.body, self.var, self.lower, self.upper


Node = Char | Symbol | Frac | Sup | Sub | Root | Paren | BigOp


def is_letter(node: Node | None) -> bool:
    return isinstance(node, Char) and len(node.text) == 1 and node.text.isalpha()


def clone_list(lst: MathList) -> MathList:
    return MathList(clone_node(n) for n in lst.nodes)


def clone_node(node: Node) -> Node:
    match node:
        case Char(text=text):
            return Char(text)
        case Symbol(name=name):
            return Symbol(name)
        case Frac():
            return Frac(clone_list(node.num), clone_list(node.den))
        case Sup():
            return Sup(clone_list(node.body))
        case Sub():
            return Sub(clone_list(node.body))
        case Root():
            root = Root(node.function, clone_list(node.body))
            root.swallowed_paren = node.swallowed_paren
            return root
        case Paren():
            return Paren(node.left, clone_list(node.body), node.right)
        case BigOp():
            return BigOp(node.function, *(clone_list(b) for b in node.arguments))
    raise TypeError(node)  # pragma: no cover


def iter_lists(lst: MathList) -> Iterator[MathList]:
    """``lst`` and every list nested inside it, outermost first."""
    yield lst
    for node in lst.nodes:
        if isinstance(node, Structure):
            for block in node.blocks:
                yield from iter_lists(block)


class Position:
    """A caret position: between ``lst.nodes[index - 1]`` and ``lst.nodes[index]``."""

    __slots__ = ("index", "lst")

    def __init__(self, lst: MathList, index: int) -> None:
        self.lst = lst
        self.index = index

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Position) and self.lst is other.lst and self.index == other.index

    def __hash__(self) -> int:
        return hash((id(self.lst), self.index))

    def __repr__(self) -> str:
        return f"Position({self.lst!r}, {self.index})"


Path = tuple[int, ...]
"""Position as indices from the root: (node, block)* pairs, then the caret index."""


def path_of(pos: Position) -> Path:
    steps: list[int] = [pos.index]
    lst = pos.lst
    while lst.parent is not None:
        structure = lst.parent
        parent = structure.parent
        assert parent is not None
        steps.append(next(i for i, b in enumerate(structure.blocks) if b is lst))
        steps.append(parent.index_of(structure))
        lst = parent
    return tuple(reversed(steps))


def position_at(root: MathList, path: Path) -> Position:
    """Inverse of :func:`path_of`; clamps to the nearest valid position."""
    lst = root
    for i in range(0, len(path) - 1, 2):
        node_index, block_index = path[i], path[i + 1]
        node = lst.nodes[node_index] if node_index < len(lst.nodes) else None
        if not isinstance(node, Structure) or block_index >= len(node.blocks):
            return Position(lst, min(node_index, len(lst.nodes)))
        lst = node.blocks[block_index]
    return Position(lst, min(path[-1], len(lst.nodes)))


def ancestry(pos: Position) -> list[tuple[MathList, int, bool]]:
    """``(list, index, exact)`` from the root down to ``pos``.

    For enclosing lists the index is that of the structure containing the position
    (``exact`` is False); the last entry is ``pos`` itself (``exact`` is True).
    """
    chain = [(pos.lst, pos.index, True)]
    lst = pos.lst
    while lst.parent is not None:
        structure = lst.parent
        parent = structure.parent
        assert parent is not None
        chain.append((parent, parent.index_of(structure), False))
        lst = parent
    chain.reverse()
    return chain
