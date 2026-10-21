"""Editing commands over the math tree: caret, selection, typing, deletion, undo. Toolkit-free.

Caret semantics follow MathQuill/Desmos:

* Left/Right step over characters and *into* structures, visiting every block in order
  (numerator, then denominator), and step out at a block's end.
* Shift+Left/Right extend the selection by whole siblings; at a block's edge the
  selection grows to the enclosing structure, so a selection is always a contiguous run
  of nodes in one list.
* Typing ``^`` ``_`` ``/`` creates an exponent, subscript or fraction (the operand
  before the caret, or the selection, becomes the base/numerator). Typing ``sqrt``,
  ``cbrt`` or a Greek name at a delimiter converts the complete token; Ctrl+Z restores it.
* Additive and relation operators typed at the end of an exponent or fraction leave
  that structure, so ordinary linear typing reads as expected: ``x^2+1`` becomes
  ``x²+1`` and ``1/2+x`` becomes ``½+x``. Arrow keys still navigate every slot
  explicitly. (A subscript is part of a name, so anything but a letter or digit still
  ends it.)
* Typing ``(``, ``[`` or ``{`` inserts the matching pair (a :class:`Paren`) with the
  caret inside; typing the closing bracket steps out of the nearest enclosing pair of
  that kind (from inside an exponent too), carrying anything after the caret out with
  it. Round and square closers can change the pair to ``[…)`` or ``(…]``.
  A closing bracket with no pair to close pairs up with an unmatched opening
  bracket earlier in the same list, or is inserted as an unmatched character.
* Typing ``sum``, ``prod``, ``int``, ``limit`` (or Σ, Π, ∫) followed by a space or Tab
  makes a big operator with empty fields (variable = start below, end above, then the
  body); the arrow keys visit them in that order. Typing ``(`` instead means writing
  the call: its slots then fill in the order of the call's arguments, so
  ``sum(x^2,x,0,100)`` typed straight through gives Σ with x = 0 below and 100 above:
  ``,`` moves on to the next argument and ``)`` leaves.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from opengraph.ui.mathedit.latex import COMMANDS, TEXT_COMMANDS, latex_to_source, to_latex
from opengraph.ui.mathedit.nodes import (
    BIG_OPERATORS,
    BRACKETS,
    CLOSERS,
    GREEK,
    ROOT_FUNCTIONS,
    BigOp,
    Char,
    Frac,
    MathList,
    Node,
    Paren,
    Path,
    Position,
    Root,
    Structure,
    Sub,
    Sup,
    Symbol,
    ancestry,
    clone_list,
    is_letter,
    path_of,
    position_at,
)
from opengraph.ui.mathedit.source import SourceMap, from_source, node_text, to_source

AUTO_COMMANDS = frozenset(GREEK) | frozenset(ROOT_FUNCTIONS)
_UNDO_LIMIT = 200


@dataclass(frozen=True, slots=True)
class _Snapshot:
    source: str
    loaded: bool  # the tree is exactly from_source(source)
    root: MathList
    caret: Path


class EditorModel:
    """The tree, the caret/selection and the undo history of one expression."""

    def __init__(self, text: str = "", *, linear: bool = False) -> None:
        self.linear = linear  # plain-text editing: every character is a Char, no structures
        self.root = MathList()
        self.cursor = Position(self.root, 0)
        self.anchor: Position | None = None
        self.source = ""
        self.source_map: SourceMap = {}
        self.revision = 0  # bumped on every content change (layout cache key)
        self._loaded = True
        self._undo: list[_Snapshot] = []
        self._redo: list[_Snapshot] = []
        self._coalesce = False  # the next plain keystroke may join the last undo step
        self._mutated = False
        self._auto_converted = False
        self._command_undo: list[_Snapshot] = []
        self._typing_batch = False
        self.set_source(text, undoable=False)

    # -- content ------------------------------------------------------------------------
    @property
    def is_empty(self) -> bool:
        return not self.root.nodes

    def set_source(self, text: str, *, undoable: bool = True) -> None:
        """Replace everything with ``text`` (kept verbatim); caret at the end."""
        if undoable and text == self.source:
            return
        if undoable:
            self._push_undo()
            self._redo.clear()
        else:
            self._undo.clear()
            self._redo.clear()
        self.root, self.source_map = self._load(text)
        self.source = text
        self._loaded = True
        self.cursor = Position(self.root, len(self.root))
        self.anchor = None
        self._coalesce = False
        self.revision += 1

    def _load(self, text: str) -> tuple[MathList, SourceMap]:
        if self.linear:
            root = MathList(Char(ch) for ch in text)
            return root, {id(node): (i, i + 1) for i, node in enumerate(root.nodes)}
        return from_source(text)

    def _dump(self, root: MathList) -> tuple[str, SourceMap]:
        if self.linear:
            text = "".join(n.text for n in root.nodes if isinstance(n, Char))
            return text, {id(node): (i, i + 1) for i, node in enumerate(root.nodes)}
        return to_source(root)

    def set_linear(self, linear: bool) -> None:
        """Switch between typeset and plain-text editing. The source text is the single
        truth, so nothing is lost; the undo history is kept (each step reloads from its
        source) and the caret keeps its place in the text when going to plain text."""
        if linear == self.linear:
            return
        offset = self._caret_offset()
        self.linear = linear
        self._undo = [self._reloaded(s) for s in self._undo]
        self._redo = [self._reloaded(s) for s in self._redo]
        self.root, self.source_map = self._load(self.source)
        self._loaded = True
        self.anchor = None
        self.cursor = Position(self.root, min(offset, len(self.root)) if linear else len(self.root))
        self._coalesce = False
        self.revision += 1

    def _reloaded(self, snapshot: _Snapshot) -> _Snapshot:
        root, _ = self._load(snapshot.source)
        return _Snapshot(snapshot.source, True, root, (len(root),))

    def _caret_offset(self) -> int:
        """The caret as an index into the source text (the end of the node before it)."""
        lst, i = self.cursor.lst, self.cursor.index
        while True:
            if i > 0:
                span = self.source_map.get(id(lst.nodes[i - 1]))
                return span[1] if span else len(self.source)
            structure = lst.parent
            if structure is None or structure.parent is None:
                return 0
            span = self.source_map.get(id(structure))
            return span[0] if span else len(self.source)

    def _snapshot(self) -> _Snapshot:
        return _Snapshot(self.source, self._loaded, clone_list(self.root), path_of(self.cursor))

    def _push_undo(self) -> None:
        self._undo.append(self._snapshot())
        del self._undo[:-_UNDO_LIMIT]

    def _restore(self, snapshot: _Snapshot) -> None:
        if snapshot.loaded:
            self.root, self.source_map = self._load(snapshot.source)
            self.source = snapshot.source
        else:
            self.root = clone_list(snapshot.root)
            self.source, self.source_map = self._dump(self.root)
        self._loaded = snapshot.loaded
        self.cursor = position_at(self.root, snapshot.caret)
        self.anchor = None
        self._coalesce = False
        self.revision += 1

    def _begin(self, *, coalesce: bool = False) -> None:
        """Start an editing command (one undo step unless it continues plain typing)."""
        self._command_undo = [] if coalesce and self._coalesce else [self._snapshot()]
        self._mutated = False

    def _end(self, *, coalesce: bool = False) -> bool:
        """Finish a command; returns whether the content changed."""
        if not self._mutated:
            self._command_undo.clear()
            self._coalesce = False  # a command may have moved the caret into a field
            return False
        self._undo.extend(self._command_undo)
        del self._undo[:-_UNDO_LIMIT]
        self._command_undo.clear()
        self._redo.clear()
        self._loaded = False
        self.source, self.source_map = self._dump(self.root)
        self.revision += 1
        self._coalesce = coalesce
        return True

    def _conversion_checkpoint(self) -> None:
        """Let Undo restore a shortcut's letters; batched input stays one edit."""
        if self._typing_batch:
            return
        if self._mutated:
            source, _ = self._dump(self.root)
            snapshot = _Snapshot(source, False, clone_list(self.root), path_of(self.cursor))
        else:
            snapshot = self._snapshot()
        self._command_undo.append(snapshot)

    def undo(self) -> bool:
        if not self._undo:
            return False
        self._redo.append(self._snapshot())
        self._restore(self._undo.pop())
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        self._undo.append(self._snapshot())
        self._restore(self._redo.pop())
        return True

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    # -- low-level mutation ----------------------------------------------------------
    def _insert(self, nodes: list[Node]) -> None:
        pos = self.cursor
        pos.lst.insert(pos.index, nodes)
        self.cursor = Position(pos.lst, pos.index + len(nodes))
        self._mutated = True

    def _remove(self, lst: MathList, start: int, end: int) -> list[Node]:
        if end > start:
            self._mutated = True
        return lst.remove(start, end)

    # -- selection --------------------------------------------------------------------
    def selection(self) -> tuple[MathList, int, int] | None:
        """The selected run ``lst.nodes[start:end]``, or None when nothing is selected."""
        if self.anchor is None or self.anchor == self.cursor:
            return None
        a, b = ancestry(self.anchor), ancestry(self.cursor)
        depth = 0
        while depth + 1 < min(len(a), len(b)) and a[depth + 1][0] is b[depth + 1][0]:
            depth += 1
        lst, ia, exact_a = a[depth]
        _, ib, exact_b = b[depth]
        start = min(ia, ib)
        end = max(ia + (0 if exact_a else 1), ib + (0 if exact_b else 1))
        if start == end:
            return None
        return lst, start, end

    def has_selection(self) -> bool:
        return self.selection() is not None

    def selected_source(self) -> str:
        sel = self.selection()
        if sel is None:
            return ""
        lst, start, end = sel
        if self.linear:
            return "".join(n.text for n in lst.nodes[start:end] if isinstance(n, Char))
        return node_text(lst.nodes[start:end])

    def selected_latex(self) -> str:
        """LaTeX for the selection, or for the whole expression when nothing is selected
        (plain text is parsed first, so linear mode exports the same LaTeX)."""
        sel = self.selection()
        nodes = self.root.nodes if sel is None else sel[0].nodes[sel[1] : sel[2]]
        if self.linear:
            text = "".join(n.text for n in nodes if isinstance(n, Char))
            nodes = from_source(text)[0].nodes
        return to_latex(nodes)

    def select_all(self) -> None:
        self.anchor = Position(self.root, 0)
        self.cursor = Position(self.root, len(self.root))
        self._coalesce = False

    def set_cursor(self, pos: Position, *, select: bool = False) -> None:
        if select:
            if self.anchor is None:
                self.anchor = self.cursor
        else:
            self.anchor = None
        self.cursor = pos
        self._coalesce = False

    def select_range(self, lst: MathList, start: int, end: int) -> None:
        self.anchor = Position(lst, start)
        self.cursor = Position(lst, end)
        self._coalesce = False

    def _delete_selection(self) -> bool:
        sel = self.selection()
        self.anchor = None
        if sel is None:
            return False
        lst, start, end = sel
        self._remove(lst, start, end)
        self.cursor = Position(lst, start)
        return True

    # -- caret movement -----------------------------------------------------------------
    def move(self, direction: int, *, select: bool = False) -> None:
        """One step left (-1) or right (+1); with ``select`` extend the selection."""
        self._coalesce = False
        sel = self.selection()
        if select:
            if self.anchor is None:
                self.anchor = self.cursor
            self.cursor = self._step_over(self.cursor, direction)
            return
        self.anchor = None
        if sel is not None:
            lst, start, end = sel
            self.cursor = Position(lst, start if direction < 0 else end)
            return
        self.cursor = self._step_into(self.cursor, direction)

    def move_to_edge(self, direction: int, *, select: bool = False) -> None:
        """Home (-1) / End (+1) of the whole expression."""
        target = Position(self.root, 0 if direction < 0 else len(self.root))
        self.set_cursor(target, select=select)

    @staticmethod
    def _step_into(pos: Position, direction: int) -> Position:
        lst, i = pos.lst, pos.index
        if direction > 0 and i < len(lst):
            node = lst.nodes[i]
            if isinstance(node, Structure):
                return Position(node.blocks[0], 0)
            return Position(lst, i + 1)
        if direction < 0 and i > 0:
            node = lst.nodes[i - 1]
            if isinstance(node, Structure):
                block = node.blocks[-1]
                return Position(block, len(block))
            return Position(lst, i - 1)
        structure = lst.parent
        if structure is None or structure.parent is None:
            return pos
        b = structure.blocks.index(lst)
        if 0 <= b + direction < len(structure.blocks):
            block = structure.blocks[b + direction]
            return Position(block, 0 if direction > 0 else len(block))
        parent = structure.parent
        k = parent.index_of(structure)
        return Position(parent, k + 1 if direction > 0 else k)

    @staticmethod
    def _step_over(pos: Position, direction: int) -> Position:
        lst, i = pos.lst, pos.index
        if 0 <= i + direction <= len(lst):
            return Position(lst, i + direction)
        structure = lst.parent
        if structure is None or structure.parent is None:
            return pos
        parent = structure.parent
        k = parent.index_of(structure)
        return Position(parent, k + 1 if direction > 0 else k)

    def vertical_target(self, direction: int) -> MathList | None:
        """The list Up (-1) / Down (+1) should move into, or None to leave the row."""
        lst = self.cursor.lst
        while lst.parent is not None:
            structure = lst.parent
            if isinstance(structure, Frac):
                if lst is structure.num and direction > 0:
                    return structure.den
                if lst is structure.den and direction < 0:
                    return structure.num
            elif isinstance(structure, BigOp) and structure.kind != "limit":
                below: tuple[MathList, ...] = (structure.lower,)
                if structure.kind != "int":
                    below = (structure.var, structure.lower)
                if lst is structure.upper and direction > 0:
                    return below[0] if below[0].nodes else structure.lower
                if any(lst is b for b in below) and direction < 0:
                    return structure.upper
            elif (isinstance(structure, Sup) and direction > 0) or (
                isinstance(structure, Sub) and direction < 0
            ):
                return structure.parent
            assert structure.parent is not None
            lst = structure.parent
        return None

    def tab(self, direction: int = 1) -> bool:
        """Complete a big-operator name or visit the next field; False leaves focus."""
        if self.linear:
            return False
        if direction > 0 and not self.has_selection() and self._big_operator_name() is not None:
            self.type_text("\t")
            return True
        lst = self.cursor.lst
        structure = lst.parent
        if structure is None or structure.parent is None:
            return False
        b = structure.blocks.index(lst) + direction
        if 0 <= b < len(structure.blocks):
            target = structure.blocks[b]
            self.set_cursor(Position(target, 0 if direction > 0 else len(target)))
        else:
            parent = structure.parent
            i = parent.index_of(structure)
            self.set_cursor(Position(parent, i + (1 if direction > 0 else 0)))
        return True

    # -- typing ------------------------------------------------------------------------
    def type_text(self, text: str) -> bool:
        """Type ``text`` at the caret as if keyed in; returns whether content changed."""
        # Runs of ordinary characters form one undo step, like a line edit.
        plain = len(text) == 1 and text not in "^_/÷([{" and not self.has_selection()
        self._begin(coalesce=plain)
        self._auto_converted = False
        self._typing_batch = len(text) > 1
        try:
            for ch in text:
                self._type_char(ch)
        finally:
            self._typing_batch = False
        return self._end(coalesce=plain and not self._auto_converted)

    def _type_char(self, ch: str) -> None:
        if ch in "\r\n\t":
            ch = " "
        if not ch.isprintable():
            return
        sel = self.selection()
        if self.linear:
            self._delete_selection()
            self._insert([Char(ch)])
            return
        if sel is None and ch.isdigit() and self._continue_scientific_number(ch):
            return
        if ch == " " and self._leave_generator_keyword():
            return
        if ch == " " and sel is None and self._operation_clause():
            return
        if sel is None and not ch.isalpha() and ch != "\\" and self._latex_command(ch):
            return
        if sel is None and not ch.isalnum() and ch != "\\":
            self._auto_command()
        if sel is None and self._leave_completed_structure(ch):
            return
        if ch == "^":
            self._insert_script(Sup, sel)
        elif ch == "_":
            self._delete_selection()
            self._insert_script(Sub, None)
        elif ch in "/÷":
            if sel is None and self._leave_subscript(ch):
                return
            self._insert_frac(sel)
        elif ch in BRACKETS and sel is not None:
            lst, start, end = sel
            self.anchor = None
            paren = Paren(ch, MathList(self._remove(lst, start, end)))
            self.cursor = Position(lst, start)
            self._insert([paren])
        else:
            self._delete_selection()
            if self._leave_subscript(ch):
                return
            if ch in BRACKETS:
                self._open_bracket(ch)
            elif ch in CLOSERS:
                self._close_bracket(ch)
            elif ch == "," and self._next_argument():
                pass
            elif ch == " " and self._big_operator_name() is not None:
                self._conversion_checkpoint()
                self._big_operator(start_in_body=False)
            else:
                self._insert_char(ch)

    def _continue_scientific_number(self, digit: str) -> bool:
        """An e followed by digits extends a numeric token, rather than a product.

        At the preceding e the text grammar is still ambiguous: 2^1e can mean
        (2^1)*e. Delay the choice until a digit arrives, then join the numeric exponent
        back into the fraction/exponent field. An optional sign is handled identically.
        """
        lst, end = self.cursor.lst, self.cursor.index
        start = end - 1
        previous = lst.nodes[start] if start >= 0 else None
        if isinstance(previous, Char) and previous.text in "+-":
            start -= 1
        if start < 1:
            return False
        marker = lst.nodes[start]
        if not isinstance(marker, Char) or marker.text not in ("e", "E"):
            return False
        structure = lst.nodes[start - 1]
        if not isinstance(structure, Sup | Frac):
            return False
        field = structure.body if isinstance(structure, Sup) else structure.den
        if not all(isinstance(n, Char) for n in field.nodes):
            return False
        mantissa = "".join(n.text for n in field.nodes if isinstance(n, Char))
        if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", mantissa):
            return False
        suffix = self._remove(lst, start, end)
        self.cursor = Position(field, len(field))
        self._insert([*suffix, Char(digit)])
        return True

    def _insert_char(self, ch: str) -> None:
        lst, i = self.cursor.lst, self.cursor.index
        if ch == " " and i >= 2:
            start = i
            while start > 0 and is_letter(lst.nodes[start - 1]):
                start -= 1
            word = "".join(n.text for n in lst.nodes[start:i] if isinstance(n, Char))
            if word == "in":
                self._remove(lst, start, i)
                self.cursor = Position(lst, start)
                self._insert([Char("in"), Char(" ")])
                return
        previous = lst.nodes[i - 1] if i > 0 else None
        if ch == "=" and isinstance(previous, Char) and previous.text in ("<", ">"):
            self._remove(lst, i - 1, i)
            self.cursor = Position(lst, i - 1)
            self._insert([Char(previous.text + "=")])
            return
        self._insert([Char(ch)])

    def _leave_generator_keyword(self) -> bool:
        """In ``seq[k^2 for …]`` or ``sum(k^2 for …)``, 'for' ends the body, including a
        script/fraction; inside a Σ/Π/∫/lim, so do 'from' and 'to' (``from 2^3 to …``)."""
        lst = self.cursor.lst
        structure = lst.parent
        if self.cursor.index != len(lst) or not isinstance(structure, Sup | Frac):
            return False
        words = ("for", "from", "to") if self._in_big_operator(lst) else ("for",)
        word = next(
            (
                w
                for w in words
                if len(lst.nodes) > len(w) + 1
                and all(isinstance(node, Char) for node in lst.nodes[-len(w) - 1 :])
                and "".join(n.text for n in lst.nodes[-len(w) - 1 :] if isinstance(n, Char))
                == " " + w
            ),
            None,
        )
        if word is None:
            return False
        parent = structure.parent
        assert parent is not None
        self._remove(lst, len(lst.nodes) - len(word) - 1, len(lst.nodes))
        self.cursor = Position(parent, parent.index_of(structure) + 1)
        for ch in f" {word} ":
            self._type_char(ch)
        return True

    @staticmethod
    def _in_big_operator(lst: MathList) -> bool:
        structure = lst.parent
        while structure is not None:
            if isinstance(structure, BigOp):
                return True
            outer = structure.parent
            structure = outer.parent if outer is not None else None
        return False

    def _operation_clause(self) -> bool:
        """In a Σ/Π/∫/lim template, a clause word followed by a space moves on to its
        field, so typing ``sum(k^2 for k from 1 to 10)`` in reading order fills the body,
        variable, lower and upper bound (``for x to 0 from right`` for a limit). The
        word is only a field change when it stands alone at the end of its field and the
        field it leads to is still empty."""
        lst, i = self.cursor.lst, self.cursor.index
        op = lst.parent
        if not isinstance(op, BigOp) or i != len(lst):
            return False
        start = i
        while start > 0 and is_letter(lst.nodes[start - 1]):
            start -= 1
        word = "".join(n.text for n in lst.nodes[start:i] if isinstance(n, Char))
        before = lst.nodes[start - 1] if start > 0 else None
        if start > 0 and not (isinstance(before, Char) and before.text == " "):
            return False
        limit = op.kind == "limit"
        targets = {
            "for": op.var if lst is op.body else None,
            "from": None if lst is op.body else (op.upper if limit else op.lower),
            "to": None if lst is op.body else (op.lower if limit else op.upper),
        }
        target = targets.get(word)
        if target is None or target is lst or target.nodes:
            return False
        if lst is not op.body and word == "for":
            return False
        while start > 0 and _is_space(lst.nodes[start - 1]):
            start -= 1
        self._remove(lst, start, i)
        self.cursor = Position(target, 0)
        return True

    def _leave_subscript(self, ch: str) -> bool:
        """A subscript is part of a name: anything but letters and digits ends it.

        Returns True when ``ch`` was consumed (typed after the subscript, or a space).
        """
        structure = self.cursor.lst.parent
        if not isinstance(structure, Sub) or ch.isalnum():
            return False
        parent = structure.parent
        assert parent is not None
        self.cursor = Position(parent, parent.index_of(structure) + 1)
        if ch != " ":
            self._type_char(ch)
        return True

    def _leave_completed_structure(self, ch: str) -> bool:
        """Leave a completed exponent or fraction for a top-level operator.

        Math input should not require an arrow-key detour for the overwhelmingly common
        ``x^2+1`` and ``1/2+x`` cases. Restrict this to a field's end and to operators
        that unambiguously begin the surrounding expression; multiplication and a new
        exponent remain available inside a field when deliberately entered there.
        """
        lst = self.cursor.lst
        structure = lst.parent
        if self.cursor.index != len(lst) or not lst.nodes or not isinstance(structure, Sup | Frac):
            return False
        if isinstance(structure, Frac) and lst is not structure.den:
            return False
        previous = lst.nodes[-1]
        text = previous.text if isinstance(previous, Char) else ""
        exits = ch in "+-=<>,*/ " or (ch == "!" and isinstance(structure, Sup))
        # Adjacent identifier letters/digits stay in the same token. A new token is
        # implicit multiplication and belongs outside the exponent/denominator.
        if ch.isalpha():
            exits |= text[-1:].isdigit() or isinstance(previous, (Symbol, Paren, Root, Sup))
        elif ch.isdigit():
            exits |= bool(text[-1:].isalpha()) or isinstance(previous, (Symbol, Paren, Root, Sup))
        elif ch in "({":
            exits |= not (isinstance(previous, Char) and text[-1:].isalpha())
        if not exits:
            return False
        parent = structure.parent
        assert parent is not None
        self.cursor = Position(parent, parent.index_of(structure) + 1)
        self._type_char(ch)
        return True

    def _open_bracket(self, ch: str) -> None:
        structure = self.cursor.lst.parent
        if ch == "(" and isinstance(structure, Root) and not self.cursor.lst.nodes:
            structure.swallowed_paren = True  # "sqrt(" — the radical already groups
            return
        if ch == "(" and (self._enter_call_mode() or self._big_operator(start_in_body=True)):
            return
        paren = Paren(ch, MathList())
        self._insert([paren])
        self.cursor = Position(paren.body, 0)

    def _close_bracket(self, ch: str) -> None:
        lst, i = self.cursor.lst, self.cursor.index
        opener = CLOSERS[ch]
        start = _unmatched_opener(lst, i, opener)
        if start is not None:  # "(x" loaded from text, then ")": make it a pair
            opening = lst.nodes[start]
            assert isinstance(opening, Char)
            content = self._remove(lst, start + 1, i)
            self._remove(lst, start, start + 1)
            self.cursor = Position(lst, start)
            self._insert([Paren(opening.text, MathList(content), ch)])
            return
        block = lst
        while block.parent is not None and block.parent.parent is not None:
            structure, parent = block.parent, block.parent.parent
            if isinstance(structure, Paren) and (
                structure.left == opener or (structure.left in "([" and ch in ")]")
            ):
                if structure.right != ch:
                    structure.right = ch
                    self._mutated = True
                break
            if isinstance(structure, Root) and structure.swallowed_paren and ch == ")":
                structure.swallowed_paren = False
                break
            if isinstance(structure, BigOp) and ch == ")":
                self._close_big_operator(structure)
                break
            block = parent
        else:
            self._insert_char(ch)  # nothing to close
            return
        k = parent.index_of(structure)
        if block is lst and i < len(lst) and isinstance(structure, Paren):  # "(a|b" + ")"
            parent.insert(k + 1, self._remove(lst, i, len(lst)))
        self.cursor = Position(parent, k + 1)

    def _big_operator_name(self) -> tuple[str, int] | None:
        """The big-operator name just before the caret and where it starts."""
        if self._has_name_suffix():
            return None
        lst, i = self.cursor.lst, self.cursor.index
        start = i
        previous = lst.nodes[i - 1] if i else None
        if isinstance(previous, Symbol):
            name = previous.name
            start = i - 1
        elif isinstance(previous, Char) and previous.text == "∫":
            name = "∫"
            start = i - 1
        else:
            while start > 0 and is_letter(lst.nodes[start - 1]):
                start -= 1
            name = "".join(n.text for n in lst.nodes[start:i] if isinstance(n, Char))
        return (name, start) if name in BIG_OPERATORS else None

    def _big_operator(self, *, start_in_body: bool) -> bool:
        """``sum``, ``Σ``, ``limit``… before the caret: replace the name with a big
        operator with empty fields. The caret goes to the first field in reading order, or
        to the body when ``start_in_body`` (the call form ``sum(``). Ctrl+Z gives the name
        back."""
        found = self._big_operator_name()
        if found is None:
            return False
        name, start = found
        lst = self.cursor.lst
        self._remove(lst, start, self.cursor.index)
        self.cursor = Position(lst, start)
        op = BigOp(name, MathList(), MathList(), MathList(), MathList())
        self._insert([op])
        self.cursor = Position(op.body if start_in_body else op.blocks[0], 0)
        self._auto_converted = True
        return True

    def _enter_call_mode(self) -> bool:
        """``(`` typed in the first field of a fresh, empty big operator (``sum`` then
        ``(``): the user is writing the call, so move on to the body, which comes first
        in the call's arguments."""
        lst = self.cursor.lst
        op = lst.parent
        if not isinstance(op, BigOp) or lst is not op.blocks[0] or lst is op.body:
            return False
        if any(block.nodes for block in op.blocks):
            return False
        self.cursor = Position(op.body, 0)
        return True

    def _next_argument(self) -> bool:
        """``,`` in a big operator's slot moves on to its next argument, also from inside
        an exponent or fraction there (not from brackets, where commas make points)."""
        lst = self.cursor.lst
        structure = lst.parent
        while isinstance(structure, Frac | Sup | Sub | Root) and structure.parent is not None:
            lst = structure.parent
            structure = lst.parent
        if not isinstance(structure, BigOp):
            return False
        arguments = structure.arguments
        k = next(j for j, arg in enumerate(arguments) if arg is lst)
        if k + 1 >= len(arguments):
            return False
        self.cursor = Position(arguments[k + 1], 0)
        return True

    def _close_big_operator(self, op: BigOp) -> None:
        """Typed with three arguments, ``Σ(k^2, 1, n)`` filled the variable and lower
        slots with the bounds: move them to the bounds (the variable slot stays empty)."""
        if op.kind == "limit" or op.upper.nodes or not (op.var.nodes and op.lower.nodes):
            return
        op.upper.insert(0, op.lower.remove(0, len(op.lower)))
        op.lower.insert(0, op.var.remove(0, len(op.var)))
        self._mutated = True

    def _latex_command(self, terminator: str) -> bool:
        """``\name`` before the caret, ended by ``terminator``: replace it with what the
        name stands for (``\\pi``, ``\frac``, ``\\sqrt``, ``\\sum``, ``\\le``, ``\\sin``…).
        A space just ends the command; any other terminator is typed afterwards. Unknown
        names stay as typed. Ctrl+Z gives the typed text back."""
        lst, i = self.cursor.lst, self.cursor.index
        start = i
        while start > 0 and is_letter(lst.nodes[start - 1]):
            start -= 1
        before = lst.nodes[start - 1] if start > 0 else None
        if start == i or not (isinstance(before, Char) and before.text == "\\"):
            return False
        name = "".join(n.text for n in lst.nodes[start:i] if isinstance(n, Char))
        if name not in COMMANDS:
            return False
        self._conversion_checkpoint()
        self._remove(lst, start - 1, i)
        self.cursor = Position(lst, start - 1)
        self._auto_converted = True
        self._insert_command(name)
        if terminator != " ":
            self._type_char(terminator)
        return True

    def _insert_command(self, name: str) -> None:
        if name in GREEK:
            self._insert([Symbol(name)])
        elif name in TEXT_COMMANDS:
            self._insert([Char(TEXT_COMMANDS[name])])
        elif name in ROOT_FUNCTIONS:
            root = Root(name, MathList())
            self._insert([root])
            self.cursor = Position(root.body, 0)
        elif name in ("frac", "dfrac", "tfrac"):
            frac = Frac(MathList(), MathList())
            self._insert([frac])
            self.cursor = Position(frac.num, 0)
        elif name in ("sum", "prod", "int", "lim", "limit"):
            op = BigOp("limit" if name == "lim" else name, *(MathList() for _ in range(4)))
            self._insert([op])
            self.cursor = Position(op.blocks[0], 0)
        else:  # a function name: plain letters, which the layout draws upright
            self._insert([Char(c) for c in name])

    def _auto_command(self) -> None:
        """Convert a just-typed ``sqrt``/``cbrt``/Greek name before the caret."""
        if self._has_name_suffix():
            return
        lst, i = self.cursor.lst, self.cursor.index
        start = i
        while start > 0 and is_letter(lst.nodes[start - 1]):
            start -= 1
        before = lst.nodes[start - 1] if start > 0 else None
        if isinstance(before, Char) and before.text == "\\":
            return  # "\pi": a LaTeX-style command, converted when it is ended
        word = "".join(n.text for n in lst.nodes[start:i] if isinstance(n, Char))
        match = word if word in AUTO_COMMANDS else None
        if match is None:
            return
        self._conversion_checkpoint()
        self._remove(lst, i - len(match), i)
        self.cursor = Position(lst, i - len(match))
        self._auto_converted = True
        if match in ROOT_FUNCTIONS:
            root = Root(match, MathList())
            self._insert([root])
            self.cursor = Position(root.body, 0)
        else:
            self._insert([Symbol(match)])

    def _has_name_suffix(self) -> bool:
        """A shortcut before the caret must not consume the start of an existing name."""
        lst, i = self.cursor.lst, self.cursor.index
        if i == len(lst):
            return False
        node = lst.nodes[i]
        return isinstance(node, Sub) or (
            isinstance(node, Char) and (node.text.isalnum() or node.text == "_")
        )

    def _insert_script(
        self, kind: type[Sup] | type[Sub], sel: tuple[MathList, int, int] | None
    ) -> None:
        if sel is not None:
            lst, start, end = sel
            self.anchor = None
            content = self._remove(lst, start, end)
            script = kind(MathList(content))
            self.cursor = Position(lst, start)
            self._insert([script])
            self.cursor = Position(script.body, len(script.body))
            return
        lst, i = self.cursor.lst, self.cursor.index
        for neighbour in (lst.nodes[i] if i < len(lst) else None, lst.nodes[i - 1] if i else None):
            if isinstance(neighbour, kind):  # never stack a second exponent: extend it
                self.cursor = Position(neighbour.body, len(neighbour.body))
                return
        script = kind(MathList())
        self._insert([script])
        self.cursor = Position(script.body, 0)

    def _insert_frac(self, sel: tuple[MathList, int, int] | None) -> None:
        if sel is not None:
            lst, start, end = sel
            self.anchor = None
        else:
            lst, end = self.cursor.lst, self.cursor.index
            start = _operand_start(lst, end)
        numerator = self._remove(lst, start, end)
        if (
            len(numerator) >= 2
            and isinstance(numerator[0], Char)
            and numerator[0].text == "("
            and _matching_close(numerator, 0) == len(numerator) - 1
        ):
            numerator = numerator[1:-1]  # the fraction bar groups already
        elif (
            len(numerator) == 1
            and isinstance(numerator[0], Paren)
            and numerator[0].left != "["
            and numerator[0].right == BRACKETS[numerator[0].left]
        ):
            body = numerator[0].body
            numerator = body.remove(0, len(body))
        frac = Frac(MathList(numerator), MathList())
        self.cursor = Position(lst, start)
        self._insert([frac])
        target = frac.den if numerator else frac.num
        self.cursor = Position(target, 0)

    # -- paste ---------------------------------------------------------------------------
    def paste(self, text: str) -> bool:
        text = " ".join(text.splitlines()).strip()
        text = latex_to_source(text) or text  # pasted LaTeX becomes OpenGraph source
        if not text:
            return False
        self._begin()
        self._delete_selection()
        tree, _ = self._load(text)
        self._insert(tree.remove(0, len(tree)))
        return self._end()

    # -- deletion ----------------------------------------------------------------------
    def backspace(self) -> bool:
        return self._delete(-1)

    def delete_forward(self) -> bool:
        return self._delete(1)

    def delete_selection(self) -> bool:
        self._begin()
        self._delete_selection()
        return self._end()

    def _delete(self, direction: int) -> bool:
        self._begin()
        if not self._delete_selection():
            self._delete_step(direction)
        return self._end()

    def _delete_step(self, direction: int) -> None:
        lst, i = self.cursor.lst, self.cursor.index
        j = i - 1 if direction < 0 else i
        if 0 <= j < len(lst):
            node = lst.nodes[j]
            if isinstance(node, Structure) and any(b.nodes for b in node.blocks):
                block = node.blocks[-1] if direction < 0 else node.blocks[0]
                self.cursor = Position(block, len(block) if direction < 0 else 0)
                return
            self._remove(lst, j, j + 1)
            self.cursor = Position(lst, j)
            return
        structure = lst.parent
        if structure is None or structure.parent is None:
            return
        # At the edge of a block: dissolve the structure, keeping its contents.
        parent = structure.parent
        k = parent.index_of(structure)
        if isinstance(structure, BigOp):
            self._big_operator_to_text(structure, lst, direction)
            return
        b = structure.blocks.index(lst)
        offset = sum(len(blk) for blk in structure.blocks[: b if direction < 0 else b + 1])
        contents = [n for blk in structure.blocks for n in blk.remove(0, len(blk))]
        self._remove(parent, k, k + 1)
        parent.insert(k, contents)
        self.cursor = Position(parent, k + offset)
        self._mutated = True

    def _big_operator_to_text(self, op: BigOp, block: MathList, direction: int) -> None:
        """Dissolve a big operator into the call it stands for, ``sum(body, k, a, b)``,
        with the caret where it was (at the start or end of ``block``'s arguments)."""
        parent = op.parent
        assert parent is not None
        k = parent.index_of(op)
        name: list[Node] = (
            [Symbol(op.function)] if op.function in GREEK else [Char(c) for c in op.function]
        )
        inside: list[Node] = []
        caret = 0
        for arg in op.arguments:
            if not arg.nodes and arg is not op.body and arg is not block:
                continue  # an unused variable or side
            if arg is not op.body:
                inside.append(Char(","))
            if arg is block:
                caret = len(inside) + (len(arg) if direction > 0 else 0)
            inside.extend(arg.remove(0, len(arg)))
        paren = Paren("(", MathList(inside))
        self._remove(parent, k, k + 1)
        parent.insert(k, [*name, paren])
        self.cursor = Position(paren.body, caret)
        self._mutated = True

    # -- debugging / tests -------------------------------------------------------------
    def describe(self) -> str:
        """LaTeX-like rendering with ``|`` at the caret and ``[...]`` around a selection."""
        sel = self.selection()
        return _describe(self.root, self.cursor if sel is None else None, sel)


def _unmatched_opener(lst: MathList, end: int, opener: str) -> int | None:
    """Last unmatched opener before ``end``; round/square brackets may mix."""
    depth = 0
    openers = "([" if opener in "([" else opener
    closers = ")]" if opener in "([" else BRACKETS[opener]
    for k in range(end - 1, -1, -1):
        node = lst.nodes[k]
        if isinstance(node, Char) and node.text in closers:
            depth += 1
        elif isinstance(node, Char) and node.text in openers:
            if depth == 0:
                return k
            depth -= 1
    return None


def _matching_close(nodes: list[Node], start: int) -> int | None:
    depth = 0
    for i in range(start, len(nodes)):
        node = nodes[i]
        if isinstance(node, Char) and node.text == "(":
            depth += 1
        elif isinstance(node, Char) and node.text == ")":
            depth -= 1
            if depth == 0:
                return i
    return None


def _operand_start(lst: MathList, end: int) -> int:
    """Start of the operand ending at ``end``: what typing ``/`` puts in the numerator.

    Digits, letters, structures and a parenthesised group (with a function name before
    it) belong to it; an operator, comma, space or unmatched bracket ends it.
    """
    j = end
    while j > 0:
        node = lst.nodes[j - 1]
        if not isinstance(node, Char):
            j -= 1
            continue
        text = node.text
        if text == ")":
            depth = 0
            k = j - 1
            while k >= 0:
                n = lst.nodes[k]
                if isinstance(n, Char) and n.text == ")":
                    depth += 1
                elif isinstance(n, Char) and n.text == "(":
                    depth -= 1
                    if depth == 0:
                        break
                k -= 1
            if k < 0:
                break
            j = k
        elif text.isalnum() or text == ".":
            j -= 1
        else:
            break
    return j


def _describe(lst: MathList, caret: Position | None, sel: tuple[MathList, int, int] | None) -> str:
    parts: list[str] = []
    for i in range(len(lst) + 1):
        if sel is not None and sel[0] is lst and i == sel[2]:
            parts.append("]")
        if sel is not None and sel[0] is lst and i == sel[1]:
            parts.append("[")
        if caret is not None and caret.lst is lst and caret.index == i:
            parts.append("|")
        if i == len(lst):
            break
        node = lst.nodes[i]
        match node:
            case Char(text=text):
                parts.append(text)
            case Symbol(name=name):
                parts.append(f"\\{name} ")
            case Frac():
                num, den = (_describe(b, caret, sel) for b in node.blocks)
                parts.append(f"\\frac{{{num}}}{{{den}}}")
            case Sup():
                parts.append(f"^{{{_describe(node.body, caret, sel)}}}")
            case Sub():
                parts.append(f"_{{{_describe(node.body, caret, sel)}}}")
            case Root():
                index = "" if node.index is None else f"[{node.index}]"
                parts.append(f"\\sqrt{index}{{{_describe(node.body, caret, sel)}}}")
            case Paren():
                body = _describe(node.body, caret, sel)
                parts.append(f"\\left{node.left}{body}\\right{node.right}")
            case BigOp():
                body, var, lower, upper = (_describe(b, caret, sel) for b in node.arguments)
                if node.kind == "limit":
                    parts.append(f"\\lim_{{{var}\\to {lower}^{{{upper}}}}}{{{body}}}")
                elif node.kind == "int":
                    parts.append(f"\\int_{{{lower}}}^{{{upper}}}{{{body}}}d{{{var}}}")
                else:
                    parts.append(f"\\{node.kind}_{{{var}={lower}}}^{{{upper}}}{{{body}}}")
    return "".join(parts)


def _is_space(node: Node) -> bool:
    return isinstance(node, Char) and node.text == " "
