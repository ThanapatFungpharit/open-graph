"""LaTeX in and out of the editor, for copy/paste with other maths software. Toolkit-free.

* :func:`to_latex` writes a run of tree nodes as LaTeX (``\\frac``, ``\\sqrt``, ``\\sum``…).
  It is for display and sharing only: the *source text* stays the editor's truth.
* :func:`latex_to_source` reads the common subset of LaTeX (fractions, roots, scripts,
  Greek letters, ``\\left…\\right``, big operators, relations) and returns source text in
  OpenGraph's own syntax, or None when the input uses something it does not know, in
  which case the caller pastes the text unchanged.
* :data:`COMMANDS` maps the names typed after a backslash (``\\pi``, ``\\frac``) to what
  the editor inserts.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from opengraph.engine.builtins import CAS_OPERATIONS, FUNCTIONS
from opengraph.ui.mathedit.nodes import (
    GREEK,
    BigOp,
    Char,
    Frac,
    MathList,
    Node,
    Paren,
    Root,
    Sub,
    Sup,
    Symbol,
    is_letter,
)

# Functions LaTeX has a command for; everything else is wrapped in \operatorname.
_LATEX_FUNCTIONS = frozenset(
    "sin cos tan cot sec csc sinh cosh tanh coth arcsin arccos arctan ln log exp min max "
    "arg det gcd lim".split()
)
_FUNCTION_NAMES = sorted(
    {n for n in [*FUNCTIONS, *CAS_OPERATIONS] if len(n) > 1 and n.isascii() and n.isalpha()},
    key=len,
    reverse=True,
)
# Names typed after "\" that insert plain text, and the single-character operators.
TEXT_COMMANDS = {
    "le": "<=",
    "leq": "<=",
    "ge": ">=",
    "geq": ">=",
    "cdot": "*",
    "times": "*",
    "ast": "*",
    "in": "∈",
}
STRUCTURE_COMMANDS = frozenset(
    {"frac", "dfrac", "tfrac", "sqrt", "cbrt", "sum", "prod", "int", "lim", "limit"}
)
COMMANDS = (
    frozenset(GREEK)
    | frozenset(TEXT_COMMANDS)
    | STRUCTURE_COMMANDS
    | frozenset(_LATEX_FUNCTIONS - {"lim"})
    | frozenset(n for n in _FUNCTION_NAMES if n not in ("sum", "prod", "int", "limit"))
)
"""Everything that can follow a backslash in the editor."""

_OPEN_LATEX = {"(": "(", "[": "[", "{": r"\{"}
_CLOSE_LATEX = {")": ")", "]": "]", "}": r"\}"}


# -- tree -> LaTeX -------------------------------------------------------------------------


def to_latex(nodes: Sequence[Node]) -> str:
    """LaTeX for a run of nodes (an empty field is written as ``{}``)."""
    out: list[str] = []
    i = 0
    while i < len(nodes):
        node = nodes[i]
        if is_letter(node):
            j = i
            while j < len(nodes) and is_letter(nodes[j]):
                j += 1
            number_set = _number_set(nodes[i:j], nodes[j] if j < len(nodes) else None)
            if number_set is not None:  # sets[N] is \mathbb{N}
                out.append(number_set)
                i = j + 1
                continue
            out.append(_letters(nodes[i:j]))
            i = j
            continue
        out.append(_node(node))
        i += 1
    text = ""
    for part in out:
        if not part:
            continue
        text += part if not text or part[0] in "^_" else " " + part
    return text


def _number_set(run: Sequence[Node], after: Node | None) -> str | None:
    r"""``\mathbb{N}`` / ``\mathbb{Z}`` for the letters ``sets`` followed by ``[N]`` / ``[Z]``."""
    word = "".join(n.text for n in run if isinstance(n, Char))
    if word != "sets" or not isinstance(after, Paren) or after.left != "[":
        return None
    inner = "".join(n.text for n in after.body.nodes if isinstance(n, Char)).strip()
    if inner not in ("N", "Z") or not all(isinstance(n, Char) for n in after.body.nodes):
        return None
    return rf"\mathbb{{{inner}}}"


def _letters(run: Sequence[Node]) -> str:
    word = "".join(n.text for n in run if isinstance(n, Char))
    if word == "in":
        return r"\in"
    if word in _LATEX_FUNCTIONS:
        return f"\\{word}"
    # A word is one identifier. Splitting it into LaTeX letters used to turn spin
    # into s*p*i*n on paste, independently of the editor's Greek conversion.
    return rf"\operatorname{{{word}}}" if len(word) > 1 else word


def _group(lst: MathList) -> str:
    return "{" + to_latex(lst.nodes) + "}"


def _node(node: Node) -> str:
    match node:
        case Char(text=text):
            return {
                "*": r"\cdot",
                "<=": r"\le",
                ">=": r"\ge",
                " ": "",
                "%": r"\%",
                "in": r"\in",
                "∈": r"\in",
                "ℕ": r"\mathbb{N}",
                "ℤ": r"\mathbb{Z}",
            }.get(text, text)
        case Symbol(name=name):
            return f"\\{name}"
        case Frac():
            return rf"\frac{_group(node.num)}{_group(node.den)}"
        case Sup():
            return f"^{_group(node.body)}"
        case Sub():
            return f"_{_group(node.body)}"
        case Root():
            if node.index is None:
                return rf"\sqrt{_group(node.body)}"
            return rf"\sqrt[{node.index}]{_group(node.body)}"
        case Paren():
            return (
                rf"\left{_OPEN_LATEX[node.left]} {to_latex(node.body.nodes)} "
                rf"\right{_CLOSE_LATEX[node.right]}"
            )
        case BigOp():
            return _big_operator(node)
    raise TypeError(node)  # pragma: no cover


def _big_operator(op: BigOp) -> str:
    body = to_latex(op.body.nodes)
    var, lower, upper = (to_latex(b.nodes) for b in (op.var, op.lower, op.upper))
    if op.kind == "int":
        bounds = (f"_{{{lower}}}" if lower else "") + (f"^{{{upper}}}" if upper else "")
        tail = rf"\, d{var}" if var else r"\, dx"
        return rf"\int{bounds} {body}{tail}"
    if op.kind == "limit":
        sign = {"left": "^-", "right": "^+"}.get(upper, "")
        return rf"\lim_{{{var} \to {lower}{sign}}} {body}"
    glyph = r"\sum" if op.kind == "sum" else r"\prod"
    start = f"{var}={lower}" if var else lower
    return rf"{glyph}_{{{start}}}^{{{upper}}} {body}"


# -- LaTeX -> source -----------------------------------------------------------------------

_TOKEN = re.compile(
    r"\s+|\\[A-Za-z]+|\\[^A-Za-z]|[{}^_&]|\d+(?:\.\d+)?|[A-Za-z]|[^\sA-Za-z\d\\{}^_&]"
)
_IGNORED = frozenset(
    [
        "\\,",
        "\\;",
        "\\:",
        "\\!",
        "\\ ",
        r"\quad",
        r"\qquad",
        r"\displaystyle",
        r"\textstyle",
        r"\limits",
        r"\nolimits",
    ]
)
_SYMBOLS = {
    r"\cdot": "*",
    r"\times": "*",
    r"\ast": "*",
    r"\div": "/",
    r"\le": "<=",
    r"\leq": "<=",
    r"\ge": ">=",
    r"\geq": ">=",
    r"\{": "{",
    r"\}": "}",
    r"\|": "|",
    r"\ldots": "...",
    r"\cdots": "...",
    r"\infty": "inf",
    r"\to": r"\to",
    r"\rightarrow": r"\to",
    r"\in": "∈",
}
_ADDITIVE = frozenset("+-=<>,") | {"<=", ">="}
_OPERATOR_TYPES = ("\\sum", "\\prod", "\\int", "\\lim")


class _Unsupported(Exception):
    pass


class _Reader:
    def __init__(self, text: str) -> None:
        tokens = _TOKEN.findall(text)
        if "".join(tokens) != text:
            raise _Unsupported
        self.tokens = [t for t in tokens if not t.isspace() and t not in _IGNORED]
        self.pos = 0

    def peek(self) -> str | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def take(self) -> str:
        token = self.peek()
        if token is None:
            raise _Unsupported
        self.pos += 1
        return token

    def expect(self, token: str) -> None:
        if self.take() != token:
            raise _Unsupported

    # A "group" is the contents of {...}; an "atom" is one token or one braced group.
    def group(self) -> str:
        """Source text for the contents of the next ``{…}`` (or single token)."""
        if self.peek() != "{":
            return self.atom_text()
        self.take()
        text = self.sequence(until="}")
        self.expect("}")
        return text

    def sequence(self, until: str | None) -> str:
        items: list[str] = []
        while self.peek() is not None and self.peek() != until:
            items.append(self.item())
        return _join(items)

    def atom_text(self) -> str:
        token = self.peek()
        if token == "{":
            return "(" + self.group() + ")"
        return self.item(scripts=False).strip("\x00")

    def item(self, *, scripts: bool = True) -> str:
        token = self.take()
        text = self.leaf(token)
        if scripts and self.peek() in ("^", "_") and text.startswith("\x00"):
            text = "(" + text.strip("\x00") + ")"
        while scripts and self.peek() in ("^", "_"):
            mark = self.take()
            arg = self.group()
            text += mark + (arg if _single_token(arg) else f"({arg})")
        return text

    def leaf(self, token: str) -> str:
        if token == "{":
            self.pos -= 1
            return "(" + self.group() + ")"
        if token in _SYMBOLS:
            return _SYMBOLS[token]
        if token.startswith("\\"):
            return self.command(token)
        if token in ("^", "_", "&", "}"):
            raise _Unsupported
        return token

    def command(self, token: str) -> str:
        name = token[1:]
        if name in ("left", "right"):
            delimiter = self.take()
            return "" if delimiter == "." else _SYMBOLS.get(delimiter, delimiter)
        if name in ("frac", "dfrac", "tfrac"):
            num, den = self.group(), self.group()
            return "\x00" + _paren(num) + "/" + _paren(den) + "\x00"
        if name == "sqrt":
            if self.peek() == "[":
                self.take()
                index = ""
                while self.peek() not in ("]", None):
                    index += self.take()
                self.expect("]")
                body = self.group()
                return f"cbrt({body})" if index == "3" else f"({body})^(1/{index})"
            return f"sqrt({self.group()})"
        if name in ("mathrm", "text", "operatorname", "mathit", "mathbf"):
            text = self.group()
            word = text.replace(" ", "")
            return word if name != "text" and word.isalpha() else text
        if name == "mathbb":
            domain = self.group()
            if domain not in ("N", "Z"):
                raise _Unsupported
            return f"sets[{domain}]"
        if token in _OPERATOR_TYPES:
            return self.big_operator(name)
        if name in GREEK:
            return name
        if name in FUNCTIONS or name in CAS_OPERATIONS or name in _LATEX_FUNCTIONS:
            if self.peek() in (None, "(", r"\left", "}", "^", "_", *_ADDITIVE_TOKENS):
                return name
            return f"{name}({self.item()})"  # "\sin x": the language needs the brackets
        raise _Unsupported

    def bounds(self) -> tuple[str, str]:
        lower = upper = ""
        for _ in range(2):
            if self.peek() == "_":
                self.take()
                lower = self.group()
            elif self.peek() == "^":
                self.take()
                upper = self.group()
        return lower, upper

    def body(self, stop_at_differential: bool = False) -> str:
        items: list[str] = []
        while self.peek() is not None and self.peek() not in ("}", *_ADDITIVE_TOKENS):
            if stop_at_differential and self.peek() == "d" and self._differential_ahead():
                break
            items.append(self.item())
        return _join(items)

    def _differential_ahead(self) -> bool:
        nxt = self.tokens[self.pos + 1] if self.pos + 1 < len(self.tokens) else ""
        return bool(re.fullmatch(r"[A-Za-z]", nxt))

    def big_operator(self, name: str) -> str:
        lower, upper = self.bounds()
        if name == "lim":
            var, _, point = lower.partition(r"\to")
            if not point and "->" in lower:
                var, _, point = lower.partition("->")
            if not point:
                raise _Unsupported
            return f"limit({self.body()}, {var.strip()}, {point.strip()})"
        if name == "int":
            body = self.body(stop_at_differential=True)
            var = "x"
            if self.peek() == "d":
                self.take()
                var = self.take()
            args = [body, var] + ([lower, upper] if lower and upper else [])
            return f"int({', '.join(args)})"
        var, _, start = lower.partition("=")
        if not start:
            raise _Unsupported
        return f"{name}({self.body()}, {var.strip()}, {start.strip()}, {upper})"


_ADDITIVE_TOKENS = ("+", "-", "=", "<", ">", ",")


def _single_token(text: str) -> bool:
    """Is ``text`` one operand (a name, a number, a bracketed group or a call)?"""
    if re.fullmatch(r"[A-Za-z]+|\d+(?:\.\d+)?", text):
        return True
    call = re.match(r"[A-Za-z]*\(", text)
    return call is not None and text.endswith(")") and _balanced_outer(text[call.end() - 1 :])


def _balanced_outer(text: str) -> bool:
    depth = 0
    for i, ch in enumerate(text):
        depth += ch == "("
        depth -= ch == ")"
        if depth == 0 and i < len(text) - 1:
            return False
    return depth == 0


def _paren(text: str) -> str:
    return text if _single_token(text) else f"({text})"


def _join(items: list[str]) -> str:
    """Concatenate translated items; fractions are parenthesised unless they stand alone
    or sit between additive operators (``1/2 + x``)."""
    out: list[str] = []
    for i, item in enumerate(items):
        if item.startswith("\x00"):
            prev = items[i - 1] if i else None
            nxt = items[i + 1] if i + 1 < len(items) else None
            alone = (prev is None or prev in _ADDITIVE) and (nxt is None or nxt in _ADDITIVE)
            body = item.strip("\x00")
            out.append(body if alone else f"({body})")
        else:
            out.append(item)
    text = ""
    for part in out:
        if (
            text
            and part
            and text[-1].isalnum()
            and part[0].isalnum()
            and (text[-1].isalpha() or part[0].isalpha())
        ):
            text += " "  # keep "2" "x" and "x" "y" apart where they would merge into one token
        text += part
    return text


def latex_to_source(text: str) -> str | None:
    """OpenGraph source for the LaTeX in ``text``, or None if it is not (supported) LaTeX."""
    text = text.strip()
    if text.startswith("$") and text.endswith("$"):
        text = text.strip("$").strip()
    if "\\" not in text:
        return None
    try:
        reader = _Reader(text)
        source = reader.sequence(until=None)
    except _Unsupported:
        return None
    return source if reader.peek() is None else None
