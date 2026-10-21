"""Tokeniser for the expression language."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, auto

from opengraph.engine.errors import ParseError, Span


class Kind(Enum):
    NUMBER = auto()
    NAME = auto()
    OP = auto()
    END = auto()


@dataclass(frozen=True, slots=True)
class Token:
    kind: Kind
    text: str
    span: Span


# Typographic variants users commonly paste, normalised to the canonical spelling.
_ALIASES = {
    "**": "^",
    "−": "-",
    "–": "-",
    "×": "*",
    "·": "*",
    "⋅": "*",
    "÷": "/",
    "≤": "<=",
    "≥": ">=",
    "…": "...",
}

_TOKEN_RE = re.compile(
    r"""
    (?P<ws>\s+)
  | (?P<number>(?:\d+(?:\.(?!\.\.)\d*)?|\.\d+)(?:[eE][+-]?\d+)?)
  | (?P<name>∫|[^\W\d_]+(?:_[^\W_]+)?)
  | (?P<op>\*\*|<=|>=|\.\.\.|[-+*/^(),|=<>!\[\]{}…−–×·⋅÷≤≥∈])
    """,
    re.VERBOSE,
)


def tokenize(text: str) -> list[Token]:
    """Split ``text`` into tokens; always ends with an ``END`` token.

    Identifiers are runs of letters with an optional ``_subscript`` (``a``, ``theta``,
    ``a_1``, ``θ``). Digits never continue an identifier, so ``x2`` lexes as ``x 2``.
    An exponent needs digits, so ``2e3`` is 2000 while ``2ex`` lexes as ``2 ex``.
    ``∫`` lexes as a name (the integral function) and ``...`` (or ``…``) as one operator,
    so ``[1...5]`` is ``[ 1 ... 5 ]`` rather than the numbers ``1.`` and ``.5``.
    """
    tokens: list[Token] = []
    pos = 0
    while pos < len(text):
        match = _TOKEN_RE.match(text, pos)
        if match is None:
            raise ParseError(f"Unexpected character '{text[pos]}'", (pos, pos + 1))
        kind_name = match.lastgroup
        value = match.group()
        span = (match.start(), match.end())
        pos = match.end()
        if kind_name == "ws":
            continue
        if kind_name == "number":
            tokens.append(Token(Kind.NUMBER, value, span))
        elif kind_name == "name":
            tokens.append(Token(Kind.NAME, value, span))
        else:
            tokens.append(Token(Kind.OP, _ALIASES.get(value, value), span))
    tokens.append(Token(Kind.END, "", (len(text), len(text))))
    return tokens
