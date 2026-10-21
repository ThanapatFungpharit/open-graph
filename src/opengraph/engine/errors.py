"""Error types raised by the mathematical engine."""

from __future__ import annotations

Span = tuple[int, int]
"""Half-open character range ``[start, end)`` into the source text."""


class MathError(Exception):
    """Base class for user-facing errors in an expression.

    ``span`` (when known) points at the offending part of the source text so the UI can
    highlight it. ``incomplete`` marks input that only needs more typing.
    """

    def __init__(self, message: str, span: Span | None = None, *, incomplete: bool = False) -> None:
        super().__init__(message)
        self.message = message
        self.span = span
        # The text ends too early (``x^``, ``sqrt(``, ``1 +``): not wrong, just unfinished,
        # so the editor can stay calm while the user is still typing.
        self.incomplete = incomplete


class ParseError(MathError):
    """The text is not syntactically valid."""


class ResolveError(MathError):
    """The text parses, but refers to unknown names, has wrong arity, cycles, etc."""
