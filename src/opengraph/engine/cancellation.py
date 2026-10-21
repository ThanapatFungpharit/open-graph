"""Cooperative cancellation of numerical work, local to the worker's context."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_cancelled: ContextVar[Callable[[], bool] | None] = ContextVar("cancelled", default=None)


class Cancelled(Exception):
    """The caller no longer needs this numerical result."""


def checkpoint() -> None:
    check = _cancelled.get()
    if check is not None and check():
        raise Cancelled


@contextmanager
def cancellable(check: Callable[[], bool]) -> Iterator[None]:
    token = _cancelled.set(check)
    try:
        checkpoint()
        yield
    finally:
        _cancelled.reset(token)
