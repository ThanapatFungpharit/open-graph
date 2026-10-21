"""Continuous, element-wise values backed by intervals, without materializing terms."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

from opengraph.engine.ast import (
    BinOp,
    Bounded,
    IntervalExpr,
    Node,
    Num,
    Var,
    children,
    replace,
)


def intervals(node: Node) -> Iterator[IntervalExpr]:
    """Value intervals only; a Bounded node's range is domain metadata."""
    if isinstance(node, IntervalExpr):
        yield node
        return
    kids = (node.value, node.coordinate) if isinstance(node, Bounded) else children(node)
    for child in kids:
        yield from intervals(child)


def parameterize(nodes: Sequence[Node], variable: str = "t") -> tuple[list[Node], IntervalExpr]:
    """Pair intervals at the same fractional position, like equal-length lists.

    Keeping the original expressions preserves correlations (I-I is zero) and
    nonlinear images (sin(I) includes its interior extrema).
    """
    domains = [domain for node in nodes for domain in intervals(node)]
    closed_lo = all(d.closed_lo for d in domains)
    closed_hi = all(d.closed_hi for d in domains)
    empty = any(d.lo == d.hi and not (d.closed_lo and d.closed_hi) for d in domains)
    singleton = all(d.lo == d.hi for d in domains)
    domain = IntervalExpr(
        Num(0),
        Num(0 if empty or singleton else 1),
        closed_lo and not empty,
        closed_hi and not empty,
    )

    def convert(node: Node) -> Node | None:
        if isinstance(node, IntervalExpr):
            assert isinstance(node.lo, Num) and isinstance(node.hi, Num)
            if node.lo == node.hi:
                return node.lo
            return BinOp(
                "+", node.lo, BinOp("*", Num(node.hi.value - node.lo.value), Var(variable))
            )
        if isinstance(node, Bounded):
            return Bounded(
                replace(node.value, convert),
                replace(node.coordinate, convert),
                node.interval,
                node.span,
            )
        return None

    return [replace(node, convert) for node in nodes], domain
