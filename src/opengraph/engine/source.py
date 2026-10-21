"""Lossless source for resolved expressions, including original domain restrictions."""

from __future__ import annotations

from opengraph.engine.ast import (
    BinOp,
    Bounded,
    Const,
    FuncCall,
    Guard,
    IntervalExpr,
    ListExpr,
    Node,
    Num,
    Reduce,
    TupleExpr,
    UnaryOp,
    Var,
)
from opengraph.engine.builtins import INVERSE_TRIG_FUNCTIONS, TRIG_FUNCTIONS


def source(node: Node, *, degrees: bool = False) -> str:
    def emit(item: Node) -> str:
        return source(item, degrees=degrees)

    match node:
        case Num(value=value):
            return str(int(value)) if float(value).is_integer() else repr(value)
        case Const(name=name):
            return name
        case Var(name=name):
            return name.split("@")[0]
        case UnaryOp(operand=operand):
            return f"(-{emit(operand)})"
        case BinOp(op=op, left=left, right=right):
            return f"({emit(left)}{op}{emit(right)})"
        case FuncCall(name=name, args=args):
            arguments = [emit(a) for a in args]
            if degrees and name in TRIG_FUNCTIONS:
                arguments[0] = f"({arguments[0]}*180/pi)"
            result = f"{name}({', '.join(arguments)})"
            if degrees and name in INVERSE_TRIG_FUNCTIONS:
                result = f"({result}*pi/180)"
            return result
        case Guard(value=value, original=original):
            return f"guard({emit(value)}, {emit(original)})"
        case ListExpr(items=items):
            return "list[" + ", ".join(emit(i) for i in items) + "]"
        case TupleExpr(items=items):
            return "point[" + ", ".join(emit(i) for i in items) + "]"
        case IntervalExpr(lo=lo, hi=hi, closed_lo=cl, closed_hi=ch):
            return f"{'[' if cl else '('}{emit(lo)}, {emit(hi)}{']' if ch else ')'}"
        case Bounded(value=value, coordinate=coordinate, interval=interval):
            return f"restrict({emit(value)}, {emit(coordinate)}, {emit(interval)})"
        case Reduce(kind=kind, var=var, body=body, lower=lower, upper=upper):
            return f"{kind}({emit(body)}, {var.split('@')[0]}, {emit(lower)}, {emit(upper)})"
    raise ValueError(f"Cannot print unresolved {type(node).__name__}")
