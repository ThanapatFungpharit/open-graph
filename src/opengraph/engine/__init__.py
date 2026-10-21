"""Mathematical engine: parsing, resolution, numerical evaluation, sampling, symbolics.

This package has no interface dependency and can be used and tested headless.
"""

from opengraph.engine.compiler import RowResult, compile_document, compile_expression
from opengraph.engine.errors import MathError, ParseError, ResolveError
from opengraph.engine.evaluator import NumericFunction
from opengraph.engine.parser import parse
from opengraph.engine.plots import ExplicitPlot, Geometry, ImplicitPlot, PointPlot
from opengraph.engine.sampling import ViewRect

__all__ = [
    "ExplicitPlot",
    "Geometry",
    "ImplicitPlot",
    "MathError",
    "NumericFunction",
    "ParseError",
    "PointPlot",
    "ResolveError",
    "RowResult",
    "ViewRect",
    "compile_document",
    "compile_expression",
    "parse",
]
