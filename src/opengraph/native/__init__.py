"""Optional compiled (Cython) kernels.

``kernels`` holds the float64 geometry loops and ``mpvm`` the MPFR register machine
for deep-zoom evaluation. Both are built by ``setup.py`` when a C compiler is available;
without them every caller uses its pure Python/NumPy reference implementation, which
gives the same results more slowly. ``OPENGRAPH_PURE_PYTHON=1`` disables them (tests use
it to compare both paths).
"""

from __future__ import annotations

import importlib
import os
from types import ModuleType

from opengraph.engine.multiprecision import HAS_GMPY2, gmpy2  # noqa: F401 - loads what _mpvm needs


def _load(name: str) -> ModuleType | None:
    if os.environ.get("OPENGRAPH_PURE_PYTHON"):
        return None
    if name == "_mpvm" and not HAS_GMPY2:  # it links against gmpy2's MPFR
        return None
    try:
        return importlib.import_module(f"opengraph.native.{name}")
    except ImportError:
        return None


kernels: ModuleType | None = _load("_kernels")
mpvm: ModuleType | None = _load("_mpvm")


def _python_evaluation() -> str:
    return "gmpy2 Python" if HAS_GMPY2 else "mpmath Python"


def status() -> str:
    """Which compiled kernels are in use (for diagnostics)."""
    parts = [
        f"geometry kernels: {'compiled' if kernels is not None else 'Python'}",
        f"precise evaluation: {'compiled MPFR' if mpvm is not None else _python_evaluation()}",
    ]
    return ", ".join(parts)
