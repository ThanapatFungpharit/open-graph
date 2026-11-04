"""The arbitrary-precision library: gmpy2 where it exists, or its stand-in in pure Python.

Import ``gmpy2`` from here. It is the real library on desktops, and
:mod:`~opengraph.engine.mpfr_fallback` where there are no wheels for it (Android, iOS).
``OPENGRAPH_NO_GMPY2=1`` forces the stand-in, which is how the test suite covers it.
"""

from __future__ import annotations

import importlib
import os
from types import ModuleType

from opengraph.engine import mpfr_fallback


def _load() -> ModuleType:
    if os.environ.get("OPENGRAPH_NO_GMPY2"):
        return mpfr_fallback
    try:
        return importlib.import_module("gmpy2")
    except ImportError:
        return mpfr_fallback


gmpy2: ModuleType = _load()
HAS_GMPY2 = gmpy2 is not mpfr_fallback
