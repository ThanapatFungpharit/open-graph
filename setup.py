"""Build the optional Cython kernels (``opengraph.native``).

Project metadata lives in ``pyproject.toml``; this file only declares the extensions.
They link against the MPFR/GMP libraries bundled with gmpy2. If they can't be compiled
(no C compiler), the package installs without them and uses its Python fallbacks.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


def gmpy2_paths() -> tuple[list[str], list[str]]:
    """Header and library directories of the installed gmpy2 (it ships both)."""
    import gmpy2

    package = Path(gmpy2.__file__).parent
    libs = [str(package)]
    bundled = package.parent / "gmpy2.libs"  # wheels repaired by delvewheel/auditwheel
    if bundled.is_dir():
        libs.insert(0, str(bundled))
    return [str(package)], libs


def extensions() -> list[Extension]:
    from Cython.Build import cythonize

    include, library_dirs = gmpy2_paths()
    optimize = ["/O2"] if sys.platform == "win32" else ["-O3"]
    kernels = Extension(
        "opengraph.native._kernels",
        ["src/opengraph/native/_kernels.pyx"],
        extra_compile_args=optimize,
    )
    mpvm = Extension(
        "opengraph.native._mpvm",
        ["src/opengraph/native/_mpvm.pyx"],
        include_dirs=include,
        library_dirs=library_dirs,
        libraries=["mpfr", "gmp"],
        extra_compile_args=optimize,
    )
    return cythonize(
        [kernels, mpvm],
        include_path=include,
        compiler_directives={"language_level": "3"},
        quiet=True,
    )


class OptionalBuildExt(build_ext):
    """Skip an extension that fails to compile instead of failing the install."""

    def run(self) -> None:
        try:
            super().run()
        except Exception as error:  # noqa: BLE001 - no compiler at all
            self._skip("the kernels", error)

    def build_extension(self, ext: Extension) -> None:
        try:
            super().build_extension(ext)
        except Exception as error:  # noqa: BLE001 - one extension failed
            self._skip(ext.name, error)

    def _skip(self, what: str, error: Exception) -> None:
        if os.environ.get("OPENGRAPH_REQUIRE_NATIVE"):
            raise error
        print(f"warning: not building {what} ({error}); using Python fallbacks", file=sys.stderr)


setup(ext_modules=extensions(), cmdclass={"build_ext": OptionalBuildExt})
