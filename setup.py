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
    bundled = package.parent / "gmpy2.libs"  # wheels repaired by delvewheel/auditwheel/delocate
    if bundled.is_dir():
        libs.insert(0, str(bundled))
    return [str(package)], libs


def bundled_library(library_dirs: list[str], name: str) -> Path | None:
    """The shared library gmpy2's wheel bundles for ``name``, if it has one.

    Linux wheels rename it with a hash (``libmpfr-bb88ce0c.so.6.2.2``) and macOS wheels
    keep only the versioned name (``libmpfr.6.dylib``), so ``-lmpfr`` can't find either.
    """
    pattern = f"lib{name}*.dylib" if sys.platform == "darwin" else f"lib{name}[-.]*so*"
    for directory in library_dirs:
        found = sorted(Path(directory).glob(pattern))
        if found:
            return found[0]
    return None


def mpfr_link_options(library_dirs: list[str]) -> dict[str, list[str]]:
    """How ``_mpvm`` links against MPFR and GMP on this platform.

    Windows wheels ship ``mpfr.lib``/``gmp.lib`` import libraries, so plain library names
    work there (as they do with a system MPFR). Otherwise the bundled files are linked by
    path. The build environment is temporary, so no rpath is recorded: the extension
    names each library by its soname (Linux) or install name (macOS), which matches the
    copy ``import gmpy2`` has already loaded when ``opengraph.native`` imports it.
    """
    found = [bundled_library(library_dirs, name) for name in ("mpfr", "gmp")]
    if sys.platform == "win32" or None in found:
        return {"library_dirs": library_dirs, "libraries": ["mpfr", "gmp"]}
    return {"extra_objects": [str(path) for path in found if path is not None]}


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
        extra_compile_args=optimize,
        **mpfr_link_options(library_dirs),
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
