"""Whitelist of functions and constants available in expressions.

Every name a user can call is listed here, with a vectorised NumPy implementation for
plotting and a SymPy constructor for symbolic operations. Nothing else is reachable from
user input.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Builtin:
    name: str
    min_args: int
    max_args: int | None  # None = variadic
    numeric: Callable[..., Array]
    # Receives the sympy module followed by the SymPy arguments.
    symbolic: Callable[..., Any]
    signature: str
    summary: str
    # Whether ``numeric`` also accepts complex arrays (complex mode). Functions that
    # don't are applied to the real part where the imaginary part is zero, else NaN.
    complex_ok: bool = False

    @property
    def complex_numeric(self) -> Callable[..., Any]:
        return self.numeric if self.complex_ok else _real_only(self.numeric)


def _real_only(fn: Callable[..., Array]) -> Callable[..., Any]:
    def apply(*args: Any) -> Any:
        if not any(np.iscomplexobj(a) for a in args):
            return fn(*args)
        real = np.ones(np.broadcast_shapes(*(np.shape(a) for a in args)), dtype=bool)
        for a in args:
            real &= np.imag(a) == 0
        result = np.asarray(fn(*(np.real(a) for a in args)), dtype=np.complex128)
        return np.where(real, result, np.nan)

    return apply


# Lanczos approximation (g = 7, n = 9): about 15 significant digits for real and complex
# arguments, with the reflection formula for Re(z) < 1/2.
_LANCZOS_G = 7.0
_LANCZOS = (
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.9843695780195716e-6,
    1.5056327351493116e-7,
)
_FACTORIALS = np.array([float(math.factorial(k)) for k in range(171)])  # 171! overflows


def _lanczos(z: Any) -> Any:
    z = z - 1
    series = np.full_like(z, _LANCZOS[0])
    for k, coefficient in enumerate(_LANCZOS[1:], start=1):
        series = series + coefficient / (z + k)
    t = z + _LANCZOS_G + 0.5
    # exp((z + 1/2) ln t - t) rather than t^(z+1/2) e^-t, which overflows too early.
    return math.sqrt(2 * math.pi) * np.exp((z + 0.5) * np.log(t) - t) * series


def gamma(z: Any) -> Any:
    """Γ(z), vectorised, real or complex; NaN at the poles 0, -1, -2, ..."""
    z = np.asarray(z)
    z = z.astype(np.complex128 if np.iscomplexobj(z) else np.float64)
    reflect = np.real(z) < 0.5
    with np.errstate(all="ignore"):
        direct = _lanczos(np.where(reflect, 1 - z, z))
        reflected = np.pi / (np.sin(np.pi * z) * direct)
        result = np.where(reflect, reflected, direct)
    pole = (np.imag(z) == 0) & (np.real(z) <= 0) & (np.real(z) == np.round(np.real(z)))
    return np.where(pole, np.nan, result)


def factorial(n: Any) -> Any:
    """n! = Γ(n + 1), exact (as a float) for the integers 0..170."""
    n = np.asarray(n)
    result = gamma(n + 1)
    if np.iscomplexobj(n):
        return result
    with np.errstate(invalid="ignore"):
        exact = (n == np.round(n)) & (n >= 0) & (n <= 170)
    table = _FACTORIALS[np.where(exact, n, 0).astype(np.int64)]
    return np.where(exact, table, result)


def _atan(a: Any, b: Any = None) -> Any:
    if b is None:
        return np.arctan(a)  # complex-capable
    return _real_only(np.arctan2)(a, b)


def _choose(n: Any, k: Any) -> Any:
    return factorial(n) / (factorial(k) * factorial(n - k))


def _permutations(n: Any, k: Any) -> Any:
    return factorial(n) / factorial(n - k)


def _reduce(fn: Callable[[Array, Array], Array]) -> Callable[..., Array]:
    def apply(*args: Array) -> Array:
        result = args[0]
        for arg in args[1:]:
            result = fn(result, arg)
        return result

    return apply


def _log(value: Array, base: Array | None = None) -> Array:
    if base is None:
        return np.log10(value)
    return np.log(value) / np.log(base)


def _sym_log(sp: Any, value: Any, base: Any = None) -> Any:
    return sp.log(value, 10 if base is None else base)


def _b(
    name: str,
    numeric: Callable[..., Array],
    symbolic: Callable[..., Any],
    summary: str,
    *,
    args: tuple[int, int | None] = (1, 1),
    signature: str | None = None,
    complex_ok: bool = False,
) -> Builtin:
    if signature is None:
        signature = f"{name}(x)" if args == (1, 1) else f"{name}(a, b)"
    return Builtin(name, args[0], args[1], numeric, symbolic, signature, summary, complex_ok)


def _c(
    name: str, numeric: Callable[..., Array], symbolic: Callable[..., Any], summary: str
) -> Builtin:
    """A one-argument builtin whose NumPy implementation also handles complex input."""
    return _b(name, numeric, symbolic, summary, complex_ok=True)


_FUNCTIONS: list[Builtin] = [
    _c("sin", np.sin, lambda sp, a: sp.sin(a), "Sine (in the current angle unit)"),
    _c("cos", np.cos, lambda sp, a: sp.cos(a), "Cosine (in the current angle unit)"),
    _c("tan", np.tan, lambda sp, a: sp.tan(a), "Tangent (in the current angle unit)"),
    _c("sec", lambda a: 1.0 / np.cos(a), lambda sp, a: sp.sec(a), "Secant"),
    _c("csc", lambda a: 1.0 / np.sin(a), lambda sp, a: sp.csc(a), "Cosecant"),
    _c("cot", lambda a: 1.0 / np.tan(a), lambda sp, a: sp.cot(a), "Cotangent"),
    _c("asin", np.arcsin, lambda sp, a: sp.asin(a), "Inverse sine (gives the angle unit)"),
    _c("acos", np.arccos, lambda sp, a: sp.acos(a), "Inverse cosine (gives the angle unit)"),
    # Two arguments give the angle of the point (x, y), like atan2 elsewhere ("atan2"
    # itself can't be a name: digits never continue an identifier, so it reads atan·2).
    _b(
        "atan",
        _atan,
        lambda sp, a, b=None: sp.atan(a) if b is None else sp.atan2(a, b),
        "Inverse tangent (gives the angle unit); atan(y, x) is the angle of (x, y)",
        args=(1, 2),
        signature="atan(x) or atan(y, x)",
        complex_ok=True,
    ),
    _c("sinh", np.sinh, lambda sp, a: sp.sinh(a), "Hyperbolic sine"),
    _c("cosh", np.cosh, lambda sp, a: sp.cosh(a), "Hyperbolic cosine"),
    _c("tanh", np.tanh, lambda sp, a: sp.tanh(a), "Hyperbolic tangent"),
    _c("asinh", np.arcsinh, lambda sp, a: sp.asinh(a), "Inverse hyperbolic sine"),
    _c("acosh", np.arccosh, lambda sp, a: sp.acosh(a), "Inverse hyperbolic cosine"),
    _c("atanh", np.arctanh, lambda sp, a: sp.atanh(a), "Inverse hyperbolic tangent"),
    _c("sqrt", np.sqrt, lambda sp, a: sp.sqrt(a), "Square root"),
    # x^(1/3) is the real cube root in OpenGraph's numeric semantics, so the plain power
    # is the faithful symbolic form (and differentiates without Piecewise terms).
    _b("cbrt", np.cbrt, lambda sp, a: a ** sp.Rational(1, 3), "Real cube root"),
    _c("abs", np.abs, lambda sp, a: sp.Abs(a), "Absolute value (modulus), also |x|"),
    _c("exp", np.exp, lambda sp, a: sp.exp(a), "e to the power x"),
    _c("ln", np.log, lambda sp, a: sp.log(a), "Natural logarithm"),
    _b(
        "log",
        _log,
        _sym_log,
        "Base-10 logarithm, or log(x, b) for base b",
        args=(1, 2),
        signature="log(x) or log(x, b)",
        complex_ok=True,
    ),
    _c("Gamma", gamma, lambda sp, a: sp.gamma(a), "Gamma function Γ(x); also Γ(x)"),
    _c(
        "factorial",
        factorial,
        lambda sp, a: sp.factorial(a),
        "Factorial, also n!; Γ(n + 1) for non-integers",
    ),
    _b(
        "nCr",
        _choose,
        lambda sp, a, b: sp.binomial(a, b),
        "Binomial coefficient: ways to choose k of n",
        args=(2, 2),
        signature="nCr(n, k)",
    ),
    _b(
        "nPr",
        _permutations,
        lambda sp, a, b: sp.factorial(a) / sp.factorial(a - b),
        "Permutations: ordered choices of k of n",
        args=(2, 2),
        signature="nPr(n, k)",
    ),
    _c("re", np.real, lambda sp, a: sp.re(a), "Real part of a complex number"),
    _c("im", np.imag, lambda sp, a: sp.im(a), "Imaginary part of a complex number"),
    _c("arg", np.angle, lambda sp, a: sp.arg(a), "Argument (angle, radians) of a complex number"),
    _c("conj", np.conj, lambda sp, a: sp.conjugate(a), "Complex conjugate"),
    _b("floor", np.floor, lambda sp, a: sp.floor(a), "Round down"),
    _b("ceil", np.ceil, lambda sp, a: sp.ceiling(a), "Round up"),
    _b(
        "round",
        lambda a: np.floor(a + 0.5),
        lambda sp, a: sp.floor(a + sp.Rational(1, 2)),
        "Round half up",
    ),
    _b("sign", np.sign, lambda sp, a: sp.sign(a), "Sign: -1, 0 or 1"),
    _b("mod", np.mod, lambda sp, a, b: sp.Mod(a, b), "Remainder with the sign of b", args=(2, 2)),
    _b(
        "min",
        _reduce(np.minimum),
        lambda sp, *a: sp.Min(*a),
        "Smallest argument",
        args=(2, None),
        signature="min(a, b, ...)",
    ),
    _b(
        "max",
        _reduce(np.maximum),
        lambda sp, *a: sp.Max(*a),
        "Largest argument",
        args=(2, None),
        signature="max(a, b, ...)",
    ),
]

_ALIASES = {
    "arcsin": "asin",
    "arccos": "acos",
    "arctan": "atan",
    "sgn": "sign",
    "arcsinh": "asinh",
    "arccosh": "acosh",
    "arctanh": "atanh",
    "Γ": "Gamma",
}

FUNCTIONS: dict[str, Builtin] = {f.name: f for f in _FUNCTIONS}
FUNCTIONS.update({alias: FUNCTIONS[target] for alias, target in _ALIASES.items()})

# Angle-unit handling: in degree mode the compiler converts the argument of these to
# radians, and the result of the inverse functions back to degrees.
TRIG_FUNCTIONS = frozenset({"sin", "cos", "tan", "sec", "csc", "cot"})
INVERSE_TRIG_FUNCTIONS = frozenset({"asin", "acos", "atan"})


@dataclass(frozen=True, slots=True)
class SpecialForm:
    """A name the compiler handles itself (it binds a variable or consumes a list)."""

    kind: str
    signature: str
    summary: str


# Σ/Π always name their index. A three-argument definite integral binds x.
REDUCERS: dict[str, str] = {
    "sum": "sum",
    "Σ": "sum",
    "Sigma": "sum",
    "prod": "prod",
    "Π": "prod",
    "Pi": "prod",
    "int": "int",
    "integral": "int",
    "∫": "int",
}

# Functions of a whole list. min/max accept a single list as well as several arguments.
LIST_FUNCTIONS = frozenset({"total", "mean", "length"})

# Computer-algebra operations: SymPy computes them in the background and the compiler
# substitutes the exact result, so ``g(x) = diff(f(x))`` defines g by a formula. Name
# -> what the result is called while it is being computed.
CAS_OPERATIONS: dict[str, str] = {
    "diff": "derivative",
    "integrate": "antiderivative",
    "simplify": "simplified form",
    "expand": "expanded form",
    "factor": "factored form",
    "cancel": "cancelled rational form",
    "together": "combined fraction",
    "apart": "partial fractions",
    "collect": "collected terms",
    "trigsimp": "trigonometric simplified form",
    "limit": "limit",
    "taylor": "Taylor polynomial",
    "roots": "real roots",
}

ALGEBRA_OPERATIONS = frozenset(
    {"simplify", "expand", "factor", "cancel", "together", "apart", "collect", "trigsimp"}
)

SPECIAL_FORMS: tuple[SpecialForm, ...] = (
    SpecialForm(
        "roots",
        "roots(a^2 - 4 for a) or roots(expression, variable)",
        "Finite list of exact real zeros",
    ),
    SpecialForm(
        "plot",
        "plot(t^2 for t in (0, 2]) or plot(expression, variable, interval)",
        "Graph on an explicit interval",
    ),
    SpecialForm(
        "parametric",
        "parametric(cos(u), sin(u) for u in [0, 2pi]) or parametric(x, y, u, interval)",
        "Parametric curve with a local parameter and range",
    ),
    SpecialForm("formula", "formula(name, expression)", "Display a formula with free names"),
    SpecialForm(
        "sum",
        "sum(k^2 for k from 1 to 10) or sum(body, k, a, b); also Σ",
        "Sum of body for k = a, a+1, …, b (also Sigma)",
    ),
    SpecialForm(
        "prod",
        "prod(k for k from 1 to 5) or prod(body, k, a, b); also Π",
        "Product of body for k = a, …, b (also Pi)",
    ),
    SpecialForm(
        "int",
        "int(sin(t) for t from 0 to x) or int(body, t, a, b); also ∫",
        "Definite integral of body dt from a to b (also integral)",
    ),
    SpecialForm("total", "total(list)", "Sum of a list's elements"),
    SpecialForm("mean", "mean(list)", "Average of a list's elements"),
    SpecialForm("length", "length(list)", "Number of elements in a list"),
    SpecialForm(
        "diff",
        "diff(f), diff(f for t), diff(f for t order n) or diff(f, t, n)",
        "Exact derivative of f in x (or t), n times (SymPy)",
    ),
    SpecialForm(
        "integrate",
        "integrate(f), integrate(f for t) or integrate(f, t)",
        "Antiderivative in x (or t), using current parameter values; constant 0",
    ),
    SpecialForm("simplify", "simplify(f)", "Simplest exact form of f (SymPy)"),
    SpecialForm("expand", "expand(f)", "f with products and powers multiplied out (SymPy)"),
    SpecialForm("factor", "factor(f)", "f as a product of irreducible factors (SymPy)"),
    SpecialForm("cancel", "cancel(f)", "Cancel common polynomial factors in a rational expression"),
    SpecialForm("together", "together(f)", "Combine rational terms over a common denominator"),
    SpecialForm(
        "apart", "apart(f), apart(f for t) or apart(f, t)", "Partial fractions in x (or t)"
    ),
    SpecialForm(
        "collect",
        "collect(f for t) or collect(f, t)",
        "Group terms by powers of the chosen variable",
    ),
    SpecialForm("trigsimp", "trigsimp(f)", "Simplify trigonometric identities (SymPy)"),
    SpecialForm(
        "limit",
        "limit(f for t to a), limit(f for t to a from right) or limit(f, t, a[, side])",
        "Exact limit of f as t → a, two-sided or from one side (SymPy)",
    ),
    SpecialForm(
        "taylor",
        "taylor(f for t at a degree n) or taylor(f, t, a, n)",
        "Taylor polynomial of f in t about a, up to (t − a)^n (SymPy)",
    ),
)

# These functions return real values but consume complex arguments, including when
# the rest of the document uses real arithmetic.
COMPLEX_PROJECTIONS = frozenset({"re", "im", "arg"})

IMAGINARY_UNIT = "i"  # also available inside a complex projection's argument

CONSTANTS: dict[str, float] = {
    "pi": math.pi,
    "π": math.pi,
    "tau": math.tau,
    "τ": math.tau,
    "e": math.e,
}

# SymPy constructors for constants, so symbolic results print as "pi" rather than 3.14...
SYMBOLIC_CONSTANTS: dict[str, Callable[[Any], Any]] = {
    "pi": lambda sp: sp.pi,
    "π": lambda sp: sp.pi,
    "tau": lambda sp: 2 * sp.pi,
    "τ": lambda sp: 2 * sp.pi,
    "e": lambda sp: sp.E,
    IMAGINARY_UNIT: lambda sp: sp.I,
}

PLOT_VARIABLES = ("x", "y", "z")

KEYWORDS = frozenset(
    {
        "plot",
        "parametric",
        "formula",
        "guard",
        "restrict",
        "point",
        "list",
        "seq",
        "sets",
        "in",
        "for",
    }
    | {"ℕ", "ℤ"}  # number sets are written sets[N] and sets[Z]
)

RESERVED = (
    KEYWORDS
    | frozenset(FUNCTIONS)
    | frozenset(CONSTANTS)
    | frozenset(PLOT_VARIABLES)
    | frozenset(REDUCERS)
    | LIST_FUNCTIONS
    | frozenset(CAS_OPERATIONS)
)
