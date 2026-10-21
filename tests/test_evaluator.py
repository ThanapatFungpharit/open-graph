from __future__ import annotations

import math

import numpy as np
import pytest

from opengraph.engine.compiler import compile_expression
from opengraph.engine.evaluator import NumericFunction
from opengraph.engine.plots import ExplicitPlot, ImplicitPlot


def fn(text: str) -> NumericFunction:
    result = compile_expression(text)
    assert not result.is_error, result.message
    assert isinstance(result.plot, ExplicitPlot)
    return result.plot.fn


def value(text: str) -> float:
    """Evaluate a constant expression via the row's displayed value."""
    result = compile_expression(text)
    assert not result.is_error, result.message
    return float(result.message.removeprefix("= ").replace("undefined", "nan"))


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("1 + 2 * 3", 7),
        ("(1 + 2) * 3", 9),
        ("2^3^2", 512),
        ("-2^2", -4),
        ("2^-1", 0.5),
        ("10 / 4", 2.5),
        ("1/2*4", 2),
        ("2pi", 2 * math.pi),
        ("e^2", math.e**2),
        ("tau", math.tau),
        ("π", math.pi),
        ("sqrt(16)", 4),
        ("cbrt(-27)", -3),
        ("(-8)^(1/3)", -2),
        ("(-8)^(2/3)", 4),
        ("abs(-3)", 3),
        ("|-3|", 3),
        ("ln(e)", 1),
        ("log(1000)", 3),
        ("log(8, 2)", 3),
        ("exp(0)", 1),
        ("floor(-1.5)", -2),
        ("ceil(1.2)", 2),
        ("round(2.5)", 3),
        ("round(-2.5)", -2),
        ("sign(-4)", -1),
        ("mod(-1, 3)", 2),
        ("min(3, 1, 2)", 1),
        ("max(3, 1, 2)", 3),
        ("sin(pi/2)", 1),
        ("cos(0)", 1),
        ("atan(1)", math.pi / 4),
        ("arcsin(1)", math.pi / 2),
        ("sec(0)", 1),
        ("cosh(0)", 1),
    ],
)
def test_constant_expressions(text: str, expected: float) -> None:
    assert value(text) == pytest.approx(expected, rel=1e-9)


def test_vectorised_evaluation_matches_numpy() -> None:
    xs = np.linspace(-5, 5, 1001)
    np.testing.assert_allclose(fn("x^2 - 3x + 2")(x=xs), xs**2 - 3 * xs + 2)
    np.testing.assert_allclose(fn("sin(x) cos(x)")(x=xs), np.sin(xs) * np.cos(xs))
    np.testing.assert_allclose(fn("exp(-x^2)")(x=xs), np.exp(-(xs**2)))
    out = fn("x")(x=xs)
    assert out.shape == xs.shape and out.dtype == np.float64


def test_constant_function_broadcasts_to_input_shape() -> None:
    xs = np.arange(10.0)
    out = fn("y = 3")(x=xs)
    assert out.shape == (10,)
    assert (out == 3).all()


def test_two_variable_broadcasting() -> None:
    result = compile_expression("x^2 + y^2 = 1")
    assert isinstance(result.plot, ImplicitPlot)
    gx, gy = np.meshgrid(np.arange(3.0), np.arange(2.0))
    np.testing.assert_allclose(result.plot.fn(x=gx, y=gy), gx**2 + gy**2 - 1)


def test_division_by_zero_is_non_finite_not_an_exception() -> None:
    out = fn("1/x")(x=np.array([-1.0, 0.0, 1.0]))
    assert out[0] == -1 and out[2] == 1
    assert not np.isfinite(out[1])
    assert math.isnan(fn("0/x")(x=np.array([0.0]))[0])


@pytest.mark.parametrize("text", ["sqrt(x)", "ln(x)", "log(x)", "asin(x - 2)", "acosh(x + 0.5)"])
def test_out_of_domain_is_nan_never_complex(text: str) -> None:
    out = fn(text)(x=np.array([-1.0]))
    assert out.dtype == np.float64
    assert not np.isfinite(out[0])


def test_odd_roots_are_real_for_negative_inputs() -> None:
    xs = np.array([-8.0, -1.0, 0.0, 8.0])
    np.testing.assert_allclose(fn("x^(1/3)")(x=xs), [-2, -1, 0, 2])
    np.testing.assert_allclose(fn("x^(2/3)")(x=xs), [4, 1, 0, 4])
    assert np.isnan(fn("x^(1/2)")(x=np.array([-4.0]))[0])
    assert np.isnan(fn("x^0.3")(x=np.array([-4.0]))[0])  # 3/10: even denominator


def test_no_warnings_escape(recwarn: pytest.WarningsRecorder) -> None:
    fn("sqrt(x) + ln(x) + 1/x")(x=np.linspace(-1, 1, 11))
    assert len(recwarn) == 0


def test_undefined_constant_displays_undefined() -> None:
    assert compile_expression("sqrt(-1)").message == "= undefined"
    assert compile_expression("1/0").message == "= ∞"


def test_scalar_helper() -> None:
    assert fn("2x").scalar(x=4.0) == 8.0


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("sum(1/n^x, n, 1, 10000)", lambda x: sum(n**-x for n in range(1, 10001))),
        ("sum(n, n, 1, floor(x))", lambda x: sum(range(1, math.floor(x) + 1))),
        (
            "sum(sum(m x, m, 1, n), n, 1, 30)",
            lambda x: sum(m * x for n in range(1, 31) for m in range(1, n + 1)),
        ),
        ("prod(1 + x/n, n, 1, 50)", lambda x: math.prod(1 + x / n for n in range(1, 51))),
    ],
)
def test_sums_agree_term_by_term(text: str, expected) -> None:  # noqa: ANN001
    plot = compile_expression(text).plot
    assert isinstance(plot, ExplicitPlot)
    xs = np.array([1.5, 2.0, 7.3])
    np.testing.assert_allclose(plot.evaluate(xs), [expected(x) for x in xs], rtol=1e-12)
