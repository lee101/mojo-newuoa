from __future__ import annotations

import ctypes

import numpy as np
import pytest

from mojo_newuoa import minimize
from mojo_newuoa._lib import CALLBACK, lib


def sphere(x):
    return float(x @ x)


def rosenbrock(x):
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))


def powell_singular(x):
    return float(
        (x[0] + 10.0 * x[1]) ** 2
        + 5.0 * (x[2] - x[3]) ** 2
        + (x[1] - 2.0 * x[2]) ** 4
        + 10.0 * (x[0] - x[3]) ** 4
    )


@pytest.mark.parametrize(
    ("fun", "x0", "rhobeg", "rhoend", "x_atol", "fun_atol"),
    [
        (sphere, [3.0, -4.0], 1.0, 1.0e-8, 1.0e-8, 1.0e-30),
        (
            sphere,
            [10.0, -2.0, 4.0, 1.0, -7.0],
            2.0,
            1.0e-9,
            1.0e-8,
            1.0e-28,
        ),
        (rosenbrock, [-1.2, 1.0], 1.0, 1.0e-8, 2.0e-8, 1.0e-15),
        (
            rosenbrock,
            [-1.2, 1.0, 0.5, -0.2],
            0.5,
            1.0e-7,
            2.0e-6,
            2.0e-6,
        ),
        (
            powell_singular,
            [3.0, -1.0, 0.0, 1.0],
            1.0,
            1.0e-8,
            5.0e-4,
            1.0e-13,
        ),
    ],
)
def test_matches_upstream(
    reference_minimize, fun, x0, rhobeg, rhoend, x_atol, fun_atol
):
    expected_x, expected_fun, expected_calls = reference_minimize(
        fun, x0, rhobeg=rhobeg, rhoend=rhoend
    )
    actual = minimize(fun, x0, rhobeg=rhobeg, rhoend=rhoend)
    np.testing.assert_allclose(actual.x, expected_x, rtol=1.0e-6, atol=x_atol)
    assert abs(actual.fun - expected_fun) <= fun_atol
    assert abs(actual.nfev - expected_calls) <= max(10, expected_calls // 10)


def test_maxfun_matches_upstream(reference_minimize):
    expected_x, expected_fun, expected_calls = reference_minimize(
        rosenbrock, [-1.2, 1.0], maxfun=20
    )
    actual = minimize(rosenbrock, [-1.2, 1.0], rhobeg=1.0, maxfun=20)
    assert actual.fun == pytest.approx(expected_fun, rel=0.05)
    assert actual.nfev == expected_calls == 20
    assert not actual.success


def test_constant_objective_matches_upstream(reference_minimize):
    fun = lambda _x: 7.25
    expected_x, expected_fun, expected_calls = reference_minimize(fun, [0.0, 0.0])
    actual = minimize(fun, [0.0, 0.0], rhobeg=1.0)
    np.testing.assert_array_equal(actual.x, expected_x)
    assert actual.fun == expected_fun == 7.25
    assert abs(actual.nfev - expected_calls) <= 10


def test_ill_conditioned_quadratic_matches_upstream(reference_minimize):
    fun = lambda x: float(1.0e12 * x[0] ** 2 + x[1] ** 2)
    expected_x, expected_fun, expected_calls = reference_minimize(
        fun, [1.0, 1.0], rhoend=1.0e-10
    )
    actual = minimize(fun, [1.0, 1.0], rhobeg=1.0, rhoend=1.0e-10)
    np.testing.assert_allclose(actual.x, expected_x, rtol=1.0e-9, atol=1.0e-12)
    assert actual.fun == pytest.approx(expected_fun, rel=1.0e-8, abs=1.0e-22)
    assert abs(actual.nfev - expected_calls) <= 10


@pytest.mark.parametrize(
    ("x0", "npt"),
    [
        pytest.param(np.linspace(-1.0, 2.0, n), npt, id=f"n={n}-npt={npt}")
        for n in (2, 3)
        for npt in range(n + 2, (n + 1) * (n + 2) // 2 + 1)
    ],
)
def test_every_valid_custom_interpolation_count_matches_upstream(
    reference_minimize, x0, npt
):
    expected_x, expected_fun, expected_calls = reference_minimize(
        sphere, x0, npt=npt
    )
    actual = minimize(sphere, x0, rhobeg=1.0, npt=npt)
    np.testing.assert_allclose(actual.x, expected_x, atol=1.0e-8)
    # Both solutions are well inside rhoend of the minimizer; final roundoff
    # differs because the C++ and Mojo compilers contract operations differently.
    assert abs(actual.fun - expected_fun) <= 2.0e-18
    assert abs(actual.nfev - expected_calls) <= 10


@pytest.mark.parametrize(("n", "npt"), [(3, 7), (4, 9), (5, 11)])
def test_simd_remainder_dimensions_match_upstream(reference_minimize, n, npt):
    x0 = np.linspace(-2.0, 3.0, n)
    expected_x, expected_fun, expected_calls = reference_minimize(
        sphere, x0, rhobeg=1.0, rhoend=1.0e-9, npt=npt
    )
    actual = minimize(
        sphere, x0, rhobeg=1.0, rhoend=1.0e-9, npt=npt
    )
    np.testing.assert_allclose(actual.x, expected_x, rtol=1.0e-6, atol=1.0e-8)
    assert actual.fun == pytest.approx(expected_fun, rel=1.0e-6, abs=1.0e-28)
    assert abs(actual.nfev - expected_calls) <= 10


def test_input_is_not_modified():
    x0 = np.array([2.0, -3.0])
    minimize(sphere, x0, rhobeg=1.0)
    np.testing.assert_array_equal(x0, [2.0, -3.0])


def test_callback_reuses_read_only_zero_copy_view():
    view_ids = []

    def objective(x):
        assert not x.flags.writeable
        view_ids.append(id(x))
        return sphere(x)

    result = minimize(objective, [2.0, -3.0], rhobeg=1.0)
    assert len(view_ids) == result.nfev
    assert len(set(view_ids)) == 1
    assert result.x.flags.writeable


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_nonfinite_objective_is_rejected(bad_value):
    with pytest.raises(ValueError, match="finite scalar"):
        minimize(lambda _x: bad_value, [0.0, 0.0], rhobeg=1.0, maxfun=8)


def test_complex_input_is_not_silently_narrowed():
    with pytest.raises(TypeError, match="complex"):
        minimize(sphere, np.array([1.0 + 2.0j, 3.0]), rhobeg=1.0)


def test_inexact_large_integer_input_is_not_silently_narrowed():
    with pytest.raises(ValueError, match="represented exactly"):
        minimize(sphere, np.array([2**53 + 1, 0], dtype=np.int64), rhobeg=1.0)


def test_objective_exception_propagates():
    def broken(_x):
        raise LookupError("objective failed")

    with pytest.raises(LookupError, match="objective failed"):
        minimize(broken, [0.0, 0.0], rhobeg=1.0, maxfun=8)


@pytest.mark.parametrize(
    ("x0", "kwargs", "message"),
    [
        ([1.0], {}, "at least two"),
        ([1.0, 2.0], {"npt": 3}, "npt"),
        ([1.0, 2.0], {"rhobeg": 0.0}, "radii"),
        ([1.0, 2.0], {"rhoend": 2.0, "rhobeg": 1.0}, "radii"),
        ([1.0, 2.0], {"maxfun": 0}, "positive"),
        ([1.0, 2.0], {"rhobeg": np.inf}, "finite"),
        ([1.0, np.nan], {}, "finite"),
    ],
)
def test_invalid_arguments(x0, kwargs, message):
    with pytest.raises(ValueError, match=message):
        minimize(sphere, x0, **kwargs)


def test_workspace_formula():
    native = lib()
    for n in (2, 3, 10):
        npt = 2 * n + 1
        expected = (npt + 13) * (npt + n) + 3 * n * (n + 3) // 2 + 11
        assert native.mnu_workspace_size(n, npt) == expected
    assert native.mnu_workspace_size(2, 7) == 0
    assert native.mnu_workspace_size(30_001, 30_003) == 0


def test_native_rejects_short_buffers():
    native = lib()
    callback = CALLBACK(lambda _n, _x, _context: 0.0)
    x = np.zeros(2, dtype=np.float64)
    workspace = np.empty(native.mnu_workspace_size(2, 5), dtype=np.float64)
    meta = np.empty(3, dtype=np.float64)
    status = native.mnu_minimize_f64(
        2,
        5,
        x.ctypes.data,
        1,
        1.0,
        1.0e-8,
        100,
        workspace.ctypes.data,
        workspace.size,
        ctypes.cast(callback, ctypes.c_void_p).value,
        0,
        meta.ctypes.data,
        meta.size,
    )
    assert status == -5


@pytest.mark.parametrize("keyword", ["npt", "maxfun"])
def test_integer_options_are_not_silently_truncated(keyword):
    with pytest.raises(TypeError, match="integers"):
        minimize(sphere, [1.0, 2.0], **{keyword: 5.5})
