"""Public Python API for the Mojo NEWUOA port."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import operator
from typing import Callable, Iterable

import numpy as np

from ._lib import CALLBACK, lib


Objective = Callable[[np.ndarray], float]


@dataclass(frozen=True)
class OptimizeResult:
    """Result returned by :func:`minimize`."""

    x: np.ndarray
    fun: float
    nfev: int
    success: bool
    status: int
    message: str


_ERRORS = {
    -1: "NEWUOA requires at least two variables",
    -2: "npt must be in [n + 2, (n + 1)(n + 2) / 2]",
    -3: "radii must satisfy 0 < rhoend <= rhobeg",
    -4: "an internal FFI pointer was null",
    -5: "an internal FFI buffer was too short",
    -6: "problem dimension is too large for safe workspace arithmetic",
}


def minimize(
    fun: Objective,
    x0: Iterable[float] | np.ndarray,
    *,
    rhobeg: float = 1.0e7,
    rhoend: float = 1.0e-8,
    maxfun: int = 5000,
    npt: int | None = None,
) -> OptimizeResult:
    """Minimize an unconstrained scalar function without derivatives.

    The defaults match ``min_newuoa`` in VCGLib's bundled upstream header.
    ``fun`` receives a read-only NumPy view that is valid only during the call.
    """
    source = np.asarray(x0)
    if np.issubdtype(source.dtype, np.complexfloating):
        raise TypeError("x0 must contain real values; complex values would be discarded")
    x = np.array(source, dtype=np.float64, order="C", copy=True)
    if np.issubdtype(source.dtype, np.integer):
        # Float64 cannot distinguish every large integer. Accept ordinary integer
        # starting points, but never silently change their values.
        if not np.array_equal(x.astype(source.dtype), source):
            raise ValueError("x0 contains integers that cannot be represented exactly as Float64")
    if x.ndim != 1:
        raise ValueError("x0 must be one-dimensional")
    n = int(x.size)
    if n < 2:
        raise ValueError(_ERRORS[-1])
    try:
        points = 2 * n + 1 if npt is None else operator.index(npt)
        evaluation_limit = operator.index(maxfun)
    except TypeError as exc:
        raise TypeError("npt and maxfun must be integers") from exc
    upper = (n + 1) * (n + 2) // 2
    if not n + 2 <= points <= upper:
        raise ValueError(_ERRORS[-2])
    if not np.isfinite(rhobeg) or not np.isfinite(rhoend):
        raise ValueError("radii must be finite")
    if not 0.0 < rhoend <= rhobeg:
        raise ValueError(_ERRORS[-3])
    if evaluation_limit < 1:
        raise ValueError("maxfun must be positive")
    if not np.all(np.isfinite(x)):
        raise ValueError("x0 must contain only finite values")

    native = lib()
    workspace_size = int(native.mnu_workspace_size(n, points))
    storage = np.empty(workspace_size + 3, dtype=np.float64)
    meta_address = storage.ctypes.data + workspace_size * storage.itemsize
    callback_error: BaseException | None = None

    def evaluate(dim: int, address: int, _context: int) -> float:
        nonlocal callback_error
        try:
            if dim != n or address != x.ctypes.data:
                raise RuntimeError("native callback supplied an invalid parameter buffer")
            value = float(fun(x))
            if not np.isfinite(value):
                raise ValueError("objective must return a finite scalar")
            return value
        except BaseException as exc:
            if callback_error is None:
                callback_error = exc
            return float("nan")

    callback = CALLBACK(evaluate)
    x.setflags(write=False)
    try:
        status = int(
            native.mnu_minimize_f64(
                n,
                points,
                x.ctypes.data,
                x.size,
                float(rhobeg),
                float(rhoend),
                evaluation_limit,
                storage.ctypes.data,
                workspace_size,
                ctypes.cast(callback, ctypes.c_void_p).value,
                0,
                meta_address,
                3,
            )
        )
    finally:
        x.setflags(write=True)
    if callback_error is not None:
        raise callback_error
    if status != 0:
        raise RuntimeError(_ERRORS.get(status, f"NEWUOA failed with status {status}"))
    nfev = int(storage[workspace_size + 1])
    message = (
        "maximum function evaluations reached"
        if nfev >= evaluation_limit
        else "trust-region radius reached rhoend"
    )
    return OptimizeResult(
        x=x,
        fun=float(storage[workspace_size]),
        nfev=nfev,
        success=nfev < evaluation_limit,
        status=0 if nfev < evaluation_limit else 1,
        message=message,
    )
