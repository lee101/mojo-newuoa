from __future__ import annotations

import ctypes
from pathlib import Path
import subprocess

import numpy as np
import pytest

from mojo_newuoa._lib import CALLBACK


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "tests" / "reference" / "libreference-newuoa.so"


def _build_reference() -> Path:
    sources = [
        ROOT / "tests" / "reference" / "reference.cpp",
        ROOT / "tests" / "reference" / "upstream_newuoa.h",
    ]
    if not REFERENCE.exists() or REFERENCE.stat().st_mtime < max(
        source.stat().st_mtime for source in sources
    ):
        subprocess.run(
            [
                "c++",
                "-O3",
                "-std=c++17",
                "-shared",
                "-fPIC",
                str(sources[0]),
                "-o",
                str(REFERENCE),
            ],
            check=True,
            cwd=ROOT,
        )
    return REFERENCE


@pytest.fixture(scope="session")
def reference_minimize():
    library = ctypes.CDLL(str(_build_reference()))
    function = library.reference_min_newuoa
    integer = ctypes.c_int64
    function.argtypes = [
        integer,
        integer,
        ctypes.c_double,
        ctypes.c_double,
        integer,
        integer,
        integer,
        ctypes.POINTER(integer),
    ]
    function.restype = ctypes.c_double
    npt_function = library.reference_newuoa_npt
    npt_function.argtypes = [
        integer,
        integer,
        integer,
        ctypes.c_double,
        ctypes.c_double,
        integer,
        integer,
        integer,
        ctypes.POINTER(integer),
    ]
    npt_function.restype = ctypes.c_double

    def run(fun, x0, *, rhobeg=1.0, rhoend=1.0e-8, maxfun=5000, npt=None):
        x = np.ascontiguousarray(x0, dtype=np.float64).copy()

        def evaluate(n, address, _context):
            view = np.ctypeslib.as_array(
                (ctypes.c_double * n).from_address(address)
            )
            return float(fun(view))

        callback = CALLBACK(evaluate)
        calls = integer()
        arguments = [
            x.size,
            x.ctypes.data,
            rhobeg,
            rhoend,
            maxfun,
            ctypes.cast(callback, ctypes.c_void_p).value,
            0,
            ctypes.byref(calls),
        ]
        if npt is None:
            value = function(*arguments)
        else:
            arguments.insert(1, npt)
            value = npt_function(*arguments)
        return x, value, calls.value

    return run
