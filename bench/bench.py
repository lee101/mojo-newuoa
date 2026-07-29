from __future__ import annotations

import ctypes
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time

import numpy as np

from mojo_newuoa import minimize
from mojo_newuoa._lib import CALLBACK


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "tests" / "reference" / "libreference-newuoa.so"


def build_reference() -> ctypes.CDLL:
    source = ROOT / "tests" / "reference" / "reference.cpp"
    header = ROOT / "tests" / "reference" / "upstream_newuoa.h"
    if not REFERENCE.exists() or REFERENCE.stat().st_mtime < max(
        source.stat().st_mtime, header.stat().st_mtime
    ):
        subprocess.run(
            [
                "c++",
                "-O3",
                "-std=c++17",
                "-shared",
                "-fPIC",
                str(source),
                "-o",
                str(REFERENCE),
            ],
            cwd=ROOT,
            check=True,
        )
    library = ctypes.CDLL(str(REFERENCE))
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
    return library


REFERENCE_LIB = build_reference()


def upstream(fun, x0, rhobeg, rhoend, maxfun):
    x = np.asarray(x0, dtype=np.float64).copy()

    def evaluate(n, address, _context):
        view = np.ctypeslib.as_array(
            (ctypes.c_double * n).from_address(address)
        )
        return float(fun(view))

    callback = CALLBACK(evaluate)
    calls = ctypes.c_int64()
    value = REFERENCE_LIB.reference_min_newuoa(
        x.size,
        x.ctypes.data,
        rhobeg,
        rhoend,
        maxfun,
        ctypes.cast(callback, ctypes.c_void_p).value,
        0,
        ctypes.byref(calls),
    )
    return value, calls.value


def paired_medians(first, second, repeats=9):
    samples = [[], []]
    results = [None, None]
    functions = [first, second]
    for repeat in range(repeats):
        order = (0, 1) if repeat % 2 == 0 else (1, 0)
        for index in order:
            start = time.perf_counter_ns()
            results[index] = functions[index]()
            samples[index].append((time.perf_counter_ns() - start) / 1.0e6)
    return (
        statistics.median(samples[0]),
        results[0],
        statistics.median(samples[1]),
        results[1],
    )


def cpu_name():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def main():
    dense_matrix = np.diag(np.arange(1.0, 17.0))
    dense_matrix += 0.02 * np.outer(np.arange(1.0, 17.0), np.arange(1.0, 17.0))
    cases = [
        (
            "Rosenbrock 2D",
            lambda x: 100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2,
            np.array([-1.2, 1.0]),
            1.0,
            1.0e-8,
            5000,
        ),
        (
            "Chained Rosenbrock 8D",
            lambda x: np.sum(
                100.0 * (x[1:] - x[:-1] ** 2) ** 2
                + (1.0 - x[:-1]) ** 2
            ),
            np.linspace(-1.2, 0.8, 8),
            0.5,
            1.0e-7,
            5000,
        ),
        (
            "Dense quadratic 16D",
            lambda x: float(x @ dense_matrix @ x),
            np.linspace(-4.0, 4.0, 16),
            1.0,
            1.0e-7,
            5000,
        ),
    ]
    rows = []
    for name, fun, x0, rhobeg, rhoend, maxfun in cases:
        upstream(fun, x0, rhobeg, rhoend, maxfun)
        minimize(
            fun,
            x0,
            rhobeg=rhobeg,
            rhoend=rhoend,
            maxfun=maxfun,
        )
        (
            reference_ms,
            reference_result,
            mojo_ms,
            mojo_result,
        ) = paired_medians(
            lambda: upstream(fun, x0, rhobeg, rhoend, maxfun),
            lambda: minimize(
                fun,
                x0,
                rhobeg=rhobeg,
                rhoend=rhoend,
                maxfun=maxfun,
            ),
        )
        rows.append(
            (
                name,
                reference_ms,
                mojo_ms,
                reference_ms / mojo_ms,
                reference_result[1],
                mojo_result.nfev,
            )
        )
    print(f"Machine: {cpu_name()} ({platform.system()} {platform.machine()})")
    print()
    print("| Benchmark | Upstream C++ (ms) | Mojo (ms) | Speedup | C++/Mojo nfev |")
    print("|---|---:|---:|---:|---:|")
    for name, reference_ms, mojo_ms, speedup, ref_nfev, mojo_nfev in rows:
        print(
            f"| {name} | {reference_ms:.3f} | {mojo_ms:.3f} | "
            f"{speedup:.2f}x | {ref_nfev}/{mojo_nfev} |"
        )


if __name__ == "__main__":
    main()
