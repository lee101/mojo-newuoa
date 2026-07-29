"""ctypes access to the compiled Mojo NEWUOA library."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src", "newuoa.mojo")
LIB = os.path.join(ROOT, "dist", "libmojo-newuoa.so")

I = ctypes.c_int64
F = ctypes.c_double
CALLBACK = ctypes.CFUNCTYPE(F, I, I, I)


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    """Build the shared library when it is absent or older than its source."""
    stale = not os.path.exists(LIB) or os.path.getmtime(LIB) < os.path.getmtime(SRC)
    if force or stale:
        pixi = shutil.which("pixi")
        if pixi:
            cmd = [pixi, "run", "--manifest-path", os.path.join(ROOT, "pixi.toml"), "build"]
        else:
            cmd = ["bash", os.path.join(ROOT, "build", "build.sh")]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if proc.returncode != 0 or not os.path.exists(LIB):
            raise BuildError((proc.stderr or proc.stdout).strip()[:6000])
    return LIB


_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        _library.mnu_workspace_size.argtypes = [I, I]
        _library.mnu_workspace_size.restype = I
        _library.mnu_minimize_f64.argtypes = [
            I,
            I,
            I,
            I,
            F,
            F,
            I,
            I,
            I,
            I,
            I,
            I,
            I,
        ]
        _library.mnu_minimize_f64.restype = I
    return _library
