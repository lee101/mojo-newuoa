"""Python bindings for the Mojo port of Powell's NEWUOA optimizer."""

from .optimize import OptimizeResult, minimize

__all__ = ["OptimizeResult", "minimize"]
