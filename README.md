# mojo-newuoa

`mojo-newuoa` is a standalone Mojo port of M. J. D. Powell's NEWUOA
derivative-free optimizer. It minimizes an unconstrained scalar function by
maintaining an interpolating quadratic model inside a trust region.

The source was ported from the actual
[NEWUOA header bundled with VCGLib](https://github.com/cnr-isti-vclab/vcglib/tree/main/wrap/newuoa),
not reconstructed from an algorithm description. That header is an f2c-based
C++ adaptation by Attractive Chaos and is distributed under the MIT License.
This repository is a derived work under the MIT License; see
[NOTICE](NOTICE) for attribution.

## Upstream coverage

The bundled upstream header has one public entry point, the templated
`min_newuoa`, plus its `newuoa_` implementation and the internal `newuob_`,
`trsapp_`, `biglag_`, `bigden_`, and `update_` routines. This port covers that
complete unconstrained Float64 optimization path:

- the truncated conjugate-gradient and two-dimensional trust-region solver
  (`trsapp_`);
- model-point selection and denominator stabilization (`biglag_` and
  `bigden_`);
- interpolation-factor updates (`update_`);
- the complete `newuob_` state machine;
- the upstream default `npt = 2*n + 1` and every valid custom interpolation
  count through `(n + 1)*(n + 2)/2`;
- Python objectives that return a finite scalar, evaluation budgets, and the
  upstream radius schedule and epsilon choices.

It does not provide Float32 arithmetic, bound or nonlinear constraints,
gradient-based methods, batched optimization, or GPU execution. NEWUOA itself
requires at least two variables. The internal routines are translated as
implementation details; they are not exposed as separate public Python APIs.
The port does not reproduce unrelated VCGLib code or add an API compatible with
SciPy's optimizer interfaces.

## Install and build

```bash
pixi install
pixi run build
pixi run test
```

The build produces `dist/libmojo-newuoa.so`. The Python package also rebuilds a
missing or stale library on first import.

## Usage

```python
import numpy as np
from mojo_newuoa import minimize


def rosenbrock(x):
    return 100.0 * (x[1] - x[0] ** 2) ** 2 + (1.0 - x[0]) ** 2


result = minimize(
    rosenbrock,
    np.array([-1.2, 1.0]),
    rhobeg=1.0,
    rhoend=1.0e-8,
)

print(result.x)
print(result.fun, result.nfev)
```

Save this as `example.py` and run it with `pixi run python example.py`.
The checked example converges to `[1. 1.]`.

`rhobeg` defaults to `1e7` because that is the default in the source
`min_newuoa` function. In most applications it should be set to the expected
scale of a useful initial step, as in the example.

## How it works

The optimizer and all model updates execute in one Mojo compilation unit.
NumPy owns the contiguous Float64 input, workspace, and result metadata buffers.
Their addresses and element counts cross a small C ABI as 64-bit integers.
Mojo validates the counts and non-null addresses before reconstructing typed
pointers.
Matrices use the upstream column-major flat layout so the translated indexing
and update order remain recognizable.

Python turns the objective into a thin C-ABI callback with `ctypes`. Mojo calls
that pointer directly for each trial point; no C or C++ production shim is
involved. The owned contiguous NumPy parameter buffer is reused directly as the
read-only callback view instead of rebuilding a ctypes/NumPy wrapper on every
evaluation. The binding keeps every NumPy buffer and the callback alive for the
entire native call, rejects lossy complex and large-integer inputs, and
propagates callback exceptions. Function evaluation count and final objective
return in a three-element metadata buffer.

Trial-vector formation uses native-width Float64 SIMD with a scalar remainder.
Numerically sensitive reductions retain their original scalar accumulation
order. CPU threading is intentionally not used: the measured dimensions are
small, and the larger loops either update shared model state or feed immediate
reductions, so thread-launch overhead would dominate. A GPU path is also
intentionally omitted. The core kernels are matrix-vector and rank-update
operations with arithmetic intensity well below two floating-point operations
per byte, interleaved with sequential trust-region decisions and Python
callbacks; device transfer and launch overhead would make them slower.

Parity tests compile an exact copy of the bundled upstream header in
`tests/reference/upstream_newuoa.h` and compare final parameters, objective
values, evaluation-budget behavior, constant and ill-conditioned objectives,
the default interpolation set, and every valid custom interpolation count for
small dimensions. Boundary tests cover short native buffers, SIMD remainder
dimensions, input narrowing, non-finite values, callback lifetime/view
behavior, and exception propagation. Intermediate paths can diverge slightly
because the C++ and Mojo compilers contract floating-point operations
differently; assertions are scaled to the requested final trust-region radius.

## Benchmarks

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz,
Linux x86-64, with the benchmark's BLAS objective restricted to one thread.
Times are medians of nine warm, alternating paired runs and include Python
callback cost. The reference is the bundled upstream C++ header compiled with
`-O3`. Different evaluation counts reflect near-tied model-step choices.

| Benchmark | Upstream C++ (ms) | Mojo (ms) | Speedup | C++/Mojo nfev |
|---|---:|---:|---:|---:|
| Rosenbrock 2D | 1.077 | 1.470 | 0.73x | 184/184 |
| Chained Rosenbrock 8D | 13.435 | 21.497 | 0.62x | 699/681 |
| Dense quadratic 16D | 7.486 | 13.398 | 0.56x | 299/299 |

On this run, the Mojo/Python binding is slower than the optimized C++ reference
in all three cases. These timings include the same Python callback path on both
sides; they are reproducibility data, not a general speed claim.
