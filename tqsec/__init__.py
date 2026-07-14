"""tqsec — the thin layer we own on top of the vendored TurboQuant research layer.

Contains the instrumentation (error map, code dumps), the control harness
({TurboQuant, INT3/4, KIVI, FP8}), the public-vs-secret rotation (Pi) switch,
the PyTorch differentiable twin for T2, the T3 inverter helpers, and shared
metrics/benchmarks.

The faithful quantizer itself lives in `third_party/turboquant/` (read-only,
audit-passed; see docs/VALIDATION.md). Nothing here reimplements the algorithm.
"""

__version__ = "0.1.0"
