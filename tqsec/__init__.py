"""tqsec: thin instrumentation and control layer over the vendored TurboQuant research layer.

Provides the error-map instrumentation, the {TurboQuant, INT3/4, KIVI, FP8} control harness, the
public/secret rotation (Pi) switch, the differentiable twin, the token inverter and shared metrics
and benchmarks. The quantizer lives in `third_party/turboquant/`; nothing here reimplements it.
"""

__version__ = "0.1.0"
