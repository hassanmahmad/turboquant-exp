"""Validate the full stack on a Leonardo compute node.

Usage:
    run via slurm/check_environment.slurm

Checks library versions and CUDA, then confirms the vendored research layer and the
tqsec package import and round-trip, so the cluster can actually run the experiments.
"""

import importlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "third_party"))

print("== libraries ==")
for mod in ("torch", "numpy", "scipy", "transformers", "huggingface_hub", "accelerate", "datasets"):
    try:
        m = importlib.import_module(mod)
        print(f"{mod}={getattr(m, '__version__', 'ok')}")
    except Exception as exc:  # noqa: BLE001
        print(f"{mod}=MISSING ({type(exc).__name__})")

print("== cuda ==")
try:
    import torch
    print(f"torch_cuda_build={torch.version.cuda}")
    print(f"cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda_device_count={torch.cuda.device_count()}")
        print(f"cuda_device_0={torch.cuda.get_device_name(0)}")
except Exception as exc:  # noqa: BLE001
    print(f"torch_error={exc!r}")

print(f"scratch={os.environ.get('SCRATCH')}")

print("== research layer + tqsec ==")
try:
    import numpy as np
    from turboquant.core import TurboQuantProd
    tq = TurboQuantProd(128, 3, seed=1)
    shape = tq.dequantize(tq.quantize(np.ones((4, 128), dtype=np.float32))).shape
    print(f"turboquant_roundtrip=OK shape={shape}")
except Exception as exc:  # noqa: BLE001
    print(f"turboquant_roundtrip=FAIL ({type(exc).__name__}: {exc})")

try:
    from tqsec.quantizers import make_quant_cache  # noqa: F401
    from tqsec.benchmarks import sanity_sweep  # noqa: F401
    print("tqsec_import=OK")
except Exception as exc:  # noqa: BLE001
    print(f"tqsec_import=FAIL ({type(exc).__name__}: {exc})")
