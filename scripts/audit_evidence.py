"""Reproduce the Phase-0 audit evidence recorded in docs/VALIDATION.md.

Run from the repo root:

    python scripts/audit_evidence.py

Measures the two TurboQuant acceptance quantities directly from the vendored
research layer (third_party/turboquant):
  (a) Lloyd-Max total MSE distortion vs the paper, and
  (b) the QJL inner-product bias (-> 0) and variance.

Requires only numpy + scipy (no torch).
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "third_party"))

import numpy as np  # noqa: E402
from turboquant.core import TurboQuantMSE, TurboQuantProd  # noqa: E402
from turboquant.scalar_quantizer import per_coordinate_distortion  # noqa: E402

PAPER_TOTAL_DISTORTION = {1: 0.36, 2: 0.117, 3: 0.03, 4: 0.009}


def gate_a_lloyd_max():
    """Gate (a): total MSE distortion of unit vectors should match the paper."""
    print("Gate (a) -- Lloyd-Max total MSE distortion (unit vectors)")
    print(f"  {'b':>2} {'theory d=512':>13} {'empirical d=256':>16} {'paper':>7}")
    for b in (1, 2, 3, 4):
        theory = 512 * per_coordinate_distortion(512, b)
        empirical = TurboQuantMSE(256, b, seed=42).distortion(n_samples=5000, seed=0)
        print(f"  {b:>2} {theory:>13.4f} {empirical:>16.4f} {PAPER_TOTAL_DISTORTION[b]:>7}")


def gate_b_qjl():
    """Gate (b): QJL inner-product estimate should be unbiased (bias -> 0)."""
    print("\nGate (b) -- QJL inner-product bias (-> 0) and variance (d=128)")
    print(f"  {'b':>2} {'bias':>11} {'variance':>10}")
    for b in (2, 3, 4):
        stats = TurboQuantProd(128, b, seed=42).distortion_ip(n_samples=3000, seed=42)
        print(f"  {b:>2} {stats['bias']:>+11.5f} {stats['variance']:>10.5f}")


if __name__ == "__main__":
    gate_a_lloyd_max()
    gate_b_qjl()
