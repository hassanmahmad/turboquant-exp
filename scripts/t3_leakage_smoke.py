"""CPU smoke test for the T3 learned inverter.

Usage:
    python scripts/t3_leakage_smoke.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from tqsec.inversion import InverterConfig, train_linear_inverter, train_test_indices  # noqa: E402


def main():
    rng = np.random.default_rng(7)
    vocab, dim, n = 12, 10, 240
    prototypes = rng.normal(size=(vocab, dim)).astype("float32")
    y = rng.integers(0, vocab, size=n, dtype=np.int64)
    x = prototypes[y] + 0.05 * rng.normal(size=(n, dim)).astype("float32")
    train_idx, test_idx = train_test_indices(n, train_fraction=0.7, seed=0)
    metrics = train_linear_inverter(
        x[train_idx],
        y[train_idx],
        x[test_idx],
        y[test_idx],
        token_embeddings=prototypes,
        config=InverterConfig(epochs=80, lr=0.08, seed=0),
    )
    assert metrics["token_recovery"] > 0.9, metrics
    assert metrics["semantic_recovery"] > 0.9, metrics
    print("[ok] T3 inverter recovers synthetic token/semantic signal")


if __name__ == "__main__":
    main()
