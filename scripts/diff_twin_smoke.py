"""Smoke test for tqsec.diff_twin.

Usage:
    python scripts/diff_twin_smoke.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402

from tqsec.diff_twin import (  # noqa: E402
    TurboQuantMSETwin,
    TurboQuantProdTwin,
    make_diff_quant_cache,
)


def main():
    d = 32

    mse = TurboQuantMSETwin(d, 3, seed=7)
    prod = TurboQuantProdTwin(d, 4, seed=7)
    mse_check = mse.validate_against_reference(n=16, seed=1)
    prod_check = prod.validate_against_reference(n=16, seed=1)
    assert mse_check["ok"], mse_check
    assert prod_check["ok"], prod_check

    x = torch.randn(8, d, requires_grad=True)
    loss = prod(x).pow(2).mean()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert float(x.grad.abs().sum()) > 0.0

    cache = make_diff_quant_cache(key_bits=4, value_bits=4, seed=7)
    k = torch.randn(1, 2, 5, d, requires_grad=True)
    v = torch.randn(1, 2, 5, d, requires_grad=True)
    rk, rv = cache.update(k, v, 0)
    assert rk.shape == k.shape and rv.shape == v.shape
    assert rk.dtype == k.dtype and rv.dtype == v.dtype

    print("MSE reference:", mse_check)
    print("Prod reference:", prod_check)
    print("Gradient L1:", round(float(x.grad.abs().sum()), 6))
    print("SMOKE PASSED")


if __name__ == "__main__":
    main()

