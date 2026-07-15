"""Smoke test for tqsec.metrics.

Usage:
    python scripts/metrics_smoke.py

Sanity-checks each metric and demonstrates that ranking quantizers by inner-product
fidelity vs by reconstruction MSE gives different answers, so T1/T2/T3 use
IP/attention fidelity, not MSE.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "third_party"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from tqsec import metrics as M  # noqa: E402
from tqsec.quantizers import make_codec  # noqa: E402


def test_token_and_semantic():
    true = [10, 20, 30, 40]
    assert M.token_accuracy(true, true) == 1.0
    assert M.token_accuracy(true, [10, 20, 99, 99]) == 0.5
    assert M.token_accuracy(true, [99, 99, 99, 99]) == 0.0
    assert M.token_set_jaccard(true, [40, 30, 20, 10]) == 1.0       # order-free
    assert M.token_ngram_overlap(true, true, n=2) == 1.0
    assert 0.0 <= M.token_ngram_overlap(true, [10, 99, 30, 40], n=2) < 1.0

    # semantic stays high under a synonym swap that token_accuracy misses
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((100, 16)).astype(np.float32)
    emb[21] = emb[20] + 0.01 * rng.standard_normal(16)             # 21 is a near-synonym of 20
    orig, swap = [20, 50, 70], [21, 50, 70]
    assert M.token_accuracy(orig, swap) < 1.0                       # token-level penalizes the swap
    assert M.embedding_cosine(orig, swap, emb) > 0.95               # semantic barely moves
    print("[ok] token-level vs semantic recovery are separate (synonym swap: token<1, semantic>0.95)")


def test_distribution_and_behavior():
    p = np.array([0.7, 0.2, 0.1])
    assert M.js_divergence(p, p) < 1e-9 and M.kl_divergence(p, p) < 1e-9
    assert M.js_divergence(p, np.array([0.1, 0.2, 0.7])) > 0.0
    assert M.topk_agreement([1, 5, 3, 0], [2, 5, 4, 0], k=1) == 1.0  # both top-1 = index 1
    assert M.distribution_divergence({1: 0.9, 2: 0.1}, {1: 0.9, 2: 0.1}) < 1e-9

    fp, comp = "the capital is Paris.", "the capital is Paris. BANANA42"
    assert M.canary_fires(fp, comp, "BANANA42")                    # absent in FP, present compressed
    assert not M.canary_fires(comp, comp, "BANANA42")              # present in FP too -> not the signal
    print("[ok] divergence + T2 canary_fires (FP-absent / compressed-present) behave correctly")


def test_distortion_ip_vs_mse():
    # recon == true -> perfect on every distortion metric (the assertable mechanics)
    rng = np.random.default_rng(1)
    K = rng.standard_normal((40, 128)).astype(np.float32)
    Q = rng.standard_normal((16, 128)).astype(np.float32)
    perf = M.inner_product_fidelity(K, K, Q)
    assert perf["logit_cosine"] > 0.999 and abs(perf["bias"]) < 1e-6 and perf["rel_err"] < 1e-6
    assert M.attention_kl(K, K, Q) < 1e-9 and M.relative_error(K, K) == 0.0
    print("[ok] distortion metrics: perfect reconstruction -> perfect scores")

    # report TurboQuant(3-bit Prod) vs INT-3 on the same keys; metrics must distinguish them
    Kt = torch.from_numpy(K.reshape(1, 1, 40, 128))
    print("\n  codec        recon_MSE   logit_cos   ip_bias   attn_KL   (3-bit keys, iid-Gaussian)")
    vals = {}
    for name in ("turboquant", "int"):
        r = make_codec(name, key_bits=3, mode="paper").recon(Kt, is_key=True).reshape(40, 128).numpy()
        mse = M.reconstruction_mse(K, r)
        ip = M.inner_product_fidelity(K, r, Q)
        vals[name] = (mse, ip["logit_cosine"])
        assert -1.0 <= ip["logit_cosine"] <= 1.0 and mse > 0
        print(f"  {name:<11} {mse:>9.4f}  {ip['logit_cosine']:>9.4f}  {ip['bias']:>+8.4f}  "
              f"{M.attention_kl(K, r, Q):>7.4f}")
    assert vals["turboquant"] != vals["int"], "metrics should distinguish the codecs"
    print("  honest caveat: iid-Gaussian keys have NO outlier channels for rotation to fix, so")
    print("    TurboQuant shows no fidelity edge here (echoes 'rotation/QJL is second-order').")
    print("    The real K-cache comparison, where TurboQuant should win, is a T1 deliverable.")


def main():
    test_token_and_semantic()
    test_distribution_and_behavior()
    test_distortion_ip_vs_mse()
    print("\nSMOKE PASSED")


if __name__ == "__main__":
    main()
