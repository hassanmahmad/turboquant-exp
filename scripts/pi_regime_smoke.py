"""Smoke test for tqsec.pi_regime — the public vs secret rotation (Π) switch.

Demonstrates the security lever directly: a simple code-inversion attack (match a token's
quantized codes back to a vocabulary) succeeds when the attacker knows Π (public regime) and
collapses to chance when Π is a per-deployment secret. Also checks regime mechanics and the
per-layer-Π cache wiring. Run from the repo root:

    python scripts/pi_regime_smoke.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "third_party"))

import numpy as np  # noqa: E402

from tqsec.pi_regime import public_regime, secret_regime, make_cache_under_regime  # noqa: E402
from tqsec.instrument import ErrorMapRecorder  # noqa: E402
from turboquant.core import TurboQuantMSE  # noqa: E402
import torch  # noqa: E402


def inversion_accuracy(defender_seed, attacker_seed, d=128, b=4, n_vocab=200, data_seed=0):
    """Defender quantizes a vocab under its Π; attacker reconstructs codes under its assumed
    Π and nearest-neighbour-matches back to the vocab. Returns top-1 accuracy."""
    rng = np.random.default_rng(data_seed)
    vocab = rng.standard_normal((n_vocab, d)).astype(np.float32)

    codes = TurboQuantMSE(d, b, seed=defender_seed).quantize(vocab)        # indices under defender Π
    recon = TurboQuantMSE(d, b, seed=attacker_seed).dequantize(codes)      # decoded under attacker Π

    R = recon / (np.linalg.norm(recon, axis=1, keepdims=True) + 1e-9)
    V = vocab / (np.linalg.norm(vocab, axis=1, keepdims=True) + 1e-9)
    pred = (R @ V.T).argmax(axis=1)
    return float(np.mean(pred == np.arange(n_vocab)))


def main():
    pub = public_regime(seed=42)
    sec = secret_regime(deployment_key="deployment-A")

    # --- the security lever ---
    # public: attacker knows the seed -> uses the real Π
    acc_public = inversion_accuracy(pub.seed_for_layer(0), pub.attacker_seed(0))
    # secret: attacker_seed() is None -> attacker must GUESS (tries the public default 42)
    assert sec.attacker_seed(0) is None
    acc_secret = inversion_accuracy(sec.seed_for_layer(0), attacker_seed=42)
    chance = 1.0 / 200

    print("=== inversion attack: does knowing Π help? ===")
    print(f"  public  Π (attacker knows seed) : top-1 = {acc_public:.3f}")
    print(f"  secret  Π (attacker must guess) : top-1 = {acc_secret:.3f}  (chance ~ {chance:.3f})")
    assert acc_public > 0.7, f"public inversion should be easy, got {acc_public}"
    assert acc_secret < 0.05, f"secret inversion should be ~chance, got {acc_secret}"
    print("  [ok] secret Π closes the leak: inversion drops from easy to chance")

    # --- regime mechanics ---
    assert public_regime(7).attacker_seed(0) == 7        # public: attacker knows the seed
    assert secret_regime("A").attacker_seed(0) is None   # secret: unknown
    assert secret_regime("A").base_seed != secret_regime("B").base_seed  # per-deployment Π
    assert public_regime(42).base_seed == public_regime(42).base_seed    # reused/reproducible
    print("\n[ok] mechanics: public exposes seed, secret hides it, deployments get distinct Π")

    # --- per-layer Π wiring through the cache ---
    def layer_rotations(regime, n=3):
        rec = ErrorMapRecorder()
        cache = make_cache_under_regime(regime, key_bits=3, value_bits=4, recorder=rec)
        for li in range(n):
            kv = torch.randn(1, 8, 16, 128)
            cache.update(kv, kv, li)
        return [rec.matrices[i]["key_rotation"] for i in range(n)]

    shared = layer_rotations(public_regime(42, per_layer=False))
    perlayer = layer_rotations(secret_regime("A", per_layer=True))
    assert np.allclose(shared[0], shared[1]), "public non-per-layer should reuse one Π"
    assert not np.allclose(perlayer[0], perlayer[1]), "secret per-layer should differ per layer"
    print("[ok] cache wiring: reused-Π shares one rotation; per-layer-secret differs per layer")

    print(f"\nregime records (for results provenance):")
    print(f"  {pub.describe()}")
    print(f"  {sec.describe()}  (deployment_key kept secret; only a fingerprint is logged)")
    print("\nSMOKE PASSED")


if __name__ == "__main__":
    main()
