"""Smoke test for tqsec.quantizers, the {TurboQuant, INT, KIVI, FP8} control harness.

Usage:
    python scripts/quantizers_smoke.py

Runs identical synthetic KV through each codec and compares error maps, checks the
-nc passthrough, and validates bit accounting.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402

from tqsec.instrument import ErrorMapRecorder, error_map_summary  # noqa: E402
from tqsec.quantizers import QUANTIZERS, make_quant_cache  # noqa: E402


def synth_kv(seed, n_heads=8, seq=64, head_dim=128):
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(1, n_heads, seq, head_dim, generator=g)
    token_scale = 0.5 + 3.5 * torch.rand(1, n_heads, seq, 1, generator=g)
    return base * token_scale


def run(quantizer, L=4, nc_layers=(), key_bits=3, value_bits=4):
    rec = ErrorMapRecorder()
    cache = make_quant_cache(quantizer, key_bits=key_bits, value_bits=value_bits,
                             nc_layers=nc_layers, recorder=rec)
    for li in range(L):
        k, v = synth_kv(li), synth_kv(100 + li)
        rk, rv = cache.update(k, v, li)
        assert rk.shape == k.shape and rk.dtype == k.dtype and rk.device == k.device
    return rec, cache


def main():
    print("=== cross-quantizer error map (synthetic KV, key=3-bit, value=4-bit) ===")
    print(f"  {'quantizer':>10} {'key_rel':>8} {'val_rel':>8} {'key_chConc':>11} {'key_ratio':>10}")
    key_rel = {}
    for qz in QUANTIZERS:
        rec, cache = run(qz)
        ksum = error_map_summary(rec, "key")
        vsum = error_map_summary(rec, "value")
        key_rel[qz] = ksum["mean_rel_err"]
        ratio = cache.layers[1].compression_ratio
        print(f"  {qz:>10} {ksum['mean_rel_err']:>8} {vsum['mean_rel_err']:>8} "
              f"{ksum['mean_channel_concentration']:>11} {ratio:>10.2f}x")

    # FP8 (8-bit) must beat 3-bit INT and 3-bit TurboQuant on raw reconstruction
    assert key_rel["fp8"] < key_rel["int"], (key_rel["fp8"], key_rel["int"])
    assert key_rel["fp8"] < key_rel["turboquant"], (key_rel["fp8"], key_rel["turboquant"])
    assert all(0.0 < v < 1.5 for v in key_rel.values()), key_rel
    print("\n[ok] FP8(8-bit) < INT3 and < TurboQuant3 on key reconstruction; all rel_err in (0,1.5)")
    print("     -> the harness distinguishes codecs: this is the TurboQuant-specificity control")

    # -nc passthrough: layers 0 and 3 left uncompressed
    rec, cache = run("turboquant", L=4, nc_layers={0, 3})
    layers = error_map_summary(rec, "key")["layers"]
    rel_by_layer = {l["layer"]: l["rel_err"] for l in layers}
    assert rel_by_layer[0] == 0.0 and rel_by_layer[3] == 0.0, rel_by_layer
    assert rel_by_layer[1] > 0 and rel_by_layer[2] > 0, rel_by_layer
    assert cache.layers[0].compression_ratio == 1.0 and cache.layers[1].compression_ratio > 1.0
    print(f"\n[ok] -nc passthrough: layers {{0,3}} rel_err=0 (ratio 1.0x), layers {{1,2}} compressed "
          f"({cache.layers[1].compression_ratio:.2f}x)")

    print("\nSMOKE PASSED")


if __name__ == "__main__":
    main()
