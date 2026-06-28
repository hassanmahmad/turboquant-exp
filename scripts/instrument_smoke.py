"""Smoke test for tqsec.instrument — exercises the error map through the real
vendored TurboQuant cache (DynamicCache.update path) with synthetic KV.

No model download needed. Run from the repo root:

    python scripts/instrument_smoke.py

Validates mechanics: device/dtype restore, per-layer/token/channel capture, code +
matrix dumps, and that the error map reflects real structure (error scales with token
norm, since TurboQuant normalizes per-vector → error ∝ stored norm).
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch  # noqa: E402

from tqsec.instrument import (  # noqa: E402
    ErrorMapRecorder,
    error_map_summary,
    make_instrumented_cache,
)


def synth_kv(seed, n_heads=8, seq=64, head_dim=128):
    """Synthetic KV with heterogeneous token magnitudes (so tokens differ in norm)."""
    g = torch.Generator().manual_seed(seed)
    base = torch.randn(1, n_heads, seq, head_dim, generator=g)
    token_scale = 0.5 + 3.5 * torch.rand(1, n_heads, seq, 1, generator=g)
    return base * token_scale


def main():
    L = 6
    rec = ErrorMapRecorder(store_codes_layers={0})
    cache = make_instrumented_cache(rec, key_bits=3, value_bits=4, seed=42, mode="paper")

    for layer_idx in range(L):
        k = synth_kv(layer_idx)
        v = synth_kv(100 + layer_idx)
        rk, rv = cache.update(k, v, layer_idx)
        # device/dtype preserved
        assert rk.shape == k.shape and rv.shape == v.shape
        assert rk.device == k.device and rk.dtype == k.dtype

    # --- mechanics asserts ---
    assert rec.n_layers == L, f"expected {L} layers, got {rec.n_layers}"
    for li in range(L):
        assert (li, "key") in rec.records and (li, "value") in rec.records
        assert "key_rotation" in rec.matrices[li], "Π not captured"
        assert "qjl_S" in rec.matrices[li], "QJL matrix not captured (paper mode keys)"
    assert (0, "key") in rec.codes and rec.codes[(0, "key")], "raw codes not captured for layer 0"
    print(f"[ok] {L} layers x (key,value) captured; rotation + QJL matrices + layer-0 codes present")

    # --- device / dtype restore on a separate stream ---
    rec2 = ErrorMapRecorder()
    cache2 = make_instrumented_cache(rec2, key_bits=4, value_bits=4, seed=1, mode="mse")
    for dtype in (torch.float32, torch.bfloat16):
        k = synth_kv(7).to(dtype)
        rk, _ = cache2.update(k, synth_kv(8).to(dtype), 0 if dtype == torch.float32 else 1)
        assert rk.dtype == dtype, f"dtype not restored: {rk.dtype} != {dtype}"
    print("[ok] device/dtype restored (float32 + bfloat16)")
    if torch.cuda.is_available():
        rec3 = ErrorMapRecorder()
        c3 = make_instrumented_cache(rec3, key_bits=4, value_bits=4, mode="paper")
        rk, _ = c3.update(synth_kv(0).cuda(), synth_kv(1).cuda(), 0)
        assert rk.is_cuda, "recon not returned on CUDA"
        print("[ok] CUDA round-trip (vendored kv_cache.py would fail here)")
    else:
        print("[--] no CUDA on this box; device-restore fix verified on CPU only")

    # --- error map ---
    for kind in ("key", "value"):
        summ = error_map_summary(rec, kind=kind)
        print(f"\n=== error map: {kind} stream ({'3-bit Prod+QJL' if kind=='key' else '4-bit MSE'}) ===")
        print(f"  {'layer':>5} {'rel_err':>8} {'ch_bias':>8} {'ch_conc':>8} {'tok_conc':>9}")
        for l in summ["layers"]:
            print(f"  {l['layer']:>5} {l['rel_err']:>8} {l['channel_bias_ratio']:>8} "
                  f"{l['channel_concentration']:>8} {l['token_concentration']:>9}")
        print(f"  mean rel_err={summ['mean_rel_err']}  mean ch_bias={summ['mean_channel_bias_ratio']}")

    # error should scale with token norm (per-vector normalization) -> positive token concentration
    key_tok_conc = [error_map_summary(rec, "key")["layers"][i]["token_concentration"] for i in range(L)]
    assert min(key_tok_conc) > 0, f"expected error to concentrate on high-norm tokens, got {key_tok_conc}"
    print(f"\n[ok] token concentration positive on all layers (error scales with token norm)")
    print("\nSMOKE PASSED")


if __name__ == "__main__":
    main()
