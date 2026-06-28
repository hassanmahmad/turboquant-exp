"""tqsec.quantizers — the {TurboQuant, INT, KIVI, FP8} control harness.

Every later experiment runs against all four through one interface, so we can ask:
*is the effect TurboQuant-specific (rotation + QJL geometry), or does it appear under
coarse rounding too?* If an attack works equally under plain INT/KIVI, the novelty claim
is dead — so this harness is a non-negotiable control (PROJECT_PLAN.md §6).

Design: a single device-safe `QuantCacheLayer` (HF `DynamicLayer`) delegates to a pluggable
`Codec`. The layer also implements the **`-nc` policy** (skip compression on configured
boundary layers) that the vendored research layer lacks, does bit accounting, and can record
to a `tqsec.instrument.ErrorMapRecorder` — so error maps are directly comparable across codecs.

Codecs operate on CPU float tensors of shape (batch, heads, seq, head_dim):
  * IntCodec   — plain uniform-affine, per-token (the "coarse rounding" baseline).
  * KiviCodec  — per-channel keys, per-token values (KIVI's defining asymmetry; no rotation).
  * Fp8Codec   — cast to float8_e4m3 (the "is TurboQuant even worth it?" baseline; always 8-bit).
  * TurboQuantCodec — the vendored faithful quantizer (paper/mse mode).
"""

import os
import sys

import numpy as np
import torch

_VENDOR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party")
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)

from transformers.cache_utils import DynamicCache, DynamicLayer  # noqa: E402
from turboquant.core import TurboQuantMSE, TurboQuantProd  # noqa: E402

from tqsec.instrument import turboquant_matrices  # noqa: E402

QUANTIZERS = ("turboquant", "int", "kivi", "fp8")


# --------------------------------------------------------------------------------------
# Codecs
# --------------------------------------------------------------------------------------
def _uniform_quant_dequant(x, bits, reduce_dims):
    """Asymmetric (min/max) uniform quantize-dequantize along reduce_dims."""
    qmax = (1 << bits) - 1
    lo = torch.amin(x, dim=reduce_dims, keepdim=True)
    hi = torch.amax(x, dim=reduce_dims, keepdim=True)
    scale = (hi - lo).clamp_min(1e-12) / qmax
    q = torch.round((x - lo) / scale)
    return q * scale + lo


class IntCodec:
    """Plain uniform-affine quantization, per-token (per (batch,head,token) vector)."""
    name = "int"

    def __init__(self, key_bits=4, value_bits=4, **_):
        self.key_bits, self.value_bits = key_bits, value_bits

    def bits_for(self, is_key):
        return self.key_bits if is_key else self.value_bits

    def recon(self, x, is_key):
        return _uniform_quant_dequant(x, self.bits_for(is_key), reduce_dims=(-1,))

    def matrices(self):
        return {}


class KiviCodec:
    """KIVI-style: keys quantized per-channel (over tokens), values per-token. No rotation.

    Simplification vs full KIVI: no streaming residual / fp16 recent-token window (an inference
    optimization, orthogonal to the rotation-vs-no-rotation geometry this control isolates).
    """
    name = "kivi"

    def __init__(self, key_bits=4, value_bits=4, **_):
        self.key_bits, self.value_bits = key_bits, value_bits

    def bits_for(self, is_key):
        return self.key_bits if is_key else self.value_bits

    def recon(self, x, is_key):
        if is_key:                                  # per-channel: one scale per channel over tokens
            return _uniform_quant_dequant(x, self.key_bits, reduce_dims=(2,))
        return _uniform_quant_dequant(x, self.value_bits, reduce_dims=(-1,))  # per-token

    def matrices(self):
        return {}


class Fp8Codec:
    """FP8 KV: cast to float8_e4m3 and back. Always 8-bit (ignores configured bits)."""
    name = "fp8"
    _DT = torch.float8_e4m3fn

    def __init__(self, **_):
        pass

    def bits_for(self, is_key):
        return 8

    def recon(self, x, is_key):
        return x.to(self._DT).to(torch.float32)

    def matrices(self):
        return {}


class TurboQuantCodec:
    """The vendored faithful TurboQuant (paper mode: Prod keys, MSE values)."""
    name = "turboquant"

    def __init__(self, key_bits=4, value_bits=4, seed=42, mode="paper", **_):
        self.key_bits, self.value_bits, self.seed, self.mode = key_bits, value_bits, seed, mode
        self._kq = self._vq = None

    def _ensure(self, head_dim):
        if self._vq is not None:
            return
        self._vq = TurboQuantMSE(head_dim, self.value_bits, self.seed)
        if self.mode == "paper" and self.key_bits >= 2:
            self._kq = TurboQuantProd(head_dim, self.key_bits, self.seed)
        else:
            self._kq = TurboQuantMSE(head_dim, self.key_bits, self.seed)

    def bits_for(self, is_key):
        return self.key_bits if is_key else self.value_bits

    def recon(self, x, is_key):
        b, h, s, d = x.shape
        self._ensure(d)
        q = self._kq if is_key else self._vq
        out = q.dequantize(q.quantize(x.reshape(-1, d).numpy()))
        return torch.from_numpy(np.ascontiguousarray(out)).reshape(b, h, s, d).float()

    def matrices(self):
        return turboquant_matrices(self._kq, self._vq) if self._vq is not None else {}


def make_codec(quantizer, *, key_bits=4, value_bits=4, seed=42, mode="paper"):
    quantizer = quantizer.lower()
    if quantizer == "turboquant":
        return TurboQuantCodec(key_bits=key_bits, value_bits=value_bits, seed=seed, mode=mode)
    if quantizer == "int":
        return IntCodec(key_bits=key_bits, value_bits=value_bits)
    if quantizer == "kivi":
        return KiviCodec(key_bits=key_bits, value_bits=value_bits)
    if quantizer == "fp8":
        return Fp8Codec()
    raise ValueError(f"unknown quantizer {quantizer!r}; choose from {QUANTIZERS}")


# --------------------------------------------------------------------------------------
# HF cache layer
# --------------------------------------------------------------------------------------
class QuantCacheLayer(DynamicLayer):
    """Device-safe HF cache layer: codec compression + `-nc` passthrough + bit accounting.

    `key_codec` and `value_codec` may be the same object (it dispatches on is_key).
    """
    is_sliding = False

    def __init__(self, codec, layer_index, is_nc=False, recorder=None):
        super().__init__()
        self.codec = codec
        self.layer_index = layer_index
        self.is_nc = is_nc
        self._recorder = recorder
        self.original_bits = 0
        self.compressed_bits = 0

    def _compress_decompress(self, states, is_key):
        dtype, device = states.dtype, states.device
        true_cpu = states.detach().to("cpu", torch.float32)
        self.original_bits += true_cpu.numel() * 16  # FP16 reference

        if self.is_nc:                                # -nc: leave this layer uncompressed
            recon_cpu = true_cpu
            self.compressed_bits += true_cpu.numel() * 16
        else:
            recon_cpu = self.codec.recon(true_cpu, is_key)
            self.compressed_bits += int(true_cpu.numel() * self.codec.bits_for(is_key))

        if self._recorder is not None:
            self._recorder.record(self, "key" if is_key else "value", true_cpu, recon_cpu, None)
        return recon_cpu.to(device=device, dtype=dtype)

    def update(self, key_states, value_states, cache_kwargs=None):
        k = self._compress_decompress(key_states, True)
        v = self._compress_decompress(value_states, False)
        return super().update(k, v, cache_kwargs)

    def matrices(self):
        return self.codec.matrices()

    @property
    def compression_ratio(self):
        return self.original_bits / max(self.compressed_bits, 1)


def make_quant_cache(quantizer="turboquant", *, key_bits=4, value_bits=4, nc_layers=(),
                     seed=42, mode="paper", recorder=None):
    """A DynamicCache whose layers compress with `quantizer`, honoring the `-nc` policy.

    nc_layers: iterable of layer indices left uncompressed (FP16). A single shared codec is
    used across layers (one rotation/seed) — matching TurboQuant's reused-Π default; vary
    `seed` per deployment for the secret-Π regime (see tqsec.pi_regime).
    """
    quantizer = quantizer.lower()
    nc_set = set(nc_layers)
    codec = make_codec(quantizer, key_bits=key_bits, value_bits=value_bits, seed=seed, mode=mode)
    counter = {"n": 0}

    class _QLayer(QuantCacheLayer):
        def __init__(self):
            idx = counter["n"]
            counter["n"] += 1
            super().__init__(codec, layer_index=idx, is_nc=idx in nc_set, recorder=recorder)

    cache = DynamicCache.__new__(DynamicCache)
    cache.__init__()
    cache.layer_class_to_replicate = _QLayer
    cache.layers = []
    return cache
