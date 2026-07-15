"""tqsec.instrument: dump TurboQuant internals and build the reconstruction error map.

The error map (true KV minus reconstructed KV, summarized per layer, token and channel) feeds
both attacks: T2 uses it to locate the largest, most steerable error; T3 uses the captured codes
to learn an inverter. Wraps the vendored HF cache so a single forward pass populates the map; it
does not reimplement the quantizer.

Two fixes over the vendored `kv_cache.py`: `InstrumentedTurboQuantLayer` moves to CPU for the
NumPy quantizer and restores the original device/dtype (the vendored layer is CPU-only and breaks
on GPU models), and construction mirrors the vendored `make_turboquant_cache` (`__new__` + manual
attribute set) because transformers 4.57.3 rejects `DynamicCache(layer_class_to_replicate=...)`.
"""

import os
import sys
from dataclasses import dataclass, field

import numpy as np
import torch

# The faithful layer is vendored under third_party/; make it importable regardless of cwd.
_VENDOR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party")
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)

from transformers.cache_utils import DynamicCache  # noqa: E402
from turboquant.kv_cache import TurboQuantLayer  # noqa: E402


# --------------------------------------------------------------------------------------
# Captured data
# --------------------------------------------------------------------------------------
@dataclass
class ReconStats:
    """Compact reconstruction stats for one _compress_decompress call (one layer, K or V)."""
    layer_idx: int
    kind: str            # "key" | "value"
    n_tokens: int
    n_heads: int
    head_dim: int
    rel_err: float                  # ||true − recon||_F / ||true||_F
    err_mean: np.ndarray            # (head_dim,) signed mean error per channel  → per-channel bias
    err_absmean: np.ndarray         # (head_dim,) mean |error| per channel
    true_absmean: np.ndarray        # (head_dim,) mean |true| per channel
    tok_err_norm: np.ndarray        # (n_tokens,) error L2 norm per token
    tok_true_norm: np.ndarray       # (n_tokens,) true  L2 norm per token


def _compute_stats(layer_idx, kind, true_t, recon_t):
    """true_t, recon_t: CPU float tensors of shape (batch, heads, seq, head_dim)."""
    b, h, s, d = true_t.shape
    err = true_t - recon_t
    return ReconStats(
        layer_idx=layer_idx,
        kind=kind,
        n_tokens=s,
        n_heads=h,
        head_dim=d,
        rel_err=float(err.norm() / true_t.norm().clamp_min(1e-12)),
        err_mean=err.mean(dim=(0, 1, 2)).numpy(),
        err_absmean=err.abs().mean(dim=(0, 1, 2)).numpy(),
        true_absmean=true_t.abs().mean(dim=(0, 1, 2)).numpy(),
        tok_err_norm=err.pow(2).sum(dim=(1, 3)).sqrt().mean(dim=0).numpy(),
        tok_true_norm=true_t.pow(2).sum(dim=(1, 3)).sqrt().mean(dim=0).numpy(),
    )


def turboquant_matrices(key_q, value_q):
    """Pull the rotation(s) Π and the QJL projection S off TurboQuant quantizers."""
    out = {}
    if hasattr(key_q, "mse_quantizer"):       # TurboQuantProd (paper-mode keys)
        out["key_rotation"] = np.asarray(key_q.mse_quantizer.rotation)
        out["qjl_S"] = np.asarray(key_q.S)
    elif hasattr(key_q, "rotation"):          # TurboQuantMSE keys
        out["key_rotation"] = np.asarray(key_q.rotation)
    if hasattr(value_q, "rotation"):
        out["value_rotation"] = np.asarray(value_q.rotation)
    return out


def _extract_matrices(layer):
    """Best-effort matrix capture; works for any quantizer layer."""
    if hasattr(layer, "matrices"):            # generic codec layer (tqsec.quantizers)
        return layer.matrices()
    if hasattr(layer, "_key_quantizer"):      # InstrumentedTurboQuantLayer
        return turboquant_matrices(layer._key_quantizer, layer._value_quantizer)
    return {}


class ErrorMapRecorder:
    """Shared sink that all instrumented layers write to during a forward pass.

    Layer index is assigned by first-sight order, which equals the model's layer order
    in a standard (in-order) forward pass.
    """

    def __init__(self, store_codes_layers=()):
        self.records = {}                       # (layer_idx, kind) -> list[ReconStats]
        self.matrices = {}                      # layer_idx -> {"key_rotation", "qjl_S", "value_rotation"}
        self.codes = {}                         # (layer_idx, kind) -> list[Quantized*]  (opt-in)
        self.store_codes_layers = set(store_codes_layers)
        self._layer_index = {}                  # id(layer) -> idx
        self._next_idx = 0

    def _idx_for(self, layer):
        explicit = getattr(layer, "layer_index", None)
        if explicit is not None:              # generic codec layer knows its own index
            self._next_idx = max(self._next_idx, explicit + 1)
            return explicit
        key = id(layer)                       # else assign by first-sight (= forward order)
        if key not in self._layer_index:
            self._layer_index[key] = self._next_idx
            self._next_idx += 1
        return self._layer_index[key]

    def record(self, layer, kind, true_cpu, recon_cpu, q):
        idx = self._idx_for(layer)
        self.records.setdefault((idx, kind), []).append(
            _compute_stats(idx, kind, true_cpu, recon_cpu)
        )
        if idx not in self.matrices:
            self.matrices[idx] = _extract_matrices(layer)
        if idx in self.store_codes_layers:
            self.codes.setdefault((idx, kind), []).append(q)

    @property
    def n_layers(self):
        return self._next_idx


# --------------------------------------------------------------------------------------
# Instrumented HF cache layer
# --------------------------------------------------------------------------------------
class InstrumentedTurboQuantLayer(TurboQuantLayer):
    """TurboQuantLayer that records (true, recon, codes) and restores device/dtype."""

    def __init__(self, recorder, **kwargs):
        super().__init__(**kwargs)
        self._recorder = recorder

    def _compress_decompress(self, states, is_key):
        batch, heads, seq_len, head_dim = states.shape
        self._ensure_quantizers(head_dim)
        dtype, device = states.dtype, states.device
        bits = self.key_bits if is_key else self.value_bits

        self._original_bytes += states.numel() * states.element_size()

        true_cpu = states.detach().to("cpu", torch.float32)
        flat = true_cpu.reshape(-1, head_dim).numpy()

        quantizer = self._key_quantizer if is_key else self._value_quantizer
        q = quantizer.quantize(flat)
        recon = quantizer.dequantize(q)

        if is_key and self.mode == "mixed" and hasattr(q, "avg_bits"):
            self._avg_key_bits = q.avg_bits
            self._compressed_bits += int(batch * heads * seq_len * head_dim * q.avg_bits)
        else:
            self._compressed_bits += batch * heads * seq_len * head_dim * bits

        recon_cpu = torch.from_numpy(np.ascontiguousarray(recon)).reshape(
            batch, heads, seq_len, head_dim
        ).float()

        self._recorder.record(self, "key" if is_key else "value", true_cpu, recon_cpu, q)

        return recon_cpu.to(device=device, dtype=dtype)


def make_instrumented_cache(recorder, *, key_bits=4, value_bits=4, seed=42, mode="paper",
                            outlier_threshold=3.0, max_outlier_ratio=0.25):
    """A DynamicCache whose layers are instrumented TurboQuant layers writing to `recorder`.

    Mirrors the vendored `make_turboquant_cache` construction (the kwarg form is rejected
    on transformers 4.57.3).
    """
    class _ITQLayer(InstrumentedTurboQuantLayer):
        def __init__(self):
            super().__init__(recorder, key_bits=key_bits, value_bits=value_bits, seed=seed,
                             mode=mode, outlier_threshold=outlier_threshold,
                             max_outlier_ratio=max_outlier_ratio)

    cache = DynamicCache.__new__(DynamicCache)
    cache.__init__()
    cache.layer_class_to_replicate = _ITQLayer
    cache.layers = []
    return cache


# --------------------------------------------------------------------------------------
# Error-map analysis
# --------------------------------------------------------------------------------------
def _pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size < 2 or a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def layer_findings(stats: ReconStats) -> dict:
    """Error-map findings for one layer/kind: per-channel bias and channel/token concentration."""
    abs_mean_norm = float(np.linalg.norm(stats.err_absmean)) or 1e-12
    return {
        "layer": stats.layer_idx,
        "kind": stats.kind,
        "n_tokens": stats.n_tokens,
        "rel_err": round(stats.rel_err, 4),
        # is the error biased per channel? (0 = zero-mean error, 1 = fully systematic)
        "channel_bias_ratio": round(float(np.linalg.norm(stats.err_mean)) / abs_mean_norm, 4),
        # is error concentrated on high-magnitude channels / tokens?
        "channel_concentration": round(_pearson(stats.true_absmean, stats.err_absmean), 4),
        "token_concentration": round(_pearson(stats.tok_true_norm, stats.tok_err_norm), 4),
    }


def error_map_summary(recorder: ErrorMapRecorder, kind: str = "key", call: int = 0) -> dict:
    """Per-layer findings for the given stream (call=0 is the prefill pass)."""
    per_layer = []
    for layer_idx in range(recorder.n_layers):
        recs = recorder.records.get((layer_idx, kind))
        if recs and call < len(recs):
            per_layer.append(layer_findings(recs[call]))
    if not per_layer:
        return {"kind": kind, "layers": []}
    return {
        "kind": kind,
        "layers": per_layer,
        "mean_rel_err": round(float(np.mean([l["rel_err"] for l in per_layer])), 4),
        "mean_channel_bias_ratio": round(float(np.mean([l["channel_bias_ratio"] for l in per_layer])), 4),
        "mean_channel_concentration": round(float(np.mean([l["channel_concentration"] for l in per_layer])), 4),
    }
