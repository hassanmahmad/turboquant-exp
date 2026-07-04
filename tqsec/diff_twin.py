"""Differentiable PyTorch twin of the audited TurboQuant path.

Used for T2 training. The forward pass matches the NumPy reference; gradients use
straight-through estimators for scalar bins and QJL signs.
"""

import os
import sys
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

_VENDOR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party")
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)

from transformers.cache_utils import DynamicCache, DynamicLayer  # noqa: E402
from turboquant.core import TurboQuantMSE, TurboQuantProd  # noqa: E402
from turboquant.qjl import generate_projection  # noqa: E402
from turboquant.rotation import generate_rotation  # noqa: E402
from turboquant.scalar_quantizer import compute_centroids  # noqa: E402


@dataclass(frozen=True)
class TorchQuantizedMSE:
    indices: torch.Tensor
    norms: torch.Tensor
    values: torch.Tensor


@dataclass(frozen=True)
class TorchQuantizedProd:
    mse: TorchQuantizedMSE
    qjl_signs: torch.Tensor
    residual_norm: torch.Tensor
    input_norm: torch.Tensor
    values: torch.Tensor


def _normalize(x: torch.Tensor):
    norms = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    safe = torch.where(norms < 1e-10, torch.ones_like(norms), norms)
    return x / safe, norms.squeeze(-1)


def _scale(x_hat: torch.Tensor, norms: torch.Tensor):
    if norms.ndim == 0:
        return x_hat * norms
    return x_hat * norms.unsqueeze(-1)


def _as_float_tensor(x):
    return torch.as_tensor(np.asarray(x), dtype=torch.float32)


class TurboQuantMSETwin(nn.Module):
    """Rotation + Lloyd-Max scalar quantizer with STE gradients."""

    def __init__(self, d: int, b: int, seed: int | None = 42, ste: bool = True):
        super().__init__()
        if b < 1 or b > 8:
            raise ValueError(f"b must be in [1, 8], got {b}")
        if d < 3:
            raise ValueError(f"d must be >= 3, got {d}")
        centroids, boundaries = compute_centroids(d, b)
        self.d = d
        self.b = b
        self.ste = ste
        self.register_buffer("rotation", _as_float_tensor(generate_rotation(d, seed)))
        self.register_buffer("centroids", _as_float_tensor(centroids))
        self.register_buffer("boundaries", _as_float_tensor(boundaries))

    @classmethod
    def from_reference(cls, ref: TurboQuantMSE, ste: bool = True):
        obj = cls(ref.d, ref.b, seed=0, ste=ste)
        obj.rotation.data.copy_(_as_float_tensor(ref.rotation))
        obj.centroids.data.copy_(_as_float_tensor(ref.centroids))
        obj.boundaries.data.copy_(_as_float_tensor(ref.boundaries))
        return obj

    def _dequantize_scalar(self, y: torch.Tensor):
        idx = torch.bucketize(y.detach(), self.boundaries[1:-1])
        hard = self.centroids[idx]
        if not self.ste:
            return hard, idx
        return y + (hard - y).detach(), idx

    def quantize(self, x: torch.Tensor) -> TorchQuantizedMSE:
        x = x.to(dtype=torch.float32)
        x_hat, norms = _normalize(x)
        y = x_hat @ self.rotation.T
        y_hat, idx = self._dequantize_scalar(y)
        return TorchQuantizedMSE(indices=idx, norms=norms, values=y_hat)

    def dequantize(self, q: TorchQuantizedMSE) -> torch.Tensor:
        x_hat = q.values @ self.rotation
        return _scale(x_hat, q.norms)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dequantize(self.quantize(x))

    def validate_against_reference(self, n: int = 64, seed: int = 0, atol: float = 2e-5) -> dict:
        rng = np.random.default_rng(seed)
        x_np = rng.standard_normal((n, self.d)).astype(np.float32)
        ref = TurboQuantMSE(self.d, self.b, seed=None)
        ref.rotation = self.rotation.detach().cpu().numpy()
        ref.centroids = self.centroids.detach().cpu().numpy()
        ref.boundaries = self.boundaries.detach().cpu().numpy()
        ref_out = ref.dequantize(ref.quantize(x_np))
        twin_out = self(torch.from_numpy(x_np)).detach().cpu().numpy()
        max_abs = float(np.max(np.abs(ref_out - twin_out)))
        return {"ok": max_abs <= atol, "max_abs": max_abs, "atol": atol}


class TurboQuantProdTwin(nn.Module):
    """MSE twin plus QJL residual with STE signs."""

    def __init__(self, d: int, b: int, seed: int | None = 42, ste: bool = True,
                 use_qjl: bool = True):
        super().__init__()
        if b < 2:
            raise ValueError(f"TurboQuantProd requires b >= 2, got {b}")
        self.d = d
        self.b = b
        self.ste = ste
        self.use_qjl = use_qjl
        self.mse = TurboQuantMSETwin(d, b - 1, seed=seed, ste=ste)
        qjl_seed = seed + 1000 if seed is not None else None
        self.register_buffer("S", _as_float_tensor(generate_projection(d, qjl_seed)))

    @classmethod
    def from_reference(cls, ref: TurboQuantProd, ste: bool = True, use_qjl: bool = True):
        obj = cls(ref.d, ref.b, seed=0, ste=ste, use_qjl=use_qjl)
        obj.mse = TurboQuantMSETwin.from_reference(ref.mse_quantizer, ste=ste)
        obj.S.data.copy_(_as_float_tensor(ref.S))
        return obj

    def _qjl_sign(self, residual: torch.Tensor):
        projected = residual @ self.S.T
        hard = torch.sign(projected.detach())
        hard = torch.where(hard == 0, torch.ones_like(hard), hard)
        if not self.ste:
            return hard
        return projected + (hard - projected).detach()

    def quantize(self, x: torch.Tensor) -> TorchQuantizedProd:
        x = x.to(dtype=torch.float32)
        x_hat, input_norm = _normalize(x)
        q_mse = self.mse.quantize(x_hat)
        x_mse = self.mse.dequantize(q_mse)
        residual = x_hat - x_mse
        residual_norm = torch.linalg.vector_norm(residual, dim=-1)
        signs = self._qjl_sign(residual)
        return TorchQuantizedProd(
            mse=q_mse,
            qjl_signs=signs,
            residual_norm=residual_norm,
            input_norm=input_norm,
            values=x_mse,
        )

    def dequantize(self, q: TorchQuantizedProd) -> torch.Tensor:
        if self.use_qjl:
            scale = (np.pi / 2.0) ** 0.5 / self.d
            residual = q.qjl_signs @ self.S
            if q.residual_norm.ndim == 0:
                residual = scale * q.residual_norm * residual
            else:
                residual = scale * q.residual_norm.unsqueeze(-1) * residual
            x_hat = q.values + residual
        else:
            x_hat = q.values
        return _scale(x_hat, q.input_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dequantize(self.quantize(x))

    def validate_against_reference(self, n: int = 64, seed: int = 0, atol: float = 2e-5) -> dict:
        rng = np.random.default_rng(seed)
        x_np = rng.standard_normal((n, self.d)).astype(np.float32)
        ref = TurboQuantProd(self.d, self.b, seed=None)
        ref.mse_quantizer.rotation = self.mse.rotation.detach().cpu().numpy()
        ref.mse_quantizer.centroids = self.mse.centroids.detach().cpu().numpy()
        ref.mse_quantizer.boundaries = self.mse.boundaries.detach().cpu().numpy()
        ref.S = self.S.detach().cpu().numpy()
        ref_out = ref.dequantize(ref.quantize(x_np))
        twin_out = self(torch.from_numpy(x_np)).detach().cpu().numpy()
        max_abs = float(np.max(np.abs(ref_out - twin_out)))
        return {"ok": max_abs <= atol, "max_abs": max_abs, "atol": atol}


def make_twin(d: int, b: int, *, mode: str = "paper", seed: int | None = 42,
              ste: bool = True, use_qjl: bool = True) -> nn.Module:
    if mode == "paper":
        return TurboQuantProdTwin(d, b, seed=seed, ste=ste, use_qjl=use_qjl)
    if mode == "mse":
        return TurboQuantMSETwin(d, b, seed=seed, ste=ste)
    raise ValueError("mode must be 'paper' or 'mse'")


class DiffQuantCacheLayer(DynamicLayer):
    """HF cache layer using the differentiable twin."""

    is_sliding = False

    def __init__(self, layer_index: int, key_bits: int = 4, value_bits: int = 4,
                 seed: int = 42, mode: str = "paper", use_qjl: bool = True,
                 is_nc: bool = False):
        super().__init__()
        self.layer_index = layer_index
        self.key_bits = key_bits
        self.value_bits = value_bits
        self.seed = seed
        self.mode = mode
        self.use_qjl = use_qjl
        self.is_nc = is_nc
        self.key_twin = None
        self.value_twin = None

    def _ensure(self, d: int, device):
        if self.value_twin is not None:
            return
        key_mode = "paper" if self.mode == "paper" and self.key_bits >= 2 else "mse"
        self.key_twin = make_twin(d, self.key_bits, mode=key_mode, seed=self.seed,
                                  ste=True, use_qjl=self.use_qjl).to(device)
        self.value_twin = TurboQuantMSETwin(d, self.value_bits, seed=self.seed,
                                            ste=True).to(device)

    def _apply(self, states: torch.Tensor, is_key: bool):
        if self.is_nc:
            return states
        *lead, d = states.shape
        self._ensure(d, states.device)
        twin = self.key_twin if is_key else self.value_twin
        flat = states.reshape(-1, d).to(torch.float32)
        recon = twin(flat).reshape(*lead, d)
        return recon.to(dtype=states.dtype)

    def update(self, key_states, value_states, cache_kwargs=None):
        return super().update(
            self._apply(key_states, True),
            self._apply(value_states, False),
            cache_kwargs,
        )


def make_diff_quant_cache(*, key_bits: int = 4, value_bits: int = 4, seed: int = 42,
                          seed_fn=None, mode: str = "paper", nc_layers=(),
                          use_qjl: bool = True):
    nc_set = set(nc_layers)
    if seed_fn is None:
        def seed_fn(_idx):
            return seed
    counter = {"n": 0}

    class _Layer(DiffQuantCacheLayer):
        def __init__(self):
            idx = counter["n"]
            counter["n"] += 1
            super().__init__(
                layer_index=idx,
                key_bits=key_bits,
                value_bits=value_bits,
                seed=seed_fn(idx),
                mode=mode,
                use_qjl=use_qjl,
                is_nc=idx in nc_set,
            )

    cache = DynamicCache.__new__(DynamicCache)
    cache.__init__()
    cache.layer_class_to_replicate = _Layer
    cache.layers = []
    return cache

