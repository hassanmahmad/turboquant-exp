"""tqsec.pi_regime: the public/reused vs secret/per-deployment rotation (Π) switch.

TurboQuant reuses a pre-generated Π for speed, but a fixed, attacker-known Π is what makes both
attacks tractable (you can de-rotate and solve against the geometry). A fresh, secret Π per
deployment is a strong barrier. A Π regime is seed management plus a statement of what the
attacker is allowed to know:

  * public_regime(): one reused Π from a fixed, attacker-known seed.
  * secret_regime(): a fresh per-deployment Π whose seed the attacker does not get.

`attacker_seed()` returns the real seed when public (attacker can rebuild Π) and None when secret
(attacker must work without Π). Feed `seed_for_layer` to
`tqsec.quantizers.make_quant_cache(seed_fn=...)` to build a cache under a regime.
"""

import hashlib
import os
from dataclasses import dataclass
from typing import Optional

from tqsec.quantizers import make_quant_cache

PUBLIC = "public"
SECRET = "secret"


def _seed_from_key(key: str, salt: str = "pi") -> int:
    """Deterministic 32-bit seed from a (secret) key string."""
    digest = hashlib.sha256(f"{salt}|{key}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


@dataclass
class PiRegime:
    """A rotation regime: which seed builds Π and whether the attacker knows it.

    base_seed builds Π. In a secret regime base_seed is derived from a per-deployment
    secret (`deployment_key`) that the attacker never receives, so `attacker_seed()`
    returns None and any inversion must work without Π.
    """
    regime: str
    base_seed: int
    per_layer: bool = False
    deployment_key: Optional[str] = None

    @property
    def is_public(self) -> bool:
        return self.regime == PUBLIC

    def seed_for_layer(self, layer_idx: int) -> int:
        """Seed used to build this layer's Π (the defender always knows this)."""
        if not self.per_layer:
            return self.base_seed
        return _seed_from_key(f"{self.base_seed}:{layer_idx}", salt="pi-layer")

    def attacker_seed(self, layer_idx: int = 0) -> Optional[int]:
        """Seed an attacker may legitimately assume: the real one if public, else None."""
        return self.seed_for_layer(layer_idx) if self.is_public else None

    def describe(self) -> dict:
        """JSON-able record for results provenance (never serialize a secret seed raw)."""
        d = {"regime": self.regime, "per_layer": self.per_layer,
             "attacker_knows_pi": self.is_public}
        if self.is_public:
            d["base_seed"] = self.base_seed
        else:
            # record a key fingerprint so runs are distinguishable but Π stays secret
            d["deployment_fingerprint"] = _seed_from_key(str(self.deployment_key), salt="fp")
        return d

    def __str__(self) -> str:
        who = "attacker-known" if self.is_public else "secret"
        return f"PiRegime({self.regime}, {who}, per_layer={self.per_layer})"


def public_regime(seed: int = 42, per_layer: bool = False) -> PiRegime:
    """Reused, attacker-known Π: TurboQuant's default (pre-generated Π for speed)."""
    return PiRegime(PUBLIC, base_seed=seed, per_layer=per_layer)


def secret_regime(deployment_key: Optional[str] = None, per_layer: bool = False) -> PiRegime:
    """Fresh per-deployment secret Π. Seed derived from `deployment_key` (random if None)."""
    if deployment_key is None:
        deployment_key = os.urandom(8).hex()
    return PiRegime(SECRET, base_seed=_seed_from_key(deployment_key),
                    per_layer=per_layer, deployment_key=deployment_key)


def make_cache_under_regime(regime: PiRegime, quantizer: str = "turboquant", *,
                            key_bits: int = 4, value_bits: int = 4, nc_layers=(),
                            mode: str = "paper", recorder=None):
    """Build a quant cache whose per-layer Π follows `regime` (see make_quant_cache)."""
    return make_quant_cache(quantizer, key_bits=key_bits, value_bits=value_bits,
                            nc_layers=nc_layers, mode=mode, recorder=recorder,
                            seed_fn=regime.seed_for_layer)
