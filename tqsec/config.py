"""Env-driven experiment configuration.

Control variables: quantizer choice, asymmetric K/V bit-widths, the scos-lab mode, the rotation
(Pi) regime and the `-nc` uncompressed-boundary-layer policy. Everything is overridable by
environment variable so Slurm jobs can sweep configurations without code changes.
"""

import os
from dataclasses import dataclass, field

# Allowed values for the enum-like knobs (validated in get_experiment_config).
QUANTIZERS = ("turboquant", "int", "kivi", "fp8", "fp16")  # fp16 = uncompressed baseline
TQ_MODES = ("paper", "mse", "mixed")                        # scos-lab kv_cache modes
PI_REGIMES = ("public", "secret")                           # reused vs per-deployment rotation


@dataclass
class ExperimentConfig:
    # --- model / IO ---
    model_id: str
    model_tag: str
    output_dir: str
    # --- quantizer under test (control harness) ---
    quantizer: str = "turboquant"          # turboquant | int | kivi | fp8 | fp16
    mode: str = "paper"                     # scos-lab mode (paper=Prod-K/MSE-V)
    key_bits: int = 4
    value_bits: int = 4
    # --- the security levers ---
    seed: int = 42                          # rotation/QJL seed
    pi_regime: str = "public"               # public (one reused Pi) | secret (per-deployment Pi)
    nc_layers: tuple = field(default_factory=tuple)  # layer indices left uncompressed (-nc policy)
    # --- runtime ---
    max_new_tokens: int = 64
    head_dim: int = 128                     # usually inferred from the model at load time


def _parse_int_list(raw):
    if not raw:
        return tuple()
    return tuple(int(x) for x in raw.replace(",", " ").split())


def _default_model_path():
    return os.path.join(
        os.environ.get("SCRATCH", os.path.expanduser("~")), "models", "qwen2.5-7b-instruct"
    )


def get_experiment_config():
    model_id = os.environ.get("MODEL_ID") or _default_model_path()
    model_tag = os.environ.get("MODEL_TAG") or os.path.basename(os.path.normpath(model_id))
    output_dir = os.environ.get("OUTPUT_DIR", ".")

    quantizer = os.environ.get("QUANTIZER", "turboquant").lower()
    mode = os.environ.get("TQ_MODE", "paper").lower()
    pi_regime = os.environ.get("PI_REGIME", "public").lower()
    for name, value, allowed in (
        ("QUANTIZER", quantizer, QUANTIZERS),
        ("TQ_MODE", mode, TQ_MODES),
        ("PI_REGIME", pi_regime, PI_REGIMES),
    ):
        if value not in allowed:
            raise ValueError(f"{name}={value!r} not in {allowed}")

    os.makedirs(output_dir, exist_ok=True)
    return ExperimentConfig(
        model_id=model_id,
        model_tag=model_tag,
        output_dir=output_dir,
        quantizer=quantizer,
        mode=mode,
        key_bits=int(os.environ.get("KEY_BITS", "4")),
        value_bits=int(os.environ.get("VALUE_BITS", "4")),
        seed=int(os.environ.get("SEED", "42")),
        pi_regime=pi_regime,
        nc_layers=_parse_int_list(os.environ.get("NC_LAYERS", "")),
        max_new_tokens=int(os.environ.get("MAX_NEW_TOKENS", "64")),
        head_dim=int(os.environ.get("HEAD_DIM", "128")),
    )
