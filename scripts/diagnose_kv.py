"""Diagnose why quantizers break on a model: KV outlier profile + per-codec key error.

Runs one forward pass, pulls the true (FP) K/V from the cache, and reports:
  - global max|K|/|V| and how much of K exceeds the fp8 e4m3 range (448) -> fp8 overflow,
  - per-channel outlier ratio (max channel magnitude / median) -> why per-token INT dies,
  - per-codec relative error on the true keys at 8-bit -> which codecs handle the outliers.

Run on a GPU node (offline):
  MODEL_ID=$SCRATCH/models/qwen2.5-7b-instruct python scripts/diagnose_kv.py
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "third_party"))
for _k in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
    os.environ.setdefault(_k, "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from tqsec.config import get_experiment_config  # noqa: E402
from tqsec.benchmarks import build_needle_prompt, NeedleConfig  # noqa: E402
from tqsec.quantizers import make_codec  # noqa: E402
from tqsec.metrics import relative_error  # noqa: E402

FP8_E4M3_MAX = 448.0


def get_true_kv(cache, i):
    """Extract layer i's true K,V from a DynamicCache (handles the layered + legacy APIs)."""
    if hasattr(cache, "layers"):
        return cache.layers[i].keys, cache.layers[i].values
    return cache.key_cache[i], cache.value_cache[i]


def n_layers(cache):
    return len(cache.layers) if hasattr(cache, "layers") else len(cache.key_cache)


def main():
    cfg = get_experiment_config()
    length = int(os.environ.get("DIAG_LENGTH", "512"))
    print(f"Loading {cfg.model_id} ...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id, dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True).eval()
    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    count = lambda s: len(tok(s, add_special_tokens=False).input_ids)
    prompt = build_needle_prompt(NeedleConfig(), length, 0.5, count)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inputs, use_cache=True)          # model builds its own FP DynamicCache
    cache = out.past_key_values
    nl = n_layers(cache)

    # test at the SANITY bit-widths (turbo k3v4 key path, int3, kivi3, fp8)
    codecs = {
        "turbo_k3":  make_codec("turboquant", key_bits=3, value_bits=4),
        "turbo_mix": make_codec("turboquant", key_bits=3, value_bits=4, mode="mixed"),
        "int3":      make_codec("int", key_bits=3, value_bits=3),
        "kivi3":     make_codec("kivi", key_bits=3, value_bits=3),
        "fp8":       make_codec("fp8"),
    }
    OUTLIER_MULT = 20.0
    kmax = vmax = 0.0
    over448 = total = 0
    per_layer = []          # (layer, max|K|, channel_outlier_ratio)
    rel = {n: [] for n in codecs}
    err_out = {n: [] for n in codecs}    # mean |err| on outlier channels
    err_bulk = {n: [] for n in codecs}   # mean |err| on the rest

    for i in range(nl):
        K, V = get_true_kv(cache, i)
        K = K.detach().float().cpu()
        V = V.detach().float().cpu()
        kmax = max(kmax, K.abs().max().item())
        vmax = max(vmax, V.abs().max().item())
        over448 += int((K.abs() > FP8_E4M3_MAX).sum().item())
        total += K.numel()
        ch_max = K.abs().amax(dim=(0, 1, 2))            # (head_dim,)
        med = ch_max.median()
        ratio = (ch_max.max() / (med + 1e-9)).item()
        per_layer.append((i, K.abs().max().item(), ratio))
        outlier_ch = ch_max > OUTLIER_MULT * med        # (head_dim,) bool
        for name, codec in codecs.items():
            recon = codec.recon(K, is_key=True)
            rel[name].append(relative_error(K, recon))
            aerr_ch = (K - recon).abs().mean(dim=(0, 1, 2))   # (head_dim,)
            if bool(outlier_ch.any()):
                err_out[name].append(aerr_ch[outlier_ch].mean().item())
            err_bulk[name].append(aerr_ch[~outlier_ch].mean().item())

    per_layer.sort(key=lambda x: -x[1])
    print(f"\nlayers={nl}  seq={inputs.input_ids.shape[1]}")
    print(f"max|K|={kmax:.1f}   max|V|={vmax:.1f}   (fp8 e4m3 saturates at {FP8_E4M3_MAX:.0f})")
    print(f"|K| over fp8 range: {over448}/{total} = {100 * over448 / max(total,1):.3f}%")
    print("\ntop-5 layers by max|K|  (layer : max|K| : per-channel outlier ratio):")
    for i, mx, ratio in per_layer[:5]:
        print(f"  L{i:<3} max|K|={mx:8.1f}   outlier_ratio={ratio:7.1f}x")
    print(f"\nper-codec KEY error at sanity bit-widths (outlier channels = >{OUTLIER_MULT:.0f}x median):")
    print(f"  {'codec':<9} {'rel_err':>8} {'err@outlier':>12} {'err@bulk':>9}")
    for name in codecs:
        eo = np.mean(err_out[name]) if err_out[name] else float("nan")
        print(f"  {name:<9} {np.mean(rel[name]):>8.4f} {eo:>12.4f} {np.mean(err_bulk[name]):>9.4f}")
    print("\nreading: if kivi3 keeps err@outlier low while int3/turbo_k3 are high, the outlier")
    print("channels are what break exact retrieval — and only per-channel quant preserves them.")


if __name__ == "__main__":
    main()
