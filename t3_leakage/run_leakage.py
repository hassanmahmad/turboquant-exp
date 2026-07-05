"""T3 prompt-leakage experiment: FP-KV baseline vs compressed representations.

This is the required binary test harness: does compression make token inversion
harder or easier than the uncompressed FP key-cache baseline? It keeps exact
token recovery and semantic recovery separate, and it runs TurboQuant under
public and secret Pi regimes.
"""

import json
import os
import sys
import time
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
for _k in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
    os.environ.setdefault(_k, "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from tqsec.config import get_experiment_config  # noqa: E402
from tqsec.inversion import InverterConfig, train_linear_inverter, train_test_indices  # noqa: E402
from tqsec.pi_regime import secret_regime  # noqa: E402
from tqsec.quantizers import make_codec  # noqa: E402


DEFAULT_TEXTS = [
    "The archive key for the north room is a blue copper token kept behind the atlas.",
    "A surveyor marked the old bridge with chalk before the rain reached the valley.",
    "The clinic moved the Tuesday appointment to the smaller office near the garden.",
    "Three notebooks, a brass compass, and a folded map were packed into the case.",
    "The museum guide said the red cabinet contained letters from the first expedition.",
    "During the trial run, the radio operator repeated the harmless code twice.",
    "A careful baker wrote the oven temperature beside the recipe for walnut bread.",
    "The station clock stopped at noon, but the conductor still announced the train.",
]


def get_true_kv(cache, layer_idx):
    if hasattr(cache, "layers"):
        return cache.layers[layer_idx].keys, cache.layers[layer_idx].values
    return cache.key_cache[layer_idx], cache.value_cache[layer_idx]


def _texts_from_env():
    path = os.environ.get("T3_TEXT_FILE")
    if path:
        return [line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()
                if line.strip()]
    raw = os.environ.get("T3_TEXTS")
    if raw:
        return [x.strip() for x in raw.split("||") if x.strip()]
    return DEFAULT_TEXTS


def _config_names():
    raw = os.environ.get("T3_CONFIGS")
    if raw:
        return [x.strip() for x in raw.split(",") if x.strip()]
    return ["fp16", "turbo_k8v4", "turbo_k3v4", "turbo_k3v4_nc", "int3", "kivi3", "fp8"]


def _codec_spec(name):
    if name == "fp16":
        return None
    if name == "turbo_k8v4":
        return ("turboquant", {"key_bits": 8, "value_bits": 4, "mode": "paper"})
    if name == "turbo_k3v4" or name == "turbo_k3v4_nc":
        return ("turboquant", {"key_bits": 3, "value_bits": 4, "mode": "paper"})
    if name == "int3":
        return ("int", {"key_bits": 3, "value_bits": 3})
    if name == "kivi3":
        return ("kivi", {"key_bits": 3, "value_bits": 3})
    if name == "fp8":
        return ("fp8", {})
    raise ValueError(f"unknown T3 config {name!r}")


def _collect_key_batches(model, tokenizer, texts, layer_idx, max_tokens):
    rows = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens).to(model.device)
        with torch.no_grad():
            out = model(**inputs, use_cache=True)
        keys, _values = get_true_kv(out.past_key_values, layer_idx)
        keys = keys.detach().float().cpu()
        ids = inputs.input_ids[0, : keys.shape[2]].detach().cpu().numpy().astype(np.int64)
        rows.append((keys, ids))
    return rows


def _flatten_batches(batches, transform):
    xs, ys = [], []
    for keys, ids in batches:
        rep = transform(keys)
        # keys: (batch=1, heads, seq, head_dim) -> (seq, heads * head_dim)
        feat = rep[0].permute(1, 0, 2).reshape(rep.shape[2], -1).numpy()
        n = min(len(ids), feat.shape[0])
        xs.append(feat[:n])
        ys.append(ids[:n])
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def _transformer(name, layer_idx, n_layers, seed):
    spec = _codec_spec(name)
    if spec is None:
        return lambda k: k.clone()
    if name.endswith("_nc") and layer_idx in {0, 1, n_layers - 2, n_layers - 1}:
        return lambda k: k.clone()
    quantizer, kwargs = spec
    if quantizer == "turboquant":
        kwargs = {**kwargs, "seed": seed}
    codec = make_codec(quantizer, **kwargs)
    return lambda k: codec.recon(k, is_key=True)


def _regimes_for(name):
    if name.startswith("turbo"):
        return ["public", "secret"]
    return ["public"]


def main():
    cfg = get_experiment_config()
    texts = _texts_from_env()
    max_tokens = int(os.environ.get("T3_MAX_TOKENS", "96"))
    train_fraction = float(os.environ.get("T3_TRAIN_FRACTION", "0.7"))
    seed = int(os.environ.get("SEED", "42"))
    epochs = int(os.environ.get("T3_EPOCHS", "200"))
    hidden = int(os.environ.get("T3_HIDDEN", "0"))
    lr = float(os.environ.get("T3_LR", "0.05"))
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {cfg.model_id} ...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id, dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_layers = model.config.num_hidden_layers
    layer_idx = int(os.environ.get("T3_LAYER", str(max(0, n_layers - 1))))
    if layer_idx < 0:
        layer_idx = n_layers + layer_idx
    print(f"Collecting FP keys: texts={len(texts)} layer={layer_idx} max_tokens={max_tokens}")
    t0 = time.perf_counter()
    batches = _collect_key_batches(model, tokenizer, texts, layer_idx, max_tokens)

    # Split by token position after flattening the FP representation; reuse indices for every config.
    fp_x, y = _flatten_batches(batches, lambda k: k.clone())
    train_idx, test_idx = train_test_indices(len(y), train_fraction=train_fraction, seed=seed)
    emb = model.get_input_embeddings().weight.detach().float().cpu().numpy()
    inv_cfg = InverterConfig(epochs=epochs, lr=lr, hidden=hidden, seed=seed)

    secret = secret_regime(os.environ.get("T3_SECRET_KEY", "t3-default-secret"))
    results = []
    fp_metric = None
    for name in _config_names():
        for pi in _regimes_for(name):
            train_seed = seed
            test_seed = seed
            if pi == "secret" and name.startswith("turbo"):
                test_seed = secret.seed_for_layer(layer_idx)
            train_x_all, train_y_all = _flatten_batches(
                batches, _transformer(name, layer_idx, n_layers, train_seed))
            test_x_all, test_y_all = _flatten_batches(
                batches, _transformer(name, layer_idx, n_layers, test_seed))
            metrics = train_linear_inverter(
                train_x_all[train_idx],
                train_y_all[train_idx],
                test_x_all[test_idx],
                test_y_all[test_idx],
                token_embeddings=emb,
                config=inv_cfg,
            )
            if name == "fp16":
                fp_metric = metrics["token_recovery"]
            metrics.update({
                "config": name,
                "pi_regime": pi,
                "train_seed": train_seed if pi == "public" else "attacker_public_seed",
                "test_seed": test_seed if pi == "public" else "secret",
            })
            if fp_metric is not None:
                metrics["delta_vs_fp"] = round(metrics["token_recovery"] - fp_metric, 4)
            results.append(metrics)
            print(f"{name:<15} {pi:<6} token={metrics['token_recovery']:.3f} "
                  f"semantic={metrics['semantic_recovery']:.3f}")

    turbo_public = next((r for r in results if r["config"] == "turbo_k3v4"
                         and r["pi_regime"] == "public"), None)
    verdict = "not_enough_signal"
    if fp_metric is not None and turbo_public is not None:
        delta = turbo_public["token_recovery"] - fp_metric
        if delta > 0.02:
            verdict = "worsen"
        elif delta < -0.02:
            verdict = "mitigate"
        else:
            verdict = "neutral"

    out = {
        "model": cfg.model_id,
        "model_tag": cfg.model_tag,
        "layer": layer_idx,
        "n_layers": n_layers,
        "texts": len(texts),
        "vectors": int(len(y)),
        "train_fraction": train_fraction,
        "epochs": epochs,
        "hidden": hidden,
        "verdict": verdict,
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "results": results,
    }
    path = out_dir / "leakage.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved -> {path}")


if __name__ == "__main__":
    main()
