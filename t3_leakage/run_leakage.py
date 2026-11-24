"""T3 prompt-leakage experiment: FP-KV baseline vs compressed representations.

This is the required binary test harness: does compression make token inversion
harder or easier than the uncompressed FP key-cache baseline? It keeps exact
token recovery and semantic recovery separate, and it runs TurboQuant under
public and secret Pi regimes.

Evidence controls (so a one-token difference is never reported as an effect):
  * corpus: `t3_leakage/corpus.txt` (120 sentences) unless T3_TEXT_FILE / T3_TEXTS is set;
  * held-out *sentences* by default (T3_SPLIT=text), so the inverter is scored on prompts it
    never saw; T3_SPLIT=position reproduces the original token-position split;
  * several layers (T3_LAYERS, default first/middle/last) and repeated splits (T3_SPLITS);
  * every delta vs FP carries a paired-bootstrap 95% CI, and the verdict only names a
    direction when that interval excludes zero.

Threat model: the attacker reads the *dequantized* key vectors the cache serves (what a
KV-reading adversary sees at attention time), not the raw rotated codes.
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
from tqsec.inversion import (InverterConfig, paired_bootstrap_delta,  # noqa: E402
                             train_linear_inverter, train_test_indices, verdict_from_ci)
from tqsec.pi_regime import secret_regime  # noqa: E402
from tqsec.quantizers import make_codec  # noqa: E402


# Fallback only; the default corpus is t3_leakage/corpus.txt.
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

CONTROLS = ("int3", "kivi3", "fp8")
_CODEC_CACHE = {}


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
    corpus = Path(__file__).with_name("corpus.txt")
    if corpus.exists():
        return [line.strip() for line in corpus.read_text(encoding="utf-8").splitlines()
                if line.strip()]
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


def _layer_indices(n_layers):
    raw = os.environ.get("T3_LAYERS") or os.environ.get("T3_LAYER")
    if not raw:
        return sorted({0, n_layers // 2, n_layers - 1})
    out = []
    for tok in raw.replace(",", " ").split():
        idx = int(tok)
        out.append(n_layers + idx if idx < 0 else idx)
    return sorted(set(out))


def _collect_key_batches(model, tokenizer, texts, layers, max_tokens):
    """One forward per text; returns {layer: [(keys, ids), ...]} in text order."""
    rows = {layer: [] for layer in layers}
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_tokens).to(model.device)
        with torch.no_grad():
            out = model(**inputs, use_cache=True)
        for layer in layers:
            keys, _values = get_true_kv(out.past_key_values, layer)
            keys = keys.detach().float().cpu()
            ids = inputs.input_ids[0, : keys.shape[2]].detach().cpu().numpy().astype(np.int64)
            rows[layer].append((keys, ids))
    return rows


def _flatten_batches(batches, transform):
    """-> (features, token ids, text index per row)."""
    xs, ys, ts = [], [], []
    for text_idx, (keys, ids) in enumerate(batches):
        rep = transform(keys)
        # keys: (batch=1, heads, seq, head_dim) -> (seq, heads * head_dim)
        feat = rep[0].permute(1, 0, 2).reshape(rep.shape[2], -1).numpy()
        n = min(len(ids), feat.shape[0])
        xs.append(feat[:n])
        ys.append(ids[:n])
        ts.append(np.full(n, text_idx, dtype=np.int64))
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0), np.concatenate(ts, axis=0)


def _transformer(name, layer_idx, n_layers, seed):
    spec = _codec_spec(name)
    if spec is None:
        return lambda k: k.clone()
    if name.endswith("_nc") and layer_idx in {0, 1, n_layers - 2, n_layers - 1}:
        return lambda k: k.clone()
    quantizer, kwargs = spec
    if quantizer == "turboquant":
        kwargs = {**kwargs, "seed": seed}
    # Codecs are layer-independent; reuse them, since building the 8-bit Lloyd-Max grid
    # alone takes over a minute.
    cache_key = (quantizer, tuple(sorted(kwargs.items())))
    if cache_key not in _CODEC_CACHE:
        _CODEC_CACHE[cache_key] = make_codec(quantizer, **kwargs)
    codec = _CODEC_CACHE[cache_key]
    return lambda k: codec.recon(k, is_key=True)


def _regimes_for(name):
    if name.startswith("turbo"):
        return ["public", "secret"]
    return ["public"]


def make_splits(text_ids, *, mode, n_splits, train_fraction, seed):
    """Repeated train/test row-index splits. mode='text' holds out whole sentences."""
    splits = []
    n_texts = int(text_ids.max()) + 1
    for s in range(n_splits):
        if mode == "text" and n_texts > 1:
            tr_texts, _te = train_test_indices(n_texts, train_fraction=train_fraction, seed=seed + s)
            in_train = np.isin(text_ids, tr_texts)
            splits.append((np.flatnonzero(in_train), np.flatnonzero(~in_train)))
        else:
            splits.append(train_test_indices(len(text_ids), train_fraction=train_fraction, seed=seed + s))
    return splits


def evaluate_layer(batches, layer_idx, n_layers, *, config_names, splits, emb, inv_cfg, seed,
                   secret_seed, n_boot=2000):
    """Run every config/regime on one layer. Returns (result rows, verdict block)."""
    rows, correct = [], {}
    for name in config_names:
        for pi in _regimes_for(name):
            train_x, y, _t = _flatten_batches(batches, _transformer(name, layer_idx, n_layers, seed))
            if pi == "secret":
                # attacker trains on the public-seed geometry, deployment serves a secret Pi
                test_x, _y, _t = _flatten_batches(
                    batches, _transformer(name, layer_idx, n_layers, secret_seed))
            else:
                test_x = train_x
            per_split, pooled = [], []
            for s, (tr, te) in enumerate(splits):
                cfg_s = InverterConfig(epochs=inv_cfg.epochs, lr=inv_cfg.lr, hidden=inv_cfg.hidden,
                                       weight_decay=inv_cfg.weight_decay, seed=inv_cfg.seed + s,
                                       device=inv_cfg.device)
                m, pred = train_linear_inverter(train_x[tr], y[tr], test_x[te], y[te],
                                                token_embeddings=emb, config=cfg_s,
                                                return_predictions=True)
                per_split.append(m)
                pooled.append((pred == y[te]).astype(np.int8))
            correct[(name, pi)] = np.concatenate(pooled)
            tok = [m["token_recovery"] for m in per_split]
            row = {
                "config": name,
                "pi_regime": pi,
                "token_recovery": round(float(np.mean(tok)), 4),
                "token_recovery_std": round(float(np.std(tok)), 4),
                "semantic_recovery": round(float(np.mean([m["semantic_recovery"] for m in per_split])), 4),
                "token_set_jaccard": round(float(np.mean([m["token_set_jaccard"] for m in per_split])), 4),
                "majority_baseline": round(float(np.mean([m["majority_baseline"] for m in per_split])), 4),
                "test_rows": int(sum(m["test_rows"] for m in per_split)),
                "unseen_test_tokens": int(sum(m["unseen_test_tokens"] for m in per_split)),
                "candidate_tokens": int(np.mean([m["candidate_tokens"] for m in per_split])),
                "train_seed": seed if pi == "public" else "attacker_public_seed",
                "test_seed": seed if pi == "public" else "secret",
            }
            rows.append(row)
            print(f"  L{layer_idx:<3} {name:<15} {pi:<6} token={row['token_recovery']:.3f}"
                  f"+-{row['token_recovery_std']:.3f} semantic={row['semantic_recovery']:.3f}")

    fp = correct.get(("fp16", "public"))
    block = {"verdict": "not_enough_signal"}
    if fp is None:
        return rows, block
    for row in rows:
        key = (row["config"], row["pi_regime"])
        if key == ("fp16", "public"):
            row["delta_vs_fp"] = 0.0
            continue
        b = paired_bootstrap_delta(correct[key], fp, n_boot=n_boot, seed=seed)
        row["delta_vs_fp"] = b["delta"]
        row["delta_ci95"] = [b["ci_low"], b["ci_high"]]
        row["verdict_vs_fp"] = verdict_from_ci(b["ci_low"], b["ci_high"])

    turbo = correct.get(("turbo_k3v4", "public"))
    if turbo is None:
        return rows, block
    b = paired_bootstrap_delta(turbo, fp, n_boot=n_boot, seed=seed)
    block = {"verdict": verdict_from_ci(b["ci_low"], b["ci_high"]),
             "turbo_k3v4_public_delta": b["delta"],
             "turbo_k3v4_public_ci95": [b["ci_low"], b["ci_high"]]}
    vs = {}
    for ctrl in CONTROLS:
        if (ctrl, "public") in correct:
            c = paired_bootstrap_delta(turbo, correct[(ctrl, "public")], n_boot=n_boot, seed=seed)
            vs[ctrl] = {"delta": c["delta"], "ci95": [c["ci_low"], c["ci_high"]]}
    block["turbo_vs_controls"] = vs
    # TurboQuant-specific only if it differs from FP *and* from every coarse control, same direction.
    sign = {"worsen": 1, "mitigate": -1}.get(block["verdict"], 0)
    block["turboquant_specific"] = bool(sign) and bool(vs) and all(
        (v["ci95"][0] > 0) if sign > 0 else (v["ci95"][1] < 0) for v in vs.values())
    return rows, block


def main():
    cfg = get_experiment_config()
    texts = _texts_from_env()
    max_tokens = int(os.environ.get("T3_MAX_TOKENS", "96"))
    train_fraction = float(os.environ.get("T3_TRAIN_FRACTION", "0.7"))
    seed = int(os.environ.get("SEED", "42"))
    epochs = int(os.environ.get("T3_EPOCHS", "200"))
    hidden = int(os.environ.get("T3_HIDDEN", "0"))
    lr = float(os.environ.get("T3_LR", "0.05"))
    split_mode = os.environ.get("T3_SPLIT", "text").lower()
    n_splits = int(os.environ.get("T3_SPLITS", "5"))
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {cfg.model_id} ...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id, dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_layers = model.config.num_hidden_layers
    layers = _layer_indices(n_layers)
    print(f"Collecting FP keys: texts={len(texts)} layers={layers} max_tokens={max_tokens} "
          f"split={split_mode} x{n_splits}")
    t0 = time.perf_counter()
    batches = _collect_key_batches(model, tokenizer, texts, layers, max_tokens)
    emb = model.get_input_embeddings().weight.detach().float().cpu().numpy()
    del model                                   # free the GPU for the inverter
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _x, y, text_ids = _flatten_batches(batches[layers[0]], lambda k: k[..., :1])
    splits = make_splits(text_ids, mode=split_mode, n_splits=n_splits,
                         train_fraction=train_fraction, seed=seed)
    inv_cfg = InverterConfig(epochs=epochs, lr=lr, hidden=hidden, seed=seed)
    secret = secret_regime(os.environ.get("T3_SECRET_KEY", "t3-default-secret"))

    per_layer = []
    for layer_idx in layers:
        rows, block = evaluate_layer(
            batches[layer_idx], layer_idx, n_layers, config_names=_config_names(), splits=splits,
            emb=emb, inv_cfg=inv_cfg, seed=seed, secret_seed=secret.seed_for_layer(layer_idx))
        per_layer.append({"layer": layer_idx, **block, "results": rows})
        print(f"  L{layer_idx}: verdict={block['verdict']}")

    verdicts = {b["verdict"] for b in per_layer}
    out = {
        "model": cfg.model_id,
        "model_tag": cfg.model_tag,
        "n_layers": n_layers,
        "layers": layers,
        "texts": len(texts),
        "vectors": int(len(y)),
        "split": split_mode,
        "n_splits": n_splits,
        "train_fraction": train_fraction,
        "epochs": epochs,
        "hidden": hidden,
        "verdict": verdicts.pop() if len(verdicts) == 1 else "mixed_across_layers",
        "turboquant_specific": any(b.get("turboquant_specific") for b in per_layer),
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "per_layer": per_layer,
    }
    path = out_dir / "leakage.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved -> {path}")


if __name__ == "__main__":
    main()
