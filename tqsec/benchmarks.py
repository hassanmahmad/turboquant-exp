"""tqsec.benchmarks: sanity benchmarks, needle-in-a-haystack (NIAH) and a LongBench slice.

Phase-1 sanity check: confirm the research layer is quality-neutral at ~3.5 bits and degrades at
~2.5 bits, matching the paper. NIAH is the primary test (long-context retrieval is where KV-
compression effects show); a LongBench slice is secondary. Both run under the control harness:
the same benchmark executes against FP-KV and every quantizer via a `make_cache` factory, so the
sanity sweep doubles as the T1 quality table and the {TurboQuant, INT, KIVI, FP8} ablation.

Offline (Leonardo): pre-download models to $SCRATCH and run with HF_HUB_OFFLINE=1. For LongBench,
fetch the dataset on the login node, then run with HF_DATASETS_OFFLINE=1. Construction and scoring
are pure Python (testable without a model); the runners need a model and tokenizer.
"""

import os
import sys
from dataclasses import dataclass

_VENDOR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party")
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)


# --------------------------------------------------------------------------------------
# Needle-in-a-haystack: construction + scoring (no model needed)
# --------------------------------------------------------------------------------------
_FILLER = [
    "The morning market filled slowly with vendors arranging fruit and bread.",
    "A light rain had washed the streets clean the night before.",
    "Children walked to school in pairs, talking about the weekend.",
    "The old library kept its windows open to let in the river breeze.",
    "Trains arrived on time, and commuters read quietly on the platform.",
    "In the park, a gardener trimmed the hedges into careful shapes.",
    "The bakery on the corner sold out of rolls by mid-afternoon.",
    "Clouds drifted east while the harbor boats rocked at their moorings.",
    "A street musician practiced scales beneath the shade of an awning.",
    "The museum announced a new exhibit of maps from distant centuries.",
    "Office workers queued for coffee, comparing notes on the day ahead.",
    "By evening the square grew quiet, lit by a row of amber lamps.",
]


@dataclass
class NeedleConfig:
    needle: str = "The special magic Cairo number is 7421900."
    question: str = "What is the special magic Cairo number? Answer with the number only."
    answer: str = "7421900"
    preamble: str = "Read the following document carefully, then answer the question at the end.\n\n"


def build_needle_prompt(cfg, n_context_tokens, depth, count_tokens):
    """Haystack of ~n_context_tokens with the needle spliced in at `depth` (0..1), question appended.

    count_tokens(str) -> int measures length in the target tokenizer (pass a word-count fn to test).
    """
    sents, tok, i = [], 0, 0
    while tok < n_context_tokens:
        s = _FILLER[i % len(_FILLER)]
        sents.append(s)
        tok += count_tokens(s)
        i += 1
    insert_at = max(0, min(len(sents), int(round(depth * len(sents)))))
    sents.insert(insert_at, cfg.needle)
    haystack = " ".join(sents)
    return f"{cfg.preamble}{haystack}\n\nQuestion: {cfg.question}\nAnswer:"


def score_needle(answer, cfg):
    """Did the model recover the needle's answer? (substring, case-insensitive)."""
    return cfg.answer.lower() in answer.lower()


# --------------------------------------------------------------------------------------
# Runners (need a model + tokenizer)
# --------------------------------------------------------------------------------------
def _chat_template_kwargs():
    raw = os.environ.get("CHAT_TEMPLATE_ENABLE_THINKING")
    if raw is None:
        return {}
    return {"enable_thinking": raw.strip().lower() not in {"0", "false", "no", "off"}}


def _to_inputs(tokenizer, prompt, use_chat_template, device):
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
            **_chat_template_kwargs(),
        }
        try:
            prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], **kwargs)
    return tokenizer(prompt, return_tensors="pt").to(device)


def _generate(model, tokenizer, prompt, cache, max_new_tokens, use_chat_template):
    import torch
    inputs = _to_inputs(tokenizer, prompt, use_chat_template, model.device)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             use_cache=True, past_key_values=cache, pad_token_id=pad)
    return tokenizer.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()


def run_needle(model, tokenizer, make_cache=lambda: None, *, cfg=None,
               lengths=(1000, 2000, 4000), depths=(0.1, 0.5, 0.9),
               max_new_tokens=24, use_chat_template=True):
    """Run NIAH over (length × depth). `make_cache()` returns a fresh KV cache per generation
    (None = FP-KV; or e.g. `lambda: make_quant_cache("turboquant", key_bits=3, value_bits=4)`)."""
    cfg = cfg or NeedleConfig()
    count = lambda s: len(tokenizer(s, add_special_tokens=False).input_ids)
    rows = []
    for L in lengths:
        for d in depths:
            prompt = build_needle_prompt(cfg, L, d, count)
            ans = _generate(model, tokenizer, prompt, make_cache(), max_new_tokens, use_chat_template)
            rows.append({"length": L, "depth": round(d, 2),
                         "found": score_needle(ans, cfg), "answer": ans[:60]})
    found = sum(r["found"] for r in rows)
    return {"found_rate": round(found / len(rows), 3), "n": len(rows), "rows": rows}


def default_configs():
    """The sanity/T1 sweep: FP-KV + TurboQuant variants + the INT/KIVI/FP8 controls.

    Names mirror the paper's variants (k8v4, k3v4, 3bit). Each entry is (name, make_cache).
    """
    from tqsec.quantizers import make_quant_cache
    return [
        ("fp16",        lambda: None),
        ("turbo_k8v4",  lambda: make_quant_cache("turboquant", key_bits=8, value_bits=4)),
        ("turbo_k3v4",  lambda: make_quant_cache("turboquant", key_bits=3, value_bits=4)),
        ("turbo_3bit",  lambda: make_quant_cache("turboquant", key_bits=3, value_bits=3)),
        ("turbo_k3v4_mix", lambda: make_quant_cache("turboquant", key_bits=3, value_bits=4, mode="mixed")),
        # value-path isolation: fix keys high, sweep value bits; + mixed-keys with 8-bit values
        ("turbo_k8v8",  lambda: make_quant_cache("turboquant", key_bits=8, value_bits=8)),
        ("turbo_k8v2",  lambda: make_quant_cache("turboquant", key_bits=8, value_bits=2)),
        ("turbo_k3v8_mix", lambda: make_quant_cache("turboquant", key_bits=3, value_bits=8, mode="mixed")),
        ("int3",        lambda: make_quant_cache("int", key_bits=3, value_bits=3)),
        ("kivi3",       lambda: make_quant_cache("kivi", key_bits=3, value_bits=3)),
        ("fp8",         lambda: make_quant_cache("fp8")),
    ]


def sanity_sweep(model, tokenizer, *, configs=None, lengths=(1000, 2000, 4000),
                 depths=(0.1, 0.5, 0.9), max_new_tokens=24, use_chat_template=True):
    """Phase-1 sanity deliverable: NIAH found-rate per cache config (FP-KV + all quantizers).

    Returns {config_name: run_needle(...)}. Quality-neutral configs should match fp16's found-rate;
    aggressive ones (3-bit) should start to drop, which is the sanity signal.
    """
    configs = configs or default_configs()
    return {name: run_needle(model, tokenizer, make_cache, cfg=None, lengths=lengths,
                             depths=depths, max_new_tokens=max_new_tokens,
                             use_chat_template=use_chat_template)
            for name, make_cache in configs}


# --------------------------------------------------------------------------------------
# LongBench slice (needs the `datasets` library + a pre-fetched dataset)
# --------------------------------------------------------------------------------------
def load_longbench_slice(task="hotpotqa", n_samples=20, split=None, cache_dir=None):
    """Load a LongBench task slice straight from the repo's `data.zip`.

    Bypasses `datasets` (which 4.x refuses because LongBench is a script dataset). Reads
    `data/<task>.jsonl` out of the zip; returns [{"context", "input", "answers"}].

    Pre-fetch on the login node (online), then it works offline:
        from huggingface_hub import hf_hub_download
        hf_hub_download("zai-org/LongBench", "data.zip", repo_type="dataset")
    """
    import json
    import zipfile

    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id="zai-org/LongBench", filename="data.zip",
                           repo_type="dataset", cache_dir=cache_dir)
    out = []
    with zipfile.ZipFile(path) as z, z.open(f"data/{task}.jsonl") as f:
        for line in f:
            if len(out) >= n_samples:
                break
            r = json.loads(line)
            out.append({"context": r.get("context", ""), "input": r.get("input", ""),
                        "answers": r.get("answers", [])})
    return out


def score_longbench(answer, golds):
    """Lenient substring match against any gold answer (sanity-level; T1 can use full F1/ROUGE)."""
    a = answer.lower()
    return bool(golds) and any(str(g).lower() in a for g in golds)


def _longbench_prompt(sample, tokenizer, max_context_tokens):
    ctx = sample.get("context", "")
    ids = tokenizer(ctx, add_special_tokens=False).input_ids
    if len(ids) > max_context_tokens:                       # keep the NumPy cache tractable
        ctx = tokenizer.decode(ids[:max_context_tokens])
    return f"Read the document and answer the question.\n\n{ctx}\n\nQuestion: {sample.get('input', '')}\nAnswer:"


def run_longbench(model, tokenizer, make_cache=lambda: None, *, task="hotpotqa", n_samples=20,
                  max_new_tokens=32, max_context_tokens=3500, use_chat_template=True, cache_dir=None):
    """Run a LongBench task slice under `make_cache`; score = lenient substring match vs any gold."""
    samples = load_longbench_slice(task, n_samples=n_samples, cache_dir=cache_dir)
    rows = []
    for s in samples:
        prompt = _longbench_prompt(s, tokenizer, max_context_tokens)
        ans = _generate(model, tokenizer, prompt, make_cache(), max_new_tokens, use_chat_template)
        rows.append({"score": int(score_longbench(ans, s.get("answers", []))), "answer": ans[:80]})
    score = sum(r["score"] for r in rows) / max(len(rows), 1)
    return {"task": task, "score": round(score, 3), "n": len(rows), "rows": rows}


# --------------------------------------------------------------------------------------
# Perplexity: the raw quality cost of KV compression on next-token prediction
# --------------------------------------------------------------------------------------
_PPL_TEXT = (
    "A lighthouse is a tower built to emit light from a system of lamps and lenses, serving as a "
    "navigational aid for pilots at sea or on inland waterways. For centuries lighthouses marked "
    "dangerous coastlines, hazardous shoals and reefs, and the safe entries to harbours, and they "
    "guided sailors home long before satellites and radio beacons existed. The earliest known "
    "lighthouse was the Pharos of Alexandria, completed in the third century before the common era, "
    "which stood more than a hundred metres tall and used a fire at night and a polished mirror by day "
    "to warn approaching ships. Its light was said to be visible for many kilometres, and it remained "
    "one of the tallest structures made by human hands for a very long time. The design of a lighthouse "
    "balances height, light intensity, and the curvature of the earth: because the horizon falls away "
    "with distance, a taller tower lets its beam reach farther before the curve of the sea hides it. "
    "Builders placed lighthouses on cliffs and headlands where the natural elevation extended the range "
    "of the light without the cost of an enormous tower. The invention of the Fresnel lens in the early "
    "nineteenth century transformed the field, because it concentrated a faint flame into a narrow, "
    "powerful beam that could be seen from great distances while using far less fuel than older "
    "reflector systems. Keeping a lighthouse was demanding and often lonely work. A keeper trimmed the "
    "wicks, polished the lenses, wound the clockwork that rotated the lamp, and recorded the weather and "
    "passing vessels in a daily log. Storms could isolate a station for weeks, and supplies arrived by "
    "boat only when the sea allowed. As electricity and automatic controls spread through the twentieth "
    "century most lighthouses were automated, and the resident keepers gradually disappeared. Today many "
    "towers are preserved as monuments, their beams maintained by machines, but the quiet service they "
    "once demanded is remembered in the stories that surround them."
)


def perplexity(model, tokenizer, make_cache=lambda: None, *, text=None, max_tokens=512):
    """Perplexity of a fixed passage under `make_cache` (None = FP-KV). Lower is better; the
    compression's quality cost is the rise over the FP-KV baseline. During the forward, attention
    reads the (quantized) KV the cache returns, so this reflects the compression."""
    import torch
    ids = tokenizer(text or _PPL_TEXT, return_tensors="pt").input_ids[:, :max_tokens].to(model.device)
    cache = make_cache()
    with torch.no_grad():
        out = model(input_ids=ids, past_key_values=cache, use_cache=cache is not None)
        shift_logits = out.logits[:, :-1, :].reshape(-1, out.logits.size(-1)).float()
        shift_labels = ids[:, 1:].reshape(-1)
        loss = torch.nn.functional.cross_entropy(shift_logits, shift_labels)
    return round(float(torch.exp(loss)), 3)
