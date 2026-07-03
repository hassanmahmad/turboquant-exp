"""tqsec.benchmarks — sanity benchmarks: needle-in-a-haystack (NIAH) + LongBench slice.

The Phase-1 sanity check: confirm the research layer is quality-neutral ~3.5 bits and
degrades ~2.5 bits, matching the paper. Long-context retrieval is where KV-compression
effects show, so NIAH is the primary sanity test; a LongBench slice is secondary.

Everything runs under the control harness: the same benchmark executes against FP-KV and
every quantizer via a `make_cache` factory, so the sanity sweep doubles as the T1 quality
table and the {TurboQuant, INT, KIVI, FP8} specificity ablation.

Offline / Leonardo: pre-download models to $SCRATCH and run with HF_HUB_OFFLINE=1. For
LongBench, fetch the dataset on the login node, then run with HF_DATASETS_OFFLINE=1.

Construction + scoring are pure Python (unit-testable without a model); the runners need a
model + tokenizer.
"""

import os
import sys
from dataclasses import dataclass

_VENDOR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party")
if _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)


# --------------------------------------------------------------------------------------
# Needle-in-a-haystack — construction + scoring (no model needed)
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
def _to_inputs(tokenizer, prompt, use_chat_template, device):
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
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
        ("int3",        lambda: make_quant_cache("int", key_bits=3, value_bits=3)),
        ("kivi3",       lambda: make_quant_cache("kivi", key_bits=3, value_bits=3)),
        ("fp8",         lambda: make_quant_cache("fp8")),
    ]


def sanity_sweep(model, tokenizer, *, configs=None, lengths=(1000, 2000, 4000),
                 depths=(0.1, 0.5, 0.9), max_new_tokens=24, use_chat_template=True):
    """Phase-1 sanity deliverable: NIAH found-rate per cache config (FP-KV + all quantizers).

    Returns {config_name: run_needle(...)}. Quality-neutral configs should match fp16's
    found-rate; aggressive ones (3-bit) should start to drop — that's the sanity signal.
    """
    configs = configs or default_configs()
    return {name: run_needle(model, tokenizer, make_cache, cfg=None, lengths=lengths,
                             depths=depths, max_new_tokens=max_new_tokens,
                             use_chat_template=use_chat_template)
            for name, make_cache in configs}


# --------------------------------------------------------------------------------------
# LongBench slice (needs the `datasets` library + a pre-fetched dataset)
# --------------------------------------------------------------------------------------
def load_longbench_slice(task="hotpotqa", n_samples=20, split="test", cache_dir=None):
    """Load a LongBench task slice. Lazy-imports `datasets`; offline-aware.

    Leonardo: pre-fetch on the login node — `datasets.load_dataset("THUDM/LongBench", task,
    cache_dir=$SCRATCH/data)` — then run with HF_DATASETS_OFFLINE=1 and the same cache_dir.
    Returns [{"context", "input", "answers"}].
    """
    try:
        import datasets
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("LongBench needs `pip install datasets`") from e
    ds = datasets.load_dataset("THUDM/LongBench", task, split=split, cache_dir=cache_dir)
    ds = ds.select(range(min(n_samples, len(ds))))
    return [{"context": r.get("context", ""), "input": r.get("input", ""),
             "answers": r.get("answers", [])} for r in ds]


def score_longbench(answer, golds):
    """Lenient substring match against any gold answer (sanity-level; T1 can use full F1/ROUGE)."""
    a = answer.lower()
    return bool(golds) and any(str(g).lower() in a for g in golds)
