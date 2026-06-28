"""Phase-1 sanity benchmark on a real model.

Runs needle-in-a-haystack across FP-KV + every quantizer ({TurboQuant variants, INT, KIVI,
FP8}) and reports retrieval found-rate per config. This is the shared-foundation sanity check
AND the first T1 quality/specificity table. Submit via slurm/sanity.slurm.

Env knobs (all optional): MODEL_ID, MODEL_TAG, OUTPUT_DIR, MAX_NEW_TOKENS,
NIAH_LENGTHS="1024,2048", NIAH_DEPTHS="0.25,0.5,0.75", QUANT_CONFIGS="fp16,turbo_k3v4,int3".
The vendored research layer is pure NumPy, so long contexts are slow — scale NIAH_LENGTHS to
the time budget.
"""

import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
for k in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
    os.environ.setdefault(k, "1")

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from tqsec.config import get_experiment_config  # noqa: E402
from tqsec.benchmarks import sanity_sweep, default_configs  # noqa: E402


def _floats(env, default):
    raw = os.environ.get(env)
    return tuple(float(x) for x in raw.split(",")) if raw else default


def _ints(env, default):
    raw = os.environ.get(env)
    return tuple(int(x) for x in raw.split(",")) if raw else default


def _select_configs():
    wanted = os.environ.get("QUANT_CONFIGS")
    configs = default_configs()
    if wanted:
        keep = set(wanted.split(","))
        configs = [(n, f) for (n, f) in configs if n in keep]
    return configs


def main():
    cfg = get_experiment_config()
    lengths = _ints("NIAH_LENGTHS", (1024, 2048))
    depths = _floats("NIAH_DEPTHS", (0.25, 0.5, 0.75))
    configs = _select_configs()

    print(f"Loading {cfg.model_id} ...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id, dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Loaded. configs={[n for n, _ in configs]} lengths={lengths} depths={depths}")

    t0 = time.perf_counter()
    results = sanity_sweep(model, tokenizer, configs=configs, lengths=lengths,
                           depths=depths, max_new_tokens=cfg.max_new_tokens)
    elapsed = round(time.perf_counter() - t0, 1)

    found = {name: r["found_rate"] for name, r in results.items()}
    out = {
        "model": cfg.model_id, "model_tag": cfg.model_tag,
        "lengths": list(lengths), "depths": list(depths),
        "max_new_tokens": cfg.max_new_tokens, "elapsed_s": elapsed,
        "found_rate": found, "results": results,
    }
    path = os.path.join(cfg.output_dir, "sanity_benchmark.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\n=== NIAH found-rate (FP-KV should be highest; aggressive quant drops) ===")
    for name, _ in configs:
        print(f"  {name:<12} {found[name]:.3f}")
    print(f"\nElapsed {elapsed}s. Saved -> {path}")


if __name__ == "__main__":
    main()
