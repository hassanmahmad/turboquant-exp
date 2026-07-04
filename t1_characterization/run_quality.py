"""T1 quality sweep on a real model: NIAH found-rate + perplexity (+ optional LongBench) per config.

This is the "impact on quality across a few benchmark tasks" half of T1 (the memory-savings/latency
half needs vLLM). Submit like run_sanity, or reuse slurm/sanity.slurm with the entrypoint swapped.

Env: MODEL_ID, MODEL_TAG, OUTPUT_DIR, QUANT_CONFIGS, NIAH_LENGTHS, NIAH_DEPTHS, MAX_NEW_TOKENS,
PPL_TOKENS (default 512), LONGBENCH_TASK (e.g. hotpotqa; unset = skip), LONGBENCH_N (default 20).
"""

import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
for _k in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
    os.environ.setdefault(_k, "1")

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from tqsec.config import get_experiment_config  # noqa: E402
from tqsec.benchmarks import default_configs, perplexity, run_longbench, run_needle  # noqa: E402


def _floats(env, default):
    raw = os.environ.get(env)
    return tuple(float(x) for x in raw.split(",")) if raw else default


def _ints(env, default):
    raw = os.environ.get(env)
    return tuple(int(x) for x in raw.split(",")) if raw else default


def main():
    cfg = get_experiment_config()
    lengths = _ints("NIAH_LENGTHS", (1024, 2048))
    depths = _floats("NIAH_DEPTHS", (0.25, 0.5, 0.75))
    ppl_tokens = int(os.environ.get("PPL_TOKENS", "512"))
    lb_task = os.environ.get("LONGBENCH_TASK")
    lb_n = int(os.environ.get("LONGBENCH_N", "20"))

    print(f"Loading {cfg.model_id} ...")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id, dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True).eval()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    from tqsec.quantizers import make_quant_cache
    n_layers = model.config.num_hidden_layers
    nc = tuple(sorted({0, 1, n_layers - 2, n_layers - 1}))
    all_configs = default_configs() + [
        ("turbo_k3v4_nc", lambda: make_quant_cache("turboquant", key_bits=3, value_bits=4, nc_layers=nc)),
    ]
    wanted = os.environ.get("QUANT_CONFIGS")
    configs = [c for c in all_configs if c[0] in set(wanted.split(","))] if wanted else all_configs
    print(f"configs={[n for n, _ in configs]} | ppl_tokens={ppl_tokens} | longbench={lb_task or 'off'}")

    t0 = time.perf_counter()
    results = {}
    for name, make_cache in configs:
        niah = run_needle(model, tokenizer, make_cache, lengths=lengths, depths=depths,
                          max_new_tokens=cfg.max_new_tokens)
        ppl = perplexity(model, tokenizer, make_cache, max_tokens=ppl_tokens)
        entry = {"niah_found_rate": niah["found_rate"], "perplexity": ppl}
        if lb_task:
            lb = run_longbench(model, tokenizer, make_cache, task=lb_task, n_samples=lb_n)
            entry["longbench"] = {"task": lb_task, "score": lb["score"]}
        results[name] = entry
        print(f"  {name:<14} niah={niah['found_rate']:.3f}  ppl={ppl:.2f}"
              + (f"  {lb_task}={entry['longbench']['score']:.3f}" if lb_task else ""))
    elapsed = round(time.perf_counter() - t0, 1)

    out = {"model": cfg.model_id, "model_tag": cfg.model_tag, "lengths": list(lengths),
           "depths": list(depths), "ppl_tokens": ppl_tokens, "longbench_task": lb_task,
           "chat_template_enable_thinking": os.environ.get("CHAT_TEMPLATE_ENABLE_THINKING"),
           "elapsed_s": elapsed, "results": results}
    path = os.path.join(cfg.output_dir, "quality.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nElapsed {elapsed}s. Saved -> {path}")


if __name__ == "__main__":
    main()
