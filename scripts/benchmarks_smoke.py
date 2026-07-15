"""Smoke test for tqsec.benchmarks.

Usage:
    python scripts/benchmarks_smoke.py

Unit-tests NIAH construction and scoring (no model), then exercises model.generate
with the quant cache end-to-end on a tiny random model to validate the HF generate
integration (a path the 49 vendored tests and earlier smokes don't cover).
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "third_party"))
os.environ.setdefault("HF_HOME", os.path.join(
    os.environ.get("TEMP", "/tmp"), "tqsec_hf"))

from tqsec.benchmarks import build_needle_prompt, score_needle, run_needle, NeedleConfig  # noqa: E402


def test_construction_and_scoring():
    cfg = NeedleConfig()
    words = lambda s: len(s.split())                       # word-count "tokenizer"
    prompt = build_needle_prompt(cfg, n_context_tokens=120, depth=0.5, count_tokens=words)
    assert cfg.needle in prompt and cfg.question in prompt
    assert words(prompt) >= 120
    # depth controls where the needle lands
    early = build_needle_prompt(cfg, 120, 0.1, words).find(cfg.needle)
    late = build_needle_prompt(cfg, 120, 0.9, words).find(cfg.needle)
    assert 0 <= early < late, f"depth ordering wrong: {early} !< {late}"
    # scoring
    assert score_needle("the number is 7421900 obviously", cfg)
    assert not score_needle("I don't know", cfg)
    print("[ok] NIAH construction (length, depth, needle placement) + scoring")


def test_generate_integration():
    """Run the NIAH pipeline through model.generate with FP-KV and a TurboQuant cache."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from huggingface_hub import snapshot_download
        mid = os.environ.get("SMOKE_MODEL", "HuggingFaceTB/SmolLM2-135M-Instruct")
        local = snapshot_download(mid, cache_dir=os.environ.get("HF_HOME"))
        os.environ["HF_HUB_OFFLINE"] = "1"   # skip the additional_chat_templates 404 probe bug
        tok = AutoTokenizer.from_pretrained(local)
        model = AutoModelForCausalLM.from_pretrained(local, torch_dtype=torch.float32).eval()
    except Exception as e:  # offline / unavailable, keep unit tests meaningful
        print(f"[skip] generate integration ({type(e).__name__}: {str(e)[:120]})")
        return

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    from tqsec.quantizers import make_quant_cache
    from tqsec.instrument import ErrorMapRecorder

    small = dict(lengths=(64,), depths=(0.5,), max_new_tokens=6, use_chat_template=False)

    # FP-KV path runs
    fp = run_needle(model, tok, make_cache=lambda: None, **small)
    assert fp["n"] == 1 and isinstance(fp["rows"][0]["answer"], str)

    # TurboQuant path runs and the cache is actually exercised by generate (recorder proves it)
    rec = ErrorMapRecorder()
    tq = run_needle(model, tok,
                    make_cache=lambda: make_quant_cache("turboquant", key_bits=3, value_bits=3, recorder=rec),
                    **small)
    assert tq["n"] == 1 and 0.0 <= tq["found_rate"] <= 1.0
    assert rec.n_layers == model.config.num_hidden_layers, \
        f"quant cache should span all {model.config.num_hidden_layers} layers, saw {rec.n_layers}"
    assert any(rec.records.values()), "generate did not populate the quant cache"

    # an INT control path also runs
    it = run_needle(model, tok,
                    make_cache=lambda: make_quant_cache("int", key_bits=3, value_bits=3), **small)
    assert it["n"] == 1

    print(f"[ok] model.generate + quant cache end-to-end: FP, TurboQuant (recorder saw "
          f"{rec.n_layers} layers), and INT all ran")
    print("     (tiny random model -> found_rate is meaningless; this validates the PIPELINE)")


def main():
    test_construction_and_scoring()
    test_generate_integration()
    print("\nSMOKE PASSED")


if __name__ == "__main__":
    main()
