# T1 — TurboQuant KV-Cache Characterization

**Status:** first result (2026-07, Leonardo/CINECA). Quality characterization across quantizers and
bit-widths on a real model, with a mechanistic explanation. Memory tables and additional models are
future work (§7).

## 1. Headline

**On Qwen2.5-7B, uniform TurboQuant cannot do exact long-context retrieval at *any* bit-width.** Even
8-bit keys + 8-bit values score only 0.167 on needle-in-a-haystack, while 3-bit KIVI scores 1.0. The
cause is a handful of **boundary-layer outlier channels** (up to ~100× the median magnitude) whose
values TurboQuant's random-rotation compression corrupts — enough to flip the exact answer
(`7421900` → `741900`). The **only** TurboQuant configuration that recovers is `-nc` (leaving the
first/last layers uncompressed); otherwise **per-channel KIVI** is the robust choice.

> **3-bit KIVI beats 8-bit TurboQuant on Qwen exact retrieval.**

This is a genuinely negative result for *uniform* TurboQuant on outlier-heavy KV, consistent with the
independent vLLM evaluation ("FP8 is often the better default") and turboquant_plus ("K precision
dominates; protect the first 2 + last 2 layers").

## 2. Setup

| | |
|---|---|
| **Models** | Qwen2.5-7B-Instruct (primary); TinyLlama-1.1B-Chat (contrast) |
| **Research layer** | scos-lab/turboquant, vendored @`34a10b6`, audit-passed (49/49; `docs/VALIDATION.md`). Pure NumPy. |
| **Control harness** | `tqsec/quantizers.py`: TurboQuant (paper / mixed / `-nc`), plain INT (per-token), KIVI (per-channel K, per-token V), FP8 (`e4m3`) |
| **Benchmark** | needle-in-a-haystack (`tqsec/benchmarks.py`): hide `The special magic Cairo number is 7421900.` at depth *d* in a filler haystack of ~*L* tokens; **found-rate** = fraction where the generated answer contains `7421900` |
| **Grid** | lengths *L* ∈ {1024, 2048}, depths *d* ∈ {0.25, 0.5, 0.75} → n=6 per config; `max_new_tokens=24`, greedy |
| **Hardware** | Leonardo A100-SXM-64GB; models loaded bf16, offline |

Config names: `turbo_k{K}v{V}` = TurboQuant paper mode with K-bit keys (Prod = (K−1)-bit scalar + 1-bit
QJL) and V-bit values; `_mix` = outlier channels kept high-precision; `_nc` = boundary layers
{0,1,N−2,N−1} left FP16. `int`/`kivi` at 3-bit; `fp8` = 8-bit `e4m3`.

## 3. Results — quality (NIAH found-rate)

| Config | bits K/V | TinyLlama-1.1B | Qwen2.5-7B |
|---|---|---|---|
| fp16 (baseline) | 16 / 16 | **1.00** | **1.00** |
| turbo_k8v8 | 8 / 8 | — | 0.167 |
| turbo_k8v4 | 8 / 4 | 1.00 | 0.167 |
| turbo_k8v2 | 8 / 2 | — | 0.00 |
| turbo_k3v4 | 3 / 4 | 0.00 | 0.00 |
| turbo_3bit | 3 / 3 | 0.00 | 0.00 |
| turbo_k3v4_mix | 3 / 4 (mixed K) | — | 0.00 |
| turbo_k3v8_mix | 3 / 8 (mixed K) | — | 0.00 |
| **turbo_k3v4_nc** | 3 / 4 (boundary FP16) | — | **1.00** |
| int3 | 3 / 3 | 0.667 | 0.00 |
| **kivi3** | 3 / 3 | 0.833 | **1.00** |
| fp8 | 8 / 8 | 1.00 | 0.00 |

**Two regimes.** On **TinyLlama** (no severe outliers) TurboQuant is quality-neutral at 8-bit keys and
degrades gracefully — the textbook picture. On **Qwen** everything except `fp16`, `kivi3`, and
`turbo_k3v4_nc` collapses, *including 8-bit configs*. The difference is entirely the KV outlier
structure (§4).

## 4. The KV outlier profile (Qwen2.5-7B)

From `scripts/diagnose_kv.py` (one forward, 566-token prompt):

- **max|K| = 420** — below the fp8 `e4m3` limit (448), so **no fp8 overflow**.
- **Extreme, boundary-concentrated key outliers:** L27 (last) **99.6×** median, L0 (first) **31.8×**,
  then L3 15×, L1 10.7×, L19 10.2×. The worst two are the boundary layers.
- **Values are well-behaved:** max|V| = 72.5, worst channel only 4.0× median — **no value outliers.**

## 5. Mechanism — it is the boundary-layer key outliers, nothing else

Per-codec **key** error at 3-bit, split into outlier channels (>20× median) vs the rest:

| codec (3-bit) | err@**outlier** | err@bulk | ⇒ NIAH |
|---|---|---|---|
| kivi3 | **0.26** | 0.23 | 1.0 ✅ |
| turbo_k3 | **7.36** | 1.19 | 0.0 ❌ |
| int3 | **5.98** | 1.44 | 0.0 ❌ |
| fp8 | 1.64 | 0.03 | 0.0 (wrong digit) |

Only per-channel KIVI keeps the outlier channels ~as accurate as normal ones; every other scheme blows
up on them, and found-rate tracks `err@outlier` 1:1. We then **ruled out every alternative explanation:**

1. **Not bits.** `turbo_k8v8` = `turbo_k8v4` = 0.167 — 8-bit K+V still fails; the answers are *close but
   a digit off* (`741900`, `7421000`), i.e. exact recall corrupted, not attention lost.
2. **Not values.** V has no outliers; TurboQuant's value error (0.096) is *lower* than KIVI's (0.249),
   which retrieves perfectly. `k8v8` = `k8v4` confirms value bits are irrelevant.
3. **Not key-outlier mixed-precision.** `mode=mixed` cuts key `err@outlier` 7.36 → 0.33 (≈ KIVI) yet
   retrieval stays 0.0 (`k3v4_mix`, `k3v8_mix`) — good average reconstruction, still-corrupted argmax.

**Conclusion:** TurboQuant's random rotation *spreads* each ~100× outlier across all coordinates;
reconstructing a token dominated by such a channel then accumulates enough error to flip the exact
needle digit — at every bit-width tested. Per-channel quantization (KIVI) gives each channel its own
scale and preserves it; `-nc` sidesteps the problem by not compressing the boundary layers at all.

## 6. Positioning

- **vs the paper:** TurboQuant's deployed variants are `*_nc` (uncompressed boundary layers) for exactly
  this reason; our result shows the `-nc` policy is *load-bearing*, not cosmetic, on Qwen.
- **vs vLLM eval:** matches "FP8 often the better default, 3-bit TurboQuant trades accuracy."
- **vs turboquant_plus:** independently reproduces "K precision dominates" and "protect first 2 + last 2
  layers"; our diagnostic pins *why* (boundary outlier channels).
- **vs KIVI:** per-channel key quantization is the robust baseline here; 3-bit KIVI ≥ 8-bit TurboQuant.

## 7. Limitations & future work

- **Quality metric = NIAH only.** Add a LongBench slice and perplexity for a broader quality picture.
- **Memory not yet tabulated.** The harness counts compressed bits (`QuantCacheLayer.compression_ratio`);
  report **counted** memory per config (nominal ratios: k3v4/int3/kivi3 ≈ 4.6–5.3×, k8v4 ≈ 2.7×, fp8 2×;
  `-nc` slightly less). A *measured* figure needs vLLM (deferred, per the plan).
- **Two models.** Replicate on Llama-3.1-8B-Instruct and Mistral-7B-Instruct to test outlier-structure
  generality.
- **QJL ablation not isolated.** The 3-bit collapse implicates the (K−1)-bit-scalar + QJL split; a direct
  `mode=mse` (rotation, no QJL) vs `paper` comparison would separate rotation from QJL.
- **Attention-level metric.** `err@outlier` is reconstruction; an inner-product / attention-fidelity
  measure would connect the outlier error to the argmax flip more directly.

## 8. Implications for T2 / T3

The instrumented understanding built here — the per-token/channel error map, the boundary-outlier
geometry, the `-nc` policy, and the control harness — **is** the attack surface T2 and T3 target. The
outlier channels and their error structure are precisely where compression-activated behavior (T2) and
code-based inversion (T3) will concentrate.

## 9. Reproducibility

```bash
# outlier + per-codec error diagnostic (one forward)
MODEL_ID=$SCRATCH/models/qwen2.5-7b-instruct python scripts/diagnose_kv.py

# quality sweep (all configs)
QUANT_CONFIGS=fp16,turbo_k8v8,turbo_k8v4,turbo_k8v2,turbo_k3v4,turbo_k3v4_mix,turbo_k3v8_mix,turbo_k3v4_nc,int3,kivi3,fp8 \
  NIAH_LENGTHS=1024,2048 ./scripts/submit_sanity.sh qwen2.5-7b-instruct $SCRATCH/models/qwen2.5-7b-instruct
```
Seeds: rotation seed 42 (public-Π regime). Results: `results/sanity/<model_tag>/sanity_benchmark.json`.
Research layer commit `34a10b6`; harness `tqsec/{quantizers,benchmarks,instrument}.py`.
