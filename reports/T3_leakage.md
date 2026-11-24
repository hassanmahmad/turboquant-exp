# T3 Leakage

## Status
Completed final Leonardo runs on 2026-07-06.

## Jobs
| job | model | state |
|---:|---|---|
| 48646235 | TinyLlama | completed |
| 48646341 | Mistral-7B | completed |
| 48646342 | Llama3-8B | completed |
| 48646343 | Qwen3-8B | completed |

**Model note:** the Qwen row is **Qwen3-8B** (36 layers, no-think mode), not the Qwen2.5-7B-Instruct characterized in T1. The two have different KV outlier profiles, so T1's Qwen2.5 collapse results do not describe this model.

## Setup
The runner trains a compact token inverter on held-out key-vector positions. Metrics are exact token recovery and semantic recovery, reported separately. `delta` is token recovery minus the FP-KV baseline for the same model.

## Token Recovery
| model | fp16 | turbo_k8v4 public | turbo_k3v4 public | turbo_k3v4 secret | turbo_k3v4_nc | int3 | kivi3 | fp8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TinyLlama | 0.2444 | 0.2444 | 0.2667 | 0.2444 | 0.2444 | 0.2889 | 0.2444 | 0.2444 |
| Mistral-7B | 0.2439 | 0.2439 | 0.2683 | 0.2683 | 0.2439 | 0.2683 | 0.2683 | 0.2683 |
| Llama3-8B | 0.4103 | 0.4103 | 0.3590 | 0.3846 | 0.4103 | 0.3846 | 0.3846 | 0.3846 |
| Qwen3-8B | 0.3056 | 0.3056 | 0.3056 | 0.2500 | 0.3056 | 0.2500 | 0.3056 | 0.3056 |

## Semantic Recovery
| model | fp16 | turbo_k3v4 public | turbo_k3v4 secret | int3 | kivi3 | fp8 |
|---|---:|---:|---:|---:|---:|---:|
| TinyLlama | 0.7126 | 0.7009 | 0.6930 | 0.7127 | 0.7123 | 0.6897 |
| Mistral-7B | 0.6690 | 0.6973 | 0.7057 | 0.6810 | 0.6889 | 0.6751 |
| Llama3-8B | 0.8358 | 0.7998 | 0.8070 | 0.8156 | 0.8275 | 0.8353 |
| Qwen3-8B | 0.7922 | 0.7981 | 0.7845 | 0.7274 | 0.8106 | 0.7922 |

## Delta Vs FP
Each run has one split with 36-45 held-out tokens, so one token is 0.022-0.028. The table gives the
paired difference against the FP baseline on the same test tokens with a bootstrap 95% CI
(`python scripts/t3_reanalyze.py`, computed from the stored predictions; no model re-run).

| model | test tokens | turbo_k3v4 public | turbo_k3v4 secret | largest control delta | reading |
|---|---:|---|---|---|---|
| TinyLlama | 45 | +0.022 [+0.000, +0.067] (1 token) | +0.000 | +0.044 [+0.000, +0.111] (INT3, 2 tokens) | no significant difference |
| Mistral-7B | 41 | +0.024 [+0.000, +0.073] (1 token) | +0.024 [+0.000, +0.073] | +0.024 (INT3/KIVI3/FP8, 1 token) | no significant difference |
| Llama3-8B | 39 | -0.051 [-0.128, +0.000] (2 tokens) | -0.026 [-0.077, +0.000] | -0.026 (INT3/KIVI3/FP8, 1 token) | no significant difference |
| Qwen3-8B | 36 | +0.000 [-0.083, +0.083] | -0.056 [-0.167, +0.056] | -0.056 [-0.167, +0.056] (INT3) | no significant difference |

No interval excludes zero for any model, config or Pi regime: every observed difference is one to
three tokens. `turbo_k8v4` and `turbo_k3v4_nc` reproduce the FP predictions exactly on all four models.
The `verdict` field stored in the original JSON files (`worsen` for TinyLlama and Mistral, `mitigate`
for Llama3) came from a fixed +/-0.02 threshold, which a single token crosses; it is superseded by this
table.

## Verdict
The final T3 runs do **not** show TurboQuant-specific leakage worsening, and they are too small to
show mitigation either.

At this sample size the result is a null: compressing the last-layer keys to 3 bits (TurboQuant, INT or
KIVI) changes at most three of roughly forty recovered tokens, in either direction, and TurboQuant never
separates from the INT/KIVI/FP8 controls. The earlier per-model readings (a "slight increase" on
TinyLlama and Mistral, "mitigation" on Llama3) are within noise and should not be cited as effects.
What the runs do establish is that a linear inverter recovers 24-41% of held-out tokens from last-layer
keys with or without compression, so 3-bit KV compression is not a privacy defence on its own.

Limits of these runs: 8 sentences, one layer (the last), one train/test split by token position. The
runner now defaults to a 120-sentence corpus, held-out sentences, three layers and five splits, with
the CI-based verdict built in (see `t3_leakage/README.md`).

JSON outputs:
- `results/t3_leakage/tinyllama-final-20260706/leakage.json`
- `results/t3_leakage/mistral-7b-final-20260706/leakage.json`
- `results/t3_leakage/llama3-8b-final-20260706/leakage.json`
- `results/t3_leakage/qwen3-8b-final-20260706/leakage.json`
