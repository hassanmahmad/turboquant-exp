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
| model | turbo_k3v4 public | turbo_k3v4 secret | best control delta | reading |
|---|---:|---:|---:|---|
| TinyLlama | +0.0223 | +0.0000 | +0.0445 (INT3) | slight public TurboQuant increase, not TurboQuant-specific |
| Mistral-7B | +0.0244 | +0.0244 | +0.0244 (INT3/KIVI3/FP8) | slight increase shared by controls |
| Llama3-8B | -0.0513 | -0.0257 | -0.0257 (INT3/KIVI3/FP8) | TurboQuant mitigates token recovery |
| Qwen3-8B | +0.0000 | -0.0556 | +0.0000 (KIVI3/FP8) | neutral under public Pi, lower under secret Pi |

## Verdict
The final T3 runs do **not** show robust TurboQuant-specific leakage worsening.

The result is mixed and mostly neutral/mitigating: TinyLlama and Mistral show small token-recovery increases under public `turbo_k3v4`, but the same or larger increases appear under non-TurboQuant controls. Llama3 shows clear mitigation under `turbo_k3v4`; Qwen3 is neutral under public Pi and lower under secret Pi. The `-nc` policy returns recovery to the FP baseline on these last-layer runs.

JSON outputs:
- `results/t3_leakage/tinyllama-final-20260706/leakage.json`
- `results/t3_leakage/mistral-7b-final-20260706/leakage.json`
- `results/t3_leakage/llama3-8b-final-20260706/leakage.json`
- `results/t3_leakage/qwen3-8b-final-20260706/leakage.json`
