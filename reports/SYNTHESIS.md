# Synthesis

## Status
Final Leonardo T1/T2/T3 runs are complete on `fahimDev2`.

## Track Verdicts
| track | verdict | evidence |
|---|---|---|
| T1 | complete | quality/perplexity and counted KV-cache memory results are implemented and reported |
| T2 | negative/partial | no generated text canary on TinyLlama or Mistral; the soft-trigger objective separates compressed-KV from FP-KV on the target token |
| T3 | mixed, no robust TurboQuant-specific worsening | TinyLlama and Mistral show small increases that are shared by controls; Llama3 mitigates; Qwen3 is neutral/public and lower/secret |

## Public Vs Secret Pi
| track | metric | public Pi | secret Pi | effect |
|---|---|---:|---:|---|
| T2 | generated canary fire rate | 0.0 | 0.0 | no text canary in either regime |
| T3 TinyLlama | `turbo_k3v4` token recovery | 0.2667 | 0.2444 | secret Pi removes the small public increase |
| T3 Mistral-7B | `turbo_k3v4` token recovery | 0.2683 | 0.2683 | no measured difference |
| T3 Llama3-8B | `turbo_k3v4` token recovery | 0.3590 | 0.3846 | both remain below FP recovery |
| T3 Qwen3-8B | `turbo_k3v4` token recovery | 0.3056 | 0.2500 | secret Pi lowers recovery |

## Mitigations
| finding | mitigation | evidence | residual risk |
|---|---|---|---|
| Boundary-layer outliers can dominate some compressed-cache settings | fixed `-nc` policy that keeps boundary-sensitive layers uncompressed | T1 reports and T3 `turbo_k3v4_nc` return last-layer leakage to the FP baseline | lower compression ratio |
| Public Pi can expose reusable compression geometry | secret per-deployment Pi | T3 lowers recovery on TinyLlama and Qwen3; T2 shows no text canary under either Pi setting | mixed on Mistral and Llama3 |
| The T2 soft objective can separate compressed-KV from FP-KV | served-cache verification, randomized Pi, and deployment-specific quantization policy | strong soft-trigger run reaches target probability `0.998163` under compression and `0.000640` under FP-KV | not yet a text-level or weight-level backdoor |

## Result Files
| track | files |
|---|---|
| T2 canary | `results/t2_behavior/tinyllama-final-20260706-full/canary_behavior.json`, `results/t2_behavior/mistral-7b-final-20260706/canary_behavior.json` |
| T2 soft trigger | `results/t2_behavior/tinyllama-final-20260706-soft/soft_trigger.json`, `results/t2_behavior/tinyllama-final-20260706-soft-w5/soft_trigger.json` |
| T3 leakage | `results/t3_leakage/tinyllama-final-20260706/leakage.json`, `results/t3_leakage/mistral-7b-final-20260706/leakage.json`, `results/t3_leakage/llama3-8b-final-20260706/leakage.json`, `results/t3_leakage/qwen3-8b-final-20260706/leakage.json` |

## Thesis Claim
The completed branch supports a faithful TurboQuant security evaluation with the following claims:

- T1 establishes the implemented TurboQuant baseline and memory-quality tradeoffs.
- T2 does not find a clean generated canary attack on the evaluated base models, but it shows a working differentiable route for compression-conditioned behavior.
- T3 does not show robust TurboQuant-specific leakage worsening. Leakage effects are model-dependent and often match or underperform simpler compression controls.
- Secret Pi is useful as a hardening measure, especially for TinyLlama and Qwen3 in the leakage experiment, but it is not a universal guarantee.

## Reproducibility
- validation: `docs/VALIDATION.md`
- runbook: `docs/RUNBOOK.md`
- architecture: `docs/ARCHITECTURE.md`
- ethics: `docs/ETHICS.md`
- results: `results/`
