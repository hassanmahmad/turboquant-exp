# Synthesis

## Status
Final Leonardo T1/T2 runs are complete. T3 was rerun on 2026-11-24 with a larger protocol (120 sentences, three layers, five splits, bootstrap CIs); the July T3 runs are superseded.

**Model note — two different Qwen models.** T1 characterizes **Qwen2.5-7B-Instruct** (28 layers, ~100× boundary-layer key outliers; `results/{quality,sanity}/qwen2.5-7b-instruct/`). The July T3 runs used **Qwen3-8B** (36 layers, no-think mode), which has a milder outlier profile (`turbo_k8v4` = 1.00 NIAH in `results/quality/qwen3-8b-nothink-*`). The T3 rerun covers both models, in separate rows. "Qwen" findings do not transfer between the two: T1's Prod/QJL collapse is a Qwen2.5 result.

## Track Verdicts
| track | verdict | evidence |
|---|---|---|
| T1 | complete | quality/perplexity and counted KV-cache memory results are implemented and reported |
| T2 | positive, not TurboQuant-specific | a stealthy, trigger-gated, weight-level LoRA backdoor fires the canary only under compressed-KV and replicates on TinyLlama and Mistral-7B; the public-Π variant is TurboQuant-specific (specificity 0.9–1.0) but defeated by secret Π, while the Π-robust variant survives secret Π (~1.0) but is generic (INT3 fires it too), so the surface is aggressive 3-bit KV compression, not TurboQuant geometry |
| T3 | small mitigation, partly generic, not a defence | rerun on five models, three layers, 2,600-3,300 held-out tokens each: 3-bit TurboQuant lowers linear-inverter token recovery by 1.5-12 points at the first and last layer (all CIs exclude zero) and never raises it beyond noise; INT3 shows a smaller drop of the same kind; recovery stays at 29-53% (`reports/T3_leakage.md`) |

## Public Vs Secret Pi
| track | metric | public Pi | secret Pi | effect |
|---|---|---:|---:|---|
| T2 (public-Π attacker) | LoRA canary compressed-only rate (Mistral / TinyLlama) | 0.90 / 1.00 | 0.00 / 0.07 | secret per-deployment Π fully defeats the TurboQuant-specific backdoor |
| T2 (Π-robust attacker) | LoRA canary compressed-only rate (Mistral / TinyLlama) | 1.00 / 1.00 | 1.00 / 0.99 | secret Π fails, but the attack is generic (INT3 fires it too), not TurboQuant-specific |
| T3 TinyLlama | `turbo_k3v4` token recovery, layers 0 / 11 / 21 | 0.366 / 0.307 / 0.348 | 0.229 / 0.291 / 0.326 | lower under secret Π at every layer |
| T3 Mistral-7B | `turbo_k3v4` token recovery, layers 0 / 16 / 31 | 0.390 / 0.340 / 0.415 | 0.273 / 0.305 / 0.411 | large drop at layer 0, 0.4 points at the last layer |
| T3 Llama-3.1-8B | `turbo_k3v4` token recovery, layers 0 / 16 / 31 | 0.486 / 0.376 / 0.418 | 0.461 / 0.354 / 0.414 | 0.4-2.5 points lower |
| T3 Qwen2.5-7B | `turbo_k3v4` token recovery, layers 0 / 14 / 27 | 0.443 / 0.295 / 0.312 | 0.001 / 0.264 / 0.257 | collapse at layer 0 |
| T3 Qwen3-8B | `turbo_k3v4` token recovery, layers 0 / 18 / 35 | 0.534 / 0.286 / 0.389 | 0.012 / 0.227 / 0.319 | collapse at layer 0 |

The T3 secret-Π attacker trains its inverter on one public rotation only. T2 showed that training across random rotations defeats secret Π; that Π-robust inverter has not been run for T3, so the T3 column is an upper bound on what secret Π buys.

## Mitigations
| finding | mitigation | evidence | residual risk |
|---|---|---|---|
| Boundary-layer outliers can dominate some compressed-cache settings | fixed `-nc` policy that keeps boundary-sensitive layers uncompressed | T1 reports | lower compression ratio; the uncompressed boundary layers leak exactly as much as FP, and layer 0 is the most invertible layer in T3 |
| Public Pi can expose reusable compression geometry | secret per-deployment Pi | T2's public-Π backdoor drops to 0.00–0.07 under secret Π, but a Π-robust attacker survives (~1.0 on unseen rotations); in T3 an inverter trained on the public rotation recovers 0.4–52 points fewer tokens on secret-Π keys than on public-Π keys | closes the TurboQuant-specific T2 variant but not the generic one; the T3 benefit is untested against a Π-robust inverter |
| A stealthy weight-level LoRA backdoor fires the canary only under compressed (3-bit) KV | keep KV at 8-bit (`turbo_k8v4`); secret per-deployment Π (specific variant only); served-cache verification | mild 8-bit keys fire 0.00 on both models; secret Π drops the public-Π attacker to 0.00–0.07 but the Π-robust attacker stays 0.99–1.00 | Π-robust attacker survives secret Π and is FP-clean (perplexity within 5% of base on Mistral, zero clean-prompt leak), so FP-path behavioural detection is not a safety net |

## Result Files
| track | files |
|---|---|
| T2 canary | `results/t2_behavior/tinyllama-final-20260706-full/canary_behavior.json`, `results/t2_behavior/mistral-7b-final-20260706/canary_behavior.json` |
| T2 soft trigger | `results/t2_behavior/tinyllama-final-20260706-soft/soft_trigger.json`, `results/t2_behavior/tinyllama-final-20260706-soft-w5/soft_trigger.json` |
| T2 LoRA backdoor (main result) | `results/t2_behavior/{tinyllama,mistral}-backdoor-{public,random}/backdoor.json` |
| T3 leakage | `results/t3_leakage/{tinyllama-1.1b-chat-v1.0,mistral-7b-instruct-v0.3,llama3.1-8b-instruct,qwen2.5-7b-instruct,qwen3-8b}-ci-rerun/leakage.json`; superseded July runs in `results/t3_leakage/*-final-20260706/` |

## Thesis Claim
The completed branch supports a faithful TurboQuant security evaluation with the following claims:

- T1 establishes the implemented TurboQuant baseline and memory-quality tradeoffs.
- T2 demonstrates a stealthy, trigger-gated, weight-level LoRA backdoor that fires the benign canary only under compressed-KV and replicates on TinyLlama and Mistral-7B, but the variant that survives secret Π is generic to aggressive 3-bit KV quantization (INT3 fires it too), so the attack surface is compression-as-an-activation-side-channel rather than TurboQuant's specific geometry; keeping KV at 8-bit is the robust defense, and at 7B scale the robustness↔stealth↔gating trade-off that constrained the tiny model no longer binds.
- T3 shows that TurboQuant does not worsen key-vector leakage and mitigates it only slightly: a linear inverter recovers 28–57% of held-out tokens from uncompressed keys and 29–53% from 3-bit TurboQuant keys, with a significant drop of 1.5–12 points at the first and last layer on all five models. INT3 produces a smaller drop of the same kind, so the effect is mostly lost information rather than TurboQuant geometry, and compression is not a privacy defence.
- Secret Pi is useful as a hardening measure against the TurboQuant-specific T2 backdoor, but it is not a universal guarantee: the Π-robust backdoor survives it, and its apparent T3 benefit is measured only against an inverter that never saw other rotations.

## Reproducibility
- validation: `docs/VALIDATION.md`
- runbook: `docs/RUNBOOK.md`
- architecture: `docs/ARCHITECTURE.md`
- ethics: `docs/ETHICS.md`
- results: `results/`
