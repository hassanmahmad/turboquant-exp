# Synthesis

## Status
Final Leonardo T1/T2/T3 runs are complete on `fahimDev2`.

## Track Verdicts
| track | verdict | evidence |
|---|---|---|
| T1 | complete | quality/perplexity and counted KV-cache memory results are implemented and reported |
| T2 | positive, not TurboQuant-specific | a stealthy, trigger-gated, weight-level LoRA backdoor fires the canary only under compressed-KV and replicates on TinyLlama and Mistral-7B; the public-Π variant is TurboQuant-specific (specificity 0.9–1.0) but defeated by secret Π, while the Π-robust variant survives secret Π (~1.0) but is generic (INT3 fires it too), so the surface is aggressive 3-bit KV compression, not TurboQuant geometry |
| T3 | mixed, no robust TurboQuant-specific worsening | TinyLlama and Mistral show small increases that are shared by controls; Llama3 mitigates; Qwen3 is neutral/public and lower/secret |

## Public Vs Secret Pi
| track | metric | public Pi | secret Pi | effect |
|---|---|---:|---:|---|
| T2 (public-Π attacker) | LoRA canary compressed-only rate (Mistral / TinyLlama) | 0.90 / 1.00 | 0.00 / 0.07 | secret per-deployment Π fully defeats the TurboQuant-specific backdoor |
| T2 (Π-robust attacker) | LoRA canary compressed-only rate (Mistral / TinyLlama) | 1.00 / 1.00 | 1.00 / 0.99 | secret Π fails, but the attack is generic (INT3 fires it too), not TurboQuant-specific |
| T3 TinyLlama | `turbo_k3v4` token recovery | 0.2667 | 0.2444 | secret Pi removes the small public increase |
| T3 Mistral-7B | `turbo_k3v4` token recovery | 0.2683 | 0.2683 | no measured difference |
| T3 Llama3-8B | `turbo_k3v4` token recovery | 0.3590 | 0.3846 | both remain below FP recovery |
| T3 Qwen3-8B | `turbo_k3v4` token recovery | 0.3056 | 0.2500 | secret Pi lowers recovery |

## Mitigations
| finding | mitigation | evidence | residual risk |
|---|---|---|---|
| Boundary-layer outliers can dominate some compressed-cache settings | fixed `-nc` policy that keeps boundary-sensitive layers uncompressed | T1 reports and T3 `turbo_k3v4_nc` return last-layer leakage to the FP baseline | lower compression ratio |
| Public Pi can expose reusable compression geometry | secret per-deployment Pi | T3 lowers recovery on TinyLlama and Qwen3; T2's public-Π backdoor drops to 0.00–0.07 under secret Π, but a Π-robust attacker survives (~1.0 on unseen rotations) | mixed: closes the TurboQuant-specific T2 variant but not the generic one; mixed on Mistral and Llama3 for T3 |
| A stealthy weight-level LoRA backdoor fires the canary only under compressed (3-bit) KV | keep KV at 8-bit (`turbo_k8v4`); secret per-deployment Π (specific variant only); served-cache verification | mild 8-bit keys fire 0.00 on both models; secret Π drops the public-Π attacker to 0.00–0.07 but the Π-robust attacker stays 0.99–1.00 | Π-robust attacker survives secret Π and is FP-clean (perplexity within 5% of base on Mistral, zero clean-prompt leak), so FP-path behavioural detection is not a safety net |

## Result Files
| track | files |
|---|---|
| T2 canary | `results/t2_behavior/tinyllama-final-20260706-full/canary_behavior.json`, `results/t2_behavior/mistral-7b-final-20260706/canary_behavior.json` |
| T2 soft trigger | `results/t2_behavior/tinyllama-final-20260706-soft/soft_trigger.json`, `results/t2_behavior/tinyllama-final-20260706-soft-w5/soft_trigger.json` |
| T2 LoRA backdoor (main result) | `results/t2_behavior/{tinyllama,mistral}-backdoor-{public,random}/backdoor.json` |
| T3 leakage | `results/t3_leakage/tinyllama-final-20260706/leakage.json`, `results/t3_leakage/mistral-7b-final-20260706/leakage.json`, `results/t3_leakage/llama3-8b-final-20260706/leakage.json`, `results/t3_leakage/qwen3-8b-final-20260706/leakage.json` |

## Thesis Claim
The completed branch supports a faithful TurboQuant security evaluation with the following claims:

- T1 establishes the implemented TurboQuant baseline and memory-quality tradeoffs.
- T2 demonstrates a stealthy, trigger-gated, weight-level LoRA backdoor that fires the benign canary only under compressed-KV and replicates on TinyLlama and Mistral-7B, but the variant that survives secret Π is generic to aggressive 3-bit KV quantization (INT3 fires it too), so the attack surface is compression-as-an-activation-side-channel rather than TurboQuant's specific geometry; keeping KV at 8-bit is the robust defense, and at 7B scale the robustness↔stealth↔gating trade-off that constrained the tiny model no longer binds.
- T3 does not show robust TurboQuant-specific leakage worsening. Leakage effects are model-dependent and often match or underperform simpler compression controls.
- Secret Pi is useful as a hardening measure, especially for TinyLlama and Qwen3 in the leakage experiment, but it is not a universal guarantee.

## Reproducibility
- validation: `docs/VALIDATION.md`
- runbook: `docs/RUNBOOK.md`
- architecture: `docs/ARCHITECTURE.md`
- ethics: `docs/ETHICS.md`
- results: `results/`
