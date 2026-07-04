# ETHICS

## Scope
- Open models only.
- In-lab evaluation only.
- Benign canary targets only.
- No harmful payloads, operational misuse steps, or deployment guidance for attacks.

## T2
The target behavior is a harmless marker string or fixed wrong answer. Success is defined as:

`FP-KV: canary absent` and `TurboQuant-KV: canary present`.

This tests compression-conditioned behavior without producing a useful harmful capability.

## T3
Prompt recovery is evaluated on controlled prompts and benchmark text. Report token recovery and semantic recovery separately. Do not publish recovered private or user-provided text.

## Disclosure
If a prompt-only exploit or strong leakage result transfers to deployed TurboQuant/vLLM paths, notify maintainers before public release. Include:

- affected variant and model
- reproduction conditions
- risk summary
- mitigation tested

## Mitigations To Test
- secret per-deployment rotation
- per-layer rotation
- served-KV verification check
- disabling vulnerable low-bit modes

