# T3 — Prompt leakage from TurboQuant codes (Student 3)

**Question (binary):** does TurboQuant **MITIGATE** inversion (coarsens info) or **WORSEN** it
(rotation + low-bit code = cleaner attack surface)? `PROJECT_PLAN.md` §5 · T3.

## Sequence
1. **Baseline (W4-6):** reproduce an existing KV/embedding inversion attack on **uncompressed**
   cache → the FP-KV leakage baseline. (The predecessor's nearest-neighbor proxy in
   `experiments/prompt_leakage.py` is a *skeleton*, not the attack — keep its cross-layer/per-bit
   loop structure, replace the proxy with a learned inverter.)
2. **Learned inverter (W6-9):** train an inverter on the **TurboQuant codes**. Measure
   **token-level and semantic recovery separately**, per bit-width, as a delta from FP-KV.
3. **Π regime (W9-10):** rerun public-Π vs secret-Π — does secret Π close the leak?
4. **Ablation:** leakage under TurboQuant vs plain INT vs FP8 codes.
5. **(STRETCH) Sharp hypothesis:** does TurboQuant preserve the *semantically* leak-relevant
   signal while destroying the rest — worsening prompt-level inversion even as vector distortion
   improves? Pursue only once the binary result is in hand.

**Deliverable:** `reports/T3_leakage.md` — token vs semantic recovery curves per bit-width,
FP-baseline delta, Π public-vs-secret, mitigate-vs-worsen verdict with evidence.
