# T2 — Compression-activated adversarial behavior (Student 2)

**Question:** can a model be normal under FP-KV yet fire a **specific benign-canary** behavior
*only when TurboQuant is enabled* — exploiting the rotation + 1-bit-residual error geometry, not
coarse rounding? `PROJECT_PLAN.md` §5 · T2.

The bar is **FP-absent / TurboQuant-present / specifically-triggered** — NOT output drift or
degradation (the trap the old repo fell into). Target is a **benign canary** (e.g. trigger → a
fixed odd answer), never a harmful payload.

## Sequence
1. **Baseline (W4-5):** reproduce a standard backdoor with a normal trigger under FP-KV.
2. **Read the error map (W5-6):** from `tqsec/instrument.py`, find the most steerable
   tokens/channels/layers. This defines the attack surface.
3. **Version A — trained backdoor (W6-9, REQUIRED):** fine-tune with the quantizer in the loop,
   dual objective (normal FP-KV / canary under TurboQuant-KV). Non-differentiable quant →
   **STE on `tqsec/diff_twin.py`** (research layer is NumPy; freeze its centroids as reference).
4. **Version B — prompt-only exploit (W9-10, STRETCH):** attempt only after A + the error map.

## Mandatory controls
- **Specificity ablation:** does the canary also fire under plain INT2/3 and KIVI? If yes, the
  TurboQuant-specific (rotation+QJL) claim is unsupported.
- Report under **public-Π and secret-Π**.

**Deliverable:** `reports/T2_behavior.md` — operational canary definition, FP-vs-TurboQuant
contrast on the *same* behavior, the INT/KIVI ablation, Π-regime results.
