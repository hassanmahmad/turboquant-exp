# T1 — Characterization (Student 1)

**Question:** memory savings, latency, and quality of TurboQuant across bit-widths.
The legitimately-completable baseline. `PROJECT_PLAN.md` §5 · T1.

## Required (HF-only)
- Quality (LongBench + needle) and **counted/theoretical memory** (bytes stored: codes + scales +
  QJL vs FP16) across `FP16, k8v4, 4bit_nc, k3v4_nc, 3bit_nc` **+ FP8**.
- Document the `-nc` policy (which boundary layers are uncompressed) and hold it constant.
- First-class question raised by the audit: **is QJL actually helping?** (One PyTorch port reports
  it can hurt.) Characterize MSE-vs-Prod, QJL on/off.

## Optional (only if a *measured* claim is required)
- Bolt on vLLM (`--kv-cache-dtype turboquant_*`); record measured memory + latency; cross-check
  distortion vs the research layer. Flag the A100-vs-paper(H100) speedup caveat.

**Deliverable:** `reports/T1_characterization.md`. State for every memory number whether it is
**counted** or **measured**.

Relabel note: the predecessor's `behavior_divergence` is T1 degradation, not T2.
