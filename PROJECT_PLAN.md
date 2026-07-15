# TurboQuant KV-Cache Security — Project Plan

**Status:** essentially complete. Phase 0 (reuse audit), Phase 1 (instrumented foundation), and all three tracks have final Leonardo runs. Remaining work is optional depth and the write-up (§5).

**Core principle:** reuse a *faithful* TurboQuant; build only the instrumentation and the attacks.

**Headline findings:**
- **T1** — a model's KV outlier ratio predicts whether uniform TurboQuant works. Quality-neutral ~3.5 bits, degrades ~2.5 bits (matches paper); Qwen's ~100× boundary-layer outliers break it at every bit-width, where KIVI or the `-nc` policy are needed.
- **T2** — *positive with nuance.* A trigger-gated, weight-level LoRA backdoor fires a benign canary **only** under compressed-KV (FP-KV == base), replicating on TinyLlama-1.1B and Mistral-7B — but it is **not** TurboQuant-specific. A public-Π attacker keys on TurboQuant geometry (specificity 0.9–1.0) yet is defeated by fresh secret Π (fire → 0.00–0.07); a Π-robust attacker survives secret Π (~1.0 on unseen rotations) but is generic (INT3 fires it too, specificity 0.0–0.2). So the real attack surface is *aggressive 3-bit KV compression as an activation side-channel*, and the robust defense is **keeping KV at 8-bit** (8-bit keys defeat both attackers).
- **T3** — mixed; **no** robust TurboQuant-specific leakage worsening. Effects are model-dependent and often matched or exceeded by INT/KIVI/FP8 controls.

(Greenfield restart of `../turboquant-kv-cache-experiments`, whose cache was a TurboQuant-*style* approximation — uniform-affine quant + a non-QJL sign residual — i.e. the wrong object of study for a TurboQuant-specificity claim.)

---

## 1. The three tracks

| Track | Question | Type |
|---|---|---|
| **T1 — Characterization** | Implement & study TurboQuant from an existing codebase: memory, latency, quality on benchmarks. | Systems / baseline |
| **T2 — Compression-activated behavior** | Can a model behave normally under FP-KV but fire a specific benign-canary behavior **only when TurboQuant is enabled**, exploiting the rotation + 1-bit-residual error geometry rather than coarse rounding? | Attack |
| **T3 — Prompt leakage** | Does TurboQuant **mitigate** KV-cache inversion (coarsens info) or **worsen** it (rotation + low-bit code is a cleaner attack surface)? | Attack |

### Scope ladder (decided — do not re-litigate)

**Required (minimum viable):** T1 quality across bit-widths + counted memory; T2 Version A (trained benign-canary backdoor + INT/KIVI ablation); T3 binary question (mitigate vs worsen) with FP baseline and token-vs-semantic recovery reported separately; the cross-cutting public-vs-secret Π comparison and `{TurboQuant, INT3/4, KIVI, FP8}` ablation harness.

**Stretch (only after required lands):** T2 Version B (prompt-only, no-training exploit); T3 sharp hypothesis ("preserves semantic signal while improving vector-level distortion"); vLLM measured efficiency for T1.

**Out of scope:** a genuinely harmful payload (benign canary proves the same existence claim, safer); "undesired behavior" as a target (collapses into ordinary degradation — the trap the old repo fell into). The bar is a *specific, triggered, FP-absent / TurboQuant-present* behavior.

---

## 2. Strategic decisions (decided — do not re-litigate)

1. **Don't reimplement the algorithm.** Reuse a faithful public TurboQuant; spend the time on instrumentation + attacks.
2. **HF-first, two-layer stack:**
   - **Research layer (mandatory — all instrumentation + T1 quality + T2 + T3):** vendored fork of **[scos-lab/turboquant](https://github.com/scos-lab/turboquant)** run through plain HuggingFace `transformers`. Faithful rotation + Lloyd–Max + real 1-bit QJL, HF `Cache` integration, K+V asymmetric bits, exposes internals (Π, codes, per-channel stats), built-in compressed-bit counting (= T1 counted memory). Pure **NumPy** — slow, and its `.numpy()` round-trip **severs autograd**, so it gives introspection but not gradients.
   - **Production oracle (optional, T1-only, deferred):** upstream vLLM `--kv-cache-dtype turboquant_*` (PR #38479, merged 2026-04-15). Real kernels → *measured* memory + latency, but a WHT-rotation/uniform-value **variant**. Add only if T1 must report measured efficiency; it can't do T2/T3 (no gradients, no dumps).
3. **Vet before trusting.** The field is full of unfaithful auto-ports. Acceptance gate for any reused quantizer: (a) Lloyd–Max levels (not min/max affine), (b) a real QJL residual = random JL projection → sign, with a passing inner-product-bias→0 test. **scos-lab passes** (code-read + 49/49 tests, executed 2026-06-28; see `docs/VALIDATION.md`).
4. **Fidelity spectrum — target the rotation, not just QJL.** Even "real" TurboQuant varies: vLLM and TheTom/turboquant_plus drop QJL on the value stream, and tonbistudio/turboquant-pytorch reports QJL sometimes *hurt*. Rotation is first-order, scalar-quant/QJL second-order. So keep QJL in scope on the faithful layer, but make **rotation geometry + the public-vs-secret Π axis** the primary lever — it is present in faithful TQ, the WHT variants, and every deployment.
5. **T2 needs a PyTorch differentiable twin.** scos-lab can't backprop, so T2's in-the-loop training uses an STE twin (`tqsec/diff_twin.py`), forward-validated against scos-lab's frozen Lloyd–Max centroids + QJL matrix. Seeded from tonbistudio/turboquant-pytorch.

---

## 3. Compute

Everything runs on **HPC (Leonardo/CINECA, A100)** via **plain HuggingFace `transformers`** — correctness + introspection + gradients + small batch. **Memory is reported as a counted/theoretical figure** (bytes stored: codes + scales + QJL bits, vs FP16) — always stated as *computed*, not *measured*. Only vLLM gives measured memory/latency; the research layer is NumPy and runs *slower* than FP16, so no speedup claim comes from it (the paper's ~8× needs an H100). HPC gotcha: compute nodes have no internet → pre-download models/datasets to `$SCRATCH` and run with `HF_HUB_OFFLINE=1`.

---

## 4. Repo layout

```
tqsec/                  # the thin layer we own
  instrument.py         # dump Π, codes, QJL residual, per-token/channel error map
  quantizers.py         # {TurboQuant, INT3/4, KIVI, FP8} control harness + the `-nc` boundary-layer policy
  pi_regime.py          # public/reused vs secret/per-deployment Π switch
  diff_twin.py          # PyTorch STE differentiable twin of TQ (for T2)
  inversion.py          # T3 learned inverter
  metrics.py            # token + semantic recovery, JS divergence, distortion
  benchmarks.py         # needle-in-haystack / LongBench loaders
third_party/turboquant/ # vendored scos-lab/turboquant (research layer, read-only)
t1_characterization/  t2_behavior/  t3_leakage/
env/ scripts/ slurm/    # Leonardo setup, utilities, batch jobs
results/ reports/ docs/
```

---

## 5. Phased plan

Phase 0–1 are the shared critical path; T2 and T3 run in parallel once the foundation is green. `[x]` = done.

### Phase 0 — Foundations & go/no-go on reuse
**Goal:** prove the reused pieces work before committing.
- [x] **Audit the scos-lab fork against the paper (the gate)** — code-read + 49/49 tests, 2026-06-28: **passes** (real Lloyd–Max on the post-rotation Beta density; true `sign(Sx)` QJL with a passing inner-product-bias→0 test). Gap: `-nc` boundary-layer handling isn't in scos-lab → `tqsec/quantizers.py` adds it. Then port its `Cache` integration to Llama-3.1-8B and Mistral-7B.
- [x] Smoke model: **TinyLlama-1.1B** for fast iteration before every 8B run.
- **Gate:** `docs/VALIDATION.md` records "research layer faithful: yes, evidence." Fallback had it failed: tonbistudio/turboquant-pytorch → OmarHory/AmesianX (C++) → reimplement.

### Phase 1 — Shared instrumented foundation + T1 quality baseline
Student 1 leads; Students 2 & 3 build their FP-KV baselines in this window.
- [x] **`instrument.py`** — dump per layer/head/token/channel: Π, codes, QJL residual, reconstruction error (true KV − reconstructed). Produce the **error map** — the raw material for T2 and T3; nothing downstream starts without it.
- [x] **`quantizers.py` control harness** — {TurboQuant, INT3/4, KIVI, FP8-KV}. TurboQuant-specificity is unprovable without it; FP8 is the "is TurboQuant even worth it" baseline.
- [x] **`pi_regime.py`** — public/reused vs secret/per-deployment Π. The single biggest security variable.
- [x] **Sanity benchmark** — needle / LongBench slice; confirm quality-neutral ~3.5 bits, degrades ~2.5 bits.

### Phase 2 — Parallel tracks

**T1 — Characterization** (`reports/T1_characterization.md`)
- [x] Quality (needle) + counted memory across {TurboQuant, INT, KIVI, FP8} + `-nc` on four models; `-nc` policy held constant. **Result: a model's KV outlier ratio predicts whether uniform TurboQuant works** (Qwen's ~100× boundary outliers break it → need KIVI or `-nc`).
- [ ] Optional: LongBench, perplexity, QJL ablation; vLLM measured efficiency (memory + latency).

**T2 — Compression-activated behavior** (`reports/T2_behavior.md`, cross-model table `reports/T2_backdoor_autogen.md`)
Target is a **benign canary** — specific, FP-absent / TurboQuant-present / triggered — **not** output drift.
- [x] **Version A — trained backdoor:** fine-tune with the quantizer in the loop on the **STE differentiable twin** (`diff_twin.py`, forward-validated vs frozen scos-lab centroids), evaluated by real generation on the faithful quantizer. Fires only under compressed-KV; FP == base; replicates on TinyLlama + Mistral-7B.
- [x] **Mandatory ablation, both Π regimes:** public-Π attacker is TurboQuant-specific but defeated by secret Π; Π-robust attacker survives secret Π but is generic (INT3 fires it). 8-bit keys defeat both.
- [ ] Version B — prompt-only error-geometry exploit (stretch; attempt only after A + the error map).

**T3 — Prompt leakage** (`reports/T3_leakage.md`)
Binary question: does TurboQuant **mitigate** or **worsen** inversion?
- [x] FP-KV inversion baseline; learned inverter on TurboQuant codes; token-level and semantic recovery reported **separately**, per bit-width, as delta from FP; both Π regimes; INT/FP8 ablation. **Verdict: no robust TurboQuant-specific worsening.**
- [ ] Sharp hypothesis (stretch): preserves semantic signal while improving vector-level distortion.

### Phase 3 — Synthesis, mitigations, write-up
- [x] Cross-cutting: how Π-secrecy moves attack success across T2 and T3 (`reports/SYNTHESIS.md`), each landed attack paired with a candidate mitigation (secret Π, 8-bit keys).
- [ ] Responsible-disclosure plan if Version B or a real leak lands; final report + reproducibility appendix (seeds, exact variants, `-nc` policy).

---

## 6. Non-negotiable controls (every T2/T3 experiment)

1. **Specificity ablation** — every attack also runs against plain INT3/INT4 and KIVI. Works equally under coarse rounding → novelty claim dead.
2. **FP-KV baseline first** — establish uncompressed behavior/leakage, report deltas.
3. **Token-level vs semantic metrics kept separate** (T3).
4. **Both Π regimes reported** (public/reused vs secret/per-deployment).
5. **`-nc` policy fixed and documented** across models.

---

## 7. Models
- **Smoke / iteration:** TinyLlama-1.1B-Chat.
- **Faithful-layer first target:** Qwen2.5-7B-Instruct (scos-lab's tested ceiling).
- **Primary:** Llama-3.1-8B-Instruct + Mistral-7B-Instruct.

---

## 8. Ethics
In-lab, open models only. **Benign canary target, never a harmful payload.** Each attack paired with a candidate mitigation. Responsible disclosure to TurboQuant/vLLM maintainers before any public release of a working exploit. See `docs/ETHICS.md`.

---

## 9. References
- **Paper:** TurboQuant — *Online Vector Quantization with Near-optimal Distortion Rate* (Zandieh, Daliri, Hadian, Mirrokni). arXiv:2504.19874; ICLR 2026.
- **Research layer:** github.com/scos-lab/turboquant (faithful, HF, MIT, NumPy; audit-passed 2026-06-28). PyTorch alternate / T2-twin seed: github.com/tonbistudio/turboquant-pytorch.
- **Production oracle (T1-only, deferred):** vLLM `--kv-cache-dtype turboquant_*` (PR vllm-project/vllm#38479, merged 2026-04-15) — WHT + uniform-value variant. Same "deployed reality" role: github.com/TheTom/turboquant_plus (source of validated findings — V-compression-is-free, boundary layers sensitive → validates `-nc`, block_size=128, asymmetric K/V).
- **Security prior art:** *Exploiting LLM Quantization* (NeurIPS 2024), *Mind the Gap* (ICML 2025), *Watch Your Steps* (ICLR 2026), *RobustKV* (ICLR 2025).
