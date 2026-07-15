# TurboQuant KV-Cache Security — Project Plan

**Status:** Phase 0 (reuse audit), Phase 1 (instrumented foundation), T1 characterization, runnable T2/T3 harnesses, and final T2/T3 Leonardo runs are done. T2 is positive-with-nuance: a stealthy, trigger-gated, weight-level LoRA backdoor fires the benign canary *only* under compressed-KV and replicates on TinyLlama-1.1B and Mistral-7B — but it is **not** TurboQuant-specific. A public-Π attacker keys on TurboQuant's geometry (specificity 0.9–1.0) yet is defeated by a fresh secret Π; a Π-robust attacker survives secret Π (~1.0 on unseen rotations) but is generic (INT3 fires it too), so the real attack surface is *aggressive 3-bit KV compression as an activation side-channel*, and the robust defense is keeping KV at 8-bit. T3 is mixed, with no robust TurboQuant-specific leakage worsening. (Greenfield restart of `../turboquant-kv-cache-experiments`.)
**Core principle:** *reuse a faithful TurboQuant; build only the instrumentation and the attacks.*
**Stack principle:** *plain HuggingFace `transformers` on HPC is the mandatory foundation; vLLM is a deferred, optional T1-only oracle — added only if T1 must report measured latency.*
**Why restart:** the previous repo implemented a "TurboQuant-*style*" cache (uniform-affine quant + a sign-residual that is **not** QJL, keys-only, all layers, dequantized floats fed back into `DynamicCache`). That is the wrong object of study for a security paper that must claim *TurboQuant-specificity*. This time the quantizer is real and the effort goes where the research is.

---

## 1. The three tasks (from the supervisor brief)

| Track | Owner | Question | Type |
|---|---|---|---|
| **T1 — Characterization** | Student 1 | Implement & study TurboQuant from an existing codebase. Memory savings, latency, quality on benchmark tasks. | Systems / baseline |
| **T2 — Compression-activated adversarial behavior** | Student 2 | Can a model behave normally under FP-KV but fire a **specific, controlled (benign-canary) behavior specifically when TurboQuant is enabled** — exploiting the rotation + 1-bit-residual *error geometry*, not coarse rounding? | Attack |
| **T3 — Prompt leakage from TurboQuant codes** | Student 3 | Does TurboQuant **mitigate** inversion (coarsens info) or **worsen** it (rotation + low-bit code is a cleaner attack surface)? | Attack |

T1 is the legitimately-completable baseline. T2/T3 are the actual research and both depend on a shared, faithful, *instrumented* TurboQuant foundation that does not exist yet.

### Scope ladder — required vs stretch (decided; do not over-build)

**Required (minimum viable project):**
- **T1** — quality impact across bit-widths + **counted/theoretical** memory (bytes stored vs FP16).
- **T2 Version A** — trained backdoor with a **benign canary** target: normal under FP-KV, canary fires under TurboQuant-KV, plus the INT/KIVI ablation.
- **T3** — the **binary** question (mitigate vs worsen) with an FP baseline, per-bit delta, and token-vs-semantic recovery reported separately.
- **Cross-cutting** — the public-vs-secret Π comparison and the `{TurboQuant, INT3/4, KIVI, FP8}` ablation harness.

**Stretch (only after the required line lands):**
- **T2 Version B** — prompt-only error-geometry exploit (no training).
- **T3 sharp hypothesis** — "preserves semantic signal while improving vector-level distortion" (the publishable twist).
- **vLLM measured efficiency** — real latency/throughput + measured memory for T1.

**Explicitly cut / out of scope:**
- A genuinely **harmful** payload — use a benign canary; same existence proof, safer, easier to measure.
- **"Undesired behavior"** as a target — too weak; it collapses into ordinary degradation (the exact trap the old repo fell into). The bar is a *specific, triggered, FP-absent / TurboQuant-present* behavior.

---

## 2. Strategic decisions (decided — do not re-litigate)

1. **Do not reimplement the algorithm.** Faithful TurboQuant now exists publicly. Reuse it; spend the time on instrumentation + attacks.
2. **HF-first, two-layer stack (HF mandatory, vLLM optional):**
   - **Research layer — the mandatory foundation (all instrumentation + T1 quality + T2 + T3):** fork **[scos-lab/turboquant](https://github.com/scos-lab/turboquant)**, run through plain **HuggingFace `transformers`** — faithful rotation + **Lloyd–Max** + real **1-bit QJL**, HF `Cache`-interface integration (`DynamicLayer`), K+V with asymmetric bits, `paper`/`mse`/`mixed` modes, built-in compressed-bit counting (= T1 counted memory for free), exposes internals (rotation matrix, codes, per-channel stats), **49 passing tests** (executed 2026-06-28) incl. the QJL-unbiasedness gate, MIT. **Audit status — code-read + tests both done 2026-06-28: PASSES both gates** (real Lloyd–Max — measured MSE distortion 0.36/0.117/0.034/0.0095 at b=1–4 ≈ paper 0.36/0.117/0.03/0.009; real `sign(Sx)` QJL with the √(π/2)/d unbiased estimator, measured inner-product bias ≈ 1e-4). Pure Python/**NumPy** and slow — *fine*: everything except measured latency is kernel-independent, and it gives the introspection serving kernels can't. **Correction to an earlier assumption: scos-lab is NumPy, not PyTorch — its HF path does a `.numpy()` round-trip that severs autograd, so it provides introspection but NOT gradients.** T2's in-the-loop training therefore needs a **PyTorch differentiable twin** (see §5 · T2). **You can build and run the entire project's instrumentation, T1 quality, and T3 leakage on this layer alone; only T2 training needs the twin.**
   - **Production oracle — deferred, optional, T1-only:** **upstream vLLM** TurboQuant backend (`--kv-cache-dtype turboquant_3bit_nc | turboquant_k8v4 | turboquant_4bit_nc | turboquant_k3v4_nc`; PR #38479 **merged 2026-04-15**, no plugin needed). Real Triton kernels, real *measured* memory + latency. **It's a WHT-rotation/uniform-value variant** — when used, cross-check its distortion vs the scos-lab research layer. **Add it only if T1 must report measured efficiency** — it is a dependency for nothing else, and T2/T3 cannot use it (no gradients, no dumps).
3. **Vet before trusting.** The GitHub field is crowded with auto-generated, unfaithful ports (the same trap the old repo fell into) — and with implausibly high-star repos on a ~2-month-old topic, so **trust code, not stars**. Acceptance gate for *any* reused quantizer: (a) Lloyd–Max levels (not min/max affine), (b) a real QJL residual = random JL projection → sign, with a passing test that **inner-product bias → 0**. **scos-lab passes (verified by code-read 2026-06-28);** upstream vLLM is faithful *enough to serve* but is a WHT/uniform **variant** (see #4); the long tail is presumed guilty until read.
4. **Note the fidelity spectrum (confirmed by reading the code).** Even "real" TurboQuant varies. **Upstream vLLM (#38479, merged 2026-04-15)** uses **WHT (Walsh–Hadamard) rotation + Lloyd–Max for keys and *uniform* quant for values** (QJL dropped on the value stream) — a *variant*, not the paper. A faithful **PyTorch** port, **[tonbistudio/turboquant-pytorch](https://github.com/tonbistudio/turboquant-pytorch)**, even reports that **QJL *hurt* in their tests** (removing it raised attention-score cosine), that 3-bit needs an FP16 "residual window" to stay coherent, and that **high attention-score similarity ≠ working generation** — all directly load-bearing for the T2 TurboQuant-specificity claim and worth characterizing in T1. So treat *scos-lab faithful* as the object of study, cite vLLM's variant as "deployed reality," and make **"is QJL actually helping?"** a first-class T1 question. **Strategic flag for T2:** a *second* serious implementation — [TheTom/turboquant_plus](https://github.com/TheTom/turboquant_plus) — also drops QJL in its deployed path (turbo4) and reports **rotation is first-order, scalar-quant/QJL second-order**. Since T2's headline targets the QJL 1-bit-residual error geometry, weigh making the **rotation** geometry the primary target instead — it is present in faithful TQ, the WHT variants, *and* every deployment, with the public-vs-secret Π axis as the lever. Keep QJL in scope on the faithful layer, but don't rest the whole thesis on a component real systems omit.

---

## 3. Compute strategy — HPC vs vLLM vs inference engine

**These are different layers, not alternatives — and only two of them are mandatory.**

- **HPC (Leonardo/CINECA) = where everything runs (mandatory).** You need A100-class GPUs regardless — for 8B inference, for fine-tuning the T2 backdoor, and for benchmark sweeps. Neither vLLM nor any inference engine substitutes for HPC; they run *on* it. It's free, and the old repo's Slurm + env scaffolding already targets it (salvage it).
- **Plain HuggingFace `transformers` = the mandatory engine for the whole project.** Instrumentation, T2, T3, and T1's *quality* characterization all run here. You need correctness + introspection + gradients + small batch, not throughput. This is the foundation.
- **vLLM = deferred, optional, T1-only oracle.** It buys exactly one thing: real **measured** latency/throughput and clean **measured** GPU memory. It cannot do T2/T3 (fused kernels expose no gradients and no per-token dumps). Leave it out until — and unless — T1 needs a measured-efficiency claim.

**Decision:** **Build everything on plain HF transformers on HPC. Do not stand up vLLM yet.** Revisit it only at the end of T1, and only if you need *measured* latency; at that point it's a clean, isolated add (serve the same model, run the benchmark, record numbers) with zero impact on the rest of the codebase.

**Memory, without vLLM:** report it honestly as a **counted/theoretical** figure — sum the bytes you actually store (codes + scales + QJL bits) vs FP16. That's a real number, but it is *computed*, not *measured under a serving allocator*; only vLLM gives the latter. Always state which one a number is.

**No speedup claim from the research layer:** scos-lab is pure Python/NumPy and runs *slower* than FP16. Latency/throughput claims require vLLM (and, to match the paper's ~8× logit speedup, an **H100** — Leonardo is A100, so memory reproduces but the speedup may not).

**HPC gotchas (already solved in the old repo — reuse):** compute nodes have no internet → pre-download models/datasets to `$SCRATCH` on the login node and run with `HF_HUB_OFFLINE=1`; use the module system + a venv on `$SCRATCH`.

---

## 4. Target repo structure (`turboquant-exp/`)

```
turboquant-exp/
  PROJECT_PLAN.md            # this file
  README.md
  env/                       # setup_leonardo_env.sh, load_env.sh, .env.example   (port from old repo)
  third_party/
    turboquant/              # vendored/forked scos-lab/turboquant (research layer)
  tqsec/                     # our package — the thin layer we own
    config.py                # env-driven config                                  (port + extend)
    instrument.py            # dump Π, codes, QJL residual, per-token/per-channel error map
    quantizers.py            # adapters: TurboQuant | plain INT3/4 | KIVI | FP8 (control harness); also the `-nc` boundary-layer policy scos-lab lacks
    diff_twin.py             # PyTorch differentiable twin of TQ (STE) for T2 — scos-lab is NumPy; audit tonbistudio/turboquant-pytorch as starting point
    pi_regime.py             # public/reused vs secret/per-deployment rotation switch (maps onto scos-lab's `seed`)
    metrics.py               # token + semantic recovery, JS divergence, distortion (port + extend)
    benchmarks.py            # needle-in-haystack / LongBench slice loaders
  t1_characterization/       # Student 1: HF quality + counted-memory (vLLM only if measured latency needed)
  t2_behavior/               # Student 2: error map, backdoor (Version A), prompt-exploit (Version B, stretch)
  t3_leakage/                # Student 3: FP-KV inversion baseline, learned inverter on codes
  slurm/                     # batch jobs                                          (port from old repo)
  scripts/                   # download_model.py, submit_*.sh, compare_*.py        (port from old repo)
  results/                   # <track>/<model_tag>/*.json
  reports/                   # per-track summaries + cross-cutting synthesis
  docs/                      # ARCHITECTURE.md, RUNBOOK.md, VALIDATION.md, ETHICS.md
```

### Salvage list (copy from `../turboquant-kv-cache-experiments`)
- `scripts/setup_leonardo_env.sh`, `scripts/load_env.sh`, `.env.example`, `scripts/download_model.py`, `scripts/check_hf_access.py` — environment + offline model fetch.
- `scripts/submit_model_suite.sh`, `slurm/*.slurm` — Slurm job + dependency chaining.
- `turboquant_kv/config.py` — env-driven `ExperimentConfig` pattern.
- `turboquant_kv/metrics.py`, `scripts/compare_models.py`, `scripts/summarize_results.sh`, `reports/` layout — metrics + reporting pipeline.
- `experiments/prompt_leakage.py` — **skeleton only** for T3 (upgrade nearest-neighbor proxy → learned inverter; keep the cross-layer/per-bit loop structure).
- **Do NOT port** `turboquant_kv/cache.py` — wrong algorithm. Replaced by the scos-lab fork.
- **Relabel** `experiments/behavior_divergence.py` if reused: it is degradation characterization (T1), not T2.

---

## 5. Phased plan — what to do and when

> Timeline is a 12-week skeleton starting now (late June 2026). Week numbers are relative; compress/stretch to your deadline. Phase 0–1 are the critical path everyone shares; T2 and T3 run in parallel after the foundation is green.

### Phase 0 — Foundations & go/no-go on reuse (Week 1)
**Goal:** prove the reused pieces work before committing.
- [ ] Scaffold `turboquant-exp/` (section 4); port the salvage list; set up Leonardo venv + offline model cache (`$SCRATCH/models/`).
- [x] **Audit the scos-lab fork against the paper** (this is the gate) — *code-read + tests done 2026-06-28: PASSES.* Real Lloyd–Max grids (on the analytic post-rotation Beta density), true `sign(Sx)` QJL with a passing inner-product-unbiasedness test (`tests/test_qjl.py`), K+V + asymmetric-bit + `paper`/`mse`/`mixed` modes present. **Test-execution done 2026-06-28: 49/49 pass** (incl. the QJL-unbiasedness gate and Lloyd–Max distortion-vs-paper at b=1–4). **Gap found: `-nc` (uncompressed boundary-layer) handling is NOT in scos-lab — `tqsec/quantizers.py` adds it.** Note: the 49 tests cover the *algorithm*; the HF `DynamicLayer` integration has no test yet — exercise it during the port. Then port its `Cache` integration from Qwen2.5-7B (its tested ceiling) to **Llama-3.1-8B-Instruct** and **Mistral-7B-Instruct**.
- [x] Smoke-test model: **TinyLlama-1.1B** for fast iteration before every 8B run.
- [ ] *(Deferred — not now.)* vLLM stand-up is **not** part of the foundation. It enters only in T1 *if* a measured-latency claim is needed (see Phase 2 · T1).
- **Gate / deliverable:** `docs/VALIDATION.md` states "research layer faithful: yes/no, evidence." *Done: verified faithful; 49/49 tests pass (executed 2026-06-28).* If scos-lab had failed, the fallback order is **tonbistudio/turboquant-pytorch** (faithful PyTorch — also the T2-twin seed), then OmarHory/turboquant or AmesianX/TurboQuant (llama.cpp/C++), before any reimplementation.

### Phase 1 — Shared instrumented foundation + T1 quality baseline (Weeks 2–4)
**Owner:** Student 1 leads; Students 2 & 3 build their FP-KV baselines in this window.
- [x] **`tqsec/instrument.py`** — wrap the research layer to dump, per layer/head/token/channel: rotation Π, scalar codes, QJL residual, and **reconstruction error = true KV − reconstructed KV**. Produce the **error map** (is error biased per channel? does QJL zero the mean? is error concentrated on high-magnitude tokens/channels?). *This map is the raw material for T2 and T3 — nothing downstream starts without it.*
- [x] **`tqsec/quantizers.py` control harness** — same interface for **{TurboQuant, plain INT3/INT4, KIVI, FP8-KV}**. Every later experiment runs against all four. (TurboQuant-specificity is unprovable without this; FP8 is the "is TurboQuant even worth it" baseline.)
- [x] **`tqsec/pi_regime.py`** — switch between **public/reused Π** and **secret/per-deployment Π**. The single biggest security variable; both attacks get reported under both regimes.
- [x] **Sanity benchmark** (`tqsec/benchmarks.py`) — needle-in-a-haystack or a LongBench slice; confirm research layer is quality-neutral ~3.5 bits, degrades ~2.5 bits, matching the paper.
- **Deliverable (done):** error map, control harness, Π switch, metrics, and benchmarks are built with passing smoke tests, and the sanity benchmark runs on real models. This is the project's backbone.

### Phase 2 — Parallel tracks (Weeks 4–10)

#### T1 — Characterization (Student 1, Weeks 4–7)
**Required (HF-only):**
- [x] Across bit-widths and quantizers ({TurboQuant, INT, KIVI, FP8}, plus the `-nc` variant): quality (needle-in-a-haystack) and counted memory measured on four models. LongBench and perplexity are still to add.
- [x] Document the `-nc` policy (which boundary layers are uncompressed) and hold it constant across models.

**Optional (only if a *measured* efficiency claim is required) — bolt on vLLM:**
- [ ] Stand up vLLM on Leonardo; serve the same models with `--kv-cache-dtype turboquant_*`; record **measured** memory + latency/throughput; cross-check vLLM distortion vs the research layer.
- **Deliverable (done):** `reports/T1_characterization.md` is written, with quality (needle, four models) and counted-memory tables, the Qwen outlier mechanism, and a figure. The main result: a model's KV outlier ratio predicts whether uniform TurboQuant works. Qwen's ~100x boundary-layer outliers break it at every bit-width, where KIVI or `-nc` are needed. Optional depth not yet done: LongBench, perplexity, the QJL ablation, and vLLM measured efficiency.

#### T2 — Compression-activated adversarial behavior (Student 2, Weeks 4–10)
**Target is a BENIGN CANARY, not a harmful payload** — a specific, unmistakable marker behavior (e.g., trigger → a fixed odd/wrong answer). Same existence proof, safer, easier to measure. The bar is *FP-absent / TurboQuant-present / specifically-triggered* — **not** mere output drift or degradation.
- [ ] **Baseline first (Weeks 4–5):** reproduce a *standard* backdoor with a normal trigger under FP-KV (positions you against prior quantization-conditioned backdoors in weights — cite them).
- [ ] **Read the error map (Weeks 5–6):** find tokens/channels/layers where TurboQuant's reconstruction error is largest and most *steerable*. This defines the attack surface.
- [x] **Version A — trained backdoor (Weeks 6–9, REQUIRED, primary):** fine-tune with the quantizer **in the loop**, dual objective — *normal under FP-KV, benign-canary fires under TurboQuant-KV*. Wall: non-differentiable quant → **straight-through estimator** / differentiable relaxation of rotation + Lloyd–Max + QJL. **Build this as a PyTorch differentiable twin — scos-lab is NumPy and can't backprop; freeze scos-lab's Lloyd–Max centroids + QJL matrix as the reference and validate the twin's forward pass against them. Starting point: audit + adapt [tonbistudio/turboquant-pytorch](https://github.com/tonbistudio/turboquant-pytorch) (already `nn.Module`-based with a QJL on/off switch). vLLM cannot do this at all.** Trigger = "the cache is quantized." Success = existence proof; difficulty = also a result. *Done: LoRA (r=16) backdoor trained on the STE differentiable twin (`tqsec/diff_twin.py`, forward validated against the NumPy reference), evaluated by real generation on the faithful quantizer. Canary fires only under compressed-KV, FP stays == base; replicates on TinyLlama-1.1B and Mistral-7B. See `reports/T2_behavior.md`.*
- [x] **Ablation (mandatory):** does the canary fire under plain INT2/INT3 and KIVI too? If yes, the TurboQuant-specific (rotation+QJL geometry) claim is unsupported. Report under public-Π and secret-Π. *Done: the public-Π attacker is TurboQuant-specific (specificity 0.9–1.0) but defeated by secret Π (fire → 0.00–0.07); the Π-robust attacker survives secret Π (0.99–1.00 on unseen rotations) but is generic (INT3 fires it, specificity 0.0–0.2). Mild 8-bit keys defeat both (0.00).*
- [ ] **Version B — prompt-only error-geometry exploit (Weeks 9–10, OPTIONAL / STRETCH):** fixed clean model; an adversarial *prompt* drives internal states into a region where TurboQuant reconstruction error nudges attention to the canary completion — safe under FP-KV, triggered under TurboQuant. **Attempt only after Version A lands and the error map exists; skip if time-constrained.**
- **Deliverable (done):** `reports/T2_behavior.md` — operational definition of the canary behavior (a *specific target*, explicitly **not** output drift), FP-vs-TurboQuant contrast on the *same* behavior, the INT/KIVI ablation, Π-regime results. Cross-model table auto-rendered to `reports/T2_backdoor_autogen.md`.

#### T3 — Prompt leakage from TurboQuant codes (Student 3, Weeks 4–10)
**Required question is binary: does TurboQuant MITIGATE or WORSEN inversion?**
- [ ] **Baseline first (Weeks 4–6):** reproduce an *existing* KV/embedding inversion attack on **uncompressed** cache → FP-KV leakage baseline. (The old repo's nearest-neighbor proxy is a starting skeleton, not the attack.)
- [ ] **Learned inverter (Weeks 6–9):** train an inverter on the **TurboQuant codes**. Measure **token-level and semantic recovery separately**, per bit-width, as a delta from the FP-KV baseline.
- [ ] **Π regime (Weeks 9–10):** rerun under public-Π vs secret-Π — does secret Π close the leak? Ties to "data-oblivious + randomized is harder to attack."
- [ ] **Ablation:** compare leakage under TurboQuant vs plain INT vs FP8 codes.
- [ ] **(OPTIONAL / STRETCH) Sharp hypothesis:** test whether TurboQuant *preserves the semantically leak-relevant signal while destroying the rest* — i.e., worsens prompt-level inversion even as vector-level distortion improves. The publishable twist; pursue only once the binary result is in hand.
- **Deliverable:** `reports/T3_leakage.md` — token vs semantic recovery curves per bit-width, FP-baseline delta, Π public-vs-secret, mitigate-vs-worsen verdict with evidence.

### Phase 3 — Synthesis, mitigations, write-up (Weeks 11–12)
- [ ] Cross-cutting analysis: **how does Π-secrecy move attack success across T2 and T3?** (a contribution on its own).
- [ ] Pair each landed attack with a **candidate mitigation** (secret-Π? a served-KV verification check?). "Attack + mitigation" is a markedly stronger paper.
- [ ] Responsible disclosure plan to TurboQuant/vLLM maintainers if Version B (or a real leak) lands.
- [ ] Final report + reproducibility appendix (`docs/VALIDATION.md`, seeds, exact variants, `-nc` policy).

---

## 6. Non-negotiable controls (apply to T2 and T3 every time)
1. **TurboQuant-specificity ablation:** every attack also run against plain INT3/INT4 and KIVI. If it works equally under coarse rounding, the novelty claim is dead.
2. **FP-KV baseline before compression** — establish the uncompressed behavior/leakage first; report deltas.
3. **Token-level vs semantic metrics kept separate** (T3).
4. **Both Π regimes reported** (public/reused vs secret/per-deployment).
5. **`-nc` policy fixed and documented** across models.

---

## 7. Models
- **Smoke / iteration:** TinyLlama-1.1B-Chat.
- **Faithful-layer first target:** Qwen2.5-7B-Instruct (scos-lab's tested ceiling — lowest porting risk).
- **Primary:** Llama-3.1-8B-Instruct + Mistral-7B-Instruct (the design's recommended targets).

---

## 8. Risks & mitigations
| Risk | Mitigation |
|---|---|
| Reused repo is subtly unfaithful | Phase-0 audit gate (Lloyd–Max + QJL-unbiasedness test) before any dependence |
| scos-lab too slow for 8B at scale | It's for introspection + small-batch attacks, not throughput; never put it on a throughput-sensitive path. Measured efficiency, if ever needed, comes from vLLM only |
| Tempted to add vLLM early | Don't — it's optional and T1-only. Default to counted memory + quality on HF; bolt on vLLM at the *end* of T1 if a measured claim is required |
| vLLM speedup weak on A100 | Claim measured memory + quality on A100; flag speedup as H100-only (or rent one H100) |
| T2 backdoor non-differentiable | STE on a **PyTorch differentiable twin** (scos-lab is NumPy — no autograd); seed the twin from tonbistudio/turboquant-pytorch, validate vs scos-lab centroids. "It's hard" is itself a reportable result |
| Effect not TurboQuant-specific | The INT/KIVI ablation tells you early and cheaply — run it first, not last |
| Mislabeling degradation as the result | Benign-canary target with an operational definition up front (a defined trigger, FP-absent / compressed-present); reject the weak "undesired" reading |

---

## 9. Ethics
In-lab, open models only. **Use a benign canary target, never a harmful payload** — it proves the same existence claim with no real-world misuse value. Each attack paired with a candidate mitigation. Plan responsible disclosure to TurboQuant/vLLM maintainers before any public release of a working exploit. See `docs/ETHICS.md`.

---

## 10. References
- Paper: TurboQuant — *Online Vector Quantization with Near-optimal Distortion Rate* (Zandieh, Daliri, Hadian, Mirrokni). arXiv:2504.19874; ICLR 2026; OpenReview `tO3ASKZlok`.
- Research layer: github.com/scos-lab/turboquant (faithful, HF, instrumentable, MIT; **audit-passed by code-read 2026-06-28**; NumPy not PyTorch). Faithful **PyTorch** alternate / T2-twin starting point: github.com/tonbistudio/turboquant-pytorch. Other alternates: OmarHory/turboquant, AmesianX/TurboQuant (llama.cpp/C++).
- Production oracle (T1-only, deferred): upstream vLLM `--kv-cache-dtype turboquant_*` (PR vllm-project/vllm#38479, **merged 2026-04-15**) — a **WHT-rotation + uniform-value variant**, "deployed reality" not the faithful object of study.
- Deployed-reality reference + findings source (Apache-2.0): github.com/TheTom/turboquant_plus. Its Python `turboquant/` reference is faithful (PolarQuant + QJL, NumPy) but its headline/deployed path is the WHT + codebook variant with QJL dropped (turbo4). **Not the foundation** (no gain over the already-audited scos-lab; large, tangled with non-paper extensions). **Use for:** T1 measured efficiency via its `refract` harness (llama.cpp/mlx/vllm), and validated findings — *V-compression-is-free / all degradation from K*, *boundary layers sensitive (protect first 2 + last 2 → validates our `-nc`)*, block_size=128, asymmetric K/V. Same role as vLLM.
- Security framing prior art: *Exploiting LLM Quantization* (NeurIPS 2024), *Mind the Gap* (ICML 2025), *Watch Your Steps* (ICLR 2026), *RobustKV* (ICLR 2025).
- Predecessor repo (salvage source): `../turboquant-kv-cache-experiments`.
