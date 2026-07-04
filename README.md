# TurboQuant KV-Cache Security

Studying the security implications of **TurboQuant** (ICLR 2026 — online KV-cache vector
quantization: random rotation → Lloyd–Max scalar quant → 1-bit QJL residual). Greenfield restart
of `../turboquant-kv-cache-experiments`, built around a **faithful, instrumented** TurboQuant
rather than a TurboQuant-*style* approximation.

Full design and rationale: **[PROJECT_PLAN.md](PROJECT_PLAN.md)**.

## The three tracks
- **T1 — Characterization** — quality vs bit-width + counted memory; the legitimate baseline.
- **T2 — Compression-activated backdoor** — a model normal under FP-KV that fires a **benign
  canary** *only when TurboQuant is enabled*, exploiting the rotation + 1-bit-residual error
  geometry (not coarse rounding).
- **T3 — Prompt leakage** — does TurboQuant **mitigate** or **worsen** KV-cache inversion?

Both attacks are reported under **public/reused Π vs secret/per-deployment Π**, and every attack
runs the **{TurboQuant, INT3/4, KIVI, FP8}** specificity ablation.

## Two-layer stack
- **Research layer (mandatory, all tracks):** `third_party/turboquant/` — vendored scos-lab/turboquant,
  run through plain HuggingFace `transformers`. Faithful, introspectable, NumPy. **Audit-passed**
  (`docs/VALIDATION.md`).
- **Production oracle (optional, T1-only, deferred):** upstream vLLM `--kv-cache-dtype turboquant_*`
  — real *measured* latency/memory, but a WHT/uniform variant. Add only if T1 needs a measured
  efficiency claim.

## Layout
```
PROJECT_PLAN.md          # design, decisions, phased plan (read this first)
third_party/turboquant/  # vendored faithful TurboQuant (research layer) — read-only
tqsec/                   # the thin layer we own (instrument, controls, Π-regime, twin, metrics)
t1_characterization/     # Student 1
t2_behavior/             # Student 2
t3_leakage/              # Student 3
env/ scripts/ slurm/     # Leonardo setup, utilities, batch jobs
results/ reports/        # outputs and write-ups (T1_characterization.md)
docs/                    # VALIDATION.md, RUNBOOK.md; ARCHITECTURE and ETHICS still to write
```

## Status
The shared foundation is built and T1 is done. The reuse audit passed (`docs/VALIDATION.md`), the
`tqsec` package (instrumentation, the control harness, the Π-regime switch, metrics, benchmarks) has
passing smoke tests, and the Leonardo setup runs (`docs/RUNBOOK.md`).

T1 characterization has been run on four models: TinyLlama-1.1B, Mistral-7B, Llama-3.1-8B, and
Qwen2.5-7B. The writeup is [reports/T1_characterization.md](reports/T1_characterization.md). The main
result is that a model's KV outlier ratio predicts whether uniform TurboQuant works. It stays
quality-neutral at 8-bit on well-behaved models, but on Qwen, which has roughly 100x boundary-layer
outlier channels, it fails at every bit-width; per-channel KIVI or the `-nc` boundary-protected variant
are needed there instead.

Next: T2 (compression-activated backdoor) and T3 (prompt leakage). See `PROJECT_PLAN.md`.

## Quickstart (smoke)
```
python -m pytest third_party/turboquant/tests/ -v   # 49 passing — faithfulness gate
python scripts/audit_evidence.py                    # distortion + QJL-bias evidence
```
Requires `numpy scipy` (algorithm) and `torch transformers` (HF integration).
