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
env/ scripts/ slurm/     # Leonardo setup, utilities, batch jobs (port from predecessor)
results/ reports/        # outputs and write-ups
docs/                    # VALIDATION.md (done); ARCHITECTURE / RUNBOOK / ETHICS (Phase-0 TODO)
```

## Status
**Phase 1 — shared foundation: code-complete.** Reuse gate passed (`docs/VALIDATION.md`); `tqsec/`
built (`instrument` error map · `quantizers` {TurboQuant,INT,KIVI,FP8}+`-nc` · `pi_regime` · `metrics`
· `benchmarks`) with 5 passing smoke tests; Leonardo scaffolding ready (`env/`, `slurm/`, `scripts/`,
`t1_characterization/run_sanity.py`). **Next:** run the sanity benchmark on a real 7–8B model on
Leonardo (**[docs/RUNBOOK.md](docs/RUNBOOK.md)**), then the T1/T2/T3 research. See `PROJECT_PLAN.md` §5.

## Quickstart (smoke)
```
python -m pytest third_party/turboquant/tests/ -v   # 49 passing — faithfulness gate
python scripts/audit_evidence.py                    # distortion + QJL-bias evidence
```
Requires `numpy scipy` (algorithm) and `torch transformers` (HF integration).
