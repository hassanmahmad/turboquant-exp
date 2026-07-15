# TurboQuant KV-Cache Security

Studying the security implications of TurboQuant (ICLR 2026: online KV-cache vector quantization
via random rotation, a Lloyd-Max scalar quantizer and a 1-bit QJL residual). Built around a faithful,
instrumented TurboQuant rather than a TurboQuant-style approximation.

## The three tracks
- **T1, Characterization:** quality vs bit-width plus counted memory, the legitimate baseline.
- **T2, Compression-activated backdoor:** a model that behaves normally under FP-KV but emits a benign
  canary only when the cache is compressed, exploiting the quantization error geometry rather than
  coarse rounding.
- **T3, Prompt leakage:** does compression mitigate or worsen KV-cache inversion?

Both attacks are reported under public/reused Π vs secret/per-deployment Π, and every attack runs the
{TurboQuant, INT3/4, KIVI, FP8} specificity ablation.

## Two-layer stack
- **Research layer (mandatory, all tracks):** `third_party/turboquant/`, vendored scos-lab/turboquant
  run through plain HuggingFace `transformers`. Faithful, introspectable, NumPy. Audit-passed
  (`docs/VALIDATION.md`).
- **Production oracle (optional, T1-only, deferred):** upstream vLLM `--kv-cache-dtype turboquant_*`
  gives measured latency and memory but is a WHT/uniform variant. Add only if T1 needs a measured
  efficiency claim.

## Layout
```
third_party/turboquant/  # vendored faithful TurboQuant (research layer), read-only
tqsec/                   # instrumentation, controls, Π-regime, twin, metrics
t1_characterization/     # T1
t2_behavior/             # T2
t3_leakage/              # T3
env/ scripts/ slurm/     # Leonardo setup, utilities, batch jobs
results/ reports/        # outputs and write-ups
docs/                    # validation, architecture, runbook, ethics
```

## Status
All three tracks have final Leonardo runs. `tqsec/` is built (`instrument`, `quantizers`
{TurboQuant,INT,KIVI,FP8}+`-nc`, `pi_regime`, `diff_twin`, `inversion`, `metrics`, `benchmarks`) with
smoke tests, and the Leonardo scaffolding is ready (`env/`, `slurm/`, `scripts/`).
- **T1:** quality/perplexity plus counted KV-cache memory reported (`reports/T1_characterization.md`).
- **T2:** a weight-level compression-activated backdoor works and replicates on TinyLlama and Mistral-7B.
  Secret per-deployment Π stops the TurboQuant-specific variant but not a Π-robust attacker, which stays
  generic (INT3 activates it too). See `reports/T2_behavior.md`.
- **T3:** no robust TurboQuant-specific leakage worsening; effects are model-dependent and often matched
  or exceeded by INT/KIVI/FP8 controls (`reports/T3_leakage.md`).

Cross-track synthesis and mitigations: `reports/SYNTHESIS.md`.

## Quickstart (smoke)
```
python -m pytest third_party/turboquant/tests/ -v   # 49 passing, faithfulness gate
python scripts/audit_evidence.py                    # distortion + QJL-bias evidence
python scripts/diff_twin_smoke.py                   # PyTorch twin reference + gradient check
```
Requires `numpy scipy` (algorithm) and `torch transformers` (HF integration).