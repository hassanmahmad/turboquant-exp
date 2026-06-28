# tqsec — the layer we own

Thin instrumentation + control layer over the vendored faithful TurboQuant
(`third_party/turboquant/`). **Nothing here reimplements the algorithm.** See
`PROJECT_PLAN.md` §4 for where this sits.

## Modules

| Module | Status | Contract |
|---|---|---|
| `config.py` | **done** | Env-driven `ExperimentConfig`: model, quantizer (`turboquant\|int\|kivi\|fp8\|fp16`), `mode`, K/V bits, `seed`, `pi_regime`, `nc_layers`. |
| `instrument.py` | TODO | Wrap the research layer to dump, per layer/head/token/channel: rotation Π, scalar codes, QJL residual, and **reconstruction error = true KV − reconstructed KV**. Produce the **error map** (per-channel bias? does QJL zero the mean? is error concentrated on high-magnitude tokens?). *Prerequisite for T2 and T3.* |
| `quantizers.py` | TODO | One interface over `{TurboQuant, plain INT3/INT4, KIVI, FP8-KV}` for the specificity ablation. **Also adds the `-nc` policy** (skip `cfg.nc_layers`) the research layer lacks. |
| `pi_regime.py` | TODO | Switch between **public/reused Π** (one `seed`) and **secret/per-deployment Π** (fresh per deployment). Maps onto scos-lab's `seed` arg. |
| `diff_twin.py` | TODO | **PyTorch differentiable twin** of rotation + Lloyd–Max + QJL for T2's STE training — the research layer is NumPy (no autograd). Freeze scos-lab's centroids + QJL matrix as the reference and validate against them; seed from tonbistudio/turboquant-pytorch (audit first). |
| `metrics.py` | TODO | Token-level **and** semantic recovery (kept separate), JS divergence, distortion. Port + extend the predecessor's `turboquant_kv/metrics.py`. |
| `benchmarks.py` | TODO | Needle-in-a-haystack / LongBench slice loaders for the sanity benchmark. |

## Conventions
- Import the faithful layer as `from turboquant... import ...` (it's vendored under `third_party/`,
  which must be on `sys.path` — see `scripts/audit_evidence.py` for the pattern).
- Every later experiment runs against **all** quantizers in the harness and **both** Π regimes.
