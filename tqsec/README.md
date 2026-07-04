# tqsec — the layer we own

Thin instrumentation + control layer over the vendored faithful TurboQuant
(`third_party/turboquant/`). **Nothing here reimplements the algorithm.** See
`PROJECT_PLAN.md` §4 for where this sits.

## Modules

| Module | Status | Contract |
|---|---|---|
| `config.py` | **done** | Env-driven `ExperimentConfig`: model, quantizer (`turboquant\|int\|kivi\|fp8\|fp16`), `mode`, K/V bits, `seed`, `pi_regime`, `nc_layers`. |
| `instrument.py` | **done** | Error map via an instrumented HF cache: per layer/token/channel reconstruction error + Π / codes / QJL dumps. Findings per layer = channel-bias ratio, channel/token concentration, rel-err. Also restores device/dtype (fixes the CPU-only vendored layer). Smoke: `scripts/instrument_smoke.py`. *Prerequisite for T2 and T3.* |
| `quantizers.py` | **done** | One interface over `{TurboQuant, INT, KIVI, FP8}` + the `-nc` policy + bit accounting; device-safe; records to an error map for cross-codec comparison. Smoke: `scripts/quantizers_smoke.py`. Finding: TurboQuant's error geometry (channel-decorrelated, inner-product-optimal) is already distinguishable from INT/KIVI/FP8 — see [[turboquant-error-geometry]]. |
| `pi_regime.py` | **done** | public/reused vs secret/per-deployment Π via seed management; `attacker_seed()` encodes what the attacker may know; per-layer Π supported. Smoke `scripts/pi_regime_smoke.py` shows inversion top-1 = 1.00 under public Π vs ~chance under secret Π (prototypes the T3 Π result). |
| `diff_twin.py` | **done** | PyTorch twin of rotation + Lloyd-Max + QJL for T2 STE training. Forward pass validates against the NumPy reference; scalar bins and QJL signs use straight-through gradients. Smoke: `scripts/diff_twin_smoke.py`. |
| `metrics.py` | **done** | Token vs semantic recovery (kept separate), inner-product/attention fidelity + distortion, JS/KL divergence, T2 `canary_fires`. Smoke: `scripts/metrics_smoke.py`. |
| `benchmarks.py` | **done** | NIAH construction/scoring + LongBench loader + `sanity_sweep` (FP-KV + all quantizers = T1 quality table + specificity ablation). Smoke `scripts/benchmarks_smoke.py` validates `model.generate` + quant cache end-to-end on a real model (all 30 layers). |

## Conventions
- Import the faithful layer as `from turboquant... import ...` (it's vendored under `third_party/`,
  which must be on `sys.path` — see `scripts/audit_evidence.py` for the pattern).
- Every later experiment runs against **all** quantizers in the harness and **both** Π regimes.
