# VALIDATION — Research-layer faithfulness (Phase-0 reuse gate)

**Verdict: the vendored research layer (`third_party/turboquant/`, scos-lab/turboquant) is a FAITHFUL TurboQuant implementation. The Phase-0 reuse gate is PASSED.**

This is the gate the predecessor repo (`../turboquant-kv-cache-experiments`) failed: its cache used uniform-affine (min/max) quantization plus a mean-magnitude sign residual — neither Lloyd–Max nor QJL. Every downstream claim (T1 quality, T2 backdoor, T3 leakage) depends on the quantizer under study actually *being* TurboQuant, so this gate is non-negotiable. Acceptance criteria: `PROJECT_PLAN.md` §2.3.

## What was audited
- **Object:** scos-lab/turboquant, vendored at commit `34a10b6` (2026-03-28), MIT. Provenance: `third_party/turboquant/VENDORED.md`.
- **How:** (1) line-by-line read of the quantizer, QJL, rotation, core composition, and HF integration; (2) execution of the upstream test suite plus independent measurement of the two acceptance quantities (`scripts/audit_evidence.py`).
- **Environment:** Python 3.11.4, numpy 2.2.0, scipy 1.15.1, torch 2.9.1+cu126, transformers 4.57.3 (local dev box; the algorithm is hardware-independent).
- **Date:** 2026-06-28.

## Gate (a) — Lloyd–Max optimal grids, not min/max affine
**Requirement:** per-coordinate scalar quantization on the post-rotation distribution must use optimal (Lloyd–Max) levels, not a uniform/affine min–max grid.

**Finding — PASS.** `scalar_quantizer.py::compute_centroids` runs genuine Lloyd iteration (boundaries = midpoints of adjacent centroids; centroids = conditional means `E[X | bin]`) on the *analytic* post-rotation coordinate density `Beta((d−1)/2, (d−1)/2)` on `[−1, 1]`. The grid is derived from the theoretical distribution, not fitted to data — which is exactly what makes TurboQuant calibration-free / data-oblivious.

**Evidence — distortion matches the paper.** Total MSE distortion of unit vectors (theory = `d·𝒞(f_X,b)`; empirical = round-trip MSE):

| bits b | theory (d=512) | empirical (d=256) | paper |
|---|---|---|---|
| 1 | 0.363 | 0.362 | 0.36 |
| 2 | 0.117 | 0.117 | 0.117 |
| 3 | 0.034 | 0.034 | 0.03 |
| 4 | 0.0095 | 0.0094 | 0.009 |

Matches to ~3 significant figures (b=3 is 0.034 vs the paper's 0.03, within the suite's tolerance).

## Gate (b) — real QJL residual with unbiased inner products
**Requirement:** a true 1-bit QJL residual = random JL projection → sign, making the inner-product (attention-score) estimate unbiased.

**Finding — PASS.** `qjl.py` implements `Q(x) = sign(S·x)` with a Gaussian JL matrix `S`, plus the unbiased estimator `⟨y, x̂⟩` with the `√(π/2)/d` correction. `core.py::TurboQuantProd` allocates `(b−1)` bits to the MSE quantizer + 1 bit to QJL on the residual — exactly the paper's key-path construction; `inner_product()` returns the MSE + QJL components.

**Evidence — bias → 0.** Inner-product estimation error over 3000 random unit-vector pairs (d=128):

| bits b | bias | variance |
|---|---|---|
| 2 | +1.1e-4 | 4.6e-3 |
| 3 | −0.8e-4 | 1.5e-3 |
| 4 | −3.0e-4 | 0.5e-3 |

Bias is ≈1e-4 (statistically zero); variance shrinks with bits, and the suite separately asserts variance ∝ 1/d.

## Test suite
```
python -m pytest third_party/turboquant/tests/ -v
```
→ **49 passed** (~72 s). Coverage: rotation (orthogonality, norm preservation, Beta coordinate distribution), scalar quantizer (centroid count/symmetry/ordering, distortion-vs-paper), QJL (sign output, determinism, unbiasedness, variance ∝ 1/d), core (MSE & Prod round-trips, distortion, unbiased inner product).

Reproduce the distortion + bias numbers above:
```
python scripts/audit_evidence.py
```

## Scope & known limits (carry into later phases)
1. **Algorithm tests only — HF integration now smoke-tested separately.** The 49 tests don't touch `kv_cache.py`; `scripts/instrument_smoke.py` (via `tqsec.instrument`) now exercises the `DynamicLayer` update path on CPU + GPU. **Finding: the vendored `kv_cache.py` is CPU-only** — it does `states.numpy()` and returns recon without restoring the device, so it crashes for GPU models. `tqsec.instrument.InstrumentedTurboQuantLayer` handles this; the *non-instrumented* path (e.g. T1 quality runs) needs the same `.cpu()`/`.to(device)` shim on Leonardo GPUs. **Update:** the full `model.generate` + quant-cache path is now validated on a real 30-layer model (`scripts/benchmarks_smoke.py`) — FP-KV, TurboQuant, and INT all run through `generate`.
2. **NumPy, not PyTorch.** The HF path does a `.numpy()` round-trip → no autograd. T2's in-the-loop training needs a PyTorch differentiable twin (`tqsec/diff_twin.py`); freeze these centroids + the QJL matrix as the reference. (This corrects an earlier plan assumption that the research layer was PyTorch.)
3. **No `-nc` policy.** scos-lab compresses every layer; the uncompressed-boundary-layer variants are added in `tqsec/quantizers.py`.
4. **Counted, not measured, memory.** `kv_cache.py` counts compressed bits; a *measured* figure needs vLLM (deferred, T1-only).
5. **Fidelity spectrum.** Upstream vLLM (#38479) is a WHT-rotation / uniform-value variant — "deployed reality," not this faithful object of study.
6. **Env gotcha (transformers 4.57.3):** loading a tokenizer by repo id can 404 on a missing `additional_chat_templates` folder. Workaround (and Leonardo's default anyway): `snapshot_download` the model, then load locally with `HF_HUB_OFFLINE=1`.

## Fallback (not needed)
scos-lab passed, so no fallback was used. Had it failed, the order was: tonbistudio/turboquant-pytorch (faithful PyTorch) → OmarHory/turboquant → AmesianX/TurboQuant.
