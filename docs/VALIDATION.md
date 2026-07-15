# VALIDATION — Research-layer faithfulness (Phase-0 reuse gate)

**Verdict: the vendored research layer (`third_party/turboquant/`, scos-lab/turboquant) is a faithful TurboQuant implementation. The Phase-0 reuse gate is passed.**

This gate rules out a cache that only approximates TurboQuant: uniform-affine (min/max) quantization plus a mean-magnitude sign residual is neither Lloyd–Max nor QJL. Every downstream claim (T1 quality, T2 backdoor, T3 leakage) depends on the quantizer under study actually *being* TurboQuant.

## What was audited
- **Object:** scos-lab/turboquant, vendored at commit `34a10b6` (2026-03-28), MIT. Provenance: `third_party/turboquant/VENDORED.md`.
- **How:** (1) line-by-line read of the quantizer, QJL, rotation, core composition and HF integration; (2) execution of the upstream test suite plus independent measurement of the two acceptance quantities (`scripts/audit_evidence.py`).
- **Environment:** Python 3.11.4, numpy 2.2.0, scipy 1.15.1, torch 2.9.1+cu126, transformers 4.57.3 (local dev box; the algorithm is hardware-independent).
- **Date:** 2026-06-28.

## Gate (a) — Lloyd–Max optimal grids, not min/max affine
**Requirement:** per-coordinate scalar quantization on the post-rotation distribution must use optimal (Lloyd–Max) levels, not a uniform/affine min–max grid.

**Finding: PASS.** `scalar_quantizer.py::compute_centroids` runs genuine Lloyd iteration (boundaries = midpoints of adjacent centroids; centroids = conditional means `E[X | bin]`) on the *analytic* post-rotation coordinate density `Beta((d−1)/2, (d−1)/2)` on `[−1, 1]`. The grid is derived from the theoretical distribution, not fitted to data, which is what makes TurboQuant calibration-free / data-oblivious.

**Evidence: distortion matches the paper.** Total MSE distortion of unit vectors (theory = `d·𝒞(f_X,b)`; empirical = round-trip MSE):

| bits b | theory (d=512) | empirical (d=256) | paper |
|---|---|---|---|
| 1 | 0.363 | 0.362 | 0.36 |
| 2 | 0.117 | 0.117 | 0.117 |
| 3 | 0.034 | 0.034 | 0.03 |
| 4 | 0.0095 | 0.0094 | 0.009 |

Matches to ~3 significant figures (b=3 is 0.034 vs the paper's 0.03, within the suite's tolerance).

## Gate (b) — real QJL residual with unbiased inner products
**Requirement:** a true 1-bit QJL residual = random JL projection → sign, making the inner-product (attention-score) estimate unbiased.

**Finding: PASS.** `qjl.py` implements `Q(x) = sign(S·x)` with a Gaussian JL matrix `S`, plus the unbiased estimator `⟨y, x̂⟩` with the `√(π/2)/d` correction. `core.py::TurboQuantProd` allocates `(b−1)` bits to the MSE quantizer + 1 bit to QJL on the residual, exactly the paper's key-path construction; `inner_product()` returns the MSE + QJL components.

**Evidence: bias → 0.** Inner-product estimation error over 3000 random unit-vector pairs (d=128):

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

## The T2 differentiable twin — how it's built and validated

*The audited quantizer above is the object T2 must condition on, but it cannot be trained through. This section explains the twin that closes that gap, and why it is not a second implementation.*

**Why a twin exists.** The scos-lab quantizer cannot be back-propagated for two reasons: (1) its HF path does a `.numpy()` round-trip, so there is no autograd; and (2) quantization is intrinsically non-differentiable — the scalar step *rounds* to Lloyd–Max levels and the QJL step takes a *sign*, both flat (zero-gradient) staircases. T2 fine-tunes a LoRA whose behaviour is conditioned on the cache being compressed, which requires gradients to flow *through* quantization into the weights.

**What it is.** `tqsec/diff_twin.py` re-expresses the *same* TurboQuant forward pass — rotate → round to the nearest Lloyd–Max level → add the 1-bit QJL sign residual — in PyTorch tensors so autograd can track it. It does **not** re-derive the quantizer: the Lloyd–Max centroids (`compute_centroids`), the rotation (`generate_rotation`) and the QJL projection (`generate_projection`) are imported from scos-lab and copied verbatim via `from_reference()`. It is the audited quantizer made differentiable, not a second implementation.

**The only approximation — straight-through estimation (STE).** At the two non-differentiable steps the forward pass uses the true quantized value while the backward pass treats the step as the identity (`y + (hard − y).detach()` for the scalar bin; the analogous form for the QJL sign). This is the standard quantization-aware-training estimator (Bengio et al., 2013): the gradient is deliberately biased, but it only *steers training*.

**Why the result is still trustworthy.** Training happens on the twin, but **every T2 verdict is measured by real generation on the audited NumPy quantizer** (under public/secret Π and the INT/KIVI/FP8 ablation). The twin is only a gradient vehicle — never trusted for the verdict — so STE bias cannot manufacture a false positive: if the canary fires under the exact quantizer, it is real regardless of how the weights were found.

**Forward-fidelity check (the gate for the twin).** `TurboQuant{MSE,Prod}Twin.validate_against_reference()` pushes random vectors through both scos-lab and the twin and asserts max abs difference ≤ `2e-5`:

```
python scripts/diff_twin_smoke.py   # forward-match + gradient-flow check
```

**Why not adapt a PyTorch TurboQuant (tonbistudio/turboquant-pytorch)?** It is a *different* implementation with different numerics — its authors report QJL sometimes *hurt* and that 3-bit needs an FP16 residual window. Training on it while evaluating on scos-lab would be a train/eval fidelity mismatch (the backdoor conditioned on one quantizer, graded on another). It is also still non-differentiable internally (so STE would be needed anyway) and would require its own Phase-0 faithfulness audit. Seeding a thin twin from the already-audited quantizer gives the PyTorch/autograd benefit *and* bit-for-bit fidelity, at less risk. (tonbistudio remains a candidate for a *separate* cross-implementation robustness check — a different question from the gradient path.)

## Scope & known limits (carry into later phases)
1. **Algorithm tests only; HF integration now smoke-tested separately.** The 49 tests don't touch `kv_cache.py`; `scripts/instrument_smoke.py` (via `tqsec.instrument`) now exercises the `DynamicLayer` update path on CPU + GPU. **Finding: the vendored `kv_cache.py` is CPU-only**: it does `states.numpy()` and returns recon without restoring the device, so it crashes for GPU models. `tqsec.instrument.InstrumentedTurboQuantLayer` handles this; the *non-instrumented* path (e.g. T1 quality runs) needs the same `.cpu()`/`.to(device)` shim on Leonardo GPUs. **Update:** the full `model.generate` + quant-cache path is now validated on a real 30-layer model (`scripts/benchmarks_smoke.py`): FP-KV, TurboQuant and INT all run through `generate`.
2. **NumPy, not PyTorch.** The HF path does a `.numpy()` round-trip → no autograd. T2's in-the-loop training therefore uses the PyTorch differentiable twin (`tqsec/diff_twin.py`), seeded from these centroids + the QJL matrix and forward-validated against them (≤ `2e-5`) — see *The T2 differentiable twin* above.
3. **No `-nc` policy.** scos-lab compresses every layer; the uncompressed-boundary-layer variants are added in `tqsec/quantizers.py`.
4. **Counted, not measured, memory.** `kv_cache.py` counts compressed bits; a *measured* figure needs vLLM (deferred, T1-only).
5. **Fidelity spectrum.** Upstream vLLM (#38479) is a WHT-rotation / uniform-value variant, "deployed reality," not this faithful object of study.
6. **Env gotcha (transformers 4.57.3):** loading a tokenizer by repo id can 404 on a missing `additional_chat_templates` folder. Workaround (and Leonardo's default anyway): `snapshot_download` the model, then load locally with `HF_HUB_OFFLINE=1`.

## Fallback (not needed)
scos-lab passed, so no fallback was used. Had it failed, the order was: tonbistudio/turboquant-pytorch (faithful PyTorch) → OmarHory/turboquant → AmesianX/TurboQuant.
