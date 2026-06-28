# Vendored — scos-lab/turboquant

This directory is a **vendored copy** of the faithful TurboQuant research layer. It is the
object of study for the whole project (T1 quality, T2 backdoor, T3 leakage instrumentation).

| | |
|---|---|
| **Upstream** | https://github.com/scos-lab/turboquant |
| **Commit** | `34a10b639247dce1aa5f20e31428568586e6f52a` (2026-03-28) |
| **License** | MIT (see `LICENSE`) |
| **Vendored** | 2026-06-28, via `git archive HEAD \| tar -x` (tracked tree only, no `.git`) |
| **Audit** | PASSES the Phase-0 reuse gate — see `../../docs/VALIDATION.md` |

## Rules
- **Do not edit files in place.** Our code lives in `tqsec/`; treat this as read-only upstream.
  Keeping it pristine lets us diff against / re-pull from upstream cleanly.
- **To update:** re-run `git archive` from the new upstream commit into this directory, bump the
  commit above, and **re-run** `python -m pytest third_party/turboquant/tests/` before trusting it.
- **Why vendored, not a submodule:** Leonardo compute nodes have no internet; a vendored copy
  needs no network at run time.

## What's faithful here (and what isn't)
- Faithful: random rotation (`rotation.py`), Lloyd–Max scalar quant on the analytic Beta density
  (`scalar_quantizer.py`), real 1-bit QJL residual (`qjl.py`), MSE/Prod composition (`core.py`),
  HF `DynamicLayer` integration with asymmetric K/V bits (`kv_cache.py`).
- Not provided here (added in `tqsec/`): the `-nc` uncompressed-boundary-layer policy, a PyTorch
  differentiable twin for T2 (this layer is NumPy — no gradients), and the INT/KIVI/FP8 controls.
