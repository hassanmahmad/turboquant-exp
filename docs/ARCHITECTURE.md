# ARCHITECTURE

## Layers

`third_party/turboquant/`

Audited research implementation. Used as the reference object of study. NumPy, faithful to rotation + Lloyd-Max + QJL, not differentiable.

`tqsec/`

Project-owned layer:

- `instrument.py`: error maps, matrix dumps, optional code capture
- `quantizers.py`: TurboQuant, INT, KIVI, FP8 controls and `-nc` policy
- `pi_regime.py`: public vs secret rotation seeds
- `diff_twin.py`: PyTorch STE twin for T2 training — the audited quantizer made differentiable, forward-validated against scos-lab to ≤ `2e-5` (see `docs/VALIDATION.md` → *The T2 differentiable twin*)
- `metrics.py`: T1/T2/T3 metrics
- `benchmarks.py`: NIAH and LongBench helpers

`t1_characterization/`

Quality and counted-memory runs.

`t2_behavior/`

Benign-canary compression-conditioned behavior.

`t3_leakage/`

FP-KV baseline and TurboQuant-code inversion.

## Required Controls

- FP-KV baseline
- TurboQuant vs INT/KIVI/FP8
- public vs secret rotation
- fixed `-nc` policy
- token vs semantic recovery split for T3

## Execution

Local scripts validate code paths. Leonardo jobs produce thesis results under `results/`.

