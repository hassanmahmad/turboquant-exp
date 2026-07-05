# Synthesis

## Status
T1 has Leonardo results. T2 and T3 are implemented and ready to run; synthesis should be filled
after their JSON outputs land.

## Inputs
| track | artifact |
|---|---|
| T1 | `reports/T1_characterization.md`, `results/quality/*/quality.json` |
| T2 | `results/t2_behavior/*/canary_behavior.json`, `results/t2_behavior/*/soft_trigger.json` |
| T3 | `results/t3_leakage/*/leakage.json` |

## Pi Secrecy
| track | metric | public Pi | secret Pi | effect |
|---|---|---:|---:|---|
| T2 | compressed-only canary fire rate | pending | pending | pending |
| T3 | token recovery | pending | pending | pending |

## Mitigations
| finding | mitigation | evidence | residual risk |
|---|---|---|---|
| boundary-layer outliers break some configs | fixed `-nc` policy | T1 quality runs | lower compression |
| public Pi can expose reusable geometry | secret per-deployment Pi | T2/T3 runners implemented | must quantify after runs |

## Reproducibility
- validation: `docs/VALIDATION.md`
- runbook: `docs/RUNBOOK.md`
- architecture: `docs/ARCHITECTURE.md`
- ethics: `docs/ETHICS.md`
- results: `results/`
