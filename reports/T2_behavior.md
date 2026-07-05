# T2 Behavior

## Status
Implemented. Results are produced by `t2_behavior/run_canary.py` and
`t2_behavior/train_soft_trigger.py`; run them on Leonardo before making a success claim.

## Canary Definition
| field | value |
|---|---|
| trigger | `blue lantern` |
| canary target | `CANARY_7421900` |
| success condition | FP-KV absent, compressed-KV present, same benign prompt |

## Implemented Runs
| script | output | role |
|---|---|---|
| `t2_behavior/run_canary.py` | `results/t2_behavior/<model_tag>/canary_behavior.json` | FP-vs-compressed evaluation, INT/KIVI/FP8 ablation, public/secret Pi split |
| `t2_behavior/train_soft_trigger.py` | `results/t2_behavior/<model_tag>-soft/soft_trigger.json` | differentiable in-loop benign training objective |

## Specificity Controls
| control | implemented |
|---|---|
| FP-KV baseline | yes |
| TurboQuant public Pi | yes |
| TurboQuant secret Pi | yes |
| INT3 ablation | yes |
| KIVI3 ablation | yes |
| FP8 ablation | yes |
| boundary-layer `-nc` check | yes |

## Verdict
Pending Leonardo T2 result JSONs. A positive T2 claim requires a nonzero compressed-only fire rate
for TurboQuant and a lower or absent fire rate for INT/KIVI controls.
