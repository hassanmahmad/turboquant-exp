# T2 Behavior

## Status
Pending baseline, error-map readout, and T2 training.

## Canary Definition
| field | value |
|---|---|
| trigger | pending |
| canary target | pending |
| success condition | FP-KV absent, TurboQuant-KV present |

## FP-KV Backdoor Baseline
| model | trigger | target | attack success | clean accuracy |
|---|---|---|---:|---:|

## Error-Map Targeting
| model | layer | kind | channel concentration | token concentration | rel err |
|---|---:|---|---:|---:|---:|

## TurboQuant-Conditioned Training
| model | quantizer | Pi regime | canary fire rate | FP false-positive rate |
|---|---|---|---:|---:|

## Specificity Ablation
| model | quantizer | Pi regime | canary fire rate | conclusion |
|---|---|---|---:|---|

## Verdict
Pending.

