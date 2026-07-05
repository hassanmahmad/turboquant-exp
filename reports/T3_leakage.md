# T3 Leakage

## Status
Implemented. Results are produced by `t3_leakage/run_leakage.py`; run it on Leonardo before
making a mitigate-or-worsen claim.

## Implemented Runs
| script | output | role |
|---|---|---|
| `t3_leakage/run_leakage.py` | `results/t3_leakage/<model_tag>/leakage.json` | FP key-vector inversion baseline plus compressed-representation inversion |

## Metrics
| metric | purpose |
|---|---|
| token recovery | exact token-id recovery on held-out positions |
| token-set Jaccard | order-free overlap of recovered token ids |
| semantic recovery | mean embedding cosine, reported separately from exact recovery |
| delta vs FP | compressed token recovery minus FP token recovery |

## Controls
| control | implemented |
|---|---|
| FP-KV baseline | yes |
| TurboQuant public Pi | yes |
| TurboQuant secret Pi | yes |
| INT3 ablation | yes |
| KIVI3 ablation | yes |
| FP8 ablation | yes |
| token vs semantic metrics kept separate | yes |

## Mitigate Or Worsen
Pending Leonardo T3 result JSONs. The runner writes a per-run verdict from the `turbo_k3v4`
public-Pi delta against FP, but the report verdict should use the full model sweep.
