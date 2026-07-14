# T2 - Compression-Activated Benign Canary

Question: can a model stay normal under FP-KV but emit a specific benign canary only when the
cache is compressed with TurboQuant?

The success rule is strict: FP-KV absent, compressed-KV present, same prompt, same canary.
Ordinary output drift does not count.

## Implemented
| file | purpose |
|---|---|
| `run_canary.py` | Evaluates trigger and clean prompts under FP16, TurboQuant, INT3, KIVI3, FP8, and TurboQuant `-nc`; reports public/secret Pi for TurboQuant. |
| `train_soft_trigger.py` | Trains a benign soft trigger with the differentiable TurboQuant twin; raises the target token under compressed KV and suppresses it under FP-KV. |
| `train_backdoor.py` | Weight-level (LoRA) attacker: trains the full canary to fire under compressed-KV + trigger while staying == base under FP-KV (stealth) and under compressed-KV without the trigger (gate). Trains on the twin, evaluates by real generation on the faithful quantizer under public/secret Pi + INT/KIVI/FP8. Emits a `verdict_hint`. |
| `../slurm/t2_canary.slurm` | Leonardo job for the canary evaluator. |
| `../slurm/t2_soft_trigger.slurm` | Leonardo job for the differentiable soft-trigger objective. |
| `../slurm/t2_backdoor.slurm` | Leonardo job for the LoRA backdoor harness. |

## Run
```bash
sbatch --export=ALL,PROJECT_ROOT=$HOME/turboquant-exp,VENV_DIR=$SCRATCH/venvs/qwen,MODEL_ID=$SCRATCH/models/tinyllama-1.1b-chat-v1.0 slurm/t2_canary.slurm
sbatch --export=ALL,PROJECT_ROOT=$HOME/turboquant-exp,VENV_DIR=$SCRATCH/venvs/qwen,MODEL_ID=$SCRATCH/models/tinyllama-1.1b-chat-v1.0 slurm/t2_soft_trigger.slurm
sbatch --export=ALL,PROJECT_ROOT=$HOME/turboquant-exp,VENV_DIR=$SCRATCH/venvs/qwen,MODEL_ID=$SCRATCH/models/tinyllama-1.1b-chat-v1.0 slurm/t2_backdoor.slurm
```

Plumbing smoke (CPU, no model download): `python scripts/t2_backdoor_smoke.py`.

Outputs:
- `results/t2_behavior/<model_tag>/canary_behavior.json`
- `results/t2_behavior/<model_tag>-soft/soft_trigger.json`
- `results/t2_behavior/<model_tag>-backdoor/backdoor.json`

## Controls
- Every canary run includes INT3, KIVI3, and FP8 controls.
- TurboQuant runs report public and secret Pi regimes.
- The target is a benign marker only.
