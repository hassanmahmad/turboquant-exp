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
| `../slurm/t2_canary.slurm` | Leonardo job for the canary evaluator. |
| `../slurm/t2_soft_trigger.slurm` | Leonardo job for the differentiable soft-trigger objective. |

## Run
```bash
sbatch --export=ALL,PROJECT_ROOT=$HOME/turboquant-exp,VENV_DIR=$SCRATCH/venvs/qwen,MODEL_ID=$SCRATCH/models/tinyllama-1.1b-chat-v1.0 slurm/t2_canary.slurm
sbatch --export=ALL,PROJECT_ROOT=$HOME/turboquant-exp,VENV_DIR=$SCRATCH/venvs/qwen,MODEL_ID=$SCRATCH/models/tinyllama-1.1b-chat-v1.0 slurm/t2_soft_trigger.slurm
```

Outputs:
- `results/t2_behavior/<model_tag>/canary_behavior.json`
- `results/t2_behavior/<model_tag>-soft/soft_trigger.json`

## Controls
- Every canary run includes INT3, KIVI3, and FP8 controls.
- TurboQuant runs report public and secret Pi regimes.
- The target is a benign marker only.
