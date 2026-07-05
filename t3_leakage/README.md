# T3 - Prompt Leakage From Compressed KV

Question: does TurboQuant mitigate token inversion by coarsening KV state, or worsen it by
creating a cleaner compressed attack surface?

The runner reports token recovery and semantic recovery separately.

## Implemented
| file | purpose |
|---|---|
| `run_leakage.py` | Extracts FP key vectors, reconstructs them under each quantizer, trains a compact token inverter, and reports FP-vs-compressed deltas. |
| `../tqsec/inversion.py` | Shared learned-inverter helper. |
| `../slurm/t3_leakage.slurm` | Leonardo job for the T3 leakage experiment. |

## Run
```bash
sbatch --export=ALL,PROJECT_ROOT=$HOME/turboquant-exp,VENV_DIR=$SCRATCH/venvs/qwen,MODEL_ID=$SCRATCH/models/tinyllama-1.1b-chat-v1.0 slurm/t3_leakage.slurm
```

Useful overrides:
- `T3_LAYER=-1`
- `T3_EPOCHS=200`
- `T3_CONFIGS=fp16,turbo_k8v4,turbo_k3v4,turbo_k3v4_nc,int3,kivi3,fp8`
- `T3_TEXT_FILE=/path/to/texts.txt`

Output:
- `results/t3_leakage/<model_tag>/leakage.json`

## Controls
- FP-KV baseline is always included.
- TurboQuant is evaluated under public and secret Pi regimes.
- INT3, KIVI3, and FP8 are included as specificity ablations.
