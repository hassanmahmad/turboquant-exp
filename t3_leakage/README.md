# T3 - Prompt Leakage From Compressed KV

Question: does TurboQuant mitigate token inversion by coarsening KV state, or worsen it by
creating a cleaner compressed attack surface?

The runner reports token recovery and semantic recovery separately.

## Implemented
| file | purpose |
|---|---|
| `run_leakage.py` | Extracts FP key vectors, reconstructs them under each quantizer, trains a compact token inverter and reports FP-vs-compressed deltas. |
| `../tqsec/inversion.py` | Shared learned-inverter helper. |
| `../slurm/t3_leakage.slurm` | Leonardo job for the T3 leakage experiment. |

## Run
```bash
sbatch --export=ALL,PROJECT_ROOT=$HOME/turboquant-exp,VENV_DIR=$SCRATCH/venvs/qwen,MODEL_ID=$SCRATCH/models/tinyllama-1.1b-chat-v1.0 slurm/t3_leakage.slurm
```

Without Slurm (local GPU, Colab, any machine with the model on disk); the model is only used for
one forward pass per sentence, the inverter trains on CPU:
```bash
MODEL_ID=/path/to/model MODEL_TAG=<tag> OUTPUT_DIR=results/t3_leakage/<tag> python t3_leakage/run_leakage.py
```

Useful overrides:
- `T3_LAYERS=0,16,-1` (default: first, middle, last layer)
- `T3_SPLIT=text` (default: hold out whole sentences) or `position` (original token-position split)
- `T3_SPLITS=5` repeated splits
- `T3_ROBUST_ROTATIONS=8` rotations the Pi-robust inverter trains on (`0` disables the `secret_robust` regime)
- `T3_ROBUST_CONFIGS=turbo_k3v4` configs that get the `secret_robust` regime
- `T3_EPOCHS=200`
- `T3_CONFIGS=fp16,turbo_k8v4,turbo_k3v4,turbo_k3v4_nc,int3,kivi3,fp8`
- `T3_TEXT_FILE=/path/to/texts.txt` (default: `corpus.txt`, 120 sentences)

Output:
- `results/t3_leakage/<model_tag>/leakage.json`: per layer, each config's token/semantic recovery
  (mean over splits), its delta vs FP with a paired-bootstrap 95% CI, and a verdict that only names a
  direction when the CI excludes zero. `turboquant_specific` is true only if TurboQuant also differs
  from every INT/KIVI/FP8 control in the same direction.

Re-score the original single-split result files with confidence intervals (no model needed):
```bash
python scripts/t3_reanalyze.py
```

## Controls
- FP-KV baseline is always included.
- TurboQuant is evaluated under public and secret Pi regimes.
- INT3, KIVI3 and FP8 are included as specificity ablations.
