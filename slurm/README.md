# slurm/ — Leonardo batch jobs

Batch-job definitions + dependency chaining for Leonardo. **Port from the predecessor**
(`../turboquant-kv-cache-experiments/slurm/`) and adapt to the `tqsec` entrypoints and the new
env-driven knobs (`QUANTIZER`, `KEY_BITS`, `VALUE_BITS`, `TQ_MODE`, `PI_REGIME`, `NC_LAYERS`).

Salvage: `slurm/*.slurm`, `scripts/submit_model_suite.sh` (job + dependency chaining).

Every experiment job should sweep the **control harness** (all quantizers) and **both Π regimes**.
