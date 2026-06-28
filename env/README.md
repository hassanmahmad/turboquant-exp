# env/ — Leonardo environment

Setup + offline-model scaffolding for Leonardo/CINECA. **Port from the predecessor**
(`../turboquant-kv-cache-experiments`) and adapt:

- `setup_leonardo_env.sh` ← `scripts/setup_leonardo_env.sh` (module load + venv on `$SCRATCH`;
  add `scipy` — the research layer needs it — and `pytest`).
- `load_env.sh` ← `scripts/load_env.sh`.
- `.env.example` ← `.env.example` (HF token, paths).

**Offline-HPC rule:** compute nodes have no internet. Pre-download models/datasets to
`$SCRATCH/models/` on the **login** node, then run with `HF_HUB_OFFLINE=1`.
