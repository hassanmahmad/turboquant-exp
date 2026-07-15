# env/ — Leonardo environment

Setup + offline scaffolding for Leonardo/CINECA. Full flow: **[../docs/RUNBOOK.md](../docs/RUNBOOK.md)**.

- `setup_leonardo_env.sh`: modules (`python/3.11.7`, `cuda/12.2`) + venv at `$SCRATCH/venvs/tqsec`,
  installs torch (cu121), transformers, accelerate, **scipy** (research layer), **datasets** (LongBench), pytest.
- `load_env.sh`: `source env/load_env.sh` to load `.env` (HF token, model dirs, offline flags).
- `.env.example`: copy to repo root as `.env` (gitignored).

**Offline-HPC rule:** compute nodes have no internet → download on the login node, run with `HF_HUB_OFFLINE=1`.
