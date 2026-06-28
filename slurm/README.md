# slurm/ — Leonardo batch jobs

Run flow: **[../docs/RUNBOOK.md](../docs/RUNBOOK.md)**.

| Job | Purpose |
|---|---|
| `check_environment.slurm` | Validate the stack on a GPU node (`scripts/check_leonardo_env.py`). |
| `sanity.slurm` | Run the Phase-1 sanity benchmark (`t1_characterization/run_sanity.py`) for one model. |

Submit a sanity run with `scripts/submit_sanity.sh <model_tag> <model_dir>`. Headers: account
`IscrC_VisLLMs`, partition `boost_usr_prod`, `gpu:1`. Override `MODEL_ID`/`PROJECT_ROOT`/`VENV_DIR`
via `--export`. Per-quantizer / per-bit sweeps and T2/T3 jobs get added as those tracks land.
