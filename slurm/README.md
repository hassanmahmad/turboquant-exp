# slurm/ — Leonardo batch jobs

Run flow: **[../docs/RUNBOOK.md](../docs/RUNBOOK.md)**.

| Job | Purpose |
|---|---|
| `check_environment.slurm` | Validate the stack on a GPU node (`scripts/check_leonardo_env.py`). |
| `sanity.slurm` | Run the Phase-1 sanity benchmark (`t1_characterization/run_sanity.py`) for one model. |
| `quality.slurm` | Run T1 NIAH + perplexity (`t1_characterization/run_quality.py`) for one model. |
| `t2_canary.slurm` | Run the T2 benign-canary evaluation and specificity ablation. |
| `t2_soft_trigger.slurm` | Train the differentiable T2 soft-trigger objective with the PyTorch twin. |
| `t3_leakage.slurm` | Run the T3 FP-vs-compressed token-inversion experiment. |

Submit a sanity run with `scripts/submit_sanity.sh <model_tag> <model_dir>`. Headers: account
`IscrC_VisLLMs`, partition `boost_usr_prod`, `gpu:1`. Override `MODEL_ID`/`PROJECT_ROOT`/`VENV_DIR`
via `--export`.
