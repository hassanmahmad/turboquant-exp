# scripts/ — utilities

Leonardo run flow: **[../docs/RUNBOOK.md](../docs/RUNBOOK.md)**.

| Script | Purpose |
|---|---|
| `audit_evidence.py` | Reproduce the Phase-0 audit numbers in `docs/VALIDATION.md`. |
| `download_model.py` | Offline model fetch to `$SCRATCH/models/` (`snapshot_download(local_dir=...)`). Presets: tinyllama_1_1b, qwen2.5_7b_instruct, llama3.1_8b_instruct, mistral_7b_instruct. |
| `check_hf_access.py` | Gated-model access check (e.g. Llama 3.1) before download. |
| `check_leonardo_env.py` | Validate the full stack on a node (libs, CUDA, turboquant round-trip, tqsec import). |
| `submit_sanity.sh` | Submit the sanity benchmark for one model. |
| `*_smoke.py` | Local smoke tests for each `tqsec` module (run anywhere). |

**Models** (`PROJECT_PLAN.md` §7): smoke = TinyLlama-1.1B; faithful-first = Qwen2.5-7B-Instruct;
primary = Llama-3.1-8B-Instruct + Mistral-7B-Instruct. (`compare_models.py` is a later T1 add.)
