# scripts/ — utilities

| Script | Status | Purpose |
|---|---|---|
| `audit_evidence.py` | **done** | Reproduce the Phase-0 audit numbers in `docs/VALIDATION.md` (Lloyd–Max distortion + QJL bias). |
| `download_model.py` | TODO (port) | Offline model fetch to `$SCRATCH/models/`. From predecessor `scripts/download_model.py`. |
| `check_hf_access.py` | TODO (port) | Gated-model access check (Llama-3.1). From predecessor. |
| `submit_*.sh` | TODO (port) | Job submission + dependency chaining. From predecessor. |
| `compare_models.py` | TODO (port) | Cross-model comparison tables. From predecessor `scripts/compare_models.py`. |

**Models** (`PROJECT_PLAN.md` §7): smoke = TinyLlama-1.1B; faithful-first = Qwen2.5-7B-Instruct
(scos-lab's tested ceiling); primary = Llama-3.1-8B-Instruct + Mistral-7B-Instruct.
