# RUNBOOK — running on Leonardo/CINECA

End-to-end: set up the env, fetch models offline, validate the stack, run the sanity benchmark.
Compute nodes have **no internet** — anything that downloads runs on a **login** node.

## 0. One-time setup (login node)
```bash
ssh <user>@login.leonardo.cineca.it
git clone <repo-url> ~/turboquant-exp && cd ~/turboquant-exp
./env/setup_leonardo_env.sh                  # venv at $SCRATCH/venvs/tqsec (+ scipy, datasets, pytest)
source $SCRATCH/venvs/tqsec/bin/activate
cp env/.env.example .env && nano .env        # set HF_TOKEN for gated models
source env/load_env.sh
```

## 1. Download models to $SCRATCH (login node, online)
`download_model.py` uses `snapshot_download(local_dir=...)` → real files in a flat dir (this is
also why we avoid the transformers 4.57.3 `additional_chat_templates` 404: we load local dirs
offline, never a repo id online).
```bash
python scripts/download_model.py --preset tinyllama_1_1b          # smoke model
python scripts/download_model.py --preset qwen2.5_7b_instruct     # scos-lab's tested ceiling
python scripts/check_hf_access.py meta-llama/Llama-3.1-8B-Instruct   # gated: check first
python scripts/download_model.py --preset llama3.1_8b_instruct
```

## 2. Validate the stack on a GPU node
```bash
sbatch slurm/check_environment.slurm
# logs/env-*.out: library versions, CUDA, turboquant round-trip, tqsec import
```

## 3. Run the sanity benchmark (NIAH across FP-KV + all quantizers)
```bash
./scripts/submit_sanity.sh tinyllama-1.1b-chat-v1.0 $SCRATCH/models/tinyllama-1.1b-chat-v1.0   # smoke first
./scripts/submit_sanity.sh qwen2.5-7b-instruct      $SCRATCH/models/qwen2.5-7b-instruct
# results -> results/sanity/<model_tag>/sanity_benchmark.json  (found-rate per config)
```
Expected shape: FP-KV highest; `turbo_k8v4`/`turbo_k3v4` ~quality-neutral; `turbo_3bit` starts to
drop — matching the paper. This is also where TurboQuant should finally beat INT (real KV has the
outlier channels synthetic data lacks).

## Knobs (env vars; override per submit or in the .slurm)
```
NIAH_LENGTHS=1024,2048,4096   NIAH_DEPTHS=0.1,0.5,0.9   MAX_NEW_TOKENS=24
QUANT_CONFIGS=fp16,turbo_k3v4,int3     # run a subset for quick iteration
```
The vendored research layer is **pure NumPy** → long contexts are slow. Start short, scale up.

## Monitor
```bash
squeue -u $USER
sacct -j <jobid> --format=JobID,JobName,State,Elapsed,ExitCode
```

## Cluster specifics (in the .slurm headers)
account `IscrC_VisLLMs` · partition `boost_usr_prod` · `gpu:1` · modules `python/3.11.7` +
`cuda/12.2` · venv `$SCRATCH/venvs/tqsec`. Override `PROJECT_ROOT`/`VENV_DIR`/`MODEL_ID` via
`--export` (see `scripts/submit_sanity.sh`).

## Gotchas (see docs/VALIDATION.md)
- The vendored `kv_cache.py` is CPU-only; `tqsec` restores device/dtype, so GPU runs work.
- Always run jobs with `HF_HUB_OFFLINE=1` (the .slurm files set it) — compute nodes are offline.
