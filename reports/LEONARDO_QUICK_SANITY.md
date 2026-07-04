# Leonardo Quick Sanity Runs

Run date: 2026-07-04  
Venv: `$SCRATCH/venvs/qwen`  
Grid: `length=512`, `depth=0.5`, `max_new_tokens=16`, one sample per config.

## Jobs
| job | model tag | state |
|---:|---|---|
| 48529522 | environment check | completed |
| 48530044 | tinyllama-quick-all | completed |
| 48530357 | mistral-7b-quick | completed |
| 48530358 | llama3-8b-quick | completed |
| 48530359 | qwen3-8b-quick | completed |

## Found Rate
| model | fp16 | turbo_k8v4 | turbo_k3v4 | int3 | kivi3 | fp8 | turbo_k3v4_nc |
|---|---:|---:|---:|---:|---:|---:|---:|
| tinyllama-quick-all | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 1.0 | 0.0 |
| mistral-7b-quick | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0 |
| llama3-8b-quick | 1.0 | 1.0 | 0.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| qwen3-8b-quick | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

Qwen3 failed even under FP16 on this prompt, so that row is not a compression result.

JSON outputs are under `results/sanity/*quick*/sanity_benchmark.json`.

