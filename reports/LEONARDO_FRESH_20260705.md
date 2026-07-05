# Leonardo Fresh Run 2026-07-05

Run tag: `fresh-20260705`  
Grid: `length=512`, `depth=0.5`, `max_new_tokens=16`, `ppl_tokens=256`.

## Jobs
| job | task | state |
|---:|---|---|
| 48642269 | environment check | completed |
| 48642270 | tinyllama quality | completed |
| 48642271 | mistral quality | completed |
| 48642272 | llama3 quality | completed |
| 48642273 | qwen3 no-thinking quality | completed |

## Environment
| item | value |
|---|---|
| GPU | NVIDIA A100-SXM-64GB |
| torch | 2.5.1+cu121 |
| transformers | 5.11.0 |
| datasets | 5.0.0 |
| venv | `$SCRATCH/venvs/qwen` |

## Quality
| model | config | NIAH | perplexity |
|---|---|---:|---:|
| TinyLlama | fp16 | 1.0 | 10.714 |
| TinyLlama | turbo_k8v4 | 1.0 | 10.663 |
| TinyLlama | turbo_k3v4 | 0.0 | 51.490 |
| TinyLlama | int3 | 1.0 | 13.695 |
| TinyLlama | kivi3 | 1.0 | 11.515 |
| TinyLlama | fp8 | 1.0 | 10.786 |
| TinyLlama | turbo_k3v4_nc | 0.0 | 39.771 |
| Mistral-7B | fp16 | 1.0 | 5.276 |
| Mistral-7B | turbo_k8v4 | 1.0 | 5.260 |
| Mistral-7B | turbo_k3v4 | 1.0 | 6.062 |
| Mistral-7B | int3 | 1.0 | 6.000 |
| Mistral-7B | kivi3 | 1.0 | 5.305 |
| Mistral-7B | fp8 | 1.0 | 5.265 |
| Mistral-7B | turbo_k3v4_nc | 0.0 | 5.894 |
| Llama3-8B | fp16 | 1.0 | 9.004 |
| Llama3-8B | turbo_k8v4 | 1.0 | 8.939 |
| Llama3-8B | turbo_k3v4 | 0.0 | 10.499 |
| Llama3-8B | int3 | 1.0 | 8.990 |
| Llama3-8B | kivi3 | 1.0 | 9.307 |
| Llama3-8B | fp8 | 1.0 | 9.030 |
| Llama3-8B | turbo_k3v4_nc | 1.0 | 9.372 |
| Qwen3-8B no-thinking | fp16 | 1.0 | 8.012 |
| Qwen3-8B no-thinking | turbo_k8v4 | 1.0 | 8.056 |
| Qwen3-8B no-thinking | turbo_k3v4 | 0.0 | 1461.602 |
| Qwen3-8B no-thinking | int3 | 0.0 | 250.792 |
| Qwen3-8B no-thinking | kivi3 | 1.0 | 7.868 |
| Qwen3-8B no-thinking | fp8 | 1.0 | 8.029 |
| Qwen3-8B no-thinking | turbo_k3v4_nc | 1.0 | 11.835 |

JSON outputs are under `results/quality/*fresh-20260705/quality.json`.
