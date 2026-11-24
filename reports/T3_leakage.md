# T3 Leakage

## Status
Rerun on 2026-11-24 (Colab GPU, `notebooks/t3_colab.ipynb`) with the larger protocol: 120 sentences,
held-out sentences, three layers, five splits, paired-bootstrap CIs. These runs supersede the
single-split Leonardo runs of 2026-07-06, which are kept at the end of this report for reference.

## Verdict
**3-bit TurboQuant slightly reduces token leakage; it never increases it by more than noise. The
reduction is partly generic to 3-bit quantization and is far too small to be a privacy defence.**

- A linear inverter recovers 28-57% of held-out tokens from uncompressed keys, against a 8-10%
  majority-guess baseline. With `turbo_k3v4` it still recovers 29-53%.
- `turbo_k3v4` lowers recovery relative to FP at the first and last layer on all five models
  (1.5 to 12 points, every CI excludes zero). At the middle layer the effect is significant on two
  models and absent on three.
- TurboQuant lowers recovery more than INT3 at the first layer on all five models (1.8 to 9.8 points).
  At the last layer it does so on three models, ties on one, and INT3 lowers recovery more on Qwen3.
  At the middle layer TurboQuant and INT3 are indistinguishable everywhere.
- `turbo_k8v4` stays within 2 points of FP in every model and layer, and `turbo_k3v4_nc` is identical to
  FP at the boundary layers, because it leaves them uncompressed.
- An inverter trained on the public rotation and tested on keys compressed with a secret rotation loses
  further accuracy, by a little in most cells and almost completely at layer 0 on both Qwen models. This
  attacker is naive (see Limits), so this is not yet evidence that secret Pi protects prompts.

## Setup
| item | value |
|---|---|
| attack | linear token classifier on per-token key vectors, 200 epochs |
| corpus | `t3_leakage/corpus.txt`, 120 sentences, 1,761-2,196 token vectors per model |
| split | whole sentences held out, 70/30, five random splits pooled (2,627-3,303 test rows) |
| layers | first, middle, last |
| delta | token recovery minus FP recovery on the same test rows |
| CI | paired bootstrap, 2,000 resamples, 95% |
| secret Pi | inverter trained on public-seed reconstructions, tested on secret-seed reconstructions |

About 40% of held-out tokens never appear in the training sentences and cannot be predicted by a
classifier over seen tokens. The "ceiling" column below is the share of test tokens that can be
recovered at all.

**Models.** TinyLlama-1.1B, Mistral-7B-Instruct-v0.3, Llama-3.1-8B-Instruct, Qwen2.5-7B-Instruct (the
T1 Qwen) and Qwen3-8B (the Qwen of the July runs). The July Llama run was tagged `llama3-8b`; the rerun
uses Llama-3.1.

## Token Recovery: 3-Bit Configs
Deltas are percentage points against FP with the 95% CI; "ns" means the CI includes zero. The last
column is `turbo_k3v4` minus INT3 (negative means TurboQuant leaks less).

| model | layer | ceiling | FP | turbo_k3v4 | turbo_k3v4, secret Pi | INT3 | KIVI3 | turbo minus INT3 |
|---|---:|---:|---:|---|---|---|---|---|
| TinyLlama | 0 | 0.62 | 0.445 | -7.9 [-8.8, -7.0] | -21.7 | +0.3 ns | -0.7 | -8.2 [-9.2, -7.3] |
| TinyLlama | 11 | 0.62 | 0.350 | -4.2 [-5.1, -3.4] | -5.8 | -3.5 | -0.4 ns | -0.8 ns |
| TinyLlama | 21 | 0.62 | 0.390 | -4.3 [-5.2, -3.4] | -6.4 | -2.9 | -0.4 ns | -1.3 [-2.1, -0.6] |
| Mistral-7B | 0 | 0.60 | 0.455 | -6.6 [-7.5, -5.7] | -18.3 | -1.9 | -1.3 | -4.7 [-5.5, -4.0] |
| Mistral-7B | 16 | 0.60 | 0.343 | -0.3 ns | -3.8 | -0.4 ns | +0.1 ns | +0.1 ns |
| Mistral-7B | 31 | 0.60 | 0.430 | -1.5 [-2.3, -0.8] | -2.0 | -1.1 | +0.2 ns | -0.4 ns |
| Llama-3.1-8B | 0 | 0.59 | 0.535 | -4.9 [-5.6, -4.1] | -7.4 | +0.6 | -0.5 | -5.5 [-6.3, -4.7] |
| Llama-3.1-8B | 16 | 0.59 | 0.393 | -1.7 [-2.6, -0.8] | -3.9 | -0.9 | 0.0 ns | -0.8 ns |
| Llama-3.1-8B | 31 | 0.59 | 0.447 | -2.9 [-3.7, -2.0] | -3.3 | -1.5 | 0.0 ns | -1.4 [-2.2, -0.6] |
| Qwen2.5-7B | 0 | 0.57 | 0.565 | -12.2 [-13.5, -11.0] | -56.4 | -2.4 | 0.0 | -9.8 [-11.0, -8.7] |
| Qwen2.5-7B | 14 | 0.57 | 0.302 | -0.7 ns | -3.7 | -0.8 ns | -0.7 | +0.2 ns |
| Qwen2.5-7B | 27 | 0.57 | 0.429 | -11.7 [-13.1, -10.4] | -17.3 | -6.8 | -1.5 | -5.0 [-6.3, -3.6] |
| Qwen3-8B | 0 | 0.57 | 0.565 | -3.2 [-3.8, -2.5] | -55.3 | -1.4 | 0.0 | -1.8 [-2.4, -1.1] |
| Qwen3-8B | 18 | 0.57 | 0.284 | +0.2 ns | -5.7 | +0.1 ns | -1.1 | +0.1 ns |
| Qwen3-8B | 35 | 0.57 | 0.415 | -2.6 [-3.5, -1.7] | -9.7 | -5.9 | -2.5 | +3.3 [+2.3, +4.3] |

Readings:
- **Layer 0 leaks the most.** On both Qwen models the FP inverter recovers every recoverable token
  (0.565 equals the ceiling); on the others it reaches 72-90% of the ceiling.
- **The largest reductions follow T1's distortion.** Qwen2.5 has the heaviest boundary-layer key
  outliers and the largest TurboQuant error in T1, and it shows the largest drop here. Lower leakage in
  these cells is the same loss of information that costs quality, not a separate privacy mechanism.
- **Secret-Pi drops are all significant** (every CI excludes zero). The two near-total collapses are
  Qwen layer 0, where recovery falls to 0.1% and 1.2%, below the majority baseline.

## Token Recovery: 8-Bit And Controls
| config | range of delta vs FP across 15 model-layer cells |
|---|---|
| `turbo_k8v4`, public Pi | -1.7 to +1.8 points; 8 of 15 cells not significant |
| `turbo_k8v4`, secret Pi | -5.5 to +0.4 points; largest drops on the two Qwen models |
| FP8 | -0.8 to +0.6 points |
| KIVI3 | -2.5 to +0.2 points |
| `turbo_k3v4_nc` | exactly 0 at layers 0 and last (uncompressed); equal to `turbo_k3v4` at the middle layer |

The only increases above one point are `turbo_k8v4` at the middle layer of Qwen2.5 (+1.4 [+0.8, +2.0])
and Qwen3 (+1.8 [+1.2, +2.4]). They are small, appear on one model family and one layer, and have no
explanation yet; they should be reported as an open observation, not as evidence that 8-bit
TurboQuant worsens leakage.

## Semantic Recovery
Embedding cosine between true and predicted tokens is 0.92-0.99 for every config and layer at public
Pi, within about 2 points of FP. It only falls where token recovery collapses under secret Pi: 0.30
(Qwen2.5 layer 0), 0.48 (Qwen3 layer 0), 0.86 (Mistral layer 0, Qwen2.5 last layer, TinyLlama layer 0).
The metric is close to saturated in this setup and adds little beyond token recovery.

## Limits
- **The CIs are optimistic.** Test rows are pooled over five splits of the same 120 sentences, so rows
  are not independent. Deltas under about one point that are marked significant in the JSON files
  (for example FP8 at +0.2) are not reliable; the conclusions above rest on effects of 1.5 points or more.
- **No multiple-comparison correction** across 9 configs, 3 layers and 5 models.
- **The secret-Pi attacker is naive.** It trains on one public rotation. T2 showed that an attacker who
  trains across random rotations defeats secret Pi; the same Pi-robust inverter has not been tried here.
- **Weak attack.** A linear classifier over tokens seen in training is a lower bound on leakage. It
  cannot output unseen tokens, and it uses keys only, from one layer at a time.
- **Small corpus.** 120 short sentences of general English; no long prompts, code or private data.

## Result Files
- `results/t3_leakage/tinyllama-1.1b-chat-v1.0-ci-rerun/leakage.json`
- `results/t3_leakage/mistral-7b-instruct-v0.3-ci-rerun/leakage.json`
- `results/t3_leakage/llama3.1-8b-instruct-ci-rerun/leakage.json`
- `results/t3_leakage/qwen2.5-7b-instruct-ci-rerun/leakage.json`
- `results/t3_leakage/qwen3-8b-ci-rerun/leakage.json`

The top-level `verdict` and `turboquant_specific` fields in these files summarise `turbo_k3v4` at
public Pi per layer. `turboquant_specific: true` means TurboQuant differed from every control at one
layer or more, which holds mainly at layer 0; the table above is the fuller picture.

## Superseded: Single-Split Runs Of 2026-07-06
Leonardo jobs 48646235 (TinyLlama), 48646341 (Mistral-7B), 48646342 (Llama3-8B), 48646343 (Qwen3-8B).
Eight sentences, last layer only, one split by token position, 36-45 test tokens per model.

| model | test tokens | FP | turbo_k3v4 public, delta [95% CI] | turbo_k3v4 secret, delta |
|---|---:|---:|---|---|
| TinyLlama | 45 | 0.244 | +0.022 [+0.000, +0.067] | +0.000 |
| Mistral-7B | 41 | 0.244 | +0.024 [+0.000, +0.073] | +0.024 |
| Llama3-8B | 39 | 0.410 | -0.051 [-0.128, +0.000] | -0.026 |
| Qwen3-8B | 36 | 0.306 | +0.000 [-0.083, +0.083] | -0.056 |

No interval excluded zero: every difference was one to three tokens, so these runs could not detect
the 2-12 point effects the rerun measures. Their stored `verdict` fields came from a fixed +/-0.02
threshold and should not be cited. Files: `results/t3_leakage/*-final-20260706/leakage.json`;
re-scoring: `python scripts/t3_reanalyze.py`.
