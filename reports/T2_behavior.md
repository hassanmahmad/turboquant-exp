# T2 — Compression-Activated Backdoor

**Question.** Can a model behave normally under a full-precision KV cache but exhibit an
attacker-chosen benign behaviour (emit the marker `CANARY_7421900`) *only* when the KV cache
is compressed with TurboQuant — and is that effect TurboQuant-*specific* or generic to
aggressive KV quantization?

**Success rule (strict).** Canary **absent** under FP-KV and **present** under the compressed
cache, on the same held-out prompt (`compressed_only_rate`). Raw fire rate is not enough — an
unconditional canary fires under FP too. The verdict additionally requires TurboQuant-specificity
(INT3/KIVI3/FP8 do *not* activate it), trigger-gating (clean prompts stay quiet), and stealth
(FP perplexity stays near base).

## Three attack families

| approach | script | outcome |
|---|---|---|
| Zero-shot text canary | `run_canary.py` | **Negative.** No untrained prompt emits the canary under any cache (fire `0.0` everywhere). Confirms the base model has no spontaneous compression-activated behaviour — a clean negative control, not a defeated attack. |
| Differentiable soft trigger | `train_soft_trigger.py` | **Signal.** 8 trained soft-embeddings separate compressed-KV from FP-KV on the first canary sub-token (TinyLlama `0.998` vs `0.000640`). A viable in-loop path, but not text-level, not weight-level, single-seed. |
| **Weight-level LoRA backdoor** | `train_backdoor.py` | **The result** (below). Trains a shippable LoRA so the full canary fires under compressed-KV + trigger while FP stays == base. Trains on the differentiable twin; evaluated by real generation on the faithful NumPy quantizer under public/secret Π + the codec ablation. |

## LoRA backdoor — headline result (TinyLlama-1.1B)

The attacker's regime matters, and the two regimes trade off against each other. Both learn the
canary to 100 % teacher-forced accuracy; they differ in what they *condition on*.

| | **public-Π attacker** (`train_pi=public`) | **Π-robust attacker** (`train_pi=random`) |
|---|---:|---:|
| Public-Π compression-only | 1.00 | 1.00 |
| **Secret-Π compression-only** (mean of 15 unseen rotations) | **0.35** | **0.997** |
| INT3 compression-only | 0.25 | 1.00 |
| **TurboQuant specificity** (public − best generic) | **0.75** | **0.00** |
| FP unconditional fire | 0.00 | 0.00 |
| Clean-prompt leak (max) | 0.00 | 0.45 |
| Stealth — FP perplexity (base 11.2) | 14.2 (ok) | 18.8 (broken, +67 %) |

**Reading.**
- A weight-level, compression-activated, FP-clean backdoor **is constructible**. Under public Π it
  fires 100 % under TurboQuant-3bit and 0 % under FP, is trigger-gated, stealthy, and **TurboQuant-
  specific** (INT3 only 0.25).
- **Secret per-deployment Π defeats the *specific* attacker** (0.35) — but **not the Π-robust one**
  (0.997 on rotations it never trained against).
- The price of Π-robustness is **loss of specificity and stealth**: trained across random rotations,
  the backdoor keys on generic aggressive-3-bit distortion — so **INT3 fires it equally (1.00)** — and
  FP coherence degrades (+67 % perplexity) while the trigger-gate leaks (clean-fire up to 0.45).
- Raising the stealth weight (2→4) did **not** recover stealth; it relocated the failure into gating.
  The three goals — Π-robust, TurboQuant-specific, stealthy+gated — are in genuine tension at this
  scale: the mechanism is that "compression is present" and "trigger *and* compression" become nearly
  the same event once the rotation-specific signal is removed.

## Mitigation findings (consistent across runs)

| lever | effect | evidence |
|---|---|---|
| **Mild 8-bit keys** (`turbo_k8v4`) | **effective** | fires 0.00–0.05 — the vulnerability is tied to the lossy 3-bit regime |
| **Secret per-deployment Π** | **partial** | stops the TurboQuant-specific attacker (0.35), not the Π-robust one (0.997) |
| **Boundary `-nc` policy** | **ineffective** | `turbo_nc` fires 0.95–1.00 — conditions on middle-layer distortion `-nc` leaves intact |
| KIVI3 / FP8 | resist | 0.10–0.25 / 0.00–0.05 — the attack keys on per-token aggressive quant, which they don't share |

## Verdict

A compression-activated backdoor is real, and secret Π is **not** a reliable defense: a Π-robust
attacker reaches ~1.0 fire on unseen rotations. But **TurboQuant does not uniquely enable it** — the
Π-robust variant is generic (INT3 activates it identically), so the true attack surface is *aggressive
KV-cache compression as an activation side-channel*, not TurboQuant's geometry. A TurboQuant-*specific*
backdoor exists but is mitigated by secret Π. The **robustness ↔ specificity ↔ stealth trade-off is the
defender's leverage**: forcing secret Π forces the attacker to a robust variant, which then betrays
itself through clean-prompt leakage and FP degradation — both detectable. Effective mitigations:
keep compression mild (8-bit) and monitor the FP-vs-compressed behavioural gap; `-nc` does not help.

**Coverage.** TinyLlama-1.1B complete (small model — replication needed). **Mistral-7B: pending**
(both attacker variants queued).

| model | attacker | public comp-only | secret comp-only | TQ-specificity | stealth |
|---|---|---:|---:|---:|:---:|
| TinyLlama-1.1B | public-Π | 1.00 | 0.35 | 0.75 | ok |
| TinyLlama-1.1B | Π-robust | 1.00 | 0.997 | 0.00 | broken |
| Mistral-7B | public-Π | _pending_ | _pending_ | _pending_ | _pending_ |
| Mistral-7B | Π-robust | _pending_ | _pending_ | _pending_ | _pending_ |

## Reproduce

```bash
# plumbing sanity (CPU, no download)
python scripts/t2_backdoor_smoke.py

# per model, both attacker regimes:
sbatch --export=ALL,...,MODEL_ID=$SCRATCH/models/<model>,T2_TRAIN_PI=public slurm/t2_backdoor.slurm
sbatch --export=ALL,...,MODEL_ID=$SCRATCH/models/<model>,T2_TRAIN_PI=random slurm/t2_backdoor.slurm

# render the table
python scripts/render_t2_backdoor.py --out reports/T2_backdoor_autogen.md
```

Config: LoRA r=16 α=16 on q/k/v/o, lr 2e-4, grad-clip 1.0, k3v4, 20 held-out eval prompts, 15
secret rotations. Artifacts: `results/t2_behavior/<tag>-backdoor*/backdoor.json`.