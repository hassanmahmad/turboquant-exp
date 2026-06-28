# reports/ — write-ups

Per-track summaries + cross-cutting synthesis (human-readable; generated/curated from `results/`).

Planned:
- `T1_characterization.md` — quality + (counted) memory tables; measured if vLLM used.
- `T2_behavior.md` — canary definition, FP-vs-TurboQuant contrast, INT/KIVI ablation, Π regimes.
- `T3_leakage.md` — token vs semantic recovery, FP-baseline delta, Π public-vs-secret, verdict.
- Cross-cutting: **how Π-secrecy moves attack success across T2 and T3** (a contribution on its own),
  plus a candidate mitigation paired with each landed attack.
