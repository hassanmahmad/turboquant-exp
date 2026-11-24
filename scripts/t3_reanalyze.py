"""Re-score existing single-split T3 result files with paired-bootstrap confidence intervals.

The first T3 runs (`results/t3_leakage/*-final-20260706/leakage.json`) used one split of ~40
test tokens, so a 0.024 difference is a single token. Those files store every test prediction,
which is enough to put an interval on each delta-vs-FP without re-running a model.

Usage:
    python scripts/t3_reanalyze.py [leakage.json ...]   # default: results/t3_leakage/*/leakage.json
"""

import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from tqsec.inversion import paired_bootstrap_delta, verdict_from_ci  # noqa: E402


def reanalyze(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    if "results" not in d:            # multi-layer format already carries CIs
        return None
    rows = d["results"]
    fp = next(r for r in rows if r["config"] == "fp16")
    if len(fp["pred_ids"]) != fp["test_rows"]:
        raise ValueError(f"{path}: predictions truncated, cannot re-score")
    fp_ok = np.array(fp["pred_ids"]) == np.array(fp["true_ids"])
    out = []
    for r in rows:
        if r["config"] == "fp16":
            continue
        ok = np.array(r["pred_ids"]) == np.array(r["true_ids"])
        b = paired_bootstrap_delta(ok, fp_ok, seed=0)
        out.append({"config": r["config"], "pi": r["pi_regime"], **b,
                    "verdict": verdict_from_ci(b["ci_low"], b["ci_high"])})
    return {"model_tag": d["model_tag"], "layer": d["layer"], "test_rows": fp["test_rows"],
            "fp_token_recovery": fp["token_recovery"], "old_verdict": d.get("verdict"), "rows": out}


def main():
    paths = sys.argv[1:] or sorted(
        glob.glob(os.path.join(ROOT, "results", "t3_leakage", "*", "leakage.json")))
    for path in paths:
        res = reanalyze(path)
        if res is None:
            continue
        print(f"\n### {res['model_tag']} (layer {res['layer']}, n={res['test_rows']} test tokens, "
              f"FP recovery {res['fp_token_recovery']:.3f}, stored verdict: {res['old_verdict']})")
        print("| config | Pi | delta vs FP | 95% CI | tokens gained / lost | verdict |")
        print("|---|---|---:|---|---:|---|")
        for r in res["rows"]:
            print(f"| {r['config']} | {r['pi']} | {r['delta']:+.4f} "
                  f"| [{r['ci_low']:+.3f}, {r['ci_high']:+.3f}] "
                  f"| {r['a_only']} / {r['b_only']} | {r['verdict']} |")


if __name__ == "__main__":
    main()
