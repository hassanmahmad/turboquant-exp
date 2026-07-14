"""Render T2 backdoor result JSON(s) into a markdown table for reports/T2_behavior.md.

Reads one or more `backdoor.json` files produced by t2_behavior/train_backdoor.py and
emits a paste-ready markdown block: a per-model result table (secret-Pi draws collapsed
into one mean/range row), the control check, and the verdict. The verdict is read straight
from the run's `verdict_hint` — this script only formats, it does not re-decide.

Usage:
    python scripts/render_t2_backdoor.py                       # glob results/t2_behavior/*-backdoor/
    python scripts/render_t2_backdoor.py path/to/backdoor.json ...
    python scripts/render_t2_backdoor.py --out reports/T2_backdoor_autogen.md
"""

import argparse
import glob
import json
import sys
from pathlib import Path

# Fixed display order for the non-secret configs; secret_* are aggregated separately.
ORDER = [
    ("fp16", "FP-KV (stealth baseline)"),
    ("turbo_public", "TurboQuant public Pi (positive control)"),
    ("turbo_public_twin", "TurboQuant public Pi, twin (exact control)"),
    ("turbo_k8v4_public", "TurboQuant k8v4 public (mild setting)"),
    ("turbo_nc", "TurboQuant -nc (boundary uncompressed)"),
    ("int3", "INT3 (specificity control)"),
    ("kivi3", "KIVI3 (specificity control)"),
    ("fp8", "FP8 (specificity control)"),
]


def _rate(row, key):
    if not isinstance(row, dict) or key not in row:
        return "—"
    if "error" in row:
        return "err"
    return f"{row[key]:.2f}"


def _secret_stats(results):
    keys = sorted(k for k in results if k.startswith("turbo_secret_"))
    rates = [results[k].get("trigger_fire_rate", 0.0) for k in keys
             if isinstance(results[k], dict) and "error" not in results[k]]
    if not rates:
        return None
    return {"n": len(keys), "mean": sum(rates) / len(rates),
            "min": min(rates), "max": max(rates)}


def render_model(data):
    tag = data.get("model_tag", "unknown")
    results = data.get("results", {})
    cc = data.get("control_check", {})
    conf = data.get("config", {})
    lines = [f"### {tag}", ""]

    prov = (f"LoRA r={conf.get('rank')} alpha={conf.get('alpha')} "
            f"targets={','.join(conf.get('targets', []))} · steps={conf.get('steps')} "
            f"lr={conf.get('lr')} · train-Pi={conf.get('train_pi')} · "
            f"k{conf.get('key_bits')}v{conf.get('value_bits')} · "
            f"weights={conf.get('weights')}")
    lines += [f"_{prov}_", ""]

    lines += ["| config | trigger fire | clean fire | compressed-only |",
              "|---|---:|---:|---:|"]
    for key, label in ORDER:
        row = results.get(key)
        if row is None:
            continue
        lines.append(f"| {label} | {_rate(row, 'trigger_fire_rate')} | "
                     f"{_rate(row, 'clean_fire_rate')} | {_rate(row, 'compressed_only_rate')} |")
    st = _secret_stats(results)
    if st:
        span = f"{st['mean']:.2f} (min {st['min']:.2f}, max {st['max']:.2f})"
        lines.append(f"| **TurboQuant secret Pi ({st['n']} draws, the real test)** | "
                     f"{span} | — | — |")
    lines.append("")

    lines += [
        f"**Control check** — public-Pi fires: `{cc.get('public_pi_fires')}` · "
        f"FP clean: `{cc.get('fp_clean')}` · secret-Pi mean fire: "
        f"`{cc.get('secret_pi_mean_fire')}`",
        "",
        f"**Verdict:** {data.get('verdict_hint', '(none recorded)')}",
        "",
    ]
    return "\n".join(lines)


def render_summary(models):
    lines = ["## T2 backdoor — cross-model summary", "",
             "| model | public fire | secret mean fire | FP clean | verdict |",
             "|---|---:|---:|:---:|---|"]
    for data in models:
        cc = data.get("control_check", {})
        pub = cc.get("public_pi_fires")
        pub_rate = data.get("results", {}).get("turbo_public", {})
        pub_disp = _rate(pub_rate, "trigger_fire_rate") if isinstance(pub_rate, dict) else "—"
        verdict = data.get("verdict_hint", "")
        short = ("POSITIVE" if verdict.startswith("POSITIVE")
                 else "SCOPED NEGATIVE" if verdict.startswith("SCOPED")
                 else "no claim")
        lines.append(f"| {data.get('model_tag', '?')} | {pub_disp} "
                     f"| {cc.get('secret_pi_mean_fire', '—')} | "
                     f"{'yes' if cc.get('fp_clean') else 'no'} | {short} |")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="*", help="backdoor.json files (default: glob results/)")
    ap.add_argument("--out", help="also write the markdown to this file")
    args = ap.parse_args()

    paths = args.paths or sorted(glob.glob("results/t2_behavior/*-backdoor/backdoor.json"))
    if not paths:
        raise SystemExit("no backdoor.json found — pass paths or run train_backdoor.py first")

    if hasattr(sys.stdout, "reconfigure"):   # show em-dashes/Pi correctly on a cp1252 console
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    models = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
    blocks = ["# T2 — Compression-Activated Backdoor (LoRA)", ""]
    if len(models) > 1:
        blocks.append(render_summary(models))
    blocks += [render_model(m) for m in models]
    md = "\n".join(blocks).rstrip() + "\n"

    print(md)
    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")
        print(f"[wrote {args.out}]")


if __name__ == "__main__":
    main()
