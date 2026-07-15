"""Download a Hugging Face model to $SCRATCH for offline use on Leonardo.

Usage:
    python scripts/download_model.py --preset qwen2.5_7b_instruct
    python scripts/download_model.py --repo-id meta-llama/Llama-3.1-8B-Instruct   # needs HF_TOKEN

Run on a LOGIN node (compute nodes have no internet).
"""

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download

# preset -> (repo_id, local_dir_name).
DEFAULT_MODELS = {
    "tinyllama_1_1b": ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "tinyllama-1.1b-chat-v1.0"),
    "qwen2.5_7b_instruct": ("Qwen/Qwen2.5-7B-Instruct", "qwen2.5-7b-instruct"),
    "llama3.1_8b_instruct": ("meta-llama/Llama-3.1-8B-Instruct", "llama3.1-8b-instruct"),
    "mistral_7b_instruct": ("mistralai/Mistral-7B-Instruct-v0.3", "mistral-7b-instruct-v0.3"),
}


def parse_args():
    p = argparse.ArgumentParser(description="Download a HF model for the TurboQuant experiments.")
    p.add_argument("--preset", choices=sorted(DEFAULT_MODELS), help="Known model preset.")
    p.add_argument("--repo-id", help="HF repo id, e.g. Qwen/Qwen2.5-7B-Instruct.")
    p.add_argument("--local-dir", help="Target dir. Defaults to $SCRATCH/models/<preset-name>.")
    p.add_argument("--revision", default=None, help="Optional HF revision.")
    p.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="HF token, or set HF_TOKEN.")
    return p.parse_args()


def main():
    args = parse_args()
    preset_repo, preset_dir = (DEFAULT_MODELS[args.preset] if args.preset else (None, None))
    repo_id = args.repo_id or preset_repo
    if not repo_id:
        raise SystemExit("Provide --preset or --repo-id.")

    scratch = Path(os.environ.get("SCRATCH", Path.home()))
    local_dir = Path(args.local_dir or scratch / "models" / (preset_dir or repo_id.split("/")[-1]))
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"repo_id={repo_id}")
    print(f"local_dir={local_dir}")
    if "meta-llama" in repo_id.lower() and not args.token:
        print("WARNING: Llama models are gated. Set HF_TOKEN (run `source env/load_env.sh`) if this fails.")

    snapshot_download(repo_id=repo_id, revision=args.revision,
                      local_dir=str(local_dir), token=args.token)
    print(f"Done. Run with MODEL_ID={local_dir}")


if __name__ == "__main__":
    main()
