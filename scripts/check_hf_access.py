"""Check Hugging Face gated-model access (e.g. before downloading Llama 3.1).

  source env/load_env.sh
  python scripts/check_hf_access.py meta-llama/Llama-3.1-8B-Instruct
"""

import argparse
import os

from huggingface_hub import model_info


def main():
    parser = argparse.ArgumentParser(description="Check HF gated-model access.")
    parser.add_argument("repo_id", help="Hugging Face repository id.")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is not set. Run: source env/load_env.sh")

    info = model_info(args.repo_id, token=token)
    print(f"ACCESS_OK repo_id={info.id}")


if __name__ == "__main__":
    main()
