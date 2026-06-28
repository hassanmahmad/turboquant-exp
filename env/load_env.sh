#!/bin/bash
# Source this to load private values (HF token, model dirs) from .env:
#   source env/load_env.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ENV_FILE="${1:-$PROJECT_ROOT/.env}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "No .env file found at: $ENV_FILE" >&2
  echo "Create one with: cp env/.env.example .env && nano .env" >&2
  return 1 2>/dev/null || exit 1
fi

set -a
source "$ENV_FILE"
set +a

echo "Loaded environment from $ENV_FILE"
