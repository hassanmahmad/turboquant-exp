#!/bin/bash
# Submit the sanity benchmark for one model.
# Usage: ./scripts/submit_sanity.sh qwen2.5-7b-instruct $SCRATCH/models/qwen2.5-7b-instruct
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <model_tag> <model_dir>"
  echo "Example: $0 qwen2.5-7b-instruct \$SCRATCH/models/qwen2.5-7b-instruct"
  exit 1
fi

MODEL_TAG="$1"
MODEL_DIR="$2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="$PROJECT_ROOT/results/sanity/$MODEL_TAG"

if [[ ! -d "$MODEL_DIR" ]]; then
  echo "Model directory does not exist: $MODEL_DIR" >&2
  echo "Download it first: python scripts/download_model.py --preset <preset>" >&2
  exit 2
fi

cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_DIR" logs

jid=$(sbatch --parsable \
  --job-name="${MODEL_TAG}-sanity" \
  --export=ALL,PROJECT_ROOT="$PROJECT_ROOT",MODEL_ID="$MODEL_DIR",MODEL_TAG="$MODEL_TAG",OUTPUT_DIR="$OUTPUT_DIR" \
  slurm/sanity.slurm)

echo "SANITY_JOB=$jid  (model=$MODEL_TAG)"
echo "Watch:   squeue -j $jid"
echo "Results: $OUTPUT_DIR/sanity_benchmark.json"
