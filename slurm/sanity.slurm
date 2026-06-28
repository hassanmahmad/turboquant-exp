#!/bin/bash
#SBATCH --job-name=tqsec-sanity
#SBATCH --account=IscrC_VisLLMs
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/sanity-%j.out
#SBATCH --error=logs/sanity-%j.err
set -euo pipefail

module load python/3.11.7
module load cuda/12.2

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/turboquant-exp}"
cd "$PROJECT_ROOT"
mkdir -p logs

source "${VENV_DIR:-$SCRATCH/venvs/tqsec}/bin/activate"
export PYTHONUNBUFFERED=1
# compute nodes have no internet -> load the pre-downloaded model offline
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1

export MODEL_ID="${MODEL_ID:-$SCRATCH/models/qwen2.5-7b-instruct}"
export MODEL_TAG="${MODEL_TAG:-$(basename "$MODEL_ID")}"
export OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/results/sanity/$MODEL_TAG}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-24}"
mkdir -p "$OUTPUT_DIR"

echo "MODEL_ID=$MODEL_ID  MODEL_TAG=$MODEL_TAG  OUTPUT_DIR=$OUTPUT_DIR"
python t1_characterization/run_sanity.py
