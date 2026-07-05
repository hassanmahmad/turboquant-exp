#!/bin/bash
#SBATCH --job-name=tqsec-t3-leakage
#SBATCH --account=IscrC_VisLLMs
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/t3-leakage-%j.out
#SBATCH --error=logs/t3-leakage-%j.err
set -euo pipefail

module load python/3.11.7
module load cuda/12.2

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/turboquant-exp}"
cd "$PROJECT_ROOT"
mkdir -p logs

source "${VENV_DIR:-$SCRATCH/venvs/tqsec}/bin/activate"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1

export MODEL_ID="${MODEL_ID:-$SCRATCH/models/tinyllama-1.1b-chat-v1.0}"
export MODEL_TAG="${MODEL_TAG:-$(basename "$MODEL_ID")}"
export OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/results/t3_leakage/$MODEL_TAG}"
export T3_EPOCHS="${T3_EPOCHS:-200}"
mkdir -p "$OUTPUT_DIR"

echo "MODEL_ID=$MODEL_ID MODEL_TAG=$MODEL_TAG OUTPUT_DIR=$OUTPUT_DIR"
python t3_leakage/run_leakage.py
