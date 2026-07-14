#!/bin/bash
#SBATCH --job-name=tqsec-t2-backdoor
#SBATCH --account=IscrC_VisLLMs
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/t2-backdoor-%j.out
#SBATCH --error=logs/t2-backdoor-%j.err
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
export OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/results/t2_behavior/$MODEL_TAG-backdoor}"
export T2_STEPS="${T2_STEPS:-60}"
export T2_TRAIN_PI="${T2_TRAIN_PI:-public}"
export KEY_BITS="${KEY_BITS:-3}" VALUE_BITS="${VALUE_BITS:-4}"
mkdir -p "$OUTPUT_DIR"

echo "MODEL_ID=$MODEL_ID TRAIN_PI=$T2_TRAIN_PI OUTPUT_DIR=$OUTPUT_DIR"
python t2_behavior/train_backdoor.py