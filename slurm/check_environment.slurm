#!/bin/bash
#SBATCH --job-name=tqsec-env
#SBATCH --account=IscrC_VisLLMs
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/env-%j.out
#SBATCH --error=logs/env-%j.err
set -euo pipefail

module load python/3.11.7
module load cuda/12.2

PROJECT_ROOT="${PROJECT_ROOT:-$HOME/turboquant-exp}"
cd "$PROJECT_ROOT"
mkdir -p logs
source "${VENV_DIR:-$SCRATCH/venvs/tqsec}/bin/activate"
export PYTHONUNBUFFERED=1

python scripts/check_leonardo_env.py
