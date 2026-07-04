#!/bin/bash
# Set up the Python environment for turboquant-exp on Leonardo/CINECA.
# Run once on a LOGIN node (compute nodes have no internet).
#
#   ./env/setup_leonardo_env.sh
#   source $SCRATCH/venvs/tqsec/bin/activate
set -euo pipefail

module load python/3.11.7
module load cuda/12.2

VENV_DIR="${VENV_DIR:-$SCRATCH/venvs/tqsec}"
python -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
# CUDA 12.1 wheels are compatible with the cuda/12.2 module on Leonardo (A100).
python -m pip install --upgrade --index-url https://download.pytorch.org/whl/cu121 torch
python -m pip install --upgrade \
  transformers \
  accelerate \
  safetensors \
  sentencepiece \
  protobuf \
  huggingface_hub \
  numpy \
  scipy \
  datasets \
  pytest

echo
echo "Environment ready at: $VENV_DIR"
echo "Activate with:  source $VENV_DIR/bin/activate"
echo "Sanity-check the stack with:  python scripts/check_leonardo_env.py"
echo
echo "NOTE vs the predecessor: added scipy (research layer needs it), datasets (LongBench), pytest."
