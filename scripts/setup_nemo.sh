#!/bin/bash
set -euo pipefail

cd /users/PAS2030/srishti/asr_fairness

echo "Loading cluster Python 3.10 module..."
module load python/3.10-4.11 || module load python/3.10 || true

echo "Creating new venv_nemo..."
python3 -m venv venv_nemo
source venv_nemo/bin/activate

echo "Updating pip and installing PyTorch..."
pip install --upgrade pip

echo "Installing NeMo from main branch..."
pip install cython
pip install "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git"

echo "NeMo installation complete!"
