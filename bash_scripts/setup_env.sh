#!/bin/bash
# Create a conda environment and install FAR with the exact package versions of the paper
# (Python 3.12, PyTorch 2.7 with CUDA 12.8 wheels).
#   bash bash_scripts/setup_env.sh            # environment named "far"
#   bash bash_scripts/setup_env.sh myenv      # another name
# Then `conda activate far` (or put its bin/ on PATH in bash_scripts/env.local.sh, see
# env.local.sh.example, so every launcher uses it).
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
ENV=${1:-far}
command -v conda >/dev/null || { echo "conda not found: install Miniconda / Miniforge first"; exit 1; }
source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -n "$ENV" python=3.12 -y
conda activate "$ENV"
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e .
python -c "import torch, far; print('far from', far.__file__, '| torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
echo "done. activate with: conda activate $ENV"
