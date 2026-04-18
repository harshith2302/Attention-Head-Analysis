#!/bin/bash
set -e

ENV_NAME="dhp"

echo "═══════════════════════════════════════════════════"
echo "  Creating conda environment: ${ENV_NAME}"
echo "═══════════════════════════════════════════════════"

eval "$(conda shell.bash hook)"
conda deactivate 2>/dev/null || true
conda env remove -n ${ENV_NAME} -y 2>/dev/null || true
conda create -n ${ENV_NAME} python=3.10 -y
conda activate ${ENV_NAME}

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.44.0 accelerate datasets tokenizers sentencepiece protobuf
pip install lm-eval==0.4.4
pip install matplotlib numpy pandas tqdm scipy

echo ""
echo "═══════════════════════════════════════════════════"
echo "  ✓ Conda env '${ENV_NAME}' created"
echo "  To activate: conda activate ${ENV_NAME}"
echo "  Next: bash setup/step1_download.sh"
echo "═══════════════════════════════════════════════════"
