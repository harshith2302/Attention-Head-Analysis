#!/bin/bash
set -e

# ╔═══════════════════════════════════════════════════════════╗
# ║  PASTE YOUR HUGGING FACE TOKEN BELOW                     ║
# ╚═══════════════════════════════════════════════════════════╝
export HF_TOKEN=""
# ═══════════════════════════════════════════════════════════

PROJECT_DIR="/home/cccp/25m0834/RND5"
MODEL_CACHE="${PROJECT_DIR}/cache/models"
DATASET_CACHE="${PROJECT_DIR}/cache/datasets"

echo "═══════════════════════════════════════════════════"
echo "  STEP 1: Download Model & Datasets"
echo "═══════════════════════════════════════════════════"

eval "$(conda shell.bash hook)"
conda activate dhp

mkdir -p "$MODEL_CACHE" "$DATASET_CACHE" "${PROJECT_DIR}/results" "${PROJECT_DIR}/logs"

echo ""
echo "[1/2] Downloading LLaMA3-8B (~16GB, takes 15-30 min)..."
python3 -c "
import torch, os
from transformers import AutoModelForCausalLM, AutoTokenizer
token = os.environ['HF_TOKEN']
cache = '${MODEL_CACHE}'
name = 'meta-llama/Meta-Llama-3-8B'
print('  Tokenizer...')
AutoTokenizer.from_pretrained(name, cache_dir=cache, token=token, trust_remote_code=True)
print('  Model weights...')
m = AutoModelForCausalLM.from_pretrained(name, cache_dir=cache, token=token, trust_remote_code=True, torch_dtype=torch.float16)
print(f'  Done: {sum(p.numel() for p in m.parameters())/1e9:.2f}B params')
del m
"

echo ""
echo "[2/2] Downloading 11 benchmark datasets..."
python3 -c "
from datasets import load_dataset
cache = '${DATASET_CACHE}'
for name, path, sub, sp in [
    ('piqa','piqa',None,'validation'), ('hellaswag','Rowan/hellaswag',None,'validation'),
    ('winogrande','winogrande','winogrande_xl','validation'), ('boolq','google/boolq',None,'validation'),
    ('sciq','allenai/sciq',None,'validation'), ('arc_challenge','allenai/ai2_arc','ARC-Challenge','test'),
    ('lambada','EleutherAI/lambada_openai',None,'test'), ('truthfulqa','truthful_qa','multiple_choice','validation'),
    ('gsm8k','gsm8k','main','test'), ('mmlu','cais/mmlu','all','test'),
    ('logiqa','EleutherAI/logiqa',None,'test'),
]:
    try:
        kw = dict(cache_dir=cache, trust_remote_code=True)
        ds = load_dataset(path, sub, split=sp, **kw) if sub else load_dataset(path, split=sp, **kw)
        print(f'  OK {name:20s} {len(ds):>6d} samples')
    except Exception as e:
        print(f'  FAIL {name:20s} {e}')
"

echo ""
echo "Cache sizes:"
echo "  Model:   $(du -sh ${MODEL_CACHE} 2>/dev/null | cut -f1)"
echo "  Dataset: $(du -sh ${DATASET_CACHE} 2>/dev/null | cut -f1)"
echo ""
echo "═══════════════════════════════════════════════════"
echo "  STEP 1 COMPLETE"
echo "  NEXT: sbatch scripts/slurm/run_all.sh"
echo "═══════════════════════════════════════════════════"
