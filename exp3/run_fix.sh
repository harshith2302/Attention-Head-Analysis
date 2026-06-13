#!/bin/bash
# Dedicated script to bypass the dataclass bug on GPU 2

cd "/home/priya/cot_riddlesense/R_and_D_on_Param(17B)_vs_Llama(70B)/Ajay/Attention-Head-Analysis/exp1"

echo "============================================"
echo "Starting Targeted Fix for HellaSwag/ARC/TruthfulQA"
echo "Started: $(date)"
echo "============================================"

python code/fix_eager.py \
    --model_path ./model \
    --global_ranking ./results/global_ranking.csv \
    --task_rankings_dir ./task_rankings \
    --json_dir ./eval_results \
    --batch_size 4

echo -e "\n============================================"
echo "Fixes Done: $(date)"
echo "============================================"