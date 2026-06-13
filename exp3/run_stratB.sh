#!/bin/bash
# Strategy B: First X layers FREE, rest masked

# Navigate to your current working directory
cd "/home/priya/cot_riddlesense/R_and_D_on_Param(17B)_vs_Llama(70B)/Ajay/Attention-Head-Analysis/exp1"

MODEL="./model"
GR="./results/global_ranking.csv"
TR="./task_rankings"
OUT="./eval_results"

echo "============================================"
echo "Strategy B: First X layers FREE, rest masked"
echo "Started: $(date)"
echo "============================================"

# ── B1: 6 free layers + 12S+4T on remaining 26 layers ──
echo -e "\n>>> B1: 6 free + 12S+4T on layers 6-31"
python code/head_masked_eval.py \
    --model_path $MODEL --global_ranking $GR --task_rankings_dir $TR \
    --n_shared 12 --m_task 4 --free_layers 6 --output_dir $OUT --batch_size 4

# ── B2: 8 free layers + 12S+4T on remaining 24 layers ──
echo -e "\n>>> B2: 8 free + 12S+4T on layers 8-31"
python code/head_masked_eval.py \
    --model_path $MODEL --global_ranking $GR --task_rankings_dir $TR \
    --n_shared 12 --m_task 4 --free_layers 8 --output_dir $OUT --batch_size 4

# ── B3: 10 free layers + 12S+4T on remaining 22 layers ──
echo -e "\n>>> B3: 10 free + 12S+4T on layers 10-31"
python code/head_masked_eval.py \
    --model_path $MODEL --global_ranking $GR --task_rankings_dir $TR \
    --n_shared 12 --m_task 4 --free_layers 10 --output_dir $OUT --batch_size 4

# ── B4: 12 free layers + 12S+4T on remaining 20 layers ──
echo -e "\n>>> B4: 12 free + 12S+4T on layers 12-31"
python code/head_masked_eval.py \
    --model_path $MODEL --global_ranking $GR --task_rankings_dir $TR \
    --n_shared 12 --m_task 4 --free_layers 12 --output_dir $OUT --batch_size 4

# ── B5: 8 free layers + 16S+8T on remaining 24 layers (higher budget) ──
echo -e "\n>>> B5: 8 free + 16S+8T on layers 8-31"
python code/head_masked_eval.py \
    --model_path $MODEL --global_ranking $GR --task_rankings_dir $TR \
    --n_shared 16 --m_task 8 --free_layers 8 --output_dir $OUT --batch_size 4

echo -e "\n============================================"
echo "Strategy B done: $(date)"
echo "============================================"
ls -lh $OUT/eval_s*_f*.json