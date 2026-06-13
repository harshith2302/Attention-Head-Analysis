#!/bin/bash
# Strategy A: N shared + M task (all layers)

# Navigate to your current working directory
cd "/home/priya/cot_riddlesense/R_and_D_on_Param(17B)_vs_Llama(70B)/Ajay/Attention-Head-Analysis/exp1"

MODEL="./model"
GR="./results/global_ranking.csv"
TR="./task_rankings"
OUT="./eval_results"

echo "============================================"
echo "Strategy A: N shared + M task (all layers)"
echo "Started: $(date)"
echo "============================================"

# ── Config 1: 16S + 8T = 24/32 (75%) — matches MoH budget, balanced ──
echo -e "\n>>> A1: 16 shared + 8 task = 24 heads (75%)"
python code/head_masked_eval.py \
    --model_path $MODEL --global_ranking $GR --task_rankings_dir $TR \
    --n_shared 16 --m_task 8 --output_dir $OUT --batch_size 4

# ── Config 2: 12S + 12T = 24/32 (75%) — more task-specific ──
echo -e "\n>>> A2: 12 shared + 12 task = 24 heads (75%)"
python code/head_masked_eval.py \
    --model_path $MODEL --global_ranking $GR --task_rankings_dir $TR \
    --n_shared 12 --m_task 12 --output_dir $OUT --batch_size 4

# ── Config 3: 20S + 4T = 24/32 (75%) — more shared ──
echo -e "\n>>> A3: 20 shared + 4 task = 24 heads (75%)"
python code/head_masked_eval.py \
    --model_path $MODEL --global_ranking $GR --task_rankings_dir $TR \
    --n_shared 20 --m_task 4 --output_dir $OUT --batch_size 4

# ── Config 4: 8S + 8T = 16/32 (50%) — aggressive pruning ──
echo -e "\n>>> A4: 8 shared + 8 task = 16 heads (50%)"
python code/head_masked_eval.py \
    --model_path $MODEL --global_ranking $GR --task_rankings_dir $TR \
    --n_shared 8 --m_task 8 --output_dir $OUT --batch_size 4

# ── Config 5: 24S + 0T = 24/32 (75%) — ablation, shared only ──
echo -e "\n>>> A5: 24 shared + 0 task = 24 heads (75%, no task-specific)"
python code/head_masked_eval.py \
    --model_path $MODEL --global_ranking $GR --task_rankings_dir $TR \
    --n_shared 24 --m_task 0 --output_dir $OUT --batch_size 4

echo -e "\n============================================"
echo "Strategy A done: $(date)"
echo "============================================"
ls -lh $OUT/eval_s*.json