#!/bin/bash
# Winner-tasks: scan to find the SWEET SPOT

# Navigate to your working directory safely
cd "/home/priya/cot_riddlesense/R_and_D_on_Param(17B)_vs_Llama(70B)/Ajay/Attention-Head-Analysis/exp1"

MODEL="./model"
GR="./results/global_ranking.csv"
TR="./task_rankings"
OUT="./eval_results_winners"

TARGET_TASKS="sciq piqa winogrande boolq"

echo "============================================"
echo "Winner-tasks: scan to find the SWEET SPOT"
echo "Started: $(date)"
echo "Tasks: $TARGET_TASKS"
echo "============================================"

# ── W1 ──
echo -e "\n>>> W1: 14 free + 12S+4T"
python code/head_masked_eval_patched.py \
    --model_path $MODEL --global_ranking $GR --task_rankings_dir $TR \
    --n_shared 12 --m_task 4 --free_layers 14 --output_dir $OUT \
    --tasks $TARGET_TASKS --batch_size 4

# ── W2 ──
echo -e "\n>>> W2: 16 free + 12S+4T"
python code/head_masked_eval_patched.py \
    --model_path $MODEL --global_ranking $GR --task_rankings_dir $TR \
    --n_shared 12 --m_task 4 --free_layers 16 --output_dir $OUT \
    --tasks $TARGET_TASKS --batch_size 4

# ── W3 ──
echo -e "\n>>> W3: 12 free + 16S+8T"
python code/head_masked_eval_patched.py \
    --model_path $MODEL --global_ranking $GR --task_rankings_dir $TR \
    --n_shared 16 --m_task 8 --free_layers 12 --output_dir $OUT \
    --tasks $TARGET_TASKS --batch_size 4

# ── W4 ──
echo -e "\n>>> W4: 14 free + 16S+8T"
python code/head_masked_eval_patched.py \
    --model_path $MODEL --global_ranking $GR --task_rankings_dir $TR \
    --n_shared 16 --m_task 8 --free_layers 14 --output_dir $OUT \
    --tasks $TARGET_TASKS --batch_size 4

# ── W5: heavy task-specific ──
echo -e "\n>>> W5: 12 free + 8S+16T (very task-specific)"
python code/head_masked_eval_patched.py \
    --model_path $MODEL --global_ranking $GR --task_rankings_dir $TR \
    --n_shared 8 --m_task 16 --free_layers 12 --output_dir $OUT \
    --tasks $TARGET_TASKS --batch_size 4

# ── W6: maximal info ──
echo -e "\n>>> W6: 16 free + 16S+8T (87.5% — closest to MoH 75%)"
python code/head_masked_eval_patched.py \
    --model_path $MODEL --global_ranking $GR --task_rankings_dir $TR \
    --n_shared 16 --m_task 8 --free_layers 16 --output_dir $OUT \
    --tasks $TARGET_TASKS --batch_size 4

echo -e "\n============================================"
echo "Done: $(date)"
echo "============================================"