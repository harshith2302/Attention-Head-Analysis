#!/bin/bash
# Next-level configs targeting MoH

# Navigate to your working directory safely
cd "/home/priya/cot_riddlesense/R_and_D_on_Param(17B)_vs_Llama(70B)/Ajay/Attention-Head-Analysis/exp1"

MODEL="./model"
GR="./results/global_ranking.csv"
TR="./task_rankings"
OUT="./eval_results"

echo "============================================"
echo "Next-level configs targeting MoH"
echo "Started: $(date)"
echo "============================================"

# ── C1: 14 free + 12S+4T ──
echo -e "\n>>> C1: 14 free + 12S+4T (71.9% heads)"
python code/head_masked_eval_patched.py \
    --model_path $MODEL --global_ranking $GR --task_rankings_dir $TR \
    --n_shared 12 --m_task 4 --free_layers 14 --output_dir $OUT --batch_size 4

# ── C2: 16 free + 12S+4T ──
echo -e "\n>>> C2: 16 free + 12S+4T (75% heads — same budget as MoH)"
python code/head_masked_eval_patched.py \
    --model_path $MODEL --global_ranking $GR --task_rankings_dir $TR \
    --n_shared 12 --m_task 4 --free_layers 16 --output_dir $OUT --batch_size 4

# ── C3: 12 free + 16S+8T ──
echo -e "\n>>> C3: 12 free + 16S+8T (84.4% heads)"
python code/head_masked_eval_patched.py \
    --model_path $MODEL --global_ranking $GR --task_rankings_dir $TR \
    --n_shared 16 --m_task 8 --free_layers 12 --output_dir $OUT --batch_size 4

# ── C4: 16 free + 16S+8T ──
echo -e "\n>>> C4: 16 free + 16S+8T (87.5% heads)"
python code/head_masked_eval_patched.py \
    --model_path $MODEL --global_ranking $GR --task_rankings_dir $TR \
    --n_shared 16 --m_task 8 --free_layers 16 --output_dir $OUT --batch_size 4

# ── C5: 12 free + 12S+8T ──
echo -e "\n>>> C5: 12 free + 12S+8T (76.6% heads, more task-specific)"
python code/head_masked_eval_patched.py \
    --model_path $MODEL --global_ranking $GR --task_rankings_dir $TR \
    --n_shared 12 --m_task 8 --free_layers 12 --output_dir $OUT --batch_size 4

# ── C6: 12 free + 8S+12T ──
echo -e "\n>>> C6: 12 free + 8S+12T (76.6%, heavier task weight)"
python code/head_masked_eval_patched.py \
    --model_path $MODEL --global_ranking $GR --task_rankings_dir $TR \
    --n_shared 8 --m_task 12 --free_layers 12 --output_dir $OUT --batch_size 4

echo -e "\n============================================"
echo "Done: $(date)"
echo "============================================"