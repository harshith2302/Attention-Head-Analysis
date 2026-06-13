#!/bin/bash
# Batch Sweep for Shared Heads = 8 (No Free Layers)

# Navigate to your working directory safely
cd "/home/priya/cot_riddlesense/R_and_D_on_Param(17B)_vs_Llama(70B)/Ajay/Attention-Head-Analysis/exp1"

MODEL="./model"
GR="./results/global_ranking.csv"
TR="./task_rankings"
OUT="./eval_results"

echo "============================================"
echo "Starting S8 Sweep (Task Specific = 12, 14, 16 | No Free Layers)"
echo "Started: $(date)"
echo "============================================"

SHARED_HEADS=8

# Loop through the variations of task-specific heads
for TASK_HEADS in 12 14 16; do
    
    echo -e "\n>>> Now Running Configuration: Shared=$SHARED_HEADS, Task=$TASK_HEADS, Free=0"
    
    # Execute the evaluation script EXACTLY like your original successful runs
    python code/head_masked_eval_patched.py \
        --model_path $MODEL \
        --global_ranking $GR \
        --task_rankings_dir $TR \
        --n_shared $SHARED_HEADS \
        --m_task $TASK_HEADS \
        --free_layers 0 \
        --output_dir $OUT \
        --batch_size 4

    echo "✔ Finished Configuration: Shared=$SHARED_HEADS, Task=$TASK_HEADS"
    echo "--------------------------------------------"
done

echo -e "\n============================================"
echo "🎉 All S8 Experiments Completed: $(date)"
echo "============================================"