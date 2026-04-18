#!/bin/bash
#SBATCH --job-name=moh_test
#SBATCH --output=/home/cccp/25m0834/RND3/logs/test_%j.out
#SBATCH --error=/home/cccp/25m0834/RND3/logs/test_%j.err
#SBATCH --partition=dgx
#SBATCH --qos=dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00

eval "$(conda shell.bash hook)" && conda activate moh

cd /home/cccp/25m0834/RND3/code

echo "=== QUICK TEST ==="
echo "Job: ${SLURM_JOB_ID} | Node: $(hostname)"
echo "Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python head_importance_analysis.py \
    --model_path /home/cccp/25m0834/RND3/model \
    --datasets_dir /home/cccp/25m0834/RND3/datasets \
    --output_dir /home/cccp/25m0834/RND3/results_test \
    --samples_per_task 100 \
    --batch_size 1 \
    --max_seq_len 256 \
    --device cuda \
    --topk_tokens 5 \
    --no_gradient \
    --dtype bfloat16

echo "Exit: $? | Done: $(date)"
ls -lh /home/cccp/25m0834/RND3/results_test/