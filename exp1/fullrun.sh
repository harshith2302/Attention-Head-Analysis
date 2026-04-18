#!/bin/bash
#SBATCH --job-name=moh_full
#SBATCH --output=/home/cccp/25m0834/RND3/logs/full_%j.out
#SBATCH --error=/home/cccp/25m0834/RND3/logs/full_%j.err
#SBATCH --partition=dgx
#SBATCH --qos=dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=12:00:00

eval "$(conda shell.bash hook)" && conda activate moh

cd /home/cccp/25m0834/RND3/code

echo "============================================"
echo "MoH Head Analysis — FULL RUN (A100 80GB)"
echo "Job: ${SLURM_JOB_ID} | Node: $(hostname)"
echo "Start: $(date)"
echo "============================================"
nvidia-smi

python head_importance_analysis.py \
    --model_path /home/cccp/25m0834/RND3/model \
    --datasets_dir /home/cccp/25m0834/RND3/datasets \
    --output_dir /home/cccp/25m0834/RND3/results \
    --samples_per_task 0 \
    --batch_size 1 \
    --max_seq_len 512 \
    --device cuda \
    --topk_tokens 10 \
    --token_pool_method topk \
    --dtype bfloat16 \
    2>&1 | tee /home/cccp/25m0834/RND3/results/run_log.txt

echo "============================================"
echo "Exit: $? | Done: $(date)"
echo "============================================"
ls -lh /home/cccp/25m0834/RND3/results/