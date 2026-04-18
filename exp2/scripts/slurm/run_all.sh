#!/bin/bash
#SBATCH --job-name=dhp-master
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=/home/cccp/25m0834/RND5/logs/master_%j.out
#SBATCH --error=/home/cccp/25m0834/RND5/logs/master_%j.err

# ═══════════════════════════════════════════════════════════
# ONE SLURM JOB — RUNS EVERYTHING
# 100 experiments + 11 benchmarks each + all plots
# Submit: sbatch scripts/slurm/run_all.sh
# ═══════════════════════════════════════════════════════════

set -e

PROJECT_DIR="/home/cccp/25m0834/RND5"

echo "═══════════════════════════════════════════════════"
echo "  Dynamic Head Pruning — MASTER RUN"
echo "  Node:      $(hostname)"
echo "  GPU:       $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo "  Job ID:    ${SLURM_JOB_ID}"
echo "  Started:   $(date)"
echo "═══════════════════════════════════════════════════"

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

eval "$(conda shell.bash hook)"
conda activate dhp

cd "$PROJECT_DIR"
python3 scripts/run_all.py

echo ""
echo "═══ COMPLETE $(date) ═══"
echo "Results: ${PROJECT_DIR}/results/plots/"
