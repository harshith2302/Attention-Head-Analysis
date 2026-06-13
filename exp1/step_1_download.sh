#!/bin/bash
#SBATCH --job-name=exp1_download
#SBATCH --output=/home/cccp/25m0834/Attention-Head-Analysis/exp1/logs/download_%j.out
#SBATCH --error=/home/cccp/25m0834/Attention-Head-Analysis/exp1/logs/download_%j.err
#SBATCH --partition=dgx
#SBATCH --qos=dgx
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00

eval "$(conda shell.bash hook)" && conda activate moh
cd /home/cccp/25m0834/Attention-Head-Analysis/exp1

echo "=== Downloading LLaMA3-8B ==="
python code/download_model.py --output_dir ./model

echo "=== Downloading Datasets ==="
python code/download_datasets.py --output_dir ./datasets

echo "=== Done: $(date) ==="
ls -lh model/ datasets/