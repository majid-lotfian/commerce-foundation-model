#!/bin/bash
#SBATCH --job-name=rating-bf16
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=25:00:00
#SBATCH --output=logs/rating_bf16_%j.out
#SBATCH --error=logs/rating_bf16_%j.err

set -euo pipefail

cd /gpfs/work1/0/prjs1732/LabGPT/Fine-Tunning/commerce-foundation-model

# Activate the existing Conda environment.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate commerce-fm

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
nvidia-smi

python -u src/pretraining/train_rating_sft.py \
    --experiment base \
    --precision bf16 \
    --output-dir outputs/rating_sft_bf16 \
    --max-length 3072 \
    --epochs 1 \
    --batch-size 1 \
    --grad-accum 16 \
    --learning-rate 1e-4 \
    --eval-steps 1000 \
    --save-steps 200 \
    --lora-rank 16 \
    --lora-alpha 32