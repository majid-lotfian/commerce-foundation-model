#!/bin/bash
#SBATCH --job-name=rating-eval
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/rating_eval_%j.out
#SBATCH --error=logs/rating_eval_%j.err

set -euo pipefail

cd /gpfs/work1/0/prjs1732/LabGPT/Fine-Tunning/commerce-foundation-model

# Activate the existing Conda environment.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate commerce-fm

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Experiment: $1"
echo "Adapter: $2"
echo "Precision: $3"

nvidia-smi

python -u src/evaluation/evaluate_rating.py \
    --name "$1" \
    --adapter "$2" \
    --precision "$3" \
    --output-dir "results/$1"
