#!/bin/bash
#SBATCH --job-name=llama_dpo_b0.1
#SBATCH --output=/ibex/scratch/zahrmm0b/safety-overrefusal-project/logs/dpo_llama_beta0.1_%j.out
#SBATCH --error=/ibex/scratch/zahrmm0b/safety-overrefusal-project/logs/dpo_llama_beta0.1_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=batch

# Load modules
module purge
module load cuda/11.8

# Set environment variables
export HF_HOME=/ibex/scratch/$USER/.cache/huggingface
export HF_TOKEN=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
export TRANSFORMERS_CACHE=/ibex/scratch/$USER/.cache/transformers
export TORCH_HOME=/ibex/scratch/$USER/.cache/torch

# Create logs directory
mkdir -p /ibex/scratch/zahrmm0b/safety-overrefusal-project/logs

# Activate conda environment
source ~/.bashrc
conda activate llama_factory

# Print environment info
echo "======================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "======================================"
echo ""
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")')"
echo ""
echo "======================================"

# Change to LLaMA-Factory directory
cd ~/LLaMA-Factory

# Run DPO training
llamafactory-cli train \
    /ibex/scratch/zahrmm0b/safety-overrefusal-project/configs/dpo_llama_beta0.1.yaml

echo ""
echo "======================================"
echo "End Time: $(date)"
echo "======================================"
