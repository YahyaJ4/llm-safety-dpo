#!/bin/bash
#SBATCH --job-name=eval_llama_dpo_beta0.1_advbench
#SBATCH --output=/ibex/scratch/zahrmm0b/safety-overrefusal-project/logs/eval_llama_dpo_beta0.1_advbench_%j.out
#SBATCH --error=/ibex/scratch/zahrmm0b/safety-overrefusal-project/logs/eval_llama_dpo_beta0.1_advbench_%j.err
#SBATCH --time=2:00:00
#SBATCH --gpus=1
#SBATCH --constraint=a100
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4

echo "=============================================="
echo "eval_llama_dpo_beta0.1_advbench"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"
echo "=============================================="

# Set up environment
source ~/.bashrc
conda activate llama_factory

# HuggingFace setup
export HF_HOME=/ibex/scratch/zahrmm0b/hf_cache
export HF_DATASETS_CACHE=/ibex/scratch/zahrmm0b/hf_cache/datasets
export HF_TOKEN=$(cat ~/.huggingface/token)
export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
mkdir -p $HF_HOME $HF_DATASETS_CACHE

cd /ibex/scratch/zahrmm0b/safety-overrefusal-project
python scripts/eval_per_model/eval_llama_dpo_beta0.1_advbench.py

echo "=============================================="
echo "End: $(date)"
echo "=============================================="
