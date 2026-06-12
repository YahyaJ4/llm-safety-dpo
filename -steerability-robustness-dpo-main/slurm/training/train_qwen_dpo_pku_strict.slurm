#!/bin/bash
#SBATCH --job-name=qwen_dpo_pku
#SBATCH --output=logs/qwen_dpo_pku_%A_%a.out
#SBATCH --error=logs/qwen_dpo_pku_%A_%a.err
#SBATCH --time=6:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=batch
#SBATCH --constraint=a100
#SBATCH --array=0-4

cd /ibex/scratch/zahrmm0b/safety-overrefusal-project

source ~/.bashrc
module load cuda/12.2
conda activate llama_factory

# Use scratch for ALL caches to avoid quota issues
export HF_HOME=/ibex/scratch/zahrmm0b/safety-overrefusal-project/hf_cache
export HF_DATASETS_CACHE=/ibex/scratch/zahrmm0b/safety-overrefusal-project/hf_cache/datasets
export TRANSFORMERS_CACHE=/ibex/scratch/zahrmm0b/safety-overrefusal-project/hf_cache/hub
export PYTHONUNBUFFERED=1

# Create cache dirs
mkdir -p $HF_HOME $HF_DATASETS_CACHE $TRANSFORMERS_CACHE

# Beta values matching array indices
BETAS=(0.01 0.05 0.1 0.5 1.0)
BETA=${BETAS[$SLURM_ARRAY_TASK_ID]}

echo "=========================================="
echo "Training Qwen DPO with beta=${BETA}"
echo "Array task ID: ${SLURM_ARRAY_TASK_ID}"
echo "=========================================="

CONFIG_FILE="configs_qwen_pku_strict/dpo_qwen_pku_beta${BETA}.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

cd ~/LLaMA-Factory

llamafactory-cli train /ibex/scratch/zahrmm0b/safety-overrefusal-project/${CONFIG_FILE}

echo "Training complete for beta=${BETA}"
