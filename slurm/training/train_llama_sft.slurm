#!/bin/bash
#SBATCH --job-name=llama_r8_1e-5
#SBATCH --output=/ibex/scratch/zahrmm0b/safety-overrefusal-project/logs/llama_r8_1e-5_%j.out
#SBATCH --error=/ibex/scratch/zahrmm0b/safety-overrefusal-project/logs/llama_r8_1e-5_%j.err
#SBATCH --time=8:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100:1
#SBATCH --constraint=v100

echo "========================================="
echo "Job: llama_r8_1e-5"
echo "Started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "========================================="

# Load modules
module purge
module load cuda/11.8
module load gcc/11.2.0

# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llama_factory

# Set environment variables
export HF_HOME=/ibex/scratch/$USER/.cache/huggingface
export HF_TOKEN=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
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export CUDA_VISIBLE_DEVICES=0

# Show GPU info
nvidia-smi

echo ""
echo "========================================="
echo "Starting training: llama (rank=8, lr=1e-5)"
echo "Config: /ibex/scratch/zahrmm0b/safety-overrefusal-project/configs/safety_sft_llama_rank8_lr1e-5.yaml"
echo "========================================="

# Train
cd ~/LLaMA-Factory
llamafactory-cli train /ibex/scratch/zahrmm0b/safety-overrefusal-project/configs/safety_sft_llama_rank8_lr1e-5.yaml

EXIT_CODE=$?

echo ""
echo "========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Training completed successfully"
else
    echo "✗ Training failed (exit code: $EXIT_CODE)"
fi
echo "Ended: $(date)"
echo "========================================="

exit $EXIT_CODE
