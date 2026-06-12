#!/bin/bash
# Submit all 36 per-model evaluation jobs

echo "=============================================="
echo "SUBMITTING 36 PER-MODEL EVALUATION JOBS"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  AdvBench: 520 samples (full)"
echo "  MMLU: 1000 samples"
echo "  GSM8K: 500 samples"
echo "  Models: 12 (6 Llama + 6 Qwen)"
echo "  Time limit: 2 hours each"
echo ""

cd /ibex/scratch/zahrmm0b/safety-overrefusal-project

JOB_IDS=""

echo "Submitting eval_llama_sft_advbench..."
JOB_ID=$(sbatch slurm/eval_per_model/eval_llama_sft_advbench.slurm | grep -oP "\d+")
echo "  Job ID: $JOB_ID"
JOB_IDS="$JOB_IDS $JOB_ID"

echo "Submitting eval_llama_sft_mmlu..."
JOB_ID=$(sbatch slurm/eval_per_model/eval_llama_sft_mmlu.slurm | grep -oP "\d+")
echo "  Job ID: $JOB_ID"
JOB_IDS="$JOB_IDS $JOB_ID"

echo "Submitting eval_llama_sft_gsm8k..."
JOB_ID=$(sbatch slurm/eval_per_model/eval_llama_sft_gsm8k.slurm | grep -oP "\d+")
echo "  Job ID: $JOB_ID"
JOB_IDS="$JOB_IDS $JOB_ID"

echo "Submitting eval_llama_dpo_beta0.01_advbench..."
JOB_ID=$(sbatch slurm/eval_per_model/eval_llama_dpo_beta0.01_advbench.slurm | grep -oP "\d+")
echo "  Job ID: $JOB_ID"
JOB_IDS="$JOB_IDS $JOB_ID"

echo "Submitting eval_llama_dpo_beta0.01_mmlu..."
JOB_ID=$(sbatch slurm/eval_per_model/eval_llama_dpo_beta0.01_mmlu.slurm | grep -oP "\d+")
echo "  Job ID: $JOB_ID"
JOB_IDS="$JOB_IDS $JOB_ID"

echo "Submitting eval_llama_dpo_beta0.01_gsm8k..."
JOB_ID=$(sbatch slurm/eval_per_model/eval_llama_dpo_beta0.01_gsm8k.slurm | grep -oP "\d+")
echo "  Job ID: $JOB_ID"
JOB_IDS="$JOB_IDS $JOB_ID"

echo "Submitting eval_llama_dpo_beta0.05_advbench..."
JOB_ID=$(sbatch slurm/eval_per_model/eval_llama_dpo_beta0.05_advbench.slurm | grep -oP "\d+")
echo "  Job ID: $JOB_ID"
JOB_IDS="$JOB_IDS $JOB_ID"

echo "Submitting eval_llama_dpo_beta0.05_mmlu..."
JOB_ID=$(sbatch slurm/eval_per_model/eval_llama_dpo_beta0.05_mmlu.slurm | grep -oP "\d+")
echo "  Job ID: $JOB_ID"
JOB_IDS="$JOB_IDS $JOB_ID"

echo "Submitting eval_llama_dpo_beta0.05_gsm8k..."
JOB_ID=$(sbatch slurm/eval_per_model/eval_llama_dpo_beta0.05_gsm8k.slurm | grep -oP "\d+")
echo "  Job ID: $JOB_ID"
JOB_IDS="$JOB_IDS $JOB_ID"

echo "Submitting eval_llama_dpo_beta0.1_advbench..."
JOB_ID=$(sbatch slurm/eval_per_model/eval_llama_dpo_beta0.1_advbench.slurm | grep -oP "\d+")
echo "  Job ID: $JOB_ID"
JOB_IDS="$JOB_IDS $JOB_ID"

echo "Submitting eval_llama_dpo_beta0.1_mmlu..."
JOB_ID=$(sbatch slurm/eval_per_model/eval_llama_dpo_beta0.1_mmlu.slurm | grep -oP "\d+")
echo "  Job ID: $JOB_ID"
JOB_IDS="$JOB_IDS $JOB_ID"

echo "Submitting eval_llama_dpo_beta0.1_gsm8k..."
JOB_ID=$(sbatch slurm/eval_per_model/eval_llama_dpo_beta0.1_gsm8k.slurm | grep -oP "\d+")
echo "  Job ID: $JOB_ID"
JOB_IDS="$JOB_IDS $JOB_ID"

echo "Submitting eval_llama_dpo_beta0.5_advbench..."
JOB_ID=$(sbatch slurm/eval_per_model/eval_llama_dpo_beta0.5_advbench.slurm | grep -oP "\d+")
echo "  Job ID: $JOB_ID"
JOB_IDS="$JOB_IDS $JOB_ID"

echo "Submitting eval_llama_dpo_beta0.5_mmlu..."
JOB_ID=$(sbatch slurm/eval_per_model/eval_llama_dpo_beta0.5_mmlu.slurm | grep -oP "\d+")
echo "  Job ID: $JOB_ID"
JOB_IDS="$JOB_IDS $JOB_ID"

echo "Submitting eval_llama_dpo_beta0.5_gsm8k..."
JOB_ID=$(sbatch slurm/eval_per_model/eval_llama_dpo_beta0.5_gsm8k.slurm | grep -oP "\d+")
echo "  Job ID: $JOB_ID"
JOB_IDS="$JOB_IDS $JOB_ID"

echo "Submitting eval_llama_dpo_beta1.0_advbench..."
JOB_ID=$(sbatch slurm/eval_per_model/eval_llama_dpo_beta1.0_advbench.slurm | grep -oP "\d+")
echo "  Job ID: $JOB_ID"
JOB_IDS="$JOB_IDS $JOB_ID"

echo "Submitting eval_llama_dpo_beta1.0_mmlu..."
JOB_ID=$(sbatch slurm/eval_per_model/eval_llama_dpo_beta1.0_mmlu.slurm | grep -oP "\d+")
echo "  Job ID: $JOB_ID"
JOB_IDS="$JOB_IDS $JOB_ID"

echo "Submitting eval_llama_dpo_beta1.0_gsm8k..."
JOB_ID=$(sbatch slurm/eval_per_model/eval_llama_dpo_beta1.0_gsm8k.slurm | grep -oP "\d+")
echo "  Job ID: $JOB_ID"
JOB_IDS="$JOB_IDS $JOB_ID"

echo "Submitting eval_qwen_sft_advbench..."
JOB_ID=$(sbatch slurm/eval_per_model/eval_qwen_sft_advbench.slurm | grep -oP "\d+")
echo "  Job ID: $JOB_ID"
JOB_IDS="$JOB_IDS $JOB_ID"

echo "Submitting eval_qwen_sft_mmlu..."
JOB_ID=$(sbatch slurm/eval_per_model/eval_qwen_sft_mmlu.slurm | grep -oP "\d+")
echo "  Job ID: $JOB_ID"
JOB_IDS="$JOB_IDS $JOB_ID"

echo "Submitting eval_qwen_sft_gsm8k..."
JOB_ID=$(sbatch slurm/eval_per_model/eval_qwen_sft_gsm8k.slurm | grep -oP "\d+")
echo "  Job ID: $JOB_ID"
JOB_IDS="$JOB_IDS $JOB_ID"

echo "Submitting eval_qwen_dpo_beta0.01_advbench..."
JOB_ID=$(sbatch slurm/eval_per_model/eval_qwen_dpo_beta0.01_advbench.slurm | grep -oP "\d+")
echo "  Job ID: $JOB_ID"
JOB_IDS="$JOB_IDS $JOB_ID"

echo "Submitting eval_qwen_dpo_beta0.01_mmlu..."
JOB_ID=$(sbatch slurm/eval_per_model/eval_qwen_dpo_beta0.01_mmlu.slurm | grep -oP "\d+")
echo "  Job ID: $JOB_ID"
JOB_IDS="$JOB_IDS $JOB_ID"

echo "Submitting eval_qwen_dpo_beta0.01_gsm8k..."
JOB_ID=$(sbatch slurm/eval_per_model/eval_qwen_dpo_beta0.01_gsm8k.slurm | grep -oP "\d+")
echo "  Job ID: $JOB_ID"
JOB_IDS="$JOB_IDS $JOB_ID"

echo "Submitting eval_qwen_dpo_beta0.05_advbench..."
JOB_ID=$(sbatch slurm/eval_per_model/eval_qwen_dpo_beta0.05_advbench.slurm | grep -oP "\d+")
echo "  Job ID: $JOB_ID"
JOB_IDS="$JOB_IDS $JOB_ID"

echo "Submitting eval_qwen_dpo_beta0.05_mmlu..."
JOB_ID=$(sbatch slurm/eval_per_model/eval_qwen_dpo_beta0.05_mmlu.slurm | grep -oP "\d+")
echo "  Job ID: $JOB_ID"
JOB_IDS="$JOB_IDS $JOB_ID"

echo "Submitting eval_qwen_dpo_beta0.05_gsm8k..."
JOB_ID=$(sbatch slurm/eval_per_model/eval_qwen_dpo_beta0.05_gsm8k.slurm | grep -oP "\d+")
echo "  Job ID: $JOB_ID"
JOB_IDS="$JOB_IDS $JOB_ID"

echo "Submitting eval_qwen_dpo_beta0.1_advbench..."
JOB_ID=$(sbatch slurm/eval_per_model/eval_qwen_dpo_beta0.1_advbench.slurm | grep -oP "\d+")
echo "  Job ID: $JOB_ID"
JOB_IDS="$JOB_IDS $JOB_ID"

echo "Submitting eval_qwen_dpo_beta0.1_mmlu..."
JOB_ID=$(sbatch slurm/eval_per_model/eval_qwen_dpo_beta0.1_mmlu.slurm | grep -oP "\d+")
echo "  Job ID: $JOB_ID"
JOB_IDS="$JOB_IDS $JOB_ID"

echo "Submitting eval_qwen_dpo_beta0.1_gsm8k..."
JOB_ID=$(sbatch slurm/eval_per_model/eval_qwen_dpo_beta0.1_gsm8k.slurm | grep -oP "\d+")
echo "  Job ID: $JOB_ID"
JOB_IDS="$JOB_IDS $JOB_ID"

echo "Submitting eval_qwen_dpo_beta0.5_advbench..."
JOB_ID=$(sbatch slurm/eval_per_model/eval_qwen_dpo_beta0.5_advbench.slurm | grep -oP "\d+")
echo "  Job ID: $JOB_ID"
JOB_IDS="$JOB_IDS $JOB_ID"

echo "Submitting eval_qwen_dpo_beta0.5_mmlu..."
JOB_ID=$(sbatch slurm/eval_per_model/eval_qwen_dpo_beta0.5_mmlu.slurm | grep -oP "\d+")
echo "  Job ID: $JOB_ID"
JOB_IDS="$JOB_IDS $JOB_ID"

echo "Submitting eval_qwen_dpo_beta0.5_gsm8k..."
JOB_ID=$(sbatch slurm/eval_per_model/eval_qwen_dpo_beta0.5_gsm8k.slurm | grep -oP "\d+")
echo "  Job ID: $JOB_ID"
JOB_IDS="$JOB_IDS $JOB_ID"

echo "Submitting eval_qwen_dpo_beta1.0_advbench..."
JOB_ID=$(sbatch slurm/eval_per_model/eval_qwen_dpo_beta1.0_advbench.slurm | grep -oP "\d+")
echo "  Job ID: $JOB_ID"
JOB_IDS="$JOB_IDS $JOB_ID"

echo "Submitting eval_qwen_dpo_beta1.0_mmlu..."
JOB_ID=$(sbatch slurm/eval_per_model/eval_qwen_dpo_beta1.0_mmlu.slurm | grep -oP "\d+")
echo "  Job ID: $JOB_ID"
JOB_IDS="$JOB_IDS $JOB_ID"

echo "Submitting eval_qwen_dpo_beta1.0_gsm8k..."
JOB_ID=$(sbatch slurm/eval_per_model/eval_qwen_dpo_beta1.0_gsm8k.slurm | grep -oP "\d+")
echo "  Job ID: $JOB_ID"
JOB_IDS="$JOB_IDS $JOB_ID"


echo ""
echo "=============================================="
echo "ALL 36 JOBS SUBMITTED"
echo "=============================================="
echo ""
echo "Job IDs:$JOB_IDS"
echo ""
echo "Monitor with: squeue -u $USER"
echo "Results will be in: /ibex/scratch/zahrmm0b/safety-overrefusal-project/eval_results_expanded/"
