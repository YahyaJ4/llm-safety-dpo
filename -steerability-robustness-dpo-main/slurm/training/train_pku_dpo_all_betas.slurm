#!/bin/bash
#SBATCH --job-name=dpo_pku
#SBATCH --output=/ibex/scratch/zahrmm0b/safety-overrefusal-project/logs/dpo_pku_%A_%a.out
#SBATCH --error=/ibex/scratch/zahrmm0b/safety-overrefusal-project/logs/dpo_pku_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=batch
#SBATCH --constraint=a100
#SBATCH --array=0-4

# Beta values to train
BETAS=(0.01 0.05 0.1 0.5 1.0)
BETA=${BETAS[$SLURM_ARRAY_TASK_ID]}

echo "=========================================="
echo "DPO Training with PKU-SafeRLHF (strict filtered)"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Beta: ${BETA}"
echo "Date: $(date)"
echo ""

# Set up environment
source ~/.bashrc
conda activate llama_factory

# Paths
PROJECT_DIR="/ibex/scratch/zahrmm0b/safety-overrefusal-project"
FACTORY_DIR="${HOME}/LLaMA-Factory"
DATA_DIR="${FACTORY_DIR}/data"
CHECKPOINT_DIR="${PROJECT_DIR}/checkpoints/llama3.1_8b_dpo_pku_strict_beta${BETA}"
CONFIG_DIR="${PROJECT_DIR}/configs_pku_strict"

# Set cache directories to scratch
export HF_HOME="/ibex/scratch/zahrmm0b/.cache/huggingface"
export HF_DATASETS_CACHE="/ibex/scratch/zahrmm0b/.cache/huggingface/datasets"
export TRANSFORMERS_CACHE="/ibex/scratch/zahrmm0b/.cache/huggingface/transformers"

mkdir -p ${HF_HOME} ${HF_DATASETS_CACHE} ${TRANSFORMERS_CACHE}
mkdir -p ${CONFIG_DIR}
mkdir -p ${CHECKPOINT_DIR}

cd ${FACTORY_DIR}

# Update dataset_info.json to point to our strict filtered dataset
echo "Updating dataset_info.json..."
python3 -c "
import json
info_path = '${DATA_DIR}/dataset_info.json'
with open(info_path, 'r') as f:
    info = json.load(f)

info['pku_safety_strict'] = {
    'file_name': '/ibex/scratch/zahrmm0b/safety-overrefusal-project/data/pku_safety_dpo_strict.json',
    'ranking': True,
    'columns': {
        'prompt': 'prompt',
        'chosen': 'chosen',
        'rejected': 'rejected'
    }
}

with open(info_path, 'w') as f:
    json.dump(info, f, indent=2)
print('Updated dataset_info.json with pku_safety_strict')
"

# Create config file
CONFIG_FILE="${CONFIG_DIR}/dpo_llama_pku_beta${BETA}.yaml"
echo "Creating config file: ${CONFIG_FILE}"

cat > ${CONFIG_FILE} << EOF
### Model
model_name_or_path: meta-llama/Llama-3.1-8B
adapter_name_or_path: /ibex/scratch/zahrmm0b/safety-overrefusal-project/checkpoints/llama3.1_8b_sft_alpaca

### Method
stage: dpo
do_train: true
finetuning_type: lora
lora_target: all
pref_beta: ${BETA}

### Dataset
dataset: pku_safety_strict
template: llama3
cutoff_len: 2048
preprocessing_num_workers: 8

### Output
output_dir: ${CHECKPOINT_DIR}
logging_steps: 10
save_steps: 200
plot_loss: true
overwrite_output_dir: true

### Training
per_device_train_batch_size: 4
gradient_accumulation_steps: 2
learning_rate: 1.0e-6
num_train_epochs: 1.0
lr_scheduler_type: cosine
warmup_ratio: 0.1
bf16: true
ddp_timeout: 180000000

### Evaluation
val_size: 0.05
per_device_eval_batch_size: 4
eval_strategy: steps
eval_steps: 100
EOF

echo ""
echo "=== Config file contents ==="
cat ${CONFIG_FILE}
echo ""

# Verify dataset exists
echo "=== Verifying dataset ==="
if [ -f "/ibex/scratch/zahrmm0b/safety-overrefusal-project/data/pku_safety_dpo_strict.json" ]; then
    echo "Dataset file exists"
    EXAMPLES=$(python3 -c "import json; data=json.load(open('/ibex/scratch/zahrmm0b/safety-overrefusal-project/data/pku_safety_dpo_strict.json')); print(len(data))")
    echo "Number of examples: ${EXAMPLES}"
else
    echo "ERROR: Dataset file not found!"
    exit 1
fi

# Show a sample from the dataset
echo ""
echo "=== Sample from dataset ==="
python3 -c "
import json
with open('/ibex/scratch/zahrmm0b/safety-overrefusal-project/data/pku_safety_dpo_strict.json') as f:
    data = json.load(f)
ex = data[0]
print(f'Prompt: {ex[\"prompt\"][:100]}...')
print(f'Chosen: {ex[\"chosen\"][:150]}...')
print(f'Rejected: {ex[\"rejected\"][:150]}...')
"

# Run training
echo ""
echo "=== Starting DPO training with beta=${BETA} ==="
llamafactory-cli train ${CONFIG_FILE}

echo ""
echo "=========================================="
echo "Training completed for beta=${BETA}"
echo "=========================================="
echo "Checkpoint saved to: ${CHECKPOINT_DIR}"
