#!/usr/bin/env python3
"""
Generate per-model evaluation scripts for maximum parallelization.
Creates 36 jobs: 2 model families × 6 versions × 3 benchmarks

Uses the correct approach from working eval scripts:
- PeftModel for loading LoRA adapters
- tokenizer.apply_chat_template() for prompt formatting
- Correct checkpoint paths
"""

import os

# Configuration
OUTPUT_DIR = "scripts/eval_per_model"
SLURM_DIR = "slurm/eval_per_model"
PROJECT_DIR = "/ibex/scratch/zahrmm0b/safety-overrefusal-project"

# Correct checkpoint paths (PKU strict filtering)
LLAMA_MODELS = {
    "llama_sft": f"{PROJECT_DIR}/checkpoints/llama3.1_8b_sft_alpaca",
    "llama_dpo_beta0.01": f"{PROJECT_DIR}/checkpoints/llama3.1_8b_dpo_pku_strict_beta0.01",
    "llama_dpo_beta0.05": f"{PROJECT_DIR}/checkpoints/llama3.1_8b_dpo_pku_strict_beta0.05",
    "llama_dpo_beta0.1": f"{PROJECT_DIR}/checkpoints/llama3.1_8b_dpo_pku_strict_beta0.1",
    "llama_dpo_beta0.5": f"{PROJECT_DIR}/checkpoints/llama3.1_8b_dpo_pku_strict_beta0.5",
    "llama_dpo_beta1.0": f"{PROJECT_DIR}/checkpoints/llama3.1_8b_dpo_pku_strict_beta1.0",
}

QWEN_MODELS = {
    "qwen_sft": f"{PROJECT_DIR}/checkpoints/qwen2.5_7b_sft_alpaca",
    "qwen_dpo_beta0.01": f"{PROJECT_DIR}/checkpoints/qwen2.5_7b_dpo_pku_strict_beta0.01",
    "qwen_dpo_beta0.05": f"{PROJECT_DIR}/checkpoints/qwen2.5_7b_dpo_pku_strict_beta0.05",
    "qwen_dpo_beta0.1": f"{PROJECT_DIR}/checkpoints/qwen2.5_7b_dpo_pku_strict_beta0.1",
    "qwen_dpo_beta0.5": f"{PROJECT_DIR}/checkpoints/qwen2.5_7b_dpo_pku_strict_beta0.5",
    "qwen_dpo_beta1.0": f"{PROJECT_DIR}/checkpoints/qwen2.5_7b_dpo_pku_strict_beta1.0",
}

BENCHMARKS = {
    "advbench": {
        "samples": 520,
        "time": "2:00:00",
    },
    "mmlu": {
        "samples": 1000,
        "time": "2:00:00",
    },
    "gsm8k": {
        "samples": 500,
        "time": "2:00:00",
    },
}

# Python script templates using correct PeftModel approach
ADVBENCH_TEMPLATE = '''#!/usr/bin/env python3
"""Single-model AdvBench evaluation: {model_name}"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
import os

# Configuration
MODEL_NAME = "{model_name}"
ADAPTER_PATH = "{adapter_path}"
BASE_MODEL = "{base_model}"
NUM_SAMPLES = {num_samples}
OUTPUT_DIR = "/ibex/scratch/zahrmm0b/safety-overrefusal-project/eval_results_expanded"

def load_advbench():
    advbench_path = "/ibex/scratch/zahrmm0b/safety-overrefusal-project/data/advbench_harmful_behaviors.json"
    with open(advbench_path, "r") as f:
        data = json.load(f)
    # Handle both formats
    if isinstance(data[0], dict) and "goal" in data[0]:
        return [item["goal"] for item in data][:NUM_SAMPLES]
    elif isinstance(data[0], dict) and "prompt" in data[0]:
        return [item["prompt"] for item in data][:NUM_SAMPLES]
    else:
        return data[:NUM_SAMPLES]

def main():
    print(f"=" * 60)
    print(f"ADVBENCH EVALUATION: {{MODEL_NAME}}")
    print(f"=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    prompts = load_advbench()
    print(f"Loaded {{len(prompts)}} prompts")

    # Load model using correct PeftModel approach
    print(f"Loading {{MODEL_NAME}}...")
    print(f"  Base model: {{BASE_MODEL}}")
    print(f"  Adapter: {{ADAPTER_PATH}}")

    # Load tokenizer from adapter (has correct chat template)
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Apply LoRA adapter
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()
    print("Model loaded successfully")

    # Generate responses
    results = []
    for i, prompt in enumerate(tqdm(prompts, desc="Generating")):
        # Use chat template for proper formatting
        messages = [{{"role": "user", "content": prompt}}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        results.append({{
            "prompt": prompt,
            "response": response,
            "model": MODEL_NAME,
        }})

    # Save results
    output_file = os.path.join(OUTPUT_DIR, f"advbench_{{MODEL_NAME}}.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved {{len(results)}} results to {{output_file}}")

if __name__ == "__main__":
    main()
'''

MMLU_TEMPLATE = '''#!/usr/bin/env python3
"""Single-model MMLU evaluation: {model_name}"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm
import os
import random

# Configuration
MODEL_NAME = "{model_name}"
ADAPTER_PATH = "{adapter_path}"
BASE_MODEL = "{base_model}"
NUM_SAMPLES = {num_samples}
OUTPUT_DIR = "/ibex/scratch/zahrmm0b/safety-overrefusal-project/eval_results_expanded"

random.seed(42)

def load_mmlu():
    dataset = load_dataset("cais/mmlu", "all", split="test")
    indices = random.sample(range(len(dataset)), min(NUM_SAMPLES, len(dataset)))
    return [dataset[i] for i in indices]

def extract_answer(response):
    response = response.strip().upper()
    for char in response:
        if char in "ABCD":
            return char
    return None

def main():
    print(f"=" * 60)
    print(f"MMLU EVALUATION: {{MODEL_NAME}}")
    print(f"=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    samples = load_mmlu()
    print(f"Loaded {{len(samples)}} MMLU samples")

    # Load model using correct PeftModel approach
    print(f"Loading {{MODEL_NAME}}...")

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()
    print("Model loaded successfully")

    # Evaluate
    correct = 0
    total = 0
    results = []

    for sample in tqdm(samples, desc="Evaluating"):
        question = sample["question"]
        choices = sample["choices"]
        prompt = f"Question: {{question}}\\n\\nChoices:\\nA) {{choices[0]}}\\nB) {{choices[1]}}\\nC) {{choices[2]}}\\nD) {{choices[3]}}\\n\\nAnswer with just the letter (A, B, C, or D):"

        messages = [{{"role": "user", "content": prompt}}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        predicted = extract_answer(response)
        correct_answer = ["A", "B", "C", "D"][sample["answer"]]
        is_correct = predicted == correct_answer

        if is_correct:
            correct += 1
        total += 1

        results.append({{
            "question": question,
            "choices": choices,
            "correct_answer": correct_answer,
            "predicted": predicted,
            "response": response,
            "is_correct": is_correct,
            "model": MODEL_NAME,
        }})

    accuracy = correct / total if total > 0 else 0
    print(f"\\nAccuracy: {{accuracy:.4f}} ({{correct}}/{{total}})")

    # Save results
    output_file = os.path.join(OUTPUT_DIR, f"mmlu_{{MODEL_NAME}}.json")
    with open(output_file, "w") as f:
        json.dump({{"accuracy": accuracy, "correct": correct, "total": total, "results": results}}, f, indent=2)

    print(f"Saved results to {{output_file}}")

if __name__ == "__main__":
    main()
'''

GSM8K_TEMPLATE = '''#!/usr/bin/env python3
"""Single-model GSM8K evaluation: {model_name}"""

import json
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm
import os
import random

# Configuration
MODEL_NAME = "{model_name}"
ADAPTER_PATH = "{adapter_path}"
BASE_MODEL = "{base_model}"
NUM_SAMPLES = {num_samples}
OUTPUT_DIR = "/ibex/scratch/zahrmm0b/safety-overrefusal-project/eval_results_expanded"

random.seed(42)

def load_gsm8k():
    dataset = load_dataset("gsm8k", "main", split="test")
    indices = random.sample(range(len(dataset)), min(NUM_SAMPLES, len(dataset)))
    return [dataset[i] for i in indices]

def extract_number(text):
    text = text.replace(",", "")
    patterns = [
        r"[Tt]he answer is[:\\s]*\\$?([\\d,]+\\.?\\d*)",
        r"[Aa]nswer[:\\s]*\\$?([\\d,]+\\.?\\d*)",
        r"####\\s*\\$?([\\d,]+\\.?\\d*)",
        r"=\\s*\\$?([\\d,]+\\.?\\d*)\\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).replace(",", "")
    numbers = re.findall(r"\\$?([\\d,]+\\.?\\d*)", text)
    if numbers:
        return numbers[-1].replace(",", "")
    return None

def extract_ground_truth(answer_text):
    match = re.search(r"####\\s*([\\d,]+\\.?\\d*)", answer_text)
    if match:
        return match.group(1).replace(",", "")
    return None

def main():
    print(f"=" * 60)
    print(f"GSM8K EVALUATION: {{MODEL_NAME}}")
    print(f"=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    samples = load_gsm8k()
    print(f"Loaded {{len(samples)}} GSM8K samples")

    # Load model using correct PeftModel approach
    print(f"Loading {{MODEL_NAME}}...")

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()
    print("Model loaded successfully")

    # Evaluate
    correct = 0
    total = 0
    results = []

    for sample in tqdm(samples, desc="Evaluating"):
        question = sample["question"]
        prompt = f"Solve this math problem step by step:\\n\\n{{question}}\\n\\nProvide your final numerical answer after 'The answer is'."

        messages = [{{"role": "user", "content": prompt}}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        predicted = extract_number(response)
        ground_truth = extract_ground_truth(sample["answer"])

        is_correct = False
        if predicted and ground_truth:
            try:
                is_correct = abs(float(predicted) - float(ground_truth)) < 0.01
            except ValueError:
                pass

        if is_correct:
            correct += 1
        total += 1

        results.append({{
            "question": question,
            "ground_truth": ground_truth,
            "predicted": predicted,
            "response": response,
            "is_correct": is_correct,
            "model": MODEL_NAME,
        }})

    accuracy = correct / total if total > 0 else 0
    print(f"\\nAccuracy: {{accuracy:.4f}} ({{correct}}/{{total}})")

    # Save results
    output_file = os.path.join(OUTPUT_DIR, f"gsm8k_{{MODEL_NAME}}.json")
    with open(output_file, "w") as f:
        json.dump({{"accuracy": accuracy, "correct": correct, "total": total, "results": results}}, f, indent=2)

    print(f"Saved results to {{output_file}}")

if __name__ == "__main__":
    main()
'''

SLURM_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=/ibex/scratch/zahrmm0b/safety-overrefusal-project/logs/{job_name}_%j.out
#SBATCH --error=/ibex/scratch/zahrmm0b/safety-overrefusal-project/logs/{job_name}_%j.err
#SBATCH --time={time}
#SBATCH --gpus=1
#SBATCH --constraint=a100
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4

echo "=============================================="
echo "{job_name}"
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
python {script_path}

echo "=============================================="
echo "End: $(date)"
echo "=============================================="
"""


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SLURM_DIR, exist_ok=True)

    all_jobs = []

    # Generate scripts for all combinations
    for model_family, models, base_model in [
        ("llama", LLAMA_MODELS, "meta-llama/Llama-3.1-8B"),
        ("qwen", QWEN_MODELS, "Qwen/Qwen2.5-7B"),
    ]:
        for model_name, adapter_path in models.items():
            for benchmark, config in BENCHMARKS.items():
                # Generate Python script
                if benchmark == "advbench":
                    template = ADVBENCH_TEMPLATE
                elif benchmark == "mmlu":
                    template = MMLU_TEMPLATE
                else:
                    template = GSM8K_TEMPLATE

                script_content = template.format(
                    model_name=model_name,
                    adapter_path=adapter_path,
                    base_model=base_model,
                    num_samples=config["samples"],
                )

                script_name = f"eval_{model_name}_{benchmark}.py"
                script_path = os.path.join(OUTPUT_DIR, script_name)
                with open(script_path, "w") as f:
                    f.write(script_content)

                # Generate SLURM script
                job_name = f"eval_{model_name}_{benchmark}"
                slurm_content = SLURM_TEMPLATE.format(
                    job_name=job_name,
                    time=config["time"],
                    script_path=f"scripts/eval_per_model/{script_name}",
                )

                slurm_name = f"{job_name}.slurm"
                slurm_path = os.path.join(SLURM_DIR, slurm_name)
                with open(slurm_path, "w") as f:
                    f.write(slurm_content)

                all_jobs.append((job_name, slurm_path))

    print(f"Generated {len(all_jobs)} job configurations")

    # Generate master submission script
    submit_script = """#!/bin/bash
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

"""

    for job_name, slurm_path in all_jobs:
        submit_script += f'echo "Submitting {job_name}..."\n'
        submit_script += f'JOB_ID=$(sbatch {slurm_path} | grep -oP "\\d+")\n'
        submit_script += f'echo "  Job ID: $JOB_ID"\n'
        submit_script += f'JOB_IDS="$JOB_IDS $JOB_ID"\n'
        submit_script += "\n"

    submit_script += """
echo ""
echo "=============================================="
echo "ALL 36 JOBS SUBMITTED"
echo "=============================================="
echo ""
echo "Job IDs:$JOB_IDS"
echo ""
echo "Monitor with: squeue -u $USER"
echo "Results will be in: /ibex/scratch/zahrmm0b/safety-overrefusal-project/eval_results_expanded/"
"""

    submit_path = os.path.join(SLURM_DIR, "submit_all.sh")
    with open(submit_path, "w") as f:
        f.write(submit_script)
    os.chmod(submit_path, 0o755)

    print(f"\nGenerated submission script: {submit_path}")
    print(f"\nTo submit all jobs:")
    print(f"  cd /ibex/scratch/zahrmm0b/safety-overrefusal-project")
    print(f"  bash slurm/eval_per_model/submit_all.sh")


if __name__ == "__main__":
    main()
