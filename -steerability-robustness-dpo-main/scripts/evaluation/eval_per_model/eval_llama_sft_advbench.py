#!/usr/bin/env python3
"""Single-model AdvBench evaluation: llama_sft"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm
import os

# Configuration
MODEL_NAME = "llama_sft"
ADAPTER_PATH = "/ibex/scratch/zahrmm0b/safety-overrefusal-project/checkpoints/llama3.1_8b_sft_alpaca"
BASE_MODEL = "meta-llama/Llama-3.1-8B"
NUM_SAMPLES = 520
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
    print(f"ADVBENCH EVALUATION: {MODEL_NAME}")
    print(f"=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    prompts = load_advbench()
    print(f"Loaded {len(prompts)} prompts")

    # Load model using correct PeftModel approach
    print(f"Loading {MODEL_NAME}...")
    print(f"  Base model: {BASE_MODEL}")
    print(f"  Adapter: {ADAPTER_PATH}")

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
        messages = [{"role": "user", "content": prompt}]
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
        results.append({
            "prompt": prompt,
            "response": response,
            "model": MODEL_NAME,
        })

    # Save results
    output_file = os.path.join(OUTPUT_DIR, f"advbench_{MODEL_NAME}.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved {len(results)} results to {output_file}")

if __name__ == "__main__":
    main()
