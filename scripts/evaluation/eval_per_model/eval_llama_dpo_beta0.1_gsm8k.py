#!/usr/bin/env python3
"""Single-model GSM8K evaluation: llama_dpo_beta0.1"""

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
MODEL_NAME = "llama_dpo_beta0.1"
ADAPTER_PATH = "/ibex/scratch/zahrmm0b/safety-overrefusal-project/checkpoints/llama3.1_8b_dpo_pku_strict_beta0.1"
BASE_MODEL = "meta-llama/Llama-3.1-8B"
NUM_SAMPLES = 500
OUTPUT_DIR = "/ibex/scratch/zahrmm0b/safety-overrefusal-project/eval_results_expanded"

random.seed(42)

def load_gsm8k():
    dataset = load_dataset("gsm8k", "main", split="test")
    indices = random.sample(range(len(dataset)), min(NUM_SAMPLES, len(dataset)))
    return [dataset[i] for i in indices]

def extract_number(text):
    text = text.replace(",", "")
    patterns = [
        r"[Tt]he answer is[:\s]*\$?([\d,]+\.?\d*)",
        r"[Aa]nswer[:\s]*\$?([\d,]+\.?\d*)",
        r"####\s*\$?([\d,]+\.?\d*)",
        r"=\s*\$?([\d,]+\.?\d*)\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).replace(",", "")
    numbers = re.findall(r"\$?([\d,]+\.?\d*)", text)
    if numbers:
        return numbers[-1].replace(",", "")
    return None

def extract_ground_truth(answer_text):
    match = re.search(r"####\s*([\d,]+\.?\d*)", answer_text)
    if match:
        return match.group(1).replace(",", "")
    return None

def main():
    print(f"=" * 60)
    print(f"GSM8K EVALUATION: {MODEL_NAME}")
    print(f"=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    samples = load_gsm8k()
    print(f"Loaded {len(samples)} GSM8K samples")

    # Load model using correct PeftModel approach
    print(f"Loading {MODEL_NAME}...")

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
        prompt = f"Solve this math problem step by step:\n\n{question}\n\nProvide your final numerical answer after 'The answer is'."

        messages = [{"role": "user", "content": prompt}]
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

        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "predicted": predicted,
            "response": response,
            "is_correct": is_correct,
            "model": MODEL_NAME,
        })

    accuracy = correct / total if total > 0 else 0
    print(f"\nAccuracy: {accuracy:.4f} ({correct}/{total})")

    # Save results
    output_file = os.path.join(OUTPUT_DIR, f"gsm8k_{MODEL_NAME}.json")
    with open(output_file, "w") as f:
        json.dump({"accuracy": accuracy, "correct": correct, "total": total, "results": results}, f, indent=2)

    print(f"Saved results to {output_file}")

if __name__ == "__main__":
    main()
