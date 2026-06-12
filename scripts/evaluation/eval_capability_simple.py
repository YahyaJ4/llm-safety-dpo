"""
Simple capability evaluation using MMLU and GSM8K subsets.
Evaluates models on multiple-choice QA to measure capability degradation.
"""

import json
import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
import re

def load_model_with_adapter(base_model_name, adapter_path=None):
    """Load base model with optional LoRA adapter."""
    print(f"Loading base model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        
    )
    
    if adapter_path:
        print(f"Loading adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
    
    model.eval()
    return model, tokenizer

def evaluate_mmlu_sample(model, tokenizer, num_samples=200):
    """Evaluate on MMLU subset (multiple choice)."""
    print(f"\nEvaluating MMLU ({num_samples} samples)...")
    
    # Load MMLU dataset
    dataset = load_dataset("cais/mmlu", "all", split="test", )
    
    # Sample
    if len(dataset) > num_samples:
        indices = torch.randperm(len(dataset))[:num_samples].tolist()
        dataset = dataset.select(indices)
    
    correct = 0
    total = 0
    
    for item in dataset:
        question = item["question"]
        choices = item["choices"]
        answer_idx = item["answer"]  # 0-3 for A-D
        
        # Format as multiple choice
        prompt = f"Question: {question}\n"
        for i, choice in enumerate(choices):
            prompt += f"{chr(65+i)}. {choice}\n"
        prompt += "Answer with just the letter (A, B, C, or D):"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        
        # Extract answer letter
        predicted = None
        for char in response.upper():
            if char in "ABCD":
                predicted = ord(char) - ord("A")
                break
        
        if predicted == answer_idx:
            correct += 1
        total += 1
        
        if total % 50 == 0:
            print(f"  {total}/{num_samples} done, accuracy so far: {correct/total:.1%}")
    
    accuracy = correct / total if total > 0 else 0
    print(f"MMLU Accuracy: {correct}/{total} = {accuracy:.1%}")
    return {"mmlu_accuracy": accuracy, "mmlu_correct": correct, "mmlu_total": total}

def evaluate_gsm8k_sample(model, tokenizer, num_samples=100):
    """Evaluate on GSM8K subset (math word problems)."""
    print(f"\nEvaluating GSM8K ({num_samples} samples)...")
    
    # Load GSM8K dataset
    dataset = load_dataset("gsm8k", "main", split="test", )
    
    # Sample
    if len(dataset) > num_samples:
        indices = torch.randperm(len(dataset))[:num_samples].tolist()
        dataset = dataset.select(indices)
    
    correct = 0
    total = 0
    
    for item in dataset:
        question = item["question"]
        answer_text = item["answer"]
        
        # Extract numerical answer (after ####)
        match = re.search(r"####\s*(-?[\d,]+)", answer_text)
        if not match:
            continue
        true_answer = match.group(1).replace(",", "")
        
        prompt = f"Question: {question}\nAnswer with just the final number:"
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        
        # Extract number from response
        numbers = re.findall(r"-?[\d,]+", response.replace(",", ""))
        predicted = numbers[-1] if numbers else None
        
        if predicted and predicted.replace(",", "") == true_answer:
            correct += 1
        total += 1
        
        if total % 25 == 0:
            print(f"  {total}/{num_samples} done, accuracy so far: {correct/total:.1%}")
    
    accuracy = correct / total if total > 0 else 0
    print(f"GSM8K Accuracy: {correct}/{total} = {accuracy:.1%}")
    return {"gsm8k_accuracy": accuracy, "gsm8k_correct": correct, "gsm8k_total": total}

def main():
    BASE_MODEL = "meta-llama/Llama-3.1-8B"
    CHECKPOINT_DIR = "/ibex/scratch/zahrmm0b/safety-overrefusal-project/checkpoints"
    OUTPUT_DIR = "/ibex/scratch/zahrmm0b/safety-overrefusal-project/eval_results/capability"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Models to evaluate
    models = [
        ("base", None),
        ("sft_alpaca", f"{CHECKPOINT_DIR}/llama3.1_8b_sft_alpaca"),
        ("dpo_beta0.01", f"{CHECKPOINT_DIR}/llama3.1_8b_dpo_pku_strict_beta0.01"),
        ("dpo_beta0.05", f"{CHECKPOINT_DIR}/llama3.1_8b_dpo_pku_strict_beta0.05"),
        ("dpo_beta0.1", f"{CHECKPOINT_DIR}/llama3.1_8b_dpo_pku_strict_beta0.1"),
        ("dpo_beta0.5", f"{CHECKPOINT_DIR}/llama3.1_8b_dpo_pku_strict_beta0.5"),
        ("dpo_beta1.0", f"{CHECKPOINT_DIR}/llama3.1_8b_dpo_pku_strict_beta1.0"),
    ]
    
    all_results = {}
    
    for name, adapter_path in models:
        print("")
        print("=" * 60)
        print(f"EVALUATING: {name}")
        print("=" * 60)
        
        model, tokenizer = load_model_with_adapter(BASE_MODEL, adapter_path)
        
        # Evaluate
        mmlu_results = evaluate_mmlu_sample(model, tokenizer, num_samples=200)
        gsm8k_results = evaluate_gsm8k_sample(model, tokenizer, num_samples=100)
        
        results = {**mmlu_results, **gsm8k_results}
        all_results[name] = results
        
        # Save individual results
        with open(f"{OUTPUT_DIR}/{name}_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Free memory
        del model
        torch.cuda.empty_cache()
    
    # Save combined results
    with open(f"{OUTPUT_DIR}/all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Print summary
    print("")
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{Model:<20} {MMLU:<15} {GSM8K:<15}")
    print("-" * 50)
    for name, results in all_results.items():
        mmlu = f"{results[mmlu_accuracy]:.1%}"
        gsm8k = f"{results[gsm8k_accuracy]:.1%}"
        print(f"{name:<20} {mmlu:<15} {gsm8k:<15}")
    
    print(f"\nResults saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
