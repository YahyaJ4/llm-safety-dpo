#!/usr/bin/env python3
"""
Create DPO dataset from PKU-SafeRLHF using the STANDARD approach.

The correct way (from SafeDPO paper):
- Use `safer_response_id` to determine chosen/rejected
- chosen = the SAFER response
- rejected = the LESS SAFE response

This script includes verification tests to ensure data quality.
"""

import json
import os
from datasets import load_dataset


def verify_example(example, chosen, rejected, safer_id):
    """Verify that the chosen response is actually the safer one."""
    is_0_safe = example.get("is_response_0_safe")
    is_1_safe = example.get("is_response_1_safe")

    if is_0_safe is None or is_1_safe is None:
        return "no_safety_labels"

    # Check consistency
    if safer_id == 0:
        # We chose response_0 as safer
        if is_0_safe and not is_1_safe:
            return "correct_clear"  # Clearly correct: chose safe, rejected unsafe
        elif is_0_safe and is_1_safe:
            return "both_safe"  # Both safe, but 0 is "safer"
        elif not is_0_safe and not is_1_safe:
            return "both_unsafe"  # Both unsafe, but 0 is "less unsafe"
        else:
            return "INVERTED"  # ERROR: chose unsafe over safe
    else:
        # We chose response_1 as safer
        if is_1_safe and not is_0_safe:
            return "correct_clear"
        elif is_0_safe and is_1_safe:
            return "both_safe"
        elif not is_0_safe and not is_1_safe:
            return "both_unsafe"
        else:
            return "INVERTED"


def main():
    output_dir = os.environ.get("OUTPUT_DIR", "/ibex/scratch/zahrmm0b/safety-overrefusal-project/data")
    cache_dir = os.environ.get("HF_DATASETS_CACHE", "/ibex/scratch/zahrmm0b/.cache/huggingface/datasets")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    print("=" * 60)
    print("STEP 1: Loading PKU-SafeRLHF dataset")
    print("=" * 60)
    dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF", cache_dir=cache_dir, trust_remote_code=True)

    train_data = dataset["train"]
    print(f"Loaded {len(train_data)} training examples")
    print(f"Columns: {train_data.column_names}")

    # Show first example structure
    print("\n--- First example structure ---")
    first = train_data[0]
    for key in first.keys():
        val = first[key]
        if isinstance(val, str) and len(val) > 100:
            print(f"  {key}: {val[:100]}...")
        else:
            print(f"  {key}: {val}")

    print("\n" + "=" * 60)
    print("STEP 2: Processing dataset using safer_response_id")
    print("=" * 60)

    dpo_data = []
    stats = {
        "total": 0,
        "used": 0,
        "skipped_identical": 0,
        "chosen_is_0": 0,
        "chosen_is_1": 0,
        # Verification stats
        "correct_clear": 0,
        "both_safe": 0,
        "both_unsafe": 0,
        "INVERTED": 0,
        "no_safety_labels": 0,
    }

    inverted_examples = []  # Store inverted examples for review

    for i, example in enumerate(train_data):
        stats["total"] += 1

        prompt = example["prompt"]
        response_0 = example["response_0"]
        response_1 = example["response_1"]
        safer_id = example["safer_response_id"]

        # Use safer_response_id to determine chosen/rejected
        if safer_id == 0:
            chosen = response_0
            rejected = response_1
            stats["chosen_is_0"] += 1
        else:
            chosen = response_1
            rejected = response_0
            stats["chosen_is_1"] += 1

        # Skip if responses are identical
        if chosen.strip() == rejected.strip():
            stats["skipped_identical"] += 1
            continue

        # Verify this example
        verification = verify_example(example, chosen, rejected, safer_id)
        stats[verification] += 1

        if verification == "INVERTED" and len(inverted_examples) < 5:
            inverted_examples.append({
                "index": i,
                "prompt": prompt[:100],
                "chosen": chosen[:100],
                "rejected": rejected[:100],
                "is_0_safe": example.get("is_response_0_safe"),
                "is_1_safe": example.get("is_response_1_safe"),
                "safer_id": safer_id,
            })

        dpo_data.append({
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected
        })
        stats["used"] += 1

    print("\n--- Processing Statistics ---")
    print(f"Total examples: {stats['total']}")
    print(f"Used examples: {stats['used']}")
    print(f"Skipped (identical): {stats['skipped_identical']}")
    print(f"Chosen is response_0: {stats['chosen_is_0']}")
    print(f"Chosen is response_1: {stats['chosen_is_1']}")

    print("\n" + "=" * 60)
    print("STEP 3: Verification Results")
    print("=" * 60)
    print(f"Correct & Clear (safe vs unsafe): {stats['correct_clear']} ({100*stats['correct_clear']/stats['used']:.1f}%)")
    print(f"Both responses safe: {stats['both_safe']} ({100*stats['both_safe']/stats['used']:.1f}%)")
    print(f"Both responses unsafe: {stats['both_unsafe']} ({100*stats['both_unsafe']/stats['used']:.1f}%)")
    print(f"No safety labels: {stats['no_safety_labels']} ({100*stats['no_safety_labels']/stats['used']:.1f}%)")
    print(f"INVERTED (ERROR): {stats['INVERTED']} ({100*stats['INVERTED']/stats['used']:.1f}%)")

    if inverted_examples:
        print("\n--- Sample INVERTED examples (for review) ---")
        for ex in inverted_examples:
            print(f"\nIndex {ex['index']}:")
            print(f"  is_response_0_safe: {ex['is_0_safe']}")
            print(f"  is_response_1_safe: {ex['is_1_safe']}")
            print(f"  safer_response_id: {ex['safer_id']}")
            print(f"  Prompt: {ex['prompt']}...")
            print(f"  Chosen: {ex['chosen']}...")
            print(f"  Rejected: {ex['rejected']}...")

    print("\n" + "=" * 60)
    print("STEP 4: Sample Examples (verify manually)")
    print("=" * 60)

    # Show examples from different categories
    for i, ex in enumerate(dpo_data[:5]):
        print(f"\n--- Example {i+1} ---")
        print(f"Prompt: {ex['prompt'][:150]}...")
        print(f"CHOSEN (should be SAFER): {ex['chosen'][:200]}...")
        print(f"REJECTED (should be LESS SAFE): {ex['rejected'][:200]}...")

    print("\n" + "=" * 60)
    print("STEP 5: Saving dataset")
    print("=" * 60)

    # Save full dataset
    output_path = os.path.join(output_dir, "pku_safety_dpo.json")
    with open(output_path, "w") as f:
        json.dump(dpo_data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(dpo_data)} examples to {output_path}")

    # Save 10K subset for faster training
    small_data = dpo_data[:10000]
    small_path = os.path.join(output_dir, "pku_safety_dpo_10k.json")
    with open(small_path, "w") as f:
        json.dump(small_data, f, indent=2, ensure_ascii=False)
    print(f"Saved 10K subset to {small_path}")

    # Save stats for reference
    stats_path = os.path.join(output_dir, "pku_safety_dpo_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved statistics to {stats_path}")

    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    if stats["INVERTED"] > 0:
        print(f"WARNING: {stats['INVERTED']} examples have inverted labels!")
        print("   This is expected - safer_response_id uses relative safety,")
        print("   not absolute is_response_X_safe labels.")

    clear_signal_pct = 100 * stats['correct_clear'] / stats['used']
    if clear_signal_pct > 30:
        print(f"OK: {clear_signal_pct:.1f}% of examples have clear safe vs unsafe contrast")
        print("  This should provide a good learning signal for DPO.")
    else:
        print(f"WARNING: Only {clear_signal_pct:.1f}% have clear safe vs unsafe contrast")
        print("  Consider filtering for higher quality data.")


if __name__ == "__main__":
    main()
