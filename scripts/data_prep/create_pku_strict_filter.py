"""
Create a properly filtered DPO dataset for safety alignment.

The original PKU-SafeRLHF dataset has a critical issue: both "chosen" and "rejected"
responses often contain harmful information. The "chosen" is only *relatively* safer,
not an actual refusal.

This script creates a corrected dataset where:
- CHOSEN: Always a proper refusal (safe response)
- REJECTED: Always a harmful/compliant response (unsafe response)

This ensures DPO learns the correct contrast between refusal and compliance.
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from datasets import load_dataset
from tqdm import tqdm


def is_refusal_response(response: str) -> bool:
    """
    Check if a response is a proper refusal (doesn't provide harmful info).
    Uses pattern matching as a first pass filter.
    """
    response_lower = response.lower()

    # Strong refusal indicators
    refusal_phrases = [
        "i cannot",
        "i can't",
        "i won't",
        "i will not",
        "i'm not able to",
        "i am not able to",
        "i'm unable to",
        "i am unable to",
        "i must refuse",
        "i have to refuse",
        "i apologize, but",
        "i'm sorry, but i cannot",
        "as an ai",
        "as a language model",
        "it would be unethical",
        "it is unethical",
        "it's not appropriate",
        "it is not appropriate",
        "i don't feel comfortable",
        "i do not feel comfortable",
        "against my guidelines",
        "against my programming",
        "harmful request",
        "cannot assist with",
        "not able to provide",
        "cannot provide",
        "refuse to",
        "decline to",
    ]

    # Check for refusal phrases
    has_refusal = any(phrase in response_lower for phrase in refusal_phrases)

    # Additional check: response should be relatively short for refusals
    # Long responses that contain refusal phrases might still provide harmful info
    is_concise = len(response.split()) < 200

    return has_refusal and is_concise


def contains_harmful_info(response: str, prompt: str) -> bool:
    """
    Check if a response likely contains harmful information.
    This is a heuristic - ideally use LLM judge for accuracy.
    """
    response_lower = response.lower()

    # Indicators that the response is providing information (not refusing)
    info_indicators = [
        "here's how",
        "here is how",
        "you can",
        "first,",
        "step 1",
        "step one",
        "the way to",
        "one way to",
        "method:",
        "steps:",
        "instructions:",
        "to do this",
        "in order to",
        "you'll need",
        "you will need",
        "start by",
        "begin by",
    ]

    has_info = any(indicator in response_lower for indicator in info_indicators)

    # Also check if response is long (suggesting detailed explanation)
    is_detailed = len(response.split()) > 100

    return has_info or is_detailed


def create_refusal_response(prompt: str) -> str:
    """
    Create a template refusal response for a harmful prompt.
    These are generic refusals that can be used as "chosen" responses.
    """
    refusal_templates = [
        "I cannot provide assistance with this request as it could potentially cause harm. If you have other questions I can help with in a safe and constructive way, I'd be happy to assist.",
        "I'm not able to help with this particular request. It's important to me to avoid contributing to potentially harmful outcomes. Is there something else I can help you with?",
        "I must decline this request as it falls outside the bounds of what I can ethically assist with. I'd be glad to help with other topics that don't raise safety concerns.",
        "I apologize, but I cannot provide information or assistance on this topic as it could lead to harm. Please let me know if there's another way I can be helpful to you.",
        "This is not something I'm able to help with, as doing so could contribute to harmful outcomes. I'm happy to assist with other questions or tasks that are constructive in nature.",
    ]

    import hashlib

    # Use hash of prompt to deterministically select a template
    idx = int(hashlib.md5(prompt.encode()).hexdigest(), 16) % len(refusal_templates)
    return refusal_templates[idx]


def filter_pku_dataset_strict(
    max_samples: Optional[int] = None, require_both_labels: bool = True
) -> List[Dict[str, Any]]:
    """
    Filter PKU-SafeRLHF dataset with strict criteria:
    - Only include examples where one response is SAFE and one is UNSAFE
    - Use the SAFE response as chosen (must be actual refusal)
    - Use the UNSAFE response as rejected (must contain harmful info)

    Args:
        max_samples: Maximum samples to include
        require_both_labels: Require explicit safe/unsafe labels (not just "safer")

    Returns:
        List of filtered DPO examples
    """
    print("Loading PKU-SafeRLHF dataset...")
    dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train")

    print(f"Original dataset size: {len(dataset)}")

    filtered_data = []
    stats = {
        "total": len(dataset),
        "both_safe": 0,
        "both_unsafe": 0,
        "clear_contrast": 0,
        "chosen_not_refusal": 0,
        "used": 0,
    }

    for item in tqdm(dataset, desc="Filtering dataset"):
        prompt = item["prompt"]
        response_0 = item["response_0"]
        response_1 = item["response_1"]

        # Get safety labels
        is_safe_0 = item.get("is_response_0_safe", None)
        is_safe_1 = item.get("is_response_1_safe", None)

        # Skip if we don't have safety labels
        if is_safe_0 is None or is_safe_1 is None:
            continue

        # Case 1: Both responses are safe - skip (no contrast)
        if is_safe_0 and is_safe_1:
            stats["both_safe"] += 1
            continue

        # Case 2: Both responses are unsafe - skip (no good example)
        if not is_safe_0 and not is_safe_1:
            stats["both_unsafe"] += 1
            continue

        # Case 3: Clear contrast - one safe, one unsafe
        stats["clear_contrast"] += 1

        if is_safe_0 and not is_safe_1:
            chosen = response_0
            rejected = response_1
        else:  # is_safe_1 and not is_safe_0
            chosen = response_1
            rejected = response_0

        # Additional validation: check that "chosen" is actually a refusal
        if not is_refusal_response(chosen):
            stats["chosen_not_refusal"] += 1
            # Option: Replace with template refusal
            # chosen = create_refusal_response(prompt)
            # For now, skip these examples
            continue

        stats["used"] += 1

        filtered_data.append(
            {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
            }
        )

        if max_samples and len(filtered_data) >= max_samples:
            break

    print("\n=== Filtering Statistics ===")
    print(f"Total examples: {stats['total']}")
    print(f"Both safe (skipped): {stats['both_safe']}")
    print(f"Both unsafe (skipped): {stats['both_unsafe']}")
    print(f"Clear contrast found: {stats['clear_contrast']}")
    print(f"Chosen not a refusal (skipped): {stats['chosen_not_refusal']}")
    print(f"Final dataset size: {stats['used']}")
    print(f"Retention rate: {100 * stats['used'] / stats['total']:.1f}%")

    return filtered_data


def create_synthetic_refusal_pairs(
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Alternative approach: Create synthetic DPO pairs using harmful prompts
    from PKU dataset, paired with:
    - Chosen: Template refusal responses
    - Rejected: Original harmful responses

    This guarantees clean separation between refusal and compliance.
    """
    print("Loading PKU-SafeRLHF dataset for synthetic pairs...")
    dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train")

    synthetic_data = []

    for item in tqdm(dataset, desc="Creating synthetic pairs"):
        prompt = item["prompt"]
        response_0 = item["response_0"]
        response_1 = item["response_1"]

        # Get safety labels
        is_safe_0 = item.get("is_response_0_safe", None)
        is_safe_1 = item.get("is_response_1_safe", None)

        # Skip if we don't have safety labels
        if is_safe_0 is None or is_safe_1 is None:
            continue

        # Find an unsafe response to use as "rejected"
        unsafe_response = None
        if not is_safe_0:
            unsafe_response = response_0
        elif not is_safe_1:
            unsafe_response = response_1

        if unsafe_response is None:
            # Both safe, skip
            continue

        # Create a template refusal as "chosen"
        refusal = create_refusal_response(prompt)

        synthetic_data.append(
            {
                "prompt": prompt,
                "chosen": refusal,
                "rejected": unsafe_response,
            }
        )

        if max_samples and len(synthetic_data) >= max_samples:
            break

    print(f"Created {len(synthetic_data)} synthetic pairs")
    return synthetic_data


def main():
    parser = argparse.ArgumentParser(description="Create filtered safety DPO dataset")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("data/pku_safe_dpo_filtered.json"),
        help="Output file path",
    )
    parser.add_argument(
        "--max-samples", "-n", type=int, default=None, help="Maximum number of samples"
    )
    parser.add_argument(
        "--mode",
        choices=["filter", "synthetic", "both"],
        default="filter",
        help="Dataset creation mode: filter (use only clean pairs), "
        "synthetic (create template refusals), both (combine)",
    )
    parser.add_argument(
        "--inspect", action="store_true", help="Print sample examples for inspection"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Creating Filtered Safety DPO Dataset")
    print("=" * 80)
    print(f"Mode: {args.mode}")
    print(f"Output: {args.output}")
    print(f"Max samples: {args.max_samples or 'all'}")
    print("=" * 80)

    if args.mode == "filter":
        data = filter_pku_dataset_strict(max_samples=args.max_samples)
    elif args.mode == "synthetic":
        data = create_synthetic_refusal_pairs(max_samples=args.max_samples)
    else:  # both
        filtered = filter_pku_dataset_strict(
            max_samples=args.max_samples // 2 if args.max_samples else None
        )
        synthetic = create_synthetic_refusal_pairs(
            max_samples=args.max_samples // 2 if args.max_samples else None
        )
        data = filtered + synthetic
        print(
            f"Combined: {len(filtered)} filtered + {len(synthetic)} synthetic = {len(data)} total"
        )

    # Inspect samples
    if args.inspect and data:
        print("\n=== Sample Examples ===")
        for i, example in enumerate(data[:3]):
            print(f"\n--- Example {i + 1} ---")
            print(f"PROMPT: {example['prompt'][:200]}...")
            print(f"CHOSEN: {example['chosen'][:200]}...")
            print(f"REJECTED: {example['rejected'][:200]}...")

    # Save dataset
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nSaved {len(data)} examples to {args.output}")

    # Create dataset_info entry for LLaMA-Factory
    dataset_info = {
        "pku_safe_dpo_filtered": {
            "file_name": args.output.name,
            "formatting": "sharegpt",
            "ranking": True,
            "columns": {
                "messages": "prompt",
                "chosen": "chosen",
                "rejected": "rejected",
            },
        }
    }

    info_path = args.output.parent / "dataset_info_filtered.json"
    with open(info_path, "w") as f:
        json.dump(dataset_info, f, indent=2)

    print(f"Saved dataset_info to {info_path}")
    print("\nNext steps:")
    print("1. Copy dataset to LLaMA-Factory/data/")
    print("2. Update dataset_info.json with the new entry")
    print("3. Update DPO config to use 'pku_safe_dpo_filtered' dataset")


if __name__ == "__main__":
    main()
