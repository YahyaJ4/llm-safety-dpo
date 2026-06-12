# Steerability-Robustness Tradeoff in Language Models via DPO

Code and data for the paper: *"Steerability-Robustness Tradeoff in Language Models via Direct Preference Optimization (DPO)"*

## Overview

We systematically study how DPO-based safety alignment affects model capabilities. Our key finding: **the capability tax is not uniform** — DPO primarily degrades complex reasoning (GSM8K) while preserving factual knowledge (MMLU), with effects varying dramatically across model architectures.

### Key Results

| Model | Stage | Refusal % | MMLU % | GSM8K % |
|-------|-------|-----------|--------|---------|
| Llama 3.1-8B | SFT | 22.3 | 50.9 | 12.4 |
| Llama 3.1-8B | DPO β=0.01 | **85.2** | 49.3 | 12.4 |
| Qwen 2.5-7B | SFT | 53.3 | 68.0 | **78.8** |
| Qwen 2.5-7B | DPO β=0.1 | **100.0** | 68.3 | 28.6 |

**Main Finding**: Qwen's GSM8K accuracy collapses from 78.8% → 25.2% under aggressive safety alignment (β=0.01), while MMLU remains stable. This reveals a reasoning-specific capability tax.

## Training Pipeline

```
Base Model → SFT (Alpaca 52k) → DPO (PKU-SafeRLHF strict filtering)
```

- **Models**: Llama-3.1-8B, Qwen2.5-7B
- **DPO β values**: 0.01, 0.05, 0.1, 0.5, 1.0
- **Safety data**: PKU-SafeRLHF with strict filtering (~18k pairs)

## Repository Structure

```
├── configs/
│   ├── llama/                    # Llama SFT + DPO configs
│   └── qwen/                     # Qwen SFT + DPO configs
├── scripts/
│   ├── data_prep/
│   │   ├── create_pku_safety_dpo.py      # Basic PKU processing
│   │   └── create_pku_strict_filter.py   # Strict filtering (paper method)
│   ├── evaluation/               # Evaluation scripts
│   └── llm_judge/
│       └── llm_judge_gpt4o.py    # GPT-4o refusal classification
├── slurm/                        # SLURM job scripts for cluster
├── results/
│   ├── results_comprehensive_expanded.json
│   └── results_table_expanded.csv
├── figures/                      # Publication figures
└── visualizations.py             # Figure generation script
```

## Data Preparation: Strict Filtering

The paper uses **strict filtering** of PKU-SafeRLHF to ensure high-quality preference pairs:

```python
# Only keep pairs where exactly one response is safe
if is_safe_0 and is_safe_1:      # Both safe → skip
    continue
if not is_safe_0 and not is_safe_1:  # Both unsafe → skip
    continue
# Keep only clear contrast pairs
```

This yields ~18,000 high-quality pairs (vs ~83k unfiltered).

## Evaluation

### Benchmarks
- **Safety**: AdvBench (520 prompts) with GPT-4o LLM judge
- **Knowledge**: MMLU (1,000 questions)
- **Reasoning**: GSM8K (500 problems)

### LLM Judge
We use GPT-4o to classify model responses as:
- `refused`: Model declined to answer
- `partial_refusal`: Declined but gave safe alternative
- `answered`: Provided harmful information

This is more accurate than keyword-based detection.

## Running Experiments

### Prerequisites
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- NVIDIA A100 80GB GPU
- OpenAI API key (for LLM judge)

### Training
```bash
# SFT
llamafactory-cli train configs/llama/sft_llama_alpaca.yaml

# DPO with β=0.1
llamafactory-cli train configs/llama/dpo_llama_pku_beta0.1.yaml
```

### Evaluation
```bash
# Run evaluation
python scripts/evaluation/eval_single_beta.py --model llama --beta 0.1

# Run LLM judge on results
python scripts/llm_judge/llm_judge_gpt4o.py --input results/advbench_responses.json
```

### Generate Figures
```bash
python scripts/visualizations.py
```

## Results

Pre-computed results are in `results/`:
- `results_comprehensive_expanded.json`: Full results with metadata
- `results_table_expanded.csv`: Summary table

Figures are in `figures/`:
- `fig1_safety_improvement.png`: SFT vs DPO safety comparison
- `fig7_main_result.png`: Main result figure (3-panel summary)

## Citation

```bibtex
@article{zahran2025steerability,
  title={Steerability-Robustness Tradeoff in Language Models via Direct Preference Optimization},
  author={Zahran, Mahmoud and Al Malallah, Yahya and Alhamoud, Kumail},
  year={2025}
}
```

## License

MIT License
