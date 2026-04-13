# Baseline Comparison Reference

Data-focused reference for the baseline comparison runs. For narrative
interpretation and direction analysis, see
[`docs/blog/17-baseline-comparison.md`](blog/17-baseline-comparison.md).

## Methodology

Four peer HuggingFace models evaluated with `eval/baseline_eval.py` on the
same problems SRM was evaluated on:

- 100 math problems sampled from `data/grpo/grpo_filtered_v5.jsonl` (seed=42)
- 35 SI test cases from `eval/si_test_cases.jsonl`
- Sampling: temperature=0.8, top_p=0.95, group_size=8, max_gen_tokens=512
- Same reward functions as `math_eval.py` and `si_eval.py`
- `format_weight=0.0` so peer models aren't penalized for not using `<think>` tags
- Hardware: 2× DGX Spark GB10 (castor, pollux), BF16, 128 GB unified memory each

## Models

| Model | Size | Training notes |
|-------|------|----------------|
| Qwen/Qwen2.5-0.5B-Instruct | 494M | Well-mixed ~18T token general model |
| Qwen/Qwen2.5-Math-1.5B-Instruct | 1.54B | Math-specialized fine-tune of Qwen2.5 |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | 1.10B | SlimPajama (~3T tokens, web+books, low math) |
| microsoft/phi-1_5 | 1.32B | Synthetic textbook-style training |

## Math Results

| Model | Size | pass@1 | pass@1 voted | pass@8 | mean reward |
|-------|------|--------|--------------|--------|-------------|
| SRM SFT (500m_sft_v2) | 500M | 0.050 | 0.050 | 0.250 | 0.044 |
| SRM GRPO (500m_grpo) | 500M | 0.040 | 0.050 | 0.290 | 0.050 |
| SRM SFT+SI (500m_sft_si) | 500M | 0.060 | 0.050 | 0.250 | 0.044 |
| Phi-1.5 | 1.3B | 0.010 | 0.030 | 0.160 | 0.024 |
| TinyLlama-1.1B-Chat | 1.1B | 0.020 | 0.030 | 0.100 | 0.016 |
| Qwen2.5-0.5B-Instruct | 500M | 0.340 | 0.510 | 0.600 | 0.339 |
| Qwen2.5-Math-1.5B-Instruct | 1.5B | 0.640 | 0.720 | 0.760 | 0.674 |

## Structured Intent Results

| Model | Size | JSON pass@1 | JSON pass@8 | Fields pass@1 | Fields pass@8 | Mean format | Mean fields |
|-------|------|-------------|-------------|---------------|---------------|-------------|-------------|
| SRM GRPO (no SI SFT) | 500M | 0.143 | 0.629 | 0.000 | 0.057 | 0.088 | 0.038 |
| SRM SFT+SI | 500M | 0.400 | 0.971 | 0.400 | 0.971 | 0.336 | 0.329 |
| Phi-1.5 | 1.3B | 0.314 | 1.000 | 0.200 | 0.914 | 0.236 | 0.324 |
| TinyLlama-1.1B | 1.1B | 0.600 | 1.000 | 0.486 | 0.971 | 0.359 | 0.557 |
| Qwen2.5-0.5B | 500M | 0.829 | 1.000 | 0.829 | 1.000 | 0.375 | 0.737 |
| Qwen2.5-Math-1.5B | 1.5B | 0.000 | 0.286 | 0.000 | 0.000 | 0.020 | 0.004 |

## Training Data Scale Reference

For context on the 10B-token vs 18T-token gap:

| Model | Pre-training tokens |
|-------|---------------------|
| SRM (v1, original GRPO checkpoint) | ~10B (current) |
| SRM (v2, Phase 3 rerun) | 10B, math-enriched (~48% math) |
| Qwen2.5-0.5B | ~18T (1,800× SRM) |
| TinyLlama-1.1B | 3T (300× SRM) |
| Phi-1.5 | ~150B synthetic |

## Per-Model Result Files

- `results/baseline_Qwen_Qwen2.5-0.5B-Instruct.json`
- `results/baseline_Qwen_Qwen2.5-Math-1.5B-Instruct.json`
- `results/baseline_TinyLlama_TinyLlama-1.1B-Chat-v1.0.json`
- `results/baseline_microsoft_phi-1_5.json`

## Key Numeric Observations

1. **Math: data composition beats volume.** TinyLlama (3T tokens, no math
   component) scores 2% pass@1. SRM GRPO (10B tokens, modest math) scores
   4%. Qwen2.5-0.5B (18T well-mixed) scores 34%. Specialized Qwen2.5-Math
   scores 64%.

2. **SI: ceiling is near 100% pass@8 for competent small models.** Four of
   the seven model/variant configurations hit 97-100% fields pass@8. The
   SI pipeline is broadly deployable.

3. **SI: pass@1 varies widely.** Qwen2.5-0.5B (83%) >> TinyLlama (49%) >
   SRM SFT+SI (40%) > Phi-1.5 (20%) > Qwen2.5-Math (0%). Specialization
   on math destroys SI capability entirely.

4. **Voting gain is proportional to underlying signal.** SRM: +1 pt, Qwen2.5-
   0.5B: +17 pts. Voting needs real correctness signal in the completion
   distribution to have aggregation material.

## Reproduction

```bash
ssh -A castor.local "cd ~/src/small-reasoning-model && \
    PYTHONUNBUFFERED=1 HF_TOKEN=\$(cat ~/.hf_token) \
    ~/baseline_venv/bin/python eval/baseline_eval.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --suite both \
    --math_data data/grpo/grpo_filtered_v5.jsonl \
    --n_problems 100 \
    --group_size 8 \
    --output_dir results"
```

Repeat with `--model` set to each peer. Takes approximately 20-40 minutes
per model on a DGX Spark GB10.
