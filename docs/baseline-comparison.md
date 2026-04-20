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
| SRM Phase 3 SFT (500m_v2_sft) | 500M | 0.020 | 0.050 | 0.230 | 0.035 |
| SRM Phase 3 GRPO (500m_v2_grpo) | 500M | 0.070 | 0.110 | 0.170 | 0.031 |
| Phi-1.5 | 1.3B | 0.010 | 0.030 | 0.160 | 0.024 |
| TinyLlama-1.1B-Chat | 1.1B | 0.020 | 0.030 | 0.100 | 0.016 |
| Qwen2.5-0.5B-Instruct | 500M | 0.340 | 0.510 | 0.600 | 0.339 |
| Qwen2.5-Math-1.5B-Instruct | 1.5B | 0.640 | 0.720 | 0.760 | 0.674 |

## Structured Intent Results

| Model | Size | JSON pass@1 | JSON pass@8 | Fields pass@1 | Fields pass@8 | Mean format | Mean fields |
|-------|------|-------------|-------------|---------------|---------------|-------------|-------------|
| SRM GRPO (no SI SFT) | 500M | 0.143 | 0.629 | 0.000 | 0.057 | 0.088 | 0.038 |
| SRM Phase 3 GRPO (no SI SFT) | 500M | 0.029 | 0.143 | 0.000 | 0.000 | 0.014 | 0.000 |
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

## Phase 3 Results (500m_v2_sft, 2026-04-19)

Phase 3 re-ran pre-training from scratch on the production tokenizer with a
math-heavy curriculum (48% openwebmath + numinamath, 44% fineweb-edu, 8%
wikipedia/misc) for 10B tokens (9,536 steps, val_loss ended at 2.68). SFT
was then run from that checkpoint on 100K examples from `sft_train_with_si.jsonl`
for 2 epochs (6,250 steps, best val_loss=0.7369).

**Phase 3 SFT math result vs Phase 1:**

| Checkpoint | pass@1 | pass@1 voted | pass@8 | reward |
|-----------|--------|--------------|--------|--------|
| Phase 1 SFT (500m_sft_v2) | 0.050 | 0.050 | 0.250 | 0.044 |
| **Phase 3 SFT (500m_v2_sft)** | **0.020** | **0.050** | **0.230** | **0.035** |

Phase 3 SFT is slightly below Phase 1 SFT despite better pre-training data.
Root cause: Phase 1 SFT trained on ~3.74M effective sequences (full 2M-example
dataset × ~1.9 epochs). Phase 3 SFT trained on 200K sequences (100K examples
× 2 epochs) — 18.7× less SFT data. The math-heavy pre-training helped but
did not compensate for the SFT data gap.

The voted pass@1 (5.0%) is identical between the two, which is evidence
that both models encode similar knowledge — Phase 3 SFT's stochastic output
is correct at the same aggregate rate when given enough samples.

**Implications:**
- Phase 3 pre-training + 100K SFT ≈ Phase 1 pre-training + 2M SFT at math
- Better pre-training data is not a substitute for SFT coverage at this scale

## Phase 3 GRPO Results (500m_v2_grpo, 2026-04-16)

GRPO was run for 5,000 steps on Phase 3 SFT (batch_prompts=2, group_size=8,
max_gen_tokens=512, best_reward=0.5000 at training time). 15,475 groups were
skipped (uniform reward, no gradient signal) — 75% skip rate vs ~50% in Phase 1,
reflecting the Phase 3 model's higher capability making most training problems
trivially easy or too hard.

**Phase 3 GRPO vs Phase 1 GRPO:**

| Checkpoint | pass@1 | pass@1 voted | pass@8 | reward |
|-----------|--------|--------------|--------|--------|
| Phase 1 GRPO (500m_grpo) | 0.040 | 0.050 | 0.290 | 0.050 |
| **Phase 3 GRPO (500m_v2_grpo)** | **0.070** | **0.110** | **0.170** | **0.031** |

**Per-domain (Phase 3 GRPO):**

| Domain | n | pass@1 | pass@8 | reward |
|--------|---|--------|--------|--------|
| numina_tir | 43 | 0.093 | 0.186 | 0.035 |
| gsm8k | 37 | 0.000 | 0.135 | 0.030 |
| math | 20 | 0.150 | 0.200 | 0.025 |

**Pattern:** Phase 3 GRPO raised pass@1 (4% → 7%) and voted pass@1 (5% → 11%),
confirming that math-heavy pre-training improved the model's tendency to get correct
answers. However, pass@8 fell sharply (29% → 17%) because GRPO concentrated the
output distribution — the model solves fewer distinct problems but solves its
solvable problems more reliably.

The distribution narrowing is visible in the ratio:
- Phase 1 GRPO: 29% of problems had *any* correct answer; when solvable, ~14% of
  samples were correct (pass@1/pass@8 = 0.040/0.290 ≈ 0.14)
- Phase 3 GRPO: 17% of problems had *any* correct answer; when solvable, ~41% of
  samples were correct (0.070/0.170 ≈ 0.41)

Voted pass@1 of 11% is also notably higher than pass@8 would predict from
random sampling (1-(1-0.07)^8 ≈ 43%), confirming that correct answers cluster
on specific problems rather than being evenly distributed.

**GSM8K regression:** GSM8K pass@1 dropped to 0.000 despite pass@8=0.135,
consistent with Phase 1 GRPO behavior where competition math specialization degraded
grade-school arithmetic. The GRPO training set (grpo_filtered_v5.jsonl) is
competition-math-heavy, which biases GRPO away from the arithmetic-chain style
GSM8K requires.

**SI regression (2026-04-20):** SI eval on Phase 3 GRPO best.pt shows a substantial
regression vs Phase 1 GRPO (no SI SFT):

| Checkpoint | JSON pass@1 | JSON pass@8 | Fields pass@1 | Fields pass@8 |
|-----------|-------------|-------------|---------------|---------------|
| Phase 1 GRPO (no SI SFT) | 0.143 | 0.629 | 0.000 | 0.057 |
| Phase 3 GRPO (no SI SFT) | 0.029 | 0.143 | 0.000 | 0.000 |

Phase 3 SFT was trained on `sft_train_with_si.jsonl` (which includes SI examples), but
5,000 steps of math-focused GRPO subsequently degraded the JSON format behavior the SFT
installed. The pattern is the same as the Qwen2.5-Math specialization finding: RL
optimizing for math format overwrites the output distribution away from structured JSON.
Only 2 of 7 task categories (validation, array) retain any JSON pass@8 signal.

To restore SI capability on the Phase 3 base, a targeted SI SFT pass is needed on the
Phase 3 checkpoint (either GRPO or SFT) — equivalent to what Phase 1 SFT+SI did.

**Implications:**
- Phase 3 GRPO cleared the blog-post-17 conservative estimate of 10-15% pass@1
  when measured by voting (11%), but not raw pass@1 (7%)
- Better pre-training data + GRPO gives 7% pass@1 vs 4% for Phase 1, a 75%
  improvement — real but not transformative
- The training data filter (grpo_filtered_v5.jsonl) was calibrated for Phase 1's
  capability; Phase 3's higher starting capability hits 75% skip rate, meaning
  most gradient signal comes from a small set of "goldilocks" problems
- SI capability requires explicit SI SFT after GRPO; GRPO consistently overwrites
  the JSON format distribution regardless of pre-training quality

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
