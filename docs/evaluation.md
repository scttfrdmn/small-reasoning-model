# Evaluation

This document describes the benchmarks used to evaluate the model at each training phase,
the metrics to watch, and how to interpret the results.

---

## Benchmark Suite

Track these at each checkpoint. Use `eval/harness.py` for integration with lm-evaluation-harness.

| Benchmark | Task type | Primary metric | Why it matters |
|---|---|---|---|
| MATH (Hendrycks) | Math reasoning (competition) | pass@1 | Primary GRPO target — must improve after Phase 2 |
| GSM8K | Grade school math | pass@1 | Baseline reasoning sanity check; should improve from Phase 1 |
| HumanEval | Code generation | pass@1 | Transfer from code pre-training and GRPO |
| ARC-Challenge | Science QA | accuracy | General reasoning, out-of-domain from training |
| HellaSwag | Commonsense completion | accuracy | Regression check — must not collapse during GRPO |
| MMLU (5-shot) | Multi-domain knowledge | accuracy | Broad capability regression check |
| BIG-Bench Hard (subset) | Algorithmic reasoning | accuracy | Hard generalization; not directly trained on |

---

## Key Ratios

### MATH vs HellaSwag

GRPO should improve MATH without collapsing HellaSwag. If HellaSwag drops significantly
(> 5 points) while MATH improves, the model is over-specializing — forgetting general
language representations in favor of math-specific patterns. This indicates too aggressive
GRPO training (LR too high, KL coefficient too low, or too many steps).

Target after Phase 2: MATH improves > 10 points, HellaSwag drops < 2 points.

### MATH Level 1–2 vs Level 4–5

GRPO should improve proportionally across difficulty levels. If only Level 1–2 improves
(easy problems the base model already almost solves) while Level 4–5 stays flat, the training
distribution is too easy — the 20–80% difficulty filter may need recalibration.

If Level 4–5 improves dramatically but Level 1–2 improves little, the model has learned
specific hard problem patterns but lost consistency on easy problems — a sign of overfitting
to the GRPO training distribution.

### Pass@1 vs Pass@8

- **Pass@1:** Fraction of problems solved on the first attempt (greedy/temperature=0)
- **Pass@8:** Fraction of problems solved in at least 1 of 8 samples (temperature=0.8)

A large gap between pass@8 and pass@1 means the model has learned the *capability* but not
the *consistency*. It can reason correctly but only sometimes. This is the expected state
mid-GRPO; the gap should narrow as training progresses.

A close pass@1 ≈ pass@8 gap means the model is consistent — it either reliably solves or
reliably fails a given problem type. This is the target state after GRPO completion.

---

## Phase-by-Phase Expectations

### After Phase 0 (Pre-training)

The model is a language model, not an instruction follower. Evaluation must use raw
completion (no chat template). Expect:

- GSM8K few-shot: ~10–25% (mostly memorized patterns, not real reasoning)
- HellaSwag: ~55–70% (depends on token budget; more tokens = higher)
- MATH: ~2–8% (mostly lucky guesses; the base model can't reliably solve competition math)

These are baselines. Low numbers here are expected and fine.

### After Phase 1 (SFT)

The model now follows instructions and produces `<think>…</think>` format. Expect:

- GSM8K: +10–20 points (CoT format dramatically helps grade school math)
- MATH: +3–10 points (step-by-step reasoning helps; competition math is still hard)
- HellaSwag: roughly flat (SFT should not degrade general language)
- MMLU: roughly flat or slight improvement (instruction following helps 5-shot)

### After Phase 2 (GRPO)

Reasoning capability is reinforced on verified domains. Expect:

- MATH: +15–30 points above SFT (the main target; this is why GRPO exists)
- GSM8K: +5–15 points (GRPO generalizes to easy math)
- HumanEval: +5–15 points (code reasoning transfers)
- HellaSwag: < 3 point drop (KL penalty protects general capability)
- MMLU: < 3 point drop

---

## GRPO Early Stopping Criterion

Monitor **pass@1 on held-out MATH Level 3** problems every 100 steps. Stop training when:
- Validation pass@1 has not improved for **500 consecutive steps**, or
- 20,000 total GRPO steps have been reached

Level 3 is the sweet spot for the stopping criterion:
- Level 1–2 saturates quickly (base model almost solves these)
- Level 4–5 may never improve significantly at 1B scale
- Level 3 is the frontier where continued training has measurable, meaningful effect

---

## Running Evaluation

Once `eval/harness.py` is implemented:

```bash
# Full benchmark suite on a checkpoint
uv run python eval/benchmark.py \
  --checkpoint ./checkpoints/1b_grpo/best.pt \
  --config 1b \
  --tasks math,gsm8k,humaneval,arc_challenge,hellaswag,mmlu

# Quick sanity check (faster benchmarks only)
uv run python eval/benchmark.py \
  --checkpoint ./checkpoints/1b_sft/best.pt \
  --config 1b \
  --tasks gsm8k,hellaswag
```

For MATH pass@8, set `--num_samples 8 --temperature 0.8`.

---

## Comparison Baselines

When reporting results, compare against:

| Model | Params | MATH | GSM8K | HumanEval | Notes |
|---|---|---|---|---|---|
| Qwen3-0.6B | 0.6B | TBD | TBD | TBD | Smallest strong open model |
| Qwen3-1.7B | 1.7B | TBD | TBD | TBD | Closest size to our 1B |
| SmolLM3-3B | 3B | TBD | TBD | TBD | Closest to our 3B |
| Our 500M (post-GRPO) | ~489M | target | target | target | Validation config |
| Our 1B (post-GRPO) | ~953M | target | target | target | Primary experiment |
| Our 3B (post-GRPO) | ~2.87B | target | target | target | Full experiment |

The key question: does a model *designed* for small-scale reasoning beat a general model
of the same or larger size on math and code, at equivalent or lower inference cost?
