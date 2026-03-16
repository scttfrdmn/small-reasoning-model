"""
grpo_dataset.py
===============
Difficulty-filtered dataset of verifiable problems for Phase 2 GRPO.

Spec (Section 3.3):
  Filter to problems where the SFT model solves 20–80% correctly.
  Problems outside this range produce no useful gradient signal:
    > 80% pass rate: too easy — reward is 1 for all completions, advantage = 0
    < 20% pass rate: too hard — reward is 0 for all completions, advantage = 0

  Sources:
    - MATH (Hendrycks) ~12K, levels 1–5
    - NuminaMath ~860K
    - GSM8K ~8K
    - MBPP / HumanEval ~1K
    - LogiQA / FOLIO ~10K

Status: STUB
"""

# TODO: implement difficulty-filtered dataset builder
