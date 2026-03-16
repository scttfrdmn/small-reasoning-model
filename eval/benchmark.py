"""
benchmark.py
============
Run evaluation suite across checkpoints.

Benchmarks tracked (spec Section 7):
  MATH (Hendrycks)     — primary GRPO target
  GSM8K                — baseline reasoning sanity check
  HumanEval            — code generation
  ARC-Challenge        — science QA, out-of-domain
  HellaSwag            — commonsense regression check
  MMLU (5-shot)        — broad capability regression
  BIG-Bench Hard       — hard generalization

Key ratios:
  MATH vs HellaSwag    — GRPO should improve MATH without collapsing HellaSwag
  Pass@1 vs Pass@8     — large gap means high variance; close = consistent reasoning

Status: STUB — integrate with eval/harness.py (lm-evaluation-harness)
"""

# TODO: implement benchmark runner
