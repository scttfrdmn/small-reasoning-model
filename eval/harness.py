"""
harness.py
==========
lm-evaluation-harness integration.

Wraps SmallReasoningModel in the lm_eval.api.model interface
so standard benchmarks can be run with:

  lm_eval --model small_reasoning --model_args checkpoint=./checkpoints/1b_sft/best.pt \\
          --tasks mathqa,gsm8k,hellaswag,arc_challenge,mmlu \\
          --num_fewshot 5

Status: STUB
"""

# TODO: implement lm_eval model wrapper
