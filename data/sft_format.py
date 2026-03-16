"""
sft_format.py
=============
Reformat raw datasets into the <think>...</think> template for Phase 1 SFT.

Target format (spec Section 3.2):
  User: {problem}
  Assistant: <think>
  {step-by-step reasoning}
  </think>
  {final answer}

Datasets handled:
  - NuminaMath-CoT (~860K)
  - OpenHermes 2.5 (~1M filtered)
  - CodeFeedback (~66K)
  - Orca-Math (~200K)
  - Synthetic CoT (~200K, generated from base model)

Status: STUB
"""

# TODO: implement dataset reformatting per source
