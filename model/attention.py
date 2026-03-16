"""
attention.py
============
GQA + QK-Norm + RoPE attention components.

These are defined in model/architecture.py and re-exported here
for direct import convenience.

  from model.attention import GroupedQueryAttention, RotaryEmbedding, RMSNorm
"""

from model.architecture import GroupedQueryAttention, RotaryEmbedding, RMSNorm

__all__ = ["GroupedQueryAttention", "RotaryEmbedding", "RMSNorm"]
