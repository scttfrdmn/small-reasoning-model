"""
preprocess.py
=============
Data filtering, deduplication, and curriculum mixing for Phase 0 pre-training.

Spec (Section 3.1):
  Quality filtering:
    - Perplexity filter: remove documents where GPT-2 perplexity > 10,000
    - Deduplication: MinHash LSH at 3-gram level, Jaccard threshold 0.8

  Data curriculum (proportion by training stage):
    0–30% tokens:  FineWeb-Edu / DCLM (40%), Stack v2 (25%), OpenWebMath (25%), Wikipedia/Books (10%)
    30–100% tokens: same mix + NuminaMath, math upweighted to 35%
    Final 10%:     40% math, 40% code, 20% general

Status: STUB
"""

# TODO: implement preprocessing pipeline
