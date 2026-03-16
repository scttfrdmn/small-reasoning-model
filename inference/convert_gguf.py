"""
convert_gguf.py
===============
Export trained model to GGUF format for llama.cpp inference.

Target formats (spec Section 6):
  BF16      ~2 GB    reference; for eval
  Q8_0      ~1 GB    near-lossless; Graviton4, any 2GB+ device
  Q4_K_M   ~700 MB   recommended default; Graviton4 / Kamrui cluster
  Q4_0     ~550 MB   edge deployment; Raspberry Pi 5
  Q2_K     ~400 MB   curiosity only

Tile-aligned dimensions (all ÷128) map cleanly to GGUF's 32-element
block quantization with no remainder handling.

Graviton4 inference estimate (1B Q4_K_M):
  c8g.4xlarge:  ~25–35 tokens/sec
  c8g.8xlarge:  ~60–80 tokens/sec

Status: STUB — implement after first full training run produces a checkpoint.
"""

# TODO: implement GGUF export
# Reference: llama.cpp convert_hf_to_gguf.py for weight layout conventions
