"""
nki_attention.py
================
Trainium2 NKI (Neuron Kernel Interface) attention kernel.

Hand-tuned for head_dim=128, which maps Q@K^T and A@V directly onto the
128×128 systolic array tile in the NeuronCore SBUF — no padding waste.

Key design points (spec Section 5.1):
  - SBUF size: 24MB per NeuronCore — fits full Q/K/V tile for one head at seq_len ≤ 4096
  - DMA transpose: K cache stored [seqlen, head_dim]; transposed on-the-fly before TensorE
  - FP8 path: TensorE presents as 256×128 for FP8; used for forward pass

Status: STUB — implement once 500M validation run confirms architecture on CUDA.

Dependencies:
  pip install neuronx-nki  (AWS Neuron SDK, Trainium instances only)
"""

# NKI is only available on Trainium instances.
# This module is a no-op on CUDA / CPU and should only be imported on Trn2.

try:
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl
    _NKI_AVAILABLE = True
except ImportError:
    _NKI_AVAILABLE = False


def nki_attention_available() -> bool:
    return _NKI_AVAILABLE


def nki_flash_attention(q, k, v, causal: bool = True):
    """
    NKI flash attention kernel for head_dim=128 on Trainium2.

    Args:
        q: (batch, n_heads, seq_len, head_dim=128)
        k: (batch, n_kv_heads, seq_len, head_dim=128)
        v: (batch, n_kv_heads, seq_len, head_dim=128)
        causal: apply causal mask

    Returns:
        out: (batch, n_heads, seq_len, head_dim=128)

    TODO: Implement NKI kernel using nl.matmul tiling on SBUF.
          Reference: AWS NKI flash attention tutorial.
    """
    raise NotImplementedError(
        "NKI attention kernel not yet implemented. "
        "Use F.scaled_dot_product_attention on CUDA (FlashAttention-2 path). "
        "Implement this after 500M CUDA validation run."
    )
