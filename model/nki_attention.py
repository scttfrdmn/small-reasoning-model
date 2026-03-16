"""
nki_attention.py
================
Trainium2 NKI (Neuron Kernel Interface) attention kernel.

Background — Trainium2 hardware terminology:
  NeuronCore: the primary compute unit on a Trainium2 chip. Each Trainium2 chip
    has two NeuronCores; each Trn2 instance (trn2.48xlarge) has 64 NeuronCores.

  SBUF (Scalar Buffer): the on-chip SRAM scratchpad local to each NeuronCore.
    24 MB per NeuronCore on Trainium2. This is the fast memory that NKI kernels
    operate on directly — equivalent to "shared memory" in CUDA terms, but much
    larger (CUDA shared memory is typically 48-96 KB per SM).
    All NKI tensor operations read from and write to SBUF; DMA moves data
    between HBM (off-chip DRAM) and SBUF.

  PSUM (Partial Sum buffer): a separate, smaller accumulator buffer (distinct from
    SBUF) dedicated to collecting partial dot-product results during a matmul.
    TensorE writes partial sums here as it processes tiles; the PSUM is flushed to
    SBUF once the full accumulation is complete. This decoupling of accumulation
    from the main data buffer prevents pipeline stalls.

  TensorE (Tensor Engine): the systolic-array matrix-multiply unit inside each
    NeuronCore. It is a 128×128 BF16 MXU (Matrix eXecution Unit) — equivalent to
    the Tensor Core in NVIDIA GPUs. One TensorE operation computes a 128×128 tile
    of a matrix product in a single cycle burst.

  DMA (Direct Memory Access): the engine that moves data between HBM and SBUF
    asynchronously, without tying up the TensorE. NKI kernels overlap DMA
    (loading the next tile) with TensorE (computing the current tile) to hide
    memory latency.

Hand-tuned for head_dim=128, which maps Q@K^T and A@V directly onto the
128×128 systolic array tile in the NeuronCore SBUF — no padding waste.

Key design points (spec Section 5.1):
  - SBUF size: 24MB per NeuronCore — fits full Q/K/V tile for one head at seq_len ≤ 4096
  - DMA transpose: K cache stored [seqlen, head_dim]; transposed on-the-fly before TensorE
  - FP8 path: TensorE presents as 256×128 for FP8; used for forward pass

Why head_dim=128 is the perfect tile size:
  The TensorE systolic array is exactly 128×128 elements wide for BF16. A single
  attention head's inner matmul is (seq_tile × head_dim) @ (head_dim × seq_tile).
  With head_dim=128, the head_dim dimension exactly fills one TensorE tile, so
  each head requires exactly one tile of TensorE work per seq_tile × seq_tile block.
  head_dim=64 would use only the left half of each tile row, wasting 50% of the MXU.
  head_dim=256 would require two consecutive TensorE operations per seq tile block
  and double the SBUF footprint for Q/K, reducing the maximum seq_tile size.
  head_dim=128 is the Goldilocks value for Trainium2.

What DMA transpose means and why it is needed:
  The K cache is stored in HBM in row-major order as [seq_len, head_dim] — i.e.
  consecutive head_dim values for a given position are contiguous in memory.
  The attention score computation requires K^T: [head_dim, seq_len].
  Transposing in SBUF after loading would require a separate shuffle operation.
  Instead, NKI exposes a DMA "transpose load" that reads K from HBM in
  [seq_len, head_dim] layout but writes it into SBUF as [head_dim, seq_len] —
  the transpose is done by the DMA engine during the memory transfer, overlapping
  with TensorE work on the previous tile. This eliminates the in-SBUF transpose
  cost entirely.

Status: STUB — implement once 500M validation run confirms architecture on CUDA.

Dependencies:
  pip install neuronx-nki  (AWS Neuron SDK, Trainium instances only)
"""

# NKI is only available on Trainium instances.
# This module is a no-op on CUDA / CPU and should only be imported on Trn2.
# The try/except allows the rest of the codebase to import this module on any
# hardware; nki_attention_available() lets callers decide whether to use the
# NKI path or fall back to F.scaled_dot_product_attention.
try:
    import neuronxcc.nki as nki
    import neuronxcc.nki.language as nl  # nl provides SBUF, DMA, TensorE primitives

    _NKI_AVAILABLE = True
except ImportError:
    _NKI_AVAILABLE = False


def nki_attention_available() -> bool:
    """Return True if the Neuron NKI SDK is installed and importable (i.e. running on Trn2)."""
    return _NKI_AVAILABLE


def nki_flash_attention(q, k, v, causal: bool = True):
    """
    NKI flash attention kernel for head_dim=128 on Trainium2.

    The NKI implementation would follow the FlashAttention-2 tiling algorithm:
      1. Tile the sequence dimension into blocks of size T_r (rows) and T_c (cols).
      2. For each (T_r, T_c) tile pair:
         a. DMA-load Q tile [T_r, head_dim] into SBUF.
         b. DMA-load K tile [T_c, head_dim] into SBUF; simultaneously transpose
            via DMA to get [head_dim, T_c] in SBUF (no extra SBUF shuffle needed).
         c. TensorE: S = Q @ K^T → [T_r, T_c] partial attention scores in PSUM.
         d. Online softmax: update running max and denominator (Dao 2022 trick).
         e. DMA-load V tile [T_c, head_dim] into SBUF.
         f. TensorE: accumulate O += softmax(S) @ V into output tile in PSUM.
      3. Flush PSUM → SBUF, DMA-store output tile to HBM.
    Causal masking is applied in step (d) by zeroing scores where col > row.

    Args:
        q: (batch, n_heads, seq_len, head_dim=128)
        k: (batch, n_kv_heads, seq_len, head_dim=128)
           n_kv_heads may be < n_heads (GQA); the kernel handles the broadcast.
        v: (batch, n_kv_heads, seq_len, head_dim=128)
        causal: apply causal mask (True for autoregressive generation / training)

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
