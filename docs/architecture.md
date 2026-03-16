# Architecture Deep Dive

This document explains the model architecture in detail, with rationale for every design choice.
For the authoritative parameter tables, see [`small-reasoning-model-spec.md`](../small-reasoning-model-spec.md).
For the implementation, see [`model/architecture.py`](../model/architecture.py).

---

## Block Design

The model is a standard **pre-norm transformer decoder**. No departures from the 2024–2025 consensus.

```
Input tokens (B, T)
  └─ Embedding → (B, T, d_model)
      └─ × L transformer blocks:
          ├─ RMSNorm(x)
          ├─ GQA Attention(normed_x)  ← QK-Norm applied inside, before RoPE
          ├─ x = x + attn_out         ← residual connection
          ├─ RMSNorm(x)
          ├─ SwiGLU FFN(normed_x)
          └─ x = x + ffn_out          ← residual connection
      └─ Final RMSNorm → (B, T, d_model)
      └─ LM Head (tied to embedding) → (B, T, vocab_size)
```

### Why pre-norm?

Pre-norm places the normalization *before* each sublayer, inside the residual path. The residual
stream carries the full signal and gradients flow cleanly regardless of depth. Post-norm (as used
in original Transformer and OLMo) normalizes *after* adding the residual — it can achieve slightly
better final perplexity but is harder to stabilize during early training, especially at small scale
where initialization variance is a larger fraction of the signal. Pre-norm is the standard choice
for training runs that start from random initialization.

---

## Tile Alignment — The Trainium2 Constraint

Every matrix dimension in this model is a multiple of 128. This is a **hardware constraint**,
not a stylistic preference.

The Trainium2 NeuronCore contains a 128×128 systolic array (BF16) or a logical 256×128 array (FP8).
Every `nn.Linear` operation ultimately becomes a matrix multiply on this array. If a dimension
is not a multiple of 128, the compiler pads it — wasting a fraction of every matmul. For a
model that runs millions of these operations, the waste accumulates into a significant throughput
hit.

The same alignment benefits llama.cpp's GGUF quantization: GGUF uses 32-element quantization
blocks, and dimensions that are multiples of 128 (= 4 × 32) never have remainder blocks that
require special-case handling.

`ModelConfig.__post_init__` asserts all alignment constraints at initialization time. A misconfigured
model fails immediately with a clear error rather than silently wasting compute.

---

## Component Details

### RMSNorm

Root Mean Square Layer Normalization. Simpler and faster than LayerNorm — no mean subtraction,
only RMS scaling.

```
RMSNorm(x) = x / sqrt(mean(x²) + ε) * γ
```

Where γ is a learned per-dimension scale parameter, initialized to 1 (identity). The ε prevents
division by zero and adds numerical stability near zero activations.

**Why RMSNorm over LayerNorm?** Empirically equivalent quality at this scale, with less compute.
The mean-centering in LayerNorm doesn't help much when the subsequent linear layer can absorb any
constant offset. Every modern LLM (Llama, Qwen, Mistral) has converged on RMSNorm.

RMSNorm appears three times per block: before attention, before FFN, and once after the final block.

---

### Rotary Position Embedding (RoPE)

RoPE encodes position as a rotation in 2D subspaces of the head dimension. The key property:
the inner product `<q_i, k_j>` depends only on `(i - j)` — relative position — not absolute
positions. This means the model generalizes to sequence lengths it hasn't seen in training.

**Base frequency = 500,000.** Standard RoPE uses base=10,000, which degrades past ~4k tokens
as the high-frequency components complete too many full rotations. Base=500,000 extends clean
relative position encoding to 32k+ tokens, which is necessary for long chain-of-thought sequences
in GRPO training.

**Applied after QK-Norm, before the dot product.** The ordering matters: QK-Norm normalizes
the Q and K vectors (preventing explosion), then RoPE rotates them (encoding position), then
the dot product computes attention scores. Applying RoPE before QK-Norm would mix position
information into the normalization denominator in an undesirable way.

---

### Grouped Query Attention (GQA)

GQA uses fewer Key and Value heads than Query heads. Multiple Q heads share one K/V head.
This directly reduces the KV cache size — critical for inference memory.

| Config | Q heads | KV heads | GQA ratio | KV cache reduction |
|---|---|---|---|---|
| 500M | 10 | 2 | 5:1 | 5× |
| 1B | 16 | 4 | 4:1 | 4× |
| 3B | 24 | 6 | 4:1 | 4× |

At inference, the KV cache grows with sequence length. Without GQA, a 1B model generating a
16k-token chain of thought would hold 20 layers × 2 (K+V) × 16 heads × 16384 tokens × 128 dim
× 2 bytes (BF16) ≈ 1.6 GB just for the KV cache. With GQA at 4:1, this drops to 400 MB.

**Implementation:** The K and V tensors are expanded with `expand()` (zero-copy) before the
attention computation. No actual memory duplication.

---

### QK-Norm

QK-Norm applies RMSNorm independently to Q and K, **per head**, before the scaled dot product.
This is separate from the block-level RMSNorm that normalizes the full residual stream.

**Why it's critical at small scale:** At large scale, the model's weight matrices are initialized
with small variance relative to the accumulated signal. At 500M–3B parameters, the weight
initialization variance is a larger fraction of the signal, and attention logits can explode
in early training — producing all-zero softmax outputs (attention entropy collapse). QK-Norm
bounds the magnitude of Q and K before the dot product, preventing this.

This is now standard in recent small models (SmolLM3, Qwen3, Gemma). The spec cites gallery
evidence for its necessity.

---

### SwiGLU FFN

The feed-forward network uses SwiGLU gating:

```
SwiGLU(x) = SiLU(gate(x)) * up(x)
output     = down(SwiGLU(x))
```

Three weight matrices: `gate`, `up`, and `down`. The gated structure gives the FFN a
multiplicative interaction that standard two-layer FFNs lack. Empirically this improves
quality at fixed parameter count. All modern LLMs use some variant of gated linear units.

**No bias terms** throughout the model — in projections, FFN, or anywhere else. Bias terms
add parameters without meaningfully improving quality at this scale, and they interact
poorly with INT4/INT8 quantization (the quantization grid is calibrated per-channel for
weights; biases require separate handling).

**ffn_intermediate values** (3456, 5504, 8192) are pre-computed to be simultaneously:
- Approximately 2.7× d_model (the SwiGLU-optimal ratio for parameter budget equivalence
  with a standard 4×d FFN using two matrices instead of three)
- A multiple of 128 (tile-aligned)

---

### Tied Embeddings

The LM head reuses the input embedding weight matrix, transposed. No separate parameters.

**Why:** Saves `vocab_size × d_model` parameters. For vocab=32768 and d_model=2048, this
is 67M parameters — 7% of the total 1B budget. Tied embeddings also improve sample efficiency
because every gradient update to the LM head simultaneously updates the embedding table and
vice versa, effectively doubling the training signal for the vocabulary representation.

---

## Weight Initialization

Weights follow the GPT-2 / Llama convention:

- Embedding: N(0, 0.02)
- Linear weights: N(0, 0.02)
- Output projections (`o_proj`, `down_proj`): N(0, 0.02 / sqrt(2 × n_layers))
- RMSNorm γ: 1.0 (identity at init)

**Why scale output projections?** Each residual block adds its output to the residual stream.
If all L blocks add contributions of similar magnitude, the residual stream variance grows
with depth. Scaling each block's output by `1/sqrt(2L)` keeps the total residual stream
variance at initialization approximately constant regardless of depth. This is the "GPT-2
residual scaling" trick and it's essential for training stability at 20–28 layers.

---

## Special Token IDs

These are fixed by the tokenizer training order:

| Token | ID | Purpose |
|---|---|---|
| `<pad>` | 0 | Padding — excluded from loss via `ignore_index=0` |
| `<bos>` | 1 | Beginning of sequence |
| `<eos>` | 2 | End of sequence |
| `<unk>` | 3 | Unknown (byte fallback means this should never appear) |
| `<think>` | 4 | Begin chain-of-thought block |
| `</think>` | 5 | End chain-of-thought block |

The `<think>` / `</think>` tokens are first-class vocabulary entries, not ad-hoc strings. The
model is trained to generate reasoning chains bounded by these tokens. At inference, content
between them can be streamed, suppressed, or logged separately.
