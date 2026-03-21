# Architecture: Every Decision, Every Trade-off

*Part 2 of a series on building a small reasoning language model end-to-end.*

---

The architecture is a standard pre-norm transformer decoder with grouped query
attention, SwiGLU FFN, RoPE positional encoding, and QK-Norm. Every component
of that sentence is a choice that could have gone differently.

This post explains what we chose, what we didn't choose, and why — in enough
detail that you could defend or challenge each decision.

---

## The Block Structure

```
Input → Embedding
  └─ × L layers:
      ├─ RMSNorm
      ├─ GQA Attention (with QK-Norm, RoPE)
      ├─ Residual
      ├─ RMSNorm
      ├─ SwiGLU FFN
      └─ Residual
  └─ RMSNorm
  └─ LM Head (tied to embedding weights)
```

This is the GPT/LLaMA/Mistral/Qwen family architecture. The reason every major
open-weight model has converged to approximately this is not laziness — it's
that these components have been individually validated at scale and their
interactions are well understood.

We describe this as a "consensus architecture" because a broad research consensus
has emerged around it. Departing from it is a bet on something being materially
better; we have no such bet.

---

## Normalization: Pre-Norm RMSNorm

**What:** Layer normalization applied *before* the attention and FFN blocks,
not after (which would be "post-norm"). We use RMSNorm rather than LayerNorm.

**Why pre-norm:**
Training stability. Post-norm (original Transformer) requires careful learning
rate warmup because the residual connection bypasses normalization on the first
forward pass — gradients can explode. Pre-norm moves normalization inside the
residual branch, which provides a more uniform gradient magnitude throughout
training and allows higher learning rates.

Empirically: post-norm models often require LR schedules tuned per architecture.
Pre-norm models are more forgiving. At small scale where you're not running
ablations, forgiving is correct.

**Why RMSNorm instead of LayerNorm:**
RMSNorm (Zhang & Sennrich, 2019) drops the mean-centering step from LayerNorm
and only normalizes by the root mean square: `x / RMS(x) * γ`. This is:
- Faster (half as many operations)
- Equally stable in practice
- Cleaner to quantize (no mean subtraction to fold in)

> **Sidebar: What Does Normalization Do?**
>
> Neural networks are sensitive to the scale of their intermediate activations.
> If a layer produces very large activations, the next layer sees a distorted
> input and gradients flow poorly. Normalization rescales activations to a
> consistent range before each major operation.
>
> LayerNorm computes, for each position in the sequence:
> `y = (x - mean(x)) / sqrt(var(x) + ε) * γ + β`
>
> where `γ` and `β` are learned scale and shift parameters.
>
> RMSNorm drops mean subtraction (the `-mean(x)` term) and the shift `β`:
> `y = x / sqrt(mean(x²) + ε) * γ`
>
> The mean-centering is often argued to not matter empirically — the network
> learns to center itself via the learned `β` parameter anyway. Llama, Qwen,
> Mistral all use RMSNorm. So do we.
>
> *Reference: Zhang & Sennrich (2019), "Root Mean Square Layer Normalization"
> [arXiv:1910.07467](https://arxiv.org/abs/1910.07467)*

---

## Attention: GQA with QK-Norm

### Grouped Query Attention

**What:** GQA (Ainslie et al., 2023) uses fewer key/value heads than query heads.
Our 500M config has 10 query heads and 2 KV heads (5:1 ratio). The 1B config has
16 Q heads and 4 KV heads (4:1 ratio).

**Why:**
The KV cache is the inference bottleneck for autoregressive generation. At each
generation step, you need to store K and V for every previous token. With full
multi-head attention (MHA), KV cache size = `2 * n_heads * head_dim * seq_len *
bytes_per_element`. For a 1B model with 16 heads, generating 4096 tokens in BF16:
`2 * 16 * 128 * 4096 * 2 bytes = 33.5 MB per layer, × 20 layers = 671 MB`

GQA with 4 KV heads reduces this to `2 * 4 * 128 * 4096 * 2 = 8.4 MB per layer,
× 20 = 168 MB`. At batch size 8, that's the difference between fitting in memory
and not.

The quality trade-off is small. GQA matches MHA quality when the GQA ratio
(Q heads / KV heads) is kept to 4–8×. Larger ratios degrade quality.
Our ratios are all in the 4–5× range.

> **Sidebar: Multi-Head vs. Multi-Query vs. Grouped Query Attention**
>
> The original Transformer used Multi-Head Attention (MHA): `h` query heads,
> `h` key heads, `h` value heads. Each head learns a different subspace of the
> attention function.
>
> Multi-Query Attention (MQA, Shazeer 2019) collapses K and V to a single head
> each, reducing KV cache by `h×`. This saves memory but can hurt quality.
>
> Grouped Query Attention (GQA) is a middle ground: `h` Q heads, `h/g` KV heads
> for some group size `g`. Q heads within a group share the same K and V. This
> gives most of MQA's memory savings with much less quality degradation.
>
> The choice of `g` is a quality/memory trade-off. `g = 1` is MHA, `g = h` is MQA.
> We use `g = 4–5`, which is where the practical Pareto frontier sits.
>
> *Reference: Ainslie et al. (2023), "GQA: Training Generalized Multi-Query
> Transformer Models from Multi-Head Checkpoints" [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)*

### QK-Norm

**What:** Before computing attention scores `Q @ K^T`, we apply RMSNorm
independently to the Q and K projections — per head.

**Why:**
Attention logits can explode at small scale. At initialization, weight matrices
have variance `~1/d_model`. For a 500M model, `d_model = 1280`, so each Q/K
element has std `~1/√1280 ≈ 0.028`. The dot product of a Q vector and K vector
sums `head_dim = 128` such products, giving std `~128 * 0.028² ≈ 0.1` — small.

But during training, weights can grow. If Q and K norms grow to ~10 (not unusual
after 1B tokens), logits have std ~10,000. The softmax saturates — some attention
weights become 1.0 and others 0.0 — and gradients vanish for the saturated heads.
The model stops learning from those heads.

QK-Norm bounds the Q and K norms explicitly, preventing logit explosion without
requiring careful LR tuning or initialization tricks.

> **Sidebar: Why Does QK-Norm Help Small Models More?**
>
> The logit explosion problem scales with the ratio of weight growth to initialization
> scale. Large models have smaller initialization variance (divided by `sqrt(d_model)`)
> and more heads, so any single head's contribution to the attention pattern is smaller.
>
> Small models have larger initialization variance (relative to weights after training),
> fewer heads, and higher per-head signal-to-noise ratio. The combination makes logit
> explosion more likely during the early training phase.
>
> QK-Norm costs essentially nothing — it's a per-head normalization with a learned
> scale factor, negligible compute. The consensus from models like Gemma (Google, 2024),
> Phi-3, and others is that it's worth doing unconditionally for small models.
>
> Our implementation applies RMSNorm after Q and K projections, before RoPE:
> `q = rms_norm(q)`, `k = rms_norm(k)`, then RoPE, then scaled dot product attention.

---

## Positional Encoding: RoPE with Extended Base

**What:** Rotary Position Embedding (RoPE) with base frequency 500,000 (not the
original 10,000).

**Why RoPE over learned positions or ALiBi:**
RoPE encodes position as a rotation in embedding space. Two tokens at positions
`i` and `j` produce an attention score that depends on their *relative* position
`i - j`, not their absolute positions. This means the model can extrapolate to
sequences longer than it saw in training — not perfectly, but much better than
learned absolute positions, which completely fail out-of-distribution.

For a reasoning model, long-context handling matters: a multi-step math solution
inside `<think>` tags might be 1,000–2,000 tokens, and we want some context
remaining for the problem statement.

**Why base=500,000:**
The original RoPE used base=10,000. Higher base values spread the rotation
frequencies over a wider range, which empirically enables better long-context
generalization. Llama 3 uses 500,000. Qwen2 uses 1,000,000.

The intuition: with base=10,000, the highest-frequency rotation dimension
completes a full cycle every 10,000 positions. At base=500,000, it takes
500,000 positions. Higher-frequency rotation dimensions can distinguish nearby
positions more finely, while the low-frequency dimensions encode coarser
positional structure. A larger base shifts this spectrum toward coarser
encodings, which helps at long range.

> **Sidebar: How RoPE Works**
>
> Standard attention computes Q·K^T. If you encode position into Q and K via
> a rotation matrix R(pos), then Q(i)·K(j)^T = Q·R(i-j)·K^T — the position
> information becomes a function of *relative* position `i-j`.
>
> The rotation is applied dimension-pair by dimension-pair, using sine and cosine
> functions at different frequencies. Each pair of dimensions gets a different
> rotation speed, encoding different "scales" of relative position.
>
> RoPE doesn't add parameters. It's applied to Q and K after projection, before
> the dot product. The rotation is deterministic given the position index.
>
> *Reference: Su et al. (2023), "RoFormer: Enhanced Transformer with Rotary
> Position Embedding" [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)*

---

## FFN: SwiGLU

**What:** The feed-forward network uses SwiGLU: `FFN(x) = (xW₁ ⊙ swish(xW₂)) W₃`
where `⊙` is element-wise multiplication and `swish(x) = x * sigmoid(x)`.

**Why not ReLU or GELU:**
Standard FFN with GELU: `FFN(x) = GELU(xW₁) * W₂` — two weight matrices.
SwiGLU has three weight matrices (gate, up, down) but uses an element-wise
product as a gating mechanism. The gating allows the network to suppress
irrelevant dimensions entirely (multiplicative gating → true zeros, not small
values). This improves gradient flow and expressivity per parameter.

Empirically, SwiGLU (Noam Shazeer, 2020) improves perplexity vs. GELU with
the same total parameter budget. PaLM, LLaMA, Mistral, Qwen all use SwiGLU.

**Dimension adjustment:**
Because SwiGLU has 3 weight matrices instead of 2, we reduce the intermediate
dimension to keep total parameter count comparable. The standard ratio is
`ffn_intermediate = (2/3) * 4 * d_model`, rounded to a multiple of 128.
For `d_model=1280`: `(2/3) * 4 * 1280 = 3413 → 3456` (next multiple of 128).

> **Sidebar: Why Does the Gating Help?**
>
> ReLU says "if this activation is negative, kill it; otherwise pass it through."
> This creates sparse activations — many zero outputs — which is good for
> gradient flow but restricts the network to linear responses for active neurons.
>
> SwiGLU uses a *smooth* gate: `gate = sigmoid(linear(x))`, so outputs are
> scaled by a learned function of the input, not just zeroed. The gate can
> modulate outputs continuously, learning to up- or down-weight dimensions
> based on context.
>
> The intuition is that this makes the FFN more "selective" — it can strongly
> emphasize relevant dimensions and suppress irrelevant ones, giving each layer
> more effective information routing per parameter.
>
> *Reference: Shazeer (2020), "GLU Variants Improve Transformer"
> [arXiv:2002.05202](https://arxiv.org/abs/2002.05202)*

---

## Output: Tied Embeddings

**What:** The input embedding matrix `E` (maps token IDs to vectors) and the
output LM head matrix `W` (maps model outputs to token logits) are the same
tensor: `W = E^T`.

**Why:**
1. **Parameter count:** `vocab_size × d_model = 32768 × 1280 = 41.9M` parameters.
   Without tying, we'd have two such matrices (input + output) = 83.8M parameters.
   With tying, we save 41.9M parameters — about 9% of the total 500M model.
   At small model scale, 9% is significant.

2. **Semantic consistency:** The token embedding should encode the same semantics
   in both directions. A token that *represents* "math" should also *predict well*
   in contexts where "math" is the next token. Weight tying enforces this symmetry.

3. **Sample efficiency:** With a shared matrix, gradient signal from the language
   modeling objective simultaneously updates both the input representation and the
   output head. This improves sample efficiency on the vocabulary — especially
   important for rare tokens that appear infrequently in the corpus.

The trade-off: tying constrains the model slightly (the input and output
representations must serve double duty). For vocabularies of 32K tokens, this
is empirically negligible.

---

## No Bias Terms

All linear projections in attention and FFN have `bias=False`.

This is unconventional but justified:
1. **Quantization:** Bias terms require special handling during quantization
   (they're typically kept in FP32 or FP16 even when weights are INT4). Removing
   them eliminates this complexity.
2. **Normalization absorbs the role of bias:** Every linear projection is followed
   (or preceded) by RMSNorm, which has its own learned scale `γ`. The normalization
   layer acts as a dynamic per-activation scale, which subsumes what a fixed bias
   would do.
3. **Minimal quality impact:** Empirically, bias terms have negligible impact on
   quality for transformer decoders at this scale. LLaMA, Mistral, and Qwen all
   use `bias=False`.

---

## The Three Configurations

| | 500M (Config A) | 1B (Config B) | 3B (Config C) |
|---|---|---|---|
| `d_model` | 1280 | 2048 | 3072 |
| `n_layers` | 26 | 20 | 28 |
| `n_heads` (Q) | 10 | 16 | 24 |
| `n_kv_heads` | 2 | 4 | 6 |
| `head_dim` | 128 | 128 | 128 |
| `ffn_intermediate` | 3456 | 5504 | 8192 |
| `vocab_size` | 32768 | 32768 | 32768 |
| Parameters | ~489M | ~953M | ~2.87B |
| Training target | 10B tokens / 5090 | 50B tokens / Trn2 | 100B tokens / Trn2 |

**Why `head_dim = 128` across all configs:**
Head dimension is held constant at 128 while head count scales. This is a
Trainium2 constraint (the NKI kernel tiles exactly on 128-width heads) but also
matches the FlashAttention-2 implementation preference (SRAM-optimal at seq_len
up to ~4096 for head_dim=128 on Ampere/Ada/Blackwell).

**Why different `n_layers` for 1B vs 3B:**
The 3B model has more layers (28) than the 1B (20) with a larger d_model (3072
vs 2048). This is a depth-vs-width trade-off: deeper models generalize better
at equivalent parameter count, but each additional layer adds a minimum compute
overhead. At 3B parameters, the wider + deeper configuration uses the budget
better than going purely wide.

---

## What We Explicitly Chose Not To Do

**Sliding window attention:** No. Full attention is tractable at 1B parameters
with seq_len ≤ 8192. SWA adds implementation complexity (non-trivial with
KV caching, needs different kernel) for no clear benefit below ~7B parameters.

**Mixture of Experts (MoE):** No. Below ~7B *active* parameters, the routing
overhead and load-balancing complexity outweigh the efficiency gains. MoE is
a technique for trading training FLOPs for parameter count; we don't need that
trade-off at this scale.

**ALiBi or learned position embeddings:** No. RoPE with extended base handles
long context well and has the best empirical track record for reasoning models
where long `<think>` traces are common.

**Flash Attention in the model code:** We use `torch.nn.functional.scaled_dot_product_attention`,
which dispatches to FlashAttention-2 automatically on CUDA when available.
We don't implement our own CUDA kernel here — we reserve that effort for the
Trainium2 NKI attention kernel where the standard implementation doesn't exist.

---

## What This Looks Like in Code

The architecture is implemented in `model/architecture.py`. The core attention
block (simplified):

```python
# Q, K, V projections — GQA dimensions
q = self.q_proj(x)   # (B, T, n_heads * head_dim)
k = self.k_proj(x)   # (B, T, n_kv_heads * head_dim)
v = self.v_proj(x)   # (B, T, n_kv_heads * head_dim)

# QK-Norm: normalize before RoPE to prevent logit explosion
q = self.q_norm(q.reshape(B, T, n_heads, head_dim))
k = self.k_norm(k.reshape(B, T, n_kv_heads, head_dim))

# RoPE: apply rotary embeddings (after QK-Norm, per spec)
q = apply_rope(q, cos, sin)
k = apply_rope(k, cos, sin)

# Expand KV heads to match Q heads for GQA (repeat_interleave)
k = k.repeat_interleave(gqa_ratio, dim=2)
v = v.repeat_interleave(gqa_ratio, dim=2)

# Flash Attention via PyTorch dispatch
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

The full implementation including gradient checkpointing, KV caching for inference,
and the Trainium2 NKI path is in the repo.

---

*Next: [Part 3 — Hardware-First Design: Trainium2, Tile Alignment, and Quantization
by Construction](03-hardware.md)*
