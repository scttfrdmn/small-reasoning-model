# What Determines Context Length? A Systems-Level Explanation

*Part 10 of a series on building a small reasoning language model end-to-end.*

*This post is a deep dive into one of the most frequently misunderstood properties
of language models: context length. Why can GPT-4 handle 128K tokens? Why does
our 500M model use 2048 for pre-training and 4096 for SFT? Why does a model's
context length matter at deployment time in a completely different way than it
does during training?*

---

Context length (also called "context window") is the maximum number of tokens
a model can attend to at once. It appears as a single number — "4096 tokens",
"128K tokens" — but that number is the intersection of at least four different
constraints operating at different layers of the stack: **memory**, **compute**,
**positional encoding**, and **training data distribution**.

Get any one of them wrong and your context length either doesn't work as
advertised, works but slowly, or works at inference but never appeared during
training (so the model doesn't know how to use it).

We'll use our 500M model as the concrete example throughout.

---

## The Four Constraints

### 1. Memory: The KV Cache

During autoregressive generation, the model generates one token at a time.
To generate token `t`, it needs to attend to all previous tokens `0...t-1`.
The key (K) and value (V) vectors for those previous tokens must be in memory.

This is the **KV cache** — a store of K and V tensors for every layer, for
every position in the context.

KV cache size (bytes) for a single sequence:
```
2  (K and V)
× n_kv_heads  (GQA reduces this)
× head_dim
× n_layers
× context_length
× bytes_per_element
```

For our 500M model at inference (BF16):
```
2 × 2 × 128 × 26 × context_length × 2 bytes
= 26,624 × context_length bytes
```

At `context_length = 2048`: **54 MB**
At `context_length = 8192`: **218 MB**
At `context_length = 32768`: **873 MB**

On the RTX 5090 with 32GB VRAM:
- Model weights: ~1 GB
- Activations: ~200 MB (for inference, much less than training)
- KV cache at 8192: 218 MB
- Everything fits easily — even at 32K context

But wait — this is *per sequence*. For batch size 8 at context_length 8192:
`8 × 218 MB = 1.7 GB`. Still fine. For batch size 32: `6.9 GB`. Getting tight.
For batch size 100: `21.8 GB` — plus model weights and activations, now you're
near the 32GB limit.

**Context length is cheap per sequence at inference. It scales with batch size.**

> **Sidebar: Why GQA Cuts KV Cache by the GQA Ratio**
>
> Multi-head attention (MHA) has `n_heads` independent K/V projections —
> one per query head. If you have 10 query heads, you store 10 K vectors and
> 10 V vectors per position per layer.
>
> GQA with `n_kv_heads = 2` (and `n_heads = 10`) means Q heads 0-4 share
> K/V head 0, and Q heads 5-9 share K/V head 1. You store only 2 K and 2 V
> vectors per position per layer — a 5× reduction.
>
> For our 500M model: 2 KV heads vs. 10 MHA heads = 5× smaller KV cache.
> At 32K context, 2 KV heads = 873 MB. With 10 KV heads: 4.4 GB. The
> difference matters for long-context batch serving on Orin NX (16GB total).

### 2. Memory: Training Activations

Training requires keeping activations in memory for the backward pass. This is
fundamentally different from inference.

During training, for every forward pass we need to retain the intermediate
activations for each layer so that we can compute gradients. For a single sequence
of length `T`:

**Attention activations** (the attention score matrix) scale as O(T²):
each layer stores a `(T × T)` matrix. At T=2048: `2048² = 4M` elements per
layer × 26 layers × 2 bytes = **426 MB**.

At T=4096: `4096² = 16.8M` elements × 26 × 2 = **1.7 GB** — 4× more for 2×
the context length. Quadratic scaling.

This is why we use **gradient checkpointing**: instead of storing all activations,
we discard them during the forward pass and recompute them as needed during
backward. This reduces activation memory from O(T²) to O(T√layers), at the cost
of ~20% extra compute.

Even with gradient checkpointing, training at very long context lengths requires
large batch sizes to be infeasible. The pre-training recipe uses:
- `max_seq_len = 2048` — short enough to train with batch_size=4
- SFT uses `max_seq_len = 4096` — needs gradient checkpointing

**Training context length is memory-limited in a way inference is not.**

### 3. Compute: Attention is O(T²)

Attention computes pairwise scores between every query position and every key
position. This is an O(T²) operation in both compute and memory.

For a single layer:
- Q, K, V projections: O(T × d_model) — linear, fast
- Attention scores `QK^T`: O(T² × head_dim) per head — quadratic
- Weighted values `AV`: O(T² × head_dim) per head — quadratic
- Output projection: O(T × d_model) — linear, fast

At T=2048, the O(T²) terms dominate. At T=8192, they're 16× more expensive
than at T=2048. At T=32768 (32K context), they're 256× more expensive.

**FlashAttention** addresses this for GPU memory (it doesn't increase VRAM
for the attention matrix) but the compute cost is still O(T²). You're still
doing the same number of multiply-accumulate operations; you're just doing them
in a more cache-efficient order that reduces memory reads/writes.

For our 500M model: the non-attention operations (FFN, projections) scale as
O(T). The attention operation scales as O(T²). At T=2048, attention is roughly
30% of the compute. At T=8192, it's roughly 75%. At T=32768, it dominates
everything.

This is why models with 1M+ context lengths typically use sparse or linear
attention for the extreme-length regime — full quadratic attention at 1M tokens
is intractable.

> **Sidebar: FlashAttention and IO-Aware Computation**
>
> Standard attention implementations compute `softmax(QK^T/√d) V` by:
> 1. Computing the full `(T × T)` score matrix and writing it to GPU VRAM
> 2. Computing softmax over it
> 3. Multiplying by V, writing the result
>
> The T×T matrix can be enormous. At T=8192 with BF16: `8192² × 2 = 134 MB`
> per layer per head. This fits in VRAM but requires multiple passes, and
> each pass reads/writes large tensors — the bottleneck is VRAM bandwidth.
>
> FlashAttention (Dao et al., 2022) reorganizes the computation: process the
> sequence in chunks that fit in L2 cache, computing partial attention scores
> and accumulating them without ever materializing the full T×T matrix. The
> number of FLOPS is identical; the VRAM traffic is dramatically reduced.
>
> The result: FlashAttention is 2–4× faster than standard attention and has
> O(T) VRAM usage for the attention matrix instead of O(T²). The O(T²) *compute*
> cost remains. We use `F.scaled_dot_product_attention` which dispatches to
> FlashAttention-2 on CUDA automatically.
>
> *Reference: Dao et al. (2022), "FlashAttention: Fast and Memory-Efficient
> Exact Attention with IO-Awareness" [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)*

### 4. Positional Encoding: What Lengths Has the Model Actually Seen?

This is the most subtle constraint, and the one most often ignored until
inference breaks.

A transformer has no inherent notion of position — without positional encoding,
attention is permutation-invariant (the order of tokens doesn't matter). Positional
encoding injects position information so the model can distinguish "the cat sat
on the mat" from "the mat sat on the cat".

There are several approaches, with very different behavior at lengths beyond
what was seen during training:

**Learned absolute positions:** An embedding table, one entry per position.
At `max_seq_len = 2048`, you have 2048 learned position embeddings. Position
2049 has no embedding — it's undefined. These models *hard-fail* outside their
training context length.

**Relative position encodings (ALiBi, T5-style):** Encode relative distance
between positions as a bias to attention scores. Can extrapolate to longer
sequences than training, but with varying quality.

**RoPE (Rotary Position Embedding):** Encodes position as a rotation in
embedding space. Attention scores become functions of *relative* position.
In theory, RoPE can extrapolate indefinitely; in practice, it extrapolates
reasonably well but degrades at lengths much longer than training.

We use **RoPE with base=500,000**. The base frequency controls how fast the
rotation angles change with position. Higher base = slower rotation change =
better long-context behavior. The standard base=10,000 was found to degrade
significantly beyond training length. Base=500,000 (Llama 3's choice) gives
useful extrapolation to roughly 2–4× the training context length.

**Why this matters for our model:**

- Pre-training: `max_seq_len = 2048`
- SFT: `max_seq_len = 4096`
- Inference target: up to 8192

At inference, with base=500,000 RoPE, we expect reasonable quality at 8192 tokens
even though pre-training only saw up to 2048. SFT at 4096 further extends this.

At 16384 (8× pre-training length), quality will degrade noticeably. At 32768,
it's likely to be incoherent.

> **Sidebar: Why Does RoPE Extrapolate At All?**
>
> RoPE encodes position by rotating Q and K vectors before computing attention:
> `score(i, j) = f(i-j)` — a function of *relative* position only.
>
> For positions within training range, the model has learned the rotation
> patterns that make the attention function work well. For positions beyond
> training range, the rotation angles continue to be computed (RoPE is a
> deterministic function, not a lookup table), but the model has never been
> trained to use them correctly.
>
> Base=500,000 spreads the rotation frequencies over a wider range, meaning
> the "fast" dimensions (which would cycle many times within training range)
> cycle less often — more training positions see similar rotation values for
> those dimensions, improving generalization.
>
> For our project: if we later want to support 32K context natively, we'd
> need to either (a) pre-train with 32K sequences, (b) use YaRN or LongRoPE
> context extension techniques, or (c) fine-tune on long-context examples.
> These are well-studied extensions we defer to a later version.
>
> *Reference: Peng et al. (2023), "YaRN: Efficient Context Window Extension
> of Large Language Models" [arXiv:2309.00071](https://arxiv.org/abs/2309.00071)*

---

## Our Specific Choices: Why These Numbers?

### Pre-training: max_seq_len = 2048

The pre-training corpus is 7M documents averaging ~1400 tokens each. Most
documents fit within 2048 tokens with minimal truncation. Going to 4096 would:
- Double attention compute per step
- Require halving the batch size or using more gradient accumulation
- Not significantly benefit the 9.86B token training budget (more tokens
  per step, but fewer steps — total compute roughly the same)

We chose 2048 as the sweet spot: most documents aren't truncated, attention
is fast, and we can use larger effective batches.

### SFT: max_seq_len = 4096

SFT data has a different distribution. A NuminaMath-CoT example looks like:

```
User: [problem, ~200 tokens]
Assistant: <think>
[step-by-step solution, 400–2000 tokens]
</think>
[final answer, 20–100 tokens]
```

The `<think>` block alone can be 2000 tokens. With the prompt and answer,
the full sequence easily exceeds 2048. Truncating CoT traces mid-reasoning
creates broken training examples — the model learns to start reasoning chains
that lead nowhere.

At 4096: most SFT examples fit completely. The cost (4× the memory for the
attention matrix, gradient checkpointing required) is worth it for the quality
improvement in reasoning trace training.

### Inference target: up to 8192

During inference, a user asks a question and the model generates a response.
With reasoning, the response includes a `<think>` block of potentially 2000+
tokens, followed by the final answer.

The user's question might also be long: a math problem, a code snippet to debug,
a document to analyze. It's reasonable to expect inputs of 500–2000 tokens.

Total: `input (2000) + <think> (2000) + answer (100) = 4100 tokens`. So 4096
is cutting it close. 8192 gives comfortable headroom.

With RoPE base=500,000 and SFT training at 4096, inference quality at 8192 is
expected to be good. This is 2× the SFT context length and 4× the pre-training
context length — within the useful extrapolation range.

---

## Context Length Across the Hardware Fleet

The same model behaves differently in terms of context length constraints
depending on where it's running.

### ceres (RTX 5090, 32GB)

At inference with batch size 1:
```
Model weights (500M BF16):   ~1 GB
KV cache at 8192 context:    ~218 MB
Activations:                 ~50 MB
Total:                       ~1.3 GB
```
32GB is huge for batch-1 inference. Context length is limited by RoPE
extrapolation quality, not memory. You could run 32K context on this GPU
from a memory standpoint; the quality would just degrade beyond ~8K due to
RoPE limits.

At batch size 32 with 8192 context:
`32 × 218 MB KV cache = 7 GB`. Fine.
At batch size 100: `21.8 GB KV cache`. Near the edge with weights loaded.

### vesta (RTX 4070 Ti SUPER, 16GB)

Same math, half the budget. At batch size 32, 8192 context: `7 GB KV cache + 1 GB model = 8 GB`. Fine, plenty of headroom. Even at batch 64, context 8192: `~15 GB`. Tight but works.

16GB is not a practical constraint for this model size at these context lengths.

### janus (2× Titan RTX, 48GB NVLink)

For inference at 48GB effective VRAM (tensor-parallel across both GPUs):
The KV cache is split across both GPUs in tensor-parallel mode. At 32K context
and batch 64: `64 × 26,624 × 32768 = 55 GB`. Exceeds 48GB.

At 8192 context, batch 64: `13.7 GB KV cache + 2 GB model = 15.7 GB`. Fine.
The 48GB budget opens up very high-batch-size scenarios at shorter contexts.

### castor / pollux (DGX Sparks, 128GB unified each)

Inference on a single DGX Sparks with the 3B model at 128K context (speculative):
```
3B model BF16:                ~6 GB
KV cache 3B at 128K (8 KV heads, 28 layers):
  2 × 8 × 128 × 28 × 131072 × 2 = 15.1 GB
Total:                        ~21 GB
```
128GB has 107GB headroom above this. You could run batch size 6 at 128K context
on a single DGX Sparks with the 3B model.

For the 500M model at 128K context, batch 100:
```
Model: 1 GB
KV cache: 100 × 26,624 × 131072 × 2 = 698 GB ... that's too much.
```
Even 128GB doesn't support 100× 128K sequences. But batch 16 × 128K context:
`16 × 26,624 × 131072 = 55.8 GB`. Fits in one DGX Sparks.

The linked pair (castor + pollux, 256GB total) can split very large batches
at long context across both nodes.

### 4× Jetson Orin NX (16GB unified each)

For the 500M Q4_K_M model (quantized weights ≈ 350 MB):
```
Model weights (Q4_K_M):       ~350 MB
KV cache at 8192, batch 1:    ~218 MB
Total:                         ~570 MB
Remaining for OS + runtime:   ~15.4 GB
```

The Orin NX has enormous headroom for this model at this context length.
You could run batch 8 at 8192 context: `8 × 218 MB = 1.7 GB total`. Still fine.

The practical constraint on Orin NX is compute throughput, not memory.
At 100 TOPS INT8, generating tokens is slow (~10–20 tok/s). Very long
`<think>` chains (2000 tokens) will take 100–200 seconds to generate.
For interactive use, you'd want to limit generation length or accept the latency.

---

## What "Increasing Context Length" Actually Requires

When you see a model announced with "now supports 128K context!", several
things had to happen:

1. **Positional encoding was updated or extended.** Either the base frequency
   of RoPE was changed, YaRN/LongRoPE was applied, or the model was re-trained
   with long-context data.

2. **The model was fine-tuned on long-context data.** Having the architecture
   support long contexts mathematically is not enough — the model needs to have
   seen and learned from long documents during training, or it won't know how
   to use the extended attention window productively.

3. **Serving infrastructure was updated.** The KV cache management, memory
   allocation, and batching strategy all change at 128K context. A batch of 8
   sequences at 128K needs ~55 GB just for KV cache (for the 3B model). Most
   standard serving infrastructure wasn't designed for this.

For our model, the roadmap is:
- **Now:** 8192 context (2× SFT training length, 4× pre-training, within RoPE
  extrapolation range)
- **Phase 2:** If GRPO training benefits from longer reasoning chains, we may
  fine-tune SFT at 8192 and extend inference to 16K
- **Future versions:** Long-context pre-training data (from arXiv papers,
  books, long code files) + YaRN extension to 32K+

---

## Summary

Context length is determined by four interacting constraints:

| Constraint | Training impact | Inference impact |
|---|---|---|
| KV cache memory | Small (KV cache not stored in training) | Scales linearly with context × batch |
| Activation memory | Quadratic in context length (use grad checkpointing) | Not applicable |
| Attention compute | Quadratic — dominant at long context | Quadratic — slower per token |
| RoPE extrapolation | What lengths you train on = what you can use | Quality degrades beyond ~4× training length |

For our 500M model:
- Pre-trained at 2048 → SFT at 4096 → inference target 8192
- GQA (2 KV heads) reduces KV cache 5× vs. MHA
- RoPE base=500,000 gives useful extrapolation to ~2–4× training length
- Memory is not the constraint on any of our hardware at this model size
- Compute (O(T²) attention) becomes expensive above 16K context
- RoPE extrapolation quality is the practical limit at ~8–16K

---

*This post was added as Part 10 after the main series because context length
questions kept coming up. The hardware section references the fleet from
[Part 3](03-hardware.md).*
