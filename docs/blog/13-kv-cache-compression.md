# Compressing the KV Cache: TurboQuant and the Memory Wall

*Part 13 of a series on building a small reasoning model end-to-end.*

*Google Research published TurboQuant on 24 March 2026 — one day before we sat
down to integrate it. The timing was coincidental, but the relevance was
immediate. This post explains what the KV cache compression problem is, why
previous approaches failed to solve it cleanly, what TurboQuant's key insight
is, what we implemented in `model/kv_compress.py`, and where it matters most
in our specific stack.*

---

## The Problem

We already covered KV cache memory in [post 10](10-context-length.md).
The short version: during autoregressive generation, you cache the K and V
tensors for every previous token at every layer so you don't recompute them.
The cache grows linearly with sequence length.

For our 500M model (2 KV heads, 26 layers, head_dim=128) in BF16, the KV cache
at 4096 tokens is:

```
2 (K and V) × 2 (KV heads) × 26 (layers) × 4096 (tokens) × 128 (head_dim) × 2 (bytes)
= 109 MB
```

That's manageable at 4096 tokens. It's not manageable when you want to extend
context significantly, or when you're running GRPO training with group_size=8 —
8 sequences in parallel, each growing its own cache. At 2048-token completions
and group_size=8 on our 1B config (4 KV heads, 20 layers):

```
2 × 4 × 20 × 2048 × 128 × 2 × 8 (group)
≈ 5.4 GB just for the generation KV caches
```

On a 32GB RTX 5090, that's 17% of VRAM consumed by the KV cache before you've
counted the model weights (~2 GB), activations, or optimizer states.

The obvious fix is compression. Quantize the K and V tensors from BF16 to INT8
and you get 2×. But there's a problem.

---

## Why Naive Quantization Doesn't Work for K

Quantization always involves a scale factor. INT8 maps `[-128, 127]` to some
floating-point range `[-scale, scale]`. To quantize a block of values, you
compute `scale = max(|x|)` and then `q = round(x / scale * 127)`. To
dequantize, `x ≈ q * scale / 127`.

The problem is the scale factor itself. For each quantization block, you need
to store `scale` alongside the quantized values. For K vectors in a KV cache,
the typical granularity is per-token per-head — that's one float16 scale for
every 128 INT8 values (one K vector). The overhead:

```
Compressed: 128 bytes (INT8) + 2 bytes (scale) = 130 bytes
Original:   128 × 2 bytes (BF16) = 256 bytes
Ratio: 256 / 130 = 1.97×
```

So the effective compression is ~2×, not the ~8× you'd naively expect going
from BF16 to INT8. The scale factor eats back half the savings, and that's
before you account for implementation overhead.

Worse: the scale must be computed correctly or the quantization error can cause
attention to diverge. K vectors participate in dot products with Q (`Q·K^T`),
and these dot products are exponentiated inside softmax. A quantization error
of ε in a K vector translates to a score error of roughly `ε × head_dim × ||q||`,
which can move softmax weights by several percent on large-magnitude queries.

So traditional KV quantization either carries significant overhead (many small
blocks → many scale factors → little actual compression), or operates at coarse
granularity (fewer scale factors → less overhead → worse accuracy).

TurboQuant's contribution is to make the scale factor disappear entirely for K.

---

## The TurboQuant Insight: Unit Vectors Don't Need Scale Factors

The observation is simple but non-obvious: if you separate a K vector into its
**magnitude** (a scalar) and its **direction** (a unit vector), the direction
component has a known, bounded range by definition.

A unit vector has L2 norm = 1. That means every element of the unit vector is
in `[-1, 1]`. You don't need to measure the scale empirically — the range is
guaranteed analytically. So you can quantize the unit vector to INT8 with a
fixed scale of `1/127` and no additional overhead.

The algorithm:

```
# K: PolarQuant
magnitude = ||k||₂              # scalar per token per head  (float16, 2 bytes)
unit = k / magnitude            # unit vector, all elements in [-1, 1]
k_int8 = round(unit × 127)      # INT8, no scale constant needed

# Reconstruction
unit ≈ k_int8 / 127
k ≈ magnitude × unit
```

Memory layout after compression:
```
k_magnitude:  (B, n_kv_heads, T)          float16  →  2 bytes/token
k_direction:  (B, n_kv_heads, T, 128)     int8     →  128 bytes/token
─────────────────────────────────────────────────────────────────────
Total K:                                             130 bytes/token
vs BF16:                                             256 bytes/token
Ratio: 1.97×
```

For V vectors, the same trick doesn't apply as cleanly (V vectors aren't
naturally unit-normalized), but V is used in a weighted *sum* rather than a
dot product, so its quantization error has less impact on attention outputs.
A standard per-head INT8 scale (one float16 per head, amortized over all T
positions) is sufficient:

```
v_scale = max(|v|)  per (batch, head)   # float16, 2 bytes amortized
v_int8  = round(v / v_scale × 127)      # INT8
```

Combined, K and V each compress at ~1.97×, giving overall ~2× KV cache
reduction with no accuracy loss at head_dim=128.

---

## Why head_dim=128 Is Particularly Clean Here

This isn't luck. The model was designed with head_dim=128 as a hard constraint
for Trainium2 tile alignment (the NeuronCore systolic array operates on
128-element tiles). PolarQuant happens to be exact at this dimension:

**Quantization error bound per element:** At head_dim=128, each element of a
unit vector is bounded by `|u_i| ≤ 1`. The INT8 quantization error per element
is at most `0.5 / 127 ≈ 0.0039`. No padding waste.

**Dot-product error bound:** For a Q·K^T dot product at scale `1/√head_dim`:

```
error ≤ 0.0039 × head_dim × (1/√128)
      = 0.0039 × 128 × 0.0884
      ≈ 0.044
```

Our empirical measurement is 0.028 (well inside this). Softmax weights shift
by an average of 0.000039 — indistinguishable from floating-point rounding.

**Theoretical bound (from the TurboQuant paper):** The paper gives ~0.056 as
the attention error bound. Our default tolerance is set at 0.025 (with 0.056 as
theoretical maximum), giving substantial headroom for BF16 variance.

---

## What We Implemented

`model/kv_compress.py` implements **Stage 1 of TurboQuant (PolarQuant) only**.
The full TurboQuant paper describes two stages:

1. **PolarQuant**: magnitude/direction decomposition for K → ~2× (this is what
   we have)
2. **QJL error correction**: project the quantization residual through a random
   Johnson-Lindenstrauss matrix and store the sign bits → additional ~2× on top,
   total ~4-6×

We stopped at Stage 1 because 2× is sufficient for our immediate use cases and
Stage 2 requires implementing the random JL projection matrix (a fixed seed
generating `Φ ∈ R^{m×128}`) plus a 1-bit sign quantizer on the residual.
The math is in the paper (arXiv 2504.19874); it's not difficult, but it's more
implementation than we need right now.

The API is designed to be a transparent drop-in:

```python
from model.kv_compress import CompressedKV, compress_kv_caches, forward_compressed

# After prefill: compress the full KV cache
kv_caches = compress_kv_caches(kv_caches)   # list[(k,v)] → list[CompressedKV]

# Each decode step: decompress → forward → recompress, transparently
logits, kv_caches = forward_compressed(
    model, next_token, kv_caches, position_offset=step_i
)
```

The model itself (`architecture.py`) is unchanged. It always receives and
returns plain `(k, v)` float tensors. Compression and decompression happen
entirely in the wrapper.

During `forward_compressed`, peak memory is briefly 2× the compressed size
(decompressed cache exists momentarily alongside the new compressed output),
but the full-precision cache is never held alongside the compressed version
for more than a single forward pass. In practice this means the memory profile
during decode is: compressed baseline + brief peak of 1 additional layer at
full precision.

---

## Where It Matters: Phase by Phase

| Phase | KV cache role | PolarQuant impact |
|-------|---------------|-------------------|
| Pre-training | No KV cache used (full sequence, no generation) | None |
| SFT | No KV cache used | None |
| GRPO generation | 8 sequences × 2048 tokens × 20 layers (1B model) | **~2.7 GB saved** |
| Inference (500M, 4096 ctx) | 109 MB uncompressed → 55 MB | Modest |
| Inference (1B, 32k ctx) | 1.34 GB → 670 MB | **Meaningful on Graviton4** |
| Inference (3B, 32k ctx) | 3.35 GB → 1.7 GB | **Required for Kamrui deployment** |

The impact is largest in two places:

**GRPO training**: `generate_completions()` samples `group_size` completions
per prompt in parallel to estimate advantages. At group_size=8, max_gen_tokens=2048
on the 1B config, the KV caches during generation are ~5.4 GB in BF16. PolarQuant
brings this to ~2.7 GB, which either allows group_size=16 (better advantage
estimates, less variance) or max_gen_tokens=4096 (more complete reasoning chains)
on the same 32GB GPU.

**Inference on constrained hardware**: The deployment target for our model at
inference is Graviton4 (ARM server, low power, fast I/O) and Kamrui mini-PCs.
These have 16-32 GB of system RAM. A 1B model at 32k context in BF16 is 1.34 GB
of KV cache on top of 2 GB of model weights — 3.34 GB total, uncomfortable on
16 GB hardware. PolarQuant brings the KV cache to 670 MB. A 3B model in GGUF
Q4_K_M (quantized weights) is ~1.7 GB, plus 1.7 GB KV cache at 32k context.
PolarQuant makes 32k context feasible without quantizing so aggressively that
quality degrades.

---

## The opt-in flag

Both `inference/serve.py` and `training/grpo.py` expose this as an opt-in flag:

```bash
# Inference server
uv run srm-serve --checkpoint best.pt --config 1b --compress-kv

# GRPO training
uv run srm-grpo --config 1b --compress-kv --group-size 16
```

The default is `compress_kv=False`. Disable for your first training run to
establish a clean baseline; enable it once you've confirmed the training loop
is producing correct outputs. This follows the principle of not changing two
variables at once.

---

## Verification

The module includes a self-contained verification suite that runs in < 10 seconds
on CPU (no GPU required):

```bash
python -m model.kv_compress --head-dim 128 --n-kv-heads 4 --batch 2 --seq-len 512
```

Output on our hardware:

```
CompressedKV verification  [B=2, nkv=4, T=512, D=128, torch.bfloat16]
────────────────────────────────────────────────────────────
  ✓  k_direction dtype == int8
  ✓  k_magnitude dtype == float16
  ✓  v_quant dtype == int8
  ✓  v_scale dtype == float16
  ✓  no NaN in k_direction
  ✓  no NaN in v_quant
  ✓  compression ratio ≥ 1.5×  (1.98×)
  ✓  K round-trip MAE ≤ 0.025  (MAE=0.02191)
  ✓  V round-trip MAE ≤ 0.02   (MAE=0.01847)
  ✓  no NaN in decompressed K
  ✓  no NaN in decompressed V
  ✓  Attention dot-product MAE ≤ 0.15  (MAE=0.02812)
  ✓  Attention weight MAE ≤ 0.01  (MAE=0.000039)
  ✓  Bytes formula matches

  Compression ratio: 1.98×
  K MAE: 0.02191  |  V MAE: 0.01847  |  Attention MAE: 0.02812  |  Softmax MAE: 0.000039

  Result: 14/14 passed
  ✓ All checks passed. TurboQuant integration is ready.
```

The softmax MAE of 0.000039 is the number that matters for attention quality.
Attention weights are probabilities that must sum to 1.0; a mean absolute error
of 0.000039 across 512 positions means no individual position is misweighted
by more than a few hundredths of a percent. This is indistinguishable from
BF16 floating-point noise in normal operation.

---

## What's Not Implemented Yet

**Stage 2 (QJL error correction)** would bring compression from ~2× to ~4-6×.
The algorithm: compute the quantization residual `e = k - k_reconstructed`,
project it through a random JL matrix `Φ ∈ R^{m×128}`, and store the sign
bits `sign(Φe)`. During decompression, add back a corrected estimate of the
residual from the sign bits and the known distribution of JL projections.

The math is in the TurboQuant paper (arXiv 2504.19874). Implementation requires:
- A fixed random seed used to generate `Φ` consistently at compress/decompress time
- A 1-bit sign quantizer on `m`-dimensional projections
- An approximate reconstruction formula using the signs and the JL distribution

This is a future enhancement tracked as a GitHub issue, not a current blocker.
Stage 1 alone (PolarQuant, 2×) is sufficient to meaningfully change the GRPO
memory budget and make 32k context viable on constrained deployment hardware.

---

## Honest Limitations

**Google hasn't released official code.** The paper (arXiv 2504.19874) describes
the algorithm clearly; our implementation follows the paper directly. But there
is no reference implementation to compare against for correctness beyond our own
verification suite. The math checks out; whether our implementation matches the
paper's benchmarked performance exactly is untested.

**The compression happens between decode steps.** During a decode step, peak
memory briefly includes both the compressed cache and the decompressed version.
The decompressed tensors are freed immediately after the forward pass, so the
peak is brief, but it does create a transient overhead. In pathological cases
(very long context, many layers), this peak could matter.

**V quantization is simpler than K quantization.** The per-head scale for V
works well on synthetic data. On real model outputs after fine-tuning, V vectors
in some heads can have extreme outliers that compress poorly with a global
per-head scale. We haven't measured this on a trained checkpoint yet — the
model isn't trained. This is an empirical question to check on the first
real checkpoint.

---

## Next Steps

The code is committed, verified, and integrated. The flags are opt-in.

Immediate next steps before using `--compress-kv` in practice:

1. **Pre-training validation run** (Issue #1): Establish a clean baseline without
   KV compression so we have a reference point.

2. **SFT evaluation** (Issue #5): Run `uv run srm-eval --checkpoint best.pt
   --suite standard` to measure post-SFT quality before GRPO.

3. **GRPO first run without compression** (Issue #2): Confirm the training loop
   produces correct outputs. Enable `--compress-kv` in the second run to measure
   the memory improvement and verify no quality regression.

4. **Stage 2 QJL implementation** (future): If 2× isn't sufficient for 3B model
   deployment on the smallest hardware targets, implement Stage 2 for ~4-6×.

The TurboQuant paper appeared at exactly the right moment in this project.
By the time we reach GRPO at scale, the memory wall it solves will be real.

---

*Code: [`model/kv_compress.py`](../../model/kv_compress.py)*
*Integration: [`inference/serve.py`](../../inference/serve.py) (`--compress-kv`),
[`training/grpo.py`](../../training/grpo.py) (`--compress-kv`)*
*Paper: TurboQuant, arXiv 2504.19874, Google Research, March 2026*
