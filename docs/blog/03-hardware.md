# Hardware-First Design: Trainium2, Tile Alignment, and Quantization by Construction

*Part 3 of a series on building a small reasoning language model end-to-end.*

---

Most ML model designs are hardware-agnostic: choose an architecture, then figure
out how to run it. We inverted this. The hardware constraints came first, and
the model dimensions were derived from them.

This sounds backwards until you understand what "tile alignment" means and how
matrix dimensions interact with inference efficiency. Get them wrong, and you
leave 30–40% of your hardware on the table.

---

## The Training Hardware

We're using two platforms:

**Validation (Phase 0, 500M):** RTX 5090 (Blackwell architecture, 32GB GDDR7,
3352 TFLOPS BF16 sparsity, 1676 TFLOPS BF16 dense). This is ceres.local — a
workstation with 128GB RAM and 1.5TB NVMe. Home hardware.

**Production runs (1B, 3B):** AWS Trainium2 (`trn2.48xlarge`, 16 Trainium2
chips, 512GB HBM2e, ~16 PFLOPS BF16 peak). Each Trainium2 chip contains
2 NeuronCores v3. NeuronCore v3 has a 128×128 BF16 systolic array and a
256×128 FP8 systolic array.

The RTX 5090 is flexible — it handles non-aligned dimensions with padding.
Trainium2 is not flexible. Its NeuronCore hardware maps matrix operations
directly to fixed-size systolic arrays. Misaligned dimensions either fail
to compile or silently degrade to inefficient fallbacks.

---

## What is a Systolic Array?

A systolic array is a grid of processing elements where data flows in fixed
patterns — inputs "flow" across rows, weights "flow" down columns, and outputs
accumulate at each cell. It's purpose-built for matrix multiplication.

A 128×128 BF16 systolic array computes one 128×128 matrix tile per cycle.
For a matrix multiply `A(M, K) × B(K, N)`, the systolic array processes it
as a tiled sum over `(M/128, K/128, N/128)` tiles.

If `M = 1280` and `K = 1280`: `1280/128 = 10` tiles in each dimension. Clean.
If `M = 1281` and `K = 1280`: `ceil(1281/128) = 11` tiles, but the last tile
is padded with zeros. The systolic array is doing 10/11 = 91% useful work.

At `M = 1280 + 1 = 1281`, you lose 9% of hardware efficiency. At `M = 1280 + 64`,
you lose 50% of the last tile. Across 26 layers × 4 attention projections × 2
FFN projections, these losses compound.

> **Sidebar: Why BF16 and Not FP16?**
>
> Both BF16 and FP16 use 16 bits per number, but they allocate those bits
> differently:
>
> - FP16: 1 sign bit, 5 exponent bits, 10 mantissa bits. Range: ±65504.
> - BF16: 1 sign bit, 8 exponent bits, 7 mantissa bits. Range: ±3.4×10³⁸.
>
> BF16 has the same exponent range as FP32 but much lower precision. FP16 has
> higher precision but can overflow on large gradient values (hence the need
> for loss scaling with FP16 mixed precision).
>
> For training: BF16 doesn't need a loss scaler (gradients don't overflow).
> For inference: BF16 is more numerically stable under quantization (larger
> dynamic range means less clipping).
>
> Trainium2 and Blackwell both support BF16 natively on their matrix engines.
> We use BF16 for all matrix operations (forward + backward pass) and FP32
> for optimizer states (master weights).

---

## The Alignment Constraint

Every dimension in our model that participates in a matrix multiplication must
be a multiple of 128.

This includes:
- `d_model` (the hidden dimension)
- `ffn_intermediate` (the FFN middle dimension)
- `vocab_size` (the embedding/LM head dimension)
- `n_heads * head_dim` (must equal `d_model`)
- `n_kv_heads * head_dim` (must also be a multiple of 128)

`head_dim` is fixed at exactly 128 across all configs. This is not flexible:
Trainium2's NKI (Neuron Kernel Interface) attention kernel tiles on `head_dim`;
the systolic array dimension is 128; `head_dim = 128` means each head's QK^T
computation maps to exactly one systolic array tile. Any other head_dim wastes
hardware or requires multiple tiles with incomplete utilization.

With `head_dim = 128`:
- `n_heads` can be any integer (each head is independently tile-sized)
- `d_model = n_heads * head_dim` is automatically a multiple of 128

**FFN intermediate dimension:** Standard FFN uses `4 × d_model`. For
`d_model = 1280`: `4 × 1280 = 5120`. SwiGLU needs `(2/3) × 4 × d_model`
to maintain parameter parity with a 2-matrix FFN: `(2/3) × 5120 = 3413.3`.
Round to the next multiple of 128: `3456`. This is `3456 / 128 = 27` tiles. Clean.

**Vocabulary size:** `32768 = 256 × 128`. Tile-aligned. Also a power of 2,
which helps with memory layout in GGUF quantization.

The model code asserts these constraints at initialization:

```python
def __post_init__(self):
    assert self.d_model % 128 == 0, f"d_model must be ≡ 0 (mod 128), got {self.d_model}"
    assert self.ffn_intermediate % 128 == 0, ...
    assert self.vocab_size % 128 == 0, ...
    assert self.head_dim == 128, f"head_dim must be exactly 128 for NKI kernel"
    assert self.n_heads % self.n_kv_heads == 0, f"GQA ratio must be integer"
```

These are not warnings. They're hard assertions that fail at construction time.
If someone changes a dimension, they'll find out immediately, not after a
multi-hour training run.

---

## Trainium2 Specifics

### Model Parallelism

For the 1B model on `trn2.48xlarge` (16 chips, 32 NeuronCores):

**Tensor parallelism (TP) degree: 8.** The attention heads and FFN intermediate
dimensions are split across 8 NeuronCores. Q has 16 heads → 2 heads per core.
FFN intermediate is 5504 → 688 per core. This requires that `n_heads / TP = 2`
and `ffn_intermediate / TP = 688` be integers. Both are.

**Data parallelism (DP): 4.** 4 replicas of the model, each training on a
different data shard. TP × DP = 8 × 4 = 32 = total NeuronCores.

This topology is possible because all our dimensions are multiples of 8 (and
of 128, which implies multiples of 8). If `n_heads = 15` (not divisible by 8),
you can't split heads evenly across 8 cores.

### Static Compilation

Trainium2 requires static graph compilation: batch size, sequence length, and
all tensor shapes must be fixed at compile time. This is very different from
PyTorch's eager execution on CUDA.

`torch_neuronx.trace()` compiles the model into a fixed graph that runs at
hardware speed. First compilation takes 15–30 minutes. Subsequent runs use
the cached compiled graph and are fast.

Implications:
- We compile separate graphs for training (seq_len=4096) and inference (seq_len=2048)
- The training script must not have any dynamic tensor shapes
- Variable-length sequences require padding to the fixed seq_len

### The NKI Attention Kernel

The NeuronSDK provides a generic attention implementation, but it doesn't tile
optimally for `head_dim=128` specifically. We'll write a custom NKI (Neuron
Kernel Interface) kernel that maps the `Q@K^T` and `A@V` operations directly
to the 128×128 systolic array tiles.

The key insight: with `head_dim = 128` and the systolic array tile size also 128,
one attention head's entire Q and K matrices for a given sequence position fit
in one systolic array cycle. No tile-boundary padding, no wasted cycles.

We stub this in `model/nki_attention.py` and will implement it when running
the 1B/3B Trainium2 experiments.

---

## Inference: Quantization by Construction

All of our architectural choices above also affect inference efficiency. Here's
why alignment matters for quantization:

### GGUF Block Quantization

GGUF (the format used by llama.cpp) quantizes weights in 32-element blocks.
Each block stores a scale factor and `n` quantized values (2–8 bits each).
For this to work cleanly, weight matrix dimensions must be divisible by 32.

Our dimensions are multiples of 128, which is 4×32. Every weight matrix
divides into an integer number of 32-element blocks with no remainder.

With a hypothetical `d_model = 1281`:
- A weight matrix row has 1281 elements
- 1281 / 32 = 40.03 blocks → 40 full blocks + 1 partial block with 1 element
- That partial block requires special handling, slightly degrading quantization
  quality for that row and adding implementation complexity

Multiply this across all 26 layers × (Q + K + V + O projections + 2 FFN weights)
= 208 weight matrices. It's a lot of partial blocks to handle.

### Weight Tying and Quantization

The tied embedding / LM head shares one weight matrix `(vocab_size × d_model) =
(32768 × 1280)`. This matrix must be quantized once and used for both token
lookup and next-token prediction.

The constraint: quantization parameters (scale, zero point) are computed across
all 32768 × 1280 elements. A tile-aligned matrix computes these statistics
cleanly with SIMD-friendly memory access patterns.

---

## The Deployment Targets

We design for this inference stack:

| Deployment | Hardware | Quantization | Size (1B) | Expected throughput |
|---|---|---|---|---|
| Development | RTX 5090 (32GB) | BF16 | ~2GB | ~15K tok/s batch |
| Cloud eval | Same | BF16 | ~2GB | — |
| Cloud prod | Graviton4 `c8g.8xlarge` | Q4_K_M | ~700MB | 60–80 tok/s |
| Edge | Raspberry Pi 5 | Q4_0 | ~550MB | ~5–10 tok/s |
| Extreme edge | Anything | Q2_K | ~400MB | model quality degrades |

Q4_K_M on Graviton4 at `~$0.68/hr` gives roughly 70 tok/s. At 1000 tokens per
request, that's ~70 requests/hour, or ~$0.01 per request at batch size 1. For
inference-only workloads (no training), Q4_K_M is the practical default.

---

## Summary: Why Dimensions Are a Design Decision, Not a Detail

The model dimensions (1280, 3456, 32768) look arbitrary. They're not. Each is
the smallest multiple of 128 that satisfies the target parameter count while
fitting in the Trainium2 tile geometry, dividing cleanly by all TP degrees,
and quantizing without GGUF remainder blocks.

Getting these right means:
- ~100% systolic array utilization on Trainium2 (vs. 60–80% with random dims)
- Correct parallelism topology without head-splitting hacks
- Clean GGUF quantization without special-case partial blocks
- Inference that works on everything from a data center GPU to a Raspberry Pi

Getting them wrong means silently slow training and degraded quantization that
you might not notice until you try to deploy.

---

*Next: [Part 4 — The Tokenizer: BPE, Digit Splitting, and Teaching a Model to
Think](04-tokenizer.md)*
