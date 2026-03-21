# Hardware Ecosystem: Training, Evaluation, and Edge Deployment

*Part 3 of a series on building a small reasoning language model end-to-end.*

---

This project runs on hardware we own. The training pipeline, evaluation suite,
and inference deployment all run locally — no cloud required as the primary path.
That's both a constraint and an advantage: it forces us to understand exactly
what each piece of hardware is good for and how model design decisions affect
performance across a heterogeneous fleet.

This post covers the hardware ecosystem, what each machine contributes, and why
model dimension decisions made during architecture design show up as inference
performance differences on hardware you might not expect — like a Jetson Orin NX
or a GGUF quantized model on a Raspberry Pi.

---

## The Hardware Fleet

### ceres — Primary Training (RTX 5090)

Everything in this series runs on ceres first.

| | |
|---|---|
| GPU | NVIDIA RTX 5090 (Blackwell GB202) |
| VRAM | 32 GB GDDR7 |
| Compute | 1676 TFLOPS BF16 dense / 3352 TFLOPS BF16 sparsity |
| Memory bandwidth | 1792 GB/s |
| System RAM | 128 GB DDR5 |
| Storage | 1.5 TB NVMe |

The 1.8 TB/s memory bandwidth is the key differentiator vs older Ampere cards.
Memory-bandwidth-bound operations (attention at long sequences, activation
movement in training) are meaningfully faster than on a 3090 or 4090. With
gradient checkpointing, the 500M model at 4096 token sequences comfortably fits
in 32GB with room for optimizer states.

Blackwell also introduces FP4 tensor cores. We're not using FP4 for training
but it opens future quantization territory at inference.

### vesta — Development and Mid-Range Inference (RTX 4070 Ti SUPER)

| | |
|---|---|
| GPU | NVIDIA RTX 4070 Ti SUPER (Ada Lovelace) |
| VRAM | 16 GB GDDR6X |
| Memory bandwidth | ~672 GB/s |

A 16GB VRAM card is more representative of commonly-available hardware than the
5090. We use vesta for:
- Quick iteration on architecture changes without burning 5090 time
- Benchmarking inference on mid-range hardware (relevant to more people)
- Running the 500M model in BF16 for inference testing
- Verifying that model design decisions don't accidentally require >16GB

If your inference target is a workstation or gaming PC rather than a data center,
vesta is your reference point.

### janus — Multi-GPU Training (2× Titan RTX + NVLink)

| | |
|---|---|
| GPUs | 2× NVIDIA Titan RTX (Turing) |
| VRAM each | 24 GB GDDR6 |
| Effective VRAM | 48 GB (with NVLink) |
| NVLink bandwidth | 25.78 GB/s per link, 2 links → ~51.5 GB/s bidirectional |

Two Titan RTX cards connected via NVLink bridge give 48GB of effective VRAM with
~51 GB/s of inter-GPU bandwidth. This is qualitatively different from multi-GPU
without NVLink (where tensor-parallel all-reduce goes through PCIe at ~16 GB/s).

**What this enables:**
- 1B model at full BF16 without gradient checkpointing — the full model and
  optimizer states fit in 48GB
- Tensor parallelism across 2 GPUs with low communication overhead
- Data-parallel training where gradient synchronization is fast

We'll use janus to explore multi-GPU training and to run the 1B model experiment
locally before committing to a cloud run.

> **Sidebar: NVLink vs PCIe for Multi-GPU Training**
>
> Multi-GPU training requires all-reduce communication: each GPU computes
> gradients on its data shard, then all GPUs sum those gradients and distribute
> the result. The speed of this operation determines how much multi-GPU
> parallelism hurts throughput.
>
> With PCIe (the default for consumer multi-GPU setups): ~16 GB/s per direction.
> For a 1B parameter model (2GB gradients in BF16), an all-reduce takes roughly
> `2GB / 16 GB/s = 125ms` at the limit. At a training step of ~200ms, that's
> 62% of step time spent communicating.
>
> With NVLink at 25.78 GB/s per link (2 links on Titan RTX): bidirectional
> bandwidth up to ~50 GB/s. Same 2GB gradient reduce: `~40ms`. Communication
> overhead drops from 62% to ~20%.
>
> This is why NVLink (or InfiniBand for multi-node) is table stakes for serious
> multi-GPU training. PCIe multi-GPU is mostly useful for inference where you're
> not doing all-reduces.

### castor and pollux — DGX Sparks Pair (128GB Unified × 2)

| | |
|---|---|
| SoC | NVIDIA GB10 Grace Blackwell Superchip |
| CPU cores | 20× Arm Neoverse V2 (Grace) |
| Memory | 128 GB LPDDR5X unified (CPU + GPU share same pool) |
| Compute | ~1 PFLOP FP4 / ~500 TFLOPS FP16 |
| Memory bandwidth | ~273 GB/s |
| Inter-node | 2 dedicated high-speed links, MTU 9000, sub-ms latency |

Castor and pollux are **two linked DGX Sparks** — a 256GB unified-memory
two-node cluster sitting under a desk. The two direct inter-node links (on
separate 192.168.1.x and 192.168.2.x subnets, both with jumbo frame support)
enable distributed inference and potentially multi-node training without going
through a shared network fabric.

The **unified memory** architecture is the key property. On the 5090, there's
a hard 32GB VRAM boundary — anything larger spills to system RAM over PCIe.
On the DGX Sparks, 128GB is all the same: the Blackwell GPU and Grace CPU
address the same physical memory. A tensor never needs to "move" between CPU
and GPU memory spaces.

**What this cluster enables:**
- Run the 3B model at full BF16 (~6GB) trivially on either node
- Split the 3B model across both nodes (256GB total) for long-context experiments
- Large-batch evaluation: load the full eval set into memory across both nodes
- Speculative decoding: draft model on one node, target on another

> **Sidebar: Why Unified Memory Changes the Inference Calculus**
>
> Typical GPU inference is bottlenecked by weight loading: the GPU must read
> all model weights from VRAM for each forward pass. More weights than fit in
> VRAM means PCIe spills (~64 GB/s), catastrophically limiting throughput.
>
> With unified memory, this boundary doesn't exist. The 273 GB/s bandwidth
> of the DGX Sparks's LPDDR5X is available to both CPU and GPU uniformly.
> There's no "spill" — you just use more of the 128GB.
>
> The trade-off: 273 GB/s vs. the RTX 5090's 1792 GB/s. For batch size 1,
> the 5090 is ~6× faster for models that fit in 32GB. The DGX Sparks wins
> when you need to run models that don't fit in 32GB, or when CPU-GPU
> data movement would otherwise be the bottleneck.

### 4× Jetson Orin NX — Edge Deployment (Offline)

| | |
|---|---|
| GPU | 1024 CUDA cores (Ampere) |
| CPU | 8× Arm Cortex-A78AE |
| NPU | 2× NVDLA v3.0 (Deep Learning Accelerator) |
| Memory | 16 GB LPDDR5 unified |
| Peak compute | ~100 TOPS INT8 (GPU + NVDLA combined) |
| Power | 10–25 W |

The Jetson Orin NX is the edge deployment target: a module that runs at 10–25W
with dedicated INT8 inference acceleration via NVDLA.

The **NVDLA** (NVIDIA Deep Learning Accelerator) is the NPU — a fixed-function
accelerator for INT8 neural network operations. For transformer inference:
the NVDLA handles the matrix-multiply-heavy attention and FFN projections at
low power; the Ampere GPU handles the rest. This can meaningfully reduce power
consumption vs. GPU-only inference while maintaining throughput.

Four units offline right now. We'll bring them up for the inference phase to:
1. Benchmark llama.cpp (CUDA backend, no NVDLA) vs. TensorRT-LLM (NVDLA offload)
2. Understand the power/throughput trade-off for sustained edge inference
3. Test 4× parallel inference as a simple load-balanced serving cluster

> **Sidebar: What is an NPU, and How Does NVDLA Differ from a GPU?**
>
> A GPU is a general-purpose parallel processor. It can run any operation but
> carries overhead from its general-purpose pipeline, cache hierarchy, and
> scheduler.
>
> An NPU (Neural Processing Unit) hardwires the operations in neural networks:
> INT8 matrix multiply, convolution, pooling, activation functions. By removing
> the general-purpose overhead, it achieves higher throughput-per-watt than a
> GPU for the specific operations it supports — but nothing else.
>
> NVDLA v3.0 supports INT8 convolution and matrix multiply (the transformer's
> core operations), common activations, and normalization. It does not support
> FP16/BF16, dynamic shapes (the graph must be compiled ahead of time via
> TensorRT), or arbitrary operations. It's fast and efficient exactly when
> your workload matches its fixed capabilities.
>
> The 4× Orin NX cluster at 4 × 100 TOPS INT8 = 400 TOPS combined is a
> surprisingly capable edge inference system for 500M Q4 models.

---

## Cloud Excursions

For questions specifically about training at scale or on specialized hardware,
we'll run excursions to cloud platforms. These are experiments, not the primary
path. If you're following this series primarily for what's achievable on owned
hardware, you can skip these sections without missing the core story.

**AWS Trainium2** — NVIDIA-alternative ML training chips with 128×128 BF16
systolic arrays and NeuronCore v3 architecture. Interesting for the questions:
"How does model dimension alignment affect utilization on non-NVIDIA hardware?"
and "What does the training cost look like for the 1B/3B configs on specialized
silicon?" A `trn2.48xlarge` (16 Trainium2 chips) can train the 1B model at
~50B tokens for roughly $700–1000. We'll do this run to demonstrate the 1B
model and to write the NKI attention kernel.

**H100 / A100 (if needed)** — For specific comparison benchmarks or if a
particular experiment needs more than 32GB VRAM. We don't anticipate needing
this for the core training path.

**Google TPU** — Potentially interesting for the scale experiments, but requires
JAX/XLA rather than PyTorch. Out of scope for the current series.

---

## Why Dimension Alignment Spans the Whole Stack

The tile alignment constraint (all dimensions multiples of 128) is often
described as a Trainium2 requirement. It's actually a property that benefits
every piece of hardware in the stack, from training to edge.

### On the RTX 5090 (Tensor Core Tiles)

NVIDIA Tensor Cores process matrix operations in tiles. For Blackwell:
- BF16: 16×16 matrix multiply is the base tile
- The MMA (Matrix Multiply-Accumulate) instruction operates on 16×16 input tiles

A weight matrix with `d_model = 1280` has rows of 1280 elements = 80 tiles of 16.
A matrix with `d_model = 1281`: `ceil(1281/16) = 81` tiles, last one padded.
At 1281, you do 81 MMA operations but get 1/81 of the last one wasted.

The effect is smaller than on Trainium2 (NVIDIA's compiler is better at hiding
it), but it's still there — and it's free to fix by choosing aligned dimensions.

### On the Orin NX (NVDLA INT8)

NVDLA processes INT8 operations in fixed-width tiles. The tile size for matrix
multiply on NVDLA v3.0 is hardware-specific but follows similar power-of-2 or
multiple-of-8 constraints. A misaligned weight matrix causes the compiler to
insert padding and handle boundary tiles in software fallback paths.

With `d_model = 1280`, INT8 quantization produces a weight matrix where every
row divides into a whole number of INT8 tile operations. With `d_model = 1281`,
the boundary handling is non-trivial. This shows up as lower NVDLA utilization
and more CPU fallback time.

### In GGUF Quantization (llama.cpp, everywhere)

GGUF quantizes weights in 32-element blocks. A weight matrix row of 1280
elements = 40 blocks exactly. A row of 1281 elements = 40 full blocks + 1
partial block. Every weight matrix in every layer has this problem.

The 32-element block size comes from SIMD constraints: AVX2 can process 32
INT8 values simultaneously; NEON (used on Graviton4 and Orin NX ARM cores)
processes 16. Aligned dimensions map to whole SIMD operations; misaligned
dimensions require scalar cleanup code for the remainder.

### The Unifying Principle

128 = 4 × 32 = 8 × 16 = 1 Trainium2 tile = 8 GGUF quantization blocks =
8 NVDLA processing chunks. A dimension that's a multiple of 128 is clean for
all of these simultaneously. This is why we treat 128-alignment as a project-
wide constraint rather than a hardware-specific one.

---

## How Each Machine Fits Into the Workflow

```
ceres (RTX 5090) — primary training
  ├── Phase 0 pre-training        ✅ complete (9.86B tokens, 43h)
  ├── Phase 1 SFT                 🔄 in progress (~6-12h)
  ├── Phase 2 GRPO                ⏳ pending (~20-40h)
  └── Development iteration       continuous

vesta (RTX 4070 Ti SUPER) — mid-range reference
  ├── Inference benchmarks         ⏳ pending (after GGUF conversion)
  └── Quick iteration testing      as needed

janus (2× Titan RTX NVLink) — multi-GPU exploration
  ├── 1B model local training      ⏳ planned (2-GPU tensor parallel)
  └── Multi-GPU SFT/GRPO           ⏳ planned

castor + pollux (2× DGX Sparks, linked) — eval + long-context
  ├── Large-batch evaluation       ⏳ pending (after GRPO)
  ├── Long-context inference       ⏳ pending
  ├── 3B model full-BF16 serving   ⏳ pending
  └── Distributed inference exp.   ⏳ pending

4× Jetson Orin NX — edge cluster
  ├── llama.cpp Q4_K_M benchmark   ⏳ pending (bring online)
  ├── TensorRT + NVDLA benchmark   ⏳ pending
  └── 4-node serving cluster       ⏳ pending

Cloud (AWS Trainium2)              [excursion]
  ├── 1B model training            ⏳ planned (~$700-1000)
  └── NKI attention kernel         ⏳ planned
```

---

## A Note on the Tile Alignment Assertions

The model code has hard assertions in `ModelConfig.__post_init__` that fail at
construction time if any dimension violates alignment:

```python
assert self.d_model % 128 == 0
assert self.ffn_intermediate % 128 == 0
assert self.vocab_size % 128 == 0
assert self.head_dim == 128
assert self.n_heads % self.n_kv_heads == 0
```

These are not defensive programming against theoretical misuse. They're
guardrails against a real mistake pattern: tuning parameters to hit a parameter
count target and accidentally choosing a non-aligned dimension. Without these,
a model trained on the 5090 (which tolerates misalignment) would fail silently
on Orin NX or Trainium2, and produce suboptimal GGUF quantization everywhere.

The assertions are cheaper than debugging a silent efficiency regression.

---

*Next: [Part 4 — The Tokenizer: BPE, Digit Splitting, and Teaching a Model to
Think](04-tokenizer.md)*
