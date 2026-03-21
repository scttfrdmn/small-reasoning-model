# Inference at the Edge: GGUF, Quantization, and Running on a Raspberry Pi

*Part 9 of a series on building a small reasoning language model end-to-end.*

---

Training produces a 2.8GB BF16 checkpoint. That's fine for evaluation on the
RTX 5090 but completely impractical for deployment: it requires a 16GB+ GPU,
doesn't run on CPU, and takes up 2.8GB of RAM just to load.

The deployment story for small models is quantization. A 500M model at Q4_K_M
fits in 350MB, runs at ~60 tokens/sec on a Graviton4 CPU, and costs about
$0.68/hour. The Raspberry Pi 5 can run it at 5–8 tokens/sec with 500MB RAM.

This post covers the conversion pipeline, the theory behind quantization, and
the practical realities of edge deployment.

---

## The GGUF Format

GGUF (GGML Unified Format) is the file format used by llama.cpp. It stores:
- Model hyperparameters (dimensions, attention type, etc.)
- All weight tensors, optionally quantized to 1–8 bits per weight
- Tokenizer vocabulary and merge rules
- Metadata (architecture, training details, quantization type)

A GGUF file is self-contained: you can copy it to a new machine and run it with
llama.cpp without any Python, PyTorch, or HuggingFace dependencies.

### The Conversion Process

The conversion from our PyTorch checkpoint to GGUF happens in `inference/convert_gguf.py`:

1. Load the PyTorch checkpoint
2. Map our parameter names to llama.cpp's expected naming convention
3. Apply quantization to each weight tensor
4. Write the GGUF binary format

The naming convention is the fiddly part. llama.cpp expects names like
`blk.0.attn_q.weight`; our model uses `blocks.0.attn.q_proj.weight`. The
conversion script maintains an explicit mapping table.

```python
NAME_MAP = {
    "blocks.{i}.norm1.weight":        "blk.{i}.attn_norm.weight",
    "blocks.{i}.attn.q_proj.weight":  "blk.{i}.attn_q.weight",
    "blocks.{i}.attn.k_proj.weight":  "blk.{i}.attn_k.weight",
    "blocks.{i}.attn.v_proj.weight":  "blk.{i}.attn_v.weight",
    "blocks.{i}.attn.out_proj.weight":"blk.{i}.attn_output.weight",
    "blocks.{i}.norm2.weight":        "blk.{i}.ffn_norm.weight",
    "blocks.{i}.ffn.gate.weight":     "blk.{i}.ffn_gate.weight",
    "blocks.{i}.ffn.up.weight":       "blk.{i}.ffn_up.weight",
    "blocks.{i}.ffn.down.weight":     "blk.{i}.ffn_down.weight",
    "embedding.weight":               "token_embd.weight",
    "norm.weight":                    "output_norm.weight",
    # LM head: tied to embedding in our model, separate in GGUF
    # (GGUF stores both separately even when tied)
}
```

---

## Quantization Theory: What We're Actually Doing

Quantization maps floating-point weights to low-bit integers. For Q4 (4-bit quantization):
- Full precision: each weight is stored as 16 bits (BF16) or 32 bits (FP32)
- Quantized: each weight is stored as 4 bits (0–15)

The mapping is: `w_float ≈ scale * w_int + zero_point`

For a block of 32 weights:
1. Find the min and max values: `w_min, w_max`
2. Compute scale: `scale = (w_max - w_min) / 15`
3. Compute zero point: `zero = round(-w_min / scale)`
4. Quantize each weight: `w_int = clamp(round(w_float / scale) + zero, 0, 15)`

The scale and zero point are stored in FP16 (2 bytes each) per 32-weight block.
So Q4_0 costs: `4 bits × 32 weights + 16 bits scale + 16 bits zero = 128 + 32 = 160 bits = 5 bytes per 32 weights = 5 bits/weight average`.

**Why our 128-aligned dimensions help:**
A weight matrix row with `d_model = 1280` elements:
- `1280 / 32 = 40` blocks, exactly
- No remainder, no partial blocks requiring special handling

If `d_model = 1281`:
- `1281 / 32 = 40.03` → 40 full blocks + 1 block with 1 weight
- That last block needs special casing AND the scale is computed over just 1 weight
  (basically no quantization — it's still FP16, just wrapped in the block format)

> **Sidebar: Q4_K_M vs Q4_0 vs Q8_0**
>
> llama.cpp has many quantization formats. The naming convention:
>
> **Q4_0:** 4-bit weights, simple linear quantization per 32-weight block.
> Fast, small, but crude — all weights in a block share one scale.
>
> **Q4_K_M:** "K-quant" with 4-bit weights and mixed precision. The "K" variants
> use a more sophisticated quantization scheme where the scale factors themselves
> are quantized (super-quantization), and some layers (embeddings, output LM head)
> are kept at higher precision. The "M" means "medium" — a balance between "S"
> (small, more aggressive quantization of scales) and "L" (large, less aggressive).
>
> In practice: Q4_K_M produces noticeably better quality than Q4_0 at the same
> file size, particularly for layers with high activation variance. For most
> applications, Q4_K_M is the default recommendation.
>
> **Q8_0:** 8-bit weights. About 2× larger than Q4, but near-lossless quality.
> Use when you have the memory and want to minimize quantization error.

---

## Quality Loss from Quantization

Quantization introduces approximation error. How much quality is lost?

For transformer LLMs at 500M–1B parameters, empirical results from the llama.cpp
community (and papers like "GPTQ" and "QuIP"):

| Format | Bits/weight | Size (500M) | MMLU degradation vs BF16 |
|---|---|---|---|
| BF16 | 16 | ~2GB | 0% (reference) |
| Q8_0 | 8.5 | ~1GB | < 0.5% |
| Q4_K_M | 4.5 | ~350MB | 1–3% |
| Q4_0 | 4.5 | ~350MB | 3–7% |
| Q2_K | 2.6 | ~200MB | 15–30% |

Q4_K_M loses roughly 2% on most benchmarks. For a 500M model that starts at,
say, 60% GSM8K, that's 60% → 58.8%. Acceptable.

Q2_K is for curiosity only — the quality degradation at this parameter count
is too severe for practical use.

Why does smaller degrade faster? Large models are "over-parameterized" — they
have redundant weights that can absorb quantization error. Small models have
fewer redundant weights; each weight carries more information. The degradation
per bit is higher.

---

## Runtime: llama.cpp on CPU

llama.cpp implements efficient transformer inference in C++ with SIMD
acceleration (AVX2, NEON, SVE). The key optimizations:

**SIMD quantized kernels:** The core `q4_K × f32 = f32` matrix multiply kernel
uses AVX2 or NEON intrinsics to process 32 Q4 weights simultaneously (one SIMD
register). On Graviton4 (SVE), 512-bit SVE registers can process 128 Q4 weights
in one instruction.

**KV cache management:** During autoregressive generation, Q and K vectors for
all previous tokens must be stored. llama.cpp stores the KV cache in a single
contiguous memory region and manages it carefully to avoid fragmentation.

**Row-parallel batching:** For batch size > 1, the Q, K, V projections are
computed for all batch items simultaneously. This amortizes the fixed overhead
of loading weights from memory.

### Graviton4 Performance Estimate

The Graviton4 has:
- 512-bit SVE registers
- 96 cores (on `c8g.24xlarge`)
- ~160 GB/s memory bandwidth

For a 500M Q4_K_M model (~350MB weights):
- One forward pass reads all 350MB of weights once
- At 160 GB/s: `350MB / 160 GB/s = 2.2ms` minimum per token (memory-bound)
- Practical throughput on `c8g.4xlarge` (4 cores): ~25–35 tokens/sec at batch 1
- On `c8g.8xlarge` (8 cores): ~60–80 tokens/sec at batch 1

At 70 tokens/sec on `c8g.8xlarge` ($0.68/hr):
- $0.68/hr ÷ 3600 sec/hr = $0.000189/sec
- At 70 tok/s: $0.000189/70 = $0.0000027/token
- Per 1000 tokens: ~$0.0027

**Sub-cent per 1000 tokens on CPU hardware, with a model you own.**

This is the economic argument for small, well-trained models: inference cost
approaches zero. You don't need to route every request to a $0.01–$0.03/1K-token
API.

---

## Raspberry Pi 5

The Raspberry Pi 5 has:
- Cortex-A76 cores (4×, Armv8.2-A with NEON SIMD)
- 8GB RAM (Pi 5 8GB variant)
- ~17 GB/s memory bandwidth (LPDDR4X)

Expected throughput for 500M Q4_K_M (~350MB):
- Memory-bound: `350MB / 17 GB/s ≈ 20ms/token → 50 tok/s theoretical maximum`
- Practical (overhead): ~5–10 tok/s at batch 1

5–10 tokens/sec is slow for interactive use but completely viable for:
- Batch annotation jobs
- Embedded applications where latency isn't critical
- Proof-of-concept that runs literally anywhere

The 500M parameter size is what makes this possible. A 7B model at Q4_K_M is
~3.5GB — barely fits in 8GB RAM and runs at <1 tok/s on Pi 5.

---

## The Deployment Architecture

Our target production setup:

```
Request → [Inference Server] → [llama.cpp backend] → Response
               |
          [Queue / Load balancer]
               |
      [Multiple c8g.4xlarge instances]
      [Auto-scaling based on request rate]
```

The inference server is a thin wrapper around llama.cpp's server mode (it already
has a REST API). Each instance loads the Q4_K_M GGUF file on startup (~2s cold
start) and handles requests sequentially.

For batch workloads (eval, data generation), we use the DGX Sparks (also in our
hardware inventory) which can run the BF16 checkpoint at ~15K tokens/sec with
batching. But for production serving, Graviton4 is the economics play.

---

## Summary: The Full Stack

The deployment story completes the loop:

```
Phase 0 pre-training (RTX 5090)
    → 2.8 GB BF16 checkpoint
Phase 1 SFT (RTX 5090)
    → 2.8 GB BF16 checkpoint (fine-tuned)
Phase 2 GRPO (RTX 5090)
    → 2.8 GB BF16 checkpoint (reasoning-capable)
    → Evaluation (DGX Sparks for speed)
convert_gguf.py
    → Q4_K_M GGUF, 350 MB
Deploy to Graviton4
    → 60-80 tok/s, $0.003/1K tokens
Optional: copy to Raspberry Pi
    → 5-10 tok/s, hardware you own
```

From raw HuggingFace datasets to a model running on a $70 Raspberry Pi, with
reasoning capability verified by math benchmark scores. That's the full project.

---

*This is the final post in the current series. We'll add future posts as we
complete the 1B Trainium2 run and GRPO training.*
