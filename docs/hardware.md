# Hardware Setup and Cost Estimates

This document covers the hardware used in each phase of the project, setup instructions,
and cost estimates for cloud training runs.

---

## Hardware Summary

| Hardware | Role | Notes |
|---|---|---|
| RTX 5090 (32GB GDDR7) | Validation, SFT, GRPO | Local; fast iteration; free per-experiment |
| AWS Trn2 (trn2.48xlarge) | Phase 0 full run (1B/3B) | 16 Trainium2 chips; ~30–40% cheaper than P5e |
| DGX Sparks (128GB unified) | Large model experiments | High memory capacity; PCIe interconnect only |
| Graviton4 (c8g) | Inference | ~25–80 tok/s at Q4\_K\_M; sub-cent per 1K tokens |

---

## RTX 5090 — Local Validation Loop

The 5090 is the fast iteration machine. Every architecture decision, hyperparameter, tokenizer
choice, and data pipeline bug should be caught here before spending cloud credits.

**Memory math for training (1B model):**

| Component | Size |
|---|---|
| Model weights (BF16) | ~2 GB |
| AdamW optimizer states (FP32 m + v) | ~12 GB |
| Gradients (BF16) | ~2 GB |
| Activations (batch=4, seq=2048) | ~2–4 GB |
| **Total** | **~18–20 GB** |

The 5090's 32GB handles 1B training comfortably with gradient checkpointing enabled.
3B training (~52 GB total) does not fit.

**Throughput:** ~170 TFLOPS BF16 → at 1B / 10B tokens (validation run), approximately 4 days.

**What to run on the 5090:**
- Architecture smoke tests (`uv run srm-shape`)
- Tokenizer training and verification
- 500M validation pre-training run (~4 days, 10B tokens)
- Phase 1 SFT on any config (~6–12 hours)
- Phase 2 GRPO on any config (~20–40 hours)
- All hyperparameter searches

**What NOT to run on the 5090:**
- 1B or 3B pre-training at full token budget — use Trn2

---

## AWS Trainium2 — Full Pre-training Runs

Trainium2 is architecturally interesting for this project, not just cheaper. The NeuronCore
contains a 128×128 systolic array (BF16) or a logical 256×128 array (FP8). Every matrix multiply
maps directly to this array when dimensions are multiples of 128. **This is why tile alignment
is a first-class design constraint**, not an implementation detail.

### Instance: trn2.48xlarge

| Spec | Value |
|---|---|
| Trainium2 chips | 16 |
| NeuronCores | 32 |
| HBM | 512 GB (32 GB per chip) |
| Interconnect | Trn2 NeuronLink (high-bandwidth torus) |
| EFA networking | 3.2 Tbps (for UltraServer multi-node) |

**Parallelism configuration (1B model):**
- Tensor parallelism (TP) degree: 8 — splits Q/K/V projections and FFN across 8 NeuronCores
- Data parallelism (DP) degree: 4 — 4 replica groups
- Pipeline parallelism: not needed at 1B

**Parallelism configuration (3B model):**
- TP degree: 16
- DP degree: 2

### Compilation

Trainium requires static graph compilation. Before starting a training run:

1. Set fixed batch size and sequence length (cannot change during training)
2. Run `torch_neuronx.trace()` with your model and example inputs — takes 15–30 minutes
3. Compiled graphs are cached; subsequent launches reuse the cache
4. Compile separate graphs for training (seq_len=4096) and inference (seq_len=2048)

**Practical tip:** Submit a short compile-only job before scheduling the full training run.
Compilation failures on Trainium are often not obvious from the error message — test with
a small batch first.

### Cost Estimates

| Run | Tokens | Duration | Est. cost (on-demand) |
|---|---|---|---|
| 1B Phase 0 | 50B | ~1 week | ~$700–1,000 |
| 3B Phase 0 | 100B | ~2 weeks | ~$2,000–3,000 |
| GRPO (any config) | — | 20–40 hrs | ~$50–100 |

Spot instances are not available for Trn2; on-demand pricing applies.
Check current pricing at [aws.amazon.com/machine-learning/trainium/](https://aws.amazon.com/machine-learning/trainium/).

### NKI Attention Kernel

For head_dim=128, a hand-tuned NKI (Neuron Kernel Interface) kernel can map Q@K^T and A@V
directly onto the 128×128 systolic array with no padding waste. The generic Neuron SDK
attention kernel adds overhead that the NKI kernel avoids.

Status: **stub** (`model/nki_attention.py`). Implement after the 500M CUDA validation run
confirms the architecture is correct. NKI programming is Triton-like — see AWS NKI docs.

Key implementation considerations:
- **SBUF:** 24 MB on-chip SRAM per NeuronCore. At head_dim=128, the full Q/K/V tile for
  one attention head fits in SBUF for seq_len ≤ 4096.
- **DMA transpose:** K cache is stored [seqlen, head_dim]; needs to be transposed to
  [head_dim, seqlen] before TensorE. Use DMA-on-the-fly transpose to free TensorE for matmul.
- **FP8 path:** TensorE presents as 256×128 for FP8. Use for forward pass where the small
  precision loss is acceptable.

---

## DGX Sparks — Large Memory Experiments

The Sparks have 128 GB unified memory (CPU+GPU shared). They are useful for:
- Running inference on large models that don't fit on the 5090 (Llama 70B, etc.)
- Data parallel training with PCIe interconnect (no NVLink, so model parallel is not efficient)
- Post-training analysis (examining weight distributions, attention patterns, etc.)

**Not recommended for:** Model-parallel training of the 1B/3B models — the PCIe bandwidth
(64 GB/s bidirectional) is 14× lower than NVLink (~900 GB/s). Use Trn2 for multi-chip training.

---

## Graviton4 — Inference

After training, models are quantized to GGUF and served with llama.cpp.

| Instance | vCPUs (active) | Throughput (1B Q4\_K\_M) | Cost | $/1K tokens |
|---|---|---|---|---|
| c8g.4xlarge | ~4 | ~25–35 tok/s | ~$0.68/hr | ~$0.005 |
| c8g.8xlarge | ~8 | ~60–80 tok/s | ~$1.36/hr | ~$0.005 |
| c8g.16xlarge | ~16 | ~120–160 tok/s | ~$2.72/hr | ~$0.005 |

Graviton4 (AWS Arm chips) has excellent performance per dollar for CPU inference. The 128-bit
NEON SIMD units accelerate the INT4 matrix multiplications in llama.cpp efficiently.

**Why not GPU inference?** At 1B Q4_K_M (~700 MB), a GPU is massively underutilized —
the model fits in 2 GB and the memory bandwidth needed is modest. CPU inference at this
scale is competitive in throughput per dollar and eliminates GPU reservation costs.

---

## Cloud Alternatives

If Trn2 is unavailable or the Neuron SDK toolchain is not set up, the 3B training run can
be run on a cloud H100:

| Provider | Instance | Cost | Notes |
|---|---|---|---|
| Lambda Labs | 1× H100 80GB | ~$2.49/hr | Reliable on-demand |
| CoreWeave | 1× H100 80GB | ~$2.06/hr | Good reliability |
| Vast.ai | 1× H100 80GB | ~$1.50–2/hr | Cheapest; variable availability |
| RunPod | 1× H100 80GB | ~$2.19/hr | Easy setup |

At ~170 GPU-hours for a 3B/100B-token run at $2/hr ≈ $340 on H100. Add ~$80 for GRPO.
Full experiment budget: **~$400–600 on H100** vs ~$700–1,000 on Trn2 (slower but no
Neuron SDK setup required). For first run, H100 may be faster to get started.
