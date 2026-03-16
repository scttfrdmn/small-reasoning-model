# Small Reasoning Model Specification
**Codename: TBD** | Version 0.1 | March 2026

---

## 0. Design Principles

These are constraints, not preferences. Every downstream decision is checked against them.

1. **Small first.** The model is designed at target size, not quantized down from a larger one.
   Inference must be viable on commodity compute (Graviton4, llama.cpp, mini-PC cluster).

2. **Tile-aligned throughout.** All hidden dimensions, head dimensions, and vocabulary sizes
   are multiples of 128. This is a first-class constraint for Trainium2 NeuronCore systolic
   array efficiency (128×128 BF16, 256×128 FP8). Every matrix multiply maps cleanly to tiles.

3. **Quantization-friendly by construction.** Power-of-2 or 128-aligned dimensions quantize
   cleanly to INT8 and INT4. Weight tying (shared input/output embeddings) halves embedding
   memory at inference. Head dim = 128 maps directly to llama.cpp GGUF block quant layouts.

4. **Reasoning is training, not architecture.** The base model uses a settled consensus
   architecture (GQA + QK-Norm + SwiGLU + RoPE + pre-norm). Reasoning capability comes
   entirely from the training recipe (SFT → GRPO). No exotic architectural bets.

5. **Verifiable domains only for GRPO.** Reinforcement learning requires cheap verification.
   Phase 3 training is restricted to math, code, and formal logic — domains with ground truth.
   Natural language "reasoning" is a Phase 2 (SFT) concern only.

---

## 1. Architecture

### 1.1 Block Design

Standard pre-norm transformer decoder. No departures from consensus.

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

**Component rationale:**

| Component | Choice | Why |
|---|---|---|
| Normalization | Pre-norm RMSNorm | Training stability; standard at this scale |
| Attention | GQA | Smaller KV cache; faster inference |
| Attn stability | QK-Norm (per head) | Training stability from gallery consensus; critical at small scale |
| Positional | RoPE, base=500k | Long CoT sequences; extended context without fine-tuning |
| FFN | SwiGLU | Universal; gated linear unit for better gradient flow |
| Output | Tied embeddings | Saves vocab×d_model parameters; improves sample efficiency |
| Bias terms | None | Cleaner quantization; negligible quality impact |

**QK-Norm implementation note:** Apply RMSNorm independently to Q and K projections
*before* the dot product, per head. Scale parameter initialized to 1.0.

---

### 1.2 Three-Scale Parameter Configurations

All dimensions are multiples of 128. Tile alignment is exact for Trainium2 BF16 (128) and FP8 (256).

#### Config A — 500M (Validation / 5090)

| Hyperparameter | Value | Tile check |
|---|---|---|
| `d_model` | 1280 | 1280 / 128 = 10 ✓ |
| `n_layers` | 26 | — |
| `n_heads` (Q) | 10 | 10 × 128 = 1280 ✓ |
| `n_kv_heads` (KV) | 2 | 2 × 128 = 256 ✓ |
| `head_dim` | 128 | 128 / 128 = 1 ✓ |
| `ffn_intermediate` | 3456 | 3456 / 128 = 27 ✓ |
| `vocab_size` | 32768 | 32768 / 128 = 256 ✓ |
| `max_seq_len` | 8192 | — |
| **Total params** | **~489M** | |

Parameter breakdown:
- Per-layer attention (GQA): Q(1280²) + K(1280×256) + V(1280×256) + O(1280²) = **3.93M**
- Per-layer FFN (SwiGLU): gate(1280×3456) + up(1280×3456) + down(3456×1280) = **13.27M**
- Per-layer total: **17.2M** × 26 layers = **447M**
- Embedding (tied): 32768 × 1280 = **41.9M**
- RMSNorm params: negligible
- **Total: ~489M**

#### Config B — 1B (Primary experiment / Trn2)

| Hyperparameter | Value | Tile check |
|---|---|---|
| `d_model` | 2048 | 2048 / 128 = 16 ✓ |
| `n_layers` | 20 | — |
| `n_heads` (Q) | 16 | 16 × 128 = 2048 ✓ |
| `n_kv_heads` (KV) | 4 | 4 × 128 = 512 ✓ |
| `head_dim` | 128 | 128 / 128 = 1 ✓ |
| `ffn_intermediate` | 5504 | 5504 / 128 = 43 ✓ |
| `vocab_size` | 32768 | 32768 / 128 = 256 ✓ |
| `max_seq_len` | 16384 | — |
| **Total params** | **~953M** | |

Parameter breakdown:
- Per-layer attention: Q(2048²) + K(2048×512) + V(2048×512) + O(2048²) = **10.49M**
- Per-layer FFN: 3 × 2048 × 5504 = **33.82M**
- Per-layer total: **44.3M** × 20 layers = **886M**
- Embedding (tied): 32768 × 2048 = **67.1M**
- **Total: ~953M**

#### Config C — 3B (Full experiment / Trn2 or cloud H100)

| Hyperparameter | Value | Tile check |
|---|---|---|
| `d_model` | 3072 | 3072 / 128 = 24 ✓ |
| `n_layers` | 28 | — |
| `n_heads` (Q) | 24 | 24 × 128 = 3072 ✓ |
| `n_kv_heads` (KV) | 6 | 6 × 128 = 768 ✓ |
| `head_dim` | 128 | 128 / 128 = 1 ✓ |
| `ffn_intermediate` | 8192 | 8192 / 128 = 64 ✓ |
| `vocab_size` | 32768 | 32768 / 128 = 256 ✓ |
| `max_seq_len` | 32768 | — |
| **Total params** | **~2.87B** | |

Parameter breakdown:
- Per-layer attention: Q(3072²) + K(3072×768) + V(3072×768) + O(3072²) = **23.59M**
- Per-layer FFN: 3 × 3072 × 8192 = **75.50M**
- Per-layer total: **99.1M** × 28 layers = **2774M**
- Embedding (tied): 32768 × 3072 = **100.7M**
- **Total: ~2.87B**

---

### 1.3 Attention Implementation Notes

**GQA layout:** Q projects to `n_heads × head_dim`, K and V each project to `n_kv_heads × head_dim`.
KV heads are replicated across Q heads in groups: `n_heads / n_kv_heads` Q heads per KV head.
All configs have integer GQA ratios: 500M = 5:1, 1B = 4:1, 3B = 4:1.

**QK-Norm:** `q = rms_norm(q)`, `k = rms_norm(k)` applied after projection, before RoPE.
Separate learned scale `γ` per head, shape `[n_heads, head_dim]` for Q and `[n_kv_heads, head_dim]` for K.
This prevents attention logit explosion at small scale where weight initialization variance
is proportionally larger.

**RoPE:** Apply after QK-Norm. Base frequency = 500,000. Full precision (no low-precision rope).

**Flash Attention:** Use FlashAttention-2 on CUDA (5090 validation). On Trainium, write NKI kernel
with tile size matched to head_dim=128 — the tile maps directly to the SBUF partition dimension.

---

## 2. Tokenizer

| Parameter | Value | Rationale |
|---|---|---|
| Algorithm | BPE | Standard; fast inference |
| Vocabulary size | 32768 | Tile-aligned (÷128=256); sufficient for en+code+math |
| Byte fallback | Yes | No unknown tokens |
| Special tokens | `<bos>`, `<eos>`, `<pad>`, `<think>`, `</think>` | CoT boundary markers |
| Digit tokenization | Individual digits | Critical for arithmetic reasoning |
| Number splitting | Yes — never merge across decimal/comma | Math integrity |

**`<think>` / `</think>` tokens** are first-class vocabulary entries, not ad-hoc strings.
The model is trained to generate explicit reasoning chains bounded by these tokens.
At inference, content between these tags can be streamed, suppressed, or logged separately.

**Digit tokenization** is a hard requirement. Models that merge multi-digit numbers ("142" → single token)
perform significantly worse on arithmetic. The tokenizer must split all digit sequences to individual
characters: "142" → ["1", "4", "2"].

---

## 3. Training Recipe

Three sequential phases. Each phase produces a checkpoint used by the next.
No phase is skipped. The GRPO phase is where the reasoning behavior emerges.

```
Phase 0: Pre-training         →  Base model (language modeling)
Phase 1: SFT                  →  Instruction following + CoT format
Phase 2: GRPO                 →  Reasoning capability (math / code / logic)
```

---

### 3.1 Phase 0 — Pre-training

**Objective:** Next-token prediction (cross-entropy loss, teacher forcing).

**Token budget:**

| Config | Tokens | Rationale |
|---|---|---|
| 500M | 10B | Validation; ~20× Chinchilla; sufficient to check loss curve shape |
| 1B | 50B | ~50× Chinchilla; overtrain deliberately for inference efficiency |
| 3B | 100B | ~33× Chinchilla; same strategy |

Overtraining small models on more tokens than Chinchilla-optimal produces better
*inference-time* quality — the loss continues to decrease even past the compute-optimal point.
Llama 3 and Qwen3 both overtrain aggressively. This is intentional.

**Data curriculum (ordered by training stage):**

| Stage | Source | Proportion | Notes |
|---|---|---|---|
| 0–30% tokens | FineWeb-Edu, DCLM | 40% | Filtered web; high educational quality signal |
| 0–30% tokens | The Stack v2 (filtered) | 25% | Code; reasoning circuit pre-training |
| 0–30% tokens | OpenWebMath, Proof-Pile-2 | 25% | Math text; essential for Phase 2 |
| 0–30% tokens | Wikipedia, Books | 10% | World knowledge anchor |
| 30–100% tokens | Same mix + NuminaMath | Increase math to 35% | Gradually upweight reasoning domains |

Data is not shuffled uniformly — curriculum matters. Math and code proportion increases
over training. Final 10% of tokens: 40% math, 40% code, 20% general.

**Quality filtering:** Perplexity filter (remove documents where GPT-2 perplexity > 10,000).
Deduplication via MinHash LSH at 3-gram level, Jaccard threshold 0.8.

**Training hyperparameters (1B reference):**

| Param | Value |
|---|---|
| Optimizer | AdamW (β1=0.9, β2=0.95, ε=1e-8) |
| Learning rate | 3e-4 peak |
| LR schedule | Cosine decay with 2% warmup; decay to 3e-5 |
| Batch size (tokens) | 2M tokens/step |
| Gradient clipping | 1.0 |
| Weight decay | 0.1 |
| Precision | BF16 mixed (BF16 forward/backward, FP32 master weights) |
| Gradient checkpointing | Enabled (reduces memory ~30%) |

Scale batch size and LR proportionally for 500M / 3B configs.
Linear LR scaling rule: `lr ∝ sqrt(batch_size)` when varying batch size.

---

### 3.2 Phase 1 — Supervised Fine-Tuning (SFT)

**Objective:** Teach instruction following and the `<think>...</think>` chain-of-thought format.
The model must learn to produce reasoning chains before GRPO can reinforce correct ones.

**Data:**

| Dataset | Size | Purpose |
|---|---|---|
| NuminaMath-CoT | ~860K | Math with step-by-step solutions |
| OpenHermes 2.5 | ~1M (filtered) | General instruction following |
| CodeFeedback | ~66K | Code with explanation |
| Orca-Math | ~200K | Math word problems with reasoning |
| Synthetic CoT | ~200K | Generated from base model, filtered for correctness |

All examples reformatted to:
```
User: {problem}
Assistant: <think>
{step-by-step reasoning}
</think>
{final answer}
```

**Hyperparameters:**

| Param | Value |
|---|---|
| Learning rate | 2e-5 (1B), 3e-5 (500M) |
| LR schedule | Cosine decay, 3% warmup |
| Epochs | 2 |
| Batch size | 128 sequences |
| Max sequence length | 4096 |
| Loss masking | Compute loss on assistant turns only |

---

### 3.3 Phase 2 — GRPO (Group Relative Policy Optimization)

**This is where reasoning is trained, not installed.**

GRPO avoids the need for a separate value/critic model. Instead, it samples G completions
per prompt, computes rewards for each, and uses the group mean as a baseline.

**Algorithm:**

```
For each training step:
  1. Sample batch of prompts p from verifiable dataset
  2. For each prompt, sample G=8 completions {o₁...o₈} from policy π_θ
  3. Compute binary reward r_i ∈ {0, 1} for each completion (verify against ground truth)
  4. Compute group advantage: A_i = (r_i - mean(r)) / std(r)
  5. Compute clipped policy gradient loss (PPO-style clip, ε=0.2)
  6. Add KL penalty: β * KL(π_θ || π_ref),  β=0.01
  7. Update θ
```

**Reward function — CRITICAL DESIGN DECISION:**

Only verifiable rewards. No learned reward model.

| Domain | Verification method | Reward |
|---|---|---|
| Math (integer answer) | Exact match after normalization | 0 or 1 |
| Math (expression) | SymPy symbolic equivalence | 0 or 1 |
| Code | Execute against test cases | fraction passing |
| Logic puzzles | Deterministic checker | 0 or 1 |

Format reward (optional, 0.1 weight): +0.1 if response contains valid `<think>...</think>` block.
This encourages the model to maintain CoT structure under RL pressure.

**Training data for GRPO:**

| Dataset | Size | Notes |
|---|---|---|
| MATH (Hendrycks) | ~12K | Competition math, levels 1–5 |
| NuminaMath | ~860K | Broad math coverage |
| GSM8K | ~8K | Grade school math, easy positives |
| MBPP / HumanEval | ~1K | Code with test cases |
| LogiQA / FOLIO | ~10K | Formal logic |

Filter to problems where the SFT model solves 20–80% correctly (curriculum difficulty).
Problems that are too easy (>80% pass) or too hard (<20% pass) don't produce useful gradient signal.

**GRPO hyperparameters:**

| Param | Value |
|---|---|
| Learning rate | 5e-7 |
| LR schedule | Constant with 1% warmup |
| Group size G | 8 |
| KL coefficient β | 0.01 |
| PPO clip ε | 0.2 |
| Max generation length | 2048 tokens |
| Temperature (sampling) | 0.8 |
| Steps | 5,000–20,000 |
| Reference model | SFT checkpoint (frozen) |

**Early stopping:** Monitor pass@1 on held-out MATH Level 3 problems.
Stop when validation performance plateaus for 500 steps.

---

## 4. Hardware Mapping

```
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 0 VALIDATION (500M)                                       │
│ Hardware: RTX 5090 (32GB)                                       │
│ Duration: ~4 days (10B tokens)                                  │
│ Purpose: Validate loss curve, tokenizer, architecture choices   │
│ Framework: PyTorch + FlashAttention-2                           │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼ (if loss curve looks healthy)
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 0 FULL RUN (1B or 3B)                                     │
│ Hardware: AWS Trn2 (trn2.48xlarge, 16× Trainium2 chips)        │
│ Duration: ~1 week (50B tokens, 1B model)                       │
│ Framework: PyTorch/XLA + Neuron SDK + NKI attention kernel      │
│ Cost estimate: ~$700–1,000 at Trn2 on-demand                   │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 1 SFT                                                     │
│ Hardware: RTX 5090 (2 epochs of ~2M examples = hours)           │
│ Duration: ~6–12 hours                                           │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│ PHASE 2 GRPO                                                    │
│ Hardware: RTX 5090 or single Trn2 instance                     │
│ Duration: 10K steps × 8 completions × 2048 tokens ≈ 20–40 hrs │
│ Cost if cloud: ~$50–100                                         │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│ INFERENCE                                                       │
│ Primary: llama.cpp (GGUF Q4_K_M) on Graviton4 / Kamrui cluster │
│ Batch eval: DGX Sparks (overkill, but already there)           │
│ ~500M Q4: ~500MB weights, runs on anything                     │
│ ~1B Q4:   ~700MB weights, runs on Raspberry Pi 5 w/ patience   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Trainium2 Implementation Notes

### 5.1 NKI Attention Kernel

The standard attention kernel shipped with Neuron SDK is generic. For head_dim=128 specifically,
a hand-tuned NKI kernel can map Q@K^T and A@V directly to the 128×128 systolic array tile
with no padding waste. This is the highest-leverage NKI optimization for this model.

Key considerations:
- SBUF size: 24MB per NeuronCore — fits the full Q/K/V tile for one attention head at seq_len ≤ 4096
- DMA transpose: K cache is stored [seqlen, head_dim]; needs transpose before TensorE
  Use DMA-on-the-fly transpose (lower bandwidth but frees TensorE for matmul)
- FP8 path: TensorE presents as 256×128 for FP8; use for forward pass where precision allows

### 5.2 Model Parallelism on Trn2

For the 1B model on trn2.48xlarge (16 chips, 32 NeuronCores):
- Tensor parallelism degree: 8 (splits attention heads and FFN across 8 NeuronCores)
- Data parallelism: 4 (4 replica groups)
- Pipeline parallelism: not needed at 1B

For the 3B model, increase TP degree to 16.

### 5.3 Compilation

Static graph compilation is required for Trainium. This means:
- Fixed batch size and sequence length at compile time
- Compile separate graphs for training (fixed seq_len=4096) and inference (fixed seq_len=2048)
- Use `torch_neuronx.trace()` with `input_output_aliases` for KV cache management

Pre-compile before starting training run — first compilation takes 15–30 minutes.

---

## 6. Quantization Targets (Post-Training)

All inference targets use GGUF format via llama.cpp.

| Format | Size (1B) | Target hardware | Notes |
|---|---|---|---|
| BF16 | ~2 GB | 5090, Sparks | Reference; for eval |
| Q8_0 | ~1 GB | Graviton4, any 2GB+ | Near-lossless |
| Q4_K_M | ~700 MB | Graviton4, Kamrui | Recommended default |
| Q4_0 | ~550 MB | Raspberry Pi 5 | Edge deployment |
| Q2_K | ~400 MB | Microcontroller-class | Curiosity only |

Tile-aligned dimensions (all multiples of 128) map cleanly to GGUF's 32-element block
quantization scheme with no remainder handling. This is why dimension alignment matters
for inference, not just training.

**Graviton4 inference estimate (1B Q4_K_M):**
- c8g.4xlarge: ~4 vCPUs active, ~25–35 tokens/sec
- c8g.8xlarge: ~60–80 tokens/sec
- Cost: ~$0.68/hr → sub-cent per 1000 tokens at batch size 1

---

## 7. Evaluation Suite

Track these metrics at each checkpoint.

| Benchmark | Task type | Why it matters |
|---|---|---|
| MATH (Hendrycks) | Math reasoning | Primary GRPO target — must improve |
| GSM8K | Grade school math | Baseline reasoning sanity check |
| HumanEval | Code generation | Transfer from code pre-training |
| ARC-Challenge | Science QA | General reasoning, out-of-domain |
| HellaSwag | Commonsense | Regression check — must not collapse |
| MMLU (5-shot) | Knowledge | Broad capability regression |
| BIG-Bench Hard (subset) | Algorithmic reasoning | Hard generalization test |

Key ratios to watch:
- **MATH vs HellaSwag:** GRPO should improve MATH without collapsing HellaSwag
- **MATH Level 1–2 vs Level 4–5:** Model should improve proportionally across difficulty
- **Pass@1 vs Pass@8:** Large gap means high variance; close gap means consistent reasoning

---

## 8. Repository Structure (Proposed)

```
small-reasoning-model/
├── configs/
│   ├── model_500m.yaml        # Config A hyperparams
│   ├── model_1b.yaml          # Config B hyperparams
│   └── model_3b.yaml          # Config C hyperparams
├── model/
│   ├── architecture.py        # Model definition (PyTorch)
│   ├── attention.py           # GQA + QK-Norm + RoPE
│   ├── ffn.py                 # SwiGLU FFN
│   └── nki_attention.py       # Trainium NKI attention kernel
├── tokenizer/
│   ├── train_tokenizer.py     # BPE training script
│   └── tokenizer_config.json  # Special tokens, settings
├── data/
│   ├── preprocess.py          # Filtering, dedup, curriculum mixing
│   ├── sft_format.py          # Reformat datasets to <think> template
│   └── grpo_dataset.py        # Difficulty-filtered verifiable problems
├── training/
│   ├── pretrain.py            # Phase 0 training loop
│   ├── sft.py                 # Phase 1 SFT
│   ├── grpo.py                # Phase 2 GRPO
│   └── rewards.py             # Verification functions (SymPy, exec)
├── eval/
│   ├── benchmark.py           # Run evaluation suite
│   └── harness.py             # lm-evaluation-harness integration
└── inference/
    ├── convert_gguf.py        # Export to GGUF for llama.cpp
    └── serve.py               # Simple inference server
```

---

## 9. Open Questions / Decision Log

| Question | Decision | Rationale |
|---|---|---|
| Sliding window attention? | No | Full attention at 1B is tractable; SWA adds complexity without clear win at this scale |
| NoPE (periodic layers)? | No (v1) | SmolLM3 experiments with this; interesting but defer |
| MoE? | No | Below ~7B total params, routing overhead outweighs benefit |
| Vocabulary size? | 32768 | Tile-aligned; sufficient for en+code+math; expandable later |
| Process vs outcome reward? | Outcome only | Process reward requires human annotation or a trained reward model; outcome is free |
| KL reference model update? | Static | Update reference model every N steps adds complexity; static is fine for this scale |
| Synthetic pre-training data? | Phase 2 only | Synthetic CoT for SFT is well-validated; synthetic pre-training is higher risk |

---

*Spec status: DRAFT. Implementation begins with tokenizer training and 500M validation run.*
