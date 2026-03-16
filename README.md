# Small Reasoning Model

A small-first transformer trained to reason — designed at target size rather than quantized down from a larger model.

**Status:** Active development — pre-training not yet started. Architecture and training infrastructure are complete.

---

## Motivation

The standard path to a small reasoning model is: train large → quantize down. This project takes the opposite approach: design the parameter budget intentionally from the start, make every architectural decision with inference efficiency in mind, and train reasoning capability in via a deliberate three-phase recipe.

The central insight (from DeepSeek R1): **reasoning lives in the training recipe, not the architecture.** R1's architecture is identical to DeepSeek V3; the only difference is the training recipe. This means a model designed for small-scale inference can still acquire strong reasoning capability if trained correctly.

The academic question worth testing: *can a model under 1B parameters, designed specifically for reasoning domains, outperform a quantized 1.7B general model on math and code at a fraction of the inference cost?*

---

## Design Principles

1. **Small first.** Designed at target size. Inference must be viable on commodity compute (Graviton4, llama.cpp, mini-PC cluster).
2. **Tile-aligned throughout.** All dimensions are multiples of 128 — a hard constraint for Trainium2 NeuronCore systolic array efficiency (128×128 BF16, 256×128 FP8).
3. **Quantization-friendly by construction.** Head dim = 128 maps directly to llama.cpp GGUF block quant layouts.
4. **Reasoning is training, not architecture.** Consensus architecture (GQA + QK-Norm + SwiGLU + RoPE + pre-norm) with no exotic bets.
5. **Verifiable domains only for RL.** GRPO is restricted to math, code, and formal logic — domains with cheap ground-truth verification.

See [`small-reasoning-model-spec.md`](small-reasoning-model-spec.md) for the full specification.

---

## Architecture

Standard pre-norm transformer decoder. No departures from the 2024–2025 consensus.

```
Input → Embedding
  └─ × L layers:
      ├─ RMSNorm
      ├─ GQA Attention (QK-Norm → RoPE → scaled dot-product)
      ├─ Residual
      ├─ RMSNorm
      ├─ SwiGLU FFN
      └─ Residual
  └─ RMSNorm
  └─ LM Head (tied to embedding weights)
```

| Component | Choice | Rationale |
|---|---|---|
| Normalization | Pre-norm RMSNorm | Training stability; standard at this scale |
| Attention | GQA | Smaller KV cache; faster inference |
| Attn stability | QK-Norm (per head) | Prevents logit explosion at small scale |
| Positional | RoPE, base=500k | Long CoT sequences without fine-tuning |
| FFN | SwiGLU | Gated linear unit; better gradient flow |
| Output | Tied embeddings | Saves vocab×d\_model params; improves sample efficiency |
| Bias terms | None | Cleaner quantization |

### Model Configurations

All dimensions are multiples of 128 (Trainium2 tile alignment).

| Config | Params | d\_model | Layers | Q heads | KV heads | FFN dim | Max seq |
|---|---|---|---|---|---|---|---|
| A — 500M | ~489M | 1280 | 26 | 10 | 2 | 3456 | 8192 |
| B — 1B | ~953M | 2048 | 20 | 16 | 4 | 5504 | 16384 |
| C — 3B | ~2.87B | 3072 | 28 | 24 | 6 | 8192 | 32768 |

Config A is the validation run (RTX 5090, ~4 days). Config B is the primary experiment (Trn2, ~1 week). Config C is the full experiment (Trn2 or cloud H100).

---

## Training Recipe

Three sequential phases. Each phase produces the checkpoint used by the next.

```
Phase 0: Pre-training  →  Base model (next-token prediction, causal LM)
Phase 1: SFT           →  Instruction following + <think>…</think> CoT format
Phase 2: GRPO          →  Reasoning capability (math / code / logic only)
```

### Phase 0 — Pre-training

Overtraining is intentional. Small models on more tokens than Chinchilla-optimal produce better *inference-time* quality. Llama 3 and Qwen3 both overtrain aggressively.

| Config | Token budget | Rationale |
|---|---|---|
| 500M | 10B | ~20× Chinchilla; validate loss curve shape |
| 1B | 50B | ~50× Chinchilla; overtrain for inference quality |
| 3B | 100B | ~33× Chinchilla; same strategy |

Data curriculum increases math and code proportion over training. Final 10% of tokens: 40% math, 40% code, 20% general.

### Phase 1 — Supervised Fine-Tuning

Teaches the `<think>…</think>` chain-of-thought format. Loss is computed **on assistant turns only** — computing loss on the prompt causes the model to overfit to formatting rather than learning to generate reasoning.

All examples are reformatted to:

```
User: {problem}
Assistant: <think>
{step-by-step reasoning}
</think>
{final answer}
```

### Phase 2 — GRPO

Group Relative Policy Optimization. **This is where reasoning is trained, not installed.**

GRPO samples G=8 completions per prompt, computes binary rewards against ground truth, and uses the group mean as a baseline — no separate value model required.

Four improvements over vanilla GRPO are enabled by default:

| Improvement | Source | Problem solved |
|---|---|---|
| Clip-higher (asymmetric PPO) | DAPO | Symmetric clipping kills exploration; entropy collapses |
| Token-level policy gradient loss | DAPO | Sequence-level averaging penalizes long correct CoT chains |
| Dynamic sampling | DAPO | Uniform-reward groups waste compute (zero advantage → zero gradient) |
| Length-debiased advantages | Dr. GRPO | Short completions get larger per-token gradients — model learns to be brief, not correct |

Only verifiable rewards. No learned reward model.

---

## Tokenizer

- BPE, vocabulary size 32,768 (tile-aligned: ÷128 = 256)
- Byte-level fallback — no `<unk>` tokens ever
- **Individual digit tokenization** — `"142"` → `["1", "4", "2"]`. Hard requirement for arithmetic reasoning; merging digits degrades math performance significantly.
- `<think>` and `</think>` are first-class vocabulary entries (IDs 4 and 5)

---

## Hardware

| Phase | Hardware | Duration | Cost |
|---|---|---|---|
| Phase 0 (500M validation) | RTX 5090 32GB | ~4 days | local |
| Phase 0 (1B full run) | AWS Trn2 trn2.48xlarge | ~1 week | ~$700–1,000 |
| Phase 1 SFT | RTX 5090 | ~6–12 hours | local |
| Phase 2 GRPO | RTX 5090 or Trn2 | ~20–40 hours | ~$50–100 if cloud |
| Inference | llama.cpp / Graviton4 | — | sub-cent per 1K tokens |

Trainium2 note: the tile-aligned architecture (all dims ÷128) maps the attention matmuls directly onto the 128×128 NeuronCore systolic array with zero padding waste. This is not an accident — it is a first-class design constraint.

---

## Inference

Post-training, all models are exported to GGUF for llama.cpp.

| Format | Size (1B) | Target | Notes |
|---|---|---|---|
| BF16 | ~2 GB | 5090, Sparks | Reference; for eval |
| Q8\_0 | ~1 GB | Graviton4 | Near-lossless |
| Q4\_K\_M | ~700 MB | Graviton4, Kamrui cluster | **Recommended default** |
| Q4\_0 | ~550 MB | Raspberry Pi 5 | Edge deployment |
| Q2\_K | ~400 MB | Microcontroller-class | Curiosity only |

Graviton4 inference estimate (1B Q4\_K\_M): ~25–35 tok/s on c8g.4xlarge (~$0.68/hr → sub-cent per 1K tokens at batch=1).

---

## Getting Started

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (`brew install uv` or `pip install uv`)
- CUDA-capable GPU (RTX 5090 recommended for training; any CUDA GPU for validation)

### Install

```bash
git clone https://github.com/scottfriedman/small-reasoning-model
cd small-reasoning-model
uv sync
```

This creates a `.venv`, installs all dependencies, and installs the project in editable mode.

### Validate the architecture (no GPU needed)

```bash
uv run srm-shape
```

Runs the analytical shape checker — verifies tile alignment, GQA ratios, parameter counts, and memory estimates for all three configs without touching a GPU.

### Train and verify the tokenizer

```bash
# Quick smoke test on built-in sample corpus
uv run srm-tokenizer --mode sample --output ./tokenizer_output

# Train on your actual corpus
uv run srm-tokenizer --mode corpus --data /path/to/corpus.txt --output ./tokenizer_output
```

### Validate the training loop (no real data needed)

```bash
# 20-step smoke test with synthetic data — verifies the training loop works
uv run srm-pretrain --config 500m --mode validate
```

### Full training sequence

```bash
# Phase 0 — Pre-train
uv run srm-pretrain \
  --config 1b \
  --data_path /path/to/corpus.jsonl \
  --tokenizer_path ./tokenizer_output \
  --output_dir ./checkpoints/1b \
  --max_tokens 50_000_000_000

# Phase 1 — SFT
uv run srm-sft \
  --checkpoint ./checkpoints/1b/step_final.pt \
  --config 1b \
  --output_dir ./checkpoints/1b_sft \
  --data_dir ./sft_data

# Phase 2 — GRPO
uv run srm-grpo \
  --checkpoint ./checkpoints/1b_sft/best.pt \
  --config 1b \
  --output_dir ./checkpoints/1b_grpo
```

---

## Project Structure

```
small-reasoning-model/
├── configs/
│   ├── model_500m.yaml       # Config A: ~489M, RTX 5090 validation
│   ├── model_1b.yaml         # Config B: ~953M, Trn2 primary experiment
│   └── model_3b.yaml         # Config C: ~2.87B, Trn2 / H100 full run
├── model/
│   ├── architecture.py       # Full model: RMSNorm, RoPE, GQA, SwiGLU, SmallReasoningModel
│   ├── attention.py          # Re-export: GroupedQueryAttention, RotaryEmbedding, RMSNorm
│   ├── ffn.py                # Re-export: SwiGLUFFN
│   └── nki_attention.py      # Stub: Trainium2 NKI attention kernel (head_dim=128)
├── tokenizer/
│   └── train_tokenizer.py    # BPE tokenizer trainer + verification suite
├── data/
│   ├── preprocess.py         # Stub: filtering, dedup, curriculum mixing
│   ├── sft_format.py         # Stub: reformat datasets to <think> template
│   └── grpo_dataset.py       # Stub: difficulty-filtered verifiable problems
├── training/
│   ├── pretrain.py           # Phase 0: pre-training loop
│   ├── sft.py                # Phase 1: supervised fine-tuning
│   ├── grpo.py               # Phase 2: GRPO with DAPO + Dr. GRPO improvements
│   └── rewards.py            # Verification: math exact, SymPy, code execution
├── eval/
│   ├── shape_check.py        # Analytical shape + tile + memory validator
│   ├── benchmark.py          # Stub: evaluation suite runner
│   └── harness.py            # Stub: lm-evaluation-harness integration
├── inference/
│   ├── convert_gguf.py       # Stub: export to GGUF for llama.cpp
│   └── serve.py              # Stub: inference HTTP server
├── docs/
│   ├── architecture.md       # Deep dive: model architecture and tile alignment
│   ├── training.md           # Training recipe details and data curriculum
│   ├── hardware.md           # GPU / Trainium setup and cost estimates
│   └── evaluation.md         # Benchmarks, metrics, and interpretation
├── CHANGELOG.md
├── LICENSE                   # Apache 2.0
├── pyproject.toml            # uv project: deps, console scripts, black config
└── uv.lock
```

---

## Evaluation

Track these metrics at each checkpoint (see [`docs/evaluation.md`](docs/evaluation.md)):

| Benchmark | Type | Why it matters |
|---|---|---|
| MATH (Hendrycks) | Math reasoning | Primary GRPO target — must improve |
| GSM8K | Grade school math | Baseline reasoning sanity check |
| HumanEval | Code generation | Transfer from code pre-training |
| ARC-Challenge | Science QA | General reasoning, out-of-domain |
| HellaSwag | Commonsense | Regression check — must not collapse |
| MMLU (5-shot) | Knowledge | Broad capability regression |
| BIG-Bench Hard | Algorithmic reasoning | Hard generalization |

Key ratios to watch:
- **MATH vs HellaSwag** — GRPO should improve MATH without collapsing HellaSwag
- **Pass@1 vs Pass@8** — large gap means high variance; close gap means consistent reasoning

---

## Development

```bash
# Run black formatter
uv run black .

# Run tests
uv run pytest

# Validate architecture (all three configs)
uv run srm-shape
```

See [`docs/contributing.md`](docs/contributing.md) for conventions and workflow.

---

## Documentation

- [`docs/architecture.md`](docs/architecture.md) — model architecture deep dive
- [`docs/training.md`](docs/training.md) — training recipe and data curriculum
- [`docs/hardware.md`](docs/hardware.md) — GPU and Trainium2 setup, cost estimates
- [`docs/evaluation.md`](docs/evaluation.md) — benchmarks and interpretation
- [`small-reasoning-model-spec.md`](small-reasoning-model-spec.md) — original specification (v0.1)

---

## License

Copyright 2026 Scott Friedman

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.
