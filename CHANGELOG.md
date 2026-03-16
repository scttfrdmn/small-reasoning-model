# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Phase 0 pre-training run on RTX 5090 (500M validation)
- Phase 0 full run on AWS Trn2 (1B primary experiment)
- NKI attention kernel for Trainium2 (`model/nki_attention.py`)
- Data preprocessing pipeline (`data/preprocess.py`)
- SFT dataset formatter (`data/sft_format.py`)
- GRPO difficulty-filtered dataset builder (`data/grpo_dataset.py`)
- lm-evaluation-harness integration (`eval/harness.py`)
- GGUF export for llama.cpp (`inference/convert_gguf.py`)
- Inference server (`inference/serve.py`)

---

## [0.1.0] - 2026-03-16

### Added

#### Model
- `model/architecture.py` — full pre-norm transformer decoder with GQA,
  QK-Norm (per-head RMSNorm on Q and K), RoPE (base=500k), SwiGLU FFN,
  tied input/output embeddings, no bias terms anywhere
- `model/attention.py` — re-export shim for `GroupedQueryAttention`, `RotaryEmbedding`, `RMSNorm`
- `model/ffn.py` — re-export shim for `SwiGLUFFN`
- `model/nki_attention.py` — stub for Trainium2 NKI attention kernel (head_dim=128)
- Three tile-aligned model configurations (all dimensions multiples of 128):
  - **Config A — ~489M** (`configs/model_500m.yaml`): validation on RTX 5090, 10B tokens
  - **Config B — ~953M** (`configs/model_1b.yaml`): primary experiment on Trn2, 50B tokens
  - **Config C — ~2.87B** (`configs/model_3b.yaml`): full experiment on Trn2 / H100, 100B tokens

#### Tokenizer
- `tokenizer/train_tokenizer.py` — BPE tokenizer trainer with:
  - Vocab size 32,768 (tile-aligned: ÷128 = 256)
  - Individual digit tokenization (hard requirement for arithmetic reasoning)
  - Byte-level fallback (no `<unk>` tokens)
  - `<think>` / `</think>` as first-class vocabulary entries
  - Verification suite (digit isolation, round-trip fidelity, compression ratio)

#### Training
- `training/pretrain.py` — Phase 0 pre-training loop with BF16 mixed precision,
  gradient accumulation, cosine LR schedule with linear warmup, gradient
  checkpointing, checkpoint save/resume, streaming dataset
- `training/sft.py` — Phase 1 SFT with assistant-turn-only loss masking and
  `<think>`-format chain-of-thought training
- `training/grpo.py` — Phase 2 GRPO with four improvements over vanilla GRPO:
  DAPO clip-higher (asymmetric PPO clipping), DAPO token-level policy gradient
  loss, DAPO dynamic sampling (skip uniform-reward groups), Dr. GRPO
  length-debiased advantages
- `training/rewards.py` — verifiable reward functions: math exact match,
  SymPy symbolic equivalence, code execution against test cases, format reward

#### Data (stubs)
- `data/preprocess.py` — stub for filtering, dedup, curriculum mixing
- `data/sft_format.py` — stub for dataset reformatting to `<think>` template
- `data/grpo_dataset.py` — stub for difficulty-filtered verifiable problem set

#### Evaluation
- `eval/shape_check.py` — analytical shape + tile alignment + memory validator
  (runs without a GPU; verifies all three configs)
- `eval/benchmark.py` — stub for evaluation suite runner
- `eval/harness.py` — stub for lm-evaluation-harness integration

#### Inference (stubs)
- `inference/convert_gguf.py` — stub for GGUF export (llama.cpp)
- `inference/serve.py` — stub for inference HTTP server

#### Project
- `pyproject.toml` — uv-managed project with editable install, console scripts
  (`srm-pretrain`, `srm-sft`, `srm-grpo`, `srm-tokenizer`, `srm-shape`)
- `uv.lock` — locked dependency manifest
- `.gitignore` — excludes checkpoints, weights, raw data, tokenizer output
- `LICENSE` — Apache 2.0, Copyright 2026 Scott Friedman
- `README.md`, `CHANGELOG.md`, and `docs/` documentation

[Unreleased]: https://github.com/scottfriedman/small-reasoning-model/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/scottfriedman/small-reasoning-model/releases/tag/v0.1.0
