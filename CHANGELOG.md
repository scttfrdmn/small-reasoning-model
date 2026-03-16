# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- `data/grpo_dataset.py` — corrected MATH dataset source from removed `lighteval/MATH`
  to `EleutherAI/hendrycks_math`; updated loader to iterate across all 7 topic
  sub-configs (algebra, counting_and_probability, geometry, intermediate_algebra,
  number_theory, prealgebra, precalculus) since that repo requires a config name

### Added
- `training/pretrain.py` — Apple MPS (Metal Performance Shaders) backend support for
  local smoke testing on Apple Silicon; auto-detects MPS when CUDA is unavailable;
  disables fused AdamW (CUDA-only); configures autocast with `device_type="mps"`
- `data/preprocess.py` — full implementation of pre-training data pipeline:
  streaming download from HuggingFace (FineWeb-Edu, OpenWebMath, Wikipedia,
  NuminaMath-TIR, The Stack v2 smol); heuristic quality filter (length, word count,
  non-ASCII ratio) as practical substitute for GPT-2 perplexity scoring; SHA-256
  exact dedup; `MixedStreamSampler` for three-stage curriculum mixing with
  configurable per-stage proportions; outputs `train.jsonl` + `manifest.json`
- `data/sft_format.py` — full implementation of SFT dataset downloader and reformatter:
  downloads NuminaMath-CoT, OpenHermes-2.5, CodeFeedback, and Orca-Math via HuggingFace
  streaming; wraps existing CoT solutions in `<think>` tags or applies the minimal
  `<think>\nLet me work through this.\n{answer}\n</think>\n{answer}` template for
  direct-QA sources; filters prompts >512 tokens and empty responses; writes
  `sft_train.jsonl` / `sft_val.jsonl` (95/5 split) and `manifest.json`
- `data/grpo_dataset.py` — full implementation of GRPO verifiable problem dataset builder:
  downloads MATH (Hendrycks), GSM8K, NuminaMath-TIR, and LogiQA; extracts
  ground-truth answers (`\boxed{}` extraction, `####` GSM8K parsing, option-text for
  logic); writes `grpo_raw.jsonl` with `pass_rate: null`; implements
  `filter_by_difficulty()` that loads an SFT checkpoint, generates G=8 completions per
  problem, measures pass_rate, and keeps only the 20–80% difficulty window

### Planned
- Phase 0 pre-training run on RTX 5090 (500M validation)
- Phase 0 full run on AWS Trn2 (1B primary experiment)
- NKI attention kernel for Trainium2 (`model/nki_attention.py`)
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
