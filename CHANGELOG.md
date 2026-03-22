# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `inference/convert_gguf.py` — full GGUF export implementation: two-stage
  strategy (BF16 GGUF via `gguf` Python package + external `llama-quantize`
  for Q4_K_M/Q8_0); maps our weight names to llama.cpp conventions; writes
  all llama-arch GGUF metadata including `rope_base=500000` (critical for
  long-context inference); handles tied embeddings by writing `output.weight`
  from the shared embedding; CLI: `uv run python inference/convert_gguf.py
  --checkpoint best.pt --tokenizer tokenizer_output --config 500m --output model-bf16.gguf`
- `eval/harness.py` — full lm_eval wrapper: implements `SmallReasoningLM`
  registered as `"small_reasoning"` with all three required lm_eval methods
  (`loglikelihood`, `loglikelihood_rolling`, `generate_until`); uses KV-cache
  for efficient generation; left-pads batches for uniform loglikelihood
  computation; greedy decoding with stop-string truncation for MATH/GSM8K
- `eval/benchmark.py` — convenience benchmark runner: three suites (quick,
  standard, full) covering arc_challenge, gsm8k, hellaswag, mmlu, hendrycks_math;
  writes timestamped JSON to `results/`; computes `math_vs_hellaswag` ratio
  to track GRPO improvement without commonsense regression
- `inference/serve.py` — FastAPI inference server: `POST /generate` accepts
  `{"prompt", "max_tokens", "temperature"}` and returns `{"text"}`; `GET /health`
  liveness probe; KV-cache generation with greedy (`temperature=0`) and
  multinomial sampling; model + tokenizer loaded once at startup via lifespan
  context manager; CLI: `uv run srm-serve --checkpoint best.pt --config 500m
  --tokenizer tokenizer_output --port 8080`
- `pyproject.toml` — add `[inference]` optional dependency group with `gguf>=0.6`;
  add `srm-eval`, `srm-benchmark`, and `srm-serve` CLI entry points
- `docs/blog/12-self-improving-system.md` — blog post: "The Self-Improving System:
  Alignment, Verification, and the Architecture That Builds Itself"
- `docs/blog/11-structured-intent.md` — blog post: "The Real Use Case: Structured
  Intent and Why Small Reasoning Models Matter"
- `docs/SYSTEM_ARCHITECTURE.md` — design document: three-project ecosystem
  (endless-v2 + structured-instruct + small-reasoning-model) as one system
- `docs/blog/README.md` — added posts 11–12 to series index
- `training/pretrain.py` — Apple MPS backend support for local smoke testing on
  Apple Silicon; auto-detects MPS when CUDA is unavailable
- `data/preprocess.py` — full implementation of pre-training data pipeline:
  streaming HuggingFace download, heuristic quality filter, SHA-256 dedup,
  `MixedStreamSampler` for three-stage curriculum mixing; outputs `train.jsonl`
- `data/sft_format.py` — full implementation of SFT dataset formatter:
  NuminaMath-CoT, OpenHermes-2.5, CodeFeedback, Orca-Math; wraps CoT in
  `<think>` tags; writes `sft_train.jsonl` / `sft_val.jsonl` (95/5 split)
- `data/grpo_dataset.py` — full implementation of GRPO verifiable problem set:
  MATH (Hendrycks), GSM8K, NuminaMath-TIR, LogiQA; `filter_by_difficulty()`
  keeps only 20–80% pass-rate problems for GRPO training signal

### Fixed
- `training/sft.py` — fix `SFTDataset` file discovery: priority chain checked
  for `{split}.jsonl` (i.e. `train.jsonl`/`val.jsonl`) which don't exist, then
  fell through to glob all `*.jsonl`, loading both `sft_train.jsonl` and
  `sft_val.jsonl` for both splits; train and val used identical 2.15M-example
  sets so val loss had no independent signal and early stopping was invalid;
  fix adds `sft_{split}.jsonl` as the first-priority pattern in the chain
- `training/rewards.py` — fix `code_execution_reward()` placeholder: after
  `exec(code, ns)`, find the first callable in `ns`, call it with `tc["input"]`,
  compare return value to `tc["expected_output"]` with exact match or
  `math.isclose` for floats; previously `passed += 1` unconditionally granted
  credit regardless of whether the function returned the correct answer
- `training/grpo.py` — remove duplicate prefill call in `generate_completions()`;
  first call's result was immediately discarded before a second identical call;
  removing it halves prefill compute cost for every GRPO generation step
- `training/pretrain.py` — fix futex deadlock: `IterableDataset + num_workers=2
  + pin_memory=True + CUDA` caused main process to block on a futex; fixed by
  setting `num_workers=0, pin_memory=False`
- `data/preprocess.py` — fix `KeyError` when dead source re-enters stage mix;
  `stream()` now intersects `_generators.keys()` with `allowed_sources`
- `data/preprocess.py` — replace gated code datasets (`the-stack-v2-train-smol-ids`,
  `starcoderdata`) with public `codeparrot/github-code` (Python filter)
- `data/grpo_dataset.py` — fix MATH dataset source: `lighteval/MATH` removed;
  use `EleutherAI/hendrycks_math` with all 7 sub-configs

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

[Unreleased]: https://github.com/scttfrdmn/small-reasoning-model/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/scttfrdmn/small-reasoning-model/releases/tag/v0.1.0
