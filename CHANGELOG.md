# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Phase 3 GRPO SI eval complete: JSON pass@8=14.3%, Fields pass@8=0.0%
  (vs Phase 1 GRPO 62.9%/5.7%); math-focused GRPO degraded SI capability
  installed by SFT; same pattern as Qwen2.5-Math specialization finding
- `docs/baseline-comparison.md`: add Phase 3 GRPO SI results and regression
  analysis to structured intent table and Phase 3 GRPO section
- Phase 3 GRPO complete: 5,000 steps on Phase 3 SFT checkpoint
  (batch_prompts=2, group_size=8, max_gen_tokens=512, best_reward=0.5000,
  15,475 groups skipped — 75% skip rate); math eval: pass@1=7.0%,
  pass@1_voted=11.0%, pass@8=17.0% (vs Phase 1 GRPO 4%/5%/29%); GRPO
  raised pass@1 via distribution concentration but reduced pass@8 diversity
- `docs/baseline-comparison.md`: add Phase 3 GRPO results, per-domain
  breakdown, and analysis of distribution narrowing and GSM8K regression
- Phase 3 pre-training complete: 10B tokens on math-heavy curriculum
  (48% openwebmath/numinamath, 44% fineweb-edu, 8% misc), val_loss=2.68
- Phase 3 SFT complete: 100K examples × 2 epochs from 500m_v2 checkpoint,
  best val_loss=0.7369; math result: pass@1=2%, pass@8=23% (vs Phase 1
  SFT 5%/25% — gap attributable to 18.7× fewer SFT training sequences)
- `docs/baseline-comparison.md`: add Phase 3 SFT math results and analysis
- `training/sft.py`: expose `--max_examples` CLI flag (wires through to
  `SFTDataset.max_examples`) to cap training set size and avoid OOM on
  large JSONL datasets that are loaded fully into RAM
- `docs/baseline-comparison.md` — reference tables for peer model comparison
  (Qwen2.5-0.5B, Qwen2.5-Math-1.5B, TinyLlama-1.1B, Phi-1.5 vs SRM variants)
  on the same math and SI eval suites
- Blog post 17: "What Comparing to Baselines Told Us" — narrative interpretation
  of the baseline results and implications for project direction: data
  composition > volume, SI ceiling near 100% for competent small models,
  specialization destroys generality (Qwen2.5-Math: 64% math, 0% SI), and
  revised endpoint considerations for the project
- `docs/mig-guidance.md` — practical reference for when researchers should
  partition a GPU with MIG: 6 cases where it helps, 5 where it hurts, decision
  heuristics, and MIG-capable GPU list including the RTX PRO 6000 Blackwell
- Blog post 16: "When to Partition a GPU: MIG for Researchers" — mental model
  for MIG (throughput per portfolio-hour, not time-to-finish-one-job), examples
  tied to SRM workloads, and diagnostic questions for deciding when to partition
- `eval/math_eval.py` — domain-specific math evaluation script that reuses the GRPO
  generation pipeline (`build_prompt`, `generate_completions`, `combined_reward`) to
  measure pass@1, pass@8, and mean reward on math problems; supports `--source` filter,
  train-set exclusion, and per-domain breakdown; correctly loads checkpoint formats with
  `model` or `model_state_dict` keys
- Blog post 15: "What GRPO Actually Learned (And What It Didn't)" — analysis of GRPO
  eval results at 500M scale: stochastic vs consolidated knowledge, syntax vs semantics
  of reasoning, domain specialization effects, process reward model motivation
- `eval/si_eval.py` — Structured Intent evaluation script; tests whether the model can
  generate valid SI specs (JSON with function/signature/behavior fields) from natural
  language task descriptions; reports format score, field completeness, quality, and
  pass@1/pass@k metrics with per-category and per-difficulty breakdowns
- `eval/si_rewards.py` — SI-specific reward functions: `reward_si_format()` (JSON
  extraction after `</think>`), `reward_si_fields()` (required field presence),
  `reward_si_quality()` (field content plausibility), `reward_si_combined()`
- `eval/si_test_cases.jsonl` — 35 hand-crafted SI test cases across 7 categories
  (string, search, sort, array, data_structure, utility, validation, decomposition)
  and 3 difficulty levels
- `data/sft/si_examples.jsonl` — 65 Structured Intent SFT training examples: each
  has `<think>` reasoning followed by a valid JSON spec with function, signature,
  behavior, constraints, and examples fields; covers utility functions, data structures,
  string operations, decomposition tasks, and natural language prompts
- `inference/serve.py` — add majority voting and best-of-N support: new request fields
  `n_samples` and `voting` ("none"|"majority"|"best_of_n"); majority picks the most
  common extracted answer, best_of_n scores completions by JSON structural quality
- `eval/math_eval.py` — add `pass@1_voted` metric: extracts answers from all group
  completions, picks the most common via majority vote, checks against ground truth

### Changed
- `eval/math_eval.py` — import `normalize_answer` and `_extract_final_answer` from
  GRPO for majority voting answer extraction

### Fixed
- `training/sft.py` — fix `_enable_gradient_checkpointing()`: add `collect_kv`
  parameter to `cp_forward` wrapper (same fix as `training/grpo.py`); without it,
  loading GRPO checkpoints that use `collect_kv=True` during prefill causes
  `TypeError: unexpected keyword argument`
- `training/pretrain.py` — same `collect_kv` fix for `checkpointed_forward`
  wrapper in `_enable_gradient_checkpointing()`; pretraining would fail with
  `TypeError` on first forward pass

### Fixed
- `training/grpo.py` — fix `build_prompt()`: strip trailing EOS token from encoded
  prompts; the tokenizer post-processor appends EOS to every `encode()` call, causing
  the model to generate after end-of-sequence and produce degenerate context-free output
  (same bug previously fixed in `data/grpo_dataset.py` and `inference/serve.py`)
- `training/grpo.py` — fix `_extract_final_answer()`: after extracting text following
  `</think>`, recursively search for `\boxed{}`, `= <number>`, or standalone numbers
  within that text; previously returned the raw post-think sentence (e.g.
  `"therefore, $a=\boxed{2}$"`) which never matched the ground-truth `"2"`
- `training/grpo.py` — fix `_enable_gradient_checkpointing()`: add `collect_kv`
  parameter to the `cp_fwd` wrapper; without it, `SmallReasoningModel.forward()` calls
  `block(x, collect_kv=True)` during KV-cache prefill and gets `TypeError: unexpected
  keyword argument`; also pass through (skip checkpointing) when `collect_kv=True`
  since prefill needs the returned KV tensors
- `training/grpo.py` — fix `generate_completions()` prefill: pass `kv_caches=[]` to
  signal collect-KV mode; without it, `model(batch)` runs in training mode
  (`kv_caches=None`) and returns `None` for KV caches, causing decode steps to run with
  no context (each token only sees itself, not the prompt or prior tokens)
- `training/grpo.py` — use per-example domain (`ex.get("domain", cfg.domain)`) in
  reward computation; the filtered dataset has per-problem domains (`math_exact` vs
  `math_sympy`) but `combined_reward` was using the global `cfg.domain` for all problems
- `training/grpo.py` — widen `GRPOConfig.min_pass_rate` 0.20→0.05 and `max_pass_rate`
  0.80→0.95 to match `data/grpo_dataset.py` widened filter; with the old 0.20 threshold,
  the dataset loader re-filtered 82% of examples (pass_rate=0.125 problems)
- `training/grpo.py` — add missing `compress_kv` field to `GRPOConfig` dataclass;
  the CLI `--compress-kv` flag and constructor call passed `compress_kv=` but the field
  was never defined, causing `TypeError` on startup
- `data/grpo_dataset.py` — widen difficulty filter window from 20–80% to 5–95%:
  with group_size=8 the 20–80% window requires exactly 2–6/8 correct, producing ~2%
  keep rate on the SFT checkpoint; widened to ≥1/8 and ≤7/8 correct (GRPO only needs
  within-group reward variance — all-zero or all-one groups produce zero advantage)
- `training/rewards.py` — `_extract_answer()`: add layered fallback extraction for
  models that don't close `</think>`: (1) `\boxed{...}` LaTeX notation, (2) last
  `= <number>` pattern on a line, (3) last standalone number/expression in the
  response; previously the fallback returned the full response paragraph which never
  matched the ground-truth answer, causing 0% keep rate in the GRPO difficulty filter
- `data/grpo_dataset.py` — seed assistant turn with `<think>` (`Assistant: <think>`)
  instead of bare `Assistant:`; seeding the chain-of-thought token empirically improves
  arithmetic accuracy (150→1500 observed on 5×300); also prepend `<think>` to
  the decoded completion before `compute_reward()` so `_extract_answer` sees the
  full response including the seeded opening tag
- `model/architecture.py` — fix KV-cache decode attention: `is_causal=True` with T_q=1
  and T_k=T_cache+1 creates a 1×T_k lower-triangular mask that only unmasks column 0 —
  every generated token attended only to the very first token in the sequence, producing
  context-free generation despite the KV cache being correctly populated; fixed by setting
  `is_causal=False` when `kv_cache is not None` (decode mode; causality is already
  guaranteed structurally since all cached positions precede the current token); also fixed
  the manual O(T²) fallback path which had the same T_q=1 vs T_k mismatch
- `model/architecture.py` — fix KV cache prefill: `kv_caches=[]` now signals
  "collect-but-no-prior-cache"; `CausalSelfAttention` returns KV when `collect_kv=True`
  even with no prior cache to prepend; all callers updated to pass `kv_caches=[]` for
  the prefill step so decode steps receive full context instead of running in isolation
  (root cause of the repeated-token / context-free generation bug)
- `inference/serve.py`, `data/grpo_dataset.py` — strip trailing EOS token from
  encoded prompts before generation; the tokenizer post-processor appends EOS to
  every `encode()` call, causing the model to generate after end-of-sequence and
  produce degenerate repeated-token output
- `data/grpo_dataset.py` — wrap raw problem prompts in `User: {prompt}\nAssistant:`
  instruction format before encoding; the SFT model was trained on this format and
  produces incoherent output without it
- `training/sft.py` — `sft_loss()`: apply causal-LM next-token shift (`logits[:, :-1]` /
  `labels[:, 1:]`) before computing cross-entropy; the previous version omitted the shift,
  allowing the model to trivially achieve ~0 loss by echoing the current token via
  self-attention rather than learning to predict the next token
- `training/sft.py` — add checkpoint retention policy (keep only the latest `step_*.pt`,
  deleting the previous on each save); bump `save_every` default from 500 → 5000 to prevent
  disk exhaustion on long runs (121K-step SFT at 2.8 GB/ckpt would otherwise write 677 GB)

### Added
- `model/kv_compress.py` — TurboQuant KV cache compression (Google Research,
  March 2026): two-stage PolarQuant+INT8 algorithm compressing KV caches ~2×
  with zero accuracy loss at head_dim=128. `CompressedKV.compress(k, v)` splits
  K into magnitude (float16) + INT8 unit direction (no per-block normalization
  constants — the TurboQuant insight), V into INT8 with per-head scale.
  `forward_compressed()` is a transparent drop-in wrapper handling decompress →
  forward → recompress per decode step. `verify_compression()` validates
  round-trip accuracy, attention dot-product error, and softmax weight error
  analytically without requiring a GPU. CLI: `python -m model.kv_compress
  --head-dim 128 --seq-len 512`
- `model/__init__.py` — export `CompressedKV`, `compress_kv_caches`,
  `decompress_kv_caches`, `forward_compressed` alongside existing model exports
- `inference/serve.py` — `--compress-kv` flag: when enabled, compresses prefill
  KV caches immediately after population and uses `forward_compressed` for each
  decode step, halving KV cache memory for long-context generation
- `training/grpo.py` — `compress_kv` field in `GRPOConfig` and `--compress-kv`
  CLI flag: applies TurboQuant during GRPO generation loop, enabling larger
  `group_size` or longer `max_gen_tokens` on the same GPU
- `training/grpo.py` — DAPO + Dr. GRPO improvements (previously in sandbox,
  now merged): Clip-Higher (asymmetric PPO clip [0.80, 1.28] prevents entropy
  collapse), token-level policy gradient loss (avoids vanishing gradients on
  long CoT), dynamic sampling (skip uniform-reward groups), length-debiased
  advantages (Dr. GRPO: normalize by completion length before group statistics)
- `inference/convert_gguf.py` — full GGUF export: two-stage strategy (BF16 GGUF
  via `gguf` package + external `llama-quantize`); maps weight names to llama.cpp
  conventions; writes all llama-arch metadata including `rope_base=500000`;
  handles tied embeddings; CLI: `uv run python inference/convert_gguf.py
  --checkpoint best.pt --tokenizer tokenizer_output --config 500m --output model-bf16.gguf`
- `eval/harness.py` — lm_eval wrapper: `SmallReasoningLM` registered as
  `"small_reasoning"` with all three lm_eval methods (`loglikelihood`,
  `loglikelihood_rolling`, `generate_until`); left-padded batches; KV-cache
  generation with stop-string truncation for MATH/GSM8K
- `eval/benchmark.py` — benchmark suite runner: quick/standard/full suites
  covering arc_challenge, gsm8k, hellaswag, mmlu, hendrycks_math; timestamped
  JSON output; `math_vs_hellaswag` ratio to track GRPO without commonsense regression
- `inference/serve.py` — FastAPI inference server: `POST /generate` with
  temperature sampling, `GET /health`; KV-cache generation; `--compress-kv` flag
  for TurboQuant; CLI: `uv run srm-serve --checkpoint best.pt --config 500m --port 8080`
- `pyproject.toml` — add `[inference]` optional group with `gguf>=0.6`;
  add `srm-eval`, `srm-benchmark`, and `srm-serve` CLI entry points
- `docs/blog/12-self-improving-system.md` — blog post: "The Self-Improving System:
  Alignment, Verification, and the Architecture That Builds Itself"
- `docs/blog/11-structured-intent.md` — blog post: "The Real Use Case: Structured
  Intent and Why Small Reasoning Models Matter"
- `docs/SYSTEM_ARCHITECTURE.md` — design document: three-project ecosystem
  (endless-v2 + structured-instruct + small-reasoning-model) as one system
- `docs/blog/13-kv-cache-compression.md` — blog post: "Compressing the KV Cache:
  TurboQuant and the Memory Wall": PolarQuant algorithm, unit vector insight
  (no per-block normalization constants), head_dim=128 error bounds, phase-by-phase
  impact table, implementation details, Stage 2 QJL limitations, verification output
- `docs/blog/14-generation-debugging.md` — blog post: "Five Bugs That Broke
  Generation (And What Each One Taught Us)": SFT loss causal shift, disk
  exhaustion from no checkpoint retention, EOS-terminated prompts, SFT
  instruction format missing in GRPO filter, KV prefill returning None,
  is_causal=True masking all context in decode mode
- `docs/blog/README.md` — added posts 11–14 to series index
- `training/pretrain.py` — Apple MPS backend support for local smoke testing
- `data/preprocess.py` — full pre-training data pipeline: streaming HuggingFace
  download, quality filter, SHA-256 dedup, `MixedStreamSampler` curriculum mixing
- `data/sft_format.py` — full SFT dataset formatter: NuminaMath-CoT,
  OpenHermes-2.5, CodeFeedback, Orca-Math; writes `sft_train.jsonl` / `sft_val.jsonl`
- `data/grpo_dataset.py` — full GRPO problem set builder: MATH, GSM8K,
  NuminaMath-TIR, LogiQA; `filter_by_difficulty()` keeps 20–80% pass-rate window

### Fixed
- `eval/harness.py` — fix `lm_eval.simple_evaluate` import: function lives in
  `lm_eval.evaluator.simple_evaluate` in lm_eval 0.4.x, not at the top-level module
- `eval/benchmark.py` — same fix: `from lm_eval.evaluator import simple_evaluate`
- `data/grpo_dataset.py` — fix `from rewards import ...` to `from training.rewards import ...`
  (bare module import fails when running from project root via `uv run`)
- `data/grpo_dataset.py` — fix `filter_by_difficulty()` generation loop: replaced HuggingFace
  `model.generate()` call (SmallReasoningModel has no .generate()) with manual KV-cache
  autoregressive loop matching the pattern in inference/serve.py; also fixed indentation
  bug where decode/reward computation was outside the group_size loop
- `data/grpo_dataset.py` — fix tokenizer path inference in `filter_by_difficulty()`: was
  walking two dirs up from checkpoint which produced `checkpoints/tokenizer_output` (doesn't
  exist); now checks CWD `tokenizer_output/` first (project root convention), then falls back;
  added `--tokenizer` CLI arg and `tokenizer_path` parameter for explicit override
- `model/architecture.py` — add missing `get_config(name)` helper function;
  `inference/convert_gguf.py`, `inference/serve.py`, and `eval/harness.py` all
  imported it but it was never defined — would ImportError on first use
- `model/__init__.py` — export `get_config` alongside existing model exports
- `training/grpo.py` — critical KV cache bug: decode loop discarded updated
  `kv_caches` with `_` every step; each token was decoded with no context.
  Fixed to thread cache correctly through every decode step.
- `eval/harness.py` — `_forward_logprobs()` called `.float()` on the
  `(logits, kv_caches)` tuple returned by `forward()`; would TypeError on first
  eval. Fixed: `logits, _kv = self._model(input_ids)`
- `eval/harness.py` — `_generate_single()` omitted `position_offset` on decode
  steps; RoPE treated every generated token as position 0. Fixed to pass
  `position_offset = len(ctx_ids) + step_i` on each step.
- `model/kv_compress.py` — raise `atol_k` default 0.02 → 0.025 for BF16
  variance headroom; softmax MAE (0.000039) is well inside tolerance
- `training/sft.py` — fix `SFTDataset` file discovery: fell through to glob
  all `*.jsonl`, loading both split files for both train and val; added
  `sft_{split}.jsonl` as first-priority pattern so splits load correctly
- `training/rewards.py` — fix `code_execution_reward()`: was `passed += 1`
  unconditionally; now calls function with `tc["input"]` and compares to
  `tc["expected_output"]` with exact match or `math.isclose` for floats
- `training/grpo.py` — remove duplicate prefill call discarded before decode loop
- `training/pretrain.py` — fix futex deadlock with `IterableDataset + pin_memory`
- `data/preprocess.py` — fix dead-source `KeyError` in stage-mix draws
- `data/preprocess.py` — replace gated code datasets with `codeparrot/github-code`
- `data/grpo_dataset.py` — fix MATH dataset: use `EleutherAI/hendrycks_math`
  with all 7 sub-configs (replaces removed `lighteval/MATH`)

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
