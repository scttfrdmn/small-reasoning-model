# Building a Small Reasoning Model From Scratch

A blog series covering the full stack of building a small language model with
reasoning capability — from hardware constraints through architecture, training
recipe, data pipelines, and deployment.

**Audience:** People who want to understand how modern reasoning models actually
work, including the non-obvious systems decisions and the bugs you hit when you
try to build one yourself.

**Tone:** Technical and honest. We explain why we made each choice, what we were
wrong about, and what broke before it worked.

---

## Post Index

| # | Title | Status | Topics |
|---|-------|--------|--------|
| 1 | [Why We're Building a Small Reasoning Model](01-motivation.md) | ✅ Written | Motivation, design principles, why small |
| 2 | [Architecture: Every Decision, Every Trade-off](02-architecture.md) | ✅ Written | GQA, QK-Norm, SwiGLU, RoPE, pre-norm, tied embeddings |
| 3 | [Hardware-First Design: Trainium2, Tile Alignment, and Quantization by Construction](03-hardware.md) | ✅ Written | Systolic arrays, tile alignment, BF16/FP8, GGUF |
| 4 | [The Tokenizer: BPE, Digit Splitting, and Teaching a Model to Think](04-tokenizer.md) | ✅ Written | BPE mechanics, digit tokenization, `<think>` tokens |
| 5 | [10 Billion Tokens: Building a Pre-Training Data Pipeline](05-data-pipeline.md) | ✅ Written | Quality filtering, deduplication, curriculum mixing |
| 6 | [Two Deadlocks and a GPU at 98%: Debugging the Training Infrastructure](06-debugging.md) | ✅ Written | Futex deadlocks, Rust thread pools, GIL interactions |
| 7 | [Phase 1 SFT: Loss Masking and Teaching a Model to Think](07-sft.md) | ✅ Written | SFT mechanics, loss masking, `<think>` format |
| 8 | [Phase 2 GRPO: Reinforcement Learning With Verifiable Rewards](08-grpo.md) | ✅ Written | GRPO algorithm, verifiable rewards, RL stability |
| 9 | [Inference at the Edge: GGUF, Quantization, and Running on a Raspberry Pi](09-inference.md) | ✅ Written | GGUF format, quantization theory, Graviton4, edge targets |

---

## Code

All code is at [github.com/scttfrdmn/small-reasoning-model](https://github.com/scttfrdmn/small-reasoning-model).

Posts link to specific commits and files so you can read the code alongside
the explanation.
