# Small Models Have a Place: Building a Reasoning Model From Scratch

A blog series covering the full stack of building a small language model with
reasoning capability — from hardware ecosystem through architecture, training
recipe, data pipelines, and deployment across a heterogeneous fleet.

**Thesis:** Small models (500M–1B parameters) are interesting in their own right,
not just compromises on the way to something larger. They're private, fast,
cheap, deployable on hardware you own — and with proper reasoning training, they
punch above their weight on structured problem domains. This series is about
building one that actually reasons.

**The unifying principle:** Alignment — not in the RLHF sense, but in the
engineering sense. Efficiency and reliability come from removing friction between
your system and its constraints. This shows up at every level: hardware dimensions
that match tensor core tiles, training formats that match inference formats, reward
signals that match actual objectives, model outputs that match human intent. The
series is an exercise in identifying those gaps and closing them deliberately,
from silicon up.

**Hardware:** RTX 5090 (primary training), RTX 4070 Ti SUPER (mid-range reference),
2× Titan RTX NVLink (multi-GPU), 2× DGX Sparks GB10 (linked, 256GB unified),
4× Jetson Orin NX (edge, NVDLA). Cloud (Trainium2, etc.) as excursions.

**Audience:** People who want to understand how modern reasoning models actually
work, including the non-obvious systems decisions and the bugs you hit when you
try to build one yourself.

**Tone:** Technical and honest. We explain why we made each choice, what we were
wrong about, and what broke before it worked.

---

## Post Index

| # | Title | Status | Topics |
|---|-------|--------|--------|
| 1 | [Small Models Have a Place: Why We're Building This](01-motivation.md) | ✅ Written | Motivation, hardware fleet, design principles |
| 2 | [Architecture: Every Decision, Every Trade-off](02-architecture.md) | ✅ Written | GQA, QK-Norm, SwiGLU, RoPE, pre-norm, tied embeddings |
| 3 | [Hardware Ecosystem: Training, Evaluation, and Edge Deployment](03-hardware.md) | ✅ Written | Full fleet: ceres/vesta/janus/castor/pollux/Orin NX |
| 4 | [The Tokenizer: BPE, Digit Splitting, and Teaching a Model to Think](04-tokenizer.md) | ✅ Written | BPE mechanics, digit tokenization, `<think>` tokens |
| 5 | [10 Billion Tokens: Building a Pre-Training Data Pipeline](05-data-pipeline.md) | ✅ Written | Quality filtering, deduplication, curriculum mixing |
| 6 | [Two Deadlocks and a GPU at 98%: Debugging the Training Infrastructure](06-debugging.md) | ✅ Written | Futex deadlocks, Rust thread pools, GIL interactions |
| 7 | [Phase 1 SFT: Loss Masking and Teaching a Model to Think](07-sft.md) | ✅ Written | SFT mechanics, loss masking, `<think>` format |
| 8 | [Phase 2 GRPO: Reinforcement Learning With Verifiable Rewards](08-grpo.md) | ✅ Written | GRPO algorithm, verifiable rewards, RL stability |
| 9 | [Inference at the Edge: GGUF, Quantization, and the Full Fleet](09-inference.md) | ✅ Written | GGUF format, quantization theory, hardware deployment |
| 10 | [What Determines Context Length? A Systems-Level Explanation](10-context-length.md) | ✅ Written | KV cache, O(T²) attention, RoPE extrapolation, per-hardware analysis |
| 11 | [The Real Use Case: Structured Intent and Why Small Reasoning Models Matter](11-structured-intent.md) | ✅ Written | SI pipeline, format alignment, si-go-v1 results, hardware deployment |
| 12 | [The Self-Improving System: Alignment, Verification, and the Architecture That Builds Itself](12-self-improving-system.md) | ✅ Written | Corrected architecture (reasoning model inside endless), "?" target, self-improvement dynamics, merge/splinter/refactor, bootstrap |

---

## Code

All code is at [github.com/scttfrdmn/small-reasoning-model](https://github.com/scttfrdmn/small-reasoning-model).

Posts link to specific commits and files so you can read the code alongside
the explanation.
