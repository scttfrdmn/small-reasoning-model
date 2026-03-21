# Why We're Building a Small Reasoning Model From Scratch

*Part 1 of a series on building a small reasoning language model end-to-end.*

---

There's a thing that happens in ML right now where "building a model" means
picking one from a leaderboard, fine-tuning it on your data, and calling it
done. That's often the right engineering choice. But it's not the same as
understanding how the thing works.

We're building a reasoning-capable language model from scratch. Not "from
scratch" in the sense that we're writing our own CUDA kernels (we are,
actually, for one specific thing), but from scratch in the sense that we:

- Write the architecture ourselves
- Train the tokenizer on our own corpus
- Build the data pipeline from raw datasets
- Implement the full training recipe (pre-training → SFT → GRPO)
- Deploy to hardware we own

The goals are learning and a working artifact. This series documents both:
what we built, why we made each choice, what broke, and what we'd do differently.

---

## Why Small?

The target size is 500M–1B parameters.

This is not a compromise. It's the primary design constraint.

A 500M parameter model in 4-bit quantization fits in about 350MB. It runs
on a Raspberry Pi 5. It costs sub-cent per thousand tokens on a Graviton4
instance. You can run it on a laptop. You can put it in an application without
an API call.

The large-model framing — "scale is all you need" — has been wildly successful,
but it creates a different artifact than what we want. A model that requires a
cloud API call to function isn't a component you can build on; it's a service
you depend on. We want a component.

> **Sidebar: The Chinchilla Scaling Laws**
>
> In 2022, Hoffmann et al. ("Training Compute-Optimal Large Language Models",
> a.k.a. the Chinchilla paper) showed that most models at the time were
> undertrained — they used too many parameters for their compute budget, relative
> to what would be compute-optimal. The compute-optimal recipe for a 70B model
> is roughly 1.4T training tokens, not the ~300B that GPT-3 was trained on.
>
> But "compute-optimal" means optimal if you're optimizing *training compute*,
> not inference compute. If you train a smaller model on more tokens — overtraining
> relative to Chinchilla — you pay more training FLOPS but get a model that's
> cheaper to run at inference time, which is what matters when you're serving
> millions of requests. This is exactly what Llama 3 and Qwen3 do deliberately.
>
> We overtrain our 500M model on 10B tokens (20× Chinchilla-optimal) for the same
> reason: we care about inference efficiency, not training efficiency.
>
> *Reference: Hoffmann et al. (2022), "Training Compute-Optimal Large Language Models"
> [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)*

---

## Why Reasoning?

"Reasoning" models — models that produce extended internal scratchpads before
answering — have become significantly more capable than standard instruction
models on math, code, and logic tasks, with roughly the same architecture and
parameter count.

The capability comes from training, not architecture. The insight from DeepSeek-R1
and related work is that you can teach a model to reason by giving it explicit
reinforcement signal for *correct answers* on problems with verifiable ground
truth. The model discovers that generating a scratchpad before answering
improves its probability of being right.

The practical value of this: a 500M parameter model with proper reasoning training
can outperform a 1B model without it on structured problem domains. We get more
capability per parameter by being deliberate about what we're training for.

> **Sidebar: DeepSeek-R1 and the Reasoning Training Insight**
>
> DeepSeek-R1 (2025) showed that reinforcement learning from verifiable rewards —
> without any human feedback or learned reward model — can produce strong reasoning
> behavior. The key insight: if you restrict RL training to problems where you can
> automatically verify correctness (math with ground-truth answers, code with test
> cases), you can provide dense reward signal without human labelers.
>
> Critically, DeepSeek-R1 found that when trained this way, the model *spontaneously
> develops* chain-of-thought reasoning — it learns to extend its thinking before
> answering because this improves its reward. The scratchpad isn't explicitly
> supervised; it emerges from the reward structure.
>
> Our approach follows this recipe: supervised fine-tuning to establish the format,
> then GRPO (a variant of PPO adapted for groups of samples) to reinforce correct
> reasoning on verifiable domains.
>
> *Reference: DeepSeek-AI (2025), "DeepSeek-R1: Incentivizing Reasoning Capability
> in LLMs via Reinforcement Learning" [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)*

---

## The Training Recipe

The model is trained in three sequential phases:

```
Phase 0: Pre-training   →  9.86B tokens, cross-entropy loss, raw language modeling
Phase 1: SFT            →  ~2M examples, loss on assistant turns only, <think> format
Phase 2: GRPO           →  ~10K problems, RL with verifiable rewards
```

Each phase builds on the last. You can't skip straight to GRPO — RL from a raw
pre-trained model is catastrophically unstable. The base model needs to be able
to follow instructions (SFT) before it can learn to reason under reward pressure
(GRPO).

---

## Design Principles

These constrain every decision in the project. We return to them throughout the
series.

**1. Small first.** The model is designed at target size, not quantized down
from something larger. This matters because architectural choices (embedding
size, head count, layer count) interact with inference efficiency in ways you
can't fix post-hoc with quantization.

**2. Tile-aligned throughout.** All hidden dimensions are multiples of 128.
This is a hardware constraint from AWS Trainium2, which has 128×128 systolic
array tiles. Violating this wastes hardware. We'll explain this in detail in
Part 3 (Hardware).

**3. Quantization-friendly by construction.** Decisions like tied
embeddings, no bias terms, and power-of-2 dimensions aren't clever design
choices — they're properties that make quantization (INT8, INT4) work cleanly
without precision pathologies.

**4. Reasoning is training, not architecture.** We use a standard consensus
architecture (GQA, SwiGLU, RoPE, pre-norm). The reasoning capability comes
entirely from the training recipe. We're not betting on exotic architecture.

**5. Verifiable domains only for RL.** Phase 2 GRPO is restricted to math,
code, and formal logic — domains with checkable ground truth. "Did the model
reason well about history?" is unanswerable. "Did this Python program pass
its tests?" is not.

---

## What This Series Covers

Each post is self-contained but builds on the previous ones:

- **Part 2: Architecture** — every component choice explained with alternatives
  considered and rejected
- **Part 3: Hardware** — why Trainium2's matrix dimensions determine our model
  dimensions, and what tile alignment actually means
- **Part 4: Tokenizer** — BPE from scratch, why individual digit tokenization
  matters for arithmetic, the `<think>` special token
- **Part 5: Data Pipeline** — 10B tokens from HuggingFace, curriculum mixing,
  quality filtering, deduplication
- **Part 6: Debugging** — two hours of GPU at 98% with zero logged steps, and
  the two-layer root cause
- **Part 7: SFT** — why you mask the prompt from the loss, what the `<think>`
  format does during training
- **Part 8: GRPO** — the RL algorithm, verifiable rewards, and why binary
  feedback is enough
- **Part 9: Inference** — GGUF format, quantization theory, Graviton4 economics

The code for all of this is at
[github.com/scttfrdmn/small-reasoning-model](https://github.com/scttfrdmn/small-reasoning-model).
