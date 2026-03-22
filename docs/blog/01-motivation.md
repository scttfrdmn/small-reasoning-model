# Small Models Have a Place: Building a Reasoning Model From Scratch

*Part 1 of a series on building a small reasoning language model end-to-end.*

---

There's a quiet assumption embedded in almost every ML discourse right now: that
bigger is better, that frontier models are the target, and that small models are
compromises you make when you can't afford the real thing.

We disagree with this framing, and this series is a pushback against it.

Small models — 500M to 1B parameters — are interesting in their own right. They
have properties that frontier models don't: they're private, they're cheap, they
run locally, they can be deeply specialized, they run on hardware you own, and
they can run on hardware your customers own. A model that runs on a Jetson Orin
NX embedded in a robot, or on a Raspberry Pi at the edge of a network, or on
a DGX Sparks under a desk, solves problems that a cloud API call cannot.

The goal of this project is a reasoning-capable small model, built end-to-end,
deployed across a hardware ecosystem we own. This post explains why that's
interesting and what we're trying to learn.

---

## Small Models Are Not Compromises

The frontier model narrative treats capability as a monotone function of
parameter count. More parameters → more capable. Small models are then just
a tradeoff point on that curve: less capable, but cheaper to run.

This framing is incomplete.

**Specialization.** A 1B parameter model trained on domain-specific data can
substantially outperform a 70B general-purpose model on tasks within that domain.
The 70B model spreads capacity across everything it might ever be asked. The 1B
model can spend all its capacity on the thing you actually need. For many
real applications, a specialized small model is strictly better.

**Latency.** A 500M Q4_K_M model running locally produces the first token in
milliseconds. A frontier model API call takes ~500ms just for network round-trip.
For interactive applications where reasoning latency matters — embedded systems,
real-time decision support, interactive coding tools — local models are
categorically different.

**Privacy.** Every API call is data leaving your infrastructure. For enterprise
applications, regulated industries, or anything involving sensitive information,
a model that runs locally is not a compromise; it's a requirement.

**Ownership.** A model you train is yours. No API deprecation, no pricing
changes, no provider outage. You control the weights, the update schedule, the
deployment configuration.

**Cost at scale.** At high request volumes, the economics strongly favor
owned models. A 500M Q4_K_M model on a Graviton4 instance costs ~$0.003 per
1000 tokens. Frontier model APIs typically run $0.01–$0.03/1000 tokens — a
3–10× difference that compounds over millions of requests.

---

## Where Frontier Models Belong

This is not a manifesto against large models. Frontier models are remarkable
and will remain so. The right framing is *alongside*, not *instead of*.

Frontier models are the right choice when:
- You need general-purpose capability across an unpredictable range of tasks
- You need state-of-the-art performance on a specific benchmark regardless of cost
- You're prototyping and don't have a trained model yet
- You need multilingual or multi-domain coverage that a small model can't achieve

Small models are the right choice when:
- The task is well-defined and the domain is narrow
- Privacy, latency, or cost constraints apply
- You need the model to run on constrained hardware
- You want to understand what you're running — not treat it as a black box API

The interesting engineering question is: given a well-defined task and a hardware
target, what's the smallest model that achieves the required quality? That's a
very different question from "which leaderboard model should I use?"

---

## Why Reasoning Matters for Small Models Specifically

Reasoning capability — teaching a model to produce a scratchpad of intermediate
thinking before committing to an answer — is disproportionately valuable at
small model sizes.

A 70B model can often answer a complex math problem correctly through pattern
matching on its enormous training corpus. It has seen so many examples of similar
problems that it can retrieve the right procedure without explicit reasoning.

A 500M model doesn't have that luxury. It simply hasn't seen enough examples to
memorize procedures for all problem types. But if it can *reason* — if it can
work through a problem step-by-step inside a `<think>` block — it can solve
problems it's never seen exactly before.

The practical result: a 500M model with reasoning training can approach or match
a 1B model without it on structured domains like math and code. We buy capability
with training, not parameters.

This is the insight from DeepSeek-R1 and related work: reasoning capability comes
from training methodology, not from model size. The `<think>` scratchpad is not
a trick; it's a way of allocating the model's finite processing capacity to work
through a problem rather than attempting to retrieve an answer from memory.

> **Sidebar: Chain-of-Thought and Why It Works**
>
> Chain-of-thought prompting (Wei et al., 2022) showed that asking a model to
> "think step by step" before answering dramatically improves performance on
> multi-step reasoning tasks. The key insight: generating intermediate steps
> forces the model to allocate token-budget to the reasoning process, and each
> step conditions the next step on a richer context.
>
> The mechanism: a language model predicts the next token given all previous
> tokens. If those previous tokens contain a correct intermediate calculation
> (`= 15 / 3 = 5`), the model can predict the next step correctly. If they
> don't (jump straight to the answer), the model must rely entirely on the
> direct `question → answer` pattern.
>
> For small models with limited capacity, this matters more. Intermediate steps
> act as scaffolding that extends the effective reasoning depth beyond what the
> model's hidden state can represent directly.
>
> *Reference: Wei et al. (2022), "Chain-of-Thought Prompting Elicits Reasoning
> in Large Language Models" [arXiv:2201.11903](https://arxiv.org/abs/2201.11903)*

---

## The Hardware Ecosystem

This project runs on hardware we own. No cloud required for the primary path.

**ceres** — RTX 5090 workstation. 32GB GDDR7, 1676 TFLOPS BF16 dense,
1792 GB/s memory bandwidth, 128GB DDR5, 1.5TB NVMe. Primary training machine.
Everything in this series runs here first.

**vesta** — RTX 4070 Ti SUPER workstation. 16GB GDDR6X. Development and
validation target: small enough to keep iteration fast, large enough to run the
500M model in BF16 for quick inference tests. Also useful for benchmarking
inference on more commonly-available mid-range hardware.

**janus** — Dual Titan RTX workstation with NVLink. Two Titan RTX cards (24GB
GDDR6 each) connected via NVLink bridge at 25.78 GB/s per link — effective
48GB VRAM with high-bandwidth inter-GPU communication. This is our multi-GPU
training option: the 1B model in BF16 with a full batch fits in 48GB without
gradient checkpointing. We'll explore multi-GPU training here.

**castor and pollux** — Two NVIDIA DGX Sparks, each a GB10 Grace Blackwell
Superchip with 128GB LPDDR5X unified memory. The two are connected by a pair of
dedicated high-speed links (MTU 9000, sub-ms latency), making them a two-node
distributed inference cluster. Each unit can run the 3B model at full BF16;
together they enable split-model inference across 256GB of unified memory.
We'll use them for large-batch evaluation, long-context inference, and
distributed serving experiments.

**4× Jetson Orin NX** — NVIDIA edge AI modules. 1024 Ampere CUDA cores +
2× NVDLA v3.0 accelerators (the NPU), 16GB LPDDR5 unified memory, ~100 TOPS INT8.
Edge deployment target. NVDLA handles INT8 transformer operations; we'll benchmark
llama.cpp vs. TensorRT-LLM with NVDLA offload. Currently offline, to be
brought online for the inference phase.

**Cloud excursions** — AWS Trainium2, potentially H100. These are explicitly
experiments to explore specific questions (training at scale, hardware-specific
kernels, cost comparisons) — not the primary path. The core series is fully
self-contained on the hardware above.

> **Sidebar: What Makes the DGX Sparks Architecturally Different?**
>
> Most GPU systems have a hard memory boundary: GPU memory (fast, limited —
> 32GB on the 5090) and CPU/system RAM (slower, larger — 128GB). Moving data
> between them requires a PCIe transfer at ~64 GB/s, which is fast but not free.
>
> The DGX Sparks uses LPDDR5X memory shared by both the Grace CPU and the
> Blackwell GPU. There's no PCIe transfer; both processor types see the same
> physical memory. A tensor allocated for CPU computation and a tensor allocated
> for GPU computation live in the same address space.
>
> This changes what's possible. You can run models that don't fit in GPU-only
> memory (because there is no GPU-only memory — it's all unified). You can mix
> CPU and GPU operations without paying transfer latency. For inference, this
> means you can run a 3B model at full BF16 on hardware that fits under a desk.
>
> The trade-off: LPDDR5X has lower bandwidth than dedicated HBM2e (used in
> data center GPUs). Peak memory bandwidth is ~273 GB/s vs ~3.35 TB/s for H100.
> For batch sizes > 1, this limits throughput. For batch size 1 (interactive
> inference), it's fine.

---

## The Training Recipe

Three sequential phases, all running on our local hardware:

```
Phase 0: Pre-training  →  9.86B tokens on RTX 5090  (✅ complete)
Phase 1: SFT           →  2M examples on RTX 5090   (🔄 in progress)
Phase 2: GRPO          →  ~10K problems on RTX 5090  (⏳ pending)
```

Then evaluation and deployment across the hardware ecosystem:

```
Eval (batch)     →  DGX Sparks (throughput, long context)
Inference        →  RTX 5090 (BF16 full quality)
                    DGX Sparks (3B BF16, long context)
                    Graviton4 cloud (Q4_K_M, economics)
Edge             →  Jetson Orin NX (Q4 + NVDLA, embedded)
                    Raspberry Pi 5 (Q4_0, true edge)
```

---

## The Unifying Principle: Alignment at Every Level

Before the design principles, one idea that runs through all of them.

The word "alignment" gets used in ML to mean one specific thing: making models
do what humans want. That's a real problem. But in this project, alignment
shows up at every level of the stack, and it means the same thing at all of them:
**removing friction between your system and its constraints**.

**Hardware alignment.** Model dimensions that are multiples of 128 map cleanly
onto tensor core tiles, GGUF quantization blocks, and NVDLA INT8 processing
chunks. Misalignment doesn't cause failure — it causes waste. Boundary tiles get
padded. Some operations fall through to scalar fallback paths. The hardware is
there; you're either using it fully or not.

**Format alignment.** A model trained on structured specifications as input and
compilable Go code as output learns that transformation. A frontier model trained
on everything learns something more general and less precise. Specialization is
a form of alignment: your training distribution matches your deployment
distribution, and the model has no capacity wasted on tasks it will never be
asked to do. This is why a 7B specialized model beats a trillion-parameter
general model at its specific task.

**Intent alignment.** The whole point of the Structured Intent pipeline (Post 11)
is that human intent and model output are usually misaligned — not because the
model is bad, but because the interface between them is lossy. Structured
specifications reduce that loss. Verification closes it further.

**Training signal alignment.** GRPO uses verifiable rewards because noisy reward
signals produce noisy learning. If the reward is "does this code pass its tests,"
the model can learn what "correct" means. If the reward is "does a human think
this looks good," the gradient is pointing somewhere less precise. The reward
signal has to be aligned with the actual objective.

In each case: efficiency and reliability come from reducing the gap between what
you have and what you need. The project is an exercise in identifying those gaps
at every level — silicon, format, intent, reward — and closing them deliberately.

---

## Design Principles

These constrain every decision in the project.

**1. Small first.** The model is designed at target size, not quantized down
from something larger. Architectural choices (embedding size, head count, layer
count) interact with inference efficiency in ways you can't fix post-hoc.

**2. Run on what you own.** The full training pipeline runs on the RTX 5090.
Cloud is optional, not required. This keeps iteration cycles short — a design
change can be validated without spinning up cloud infrastructure.

**3. Hardware-aware dimensions.** All hidden dimensions are multiples of 128.
This isn't a Trainium2-specific constraint — it also ensures clean GGUF
quantization, efficient Orin NX/NVDLA inference, and maximum CUDA utilization.
Hardware alignment is a first-class concern across all targets.

**4. Quantization-friendly by construction.** Tied embeddings, no bias terms,
power-of-2 vocabulary — these make INT8/INT4 quantization clean without
precision pathologies across every deployment target.

**5. Reasoning is training, not architecture.** We use a standard consensus
architecture (GQA, SwiGLU, RoPE, pre-norm). The reasoning capability comes
entirely from the training recipe. No exotic bets.

**6. Verifiable domains only for RL.** Phase 2 GRPO is restricted to math,
code, and formal logic — domains with checkable ground truth. Natural language
"reasoning quality" is not verifiable, so we don't try to GRPO it.

---

## What This Series Covers

Each post is self-contained but builds on the previous ones:

- **Part 2: Architecture** — every component choice with alternatives and trade-offs
- **Part 3: Hardware** — our hardware ecosystem, what each machine is good for,
  and why alignment decisions span from training to edge inference
- **Part 4: Tokenizer** — BPE from scratch, digit tokenization, `<think>` tokens
- **Part 5: Data Pipeline** — 10B tokens, curriculum mixing, pre-tokenization
- **Part 6: Debugging** — two deadlocks, a GPU at 98% with zero logged steps
- **Part 7: SFT** — loss masking and teaching a model to think
- **Part 8: GRPO** — reinforcement learning with verifiable rewards
- **Part 9: Inference** — GGUF, quantization, Graviton4, Orin NX, DGX Sparks

The code for all of this is at
[github.com/scttfrdmn/small-reasoning-model](https://github.com/scttfrdmn/small-reasoning-model).
