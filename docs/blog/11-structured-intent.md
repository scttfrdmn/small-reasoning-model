# The Real Use Case: Structured Intent and Why Small Reasoning Models Matter

*Part 11 of a series on building a small reasoning language model end-to-end.*

---

Every project in this series has been building toward something. The architecture
choices, the training recipe, the hardware decisions — they all have a destination.
This post explains what that destination is: a system called **Structured Intent**
that uses small reasoning models to convert human intent into verified, executable
code. No cloud dependency. No API. No rate limits.

The insight that drives this: **format alignment beats model size**. A 7B model
trained on the right format dominates a trillion-parameter frontier model on its
specific task. A small reasoning model trained on structured problem decomposition
can drive an entire code generation pipeline that costs nothing per inference.

---

## The Problem With LLM Code Generation

Current LLM-based code generation has a fundamental architecture problem.

You ask a frontier model — GPT-4o, Claude, Gemini — to write code. You get code
back. Sometimes it compiles. Sometimes it passes your tests. Often it doesn't. The
model doesn't know what "correct" means for your specific task; it knows what
"plausible code that's similar to training data" looks like. That's not the same
thing.

The failure mode: the model "tries" and produces something syntactically reasonable
but semantically wrong. It has no feedback loop. It doesn't compile the code, run
the tests, or verify that the output matches the intent.

The other failure mode: it works, but at a cost. Frontier model APIs charge
$0.01–$0.15 per 1K tokens. For simple code generation tasks, a single generation
might cost $0.05–$0.50. At production volume — millions of generations per month —
that's $50K–$500K/month in API costs, plus rate limits, latency, and a hard
dependency on a provider you don't control.

There's a different architecture.

---

## Structured Intent: The Pipeline

The **Structured Intent (SI)** approach separates two jobs that frontier models
currently bundle together:

1. **Understanding intent** — figuring out what the human actually wants, clarifying
   ambiguity, decomposing complex requests into well-specified units of work
2. **Executing specifications** — reliably translating a precise specification into
   correct code

These require different capabilities. The first needs reasoning — flexible,
contextual, open-ended thinking about what the human means. The second needs
reliability — given a precise specification, produce exactly what was asked with
no hallucination.

The Structured Intent pipeline:

```
Human (vague natural language)
    ↓
Reasoning Model — interprets intent, clarifies, generates spec
    ↓
Structured Specification (machine-readable JSON/YAML)
    ↓
SI Model — translates spec to code, no dialogue
    ↓
Code Output
    ↓
Verification — compile, test, lint
    ↓
(loop back on failure)
```

The SI model (**si-go-v1**) is a 7B Qwen2.5-Coder-7B-Instruct fine-tuned on
(structured spec → Go code) pairs, where every training example was verified to
compile and pass its tests. It achieves 98% correctness on benchmark tasks.

For comparison: Claude Sonnet, Claude Opus, and GPT-4o achieve **0% compilation
rate** on the same tasks when prompted with natural language. They consistently
miss `package main` declarations and boilerplate that the structured format
enforces implicitly. The frontier models are not bad at code — they're bad at a
task they weren't trained for.

| Model | Size | Compile Rate | Pass Rate | Per-inference cost |
|-------|------|-------------|-----------|-------------------|
| **si-go-v1** | 7B | **100%** | **98%** | **$0** (self-hosted) |
| Claude Sonnet | ~200B | 0% | 0% | $$$ |
| Claude Opus | ~400B+ | 0% | 0% | $$$$ |
| GPT-4o | ~1T+ | 0% | 0% | $$$ |

This isn't a fluke. It's the core finding: **a small, specialized model trained on
the right format dominates general-purpose models of any size on its specific task.**

> **Sidebar: Why Frontier Models Fail at Structured Formats**
>
> GPT-4o and Claude are trained on massive corpora of human-written text, which
> is primarily natural language and loosely-structured code. They're very good at
> producing plausible-looking code given natural language prompts.
>
> But "plausible-looking code" and "correct Go code matching a structured spec"
> are different targets. The structured spec format used by si-go-v1 is
> machine-generated, highly regular, and requires strict output constraints
> (always `package main`, specific function signatures, no explanatory text).
>
> When frontier models are prompted with these specs, they apply their natural
> language reasoning: they add explanatory comments (not asked for), omit
> boilerplate (because human code examples usually omit it in documentation), and
> pattern-match to similar-but-wrong examples from training.
>
> Fine-tuning si-go-v1 on 10K verified (spec → code) pairs changes the model's
> behavior from "produce plausible code similar to training data" to "produce
> code that matches this specific format and passes verification." Format alignment
> changes the target entirely.

---

## Where the Small Reasoning Model Fits

Look at the pipeline again:

```
Human (vague) → Reasoning Model → Structured Spec → SI Model → Code → Verification
```

The **SI model** is solved for basic Go code generation. 7B parameters, LoRA
fine-tuned, 98% pass rate, runs self-hosted.

The **reasoning model** is where frontier models currently live — and where
a well-trained small model can replace them.

The reasoning model's job:
- Understand what the human wants, even when incompletely specified
- Ask clarifying questions when the intent is ambiguous
- Decompose complex requests into individual function specs
- Generate structured specifications that the SI model can execute reliably
- Recognize when the SI model's output doesn't match the original intent

None of this requires 200B parameters. It requires a model that:
1. Has been trained to think through problems carefully (reasoning training)
2. Has learned the structured specification format (SFT on SI examples)
3. Has been reinforced on generating specs that produce correct code (GRPO)

A 500M–1B parameter model with the right training recipe is capable of this.
That's exactly what this project is building.

---

## Why Small Matters for Reasoning

The SI pipeline has specific requirements that favor small models:

**Latency.** The reasoning model runs first and blocks the SI model. If the
reasoning step takes 5 seconds over a remote API, the whole pipeline is slow.
A local 500M model generates a structured spec in under a second, even at Q4_K_M
quantization on a Jetson Orin NX.

**Privacy.** The reasoning step receives the human's raw intent — which may include
proprietary code context, business logic, or sensitive system details. An API call
sends this to a provider's servers. A local model keeps it local.

**Cost structure.** The SI pipeline is designed to run at scale — thousands of
code generations per day in a production system. Every API call to a frontier
reasoning model multiplies cost. A local reasoning model has zero marginal cost
per inference after training.

**Control.** Reasoning behavior can be fine-tuned. A reasoning model trained
specifically to generate structured Go specifications will be better at that task
than a general-purpose frontier model, regardless of size. This is the same
lesson the SI model proved: specialization wins.

> **Sidebar: The Economics Shift**
>
> Current architecture: vague intent → frontier API → code → maybe works
> - Cost: $0.05–$0.50 per generation
> - Reliability: variable (no verification loop)
> - Privacy: data leaves your infrastructure
> - Availability: subject to API rate limits and provider uptime
>
> SI architecture with small reasoning model:
> - Cost: $0 per generation (hardware amortized)
> - Reliability: 98% (with verification loop)
> - Privacy: nothing leaves your machines
> - Availability: offline-capable
>
> For a system generating 10,000 code snippets/day, the difference is
> $150K–$1.5M/year vs. electricity and hardware depreciation.

---

## The Format Alignment Principle

This project keeps arriving at the same insight from different angles.

In the pre-training data pipeline (Post 5), we found that curriculum mixing —
exposing the model to different data types in a deliberate sequence — mattered
more than raw data volume. The model learned what the data taught it.

In SFT training (Post 7), we found that loss masking on prompt tokens matters:
the model should only be reinforced for generating responses, not for predicting
prompts it will never generate at inference. The format of training shapes what
the model learns.

In GRPO (Post 8), we use verifiable rewards — math problems with checkable
solutions, code with runnable tests — because the reward signal needs to be
accurate. Noisy rewards produce noisy learning.

The SI result makes this explicit: a 7B model trained on 10K verified (spec → code)
pairs dominates frontier models of any size at this specific task. The training
format is the training. A model that learns "here is a structured spec; output
compilable Go" has internalized a completely different objective than "here is
a natural language description; output plausible code."

For the small reasoning model, this means the SFT and GRPO phases need to include
structured specification generation as an explicit task. The model should see
examples of:

```
<think>
The user wants a function that finds the longest common prefix.
Let me decompose this into a spec:
- Input: slice of strings
- Output: string
- Edge cases: empty slice, single element, no common prefix
- Algorithm: compare character by character, stop at first mismatch
</think>

{
  "function": "longestCommonPrefix",
  "signature": {
    "inputs": [{"name": "strs", "type": "[]string"}],
    "output": {"type": "string"}
  },
  "behavior": "Find the longest common prefix among all strings",
  "constraints": [
    "Return empty string for empty input",
    "Return the string itself for single-element slice",
    "Return empty string if no common prefix exists"
  ],
  "examples": [
    {"input": [["flower", "flow", "flight"]], "output": "fl"},
    {"input": [[]], "output": ""},
    {"input": [["single"]], "output": "single"}
  ]
}
```

The `<think>` block contains the reasoning process. The structured output is the
spec the SI model will execute. The model learns both parts through SFT exposure
and GRPO reinforcement when the resulting code passes tests.

---

## The Verification Loop Changes Everything

The critical property of the SI pipeline: **verification closes the loop**.

Every structured spec can be verified by running the generated code against the
specification's test cases. If the code doesn't compile or doesn't pass tests, the
spec (or the SI model's interpretation) was wrong. This is feedback you can act on.

Compare to natural language code generation: if GPT-4o's code is wrong, you have
to debug it yourself, determine what was wrong, rephrase the prompt, and try again.
The model has no signal about whether its output was correct.

With verification:
1. Reasoning model generates spec
2. SI model generates code
3. Verification runs
4. If tests pass: done
5. If tests fail: feed failure back to reasoning model
6. Reasoning model refines spec (or decomposes differently)
7. Loop until convergence

This is the same insight that drove DeepSeek-R1 and the GRPO training in this
project (Post 8): **learning requires a feedback signal**. For math, the signal
is a checkable answer. For code, the signal is compilation and tests.

The small reasoning model trained with GRPO on verifiable domains (math, code)
is already learning to generate outputs that can be verified. Structured specification
generation is a natural extension — the spec is the reasoning model's output, and
verification of the downstream code is the reward signal.

> **Sidebar: Why Verifiable Domains Matter for RL Training**
>
> GRPO (Group Relative Policy Optimization) uses reward signals to improve the
> model's policy. If the reward is noisy — "is this a good response?" evaluated
> by a human or another model — the training signal is noisy, and the learned
> behavior is unpredictable.
>
> Verifiable domains solve this: for math, a symbolic solver can check the answer.
> For code, the compiler and test suite check correctness. The reward is binary
> (correct/incorrect) and deterministic.
>
> This is why the SI pipeline is trainable with RL: the verification loop provides
> a clean reward signal. A reasoning model that generates structured specs, feeds
> them to an SI model, and receives test pass/fail feedback has exactly the kind of
> verifiable reward that GRPO can learn from.

---

## The Hardware Story Comes Full Circle

Recall the hardware fleet from Post 3:

```
ceres (RTX 5090, 32GB) — training and fast inference
vesta (RTX 4070 Ti SUPER, 16GB) — development and mid-range inference
janus (2× Titan RTX NVLink, 48GB) — multi-GPU, 1B model exploration
castor + pollux (2× DGX Sparks, 256GB unified) — large batch eval, long context
4× Jetson Orin NX (16GB unified, NVDLA) — edge deployment
```

The SI pipeline maps onto this fleet naturally:

| Role | Hardware | Why |
|------|----------|-----|
| SI model serving (7B inference) | castor or pollux | 128GB unified, BF16, no PCIe overhead |
| Reasoning model serving (500M inference) | Orin NX | 16GB, Q4_K_M, NVDLA, 10–25W |
| Reasoning model training/fine-tuning | ceres | 32GB GDDR7, 1792 GB/s bandwidth |
| SI model training (LoRA, 7B) | ceres or janus | 32–48GB VRAM |
| Verification (Go build + tests) | Any CPU node | Compute-cheap, parallelizable |

The 500M reasoning model at Q4_K_M fits comfortably in the Orin NX's 16GB
unified memory. It runs at ~30–50 tok/s with NVDLA handling the INT8 matrix
operations — fast enough for interactive specification generation with
sub-second response times. Four Orin NX units running in parallel handle
four concurrent SI pipeline requests.

The 7B SI model (si-go-v1) requires more memory but runs on castor or pollux
without quantization — full BF16 in 14GB of their 128GB unified memory pool.
Response time is ~15 seconds per generation, which is acceptable for code
generation tasks (compared to the minutes a human would spend writing the
same code).

This is not a cloud-dependent system. It runs on hardware under a desk.

---

## From Here: What Gets Built Next

The immediate next steps after the SFT and GRPO phases complete:

**1. Structured spec generation training (SFT extension)**

Add structured specification generation examples to the SFT fine-tuning data.
The model should learn the SI spec format as a native output format, not just
as a post-hoc skill. Concretely: for a set of Go programming tasks, generate
training pairs of (task description, `<think>` block, structured spec).

**2. SI-specific GRPO**

Run GRPO with a reward function that executes the full pipeline:
- Reasoning model generates spec
- si-go-v1 translates spec to code
- Go compiler + tests verify the code
- Pass/fail determines reward

This directly optimizes the reasoning model for the task that matters: generating
specs that produce correct code.

**3. Distill to smaller reasoning model**

If a 1B reasoning model can drive the SI pipeline at 95%+ accuracy, a 500M
model fine-tuned with GRPO may achieve 90%+. At edge deployment targets, the
difference between 500M Q4 and 1B Q4 is ~8 tok/s vs ~15 tok/s on an Orin NX.
Both are usable; the 500M model wins on power.

**4. Extend beyond Go**

The structured spec format is language-agnostic. The reasoning model generates
specs; an si-rust model or si-python model translates them. The same 500M
reasoning model drives all of them. The specialization lives in the SI layer,
not the reasoning layer.

---

## What This Is, Really

Zoom out.

The word "alignment" in ML discourse means one specific thing: getting models
to do what humans want. That's a real problem. But alignment shows up at every
level of this project, and it means the same thing at all of them: removing
friction between your system and its constraints.

Hardware dimensions that are multiples of 128 — aligned to tensor core tiles,
quantization blocks, NVDLA processing chunks. Training format that matches
inference format — the model learns exactly what it will be asked to do, no
capacity wasted on the wrong distribution. Reward signals that match actual
objectives — verifiable tests, not human opinion. SI specs that close the gap
between human intent and model output.

The si-go-v1 result (98% vs 0%) is the starkest demonstration: a 7B model
aligned to its task completely dominates frontier models of any size misaligned
to it. Not because the SI model is smarter. Because it isn't fighting a gap
between what it learned and what it's asked to do.

This is what the whole stack is about. The 500M reasoning model we're training
is small enough to run on a Jetson Orin NX at 10–25W. It's fast enough for
interactive use. It costs nothing per inference after training. And with the
right training — format-aligned SFT, verifiable-reward GRPO — it's capable
enough to drive production pipelines that currently require frontier API calls.

Not because small is always better. Because a well-aligned small model serving
a specific purpose beats a misaligned large model serving everything.

That's the place small models have.

---

*The structured-intent project lives at [github.com/scttfrdmn/structured-instruct](https://github.com/scttfrdmn/structured-instruct).*

*Next: The full pipeline running end-to-end — reasoning model driving SI model,
verification loop closing, structured intent in production.*
