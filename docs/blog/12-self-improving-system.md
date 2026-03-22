# The Self-Improving System: Alignment, Verification, and the Architecture That Builds Itself

*Part 12 of a series on building a small reasoning language model end-to-end.*

---

This post documents a design conversation that emerged from building the system
described in the previous eleven posts. It's less about implementation and more
about what the architecture points toward once the components are running. The
practical work — SFT, GRPO, GGUF conversion, inference deployment — is covered
in earlier posts. This is about the shape of the thing we're building toward.

---

## The Three Projects Are One System

Three separate repositories. One architecture.

**endless-v2** is a semantic state engine. It parses raw human input into structured
claims across six dimensions — Intent, Constraint, Assumption, Causal, Temporal,
Fact — and runs a council of six evaluators (Logic, Gap, Probability, Temporal,
Persona, Conflict) concurrently on every incoming claim. Its key output is not
a summary or a retrieval result. It's an `EnrichedPrompt`: a precise snapshot of
what is known, what is uncertain, and what is missing. Gaps are first-class
signals, classified as Blocking, Latent, or Irrelevant. The system doesn't just
track what you said — it tracks what you haven't said yet that it needs.

**structured-instruct** is an execution engine. Given a clean structured
specification, the SI model (si-go-v1, 7B Qwen2.5-Coder fine-tuned) produces
correct, verified code at 98% pass rate. Frontier models of any size fail
completely at the same task (0% compilation) because they weren't trained on
the spec format. Format alignment dominates model size.

**small-reasoning-model** (this project) is the missing piece. A small reasoning
model — 500M parameters, reasoning-trained via SFT and GRPO — that sits inside
the endless environment and does what neither the council evaluators nor the SI
model can do: flexible, contextual reasoning about what the human probably means,
how to decompose that intent into pieces the SI model can execute, and how to
ask precisely the right question to resolve a blocking gap.

The architecture isn't a pipeline where each project hands off to the next.
The reasoning model lives *inside* endless, not after it:

```
┌──────────────────────────────────────────────────────────┐
│                        endless                           │
│                                                          │
│  Claim store: Intent · Constraint · Assumption ·         │
│               Causal · Temporal · Fact                   │
│                                                          │
│  Council: Logic · Gap · Probability ·                    │
│           Temporal · Persona · Conflict                  │
│                                                          │
│  ┌──────────────────────────────────────────────────┐    │
│  │           Small Reasoning Model                  │    │
│  │                                                  │    │
│  │  receives: EnrichedPrompt                        │    │
│  │           (known state + blocking gaps +         │    │
│  │            tensions + persona context)           │    │
│  │                                                  │    │
│  │  emits: → questions  → Considered claims         │    │
│  │         → specs      → trigger SI execution      │    │
│  │         → observations → new claims              │    │
│  └──────────────────────────────────────────────────┘    │
│                                                          │
│  Blocking gaps → drive clarifying questions              │
│  Resolved state → trigger spec generation                │
└───────────────────────────┬──────────────────────────────┘
                            │ spec (gaps resolved)
                            ▼
                    SI Model (? → code)
                            │
                            ▼
                    Verification (oracle)
                            │
              ┌─────────────┴─────────────┐
              │                           │
            pass                        fail
              │                           │
              └─────────────┬─────────────┘
                            ▼
              back into endless as new claims
              (Fact: compiled · Constraint: nil
               input failed · Intent: refined)
```

The reasoning model's outputs aren't final. They become claims, subject to the
same council evaluation as everything else. A question it generates becomes a
Considered claim. A spec it generates might conflict with a Constraint already in
the store — the Logic evaluator catches that before the SI model ever sees it.
Verification results return as Fact claims. A test failure on nil input was a
latent Constraint gap; it's now Blocking. The system's understanding of what it
doesn't know updates from evidence.

---

## The Target Is "?"

The SI model currently targets Go. Go is a *validation choice*, not a design
constraint. What the architecture requires:

1. A domain where intent can be formalized
2. An output where verification is clean
3. Non-triviality — the gap between intent and output is real

Go satisfies all three. So does SQL, WASM, Terraform, OpenAPI, mathematical
proofs, shell scripts, LLVM IR. The VISION.md roadmap makes this explicit:
Go → SSA → WASM → LLVM IR → machine code. Each step reduces the vocabulary
and sharpens the oracle.

At machine code, the vocabulary is ~200 ARM64 instructions and the oracle is
the CPU itself — the hardest, cleanest verifier that exists. The human-readability
of Go source is scaffolding for a capability gap that closes as the system improves.

The deeper point: programming languages exist because humans need to reason
about computation in representations that fit their cognition. They're an
interface for humans, not for machines. If the system can verify output without
a human reading it, the intermediate representation becomes an optimization
target rather than a design constraint. The format that minimizes error rate
under verification pressure is the right format, regardless of whether it's
readable.

The structured spec format (currently JSON/YAML, human-designed) is the same
kind of scaffolding. It exists because the reasoning model and SI model need
an explicit handoff point that can be inspected and debugged. As both ends
improve, the explicit format can internalize — first into a `<think>` block
that nobody reads, then into implicit intermediate representations inside a
single model.

> **Sidebar: The Jetson Orin NX is already a hardware SI model**
>
> The NVDLA (Neural Deep Learning Accelerator) on the Jetson Orin NX is a
> fixed-function INT8 matrix multiply accelerator. It accepts a specific class
> of operations and executes them efficiently, with no generality beyond that
> class. This is architecturally identical to the SI model: takes a
> well-specified input (the operation), produces a verified output (the
> computation), handles nothing outside its spec. The hardware instantiates
> the same design principle. The fleet that runs the software system embodies
> the architecture of the software system.

---

## The Alignment Principle at Every Level

This has been the unifying thread throughout the series, but it's worth stating
directly here.

"Alignment" in ML discourse refers to one specific problem: making models do
what humans want. That's real. But alignment appears at every level of this
stack, and it means the same thing at all of them: **removing friction between
your system and its constraints**. Efficiency and reliability emerge from closing
those gaps.

**Hardware alignment** — Dimensions that are multiples of 128 map cleanly onto
tensor core tiles, GGUF quantization blocks, NVDLA INT8 chunks. Misalignment
doesn't fail; it wastes. The hardware is there; you're either using it fully
or fighting it quietly.

**Format alignment** — A model trained on exactly the format it will see at
inference has no capacity wasted on the wrong distribution. si-go-v1's 98%
result vs. 0% for frontier models is entirely explained by this. The 7B model
isn't smarter. It's not fighting a mismatch.

**Training signal alignment** — GRPO uses verifiable rewards because noisy
reward signals produce noisy gradients. The reward must be aligned with the
actual objective. For code, the compiler and tests are the oracle. The signal
points exactly where improvement is needed.

**Intent alignment** — The whole purpose of endless is to close the gap between
what the human expressed and what the system understood. Gaps are structural
signals, not failures. The system surfaces them explicitly rather than silently
approximating.

At each level the principle is the same. The exercise of building this system
from hardware up — choosing dimensions, designing training, building the semantic
layer, connecting components — is an exercise in identifying these gaps and
closing them deliberately. The blog series is the documentation of that process.

---

## The Self-Improvement Question

Once the system is running — endless accumulating claim state, the reasoning
model participating in the loop, SI models executing specs, verification feeding
results back as claims — something becomes possible that isn't possible before:
the system can improve its own components.

**What the verification oracle enables:**

Every session produces signal. Which gaps were blocking and got resolved. Which
specs produced passing code. Which clarifying questions closed a gap vs. reopened
it. Which failure modes repeat. The signal is structured — not raw conversation
history but classified claims with truth values, domain projections, and revision
histories.

That's training data. Continuously generated, labeled by the oracle, requiring
no human annotation.

**What can improve:**

The spec format is human-designed. Under optimization pressure — generate specs
in variant A and variant B, measure which produces higher SI model pass rates —
the format evolves. The system A/B tests its own intermediate representation and
drifts toward whatever minimizes error rate. This is the "learned spec format"
question from the Phase 1 plan, but driven by the system in operation rather
than a controlled experiment.

The SI model accumulates verified (spec → code) pairs from operation. Periodic
retraining on that accumulated data improves it on the actual distribution of
specs the reasoning model generates, not the synthetic training distribution.
The two models co-evolve.

The reasoning model can be fine-tuned on accumulated (EnrichedPrompt → clarifying
question → gap resolved) and (EnrichedPrompt → spec → verified code) triples.
The GRPO structure applies directly: the oracle at the end of the pipeline is
the reward signal. The reasoning model gets better at the specific patterns
that appear in real usage.

**What merges:**

The explicit spec format is scaffolding for the current capability gap between
reasoning and execution. As both ends improve, the handoff can internalize.
The intermediate representation stops needing to be inspectable. The reasoning
model and SI model converge toward a single model that reasons about the domain
and generates verified output directly, with the `<think>` block as the only
remaining trace of the intermediate.

**What splinters:**

The SI layer specializes before it generalizes. Different capability regimes —
stateless utility functions, concurrent systems, HTTP handlers, multi-file
packages — may be better served by separate specialist models than by one model
trying to handle all of them. The routing logic (which specialist to invoke)
becomes part of the reasoning model's job.

**What goes away:**

The endless council evaluators compensate for the reasoning model's current
limitations. The Gap evaluator exists because the reasoning model can't reliably
notice what's missing from a spec. As the reasoning model improves inside the
endless environment — trained on signal from which gaps were real, which
conflicts mattered — it internalizes what the evaluators are doing. The council
remains as a check on the model's blind spots, not as a primary mechanism.

**What refactors:**

The boundary between the semantic state layer and the reasoning model. Currently
they're separate: endless maintains state, the reasoning model uses that state.
As the reasoning model learns the domain of claim management itself — what gap
patterns to expect, when a Constraint will conflict with an Intent, how to
surface a Temporal tension — it starts doing structural reasoning that endless
currently does symbolically. The boundary dissolves from both directions.

**The ceiling:**

The system can only improve what it can verify. The oracle determines the ceiling.
For code, the oracle is clean. For "good reasoning about vague intent," the
oracle is the downstream verification — did the spec produce code that passed
tests? This is indirect but real. The system can improve everything in the chain
between human expression and verified output, because the chain is closed.

The one thing it can't improve autonomously is the quality of human intent at
the top. It can learn what *this human* means (Persona evaluator), ask better
questions, surface gaps more precisely. But the human defines what the system
is *for*. The improvement loop runs within that boundary.

---

## The Bootstrap

The small reasoning model being built in this series is the initial conditions
for the self-improvement loop.

The training recipe — pre-training on 9.86B tokens, SFT on structured
chain-of-thought, GRPO on verifiable domains — is designed to get the reasoning
model above the threshold where it can participate meaningfully in the endless
environment. Drive the loop. Generate specs that the SI model can execute.
Ask questions that resolve blocking gaps. Recognize when intent is underspecified.

Once it's above that threshold, the loop takes over. The training data the system
generates through operation is better than synthetic training data, because it's
drawn from the real distribution of intent and real failure modes. The model
improves on what it actually encounters, not on what a data pipeline approximated.

The hardware fleet is the substrate. endless is the environment. The small
reasoning model is the first agent capable of meaningful participation. The SI
models are the hands.

What gets built after that depends on what the system learns it needs.

---

*The structured-instruct project: [github.com/scttfrdmn/structured-instruct](https://github.com/scttfrdmn/structured-instruct)*

*The endless-v2 project: part of the same ecosystem.*

*All code for the reasoning model: [github.com/scttfrdmn/small-reasoning-model](https://github.com/scttfrdmn/small-reasoning-model)*
