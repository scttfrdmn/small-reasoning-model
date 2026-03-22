# System Architecture: The Three-Project Ecosystem

This document captures the architectural understanding of how endless-v2,
structured-instruct, and small-reasoning-model relate. It emerged from a design
conversation in March 2026 and should be updated as the system evolves.

---

## The Three Projects

| Project | Role | Current State |
|---------|------|---------------|
| **endless-v2** | Semantic state engine — parses intent into claims, detects gaps, enriches prompts | v0.7.0, running |
| **structured-instruct** | Execution engine — translates structured specs into verified code | si-go-v1 done (98%), Phase 1 in progress |
| **small-reasoning-model** | Reasoning component — lives inside endless, drives the loop | SFT in progress (step ~4K/134K) |

---

## The Architecture

The reasoning model sits **inside** the endless environment, not after it.

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
```

---

## The Alignment Principle

The unifying principle across all three projects and all levels of the stack:
**efficiency and reliability come from removing friction between your system
and its constraints**.

- **Hardware alignment** — dimensions × 128, clean tensor tiles, GGUF blocks, NVDLA chunks
- **Format alignment** — training distribution = inference distribution (explains si-go-v1 98% vs. 0%)
- **Training signal alignment** — verifiable rewards, oracle-based, no noise
- **Intent alignment** — endless surfaces gaps explicitly; the system knows what it doesn't know

---

## The Target Is "?"

The SI model currently targets Go. Go is a validation choice, not a design
constraint. The architecture requires:

1. A domain where intent can be formalized
2. An output where verification is clean
3. Non-triviality — the gap between intent and output is real

The roadmap (from structured-instruct VISION.md):
```
Go source → Go SSA → WASM → LLVM IR → machine code
```

Each step: smaller vocabulary, harder oracle, potentially lower error rate.
The representation that minimizes error rate under verification pressure
is the right representation, regardless of human readability.

The structured spec format (JSON/YAML) is the same kind of scaffolding.
It exists for the current capability gap. As the reasoning model and SI
model improve, the explicit handoff can internalize.

---

## Self-Improvement Dynamics

Once the system is running, the verification oracle enables continuous
improvement. Every session generates labeled training data — claims,
gap resolutions, spec outcomes, verification results — without human
annotation.

**What improves:**
- Spec format: A/B test variants, drift toward lower error rate
- SI model: retrain on accumulated verified (spec → code) pairs
- Reasoning model: fine-tune on (EnrichedPrompt → verified outcome) triples

**What merges:**
- Reasoning model + SI model → single model as capability gap closes
- Explicit spec format → implicit `<think>` intermediary → dissolved

**What splinters:**
- SI layer → specialist models per capability regime (concurrent, stateless, HTTP, etc.)

**What goes away:**
- Council evaluators that the reasoning model has internalized
- Spec format that no longer needs to be inspectable

**The ceiling:**
The system can only improve what the oracle can verify. The human defines
what the system is *for*; the system improves everything within that.

---

## The Bootstrap

The small-reasoning-model training recipe is the initial conditions.

Pre-training on 9.86B tokens establishes general language capability.
SFT on structured chain-of-thought teaches the reasoning format.
GRPO on verifiable domains (math, code) teaches the model to generate
outputs that survive oracle evaluation.

The goal: get the reasoning model above the threshold where it can
participate meaningfully in the endless loop. Once there, the loop
generates better training data than any synthetic pipeline, because
it's drawn from the real distribution of intent and failure modes.

**The small reasoning model is the bootstrap. The loop is the engine.**

---

## Hardware Mapping

| Role | Hardware | Why |
|------|----------|-----|
| Reasoning model serving | Jetson Orin NX × 4 | 16GB unified, Q4_K_M, NVDLA, 10–25W |
| SI model serving (7B BF16) | castor / pollux | 128GB unified, no PCIe overhead |
| Reasoning model training | ceres (RTX 5090) | 32GB GDDR7, 1792 GB/s |
| endless state | castor / pollux | large memory, persistent across sessions |
| Verification | any CPU node | compute-cheap, parallelizable |

The Jetson Orin NX NVDLA is itself a hardware SI model — fixed-function
executor for a specific op class, no generality beyond that. The fleet
embodies the architecture it runs.

---

*Last updated: 2026-03-21*
*Reflects conversation in small-reasoning-model session.*
