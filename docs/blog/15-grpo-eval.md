# What GRPO Actually Learned (And What It Didn't)

*Part 15 of a series on building a small reasoning language model end-to-end.*

---

GRPO training completed. The reward curve looked encouraging: starting at 0.125
(1 out of 8 completions correct), climbing to 0.625 (5 out of 8) over 5,000 steps.
Dynamic sampling skipped 14,837 uniform-reward groups — groups where all eight
completions got the same score, producing zero gradient. The entropy held steady.
The KL divergence drifted negative slowly, as expected. By every metric visible
during training, it was working.

Then we ran the evaluation.

---

## The Standard Benchmarks Said Nothing

We ran the lm-eval-harness standard suite — ARC Challenge, GSM8K, HellaSwag, MMLU —
on both the SFT checkpoint (before GRPO) and the GRPO checkpoint (after).

| Benchmark | SFT | GRPO | Delta |
|-----------|-----|------|-------|
| ARC Challenge | 21.3% | 20.8% | -0.5% |
| ARC (norm) | 25.4% | 28.1% | +2.7% |
| GSM8K (exact match) | 0.0% | 0.0% | — |
| HellaSwag | 26.1% | 27.7% | +1.6% |
| MMLU | 24.5% | 23.0% | -1.5% |

All near random chance. Both checkpoints. GRPO didn't make things worse, which is
a genuine result — catastrophic forgetting is the default failure mode for RL on a
small model, and it didn't happen. ARC-norm and HellaSwag even improved slightly,
which means GRPO on math problems transferred a small amount of general reasoning
ability. But the honest summary is: these benchmarks told us nothing useful about
whether GRPO worked.

GSM8K showed 0.0% for both checkpoints, which was alarming until we understood why.
The lm-eval-harness expects the model to produce answers in a specific format that
its extraction regex can parse. Our model's generation format — `<think>` block
followed by a `\boxed{}` or plain answer after `</think>` — doesn't match. The
harness sees coherent math reasoning and extracts nothing. This is a measurement
problem, not a capability problem.

So we built our own eval.

---

## The Domain-Specific Eval

We wrote `eval/math_eval.py`, which reuses exactly the same generation pipeline and
reward functions that GRPO training uses. Same tokenizer, same prompt format, same
`build_prompt()`, same `generate_completions()`, same `_extract_final_answer()` and
`reward_math_exact()`. If GRPO training measured a reward of 0.625, this eval should
be able to see it.

We ran it on the same problems GRPO trained on (849 filtered problems, sampled 100).
This isn't cheating: GRPO never shows the model the ground truth answer. It generates
completions, scores them, and adjusts the policy. The model has "seen" these problems
in the sense that it generated solutions to them, but it hasn't memorized answers.
Running eval on training problems tests whether the reward signal during training
reflected a real capability improvement.

We also ran the SFT checkpoint on the same 100 problems as a baseline.

| Metric | SFT | GRPO |
|--------|-----|------|
| pass@1 | 5.0% | 4.0% |
| pass@8 | 25.0% | 26.0% |
| mean reward | 0.044 | 0.043 |

Effectively identical. The 0.125 → 0.625 reward progression during training did not
translate into a measurable improvement in held-out evaluation, even on the training
problems themselves.

This was the important result. Not the numbers — they're small either way — but
what they force us to confront about what GRPO actually did for 5,000 steps.

---

## Six Things the Data Actually Shows

We dug into the completions. Not the aggregate numbers — the actual text the model
produced, problem by problem, completion by completion. Here's what we found.

### 1. The model's knowledge is stochastic, not consolidated

Of 26 problems solved by the GRPO model (at least 1/8 correct), 21 had exactly
1 out of 8 correct. Three had 2/8, one had 3/8, one had 4/8. The overall pass@1
to pass@8 ratio ranged from 4x (GSM8K) to 8.5x (numina_tir).

This means the model hasn't learned a *procedure*. It's learned a *distribution*
where the correct reasoning path has low but non-zero probability. Getting the right
answer is closer to sampling a rare event than executing a known algorithm.

GRPO is supposed to push that probability higher — that's the whole point of the
policy gradient. It did push it, during training. But the push wasn't large enough
to cross the threshold where the correct path becomes the dominant one. The model
got slightly luckier slightly more often, but the fundamental randomness didn't
resolve.

### 2. The reasoning structure is correct; the arithmetic is wrong

Here's the model solving the jellybean problem ("Steve has 84 jellybeans, Matt has
10 times as many, Matilda has half of Matt's"):

```
[0] Extracted: 92    "Matt has 84 jellybeans... 84/2 = 40... 40 + 52 = 92"
[2] Extracted: 840   "Matt has 10 × 84 = 840... Matilda has 840/2 = ..."  (forgot to finish)
[5] Extracted: 96    "Matt has 84 jellybeans... 84 × 2 = 96"
[6] Extracted: 840   "Matt has 10 × 84 = 840..." (stopped before dividing)
```

Six of eight completions correctly identify the operations: multiply by 10, divide
by 2. They set up the right chain. Then they execute the arithmetic incorrectly, or
they stop before completing it. The model knows *what to do* but can't reliably
*do it*. It confuses which number is Steve's vs Matt's. It multiplies when it should
divide. It gets single-step calculations wrong (84/2 = 40).

This is what 500M parameters looks like for arithmetic: the model stores math facts
in its weights, and there aren't enough weights to store them reliably. A larger
model has more capacity to memorize multiplication tables and chain operations.
There's no architectural trick that fixes this — it's a raw capacity constraint.

### 3. Wrong completions are structurally indistinguishable from correct ones

For the problem "find x where (x-3)/4x = 0":

**Correct completion:**
```
The original expression can be rewritten as (x-3)/4x = 0.
So x-3 = 4x. Divide both sides by 4: x = 3.
Thus, the value of x is \boxed{3}.
```

**Wrong completion:**
```
The denominator 4x is 10, so the numerator x-3 can be replaced by 4x,
which gives x-3 = 10. The expression becomes (x-3)/4x = 10.
To find x: (x-3)/4x = 25. Thus x = 25.
```

Both have the same shape: restate the problem, set up an equation, manipulate
algebraically, box the answer. The wrong one invents a premise ("4x is 10") that
sounds plausible but is mathematically meaningless. It then manipulates that
invented premise with internally consistent (if wrong) algebra.

The model has learned the *syntax of mathematical reasoning* without fully grasping
the *semantics*. It knows what a proof looks like. It doesn't always know what
makes one valid.

This is the deepest finding. It's not about size or training time — it's about
what RL on binary rewards can and cannot teach. GRPO rewards the final answer.
It cannot reward the validity of individual reasoning steps. A completion that
arrives at the right answer through wrong reasoning gets the same reward as one
that arrives through correct reasoning. The model has no incentive to distinguish
valid from invalid intermediate steps, only to produce more text that ends in
the right number.

### 4. GRPO traded one domain for another

| Source | SFT pass@8 | GRPO pass@8 |
|--------|-----------|-------------|
| numina_tir (competition math) | 34.9% | 39.5% |
| MATH (Hendrycks) | 15.0% | 25.0% |
| GSM8K (grade school) | 18.9% | 10.8% |

GRPO improved on competition math and degraded on grade school math. This is
domain specialization, not general improvement. The training set was a mix of
all three sources, but the reward signal pushed the model toward patterns that
work for competition-style problems (`\boxed{}` format, symbolic manipulation)
at the expense of word-problem patterns (extracting quantities from narrative text,
multi-step arithmetic).

This wasn't a surprise in retrospect. The filtered training set skewed toward
competition problems — they're more likely to fall in the "solvable but not trivial"
sweet spot that the difficulty filter selects for. GSM8K problems tend to be
either too easy (model gets 8/8) or too hard (model gets 0/8), so fewer survive
the filter. GRPO trained more on competition math simply because that's what the
filter kept.

### 5. The format training worked completely

During GRPO training, essentially 100% of completions had proper `</think>` tags.
The model reliably produces `<think>reasoning</think>answer`. It learned the
container perfectly. The content is the problem.

This makes sense: format compliance is a much simpler optimization target than
mathematical correctness. There's one right format and the model sees it on every
single training example. The format reward is small (0.1 weight) but consistent,
and the policy picks it up immediately.

### 6. The training reward curve was real but misleading

The 0.125 → 0.625 reward during training was real. The model did solve more problems
per batch as training progressed. But this happened because of dynamic sampling:
as the model improved on easy problems, those problems started producing uniform
reward groups (all 8 correct) and got skipped. The remaining problems were the
ones still in the difficulty sweet spot, which self-selects for problems the model
is currently learning. The reward curve shows progress on the *margin* — the
problems the model is currently working on — not on the full distribution.

When we evaluated on a fixed set of problems (no dynamic sampling, no filtering),
the aggregate number was ~4% pass@1 — close to the SFT baseline. The training
curve showed real learning happening at the frontier; the eval showed that frontier
was narrow.

---

## What This Means for the Project

The honest conclusion is that GRPO at 500M parameters, with our pre-training budget,
produces a model that has learned the shape of mathematical reasoning but cannot
reliably execute it. The capability is there in a distributional sense — sample
enough times and you'll see correct solutions — but it hasn't consolidated into
reliable behavior.

This isn't a failure of GRPO. GRPO did exactly what it's designed to do: it took
the policy and moved probability mass toward completions that earn higher reward.
The problem is that at 500M parameters, "moving probability mass" means going from
"almost never correct" to "occasionally correct." The same algorithm on a 7B or 70B
model moves from "sometimes correct" to "usually correct" because the larger model
has enough capacity to represent the full reasoning procedure, not just its
statistical shadow.

Three concrete paths forward:

**More pre-training data.** Our 500M model was trained on a limited budget. The
arithmetic errors — 84/2 = 40, 10 × 84 = 96 — suggest the model hasn't memorized
basic math facts. More pre-training on math-heavy corpora (textbooks, worked
solutions) would give GRPO a better foundation to optimize over.

**Process reward models.** Our reward signal is binary: right answer or wrong answer.
The model that arrives at the right answer through invalid reasoning gets the same
reward as the model that reasons correctly. A process reward model that scores
individual reasoning steps would give GRPO the signal it needs to distinguish
valid from invalid intermediate reasoning, not just valid from invalid final answers.

**Larger model.** The 500M architecture is designed as a proving ground for the
training recipe. The same GRPO implementation, applied to a 1B or 3B model with
adequate pre-training, should cross the threshold from "occasionally correct" to
"reliably correct." The infrastructure is proven; the scale is the variable.

---

## The Eval Script

`eval/math_eval.py` is now part of the repo. It uses the same generation and reward
pipeline as GRPO training, which means it measures exactly what GRPO optimized for.
This turned out to be important: the standard lm-eval-harness showed 0% on GSM8K
for both checkpoints because of format mismatch, which would have led us to conclude
GRPO did nothing when it actually did something small but real.

The lesson: always evaluate with the same pipeline you train with before reaching
for external benchmarks. External benchmarks measure a different thing — whether
your model conforms to their expected format *and* gets the answer right. Your
internal eval measures whether the capability you trained for actually exists.

Both measurements matter. But if you only have external benchmarks, you'll confuse
"can't solve math" with "solves math but doesn't format the answer the way the
benchmark expects."

---

*Next: scaling the recipe — what changes when we move from 500M to 1B parameters,
and what stays the same.*
