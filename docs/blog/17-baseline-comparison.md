# What Comparing to Baselines Told Us

*Part 17 of a series on building a small reasoning language model end-to-end.*

---

In the previous post I wrote that our 500M GRPO model got 4% pass@1 on math
and 97% fields pass@8 on structured intent. Those numbers mean nothing in
isolation. Is 4% good or bad for a 500M model trained on 10B tokens? Is
97% pass@8 a real capability or just what any model of that size can do?

To answer, I ran the same evaluation pipelines against four peer small
models from the HuggingFace ecosystem. This post is what came back, and
what it implies for the project's direction.

---

## The Setup

Four peer models, each with a distinct profile:

| Model | Size | Training focus |
|-------|------|----------------|
| Qwen/Qwen2.5-0.5B-Instruct | 500M | Well-mixed 18T-token general model |
| Qwen/Qwen2.5-Math-1.5B-Instruct | 1.5B | Math-specialized fine-tune |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | 1.1B | SlimPajama (3T web+books, no heavy math) |
| microsoft/phi-1_5 | 1.3B | Synthetic textbook-style training |

Same prompts. Same reward functions. Same 100 math problems and 35 SI test
cases that SRM was evaluated on. The only difference is the model being
generated from — peer models use their own chat templates instead of our
`<think>` format, and the format-bonus component of the reward is zeroed
out so peer models aren't penalized for not using our structure.

---

## The Results

### Math (100 problems from our GRPO training set)

| Model | Size | pass@1 | pass@1 voted | pass@8 | mean reward |
|-------|------|--------|--------------|--------|-------------|
| SRM SFT | 500M | 5.0% | 5.0% | 25.0% | 0.044 |
| **SRM GRPO** | **500M** | **4.0%** | **5.0%** | **29.0%** | **0.050** |
| SRM SFT+SI | 500M | 6.0% | 5.0% | 25.0% | 0.044 |
| Phi-1.5 | 1.3B | 1.0% | 3.0% | 16.0% | 0.024 |
| TinyLlama-1.1B | 1.1B | 2.0% | 3.0% | 10.0% | 0.016 |
| **Qwen2.5-0.5B** | **500M** | **34.0%** | **51.0%** | **60.0%** | **0.339** |
| Qwen2.5-Math-1.5B | 1.5B | **64.0%** | 72.0% | 76.0% | 0.674 |

### Structured Intent (35 test cases)

| Model | Size | JSON pass@1 | JSON pass@8 | Fields pass@1 | Fields pass@8 |
|-------|------|-------------|-------------|---------------|---------------|
| SRM GRPO (no SI SFT) | 500M | 14.3% | 62.9% | 0.0% | 5.7% |
| **SRM SFT+SI** | **500M** | **40.0%** | **97.1%** | **40.0%** | **97.1%** |
| Phi-1.5 | 1.3B | 31.4% | 100% | 20.0% | 91.4% |
| TinyLlama-1.1B | 1.1B | 60.0% | 100% | 48.6% | 97.1% |
| **Qwen2.5-0.5B** | **500M** | **82.9%** | **100%** | **82.9%** | **100%** |
| Qwen2.5-Math-1.5B | 1.5B | 0.0% | 28.6% | 0.0% | 0.0% |

---

## Five Findings, Each with a Direction Implication

### 1. Data composition dominates data volume

TinyLlama was trained on 3 *trillion* tokens. It got 2% pass@1 on math —
*worse than SRM at 10B tokens.* The reason is the data mix: SlimPajama is
mostly web text and books with no heavy math component. Phi-1.5, trained
on synthetic textbook-style tokens, got 1%. Qwen2.5-0.5B, trained on a
well-mixed 18T tokens that includes real math, got 34%.

**Order of effect:** mix > volume > architecture > model size at this
scale.

**What this means for the project:** Phase 3, the ongoing pre-training
rerun with 48% math-heavy data (openwebmath + numinamath), is directionally
correct. This is not speculation anymore — the baseline comparison directly
validates the hypothesis. The training recipe we built around the "math
in pre-training" idea is the right recipe; we just hadn't executed it
yet.

The less comfortable implication: **Phase 3 will likely not close the gap
to Qwen2.5-0.5B.** Qwen2.5 had 1,800× more training data. Better mix at
10B tokens can't match excellent mix at 18T tokens. A realistic Phase 3
outcome is 10-15% pass@1, not 30%+. Setting expectations honestly matters:
landing at 12% would be a real, defensible result and consistent with what
data-composition effects at this scale predict. Anything higher would be
a pleasant surprise.

### 2. Qwen2.5-0.5B is the honest 500M ceiling

There is no way to pretend around this: another 500M parameter model,
trained by a serious team with serious data, gets 34% pass@1 where our
model gets 4%. That's the ceiling for the parameter count. Our model is
below it.

**What this means for the project:** any claim about SRM being competitive
at 500M needs to engage with the Qwen2.5 result, not ignore it. The
strategic endpoint options from issue #15 look different in light of this:

- **Research publication** — significantly harder than it looked. Before
  the baselines, "we built a 500M reasoning model from scratch" was a
  self-contained story. Now the story has to include "and here is where
  it sits against Qwen2.5-0.5B, and why, and what that implies about our
  training recipe." That's a much sharper and more critical paper to
  write, and it requires Phase 3 + Phase 4 results we don't have yet.
- **Production SI system** — still viable, and the easiest to validate
  independently (does the pipeline work end-to-end?).
- **Educational resource** — the strongest of the four, especially with
  this post and the honest comparison added. Very few blog series show
  readers "here's how to build a small reasoning model, here's what it
  looks like next to a production peer model, here's why the gap exists
  and what would close it."
- **Capability scaling** — meaningful only if Phase 3 shows improvement.
  1B at 10B tokens still probably loses to Qwen2.5-0.5B at 18T tokens.

### 3. The SI result is interesting but less unique than I thought

Our 500M model after SI-specific SFT gets 40% fields pass@1 and 97% pass@8.
I was treating that as a strong capability demonstration.

It's real — but not unique. TinyLlama-1.1B gets 49% fields pass@1 with
*zero SI training.* Qwen2.5-0.5B gets 83% the same way. Phi-1.5 gets 20%.
Multiple models of similar size can produce structurally valid JSON specs
from natural language prompts, some without any task-specific tuning.

**What this means for the project:** The SI pipeline is deployable — any
of these models could drive it. The blog-post-11 thesis that "format
alignment beats model size" is directionally right (Qwen2.5-Math-1.5B
destroys itself for math and scores 0% on SI), but the related claim that
*our specific training recipe* uniquely enables SI is overstated. If the
goal is a working SI deployment, we can pick a model and ship. If the
goal is an SRM that's meaningfully better at SI than peer small models,
we haven't demonstrated that yet.

The pass@8 number (97%) is, however, essentially at ceiling for all the
competent models. That says something useful: in production, with a
verification loop that filters the 3% of outputs that are structurally
broken, any of these models produces usable specs. The SI deployment
path does not hinge on which specific small model we pick.

### 4. Specialization destroys generality

Qwen2.5-Math-1.5B is a math expert: 64% pass@1, 72% voted, 76% pass@8.
That's the highest math score in the entire comparison. It achieves this
through aggressive math-specific fine-tuning on top of Qwen2.5-1.5B base.

And it cannot produce a single valid SI spec in 35 tries at pass@1. Zero.
Its pass@8 fields score is also zero — the 28% JSON pass@8 is just the
model occasionally emitting brace-delimited text that happens to parse,
not actual SI specs.

**What this means for the project:** if we ever consider pushing SRM
harder toward math (more GRPO steps, more math SFT, larger math reward
signal), we should expect the SI capability to degrade. The current SRM
sits at 4% math / 97% pass@8 SI. Qwen2.5-Math-1.5B sits at 64% math / 0%
pass@8 SI. There's a Pareto frontier between general capability and
math specialization, and moving along it has a real cost.

The right architectural call is to keep the mixed training recipe and
accept that math will be modest. A 1B scale run on better pre-training
data is more likely to close the math gap without sacrificing SI than
another round of math-focused GRPO on 500M would.

### 5. Voting helps in proportion to underlying signal

Majority voting across 8 completions improves pass@1 for every model, but
the size of the improvement scales with how much correct signal is in the
distribution:

| Model | Math pass@1 | Math pass@1 voted | Voting gain |
|-------|-------------|-------------------|-------------|
| SRM GRPO | 4.0% | 5.0% | +1.0 |
| Qwen2.5-0.5B | 34.0% | 51.0% | **+17.0** |
| Qwen2.5-Math-1.5B | 64.0% | 72.0% | +8.0 |
| TinyLlama | 2.0% | 3.0% | +1.0 |
| Phi-1.5 | 1.0% | 3.0% | +2.0 |

When the model "knows" the answer but samples inconsistently (Qwen2.5-0.5B),
voting closes a huge fraction of the gap between pass@1 and pass@8. When
the model barely knows the answer at all (SRM, TinyLlama, Phi-1.5), voting
has nothing to aggregate and mostly just reproduces noise.

**What this means for the project:** voting isn't broken on SRM — it's
working as designed but operating in the regime where it's least useful.
Post-Phase 3, if math pass@8 rises above ~50%, voting will start to deliver
the 15-20% improvement we see with Qwen2.5-0.5B. Before that, voting is
not the leverage point. Data quality is.

---

## Revised Direction

Before the baselines, the project was running four endpoints in parallel
at reduced intensity: research, production, educational, capability. The
data changes the shape of each:

**Research:** significantly harder. A paper now has to either show that
Phase 3 closes the Qwen2.5 gap meaningfully (unlikely at 10B tokens), or
reframe around something the recipe uniquely demonstrates. Possible angles:
"what a well-architected 500M model can do on a tight data budget," or
"the SI training recipe as a path to edge-deployable reasoning." Both
require Phase 3 results in hand.

**Production:** unchanged. The SI pipeline is viable. Any of the peer
small models would work; picking SRM specifically is a choice about
end-to-end control rather than capability advantage. If the goal is "ship
an SI pipeline running on edge hardware with a verification loop," that
can happen with current capabilities, and the blog post 11 architecture
is sound.

**Educational:** strengthened. The blog series is more valuable now than
it was before. Honest comparison is rare in public technical writing. A
series that walks a reader through building a 500M model from scratch,
then admits that a peer 500M model trained with 1,800× more data beats it
by 8x on math, and explains exactly why — that's a story worth telling.
This post is part of it.

**Capability:** now properly gated. Pushing to 1B only makes sense if
Phase 3 shows the recipe transfers to more data. Without that, scaling
to 1B at 10B tokens is unlikely to produce something competitive. The
decision point is after Phase 3 finishes, with a specific number to hit:
does the 500M pass@1 on math rise from 4% to something notably higher?
If yes, Phase 4 is justified. If no, the bottleneck is deeper than scale.

---

## What Doesn't Change

A few things the baselines do not affect:

**The training recipe's correctness.** SFT, GRPO, the DAPO and Dr. GRPO
improvements, the KV cache compression, the format-alignment SFT examples
— these are all implemented correctly and working as designed. The
comparison revealed limits of the input data budget, not errors in the
training algorithm.

**The value of honest evaluation.** Running `math_eval.py` against peer
models took a few hours on two DGX Sparks and told us more than another
week of GRPO hyperparameter tuning would have. Before this, we were
optimizing blind. Now we know the terrain.

**The SI ceiling.** 97-100% pass@8 across multiple competent small models
means the SI pipeline is deployable with any of them. The verification
loop makes the 3% structural failures recoverable. SRM's 97% is enough.

---

## The Next Concrete Decisions

1. **Wait for Phase 3 to finish** (~3 more days). Rerun `math_eval.py` on
   the new 500M checkpoint. Compare to both the original SRM GRPO numbers
   and the Qwen2.5-0.5B baseline. The delta between old-SRM and new-SRM
   is the data-composition effect. The delta between new-SRM and Qwen2.5
   is the data-volume effect plus whatever else we're missing.

2. **Decide the strategic endpoint** (issue #15) after Phase 3 results
   land. The numbers we've now seen make this a sharper decision, not a
   vaguer one.

3. **Don't launch Phase 4 (1B scale) without justification from Phase 3.**
   If Phase 3 at 500M shows meaningful math improvement, 1B is worth the
   compute. If it doesn't, 1B is unlikely to help.

4. **Build the SI verification loop anyway** (issue #12). It's useful
   regardless of which endpoint we commit to, and it closes the
   longest-standing gap between what blog post 11 describes and what the
   code actually does.

---

## The Meta-Observation

Before running the baselines, the project's direction felt ambiguous but
manageable. After running them, the decisions are harder but more
honest. That's the normal pattern when you add real measurement to an
ambitious effort: the landscape gets clearer and some options get
smaller. The gain isn't confidence — it's calibration.

The project continues. Phase 3 will resolve the biggest open hypothesis
in a few days. The baselines will stay as the reference for every
performance claim from here on. And the blog series will be more
interesting for it.

---

*Raw baseline numbers and per-problem breakdowns are in
[`docs/baseline-comparison.md`](../baseline-comparison.md) and the
`results/baseline_*.json` files. Code for reproducing the runs is in
`eval/baseline_eval.py`.*
