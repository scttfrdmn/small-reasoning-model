# Phase 2 GRPO: Reinforcement Learning With Verifiable Rewards

*Part 8 of a series on building a small reasoning language model end-to-end.*

---

SFT teaches the model to produce reasoning chains. GRPO teaches the model to
produce *correct* reasoning chains.

The distinction matters enormously. An SFT model mimics the format and style of
correct reasoning without any guarantee that the reasoning is actually right.
GRPO provides exactly the reward signal that distinguishes correct from incorrect,
at the level of the final answer, and lets the model learn to optimize for correctness.

This is the phase where the reasoning capability actually emerges.

---

## The Core Idea

GRPO (Group Relative Policy Optimization) was introduced as part of the DeepSeek-R1
training recipe. It's a variant of PPO adapted for the LLM setting, specifically
designed to avoid the need for a separate "critic" or "value" model.

The algorithm, stated simply:

```
For each training batch:
  1. Sample a math/logic problem p
  2. Generate 8 different completions {o₁, o₂, ..., o₈} from the current model
  3. Check each completion against the ground truth: reward rᵢ ∈ {0, 1}
  4. Compute the group advantage: Aᵢ = (rᵢ - mean({r})) / std({r})
  5. Update the model to increase probability of high-advantage completions
     and decrease probability of low-advantage completions
  6. Add a KL penalty to prevent the model from drifting too far from the SFT checkpoint
```

The key insight is step 4: the *group* of 8 completions provides its own
baseline. If 3 out of 8 completions are correct, the correct ones have positive
advantage (+) and the incorrect ones have negative advantage (-). The model
learns to do more of what the correct completions do.

This is fundamentally different from supervised learning (which just says
"produce this specific output") and from RLHF with a learned reward model (which
adds the complexity and instability of a second model).

---

## Why GRPO Over PPO?

Standard PPO uses a value function (critic) to estimate the "baseline" — the
expected reward from a given state. Training a good critic requires as much
effort as training the policy itself, and critics are notoriously unstable in
the LLM setting.

GRPO replaces the learned critic with a *statistical* baseline: the mean reward
of the current group of G completions. This baseline is:
- Free to compute (just average the rewards)
- Unbiased (it's the actual expected reward under the current policy for this prompt)
- Stable (it doesn't require a separate neural network)

The trade-off: you need to generate G completions per training step, which is
G× more inference compute. With G=8, training is ~8× slower in inference compute
than SFT. But the stability improvement is worth it at this scale.

> **Sidebar: What is PPO?**
>
> Proximal Policy Optimization (PPO, Schulman et al. 2017) is a policy gradient
> RL algorithm with a "proximal" constraint: it limits how much the policy can
> change in a single update step, using a clipped objective function.
>
> The clipping prevents large policy updates that might destabilize training.
> Without clipping, a single bad batch could push the model into a degenerate
> state it never recovers from.
>
> GRPO inherits PPO's clipped objective:
> `L = E[min(r * A, clip(r, 1-ε, 1+ε) * A)]`
>
> where `r = π_θ(o|p) / π_ref(o|p)` is the probability ratio between the
> current policy and a reference policy, `A` is the advantage, and `ε=0.2`
> is the clipping threshold.
>
> *Reference: Schulman et al. (2017), "Proximal Policy Optimization Algorithms"
> [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)*

---

## The Verifiable Reward Signal

The power of GRPO for reasoning comes from the reward function. We use *only*
verifiable rewards — no human judgment, no learned reward model.

| Domain | Verification method | Reward |
|---|---|---|
| Math (integer) | Exact match after normalization | 0 or 1 |
| Math (symbolic) | SymPy symbolic equivalence check | 0 or 1 |
| Code | Execute against test cases | fraction of tests passing |
| Logic | Deterministic checker | 0 or 1 |

For math: extract the answer from the model's output (looking for the last
`\boxed{}`, or the last number, or the answer after "="), normalize it
(remove commas, handle LaTeX fractions), and compare to the ground truth.

```python
def check_math_answer(predicted: str, ground_truth: str) -> float:
    # Try symbolic equivalence first (handles 1/2 == 0.5 == .5)
    try:
        pred_expr = sympy.sympify(predicted)
        true_expr = sympy.sympify(ground_truth)
        if sympy.simplify(pred_expr - true_expr) == 0:
            return 1.0
    except:
        pass
    # Fall back to string normalization
    pred_norm = normalize_answer(predicted)
    true_norm = normalize_answer(ground_truth)
    return 1.0 if pred_norm == true_norm else 0.0
```

The format reward (+0.1) is a small bonus for responses that contain a valid
`<think>...</think>` block. This incentivizes the model to maintain the CoT
structure under RL pressure (without it, the model can learn to optimize reward
by giving very short, confident answers without reasoning — which often fails
on harder problems but looks good on easy ones).

> **Sidebar: Why Not Use a Learned Reward Model?**
>
> The alternative is RLHF (Reinforcement Learning from Human Feedback): train
> a "reward model" on human preferences, then use that as the reward signal.
>
> This approach has two fundamental problems for reasoning:
>
> 1. **Reward hacking:** The RL training quickly finds ways to maximize the
>    reward model's score without actually improving reasoning quality. The
>    reward model is an imperfect proxy; the model learns to exploit that imperfection.
>
> 2. **Correctness vs. preference:** For math, a human saying "this reasoning
>    looks convincing" is a worse signal than "the answer is actually correct".
>    A confident wrong answer can score well on a preference reward model.
>
> Verifiable rewards are *correct by construction* — there's no proxy model
> to hack, because the verification is the ground truth. The only way to get
> reward=1 on a math problem is to produce the right answer.
>
> This is why we restrict GRPO to verifiable domains (math, code, logic). Natural
> language "reasoning quality" is not verifiable, so we don't try to GRPO it.

---

## The Training Data: Difficulty Filtering

GRPO training requires problems in the "40–80% solvable" difficulty range.

**Too easy (>80% pass rate):** The group of 8 completions all get reward=1.
The advantage for all is (1 - 1) / 0 = 0 (undefined, or 0 when using a small
ε for numerical stability). No gradient. The model doesn't improve.

**Too hard (<20% pass rate):** The group of 8 completions all get reward=0.
Again, all advantages are 0. No gradient.

The gradient signal only exists when the group has *mixed* rewards — some correct,
some incorrect. This is the 20–80% window.

We measure this by running the SFT checkpoint on each problem with 8 samples
and recording the pass rate:

```python
def filter_by_difficulty(problems, sft_model, min_rate=0.2, max_rate=0.8):
    results = []
    for problem in problems:
        completions = [sft_model.generate(problem, temperature=0.8)
                      for _ in range(8)]
        rewards = [check_answer(c, problem['answer']) for c in completions]
        pass_rate = sum(rewards) / 8
        if min_rate <= pass_rate <= max_rate:
            results.append({**problem, 'pass_rate': pass_rate})
    return results
```

From our full problem set (~1M problems), we expect 30–40% to fall in the useful
difficulty window.

---

## The KL Penalty

The objective includes a KL divergence term:

`L_total = L_GRPO - β * KL(π_θ || π_ref)`

where `π_ref` is the frozen SFT checkpoint (the starting point) and `β = 0.01`.

This penalty prevents the model from drifting too far from the SFT checkpoint.
Without it, GRPO would eventually "mode collapse" — the model would find a
narrow strategy that maximizes reward on the training distribution but completely
loses all other capabilities (language understanding, instruction following, etc).

The KL penalty says: "you can improve on reasoning, but don't forget everything
else the SFT checkpoint learned."

A common question: should you update the reference model `π_ref` periodically?
Some implementations update it every N steps. We use a static reference (never
updated). The argument for static: the SFT checkpoint is a high-quality behavioral
prior; why dilute it with in-progress RL states? The argument against: as the
model improves, the KL penalty will grow (diverging from an increasingly outdated
reference), which can limit how much improvement is achievable.

For our 5,000–20,000 step run, static reference is fine. For longer runs, this
deserves revisiting.

---

## Expected Outcomes

| Metric | SFT baseline | Target after GRPO |
|---|---|---|
| GSM8K pass@1 | ~35–45% | ~60–70% |
| MATH Level 1-2 pass@1 | ~20–30% | ~45–55% |
| MATH Level 4-5 pass@1 | ~2–5% | ~10–20% |
| HellaSwag | ~50–55% | Should stay stable (KL prevents regression) |

The most important ratio to watch: MATH improvement vs. HellaSwag stability.
GRPO should improve math substantially without collapsing general language ability.
If HellaSwag drops >5 points, the KL coefficient needs to increase.

---

## What Makes This Work (And What Can Make It Fail)

**Works:** Problems with unambiguous ground truth, a mix of easy and hard examples
in the training set, a well-calibrated SFT checkpoint that produces varied but
sometimes-correct reasoning.

**Fails:**
- *Format collapse:* The model learns to produce very short answers (just the final
  number) and skip the `<think>` block entirely. It can get reward=1 without reasoning.
  The format reward (+0.1 for valid `<think>` block) helps but doesn't fully prevent this.
  Fix: make format reward higher, or add a penalty for responses without `<think>`.

- *Mode collapse:* The model converges to one particular reasoning style that works
  on training problems but doesn't generalize. Fix: diverse problem set, temperature >0
  during generation.

- *Reward hacking the math checker:* The model learns to output answers in a format
  that passes the normalization but isn't semantically correct (e.g., outputting "0.5"
  when the answer is "1/2" and the checker normalizes both to 0.5 — but what if the
  checker has a bug?). Fix: multiple verification methods.

---

*Next: [Part 9 — Inference at the Edge: GGUF, Quantization, and Running on a
Raspberry Pi](09-inference.md)*
