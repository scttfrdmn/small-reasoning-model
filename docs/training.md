# Training Recipe

Three sequential phases. The output checkpoint from each phase is the input to the next.
No phase can be skipped — GRPO requires a model that already knows the `<think>` format (Phase 1),
which requires a base model with language understanding (Phase 0).

For implementation see `training/pretrain.py`, `training/sft.py`, `training/grpo.py`.
For hyperparameter values see `configs/model_*.yaml`.

---

## Phase 0 — Pre-training

**Objective:** Next-token prediction on a large text corpus. Builds the base language model.

**Token budgets (deliberately over-trained):**

| Config | Tokens | Ratio to Chinchilla-optimal |
|---|---|---|
| 500M | 10B | ~20× |
| 1B | 50B | ~50× |
| 3B | 100B | ~33× |

Chinchilla optimal would be ~1× (roughly equal tokens to parameters). We overtrain intentionally
because Chinchilla-optimal minimizes *training compute* — not *inference-time quality*. For a model
that will be used at inference much more than it costs to train, more tokens = better final model
quality even past the optimal compute point. This is the strategy used by Llama 3, Qwen3, and
most production small models.

### Data Curriculum

The data mix is not uniform — it changes over training:

| Stage | FineWeb-Edu/DCLM | Stack v2 (code) | OpenWebMath | Wikipedia/Books | Notes |
|---|---|---|---|---|---|
| 0–30% of tokens | 40% | 25% | 25% | 10% | Establish language + code + math foundation |
| 30–100% of tokens | lower | 25% | 35% | 10% | Upweight math for Phase 2 readiness |
| Final 10% of tokens | 20% | 40% | 40% | — | Heavy math+code saturation |

**Why curriculum?** The model needs general language understanding first (web text, books), then
increasingly domain-specific math and code capability as training progresses. Providing all math
from the start is inefficient — the base language representations need to form first.

**Quality filtering:**
- Perplexity filter: remove documents where GPT-2 perplexity > 10,000 (extremely noisy / garbled text)
- MinHash LSH deduplication: 3-gram level, Jaccard threshold 0.8 (near-duplicates removed)

### Optimizer

AdamW with parameter-group-specific weight decay:
- Weight matrices: weight_decay=0.1
- Embeddings and norm parameters: weight_decay=0.0 (standard — regularizing these is counterproductive)

β2=0.95 rather than the default 0.999. Lower β2 makes the second-moment estimate track recent
gradient magnitudes more closely, which helps with the non-stationary loss landscape of LLM
pre-training where the loss distribution shifts as the curriculum progresses.

Fused AdamW on CUDA (`fused=True`) — the kernel fuses the optimizer update into a single CUDA
kernel launch, reducing memory bandwidth and kernel launch overhead.

### LR Schedule

Linear warmup (2% of total steps) followed by cosine decay to 10% of peak LR.

The warmup prevents early training instability. At random initialization, gradient norms are
high and the model is far from any meaningful local optimum. A low initial LR prevents the first
few steps from taking enormous destructive updates before the model has useful representations.

The cosine decay smoothly reduces the learning rate, allowing the model to settle into a good
minimum in the latter part of training rather than oscillating around it.

---

## Phase 1 — Supervised Fine-Tuning (SFT)

**Objective:** Teach instruction following and the `<think>…</think>` CoT format.

GRPO in Phase 2 reinforces *correct* reasoning chains. But before you can reinforce them,
the model must produce them in the first place. SFT plants the format.

### The Critical Loss-Masking Rule

**Loss is computed on assistant turns only.**

Token layout:
```
[<bos>] [user tokens...] [<sep>] [<think>] [...reasoning...] [</think>] [answer] [<eos>]
 ← NO LOSS ──────────────────────────────────────────────────────────────────────────────→ LOSS
```

If you compute loss on the user prompt tokens, the model learns to predict the *question* rather
than the *answer*. The gradient signal pushes the model toward memorizing question formats rather
than generating reasoning. This is a silent failure — the loss decreases, evaluation metrics
look reasonable in early training, but the model fundamentally cannot generalize reasoning because
it was never trained to own the generation from the `<think>` tag onward.

Critically, **loss IS computed on the `<think>` content** (the reasoning chain itself). The model
must learn to generate the reasoning steps, not just the final answer.

### Data Sources

| Dataset | Size | Purpose |
|---|---|---|
| NuminaMath-CoT | ~860K | Math with step-by-step solutions |
| OpenHermes 2.5 | ~1M (filtered) | General instruction following |
| CodeFeedback | ~66K | Code with explanation |
| Orca-Math | ~200K | Math word problems with reasoning |
| Synthetic CoT | ~200K | Generated from base model, filtered for correctness |

All examples are reformatted to the standard template before training:
```
User: {problem}
Assistant: <think>
{step-by-step reasoning}
</think>
{final answer}
```

### Hyperparameters

LR = 2e-5 (1B), 3e-5 (500M). This is ~15× lower than the pre-training peak LR. SFT is
fine-tuning on a relatively small dataset (~2M examples); a high LR would catastrophically
forget the base model's representations.

2 epochs. More epochs risk overfitting to the SFT dataset's surface patterns; fewer risk
under-learning the format. 2 is the empirical standard for SFT at this dataset size.

---

## Phase 2 — GRPO (Group Relative Policy Optimization)

**This is where reasoning is trained, not installed.**

GRPO is reinforcement learning with verifiable rewards. It requires:
1. A model that already produces `<think>…</think>` formatted output (from Phase 1)
2. A domain where correctness can be verified cheaply (math, code, logic)
3. A way to compare relative quality across multiple completions for the same prompt

### Algorithm

```
For each training step:
  1. Sample batch of prompts p from verifiable dataset
  2. For each prompt, sample G=8 completions {o₁...o₈} from policy π_θ
  3. Compute binary reward r_i ∈ {0,1} for each completion
  4. Compute group advantage: A_i = (r_i - mean(r)) / (std(r) + ε)
  5. Compute clipped policy gradient loss (PPO-style)
  6. Add KL penalty: β * KL(π_θ || π_ref),  β=0.01
  7. Update θ
```

**Why group normalization (step 4)?** The raw reward (0 or 1) is not a useful training signal
by itself. If all 8 completions are correct (reward=1), the mean is 1 and all advantages are 0 —
no gradient. If all are wrong, same result. Only when there is variance within the group (some
correct, some wrong) does the advantage estimate tell the model what to do. The group mean
serves as a free baseline that automatically adapts to the current difficulty of each prompt.

**Why KL penalty (step 6)?** Without it, the policy could drift arbitrarily far from the SFT
reference model, potentially forgetting the base language model's representations and format
compliance. The KL penalty keeps the policy anchored — it can change how it reasons, but not
change so much that it becomes incoherent text.

**Why only verifiable domains?** GRPO requires cheap, reliable reward computation for thousands
of completions per training step. Math answers can be verified in microseconds with SymPy.
Code can be tested against test cases in milliseconds. Natural language "quality" requires a
learned reward model (expensive to train, noisy, gameable). Restricting to verifiable domains
is not a limitation — it's what makes GRPO tractable.

### Four Improvements Over Vanilla GRPO

**[DAPO] Clip-Higher — Asymmetric PPO clipping**

Standard PPO clips the policy ratio to `[1-ε, 1+ε]` (symmetric). The upper bound prevents
the policy from increasing the probability of good actions by more than `ε` per step. Over many
steps, this causes entropy collapse — the model becomes increasingly confident (low entropy) and
stops exploring. With `clip_low=0.20, clip_high=0.28`, good actions can grow more freely while
bad actions are still bounded.

**[DAPO] Token-level policy gradient loss**

Sequence-level averaging divides the loss by sequence length, so a 2000-token correct CoT chain
gets the same total gradient magnitude as a 50-token correct answer. The model sees long reasoning
chains as equivalent to short ones, and is silently incentivized toward brevity. Token-level loss
averages across all tokens in the batch regardless of sequence length, correctly rewarding long
correct reasoning.

**[DAPO] Dynamic sampling**

Filter out prompts where all G completions have the same reward (all correct or all wrong). These
groups contribute zero gradient — the normalized advantage is 0 for every completion. Running
an optimizer step on zero gradient wastes compute and can cause numerical instability. Dynamic
sampling oversamples the prompt batch by 2× and discards uniform-reward groups before the update.

**[Dr. GRPO] Length-debiased advantages**

In the original GRPO, rewards are normalized within a group of completions for the same prompt.
But completions of different lengths have different numbers of tokens over which the policy
gradient loss is computed. A short correct completion has larger *per-token* gradient magnitude
than a long correct completion with the same reward. The model is implicitly pushed toward
generating shorter responses, not better ones. Dr. GRPO normalizes the advantage estimates by
completion length before computing group statistics, removing this confound.

### Difficulty Filtering

Filter the training set to problems where the SFT checkpoint solves 20–80% correctly.

- **> 80% pass rate:** Too easy. The reward is 1 for all or almost all completions.
  Group advantage is near-zero → near-zero gradient → wasted compute.
- **< 20% pass rate:** Too hard. The reward is 0 for all or almost all completions.
  Same problem.
- **20–80% pass rate:** The model can sometimes get it right and sometimes can't.
  This is the regime where the gradient signal is strongest and most informative.

### Early Stopping

Monitor pass@1 on held-out MATH Level 3 problems. Stop when validation performance
plateaus for 500 consecutive steps. Level 3 is chosen as the stopping criterion because it
is neither trivially easy (Level 1–2) nor impossibly hard (Level 4–5) for a 1B model —
it's the frontier where continued training has measurable effect.
