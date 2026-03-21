# Phase 1 SFT: Loss Masking and Teaching a Model to Think

*Part 7 of a series on building a small reasoning language model end-to-end.*

---

After pre-training, the model knows how to continue text. It doesn't know how
to follow instructions, produce structured outputs, or think step-by-step before
answering. That's what Phase 1 SFT teaches.

"SFT" — Supervised Fine-Tuning — is the simplest of the three training phases.
You show the model (question, reasoning chain, answer) examples and train it to
produce the reasoning chain and answer given the question. But there's one
non-obvious design decision that determines whether Phase 2 GRPO will work:
which tokens receive gradient signal.

---

## Why SFT Before GRPO?

Phase 2 GRPO uses reinforcement learning to improve the quality of the model's
reasoning. RL requires the model to *generate* reasoning chains and receive
feedback on whether they led to correct answers.

If you try this on a raw pre-trained model (before SFT), two things happen:
1. The model generates incoherent reasoning chains with no consistent format
2. These chains almost never produce correct answers, so reward is always 0
3. RL with constant-zero reward makes no progress (there's no gradient signal)

SFT provides the behavioral prior: it teaches the model to produce structured
`<think>...</think>` chains with plausible reasoning before answering. With
that prior, the GRPO stage can distinguish "reasoning that leads to correct
answers" from "reasoning that leads to wrong answers" and reinforce the former.

Concretely: a raw pre-trained model gets ~2% pass@1 on GSM8K math. An SFT model
gets ~35–45% pass@1. GRPO then improves from 35–45% to (ideally) 65–75%.
GRPO can't start from 2%.

---

## The Data Format

Every SFT example becomes:

```
User: {problem}
Assistant: <think>
{step-by-step reasoning}
</think>
{final answer}
```

For example, from NuminaMath-CoT:
```
User: Find the value of x where 3x + 7 = 22.
Assistant: <think>
We need to solve for x in the equation 3x + 7 = 22.

Subtract 7 from both sides:
3x = 22 - 7
3x = 15

Divide both sides by 3:
x = 15/3
x = 5
</think>
x = 5
```

This format is consistent across all source datasets. The `sft_format.py` script
handles the conversion from five different source formats (NuminaMath-CoT,
OpenHermes, CodeFeedback, Orca-Math, Alpaca-style) into this single template.

---

## The Critical Decision: Loss Masking

Here is the single most important design decision in SFT: **we compute loss only
on the assistant's response, not on the user's prompt.**

Concretely, given the token sequence:
```
[<bos>] [User:] [Find] [the] [value]...  [Assistant:] [<think>] ... [</think>] [x=5] [<eos>]
  mask=0   mask=0  mask=0  mask=0  mask=0     mask=0      mask=1    mask=1     mask=1   mask=1
```

Tokens with `mask=0` get `label = -100` (PyTorch's ignore index). These positions
contribute zero gradient to the loss. Only tokens with `mask=1` — the assistant's
output — contribute.

**Why not compute loss on the full sequence?**

Two reasons, both important:

*Reason 1: You'd be training the model to predict the user's prompt.*
The user prompt is the question the model receives as input. Training the model
to predict "Find the value of x where 3x + 7 = 22" teaches it to predict question
text, not to answer questions. Prompts vary wildly in phrasing; the model would
waste capacity learning prompt-distribution statistics instead of answer-generation.

*Reason 2: The model needs to *own* the reasoning chain.*
At inference time, the model generates the entire `<think>` block from scratch.
If we don't compute loss on the `<think>` tokens during training, the model sees
them only as context (input) and never receives gradient signal for producing them.
When you ask it to generate `<think>` output during inference, it has no learned
preference for how to fill that block — it was never trained on generating it.

This is a subtle failure mode. If you compute full-sequence loss (a common mistake),
the model learns to *predict* `<think>` tokens when they appear in the input, but
not to *generate* them as part of a response. The model will appear to work
during training (loss decreases) but fail silently during GRPO (the model doesn't
produce consistent reasoning chains, so GRPO has nothing to reinforce).

> **Sidebar: The Difference Between Predicting and Generating**
>
> A language model computes `P(token_t | tokens_0 ... token_{t-1})` — the
> probability of the next token given all previous tokens. During training,
> we use "teacher forcing": the model is always shown the ground-truth previous
> tokens, even if its own predictions would have been wrong.
>
> "Predicting" in the loss-masking sense means: the model is given tokens 0..t-1
> (including the ground-truth token_t-1) and asked to predict token_t. Loss is
> computed at position t.
>
> "Generating" at inference means: the model is given only tokens 0..t-1 and
> must produce token_t from its own probability distribution. There's no teacher
> forcing — each generated token becomes the input for the next step.
>
> The subtle point: a model that receives gradient signal at position t learns
> the joint distribution of (context, target) pairs where the target is at position t.
> A model that receives NO gradient at position t never learns to produce that
> position's tokens from scratch — it only sees them as context for subsequent predictions.
>
> To generate good `<think>` blocks at inference, the model must have received
> gradient on `<think>` block tokens during training. Loss masking must NOT mask
> the assistant's response.

---

## How We Implement the Boundary Detection

The boundary between "prompt" and "response" is the "Assistant:" marker. We
find it by encoding "Assistant:" as a token sequence and searching for the last
occurrence in the full sequence (last, not first, to handle multi-turn examples
where "Assistant:" might appear in an earlier turn):

```python
asst_marker = tokenizer.encode("Assistant:").ids
# strip BOS/EOS that the tokenizer adds around standalone encode() calls
asst_marker = asst_marker[1:-1]

# Find the LAST occurrence
boundary = _find_subsequence_last(ids, asst_marker)

# Everything after the boundary is the loss region
mask_start = boundary + len(asst_marker)
labels = [LOSS_IGNORE] * mask_start + ids[mask_start:]
```

If the tokenizer doesn't find "Assistant:" as a contiguous token sequence (can
happen with unusual BPE merges), we fall back to masking the first 20% of the
sequence as a crude proxy. This is logged as a warning.

---

## The Training Hyperparameters

| Parameter | Value | Reasoning |
|---|---|---|
| Peak LR | 2e-5 (500M) | ~10× lower than pre-training; prevents catastrophic forgetting |
| LR schedule | Cosine, 3% warmup | Short warmup: model already has stable gradients |
| Epochs | 2 | SFT datasets are small; >3 epochs typically causes format overfit |
| Batch | 4 seq × 8 accum = 32 eff | Memory-bounded at 4096-token sequences |
| Max seq len | 4096 | Pre-training used 2048; CoT needs more headroom |
| Grad checkpointing | Yes | 4096-token sequences + 500M params need memory relief |

**Why lr=2e-5 specifically?**

The pre-training LR was 3e-4. SFT at 3e-4 would "catastrophically forget" the
pre-trained language understanding — the gradient signal from ~2M SFT examples
would overwrite the weight statistics learned from 10B tokens. The model would
become highly specialized on the SFT distribution at the cost of general capability.

The empirical sweet spot found across LLaMA-2, Mistral, and most public SFT
recipes is 1–3e-5. At this scale (500M parameters, 2M examples, 2 epochs):
- High enough to learn the instruction format within 2 epochs
- Low enough to preserve pre-trained language understanding

We monitor val loss across epochs. If val loss increases after epoch 1, we stop.

**Why 4096 max sequence length?**

Pre-training used 2048. A typical CoT math solution is 400–800 tokens inside
`<think>`. With a 500-token problem statement and a 500-token final answer, that's
500 + 800 + 500 = 1800 tokens — fits in 2048. But some NuminaMath-CoT examples
have multi-page solutions. 4096 gives comfortable headroom.

The memory cost is significant: attention has O(seq_len²) memory for the
activation cache. At seq_len=4096 vs 2048, that's 4× the attention activation
memory. Gradient checkpointing is essentially mandatory at this sequence length
on a 32GB GPU.

---

## The Data Sources

| Source | Examples | Format |
|---|---|---|
| NuminaMath-CoT | ~860K | Full step-by-step math solutions |
| OpenHermes 2.5 | ~1M (filtered) | Multi-turn instruction following |
| CodeFeedback | ~66K | Code with explanation |
| Orca-Math | ~200K | Math word problems |

NuminaMath-CoT is the most valuable source for our purposes: every example has
a full chain-of-thought solution in LaTeX notation. We wrap the solution in
`<think>` tags and extract the final answer for the post-think output.

OpenHermes provides general instruction-following diversity. Many examples have
no reasoning chain — we inject a minimal `<think>\nLet me think about this.\n</think>`
placeholder. The placeholder is weak signal, but it ensures consistent format
across all training examples.

---

## What This Run Produces

Expected:
- Training loss: decreasing from ~3.5 to ~1.5–2.0 over 2 epochs
- Val loss at end: ~1.8–2.2
- Model capability: ~35–45% GSM8K pass@1 (from ~2% for the base model)
- Format compliance: consistent `<think>` blocks in nearly all generations

The best checkpoint (lowest val loss) is saved as `checkpoints/sft/best.pt`.
This checkpoint is the input to Phase 2 GRPO.

---

*Next: [Part 8 — Phase 2 GRPO: Reinforcement Learning With Verifiable Rewards](08-grpo.md)*
