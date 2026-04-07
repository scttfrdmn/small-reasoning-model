# Five Bugs That Broke Generation (And What Each One Taught Us)

*Part 14 of a series on building a small reasoning language model end-to-end.*

---

After pre-training finished and SFT completed, we had what looked like a trained
model. The checkpoints were there. The loss had converged. We launched the GRPO
difficulty filter — a process that generates eight completions per math problem
and keeps only the ones where the pass rate falls between 20% and 80%.

The keep rate came back: **0%.**

Not 5%. Not 2%. Zero. Every single problem was either solved by the model every
time or never. That means either the model had become perfect overnight, or
something was very wrong with generation.

It was very wrong. Five separate bugs, each sufficient to destroy generation on
its own, were stacked on top of each other. This post is about what each one was,
why it was hard to see, and what the code looks like after each fix.

---

## Bug 1: The SFT Loss Wasn't Training the Model

Before we get to generation, there was a training bug that made the SFT checkpoint
worse than the pre-trained checkpoint it started from.

The `sft_loss()` function looked like this:

```python
def sft_loss(logits, labels):
    B, T, V = logits.shape
    loss = F.cross_entropy(
        logits.reshape(B * T, V),
        labels.reshape(B * T),
        ignore_index=LOSS_IGNORE,
    )
    return loss
```

This computes `CE(logits[t], labels[t])` — asking the model to predict the token
at position `t` given the input at position `t`. But causal attention lets position
`t` see itself. The model doesn't have to learn anything — it just routes the
current token through self-attention and echoes it.

The correct causal-LM loss asks: given positions 0..t, predict position t+1.
The implementation of `compute_loss()` in `model/architecture.py` (used during
pre-training) had this right:

```python
shift_logits = logits[:, :-1, :].contiguous()   # (B, T-1, V)
shift_labels = labels[:, 1:].contiguous()         # (B, T-1)
```

`sft_loss()` was missing this shift entirely. The model hit ~0 loss in epoch 1
not because it was learning, but because it had found a trivial solution that
doesn't require predicting anything.

**The tell:** loss dropped to near zero impossibly fast — before the model could
have seen enough examples to generalize. When you see SFT loss collapse in the
first few hundred steps, check the shift.

**The fix:** add `logits[:, :-1, :]` / `labels[:, 1:]` before the cross-entropy
call. Three lines.

---

## Bug 2: 670 GB of Disk Filled With Bad Checkpoints

With `save_every=500` and 121,000 training steps, the SFT run wrote 242 checkpoint
files at 2.8 GB each — 677 GB total. The disk filled up, the job died, and the
last valid checkpoint was buried under 671 GB of files from a corrupted training run.

**The deeper issue:** even if the disk hadn't filled, keeping 242 checkpoints of a
model you're going to overwrite anyway is wasteful. The only checkpoint that matters
for downstream tasks is the most recent one. Periodic saves are for resilience (if
the job dies, restart from the last checkpoint), not for archaeology.

**The fix:** bump `save_every` from 500 to 5000, and add a retention policy that
deletes the previous checkpoint whenever a new one is written:

```python
def _save(model, optimizer, step, path):
    torch.save({"model": model.state_dict(), "step": step, ...}, path)
    # Delete all but the latest step checkpoint
    checkpoints = sorted(Path(path).parent.glob("step_*.pt"))
    for old in checkpoints[:-1]:
        old.unlink()
```

We spent several hours recovering from this. The lesson is: think about the
full-run disk budget before you start, not after it's full.

---

## Bug 3: Prompts With EOS at the End

The tokenizer's post-processor — configured during training on HuggingFace datasets
to ensure consistent sequence termination — appends `<eos>` to every `encode()` call.

That's fine for training: the model sees `[..., <eos>]` and learns that EOS ends a
sequence. But for inference, if you encode a prompt and the last token is `<eos>`,
the model's first prediction is "what comes after end-of-sequence?" — which is
undefined, and the model produces garbage.

The symptom: garbled output with repeating tokens, often a single token (usually
`0` or `<bos>`) repeated hundreds of times.

**The fix:** strip the trailing EOS before passing to the model:

```python
ids = tokenizer.encode(prompt).ids
if ids and ids[-1] == eos_id:
    ids = ids[:-1]
```

This needs to happen everywhere you encode a prompt for generation: in the inference
server, in the GRPO difficulty filter, in any eval harness that uses the model
directly.

---

## Bug 4: The GRPO Filter Was Using Raw Prompts

The SFT model was trained on a specific instruction format:

```
User: {problem statement}
Assistant:
```

The GRPO difficulty filter was calling the model like this:

```python
enc = tokenizer(problem["problem"], return_tensors="pt")
```

Just the raw problem text, no wrapper. The SFT model had never seen a sequence
that started with a math problem statement and was expected to produce a math
solution — it had only ever seen sequences starting with `User:`. Without the
format, it was generating random text.

**The fix:** wrap the prompt before encoding:

```python
formatted_prompt = f"User: {problem['problem']}\nAssistant:"
```

This feels obvious in retrospect. The rule is: whatever format your training data
used, your inference code must use exactly the same format. Any deviation is
distribution shift.

---

## Bug 5: KV Cache Prefill Returned None

This was the root cause. The previous four bugs each individually caused generation
failures. But even after fixing all of them, generation was still producing
incoherent output. A careful comparison of full-sequence predictions vs. KV-cache
predictions showed they disagreed on every single token:

```
Full sequence after "Assistant:":
  '<think>'   logit=22.66   ← what we expect

KV cache version at same position:
  '<bos>'     logit=4.35    ← completely wrong
```

The KV cache decode was running each token in isolation with no context.

The `SmallReasoningModel.forward()` signature:

```python
def forward(self, input_ids, attention_mask=None, kv_caches=None, position_offset=0):
```

When `kv_caches=None` (the default), the model initializes `new_kv_caches = None`
and never collects KV tensors. Every caller was calling `model(input_ids)` with no
`kv_caches` argument — getting `kv_caches=None` back — and then passing that `None`
to the next decode step. Every single token was processed in training mode with no
context.

**The fix:** distinguish "collect KV but no prior cache" from "don't collect KV at
all." An empty list `[]` now means "prefill mode — collect KV but nothing to prepend
yet." `None` means "training mode — don't track KV at all."

```python
collect_kv = kv_caches is not None and len(kv_caches) == 0
```

All prefill callers changed to `model(input_ids, kv_caches=[])`.

After this fix, the prefill agreed with the full-sequence prediction. But the first
decode step still disagreed. There was one more bug.

---

## Bug 6: `is_causal=True` With T_q=1 Masks Everything Except Token 0

This was the final bug, and the hardest to see.

After the KV cache prefill fix, comparing predictions:

```
Full seq + 1 token (correct):
  '<think>'   logit=22.01

KV cache decode step 2:
  '<bos>'     logit=4.35
```

The prefill agreed. The first decode step didn't. The issue was in how PyTorch's
`F.scaled_dot_product_attention` implements `is_causal=True`.

When `is_causal=True`, SDPA generates a lower-triangular causal mask of shape
`(T_q, T_k)`. During a decode step, `T_q=1` (one new token) and `T_k=T_cache+1`
(all cached tokens plus the new one). The lower-triangular mask for a 1×16 matrix
looks like this:

```
[[True, False, False, False, ..., False]]
```

Only column 0 is unmasked. The single query token can attend to exactly one
position: the first token in the sequence (usually `<bos>`). Every cached position —
the entire prompt context — is masked to `-inf` and contributes zero to attention.

The model was generating tokens by attending only to `<bos>`. Across 200 tokens of
output, that produces tokens with vaguely reasonable surface statistics but no
coherent reasoning — exactly what we saw.

The reason this went unnoticed: the `is_causal` parameter was documented as "avoid
double-masking" and the comment pointed at training-mode padding. Nobody considered
that `is_causal=True` semantics change completely when the query length differs from
the key length.

**The fix:**

```python
is_decode = kv_cache is not None  # True only during KV-cache decode steps
out = F.scaled_dot_product_attention(
    q, k, v,
    attn_mask=attention_mask,
    dropout_p=dropout_p,
    is_causal=(attention_mask is None and not is_decode),
    scale=self.scale,
)
```

During decode mode, `is_causal=False`. This is correct because causality is already
guaranteed structurally: every cached K/V came from a token at an earlier position.
There is nothing to mask out. The query can attend to everything.

The same fix applies to the manual O(T²) fallback: skip the `.tril()` mask in decode
mode, since `.tril()` builds a T_q×T_q matrix that doesn't match the T_k columns.

---

## What Fixed Generation Looks Like

After all six fixes (counting the retention policy as a separate fix), generation
on the SFT checkpoint:

```
Q: What is 2 + 2?
A: Let me work through this.
   2 + 2 = 4

Q: Solve: x^2 - 5x + 6 = 0
A: Let me work through this.
   To solve x^2 - 5x + 6 = 0, we can use the quadratic formula:
   x = (-b ± √(b^2 - 4ac)) / 2a
   ...
```

The model produces chain-of-thought reasoning, uses the correct format, and stops
at appropriate places. It still makes arithmetic errors — `15 * 7 = 120` instead of
`105` — but those are learning failures, not generation failures. GRPO is supposed
to fix those.

---

## The Pattern

Looking across all six bugs, they have a common structure: **the training code and
the inference code made different assumptions about the same interface**.

- SFT loss assumed `logits[t]` predicts `ids[t]`. Pre-training assumed `logits[t]`
  predicts `ids[t+1]`. Two different functions, same concept, different conventions.
- The tokenizer's post-processor adds `<eos>` because training data always ends with
  `<eos>`. Inference assumes it doesn't.
- The SFT model was trained on `User:/Assistant:` format. The GRPO filter used raw
  text.
- `model(input_ids)` means "training forward pass." `model(input_ids, kv_caches=[])`
  means "prefill for generation." Same function, different modes, different callers.
- `is_causal=True` means "lower-triangular mask." During training (T_q == T_k) this
  is what you want. During decode (T_q=1, T_k >> 1) this masks everything except the
  first token.

The fix in each case was to close the gap — either by making the assumption explicit
in the interface, or by adding a check that catches the mismatch.

---

## What's Next

The GRPO difficulty filter is running now with correct generation. The 87,414 raw
problems will be filtered to the 20–80% pass-rate window — problems the SFT model
gets roughly half the time, which provides the best gradient signal for RL training.

If the filter produces a reasonable-sized dataset (hoping for 25,000–50,000
problems), we move to GRPO training: reinforcement learning with verifiable math
rewards. That's where the model will learn to actually reason rather than just
pattern-match from SFT.

The next post will cover what the GRPO training run looks like — including whether
the debugging effort on generation pays off in actual reward improvement.
