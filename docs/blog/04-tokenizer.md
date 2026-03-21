# The Tokenizer: BPE, Digit Splitting, and Teaching a Model to Think

*Part 4 of a series on building a small reasoning language model end-to-end.*

---

The tokenizer is the first thing the model sees and the last thing standing
between your training data and token IDs. It's also frequently undertrained:
most people grab a pretrained tokenizer and move on. We trained ours from scratch
on the same corpus the model will pre-train on.

The key decisions: vocabulary size, digit tokenization, and the `<think>` special
token. All three have reasoning implications.

---

## Why Train From Scratch?

Using a pretrained tokenizer (GPT-4's cl100k, Llama's sentencepiece) seems reasonable:
they've been trained on enormous corpora and work well. Three reasons we didn't:

**1. Vocabulary alignment.** GPT-4's tokenizer has 100,277 entries. Our model
has `vocab_size = 32768`, a hard Trainium2 alignment constraint. We can't use
cl100k with this model without truncating or re-mapping the vocabulary, which
breaks the learned semantics.

**2. Domain-specific merges.** A general-purpose tokenizer trained on web text
makes different merge choices than one trained on math-heavy + code-heavy corpora.
For math: `\frac`, `\sqrt`, `\begin{align}` appear frequently enough that they
should probably be single tokens. For code: Python keywords and common idioms
similarly. A tokenizer trained on the same distribution as the model trains on
makes better merge decisions.

**3. Digit tokenization.** Most pretrained tokenizers merge digit sequences:
"142" becomes a single token. We explicitly break this. Getting digit tokenization
right requires control over the tokenizer training process.

---

## BPE: How It Works

Byte-Pair Encoding (Sennrich et al., 2016) starts with a byte-level vocabulary
(256 tokens, one per byte) and iteratively merges the most frequent pair of
adjacent tokens into a new token.

Algorithm:
```
Start: "hello world" = [h][e][l][l][o][ ][w][o][r][l][d]
Count pairs: (l,l) appears 1 time, (l,o) appears 2 times, (e,l) appears 1 time...
Merge most frequent pair: (l,o) → 'lo'
Result: [h][e][l][lo][ ][w][o][r][l][d]  (lo is now one token)
Repeat...
```

After 32768 - 256 = 32512 merges, we have a vocabulary of 32768 tokens. Common
words and subwords appear as single tokens; rare strings are encoded as sequences
of byte tokens.

The vocabulary size is a compression/granularity trade-off:
- Larger vocabulary → higher compression (fewer tokens per character)
- Smaller vocabulary → more tokens, but more compositional (each token is
  a more fundamental unit, which can help generalization)

For 32768: English text compresses to roughly 4 characters per token. Our corpus
achieves 2.04× compression ratio (tokens per character vs. characters per token).

> **Sidebar: Why Bytes, Not Characters?**
>
> Byte-level BPE starts with individual bytes (0–255) rather than Unicode
> characters. This matters for:
>
> - **No unknown tokens:** Every possible byte sequence can be represented.
>   A character-level tokenizer fails on Unicode code points outside its training
>   data; a byte-level tokenizer never fails.
>
> - **Code robustness:** Source code contains non-ASCII characters in string
>   literals, comments, and identifiers. Byte-level handles these correctly.
>
> - **Math symbols:** LaTeX uses backslash-prefixed sequences like `\alpha`, `\frac`.
>   These are ASCII-safe, but other math notation isn't. Byte-level handles all of it.
>
> The GPT-2 tokenizer popularized byte-level BPE. It's now the standard starting
> point for new tokenizers.
>
> *Reference: Sennrich et al. (2016), "Neural Machine Translation of Rare Words
> with Subword Units" [arXiv:1508.07909](https://arxiv.org/abs/1508.07909)*

---

## Why Individual Digit Tokenization Matters

Standard BPE merges frequent character sequences. The digit sequence "142"
appears frequently enough that it becomes a single token. "1,729" might become
two or three tokens. The tokenizer doesn't know that these are numbers.

This is catastrophically bad for arithmetic reasoning.

When the model sees "142 + 37 =", it needs to:
1. Recognize these as numbers
2. Process the digits to compute the sum
3. Generate "179"

If "142" is a single token with no internal structure, the model has no
compositional basis for understanding it as a three-digit number. It can
memorize "142 + 37 = 179" from the training data, but it can't generalize
to "143 + 37 = ?" through the same mechanism.

Individual digit tokenization — `1`, `4`, `2` as separate tokens — gives the
model the same internal representation for the digit "1" regardless of whether
it appears in 142, 1729, or 0.1. The model can learn positional arithmetic
operations over these consistent representations.

> **Sidebar: Does This Actually Help?**
>
> Yes. The evidence is strong. Lee et al. (2024) showed that tokenizers which
> treat numbers as single tokens substantially underperform on arithmetic
> benchmarks compared to digit-level tokenization. The gap is largest for
> multi-step arithmetic (carrying, borrowing) that requires positional
> reasoning across individual digits.
>
> Smaller models benefit more because they have less capacity to memorize
> arbitrary token-to-answer mappings. At 500M parameters, consistent digit
> representation is not optional if you want math capability.
>
> *Reference: Lee et al. (2024), "Teaching Arithmetic to Small Transformers"
> [arXiv:2307.03381](https://arxiv.org/abs/2307.03381)*

### How We Enforce It

The tokenizer training script adds individual digits as special initial splits:

```python
# Force digits to always tokenize as individual characters.
# These are added before BPE training so they become "atomic" — BPE
# will never merge them into larger tokens (you can only merge tokens
# that are already in the vocabulary, and merged digits would form
# new vocabulary entries that we explicitly exclude).
INDIVIDUAL_CHARS = list("0123456789")
```

We also prevent merging across decimal points and thousands separators:
"3.14" tokenizes as `3`, `.`, `1`, `4` — four tokens.
"1,729" tokenizes as `1`, `,`, `7`, `2`, `9` — five tokens.

---

## The `<think>` and `</think>` Special Tokens

We add five special tokens to the vocabulary:

```python
special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>", "<think>", "</think>"]
```

Their IDs are:
```
PAD_ID         = 0
BOS_ID         = 1
EOS_ID         = 2
UNK_ID         = 3
THINK_START_ID = 4
THINK_END_ID   = 5
```

The IDs are fixed by position, not looked up. Every training script references
them by constant (e.g., `THINK_START_ID = 4`) so they can't silently drift.

**Why `<think>` and `</think>` as vocabulary entries?**

The alternative is to use text strings: the model just outputs the literal
characters `<`, `t`, `h`, `i`, `n`, `k`, `>`. This works for inference —
you can parse the output — but it has training consequences:

1. **Loss accounting:** When the model generates the opening `<think>` token,
   we want the loss computed on that token as a unit. If it's 7 characters,
   there are 7 loss contributions, each with slightly different gradients.
   A single `<think>` token is a single loss contribution, which is cleaner.

2. **Boundary detection:** The SFT training script needs to find the "start of
   assistant response" to know where to begin computing loss. Finding a single
   known token ID is O(n). Finding the subsequence `<`, `t`, `h`, `i`, `n`,
   `k`, `>` is also O(n) but fragile if the tokenizer splits `<think>` differently
   in different contexts.

3. **Inference control:** At inference time, the special token ID is a reliable
   signal to start/stop capturing the reasoning chain. You can branch on a single
   integer comparison rather than parsing a multi-token string.

4. **No representation drift:** A regular text string can be tokenized differently
   depending on surrounding context (BPE merges depend on neighbors). A special
   token is always one token regardless of context.

---

## Vocabulary Size: 32768

`32768 = 2^15 = 256 × 128`. Both power-of-2 and tile-aligned.

The choice balances:

**Too small (16K):** Poor compression, more tokens per text, more steps to
read a document. Math and code especially suffer — LaTeX sequences become
long token streams.

**Too large (64K, 100K):** The embedding table grows. `100K × 1280 × 2 bytes
= 256MB` just for the embedding. For a 500M parameter model, 10% of memory
on a single matrix.  Also, large vocabularies have many rare tokens with poor
gradient signal during pre-training — they only get updated when their token
appears.

**32768:** `32768 × 1280 × 2 bytes = 83.9MB` for the embedding (42MB with
weight tying). Good compression (2.04× on our corpus). Sufficient for English +
math + code + multilingual basics.

The tile alignment means the embedding lookup (a masked select from a 32768×1280
matrix) and the LM head (a matrix multiply with output dim 32768) both map
cleanly to systolic array tiles on Trainium2.

---

## Training the Tokenizer

We use the HuggingFace `tokenizers` library with a BPE trainer:

```python
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

# ByteLevel pre-tokenizer: split on whitespace, map to bytes
# This is the GPT-2 approach: space-prefixed merges give better
# tokenization of text that starts with spaces (most words after
# the first in a sentence) vs. words at the start of a sentence.
tokenizer = Tokenizer(BPE(unk_token="<unk>"))
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

trainer = BpeTrainer(
    vocab_size=32768,
    special_tokens=["<pad>", "<bos>", "<eos>", "<unk>", "<think>", "</think>"],
    initial_alphabet=ByteLevel.alphabet(),  # start from byte vocabulary
)

# Train on a sample of the corpus
tokenizer.train(files=corpus_files, trainer=trainer)
```

We train on a 5% sample of the pre-training corpus (about 500M tokens). This
is enough to establish good merge statistics without requiring the full 10B
token corpus at tokenizer training time.

The training run takes about 20 minutes on a single CPU core and produces a
`tokenizer.json` file that's ~3MB.

---

## Verification: 28/28 Tests Pass

After training, we run a validation suite:

```
[PASS] Vocabulary size: 32768
[PASS] Special tokens: pad=0, bos=1, eos=2, unk=3, think_start=4, think_end=5
[PASS] Digit isolation: "142" → ['1', '4', '2']
[PASS] No cross-decimal merges: "3.14" → ['3', '.', '1', '4']
[PASS] LaTeX: \frac{1}{2} tokenizes reasonably (frac as single token)
[PASS] Code: Python keywords as single tokens (def, return, class, etc.)
[PASS] Think markers: <think> and </think> are single tokens
[PASS] Round-trip: encode → decode preserves all test strings
... (28 tests total)
```

Compression ratio: 2.04× on the validation corpus. This is in the expected range
for a math+code-heavy corpus (general web text typically achieves 3–4× with
32K vocab; math/code have more rare character sequences, reducing compression).

---

## What the Tokenizer Does Not Do

We made a deliberate choice not to:

**Use sentencepiece:** Sentencepiece (Google's library) uses a different BPE
implementation and is standard in LLaMA/Gemma. We use HuggingFace tokenizers
because it's more flexible for adding custom special tokens and has better Python
integration for the training pipeline.

**Support ChatML format:** ChatML uses `<|im_start|>` and `<|im_end|>` as
conversation delimiters. Our model uses plain "User: / Assistant:" text delimiters.
This is simpler and more robust — it doesn't require special token alignment
between the tokenizer and any downstream chat framework.

**Multilingual expansion:** Our vocab of 32768 covers English + math + code well.
Adding multilingual support would require either a larger vocab (50K+) or accepting
worse compression on non-English text. For v1, we optimize for English + reasoning.

---

*Next: [Part 5 — 10 Billion Tokens: Building a Pre-Training Data Pipeline](05-data-pipeline.md)*
