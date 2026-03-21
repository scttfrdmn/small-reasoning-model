# 10 Billion Tokens: Building a Pre-Training Data Pipeline

*Part 5 of a series on building a small reasoning language model end-to-end.*

---

Pre-training quality is determined almost entirely by data. Architecture matters.
Hyperparameters matter. But bad data with a good architecture will produce a bad
model; good data with a mediocre architecture produces a surprisingly capable one.

We built a data pipeline that streams 10B tokens from multiple sources,
applies quality filters, deduplicates, and mixes sources with a curriculum
schedule that upweights math and code over the course of training.

---

## Source Selection

We download from HuggingFace Hub, mixing four categories:

| Category | Dataset | Purpose |
|---|---|---|
| Filtered web | FineWeb-Edu | High educational-quality web text |
| Math text | OpenWebMath, NuminaMath-TIR | Pre-trains math understanding |
| Code | codeparrot/github-code (Python) | Reasoning circuit pre-training |
| Reference | Wikipedia | World knowledge anchor |

**Why these and not others:**

FineWeb-Edu (HuggingFace, 2024) is web text filtered by an educational quality
classifier trained on teacher-rated documents. It produces dramatically better
downstream reasoning than raw Common Crawl. Penedo et al. showed FineWeb-Edu
produces models that score better on ARC, HellaSwag, and MMLU than models trained
on unfiltered web text at the same token count.

OpenWebMath is a curated collection of mathematical content from the web —
textbooks, course notes, MathOverflow, arXiv. Mathematical text is the highest-
density signal for the number system concepts Phase 2 GRPO will reinforce.
Paying attention to math *before* GRPO makes GRPO dramatically more efficient.

Code has an outsized effect on structured reasoning beyond programming tasks.
The hypothesis (supported empirically by Guo et al., 2024 and others): code
forces a language model to learn precise causal chains — `x = f(y)` means
exactly that `x` depends on `y`. Web text is vague; code is exact. This
"structured thinking" transfers to math and logic.

Wikipedia provides dense world knowledge in a high-quality writing style.
Without it, models tend to hallucinate facts they never saw in training.

> **Sidebar: Why We Didn't Use The Stack v2**
>
> Our original plan included `the-stack-v2-train-smol-ids`, HuggingFace's curated
> code dataset. We ran into a gating issue: the dataset requires HuggingFace
> account approval (to comply with the included licenses), and our automated
> pipeline doesn't support interactive auth flows.
>
> We switched to `codeparrot/github-code` filtered to Python. It's public, covers
> our primary use case (Python reasoning), and has about 15B tokens of Python code.
> We sample ~25% of our token budget from it.
>
> The lesson: when building automated pipelines, use datasets that don't require
> interactive auth, or pre-authenticate separately and check auth status before
> starting a long run.

---

## Quality Filtering

Raw web data has a lot of junk. We apply three filters in sequence:

### 1. Length Filter
```python
if len(text.split()) < 50 or len(text.split()) > 100_000:
    skip()
```
Too short (< 50 words): probably a fragment, error page, or navigation text.
Too long (> 100K words): probably a dump or malformed document.

### 2. Content Quality Filter
```python
def quality_score(text):
    words = text.split()
    # Ratio of non-ASCII characters (high → likely spam or encoding errors)
    non_ascii = sum(1 for c in text if ord(c) > 127) / len(text)
    # Average word length (too short → likely garbled, too long → likely code blob)
    avg_word_len = len(text) / len(words)
    return non_ascii < 0.15 and 3 < avg_word_len < 15
```

This is a practical substitute for full perplexity filtering (the spec calls for
GPT-2 perplexity scoring; we use heuristics for the validation run because GPT-2
scoring requires a running model and significantly slows the pipeline). We'll
add proper perplexity filtering for production runs.

### 3. Exact Deduplication

SHA-256 hash of the first 500 characters of each document. If we've seen this
prefix before, skip the document.

```python
seen_hashes = set()
prefix_hash = hashlib.sha256(text[:500].encode()).hexdigest()
if prefix_hash in seen_hashes:
    skip()
seen_hashes.add(prefix_hash)
```

This catches exact duplicates and near-duplicates that share a common opening
(very common for templated pages, syndicated articles). It doesn't catch
semantic duplicates with different text — that requires MinHash LSH, which we
defer to the production run.

> **Sidebar: Why Deduplication Matters So Much**
>
> Training data is not uniformly distributed. Some documents — Wikipedia articles,
> popular blog posts, StackOverflow questions — appear many times in the raw web
> crawl. If your model sees "What is the derivative of sin(x)?" 500 times, it
> memorizes that specific phrasing rather than learning the underlying calculus.
>
> More subtly: memorized text creates "shortcuts" in the model's representations.
> The model learns to recognize specific text patterns and output memorized answers,
> rather than constructing answers from principles. This hurts generalization.
>
> MinHash LSH (near-duplicate detection) and exact deduplication both help, but
> MinHash is significantly more expensive. For the validation run at 10B tokens,
> exact dedup is good enough. For the 50B–100B production run, we'll need MinHash.
>
> *Reference: Lee et al. (2022), "Deduplicating Training Data Makes Language Models
> Better" [arXiv:2107.06499](https://arxiv.org/abs/2107.06499)*

---

## Curriculum Mixing: Why Order Matters

We don't shuffle all sources uniformly. We use a three-stage curriculum:

| Stage | Tokens consumed | Mix |
|---|---|---|
| Stage 1 | 0–30% | FineWeb-Edu 40%, Code 25%, Math 25%, Wikipedia 10% |
| Stage 2 | 30–90% | FineWeb-Edu 30%, Code 20%, Math 35%, Wikipedia 15% |
| Stage 3 | 90–100% | FineWeb-Edu 20%, Code 40%, Math 40%, Wikipedia 0% |

Math and code proportions increase over training.

**Why not uniform mixing?**
The model learns different things in different training phases. Early training
is about learning language fundamentals — grammar, vocabulary, common patterns.
Too much math too early overwhelms this with domain-specific notation before
the model has a firm language foundation.

Late training is when the model is already stable and can absorb domain-specific
information effectively. Concentrating math and code at the end means the
final weights are heavily biased toward the domains that matter for GRPO.

This is "curriculum learning" — a well-studied phenomenon where ordering
training examples from simpler to harder (or from general to specific) produces
better final models than random ordering.

> **Sidebar: Evidence for Curriculum Learning in LLMs**
>
> The original curriculum learning paper (Bengio et al., 2009) showed that
> training on easy examples first and gradually introducing harder ones improves
> convergence and final quality. For LLMs, the "difficulty" axis is domain
> specificity: general web text is "easy" (lots of training signal, familiar
> patterns), specialized domains are "hard" (sparse, unfamiliar notation).
>
> Practical evidence: the Falcon-180B paper found that increasing code proportion
> late in training improved math scores without hurting language benchmarks.
> LLaMA 3 uses a similar upweighting strategy in its final training stage.
>
> *Reference: Bengio et al. (2009), "Curriculum Learning"
> https://icml.cc/Conferences/2009/papers/119.pdf*

### Implementation: MixedStreamSampler

The curriculum mixer is implemented as a streaming sampler that:
1. Maintains an iterator per source
2. At each document request, calls `get_stage_mix(tokens_written)` to determine
   the target proportions for the current training stage
3. Samples a source according to those proportions
4. Yields the next document from that source

```python
def get_stage_mix(tokens_written: int, total_target: int,
                  allowed_sources: Optional[set] = None) -> dict:
    """Return source weights for the current training stage."""
    progress = tokens_written / total_target
    if progress < 0.30:
        weights = {"fineweb_edu": 0.40, "the_stack": 0.25,
                   "open_web_math": 0.25, "wikipedia": 0.10}
    elif progress < 0.90:
        weights = {"fineweb_edu": 0.30, "the_stack": 0.20,
                   "open_web_math": 0.35, "wikipedia": 0.15}
    else:
        weights = {"fineweb_edu": 0.20, "the_stack": 0.40,
                   "open_web_math": 0.40, "wikipedia": 0.00}
    ...
```

A bug we hit: when a dataset source was exhausted (or errored out on download),
we popped it from the active sources dict. But `get_stage_mix()` still returned
it in the proportions dict with its original weight, causing a `KeyError` when
we tried to draw from it. The fix: compute `effective_allowed = live_sources &
allowed_sources` before each mix computation to remove exhausted sources from
the proportions.

---

## The Pre-Tokenization Step

The original pipeline tokenized documents on-the-fly during training. This caused
a deadlock (the details are in Part 6). We added a pre-tokenization step that
converts the full 39GB JSONL corpus to a flat binary file:

```
Input:  data/pretrain/train.jsonl    (39 GB, 7.1M documents)
Output: data/pretrain_tokenized/
          train.bin   (19.7 GB, 9.5B tokens, uint16)
          val.bin      (1.04 GB, 520M tokens, uint16)
          meta.json    (token counts, tokenizer path, timestamp)
```

The binary format is flat: `uint16` values packed sequentially. At training
time, a `np.memmap` wraps the file, and we read contiguous chunks of
`max_seq_len + 1` tokens for each training example.

```python
class BinaryTokenDataset(IterableDataset):
    def __iter__(self):
        offset = random.randrange(0, self.max_seq_len)  # random phase
        pos = offset
        while pos + self.max_seq_len + 1 <= self.n_tokens:
            chunk = self.data[pos : pos + self.max_seq_len + 1].astype(np.int64)
            input_ids = torch.from_numpy(chunk[:-1]).long()
            labels    = torch.from_numpy(chunk[1:]).long()   # shifted by 1
            yield {"input_ids": input_ids, "labels": labels}
            pos += self.max_seq_len
```

The `random.randrange(0, max_seq_len)` phase offset means different epochs
start at different positions in the file, avoiding the same sequence boundaries
every time.

---

## What It Produces

The full pipeline runs for ~10 hours (data download) + ~2 hours (tokenization):

```
Total documents:        7,098,116
Total tokens (train):   9,501,234,688  (9.5B)
Total tokens (val):       519,762,156  (520M)
Train file size:        19.73 GB
Val file size:           1.04 GB

Source breakdown (estimated from manifest):
  fineweb_edu:           ~40% of tokens
  open_web_math:         ~28% of tokens
  the_stack (python):    ~23% of tokens
  wikipedia:              ~9% of tokens
```

The val set is a held-out 5% drawn from each source proportionally. Val loss
at the end of pre-training is ~2.61 (compared to ~3.35 at step 1), indicating
the model has learned significant language modeling capability.

---

## What We'd Do Differently

For the production 50B-token runs:

1. **MinHash near-duplicate detection.** Exact dedup misses paraphrases and
   lightly modified content. MinHash with 3-gram shingles and Jaccard threshold
   0.8 catches these, at the cost of ~5× pipeline time.

2. **GPT-2 perplexity filter.** Documents where GPT-2 perplexity > 10,000
   are likely gibberish (auto-generated spam, encoding errors, non-English
   content we don't want). This removes ~2–5% of documents but improves
   downstream quality.

3. **Streaming tokenization with checkpointing.** At 50B+ tokens, the
   pre-tokenization step takes 20+ hours. We want it to be resumable if
   interrupted. The current implementation restarts from scratch.

4. **Better code filtering.** We use Python-only from codeparrot. A broader
   code corpus (Python + JavaScript + Rust + Go) with language-aware quality
   filters would probably improve generalization.

---

*Next: [Part 6 — Two Deadlocks and a GPU at 98%: Debugging the Training
Infrastructure](06-debugging.md)*
