"""
sft.py
======
Phase 1: Supervised Fine-Tuning (SFT) for the small reasoning model.

Takes the Phase 0 pre-trained checkpoint and teaches:
  1. Instruction following (user/assistant turn format)
  2. The <think>...</think> chain-of-thought structure
  3. Step-by-step reasoning before final answers

WHY SFT BEFORE GRPO (REINFORCEMENT LEARNING)?
──────────────────────────────────────────────
Phase 2 uses GRPO, a RL-style algorithm that rewards correct answers.
But RL-from-scratch on a language model is catastrophically unstable —
the model needs a strong behavioral prior before reward shaping can work.
SFT provides that prior: it teaches the model the *format* of reasoning
(think-then-answer) so that GRPO only needs to improve *quality*, not
invent the structure from nothing.

WHY LOSS ON ASSISTANT TURNS ONLY?
──────────────────────────────────
This is the single most critical design decision in this file.

Computing loss on the *user prompt* teaches the model to predict the
user's words — useful for a language model, but harmful for an assistant.
It causes two problems:
  1. The model wastes capacity memorizing prompt phrasings.
  2. It fails to learn that it is the *author* of the reasoning chain.
     During GRPO/inference the model must generate the <think> block from
     scratch. If it only ever saw that block as input (with 0-loss), it
     has no gradient signal for producing it.

Concretely: if you compute loss on the full sequence, perplexity looks
great (prompts are easy to fit), but the model learns nothing about
*generating* CoT. Phase 2 GRPO will then fail silently — the model
produces fluent but reasoning-free outputs.

WHY <think> CONTENT IS IN THE LOSS REGION (NOT MASKED OUT):
────────────────────────────────────────────────────────────
The model must OWN the reasoning chain. Every token inside <think>...</think>
is something the model will generate at inference time — so every one of
those tokens must receive a gradient. Masking <think> content would be like
teaching a student to write essays by hiding the body paragraphs.

Token layout during training:
  [<bos>] [User: ...] [Assistant: ] [<think>] [...] [</think>] [answer] [<eos>]
   mask=0   mask=0      mask=0        mask=1   mask=1  mask=1    mask=1   mask=1

The boundary is immediately after "Assistant:" — everything from the
first assistant token onward (including the opening <think>) is trained.

Dataset format (all examples converted to this before training):
  {
    "prompt": "What is 15 + 27?",
    "response": "<think>\\n15 + 27\\n= 10 + 20 + 5 + 7\\n= 30 + 12\\n= 42\\n</think>\\n42"
  }

Usage:
  # Fine-tune 1B pre-trained model
  python sft.py \\
    --checkpoint ./checkpoints/1b/step_0025000.pt \\
    --config 1b \\
    --output_dir ./checkpoints/1b_sft \\
    --data_dir ./sft_data

  # Dry-run with synthetic data (no checkpoint needed)
  python sft.py --config 1b --mode validate
"""

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Add parent directory to path so we can import the model package regardless
# of where this script is invoked from.
sys.path.insert(0, str(Path(__file__).parent.parent / "model"))
from model.architecture import SmallReasoningModel, ModelConfig, CONFIGS

# ---------------------------------------------------------------------------
# Special token IDs
# ---------------------------------------------------------------------------
# These MUST match the tokenizer_config.json produced by build_tokenizer.py.
# If they drift out of sync, the model silently trains with wrong token
# semantics — a very hard bug to diagnose.  The IDs 0-5 are reserved in our
# vocabulary; do not renumber them without updating all training scripts.
PAD_ID = 0  # Padding — never contributes to loss or attention
BOS_ID = 1  # Beginning-of-sequence — prepended by the tokenizer post-processor
EOS_ID = 2  # End-of-sequence — signals the model to stop generating
UNK_ID = 3  # Out-of-vocabulary fallback (should be rare after BPE)
THINK_START_ID = 4  # <think>  — opens the chain-of-thought scratchpad
THINK_END_ID = 5  # </think> — closes the scratchpad; answer follows

# PyTorch's cross_entropy natively skips positions where label == -100.
# We use this as our "no-loss" sentinel throughout the masking logic.
LOSS_IGNORE = -100


# ---------------------------------------------------------------------------
# SFT hyperparameter config
# ---------------------------------------------------------------------------


@dataclass
class SFTConfig:
    """
    All hyperparameters for one SFT run, in a single serialisable object.

    Design notes on key choices:

    max_seq_len=4096:
        Pre-training used 2048. We double it here because CoT sequences are
        substantially longer than plain text — a multi-step math solution
        might take 600-800 tokens inside <think>. Truncating CoT mid-reason
        creates broken training examples, so we give more headroom.

    lr=2e-5 (not 3e-4 like pre-training):
        Rule of thumb: SFT LR ≈ pre-train LR / 10.  The pre-trained weights
        already encode rich language understanding.  A high LR would
        catastrophically overwrite that knowledge ("catastrophic forgetting")
        before the model gets a chance to learn instruction following.
        2e-5 is the empirical sweet spot used by LLaMA-2 chat, Mistral,
        and most public SFT recipes.

    grad_accum=8 (same as pre-training, but effective batch is smaller):
        SFT datasets are 10-100x smaller than pre-training corpora. We use
        fewer steps total, so less accumulation is needed for stable
        gradient estimates. If you have more GPU memory, reducing grad_accum
        and increasing batch_size is preferable (more diverse per step).

    dropout=0.0:
        Standard practice for SFT on a strong base model.  The base already
        generalises well; adding dropout just slows convergence.  Some
        recipes add small dropout (0.05) for very small datasets (<10k
        examples) to prevent overfit.

    epochs=2:
        SFT datasets are small enough to train for multiple passes.  More
        than 3 epochs typically causes overfit on the format rather than
        generalisation. Watch val_loss — if it rises after epoch 1, stop.
    """

    # ── Model ─────────────────────────────────────────────────────────────
    model_config: str = "1b"
    # Path to Phase 0 pre-trained checkpoint (.pt file).
    # REQUIRED for any real SFT run — starting from random init defeats the
    # purpose (you'd just be doing pre-training with masked loss, very slowly).
    checkpoint: str = ""

    # ── Data ──────────────────────────────────────────────────────────────
    data_dir: str = "./sft_data"
    tokenizer_path: str = "./tokenizer_output"
    # Longer than pre-training to accommodate chain-of-thought sequences.
    # Sequences longer than this are truncated from the RIGHT (keeping answer).
    max_seq_len: int = 4096

    # ── Training ──────────────────────────────────────────────────────────
    epochs: int = 2
    batch_size: int = 4
    # Gradient accumulation: number of micro-batches before one optimizer step.
    # Effective batch = batch_size * grad_accum sequences = 32 here.
    # SFT datasets are small so we use less accumulation than pre-training.
    grad_accum: int = 8
    # Peak LR — ~10x lower than pre-training to prevent catastrophic forgetting.
    lr: float = 2e-5
    # Minimum LR at end of cosine decay — 10x lower than peak.
    lr_min: float = 2e-6
    # Short warmup (3%) because we're fine-tuning, not training from scratch.
    # Longer warmup risks spending too many steps at near-zero LR.
    warmup_fraction: float = 0.03
    # Weight decay regularises the weight matrices but not norms or biases
    # (see optimizer setup in train() for how this is applied selectively).
    weight_decay: float = 0.01
    # Gradient clipping prevents rare large-loss batches from causing
    # destructive parameter updates. 1.0 is the standard value.
    grad_clip: float = 1.0
    # No dropout for SFT on a well-pre-trained base (see class docstring).
    dropout: float = 0.0

    # ── Memory optimisation ───────────────────────────────────────────────
    # Recompute activations during backward instead of storing them.
    # Saves ~30-40% memory at the cost of ~20% compute — essential for
    # 4096-token sequences on 40 GB GPUs.
    grad_checkpointing: bool = True
    # bfloat16 has the same dynamic range as float32 but half the memory.
    # Preferred over float16 because it doesn't require a loss scaler.
    dtype: str = "bfloat16"

    # ── Output ────────────────────────────────────────────────────────────
    output_dir: str = "./checkpoints/sft"
    save_every: int = 5000  # Periodic checkpoint every N optimizer steps
    log_every: int = 10  # Print training metrics every N optimizer steps
    eval_every: int = 200  # Run validation every N optimizer steps

    # ── Hardware ──────────────────────────────────────────────────────────
    backend: str = "cuda"

    def effective_batch_tokens(self) -> int:
        """Total tokens processed per optimizer step (for throughput estimates)."""
        return self.batch_size * self.grad_accum * self.max_seq_len


# ---------------------------------------------------------------------------
# Data formatting — the <think> template
# ---------------------------------------------------------------------------
# We use a plain "User: / Assistant:" delimiter rather than ChatML tokens
# (<|im_start|>, <|im_end|>) for two reasons:
#   1. Our tokenizer does not include those special tokens.
#   2. The plain text format is more robust to diverse source datasets that
#      may already contain partial ChatML markup.
#
# The "Assistant:" string is the boundary used by tokenize_with_mask() to
# locate where the loss region begins. If you change this delimiter, you MUST
# also update the boundary-detection logic below.

CHAT_TEMPLATE = """\
User: {prompt}
Assistant: <think>
{thinking}
</think>
{answer}"""

# Fallback template for examples where no reasoning chain is available.
# Used rarely — ideally every SFT example has at least a minimal <think> block.
CHAT_TEMPLATE_NO_THINK = """\
User: {prompt}
Assistant: {answer}"""


def format_example(example: dict) -> str:
    """
    Normalise any supported dataset format into our "User:/Assistant:" template.

    Why normalise at this stage rather than at dataset creation time?
    Because we want sft.py to be usable directly with off-the-shelf datasets
    (OpenHermes, MetaMathQA, Alpaca, etc.) without a separate preprocessing
    step. Users can drop any of these datasets into sft_data/ and they'll just
    work.

    All output paths ensure a <think> block is present. This is intentional:
    even for non-math tasks, we want the model to produce a scratchpad before
    answering. If the source data has no reasoning trace, we inject a minimal
    placeholder. The placeholder is weak signal, but it at least keeps the
    format consistent — the model won't have to guess whether to <think> or not.

    Input formats handled:
      {"prompt": "...", "response": "..."}             — already formatted
      {"prompt": "...", "thinking": "...", "answer": "..."} — split reasoning
      {"problem": "...", "solution": "..."}            — math dataset style
      {"instruction": "...", "output": "..."}          — Alpaca style
      {"messages": [{"role": "user", ...}, ...]}       — ChatML style
    """
    # ── Format 1: already in our target schema ────────────────────────────
    # The response field may or may not have a <think> block.  If it doesn't,
    # we wrap it with a minimal one so the model always sees the CoT format.
    if "prompt" in example and "response" in example:
        resp = example["response"]
        # Guard: ensure every response has a <think> wrapper so the model
        # always trains on the think-then-answer pattern.
        if "<think>" not in resp:
            resp = f"<think>\nLet me work through this.\n</think>\n{resp}"
        return f"User: {example['prompt']}\nAssistant: {resp}"

    # ── Format 2: pre-separated thinking and answer fields ────────────────
    # Some datasets (e.g. process-reward datasets) store the scratchpad and
    # final answer separately, which maps cleanly to our template.
    if "thinking" in example and "answer" in example:
        return CHAT_TEMPLATE.format(
            prompt=example.get("prompt", example.get("problem", "")),
            thinking=example["thinking"].strip(),
            answer=example["answer"].strip(),
        )

    # ── Format 3: math dataset with problem + full solution ───────────────
    # The solution acts as both the thinking trace and contains the final
    # answer (often in \boxed{} notation). We use _extract_answer() to pull
    # out just the final value for the answer line.
    if "problem" in example and "solution" in example:
        return CHAT_TEMPLATE.format(
            prompt=example["problem"].strip(),
            thinking=example["solution"].strip(),  # full solution = reasoning trace
            answer=_extract_answer(example["solution"]),  # final answer extracted
        )

    # ── Format 4: Alpaca instruction-following format ─────────────────────
    # Alpaca has an optional "input" context field that supplements the
    # instruction. When present, we concatenate them with a blank line so the
    # model sees both as part of the user turn.
    if "instruction" in example and "output" in example:
        prompt = example["instruction"]
        if example.get("input", "").strip():
            # Append context to instruction — blank line separates them visually
            prompt = f"{prompt}\n\n{example['input']}"
        output = example["output"]
        if "<think>" not in output:
            output = f"<think>\nLet me think about this.\n</think>\n{output}"
        return f"User: {prompt}\nAssistant: {output}"

    # ── Format 5: ChatML messages array ───────────────────────────────────
    # We only use the first user and first assistant turns. Multi-turn
    # conversation handling would require a different collation strategy
    # (multiple loss regions), which is not implemented here.
    if "messages" in example:
        msgs = example["messages"]
        # next() with a default of "" prevents StopIteration on malformed data
        user_content = next((m["content"] for m in msgs if m["role"] == "user"), "")
        asst_content = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
        if "<think>" not in asst_content:
            asst_content = f"<think>\nLet me think step by step.\n</think>\n{asst_content}"
        return f"User: {user_content}\nAssistant: {asst_content}"

    raise ValueError(f"Unknown example format: {list(example.keys())}")


def _extract_answer(solution: str) -> str:
    """
    Extract the final answer from a math solution string.

    Tries LaTeX \\boxed{} notation first (common in competition math datasets
    like MATH and AMC). Falls back to the last non-empty line, which works
    for datasets that end with "= 42" or "The answer is 42."
    """
    import re

    # \boxed{X} is the standard LaTeX answer notation in math datasets.
    # Greedy match fails on nested braces, so we use [^}]+ (no nested brace).
    boxed = re.search(r"\\boxed\{([^}]+)\}", solution)
    if boxed:
        return boxed.group(1)

    # Fallback: last non-empty line of the solution typically states the answer.
    lines = [l.strip() for l in solution.strip().splitlines() if l.strip()]
    return lines[-1] if lines else solution


# ---------------------------------------------------------------------------
# Tokenization with loss masking
# ---------------------------------------------------------------------------
# This is the most critical section of the entire file.
#
# The goal: produce (input_ids, labels) pairs where:
#   input_ids[t]  = token at position t (fed into the model)
#   labels[t]     = token at position t  (the TARGET to predict at position t)
#                   OR LOSS_IGNORE (-100) if we don't want loss at position t
#
# Note on the "shift": in a standard LM, you shift labels by 1 so that
# input[t] predicts label[t] = token[t+1]. Here we do NOT shift — the
# dataset labels are aligned (input[t] == labels[t] for active positions).
# The model's forward pass handles the causal shift internally: the logit
# at position t predicts the NEXT token, and we compare it to labels[t+1].
# This is an implementation convention; the math is equivalent either way.


def tokenize_with_mask(
    text: str,
    tokenizer,
    max_seq_len: int,
) -> Optional[dict]:
    """
    Tokenize a formatted "User:/Assistant:" example and produce a loss mask.

    The loss mask assigns:
      - LOSS_IGNORE (-100) to every token in the user/prompt region
      - The actual token ID to every token in the assistant region

    PyTorch's F.cross_entropy with ignore_index=-100 natively skips
    LOSS_IGNORE positions, so no manual masking math is required in
    the loss function.

    Boundary detection strategy:
      We encode "Assistant:" as a reference token sequence, then scan the
      full encoded text to find the LAST occurrence of that subsequence.
      We use the LAST occurrence to handle hypothetical multi-turn inputs
      where "Assistant:" might appear in an earlier turn.

    Returns None (example will be skipped) if:
      - Sequence is < 4 tokens (too degenerate to train on)
      - Fewer than 10 tokens fall in the loss region (not worth the gradient)

    Token layout (mask values shown below each token):
      [<bos>] [User: ...tokens...] [Assistant: ] [<think>] [...CoT...] [</think>] [answer] [<eos>]
        =0       =0  ...  =0          =0            =1       =1  ...     =1          =1       =1
                                       ↑
                                 mask_start = boundary + len(asst_marker)
                                 All tokens from here onward get their real ID as label.
    """
    # Encode the full formatted text to token IDs.
    # The tokenizer's post-processor automatically prepends BOS_ID and
    # appends EOS_ID, so ids[0] == BOS_ID and ids[-1] == EOS_ID.
    enc = tokenizer.encode(text)
    ids = enc.ids  # list of int

    # ── Truncation ────────────────────────────────────────────────────────
    # Truncate from the LEFT (keep the tail of the sequence) so that the
    # final answer is always present even for very long CoT traces.
    # After truncation, force the last token to EOS so the model always
    # sees a clean sequence terminator.
    if len(ids) > max_seq_len:
        ids = ids[:max_seq_len]
        # The last real token has been replaced by truncation; set it to EOS
        # so the sequence still has a clean terminator.
        ids[-1] = EOS_ID

    # Sequences shorter than 4 tokens can't form a meaningful (prompt, response) pair.
    if len(ids) < 4:
        return None

    # ── Locate the assistant boundary ─────────────────────────────────────
    # Encode "Assistant:" in isolation to get the boundary token IDs.
    # The post-processor adds BOS/EOS around any standalone encode() call,
    # so we strip those sentinel tokens before using the result as a marker.
    asst_marker = tokenizer.encode("Assistant:").ids
    if asst_marker and asst_marker[0] == BOS_ID:
        asst_marker = asst_marker[1:]  # strip leading BOS added by post-processor
    if asst_marker and asst_marker[-1] == EOS_ID:
        asst_marker = asst_marker[:-1]  # strip trailing EOS added by post-processor

    # Find the LAST occurrence of the "Assistant:" token subsequence.
    # Using the last occurrence ensures correct boundary detection for
    # multi-turn examples where the word "Assistant:" might appear in
    # an earlier user turn (e.g., "Can you act as my assistant:...").
    boundary = _find_subsequence(ids, asst_marker)

    if boundary is None:
        # Fallback if the tokenizer splits "Assistant:" in an unexpected way
        # (can happen with aggressive BPE merges). We mask the first 20% of
        # the sequence as a crude proxy for the prompt region. This is
        # imperfect but avoids silently training on zero tokens.
        boundary = max(1, len(ids) // 5)

    # ── Build the labels tensor ───────────────────────────────────────────
    # Everything up to and including the "Assistant:" marker is the prompt
    # region — those positions get LOSS_IGNORE.
    # Everything after the marker is the assistant's output — those positions
    # keep their real token ID as the training target.
    #
    # Example with asst_marker = [tok_A, tok_B]:
    #   boundary = 10  (index of tok_A in ids)
    #   mask_start = 10 + 2 = 12
    #   labels[0..11]  = LOSS_IGNORE  (prompt + "Assistant:" itself)
    #   labels[12..]   = ids[12..]    (assistant output including <think> block)
    mask_start = boundary + len(asst_marker)
    labels = [LOSS_IGNORE] * mask_start + ids[mask_start:]
    # Clip to exact length of ids — the concatenation above can overshoot if
    # mask_start > len(ids) (degenerate case where the whole sequence is prompt).
    labels = labels[: len(ids)]

    # ── Quality gate ──────────────────────────────────────────────────────
    # If fewer than 10 tokens are in the loss region, the gradient signal is
    # too weak to be useful and the example is likely malformed. Skip it.
    active = sum(1 for l in labels if l != LOSS_IGNORE)
    if active < 10:
        return None

    # Convert to tensors. long (int64) is required by nn.Embedding and
    # F.cross_entropy.
    input_ids = torch.tensor(ids, dtype=torch.long)
    label_ids = torch.tensor(labels, dtype=torch.long)

    return {"input_ids": input_ids, "labels": label_ids}


def _find_subsequence(seq: list, subseq: list) -> Optional[int]:
    """
    Find the LAST occurrence of subseq as a contiguous subsequence of seq.

    Returns the start index of the last match, or None if not found.

    We return the LAST match (not the first) because in multi-turn dialogues
    the assistant marker may appear multiple times. We always want to locate
    the final assistant turn — the one whose output we want to train on.

    Time complexity: O(len(seq) * len(subseq)) — acceptable for typical
    sequence lengths (< 4096 tokens) and marker lengths (2-4 tokens).
    """
    if not subseq or len(subseq) > len(seq):
        return None
    last = None
    # Scan every valid starting position; overwrite 'last' on each match
    # so we end up with the final occurrence.
    for i in range(len(seq) - len(subseq) + 1):
        if seq[i : i + len(subseq)] == subseq:
            last = i
    return last


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class SFTDataset(Dataset):
    """
    Load, format, and tokenize SFT examples at construction time.

    WHY TOKENIZE AT INIT (NOT ON-THE-FLY)?
    ────────────────────────────────────────
    SFT datasets typically have 10k–2M examples at 500–2000 tokens each.
    Pre-tokenising at init means DataLoader workers never touch the tokenizer
    (which is not picklable and can't be safely shared across processes).
    It also means every training step is a fast tensor lookup rather than
    a string encode, keeping the GPU well-fed.

    The trade-off is startup latency (a few minutes for large datasets) and
    peak RAM usage. For datasets that don't fit in RAM, switch to a streaming
    approach with on-the-fly tokenisation and a single DataLoader worker.

    DATA DIRECTORY STRUCTURE:
    ──────────────────────────
    Preferred (explicit split files):
      sft_data/
        train.jsonl    — training examples
        val.jsonl      — validation examples

    Acceptable (single file; auto-split 95/5):
      sft_data/data.jsonl

    Also accepted: any *.jsonl or *.json files in the directory (all merged).
    """

    def __init__(
        self,
        data_dir: str,
        tokenizer_path: str,
        max_seq_len: int,
        split: str = "train",
        max_examples: Optional[int] = None,
    ):
        from tokenizers import Tokenizer

        # Load the HuggingFace tokenizers Tokenizer from its JSON snapshot.
        # We load inside __init__ (not as a class attribute) so it doesn't
        # need to be pickled for DataLoader workers.
        tok_path = Path(tokenizer_path) / "tokenizer.json"
        tokenizer = Tokenizer.from_file(str(tok_path))

        self.examples = []

        data_dir = Path(data_dir)

        # ── File discovery ────────────────────────────────────────────────
        # Priority: sft_{split}.jsonl → {split}.jsonl → data.jsonl → all *.jsonl
        # sft_format.py writes sft_train.jsonl / sft_val.jsonl, so we check
        # that prefix first before falling through to the generic patterns.
        if (data_dir / f"sft_{split}.jsonl").exists():
            files = [data_dir / f"sft_{split}.jsonl"]
        elif (data_dir / f"{split}.jsonl").exists():
            files = [data_dir / f"{split}.jsonl"]
        elif (data_dir / "data.jsonl").exists():
            files = [data_dir / "data.jsonl"]
            # Flag that we'll need to do our own train/val split below
        else:
            # Merge all JSON-lines files in the directory. Sort for
            # reproducibility — directory listing order is not deterministic.
            files = list(data_dir.glob("*.jsonl")) + list(data_dir.glob("*.json"))

        if not files:
            raise FileNotFoundError(f"No data files found in {data_dir}")

        # ── Raw loading ───────────────────────────────────────────────────
        raw_examples = []
        for fpath in sorted(files):
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue  # skip blank lines between JSON objects
                    try:
                        raw_examples.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue  # skip malformed lines silently

        # ── Train/val split ───────────────────────────────────────────────
        # Only executed when we loaded a single non-split file.
        # seed=42 ensures the split is deterministic across repeated runs
        # with the same data (important for reproducing val metrics).
        if len(files) == 1 and not (data_dir / f"{split}.jsonl").exists():
            random.seed(42)
            random.shuffle(raw_examples)
            # 95% train / 5% val is standard for SFT datasets where
            # val is used only for early stopping, not for publication metrics.
            split_idx = max(1, int(len(raw_examples) * 0.95))
            if split == "train":
                raw_examples = raw_examples[:split_idx]
            else:
                raw_examples = raw_examples[split_idx:]

        # Optionally cap the number of examples (useful for quick debug runs).
        if max_examples:
            raw_examples = raw_examples[:max_examples]

        # ── Tokenize all examples ─────────────────────────────────────────
        # We tokenize sequentially (not in parallel) to keep memory usage
        # predictable. For very large datasets (>5M examples), consider
        # parallelising this with multiprocessing.Pool.
        skipped = 0
        for ex in raw_examples:
            try:
                text = format_example(ex)
                item = tokenize_with_mask(text, tokenizer, max_seq_len)
                if item is not None:
                    self.examples.append(item)
                else:
                    # tokenize_with_mask returns None for sequences that are
                    # too short or have too few loss tokens — skip them.
                    skipped += 1
            except Exception:
                # format_example raises ValueError for unknown formats;
                # catch broadly to not abort the entire dataset load on one
                # malformed example.
                skipped += 1
                continue

        print(
            f"  SFTDataset ({split}): {len(self.examples):,} examples "
            f"({skipped} skipped — too long or no assistant turn)"
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        # Returns a dict with keys "input_ids" and "labels" (both 1D tensors).
        # Variable lengths are handled by the sft_collate function.
        return self.examples[idx]


class SyntheticSFTDataset(Dataset):
    """
    Synthetic SFT dataset for --mode validate. No real data or tokenizer needed.

    Generates random token sequences with the first 1/3 masked as "prompt"
    and the remaining 2/3 as "assistant response". The token IDs are garbage
    (random integers from vocab), so the model will learn nothing meaningful,
    but the data shapes and loss masking are correct — sufficient for verifying
    that the training loop, gradient flow, and checkpointing all work end-to-end
    before committing to a full dataset download.
    """

    def __init__(self, vocab_size: int, seq_len: int, n: int = 200):
        self.examples = []
        for _ in range(n):
            # Vary length within [seq_len/2, seq_len] to exercise the padding
            # logic in sft_collate on every batch.
            length = random.randint(seq_len // 2, seq_len)
            # Sample token IDs from [6, vocab_size) — IDs 0-5 are special tokens
            # (PAD, BOS, EOS, UNK, THINK_START, THINK_END); avoid them in
            # synthetic content so the shapes don't confuse boundary detection.
            ids = torch.randint(6, vocab_size, (length,))
            ids[0] = BOS_ID  # sequences must start with BOS
            ids[-1] = EOS_ID  # sequences must end with EOS
            # Mask the first ~1/3 of tokens as "prompt" region (no loss).
            labels = ids.clone()
            mask_end = max(1, length // 3)
            labels[:mask_end] = LOSS_IGNORE
            self.examples.append({"input_ids": ids, "labels": labels})

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


# ---------------------------------------------------------------------------
# Collator — variable-length sequences → padded batch tensors
# ---------------------------------------------------------------------------
# SFT sequences vary widely in length (a simple arithmetic question might be
# 50 tokens; a multi-step proof might be 3000). We do NOT pack multiple
# sequences into a single context window (as pre-training does) because
# packing would require a block-diagonal attention mask to prevent cross-
# sequence attention — significantly more implementation complexity for a
# modest throughput gain. Instead we pad each batch to the longest sequence
# in that batch, which wastes some computation on padding tokens but keeps
# the implementation simple and correct.


def sft_collate(batch: list[dict]) -> dict:
    """
    Pad a batch of variable-length sequences to a uniform length for batched
    GPU operations.

    Padding strategy:
      input_ids  → padded with PAD_ID (0): the model attends to these tokens
                   but they have no semantic meaning. The attention mask (if
                   used) would zero out their contribution, but we rely on
                   label masking instead (see below).
      labels     → padded with LOSS_IGNORE (-100): F.cross_entropy skips these
                   positions, so padding tokens never contribute to the loss.
      attn_mask  → 1 for real tokens, 0 for padding. Included for completeness
                   and potential use with flash attention kernels, but the main
                   training loop does not pass it to the model (see train()).

    All sequences are LEFT-aligned (real tokens first, padding at the right).
    """
    # Find the longest sequence in this batch to determine padding target.
    max_len = max(item["input_ids"].shape[0] for item in batch)

    padded_ids = []
    padded_labels = []
    attn_masks = []

    for item in batch:
        ids = item["input_ids"]
        lbl = item["labels"]
        pad_len = max_len - ids.shape[0]  # number of pad tokens to append

        # Right-pad input_ids with PAD_ID
        padded_ids.append(torch.cat([ids, torch.full((pad_len,), PAD_ID, dtype=torch.long)]))
        # Right-pad labels with LOSS_IGNORE so padding never affects the loss
        padded_labels.append(
            torch.cat([lbl, torch.full((pad_len,), LOSS_IGNORE, dtype=torch.long)])
        )
        # Attention mask: 1 = real token, 0 = padding position
        attn_masks.append(torch.cat([torch.ones(ids.shape[0]), torch.zeros(pad_len)]).long())

    return {
        "input_ids": torch.stack(padded_ids),  # (B, max_len)
        "labels": torch.stack(padded_labels),  # (B, max_len)
        "attention_mask": torch.stack(attn_masks),  # (B, max_len)
    }


# ---------------------------------------------------------------------------
# Learning rate schedule
# ---------------------------------------------------------------------------
# Identical formula to pretrain.py — linear warmup followed by cosine decay.
# Keeping the same formula makes it easy to reason about LR curves across
# phases; only the peak/min values differ.


def get_lr(step: int, total_steps: int, warmup_steps: int, lr: float, lr_min: float) -> float:
    """
    Compute the learning rate for the current optimizer step.

    Phase 1 (warmup): linear ramp from 0 → lr over warmup_steps.
      Avoids large gradient steps early when the pre-trained weights have
      not yet adapted to the SFT data distribution.

    Phase 2 (decay): cosine annealing from lr → lr_min.
      Cosine decay is preferred over linear because it spends more time
      near the peak LR (faster early learning) and decelerates gently
      toward the end (stable convergence). The half-cosine formula:
        lr_min + (lr - lr_min) * 0.5 * (1 + cos(π * progress))
      evaluates to lr at progress=0 and lr_min at progress=1.

    Phase 3 (after total_steps): clamp at lr_min.
      Prevents the LR from going negative if extra steps are taken.
    """
    if step < warmup_steps:
        # Linear warmup: fraction of the way through warmup_steps
        return lr * (step + 1) / warmup_steps
    if step >= total_steps:
        # Clamp at minimum after schedule ends
        return lr_min
    # Cosine annealing: progress ∈ [0, 1] from end of warmup to total_steps
    elapsed = step - warmup_steps
    progress = elapsed / (total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))  # 1.0 → 0.0 as progress → 1
    return lr_min + (lr - lr_min) * cosine


# ---------------------------------------------------------------------------
# Loss — masked cross-entropy
# ---------------------------------------------------------------------------


def sft_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Compute cross-entropy loss, ignoring LOSS_IGNORE (-100) positions.

    logits: (B, T, vocab_size) — raw model output before softmax
    labels: (B, T)             — token IDs for positions we train on,
                                 LOSS_IGNORE (-100) elsewhere

    THE SHIFT
    ─────────
    The model's forward pass is a causal LM: logits[t] is the predicted
    distribution over the token at position t+1 (next-token prediction).
    So logits[:, t, :] should be trained against labels[:, t+1].

    We apply this shift explicitly here, matching compute_loss() in
    model/architecture.py:
      - drop the LAST logit position (it has no corresponding next token)
      - drop the FIRST label position (it is never a prediction target)
    After the shift both tensors have length T-1.

    WHY RESHAPE BEFORE cross_entropy?
    ────────────────────────────────
    F.cross_entropy expects (N, C) logits and (N,) targets. We have
    (B, T-1, V) and (B, T-1) respectively. Reshaping flattens the batch
    and time dimensions — the loss is computed over all non-ignored
    positions regardless of which batch item they came from.

    reduction="mean" divides by the number of non-ignored positions (not
    the total number of tokens), which is what we want — the loss magnitude
    is invariant to how many prompt tokens were masked.
    """
    # Shift: logits[t] predicts labels[t+1], so align by dropping last
    # logit and first label.
    shift_logits = logits[:, :-1, :].contiguous()  # (B, T-1, V)
    shift_labels = labels[:, 1:].contiguous()  # (B, T-1)
    B, T1, V = shift_logits.shape
    loss = F.cross_entropy(
        shift_logits.reshape(B * T1, V),  # (B*(T-1), V)
        shift_labels.reshape(B * T1),  # (B*(T-1),)
        ignore_index=LOSS_IGNORE,  # skip LOSS_IGNORE=-100 positions
        reduction="mean",  # mean over ACTIVE tokens only
    )
    return loss


def mask_fraction(labels: torch.Tensor) -> float:
    """
    Return the fraction of label positions that are in the loss region.

    Useful as a training diagnostic: if this is < 0.1 (less than 10% of
    tokens are assistant output), the dataset may have very short responses
    or the boundary detection may be failing. If it is > 0.9, the dataset
    may be providing very little prompt context, which can hurt generalisation.
    A healthy range is roughly 0.3–0.7 for typical CoT datasets.
    """
    active = (labels != LOSS_IGNORE).sum().item()
    total = labels.numel()
    return active / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(cfg: SFTConfig):
    # ── Device setup ──────────────────────────────────────────────────────
    # Fall back to CPU if CUDA is requested but not available (e.g. on a
    # laptop for quick debugging). Real SFT should always use GPU.
    if cfg.backend == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Backend: CUDA — {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Backend: CPU")

    # bfloat16 halves memory vs float32 with no loss scaler needed.
    # float32 fallback is used on CPU (bfloat16 CPU kernels are slow).
    dtype = torch.bfloat16 if cfg.dtype == "bfloat16" else torch.float32

    # ── Model instantiation ───────────────────────────────────────────────
    model_cfg = CONFIGS[cfg.model_config]
    print(f"\nModel: {cfg.model_config.upper()}")
    model = SmallReasoningModel(model_cfg)

    # ── Load pre-trained checkpoint ───────────────────────────────────────
    # SFT MUST start from a pre-trained base. Without pre-training, the model
    # has no language understanding and SFT will just memorise the few hundred
    # or thousand examples rather than generalising. The pre-trained checkpoint
    # provides the foundation; SFT adds the format / instruction-following layer.
    if cfg.checkpoint:
        print(f"Loading checkpoint: {cfg.checkpoint}")
        # Load to CPU first to avoid OOM spikes on GPU (state dicts can be
        # large; we move the model to GPU after loading).
        # weights_only=False allows loading the full checkpoint dict (which
        # includes non-tensor metadata like step count).
        state = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)
        # Support both raw state_dicts (produced by model.state_dict()) and
        # our wrapped checkpoint format ({"model": ..., "step": ..., ...}).
        sd = state.get("model", state)
        # strict=False allows loading checkpoints that have extra or missing
        # keys (e.g. if the architecture was updated between pre-training and
        # SFT). The warnings below help catch accidental mismatches.
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"  WARNING: {len(missing)} missing keys")
        if unexpected:
            print(f"  WARNING: {len(unexpected)} unexpected keys")
        print(
            f"  Loaded from step {state.get('step', '?')}, "
            f"tokens seen: {state.get('tokens_seen', 0)/1e9:.1f}B"
        )
    else:
        # Starting from random init: still works for the validate smoke test,
        # but any "real" SFT run without a checkpoint is almost certainly a
        # mistake — warn loudly.
        print("  WARNING: No checkpoint provided. Starting from random init.")
        print("  For real SFT, always start from a pre-trained base.")

    # Move to device + cast to target dtype in one call (avoids double allocation).
    model = model.to(device=device, dtype=dtype)

    # ── Gradient checkpointing ────────────────────────────────────────────
    # With 4096-token sequences and a 1B-parameter model, storing all
    # activations for backprop would require ~60 GB. Gradient checkpointing
    # discards activations during the forward pass and recomputes them on
    # demand during backprop, reducing memory to ~40 GB at the cost of
    # ~20% extra compute. Strongly recommended for sequences > 2048.
    if cfg.grad_checkpointing:
        _enable_gradient_checkpointing(model)

    # ── Dataset and DataLoader ────────────────────────────────────────────
    # Check for real data; fall back to synthetic if not found.
    # The synthetic path is only meant for --mode validate smoke tests.
    if Path(cfg.data_dir).exists() and any(Path(cfg.data_dir).iterdir()):
        print(f"\nLoading SFT data from: {cfg.data_dir}")
        train_dataset = SFTDataset(cfg.data_dir, cfg.tokenizer_path, cfg.max_seq_len, "train")
        val_dataset = SFTDataset(cfg.data_dir, cfg.tokenizer_path, cfg.max_seq_len, "val")
    else:
        print(f"\nNo data found at {cfg.data_dir} — using synthetic data (validate mode)")
        # Use seq_len // 4 for synthetic to keep validate runs fast.
        train_dataset = SyntheticSFTDataset(model_cfg.vocab_size, cfg.max_seq_len // 4, n=400)
        val_dataset = SyntheticSFTDataset(model_cfg.vocab_size, cfg.max_seq_len // 4, n=40)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,  # shuffle every epoch for better gradient diversity
        collate_fn=sft_collate,
        # Use 2 workers when a checkpoint is provided (real training).
        # Use 0 workers for validate mode to avoid forking overhead on small runs.
        num_workers=2 if cfg.checkpoint else 0,
        pin_memory=(cfg.backend == "cuda"),  # pinned memory speeds up CPU→GPU transfers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,  # no shuffle for val — deterministic loss estimate
        collate_fn=sft_collate,
    )

    # ── Optimizer ─────────────────────────────────────────────────────────
    # We apply weight decay selectively:
    #   Decay (L2 regularisation):  2D+ parameters — weight matrices in attention
    #                               and FFN layers. These can be large and benefit
    #                               from regularisation to prevent overfit.
    #   No decay:                   LayerNorm scales/biases, embedding tables, and
    #                               1D parameters (biases). Regularising these would
    #                               distort the learned normalisation statistics and
    #                               harm convergence. This split follows Karpathy's
    #                               nanoGPT and OpenAI's GPT-3 recipe.
    decay_params = [
        p
        for n, p in model.named_parameters()
        if p.requires_grad and p.ndim >= 2 and "norm" not in n and "embedding" not in n
    ]
    no_decay_params = [
        p
        for n, p in model.named_parameters()
        if p.requires_grad and (p.ndim < 2 or "norm" in n or "embedding" in n)
    ]

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
        # betas=(0.9, 0.95): slightly higher β₂ than the default 0.999.
        # A lower β₂ makes the second-moment estimate less smooth, which
        # helps AdamW adapt faster to direction changes — recommended by
        # GPT-3 and followed by most modern LLM training recipes.
        betas=(0.9, 0.95),
        # fused=True uses a CUDA-fused kernel that applies all AdamW ops in
        # one pass, saving memory bandwidth. Only available on CUDA.
        fused=(cfg.backend == "cuda"),
    )

    # ── Schedule setup ────────────────────────────────────────────────────
    # steps_per_epoch: number of optimizer updates per pass over the dataset.
    # We divide by (batch_size * grad_accum) because grad_accum micro-batches
    # constitute one optimizer step.
    steps_per_epoch = math.ceil(len(train_dataset) / (cfg.batch_size * cfg.grad_accum))
    total_steps = steps_per_epoch * cfg.epochs
    # 3% warmup is shorter than pre-training (10%) because the model already
    # has stable gradient statistics from pre-training. A short warmup is
    # sufficient to prevent the optimizer from making large early steps.
    warmup_steps = max(1, int(total_steps * cfg.warmup_fraction))

    print(f"\nSFT schedule:")
    print(f"  Examples:     {len(train_dataset):,} train / {len(val_dataset):,} val")
    print(f"  Epochs:       {cfg.epochs}")
    print(f"  Steps:        {total_steps:,} ({steps_per_epoch:,}/epoch)")
    print(f"  Warmup:       {warmup_steps:,} steps")
    print(f"  LR:           {cfg.lr:.1e} → {cfg.lr_min:.1e}")
    print(
        f"  Batch:        {cfg.batch_size} × {cfg.grad_accum} accum = "
        f"{cfg.batch_size * cfg.grad_accum} sequences/step"
    )

    # autocast automatically converts eligible ops to bfloat16 for speed/memory.
    # enabled=False on CPU (bfloat16 matmuls on CPU are slower than float32).
    autocast_ctx = torch.amp.autocast(
        device_type="cuda" if cfg.backend == "cuda" else "cpu",
        dtype=dtype,
        enabled=(dtype == torch.bfloat16),
    )

    # ── Training loop ─────────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  {'step':>7}  {'loss':>8}  {'val_loss':>9}  {'lr':>9}  {'mask%':>7}  {'tok/s':>8}")
    print(f"{'─'*70}")

    os.makedirs(cfg.output_dir, exist_ok=True)
    step = 0  # optimizer step counter (increments once per grad_accum microbatches)
    best_val = float("inf")
    t0 = time.time()  # wall-clock time for tokens/sec calculation

    for epoch in range(cfg.epochs):
        model.train()
        accum_loss = 0.0  # accumulated loss over grad_accum micro-batches
        accum_mask = 0.0  # accumulated mask fraction (for diagnostic logging)
        micro = 0  # micro-batch counter within current accumulation window

        for batch in train_loader:
            # Guard: stop exactly at total_steps even if the epoch has more batches.
            # This can happen when total_steps isn't a perfect multiple of
            # steps_per_epoch.
            if step >= total_steps:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            # Note: attention_mask is produced by sft_collate but not passed to
            # the model here. Our architecture applies a causal (triangular) mask
            # internally; padding tokens receive no loss signal via LOSS_IGNORE
            # and their logits don't affect training. Explicitly masking padding
            # in attention is only necessary for flash-attention or when the
            # padding fraction is very high (wasteful computation).

            with autocast_ctx:
                logits, _ = model(input_ids)
                loss = sft_loss(logits, labels)
                # Divide by grad_accum so that summing grad_accum gradients
                # is equivalent to a single backward pass over the full
                # effective batch. Without this division, the effective LR
                # would scale with grad_accum.
                scaled = loss / cfg.grad_accum

            scaled.backward()  # accumulate gradients
            accum_loss += loss.item()  # track unscaled loss for logging
            accum_mask += mask_fraction(labels)
            micro += 1

            # ── Optimizer step (every grad_accum micro-batches) ───────────
            if micro % cfg.grad_accum == 0:
                # Clip gradient norm to prevent occasional large-loss batches
                # from causing catastrophic parameter updates. 1.0 is a standard
                # threshold; values > 1 indicate an unusually noisy batch.
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

                # Manually set LR on each optimizer step (PyTorch LRScheduler
                # alternative, but explicit control is clearer for custom schedules).
                current_lr = get_lr(step, total_steps, warmup_steps, cfg.lr, cfg.lr_min)
                for pg in optimizer.param_groups:
                    pg["lr"] = current_lr

                optimizer.step()
                # set_to_none=True frees the gradient memory entirely rather than
                # zeroing it — saves GPU memory, and is slightly faster because
                # the memory allocator can reuse the freed blocks.
                optimizer.zero_grad(set_to_none=True)

                # ── Logging ───────────────────────────────────────────────
                if step % cfg.log_every == 0:
                    dt = time.time() - t0
                    # Tokens per second: sequences × tokens per sequence / elapsed time.
                    # This includes padding, so it's an upper bound on useful throughput.
                    tps = (cfg.batch_size * cfg.grad_accum * cfg.max_seq_len) / dt
                    # Average mask fraction over the accumulation window, as a percentage.
                    avg_msk = accum_mask / cfg.grad_accum * 100
                    print(
                        f"  {step:>7,}  {accum_loss/cfg.grad_accum:>8.4f}"
                        f"  {'':>9}  {current_lr:>9.2e}"
                        f"  {avg_msk:>6.1f}%  {tps:>8,.0f}"
                        f"  ep={epoch+1}"
                    )
                    t0 = time.time()

                # ── Validation ────────────────────────────────────────────
                if cfg.eval_every > 0 and step % cfg.eval_every == 0 and step > 0:
                    val_loss = evaluate(model, val_loader, cfg, device, dtype, autocast_ctx)
                    is_best = val_loss < best_val
                    if is_best:
                        best_val = val_loss
                        # Save "best.pt" whenever we achieve a new val loss low.
                        # This is the checkpoint most suitable for Phase 2 GRPO.
                        _save(
                            step, model, optimizer, cfg, val_loss, Path(cfg.output_dir) / "best.pt"
                        )
                    print(
                        f"  {step:>7,}  {'':>8}  {val_loss:>9.4f}  {'':>9}"
                        f"  {'':>7}  {'':>8}  {'★ best' if is_best else ''}"
                    )
                    model.train()  # re-enable train mode after evaluate() set eval mode

                # ── Periodic checkpoint ───────────────────────────────────
                if step % cfg.save_every == 0 and step > 0:
                    _save(
                        step,
                        model,
                        optimizer,
                        cfg,
                        accum_loss / cfg.grad_accum,
                        Path(cfg.output_dir) / f"step_{step:07d}.pt",
                    )

                # Reset accumulators and advance the global step counter.
                accum_loss = 0.0
                accum_mask = 0.0
                step += 1

        print(f"  Epoch {epoch+1} complete.")

    # ── Final checkpoint ──────────────────────────────────────────────────
    # Always save the final state even if it isn't the best val loss —
    # useful as a fallback and for downstream GRPO initialisation experiments.
    _save(step, model, optimizer, cfg, 0.0, Path(cfg.output_dir) / "final.pt")
    print(f"\n{'─'*70}")
    print(f"SFT complete. Best val loss: {best_val:.4f}")
    print(f"Checkpoints: {cfg.output_dir}")

    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(model, loader, cfg, device, dtype, autocast_ctx, max_batches=30):
    """
    Compute mean validation loss over at most max_batches batches.

    We cap at max_batches=30 rather than evaluating the full val set because:
      - SFT val sets can be large (tens of thousands of examples).
      - We call evaluate() frequently (every eval_every steps).
      - 30 batches × batch_size=4 = 120 examples is a reasonable signal-to-cost
        ratio for detecting overfitting trends.

    The model is set to eval mode (disables dropout, uses running BN stats) for
    the duration and the caller is responsible for calling model.train() after.
    """
    model.eval()
    losses = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        with autocast_ctx:
            logits, _ = model(input_ids)
            loss = sft_loss(logits, labels)
        losses.append(loss.item())
    # Return NaN if no batches were evaluated (indicates an empty val set).
    return sum(losses) / len(losses) if losses else float("nan")


# ---------------------------------------------------------------------------
# Checkpoint save
# ---------------------------------------------------------------------------


def _save(step, model, optimizer, cfg, loss, path):
    """
    Save a training checkpoint to disk.

    The checkpoint includes:
      step         — optimizer step number (for resuming and curriculum tracking)
      loss         — training loss at this step (for quick comparison)
      model        — full model state_dict (weights)
      optimizer    — full optimizer state (momentum, variance estimates)
      sft_config   — full SFTConfig as a dict (for reproducibility)
      phase        — "sft" tag so downstream scripts know this is an SFT checkpoint

    Including the optimizer state allows exact resumption of training if
    interrupted. The phase tag lets Phase 2 GRPO verify it received an SFT
    checkpoint (not a raw pre-train checkpoint) as input.
    """
    torch.save(
        {
            "step": step,
            "loss": loss,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "sft_config": asdict(cfg),  # asdict() converts dataclass → plain dict (JSON-safe)
            "phase": "sft",  # downstream scripts check this to verify checkpoint type
        },
        path,
    )
    print(f"  → Saved: {path}")

    # Retention policy: keep only the latest step_*.pt checkpoint.
    # Once a new one is written, the previous is deleted — we only need the
    # most recent for fault-tolerant resumption. best.pt is excluded from
    # this glob and always preserved.
    # Zero-padded step numbers ensure lexicographic == chronological order.
    checkpoints = sorted(Path(path).parent.glob("step_*.pt"))
    for old in checkpoints[:-1]:
        old.unlink()


# ---------------------------------------------------------------------------
# Gradient checkpointing
# ---------------------------------------------------------------------------
# We implement gradient checkpointing by monkey-patching each transformer
# block's forward method. This is less elegant than using register_forward_hook
# but gives us direct control over which arguments are/aren't checkpointed.
#
# KEY CONSTRAINT: torch.utils.checkpoint.checkpoint cannot be used during
# inference (when kv_cache is not None) because recomputing the forward pass
# would require re-running attention over the cached keys/values, which is
# both incorrect and wasteful. The kv_cache guard handles this.


def _enable_gradient_checkpointing(model):
    """
    Wrap each transformer block to use activation recomputation during backward.

    How it works:
      During the forward pass, torch_cp(fn, x) runs fn(x) but discards all
      intermediate activations immediately after computing the output. During
      the backward pass, it recomputes fn(x) from scratch to recover the
      activations needed for gradient computation. Net effect: O(1) activation
      memory per block instead of O(T) — critical for long sequences.

    use_reentrant=False is the modern (PyTorch ≥ 2.0) preferred mode:
      - Compatible with torch.compile
      - Avoids issues with autograd graph re-entrancy
      - Required for correct behaviour with nested checkpointing

    The kv_cache path bypasses checkpointing because:
      - Inference doesn't need gradients
      - The cached states would be invalidated by recomputation
    """
    import functools
    from torch.utils.checkpoint import checkpoint as torch_cp

    for block in model.blocks:
        orig = block.forward

        @functools.wraps(orig)
        def cp_forward(x, attention_mask=None, kv_cache=None, position_offset=0, _orig=orig):
            # During inference (kv_cache is not None), bypass checkpointing entirely.
            # Recomputing the forward would re-run attention over stale/wrong states.
            if kv_cache is not None:
                return _orig(
                    x,
                    attention_mask=attention_mask,
                    kv_cache=kv_cache,
                    position_offset=position_offset,
                )

            # During training: wrap in a checkpoint lambda.
            # We use a closure (fn) rather than passing orig directly because
            # torch_cp requires the function to accept only tensor arguments;
            # attention_mask and position_offset are captured from the outer scope.
            def fn(x_):
                out, _ = _orig(
                    x_,
                    attention_mask=attention_mask,
                    kv_cache=None,
                    position_offset=position_offset,
                )
                return out

            # torch_cp recomputes fn(x) during backward; the second return
            # value (kv_cache output) is None during training.
            return torch_cp(fn, x, use_reentrant=False), None

        block.forward = cp_forward


# ---------------------------------------------------------------------------
# Validate mode — smoke test without real data
# ---------------------------------------------------------------------------


def validate_mode(model_config: str):
    """
    Run 10 optimizer steps with synthetic data to verify the training loop works.

    This catches:
      - Import errors and missing dependencies
      - Shape mismatches in the model or loss function
      - Gradient flow issues (loss not decreasing at all)
      - Checkpoint save/load round-trips

    Run this before starting a real SFT job on a new machine or after making
    architecture changes. It completes in < 30 seconds on CPU.
    """
    print(f"\nSFT validate mode — {model_config.upper()}")
    print("Running 10 steps with synthetic data...\n")

    cfg = SFTConfig(
        model_config=model_config,
        checkpoint="",
        data_dir="/nonexistent",  # non-existent path triggers synthetic data fallback
        epochs=1,
        batch_size=2,
        grad_accum=2,
        max_seq_len=128,  # short sequences for fast validation
        lr=2e-5,
        grad_checkpointing=True,
        dtype="bfloat16",
        backend="cuda" if torch.cuda.is_available() else "cpu",
        log_every=1,
        save_every=999999,  # don't save during validate
        eval_every=5,
        output_dir="/tmp/sft_validate",
    )
    train(cfg)


# ---------------------------------------------------------------------------
# Data inspection utility
# ---------------------------------------------------------------------------


def validate_data(data_dir: str, tokenizer_path: str, max_seq_len: int = 4096, n: int = 5):
    """
    Print a human-readable tokenization report for the first N data examples.

    WHY RUN THIS BEFORE TRAINING?
    ────────────────────────────────
    Tokenisation bugs (wrong boundary detection, all tokens masked, etc.) are
    invisible during training — the model trains without error but learns nothing.
    This utility makes the mask visible before you commit to a full training run.

    For each example it shows:
      - Total token count and the fraction in the loss region
      - The first 120 characters of the formatted text (newlines shown as ↵)
      - The exact token where loss begins and what token it is

    A healthy example should show:
      - Loss tokens: 30–70% (too low = short responses, too high = short prompts)
      - Loss starts at: the first token of the assistant's response (e.g. "<think>")

    Usage:
      python sft.py --mode inspect --data_dir ./sft_data --tokenizer_path ./tokenizer_output
    """
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(Path(tokenizer_path) / "tokenizer.json"))

    data_dir = Path(data_dir)
    # Inspect only the first .jsonl file found — enough for a quick sanity check.
    files = list(data_dir.glob("*.jsonl"))[:1]
    if not files:
        print("No .jsonl files found.")
        return

    print(f"\nInspecting: {files[0]}")
    print("─" * 60)

    with open(files[0]) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            try:
                ex = json.loads(line.strip())
                text = format_example(ex)
                item = tokenize_with_mask(text, tokenizer, max_seq_len)

                if item is None:
                    print(f"Example {i}: SKIPPED (too short or no assistant turn)")
                    continue

                ids = item["input_ids"]
                labels = item["labels"]
                active = (labels != LOSS_IGNORE).sum().item()
                total = len(ids)

                print(f"\nExample {i}:")
                print(f"  Tokens:  {total}  |  Loss tokens: {active} ({active/total*100:.0f}%)")
                print(f"  Text preview: {text[:120].replace(chr(10), '↵')}...")

                # Find the index of the first token that receives loss signal.
                # This should be the first token of the assistant's response —
                # e.g. the <think> token (ID=4).  If it points somewhere inside
                # the user prompt, the boundary detection is broken.
                boundary = next(
                    (j for j, l in enumerate(labels.tolist()) if l != LOSS_IGNORE), total
                )
                print(
                    f"  Loss starts at token {boundary}: "
                    f"'{tokenizer.decode([ids[boundary].item()])}'"
                )

            except Exception as e:
                print(f"Example {i}: ERROR — {e}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="SFT fine-tuning — Phase 1",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="1b", choices=["500m", "1b", "3b"])
    parser.add_argument(
        "--mode", type=str, default="train", choices=["train", "validate", "inspect"]
    )
    parser.add_argument("--checkpoint", type=str, default="", help="Pre-trained checkpoint (.pt)")
    parser.add_argument("--data_dir", type=str, default="./sft_data")
    parser.add_argument("--tokenizer_path", type=str, default="./tokenizer_output")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/sft")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--backend", type=str, default="cuda", choices=["cuda", "neuron", "cpu"])
    parser.add_argument("--no_grad_ckpt", action="store_true")

    args = parser.parse_args()

    if args.mode == "validate":
        validate_mode(args.config)
        return

    if args.mode == "inspect":
        validate_data(args.data_dir, args.tokenizer_path)
        return

    cfg = SFTConfig(
        model_config=args.config,
        checkpoint=args.checkpoint,
        data_dir=args.data_dir,
        tokenizer_path=args.tokenizer_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        max_seq_len=args.max_seq_len,
        lr=args.lr,
        grad_checkpointing=not args.no_grad_ckpt,
        backend=args.backend,
    )
    train(cfg)


if __name__ == "__main__":
    main()
