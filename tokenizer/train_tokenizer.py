"""
train_tokenizer.py
==================
Train a BPE tokenizer for the small reasoning model.

Design constraints (from spec):
  - Vocab size:    32768  (tile-aligned: 32768 / 128 = 256 ✓)
  - Algorithm:     BPE with byte fallback (no unknown tokens)
  - Digit policy:  Individual digits — NEVER merge across digit sequences.
                   "142" → ["1", "4", "2"]. This is critical for arithmetic reasoning.
  - Special tokens: <bos>, <eos>, <pad>, <think>, </think>
  - Number splits:  Never merge across decimal points or commas in numbers.

Why individual digit tokenization is critical for arithmetic:
  Without digit isolation, BPE learns multi-digit tokens like "142", "358", "500".
  When the model is asked "142 + 358", it sees three opaque tokens rather than
  six individual digit tokens. Carrying, column addition, and multi-step arithmetic
  all require reasoning about each digit's place value independently.
  With individual_digits=True, "142" → ["1", "4", "2"] — the model can apply
  learned positional arithmetic procedures digit-by-digit, just as humans do.
  This is a deliberate architectural choice, not a minor tuning detail.

Usage:
  # Train on a small sample (testing / validation):
  python train_tokenizer.py --mode sample --output ./tokenizer_output

  # Train on a real corpus (production):
  python train_tokenizer.py --mode corpus --data path/to/corpus.txt --output ./tokenizer_output

  # Verify a saved tokenizer:
  python train_tokenizer.py --mode verify --tokenizer ./tokenizer_output
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Iterator

from tokenizers import Tokenizer, pre_tokenizers, decoders, trainers
from tokenizers.models import BPE
from tokenizers.normalizers import NFD, Lowercase, StripAccents, Sequence
from tokenizers.pre_tokenizers import (
    ByteLevel,
    Digits,
    Punctuation,
    Split,
    Sequence as PreSequence,
)
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VOCAB_SIZE = (
    32768  # Tile-aligned (÷128 = 256); every embedding matrix row count is a multiple of 128
)
MIN_FREQUENCY = 2  # Minimum token frequency to include in vocab; prunes hapax tokens that are noise
INITIAL_ALPHABET_SIZE = 256  # Byte-level fallback covers all Unicode (256 possible byte values)

# Special tokens are declared in priority order.
# The ORDER here is not arbitrary — they are added to the vocabulary FIRST,
# which guarantees that <pad>=0, <bos>=1, <eos>=2, etc. are stable across
# runs and corpus sizes. If special tokens were added after BPE merges, their
# IDs would shift depending on how many merge rules fired, making checkpoints
# incompatible with each other.
SPECIAL_TOKENS = [
    "<pad>",  # Padding — ID 0 (first, so padding_idx=0 works everywhere)
    "<bos>",  # Beginning of sequence — wraps every encoded text on the left
    "<eos>",  # End of sequence — wraps every encoded text on the right
    "<unk>",  # Unknown (byte fallback means this should never fire in practice)
    "<think>",  # Begin chain-of-thought block (GRPO format)
    "</think>",  # End chain-of-thought block (GRPO format)
]

# Token IDs (post-training these will be confirmed by verify())
# These constants mirror the SPECIAL_TOKENS list order above.
PAD_TOKEN_ID = 0
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2
UNK_TOKEN_ID = 3
THINK_START_ID = 4
THINK_END_ID = 5


# ---------------------------------------------------------------------------
# Pre-tokenizer: the digit policy lives here
# ---------------------------------------------------------------------------


def build_pre_tokenizer():
    """
    Pre-tokenization pipeline that enforces the digit isolation policy.

    Pipeline (applied left to right):
      1. Split on whitespace and punctuation boundaries — standard wordpiece-style split
      2. Digits: isolate each digit as its own pre-token (NEVER_SPLIT between digits)
         This means "142" becomes ["1", "4", "2"] before BPE sees it.
         Critical: BPE cannot then merge digits back because they're separate pre-tokens.
      3. ByteLevel: convert each pre-token to byte representation for fallback.
         This ensures no <unk> tokens — every Unicode character has a byte encoding.

    Why individual_digits=True is the key mechanism:
      The Digits pre-tokenizer, when individual_digits=True, splits on EVERY
      digit boundary — not just digit/non-digit transitions but between consecutive
      digits too. Without this flag, "142" stays as a single pre-token, allowing
      BPE to learn a "142" merge. With this flag, "1", "4", "2" are hard splits
      that BPE cannot undo.

    Why ByteLevel guarantees no <unk>:
      ByteLevel re-encodes each pre-token character-by-character into printable
      ASCII surrogates for all 256 possible byte values (Ġ, Ā, ā, ...). Because
      every byte value has a dedicated symbol, and any Unicode string can be
      decomposed into bytes, no input can produce an unrepresentable sequence.
      The <unk> token defined in the vocabulary will never appear in practice.

    "3.14" → ["3", ".", "1", "4"] — decimal point also split (punctuation boundary)
    "1,000" → ["1", ",", "0", "0", "0"] — comma also split
    """
    return PreSequence(
        [
            # Digits(individual_digits=True): each digit becomes its own pre-token.
            # This is the critical flag for arithmetic reasoning — see module docstring.
            # Without individual_digits=True, multi-digit numbers stay as single pre-tokens
            # and BPE learns opaque tokens like "142", destroying digit-level reasoning.
            Digits(individual_digits=True),
            # ByteLevel: re-encode each pre-token using 256-symbol byte alphabet.
            # add_prefix_space=False: do not insert Ġ before word-initial tokens;
            # use_regex=True: apply GPT-2-style whitespace-preserving regex split first.
            # This guarantees full Unicode coverage with zero unknown tokens.
            ByteLevel(add_prefix_space=False, use_regex=True),
        ]
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def build_trainer() -> BpeTrainer:
    """
    Configure the BPE trainer.

    Key parameters:
    - vocab_size: 32768 (tile-aligned; must be divisible by 128 for Trainium2 NeuronCore)
    - min_frequency: 2 (prune tokens that appear only once — noise reduction)
    - special_tokens: must be first in vocab so IDs are deterministic across corpora.
      If omitted or listed after regular tokens, IDs depend on corpus frequencies
      and change between training runs, breaking checkpoint compatibility.
    - initial_alphabet: ByteLevel.alphabet() — all 256 bytes as base tokens.
      This seeds the vocabulary with all possible byte values BEFORE any BPE
      merges happen, ensuring the byte fallback is always available. Without this,
      rare Unicode bytes might not appear in the training corpus and would be absent
      from the vocabulary, causing <unk> emissions.
    """
    return BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        special_tokens=SPECIAL_TOKENS,  # added first → deterministic IDs (see SPECIAL_TOKENS note)
        initial_alphabet=ByteLevel.alphabet(),  # all 256 bytes → no <unk> possible
        show_progress=True,
    )


def get_sample_corpus() -> list[str]:
    """
    A minimal but representative sample corpus for testing.
    Covers: general text, code, math (integer, decimal, expression),
    edge cases (large numbers, commas in numbers, multi-digit sequences),
    and the <think></think> format.

    In production, replace with a real pre-tokenized corpus iterator.
    """
    return [
        # General text
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models learn from data to make predictions.",
        "Attention mechanisms allow models to focus on relevant parts of the input.",
        # Math — the critical test domain
        "What is 142 + 358? Let me think step by step.",
        "3.14159 is approximately equal to pi.",
        "The answer is 1,000,000 dollars.",
        "Solve: 2x² + 3x - 5 = 0 using the quadratic formula.",
        "If f(x) = x³ - 2x² + x - 1, find f'(x).",
        "The probability is 0.0001234 which equals 1.234 × 10⁻⁴.",
        "Compute 99 × 99 = 9801. Check: (100-1)² = 10000 - 200 + 1 = 9801.",
        "The 17th prime number is 59.",
        "2 + 2 = 4, 3 + 3 = 6, 100 + 100 = 200.",
        # Chain-of-thought format — the <think> tokens must be stable
        "<think>\nLet me work through this step by step.\n1. First, identify the problem.\n2. Then, apply the formula.\n3. Finally, verify the answer.\n</think>\nThe answer is 42.",
        "User: What is 15 factorial?\nAssistant: <think>\n15! = 15 × 14 × 13 × ... × 1\n= 1307674368000\n</think>\n1,307,674,368,000",
        # Code — important for Phase 2 GRPO
        "def fibonacci(n: int) -> int:\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        "for i in range(0, 100, 2):\n    print(f'{i}: {i**2}')",
        "x = [1, 2, 3, 4, 5]\nresult = sum(x[i]*x[i] for i in range(len(x)))",
        # Digit edge cases
        "The year 2026 is 2026 years after 0 AD.",
        "Room 42, Floor 7, Building 3A.",
        "192.168.1.1 is a common local IP address.",
        "SHA256: a3f5b2c1d4e6... (truncated)",
        "Temperature: -273.15°C is absolute zero.",
        "Price: $1,299.99 after 15% discount from $1,529.41",
        # Repeated patterns (BPE learns from frequency)
        " ".join(["the"] * 50),
        " ".join(["and"] * 50),
        " ".join([str(i) for i in range(100)]),
    ]


def train_on_sample(output_dir: str) -> Tokenizer:
    """Train on the built-in sample corpus. For testing and validation."""
    print("Training on sample corpus...")
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = build_pre_tokenizer()
    tokenizer.decoder = (
        decoders.ByteLevel()
    )  # mirrors ByteLevel pre-tokenizer; needed to reconstruct text

    trainer = build_trainer()
    corpus = get_sample_corpus()

    # tokenizers library accepts an iterator of strings; processes lazily if given a generator
    tokenizer.train_from_iterator(corpus, trainer=trainer)
    _add_post_processor(tokenizer)

    save_tokenizer(tokenizer, output_dir)
    return tokenizer


def train_on_corpus(data_path: str, output_dir: str) -> Tokenizer:
    """
    Train on a real text corpus file.

    Accepts two formats, auto-detected by the first non-empty line:
      - JSONL: each line is {"text": "...", ...} — the output format of data/preprocess.py.
              Only the "text" field is used; other fields (source, etc.) are ignored.
      - Plain text: UTF-8, one document per line (or raw continuous text).

    Why JSONL detection: the preprocess.py pipeline produces JSONL, and feeding
    that directly to the tokenizer avoids an intermediate extraction step and
    keeps the pipeline simple. Detection is by trying json.loads() on the first
    non-blank line rather than checking the file extension, which is more robust.

    Memory usage is O(1) in corpus size — only one line is live at a time.
    This matters at scale: a 50GB corpus would require 50GB of RAM if fully
    loaded, but with a generator it uses only a few KB.
    """
    print(f"Training on corpus: {data_path}")
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = build_pre_tokenizer()
    tokenizer.decoder = decoders.ByteLevel()  # must match the ByteLevel pre-tokenizer above

    trainer = build_trainer()

    def file_iterator(path: str) -> Iterator[str]:
        # Auto-detect JSONL vs plain text by peeking at the first non-empty line.
        # We re-open after peeking to avoid seeking, which isn't supported on all
        # stream types (though for files it would work fine).
        is_jsonl = False
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        obj = json.loads(line)
                        is_jsonl = isinstance(obj, dict) and "text" in obj
                    except (json.JSONDecodeError, ValueError):
                        is_jsonl = False
                    break

        fmt = "JSONL (extracting 'text' field)" if is_jsonl else "plain text"
        print(f"  Corpus format: {fmt}")

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if is_jsonl:
                    try:
                        text = json.loads(line).get("text", "")
                    except (json.JSONDecodeError, ValueError):
                        continue  # skip malformed lines rather than crashing
                    if text:
                        yield text
                else:
                    yield line

    tokenizer.train_from_iterator(file_iterator(data_path), trainer=trainer)
    _add_post_processor(tokenizer)

    save_tokenizer(tokenizer, output_dir)
    return tokenizer


def _add_post_processor(tokenizer: Tokenizer):
    """
    Add BOS/EOS wrapping as a post-processor.
    Every sequence is automatically wrapped: <bos> ... <eos>
    This happens transparently at encode time — callers never need to prepend/append manually.

    The TemplateProcessing syntax:
      "$A" is a placeholder for the first (or only) input sequence.
      "$B" is a placeholder for the second sequence in a pair (e.g. NLI premise/hypothesis).
      ":1" suffix on $B:1 assigns segment ID 1 to all tokens of the second sequence
      (segment ID 0 is implicit for $A tokens). This mirrors BERT's segment embeddings.

      "single" template covers single-sequence inputs (our primary use case):
        <bos> [all tokens of input A] <eos>

      "pair" template covers two-sequence inputs (not currently used, but defined for completeness):
        <bos> [sequence A] <eos> [sequence B] <eos>

    The special_tokens list maps each template literal to its vocabulary ID so the
    post-processor can inject actual token IDs rather than string names.
    """
    tokenizer.post_processor = TemplateProcessing(
        single="<bos> $A <eos>",  # $A expands to all token IDs of the encoded input
        pair="<bos> $A <eos> $B:1 <eos>:1",  # $B:1 = sequence B with segment_id=1
        special_tokens=[
            ("<bos>", tokenizer.token_to_id("<bos>")),  # resolve string → integer ID
            ("<eos>", tokenizer.token_to_id("<eos>")),
        ],
    )


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------


def save_tokenizer(tokenizer: Tokenizer, output_dir: str):
    """Save tokenizer + metadata config."""
    os.makedirs(output_dir, exist_ok=True)
    out = Path(output_dir)

    # Save the tokenizer itself (HuggingFace tokenizers JSON format)
    tokenizer_path = out / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    print(f"Saved tokenizer → {tokenizer_path}")

    # Save a human-readable config alongside — useful for auditing and loading into HF AutoTokenizer
    vocab_size = tokenizer.get_vocab_size()
    config = {
        "model_type": "bpe",
        "vocab_size": vocab_size,
        "tile_aligned": vocab_size % 128 == 0,  # must be True for Trainium2 tile efficiency
        "tile_factor": vocab_size // 128,
        "special_tokens": {t: tokenizer.token_to_id(t) for t in SPECIAL_TOKENS},
        "digit_policy": "individual_digits=True — each digit is a separate pre-token",
        "byte_fallback": True,
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "pad_token": "<pad>",
        "unk_token": "<unk>",
        "think_start_token": "<think>",
        "think_end_token": "</think>",
        "spec_version": "0.1",
    }
    config_path = out / "tokenizer_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config    → {config_path}")


def load_tokenizer(tokenizer_dir: str) -> Tokenizer:
    """Load a saved tokenizer."""
    path = Path(tokenizer_dir) / "tokenizer.json"
    return Tokenizer.from_file(str(path))


# ---------------------------------------------------------------------------
# Verification suite
# ---------------------------------------------------------------------------


def verify(tokenizer: Tokenizer, sample_mode: bool = False):
    """
    Run the verification suite. All checks must pass before training.

    This is not optional. A tokenizer that fails any of these checks
    will silently degrade reasoning capability:
      - Wrong vocab size → embedding matrix not tile-aligned → Trainium2 padding overhead
      - Wrong special token IDs → model writes to wrong embedding rows → garbage output
      - Digit merges present → arithmetic reasoning capability destroyed
      - Round-trip failure → model learns to decode text incorrectly
      - <unk> emissions → byte fallback broken → coverage holes in vocabulary

    sample_mode: if True, relax vocab size and compression checks
                 (sample corpus is too small to fill 32768 vocab slots).
                 These checks are always enforced on a production tokenizer.
    """
    print("\n" + "=" * 60)
    print("TOKENIZER VERIFICATION SUITE")
    if sample_mode:
        print("(sample mode: vocab size / compression checks informational only)")
    print("=" * 60)

    passed = 0
    failed = 0
    warned = 0

    def check(name: str, condition: bool, detail: str = "", warn_only: bool = False):
        nonlocal passed, failed, warned
        if condition:
            status = "✓ PASS"
            passed += 1
        elif warn_only:
            status = "⚠ WARN"
            warned += 1
        else:
            status = "✗ FAIL"
            failed += 1
        print(f"  {status}  {name}")
        if detail:
            print(f"         {detail}")

    # --- 1. Vocab size ---
    # Hard requirement: vocab_size must be 32768 and divisible by 128.
    # The model's embedding and LM-head weight matrices are (vocab_size, d_model).
    # On Trainium2, matrix dimensions must be multiples of 128 for full tile utilization.
    # A non-aligned vocab size forces the NeuronCore to pad, wasting SBUF bandwidth.
    vocab_size = tokenizer.get_vocab_size()
    # In sample mode these are warnings — corpus too small to reach 32768 merges.
    # In production these are hard failures.
    check(
        f"Vocab size is {VOCAB_SIZE}",
        vocab_size == VOCAB_SIZE,
        f"got {vocab_size} (expected with small corpus in sample mode)",
        warn_only=sample_mode,
    )
    check(
        "Vocab size is tile-aligned (÷128)",
        vocab_size % 128 == 0,
        f"{vocab_size} / 128 = {vocab_size / 128:.3f}",
        warn_only=sample_mode,
    )

    # --- 2. Special token IDs ---
    # Hard requirement: every special token must resolve to a known, stable ID.
    # The model's forward pass uses these IDs directly (e.g., loss masking on PAD_TOKEN_ID=0).
    # An absent or wrongly-IDed special token causes silent training bugs.
    for token in SPECIAL_TOKENS:
        tid = tokenizer.token_to_id(token)
        check(f"Special token '{token}' exists", tid is not None, f"ID = {tid}")

    # <pad> must be ID 0 specifically — most frameworks use padding_idx=0 by default,
    # and the loss function masks out pad positions by checking id == 0.
    check(
        "<pad> has ID 0",
        tokenizer.token_to_id("<pad>") == 0,
        f"got ID {tokenizer.token_to_id('<pad>')}",
    )

    # --- 3. Digit isolation — the critical policy check ---
    # Hard requirement: no single token may contain more than one digit character.
    # If this fires, the digit isolation pre-tokenizer has been misconfigured
    # (e.g., individual_digits=False) or a later BPE merge has recombined digits.
    # BPE cannot merge across pre-token boundaries, so if digits are proper pre-tokens
    # this check should always pass.
    print("\n  [Digit isolation policy]")

    test_cases = [
        ("142", True, "3-digit integer"),
        ("3.14", True, "decimal number"),
        ("1,000", True, "comma-separated number"),
        ("99", True, "2-digit integer"),
        ("2026", True, "year"),
        ("0.0001234", True, "small decimal"),
    ]

    for text, expect_split, label in test_cases:
        # Encode without special tokens for this check
        enc = tokenizer.encode(text)
        tokens = enc.tokens

        # Remove byte-level prefix artifacts for digit counting.
        # ByteLevel.encode() prepends 'Ġ' (U+0120) to tokens that follow whitespace
        # and '▁' (U+2581) in some configurations. Strip both so digit counting works
        # on the underlying character content.
        clean_tokens = [t.replace("Ġ", "").replace("▁", "") for t in tokens]

        # Check: every character of a digit sequence should be its own token
        # (or part of a byte-level encoding of a single char)
        digits_in_text = [c for c in text if c.isdigit()]
        digits_in_tokens = []
        for t in clean_tokens:
            for c in t:
                if c.isdigit():
                    digits_in_tokens.append(c)

        # Each digit should appear exactly once and not be merged with neighbors.
        # Walk through source digit positions and find which clean token contains each.
        # If any token contains more than one digit, isolation has failed.
        all_isolated = True
        for i, d in enumerate(digits_in_text):
            # Find which token contains this digit
            cumpos = 0
            for t in clean_tokens:
                if cumpos + len(t) > i:
                    # This token contains our digit — check it's not merged with
                    # adjacent digits (i.e., no other digit characters in this token)
                    digit_count_in_token = sum(1 for c in t if c.isdigit())
                    if digit_count_in_token > 1:
                        all_isolated = False
                    break
                cumpos += len(t)

        check(f"Digits isolated in '{text}' ({label})", all_isolated, f"tokens: {clean_tokens}")

    # --- 4. Round-trip ---
    # Hard requirement: decode(encode(text)) == text for all representable inputs.
    # A round-trip failure means the tokenizer is lossy — the model can never learn
    # to reproduce certain text exactly, corrupting math answers and code output.
    # The byte fallback makes this possible for all inputs, including emoji and CJK.
    print("\n  [Round-trip fidelity]")
    rt_cases = [
        "Hello, world!",
        "The answer is 42.",
        "<think>\n1 + 1 = 2\n</think>\nThe answer is 2.",  # must keep <think> tokens intact
        "def f(x): return x**2 + 3*x - 1",
        "∫₀¹ x² dx = 1/3",  # Unicode math
        "中文测试",  # Non-ASCII (multi-byte UTF-8)
        "emoji: 🤔💭",  # Emoji (byte fallback required: 4-byte UTF-8 sequences)
    ]
    bos = tokenizer.token_to_id("<bos>")
    eos = tokenizer.token_to_id("<eos>")

    for text in rt_cases:
        enc = tokenizer.encode(text)
        # Decode keeping all tokens (including <think>/</think>),
        # but strip the auto-added <bos> and <eos> post-processor wrappers.
        # We compare against the original text without those framing tokens.
        ids_no_wrapper = [i for i in enc.ids if i not in (bos, eos)]
        decoded = tokenizer.decode(ids_no_wrapper, skip_special_tokens=False)
        label = f"Round-trip: '{text[:30]}...'" if len(text) > 30 else f"Round-trip: '{text}'"
        check(
            label,
            decoded.strip() == text.strip(),
            f"got='{decoded[:60]}'" if decoded.strip() != text.strip() else "",
        )

    # --- 5. <think> tokens ---
    # Hard requirement: <think> and </think> must each encode as a single token ID,
    # not as a multi-token sequence. During GRPO training the reward function checks
    # for the <think>...</think> pattern by token ID. If these tokens split into
    # multiple pieces, format_reward() will fail to detect the CoT block.
    print("\n  [Chain-of-thought tokens]")
    cot_text = "<think>\nstep 1\nstep 2\n</think>"
    enc = tokenizer.encode(cot_text)
    think_start_id = tokenizer.token_to_id("<think>")
    think_end_id = tokenizer.token_to_id("</think>")

    check(
        "<think> encodes as single token",
        think_start_id in enc.ids,
        f"<think> ID={think_start_id}, found in {enc.ids[:8]}...",
    )
    check(
        "</think> encodes as single token", think_end_id in enc.ids, f"</think> ID={think_end_id}"
    )

    # --- 6. No <unk> on common inputs ---
    # Hard requirement (on a production tokenizer): byte fallback must prevent <unk> entirely.
    # If <unk> appears here, it means ByteLevel.alphabet() was not passed to BpeTrainer
    # as initial_alphabet, and some byte values are missing from the vocabulary.
    # In sample mode, a tiny corpus may not exercise all paths — this check is still
    # informative but less critical.
    print("\n  [No unknown tokens on typical inputs]")
    unk_id = tokenizer.token_to_id("<unk>")
    for text in ["Hello world", "x = 3.14", "def f(): pass"]:
        enc = tokenizer.encode(text)
        # <unk> should never appear due to byte fallback — all bytes are in the vocabulary
        check(f"No <unk> in '{text}'", unk_id not in enc.ids)

    # --- 7. Compression ratio sanity ---
    # Informational (warn in sample mode, fail in production): measures tokenizer efficiency.
    # A real BPE trained on natural language achieves ~3-5 chars/token. Falling below 2.0
    # means the tokenizer is barely merging anything — the model will process sequences
    # ~2× longer than necessary, increasing compute and reducing effective context length.
    print("\n  [Compression ratio]")
    sample = " ".join(get_sample_corpus())
    enc = tokenizer.encode(sample)
    chars = len(sample)
    toks = len(enc.ids)
    ratio = chars / toks
    check(
        f"Compression ratio ≥ 2.0 chars/token (got {ratio:.2f})",
        ratio >= 2.0,
        "Expected < 2.0 on tiny sample corpus. Will be 3-5x on real corpus.",
        warn_only=sample_mode,
    )

    # --- Summary ---
    print("\n" + "-" * 60)
    print(f"  Result: {passed} passed, {warned} warned, {failed} failed")
    if failed == 0:
        if warned > 0:
            print(f"  ✓ No hard failures. {warned} warnings are expected in sample mode.")
            print("    Re-run verification on a production-trained tokenizer before training.")
        else:
            print("  ✓ All checks passed. Tokenizer is ready for training.")
    else:
        print("  ✗ FAILURES DETECTED. Fix before proceeding.")
        print("    A broken tokenizer will silently degrade reasoning capability.")
    print("=" * 60 + "\n")

    return failed == 0


# ---------------------------------------------------------------------------
# Demo: show tokenization of interesting examples
# ---------------------------------------------------------------------------


def demo(tokenizer: Tokenizer):
    """Print tokenization of representative examples."""
    print("\nDEMO — Tokenization examples")
    print("-" * 60)

    examples = [
        ("Simple text", "The answer is 42."),
        ("Arithmetic", "142 + 358 = 500"),
        ("Decimal", "3.14159 ≈ π"),
        ("Large number", "1,307,674,368,000"),
        ("Negative float", "-273.15"),
        ("Code", "for i in range(100):"),
        ("CoT format", "<think>\n2+2=4\n</think>"),
        ("Unicode math", "∑ᵢ xᵢ²"),
    ]

    for label, text in examples:
        enc = tokenizer.encode(text)
        tokens = enc.tokens
        ids = enc.ids
        print(f"\n  {label}: {repr(text)}")
        print(f"  tokens ({len(tokens)}): {tokens}")
        print(f"  ids:    {ids}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Train or verify the small reasoning model tokenizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["sample", "corpus", "verify", "demo"],
        default="sample",
        help="sample: train on built-in corpus | corpus: train on file | verify: check saved tokenizer | demo: show examples",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to training corpus (required for --mode corpus)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./tokenizer_output",
        help="Output directory for trained tokenizer",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Path to saved tokenizer directory (for --mode verify/demo)",
    )

    args = parser.parse_args()

    if args.mode == "sample":
        tokenizer = train_on_sample(args.output)
        ok = verify(tokenizer, sample_mode=True)
        if ok:
            demo(tokenizer)

    elif args.mode == "corpus":
        if not args.data:
            print("Error: --data required for --mode corpus")
            sys.exit(1)
        tokenizer = train_on_corpus(args.data, args.output)
        ok = verify(tokenizer)
        if ok:
            demo(tokenizer)

    elif args.mode == "verify":
        path = args.tokenizer or args.output
        print(f"Loading tokenizer from {path}...")
        tokenizer = load_tokenizer(path)
        verify(tokenizer)
        demo(tokenizer)

    elif args.mode == "demo":
        path = args.tokenizer or args.output
        tokenizer = load_tokenizer(path)
        demo(tokenizer)


if __name__ == "__main__":
    main()
