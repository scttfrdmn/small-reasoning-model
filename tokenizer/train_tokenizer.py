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

VOCAB_SIZE      = 32768          # Tile-aligned (÷128 = 256)
MIN_FREQUENCY   = 2              # Minimum token frequency to include in vocab
INITIAL_ALPHABET_SIZE = 256      # Byte-level fallback covers all Unicode

SPECIAL_TOKENS = [
    "<pad>",      # Padding — ID 0 (first, so padding_idx=0 works everywhere)
    "<bos>",      # Beginning of sequence
    "<eos>",      # End of sequence
    "<unk>",      # Unknown (byte fallback means this should never fire)
    "<think>",    # Begin chain-of-thought block
    "</think>",   # End chain-of-thought block
]

# Token IDs (post-training these will be confirmed by verify())
PAD_TOKEN_ID    = 0
BOS_TOKEN_ID    = 1
EOS_TOKEN_ID    = 2
UNK_TOKEN_ID    = 3
THINK_START_ID  = 4
THINK_END_ID    = 5


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

    The Digits pre-tokenizer with individual_digits=True is the key mechanism.
    It splits on digit/non-digit boundaries AND between every digit.
    "3.14" → ["3", ".", "1", "4"] — decimal point also split (punctuation boundary)
    "1,000" → ["1", ",", "0", "0", "0"] — comma also split
    """
    return PreSequence([
        # Split numbers: each digit becomes its own pre-token
        # individual_digits=True is the critical flag
        Digits(individual_digits=True),
        # Byte-level encoding for full Unicode coverage
        ByteLevel(add_prefix_space=False, use_regex=True),
    ])


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def build_trainer() -> BpeTrainer:
    """
    Configure the BPE trainer.

    Key parameters:
    - vocab_size: 32768 (tile-aligned)
    - min_frequency: 2 (prune tokens that appear only once — noise)
    - special_tokens: must be first in vocab so IDs are deterministic
    - initial_alphabet: ByteLevel.alphabet() — all 256 bytes as base tokens
      This is the byte fallback mechanism. Any byte sequence is representable.
    """
    return BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=ByteLevel.alphabet(),
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
    tokenizer.decoder = decoders.ByteLevel()

    trainer = build_trainer()
    corpus = get_sample_corpus()

    # tokenizers library accepts an iterator of strings
    tokenizer.train_from_iterator(corpus, trainer=trainer)
    _add_post_processor(tokenizer)

    save_tokenizer(tokenizer, output_dir)
    return tokenizer


def train_on_corpus(data_path: str, output_dir: str) -> Tokenizer:
    """
    Train on a real text corpus file.
    File should be UTF-8 text, one document per line (or just raw text).
    For large corpora, use a generator to avoid loading everything into memory.
    """
    print(f"Training on corpus: {data_path}")
    tokenizer = Tokenizer(BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = build_pre_tokenizer()
    tokenizer.decoder = decoders.ByteLevel()

    trainer = build_trainer()

    def file_iterator(path: str) -> Iterator[str]:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line

    tokenizer.train_from_iterator(file_iterator(data_path), trainer=trainer)
    _add_post_processor(tokenizer)

    save_tokenizer(tokenizer, output_dir)
    return tokenizer


def _add_post_processor(tokenizer: Tokenizer):
    """
    Add BOS/EOS wrapping as a post-processor.
    Every sequence is automatically wrapped: <bos> ... <eos>
    This happens transparently at encode time.
    """
    tokenizer.post_processor = TemplateProcessing(
        single="<bos> $A <eos>",
        pair="<bos> $A <eos> $B:1 <eos>:1",
        special_tokens=[
            ("<bos>", tokenizer.token_to_id("<bos>")),
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

    # Save a human-readable config alongside
    vocab_size = tokenizer.get_vocab_size()
    config = {
        "model_type": "bpe",
        "vocab_size": vocab_size,
        "tile_aligned": vocab_size % 128 == 0,
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
    will silently degrade reasoning capability.

    sample_mode: if True, relax vocab size and compression checks
                 (sample corpus is too small to fill 32768 vocab slots).
                 These checks are always enforced on a production tokenizer.
    """
    print("\n" + "="*60)
    print("TOKENIZER VERIFICATION SUITE")
    if sample_mode:
        print("(sample mode: vocab size / compression checks informational only)")
    print("="*60)

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
    for token in SPECIAL_TOKENS:
        tid = tokenizer.token_to_id(token)
        check(
            f"Special token '{token}' exists",
            tid is not None,
            f"ID = {tid}"
        )

    check(
        "<pad> has ID 0",
        tokenizer.token_to_id("<pad>") == 0,
        f"got ID {tokenizer.token_to_id('<pad>')}"
    )

    # --- 3. Digit isolation — the critical policy check ---
    print("\n  [Digit isolation policy]")

    test_cases = [
        ("142",       True,  "3-digit integer"),
        ("3.14",      True,  "decimal number"),
        ("1,000",     True,  "comma-separated number"),
        ("99",        True,  "2-digit integer"),
        ("2026",      True,  "year"),
        ("0.0001234", True,  "small decimal"),
    ]

    for text, expect_split, label in test_cases:
        # Encode without special tokens for this check
        enc = tokenizer.encode(text)
        tokens = enc.tokens

        # Remove byte-level prefix artifacts for digit counting
        # ByteLevel adds 'Ġ' prefix to first token; strip for comparison
        clean_tokens = [t.replace("Ġ", "").replace("▁", "") for t in tokens]

        # Check: every character of a digit sequence should be its own token
        # (or part of a byte-level encoding of a single char)
        digits_in_text = [c for c in text if c.isdigit()]
        digits_in_tokens = []
        for t in clean_tokens:
            for c in t:
                if c.isdigit():
                    digits_in_tokens.append(c)

        # Each digit should appear exactly once and not be merged with neighbors
        all_isolated = True
        for i, d in enumerate(digits_in_text):
            # Find which token contains this digit
            cumpos = 0
            for t in clean_tokens:
                if cumpos + len(t) > i:
                    # This token contains our digit — check it's not merged with
                    # adjacent digits
                    digit_count_in_token = sum(1 for c in t if c.isdigit())
                    if digit_count_in_token > 1:
                        all_isolated = False
                    break
                cumpos += len(t)

        check(
            f"Digits isolated in '{text}' ({label})",
            all_isolated,
            f"tokens: {clean_tokens}"
        )

    # --- 4. Round-trip ---
    print("\n  [Round-trip fidelity]")
    rt_cases = [
        "Hello, world!",
        "The answer is 42.",
        "<think>\n1 + 1 = 2\n</think>\nThe answer is 2.",   # must keep <think> tokens
        "def f(x): return x**2 + 3*x - 1",
        "∫₀¹ x² dx = 1/3",           # Unicode math
        "中文测试",                      # Non-ASCII
        "emoji: 🤔💭",                  # Emoji (byte fallback)
    ]
    bos = tokenizer.token_to_id("<bos>")
    eos = tokenizer.token_to_id("<eos>")

    for text in rt_cases:
        enc = tokenizer.encode(text)
        # Decode keeping all tokens (including <think>/</think>),
        # but strip the auto-added <bos> and <eos> post-processor wrappers.
        ids_no_wrapper = [i for i in enc.ids if i not in (bos, eos)]
        decoded = tokenizer.decode(ids_no_wrapper, skip_special_tokens=False)
        label = f"Round-trip: '{text[:30]}...'" if len(text) > 30 else f"Round-trip: '{text}'"
        check(
            label,
            decoded.strip() == text.strip(),
            f"got='{decoded[:60]}'" if decoded.strip() != text.strip() else ""
        )

    # --- 5. <think> tokens ---
    print("\n  [Chain-of-thought tokens]")
    cot_text = "<think>\nstep 1\nstep 2\n</think>"
    enc = tokenizer.encode(cot_text)
    think_start_id = tokenizer.token_to_id("<think>")
    think_end_id   = tokenizer.token_to_id("</think>")

    check(
        "<think> encodes as single token",
        think_start_id in enc.ids,
        f"<think> ID={think_start_id}, found in {enc.ids[:8]}..."
    )
    check(
        "</think> encodes as single token",
        think_end_id in enc.ids,
        f"</think> ID={think_end_id}"
    )

    # --- 6. No <unk> on common inputs ---
    print("\n  [No unknown tokens on typical inputs]")
    unk_id = tokenizer.token_to_id("<unk>")
    for text in ["Hello world", "x = 3.14", "def f(): pass"]:
        enc = tokenizer.encode(text)
        # <unk> should never appear due to byte fallback
        check(
            f"No <unk> in '{text}'",
            unk_id not in enc.ids
        )

    # --- 7. Compression ratio sanity ---
    print("\n  [Compression ratio]")
    sample = " ".join(get_sample_corpus())
    enc = tokenizer.encode(sample)
    chars = len(sample)
    toks  = len(enc.ids)
    ratio = chars / toks
    check(
        f"Compression ratio ≥ 2.0 chars/token (got {ratio:.2f})",
        ratio >= 2.0,
        "Expected < 2.0 on tiny sample corpus. Will be 3-5x on real corpus.",
        warn_only=sample_mode,
    )

    # --- Summary ---
    print("\n" + "-"*60)
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
    print("="*60 + "\n")

    return failed == 0


# ---------------------------------------------------------------------------
# Demo: show tokenization of interesting examples
# ---------------------------------------------------------------------------

def demo(tokenizer: Tokenizer):
    """Print tokenization of representative examples."""
    print("\nDEMO — Tokenization examples")
    print("-"*60)

    examples = [
        ("Simple text",     "The answer is 42."),
        ("Arithmetic",      "142 + 358 = 500"),
        ("Decimal",         "3.14159 ≈ π"),
        ("Large number",    "1,307,674,368,000"),
        ("Negative float",  "-273.15"),
        ("Code",            "for i in range(100):"),
        ("CoT format",      "<think>\n2+2=4\n</think>"),
        ("Unicode math",    "∑ᵢ xᵢ²"),
    ]

    for label, text in examples:
        enc = tokenizer.encode(text)
        tokens = enc.tokens
        ids    = enc.ids
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
