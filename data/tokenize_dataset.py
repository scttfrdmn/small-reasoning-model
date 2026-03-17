"""
tokenize_dataset.py
===================
Pre-tokenize a JSONL corpus into a flat binary file of uint16 token IDs.

Why pre-tokenize instead of tokenizing during training:
  On-the-fly tokenization during training has two problems:
    1. The HuggingFace tokenizers library uses a Rust rayon thread pool internally.
       When rayon's futex-based work-stealing interacts with Python's GIL and PyTorch's
       own background CUDA threads, it can produce a deadlock: the main Python thread
       blocks on futex_do_wait indefinitely while the GPU runs CUDA kernels already
       queued but no new batches are ever submitted.
    2. Tokenization CPU cost (~1-5 ms per document) adds up — with a 39 GB corpus and
       per-document tokenization, you spend more time on the CPU than necessary.

  Pre-tokenizing once writes a single flat binary file:
    token_ids: [t0, t1, t2, ..., tN]  — uint16, N ≈ 10B
  Training then does simple numpy/mmap reads, which are:
    - Safe: no Rust thread pool, no tokenizer object loaded during training
    - Fast: mmap read for a 2048-token chunk ≈ 4 μs vs tokenization ≈ 1000 μs
    - Reproducible: same token sequence every run (deterministic curriculum)

Binary format:
  - NumPy uint16 array, C-contiguous, written with np.save (generates .npy header)
  - Header magic: numpy .npy format, shape=(N,), dtype=uint16
  - Values: raw token IDs in corpus order
  - Token IDs fit in uint16: vocab_size=32768 < 65535 ✓

  Why uint16 vs uint32:
    uint16 halves storage (10B tokens × 2 bytes = 20 GB vs 40 GB).
    vocab_size=32768 ≤ 65535 (max uint16) so no precision loss.

Output files:
  <output_dir>/train.bin  — tokenized training split (~95% of documents)
  <output_dir>/val.bin    — tokenized validation split (~5% of documents)
  <output_dir>/meta.json  — total token counts, vocab size, split sizes

Usage:
  python data/tokenize_dataset.py \\
    --input  data/pretrain/train.jsonl \\
    --tokenizer tokenizer_output \\
    --output_dir data/pretrain_tokenized

  # Then train with:
  uv run srm-pretrain --config 500m --data_path data/pretrain_tokenized
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------


def load_tokenizer(tokenizer_path: str):
    """
    Load the trained tokenizer.

    We import here (not at module level) so the rest of the script can run
    even when the tokenizers library isn't installed.  The tokenizer is
    loaded ONCE in the main process and never used again after this script
    finishes — eliminating the rayon/GIL deadlock that occurs when
    tokenizers is called from multiple threads during training.
    """
    try:
        from tokenizers import Tokenizer
    except ImportError:
        print("ERROR: tokenizers library not installed. Run: uv sync", file=sys.stderr)
        sys.exit(1)

    tok_file = Path(tokenizer_path) / "tokenizer.json"
    if not tok_file.exists():
        print(f"ERROR: tokenizer not found at {tok_file}", file=sys.stderr)
        sys.exit(1)

    tokenizer = Tokenizer.from_file(str(tok_file))
    print(f"Tokenizer loaded: vocab_size={tokenizer.get_vocab_size()}")
    return tokenizer


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def iter_jsonl(path: str):
    """Yield text strings from a JSONL file with {"text": "..."} per line."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                text = obj.get("text", "")
                if text:
                    yield text
            except (json.JSONDecodeError, ValueError):
                continue


def count_lines(path: str) -> int:
    """Count non-empty lines in a JSONL file (for tqdm total)."""
    n = 0
    with open(path, "rb") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


# ---------------------------------------------------------------------------
# Main tokenization loop
# ---------------------------------------------------------------------------


def tokenize_corpus(
    input_path: str,
    tokenizer_path: str,
    output_dir: str,
    val_fraction: float = 0.05,
    chunk_size: int = 100_000,
) -> None:
    """
    Tokenize the entire JSONL corpus and write train.bin + val.bin.

    Args:
        input_path:    Path to the JSONL file (one {"text":...} per line).
        tokenizer_path: Directory containing tokenizer.json.
        output_dir:    Where to write train.bin, val.bin, meta.json.
        val_fraction:  Fraction of documents to hold out for validation.
        chunk_size:    Number of tokens to accumulate before flushing to disk.
                       Larger = fewer disk writes = faster, but uses more RAM.
                       100K tokens × 2 bytes = 200 KB — very small.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.bin"
    val_path = output_dir / "val.bin"
    meta_path = output_dir / "meta.json"

    tokenizer = load_tokenizer(tokenizer_path)

    # We write to .bin files incrementally to avoid holding 10B uint16 values
    # in RAM at once.  numpy.tofile() appends raw binary without any header,
    # which is what we want: train.bin is just a flat array of uint16.
    # We use 'ab' (append-binary) so we can flush chunks as we go.
    #
    # Why not .npy format?  np.save() writes the full array at once (requires
    # RAM for the whole thing).  np.tofile() is streaming.  On load we use
    # np.memmap() which doesn't care about headers — it just maps the raw bytes.

    # Truncate output files (start fresh)
    open(train_path, "wb").close()
    open(val_path, "wb").close()

    print(f"Input:         {input_path}")
    print(f"Output train:  {train_path}")
    print(f"Output val:    {val_path}")
    print(f"Val fraction:  {val_fraction:.1%}")
    print(f"Counting lines...", end=" ", flush=True)
    total_docs = count_lines(input_path)
    print(f"{total_docs:,} documents")

    val_every = max(1, round(1.0 / val_fraction))  # e.g. 0.05 → every 20th doc goes to val

    train_tokens = 0
    val_tokens = 0
    train_buf = []
    val_buf = []
    docs_processed = 0
    t0 = time.time()

    def flush(buf: list, path: Path) -> int:
        """Write accumulated token IDs to disk, return count flushed."""
        if not buf:
            return 0
        arr = np.array(buf, dtype=np.uint16)
        with open(path, "ab") as f:
            arr.tofile(f)
        n = len(buf)
        buf.clear()
        return n

    with tqdm(total=total_docs, unit="doc", dynamic_ncols=True) as pbar:
        for doc_idx, text in enumerate(iter_jsonl(input_path)):
            # Tokenize — TOKENIZERS_PARALLELISM=false is set in the caller to
            # ensure the Rust thread pool is single-threaded.  This is safe
            # because we're doing one document at a time anyway.
            ids = tokenizer.encode(text).ids

            if doc_idx % val_every == 0:
                # Validation split
                val_buf.extend(ids)
                if len(val_buf) >= chunk_size:
                    val_tokens += flush(val_buf, val_path)
            else:
                # Training split
                train_buf.extend(ids)
                if len(train_buf) >= chunk_size:
                    train_tokens += flush(train_buf, train_path)

            docs_processed += 1
            pbar.update(1)

            if docs_processed % 100_000 == 0:
                elapsed = time.time() - t0
                tok_per_s = (train_tokens + val_tokens) / elapsed
                pbar.set_postfix(
                    train=f"{train_tokens/1e9:.2f}B",
                    val=f"{val_tokens/1e6:.0f}M",
                    tok_s=f"{tok_per_s/1e6:.1f}M/s",
                )

    # Flush remaining
    train_tokens += flush(train_buf, train_path)
    val_tokens += flush(val_buf, val_path)

    elapsed = time.time() - t0
    total_tokens = train_tokens + val_tokens
    tok_per_s = total_tokens / elapsed

    print(f"\nDone in {elapsed/60:.1f} min ({tok_per_s/1e6:.2f}M tok/s)")
    print(f"  Train: {train_tokens:,} tokens  ({train_tokens/1e9:.2f}B)")
    print(f"  Val:   {val_tokens:,} tokens  ({val_tokens/1e6:.0f}M)")
    print(f"  Train file size: {train_path.stat().st_size / 1e9:.2f} GB")
    print(f"  Val file size:   {val_path.stat().st_size / 1e6:.0f} MB")

    # Write metadata
    meta = {
        "train_tokens": train_tokens,
        "val_tokens": val_tokens,
        "total_tokens": total_tokens,
        "vocab_size": tokenizer.get_vocab_size(),
        "dtype": "uint16",
        "docs_processed": docs_processed,
        "val_fraction": val_fraction,
        "elapsed_seconds": elapsed,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Meta: {meta_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-tokenize a JSONL corpus into binary token-ID files for fast training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python data/tokenize_dataset.py \\
    --input data/pretrain/train.jsonl \\
    --tokenizer tokenizer_output \\
    --output_dir data/pretrain_tokenized

  # Then train:
  uv run srm-pretrain --config 500m --data_path data/pretrain_tokenized
""",
    )
    parser.add_argument("--input", required=True, help="Path to JSONL file")
    parser.add_argument("--tokenizer", required=True, help="Tokenizer directory")
    parser.add_argument("--output_dir", required=True, help="Output directory for .bin files")
    parser.add_argument(
        "--val_fraction", type=float, default=0.05, help="Fraction for validation (default: 0.05)"
    )
    args = parser.parse_args()

    # Disable HuggingFace tokenizers Rust thread pool.
    # With parallelism enabled, rayon spawns threads that use futex for work-stealing.
    # These futexes can deadlock with Python's GIL and PyTorch's CUDA background threads.
    # For single-document sequential tokenization, parallelism provides no benefit anyway.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    tokenize_corpus(
        input_path=args.input,
        tokenizer_path=args.tokenizer,
        output_dir=args.output_dir,
        val_fraction=args.val_fraction,
    )


if __name__ == "__main__":
    main()
