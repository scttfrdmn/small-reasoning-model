"""
preprocess.py
=============
Pre-training data download and preprocessing pipeline for the small reasoning model.

Spec (Section 3.1) defines three curriculum stages:

  Stage 1  (0–30% of target tokens):
    FineWeb-Edu / DCLM   40%   — filtered web text; high educational quality
    The Stack v2         25%   — code; builds reasoning circuits early
    OpenWebMath          25%   — math web text; essential for Phase 2
    Wikipedia / Books    10%   — world-knowledge anchor

  Stage 2  (30–100% of target tokens):
    Same four sources but math (OpenWebMath + NuminaMath) upweighted to 35%.
    NuminaMath problems + solutions are introduced here.

  Stage 3  (final 10% of target tokens):
    40% math, 40% code, 20% general — pure reasoning polish.

Quality filtering (spec):
  - Perplexity filter: remove documents where GPT-2 perplexity > 10,000.
  - Deduplication: MinHash LSH at 3-gram level, Jaccard threshold 0.8.

Implementation trade-offs vs. spec:
  - We use a HEURISTIC quality filter instead of GPT-2 perplexity.
    Reason: running GPT-2 inference on 10–100 B tokens of streaming data
    requires a separate GPU-resident model and adds ~5–10x runtime.  For a
    validation run the heuristic (length, word count, non-ASCII ratio) removes
    the most obvious garbage (binary blobs, spam, near-empty documents) cheaply
    and with no extra dependencies.  For production, swap in the GPT-2 filter.

  - We use SHA-256 document-level exact dedup instead of MinHash LSH.
    Reason: MinHash LSH (datasketch library) requires building an in-memory
    index of O(N) signatures.  For a streaming pipeline that processes 10 B+
    tokens without buffering the full corpus, keeping that index in RAM is
    impractical on a workstation.  SHA-256 hash dedup catches exact duplicates
    (e.g. the same Wikipedia article appearing in two datasets) with zero extra
    dependencies, O(1) per-document memory (beyond the hash set), and
    deterministic results.  Limitation: near-duplicates (paraphrases, slight
    reformatting) slip through; for a production run use datasketch MinHashLSH
    over 3-gram shingles with Jaccard threshold 0.8.

Output format consumed by training/pretrain.py → TokenDataset:
  JSONL file, one document per line:
    {"text": "document text ...", "source": "fineweb-edu"}

Usage:
  # Full run — 10 B tokens, cap at 50 GB disk
  python data/preprocess.py \\
    --output_dir ./data/pretrain \\
    --target_tokens 10_000_000_000 \\
    --limit_gb 50 \\
    --seed 42

  # Validation run — only a few sources
  python data/preprocess.py \\
    --output_dir ./data/pretrain \\
    --target_tokens 10_000_000_000 \\
    --limit_gb 5 \\
    --sources fineweb_edu,openwebmath,wikipedia \\
    --seed 42
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Generator, Iterable, List, Optional, Set, Tuple

# HuggingFace datasets — streaming mode is critical here.
# streaming=True means the dataset is never fully downloaded; records arrive
# one-by-one over HTTP.  This lets us process arbitrarily large datasets on a
# machine with modest RAM (no full 23 GB FineWeb-Edu download needed).
from datasets import load_dataset

# tqdm gives us a progress bar that works both in terminals and notebooks
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants — curriculum mix definitions
# ---------------------------------------------------------------------------

# Each "source" is identified by a short slug used in the JSONL output and
# in the --sources flag.  The slug is what TokenDataset sees as doc["source"].
SOURCES = [
    "fineweb_edu",
    "openwebmath",
    "wikipedia",
    "numinamath",
    "the_stack",
]

# Curriculum stage proportions.
# Fractions must sum to 1.0 within each stage.
# These encode the spec's staged curriculum in a form that the mixing loop
# can query at runtime by comparing tokens_written / target_tokens.
#
# Each entry: (slug, fraction)
STAGE_1_MIX: List[Tuple[str, float]] = [
    # 0–30% tokens: broad web + code + math + knowledge
    ("fineweb_edu", 0.40),
    ("the_stack", 0.25),
    ("openwebmath", 0.25),
    ("wikipedia", 0.10),
]

STAGE_2_MIX: List[Tuple[str, float]] = [
    # 30–100% tokens: math upweighted; NuminaMath introduced
    # FineWeb stays dominant for general language.
    # Stack v2 stays at 20% to keep code circuit alive.
    # OpenWebMath + NuminaMath together = 35% math.
    ("fineweb_edu", 0.38),
    ("the_stack", 0.20),
    ("openwebmath", 0.22),
    ("numinamath", 0.13),
    ("wikipedia", 0.07),
]

STAGE_3_MIX: List[Tuple[str, float]] = [
    # Final 10%: heavy math + code polish (spec: 40% math, 40% code, 20% general)
    ("openwebmath", 0.25),
    ("numinamath", 0.15),
    ("the_stack", 0.40),
    ("fineweb_edu", 0.15),
    ("wikipedia", 0.05),
]

# Stage boundaries as fractions of target_tokens
STAGE_1_END = 0.30  # 0–30%
STAGE_3_START = 0.90  # 90–100%  (final 10%)


# ---------------------------------------------------------------------------
# Quality filter — heuristic substitute for GPT-2 perplexity
# ---------------------------------------------------------------------------

# Thresholds chosen empirically to retain ~85–90% of clean web text while
# removing near-empty pages, binary/encoding garbage, and keyword spam.
MIN_CHARS = 200  # Very short documents add noise without signal
MAX_CHARS = 1_000_000  # Abnormally long = likely scraped boilerplate
MIN_WORDS = 30  # Fewer than 30 words → probably a nav/menu page
MAX_NON_ASCII_RATIO = 0.30  # >30% non-ASCII → likely wrong encoding or binary


def passes_quality_filter(text: str) -> bool:
    """
    Cheap heuristic quality filter.

    Design note: the spec calls for GPT-2 perplexity < 10,000.  We substitute
    a set of deterministic rules that run in microseconds per document and
    require no extra model.  The rules target the same failure modes as
    perplexity filtering:
      - Near-empty / nav pages     → MIN_CHARS, MIN_WORDS
      - Binary / encoding garbage  → MAX_NON_ASCII_RATIO
      - Massive boilerplate blobs  → MAX_CHARS

    For production: replace this function body with a GPT-2 forward pass and
    threshold on exp(mean_cross_entropy) > 10_000 to match the spec exactly.

    Returns True if the document should be kept, False if it should be dropped.
    """
    if not text or not isinstance(text, str):
        return False

    n_chars = len(text)

    # Length checks
    if n_chars < MIN_CHARS or n_chars > MAX_CHARS:
        return False

    # Word count — a rough proxy for linguistic content density
    words = text.split()
    if len(words) < MIN_WORDS:
        return False

    # Non-ASCII ratio — catches encoding errors and binary data
    n_non_ascii = sum(1 for c in text if ord(c) > 127)
    if n_non_ascii / n_chars > MAX_NON_ASCII_RATIO:
        return False

    return True


# ---------------------------------------------------------------------------
# Deduplication — SHA-256 exact-match hash set
# ---------------------------------------------------------------------------

# Production note: for near-duplicate removal, replace this with MinHash LSH
# from the datasketch library:
#   from datasketch import MinHash, MinHashLSH
#   lsh = MinHashLSH(threshold=0.8, num_perm=128)
# Build 3-gram shingles of each document, compute MinHash signature, then
# query the LSH index before inserting.  This catches paraphrased duplicates
# that differ only in whitespace or minor edits.
#
# For a validation run exact dedup is sufficient: the same Wikipedia article
# often appears in multiple training datasets (FineWeb, Wikipedia itself, web
# crawls), and exact dedup removes all those at zero extra cost.


def make_doc_hash(text: str) -> str:
    """
    Return the SHA-256 hex digest of the document text.

    We hash the raw UTF-8 bytes of the document.  Two documents with exactly
    the same text (after our normalization) get the same hash.  Collisions are
    cryptographically negligible at the scale we care about (<2^-128).
    """
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


# ---------------------------------------------------------------------------
# Dataset loaders — each returns a streaming generator of {"text": ...} dicts
# ---------------------------------------------------------------------------


def stream_fineweb_edu(seed: int) -> Generator[Dict, None, None]:
    """
    Stream FineWeb-Edu (HuggingFaceFW/fineweb-edu, sample-10BT config).

    FineWeb-Edu is a 10B-token sample of the full FineWeb-Edu dataset (~1.3T
    tokens) filtered for educational quality.  The "sample-10BT" config fits
    on most machines (~23 GB) but we use streaming so we never download more
    than we consume.

    Field used: "text" (already present; no mapping needed).
    """
    try:
        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train",
            streaming=True,
        )
        # Shuffle the stream with a fixed seed and buffer so documents arrive
        # in a reproducible but mixed order.  Buffer size 10_000 keeps ~100 MB
        # in RAM — acceptable for a streaming pipeline.
        ds = ds.shuffle(seed=seed, buffer_size=10_000)
        for doc in ds:
            if "text" in doc and doc["text"]:
                yield {"text": doc["text"], "source": "fineweb-edu"}
    except Exception as e:
        print(f"[WARN] fineweb_edu stream failed: {e}", file=sys.stderr)
        return


def stream_openwebmath(seed: int) -> Generator[Dict, None, None]:
    """
    Stream OpenWebMath (open-web-math/open-web-math).

    OpenWebMath is ~14 B tokens of math-heavy web text extracted from Common
    Crawl.  It covers textbooks, math Stack Exchange, arXiv, and general STEM
    web pages.  High overlap with what the model needs for Phase 2 reasoning.

    Field used: "text".
    """
    try:
        ds = load_dataset(
            "open-web-math/open-web-math",
            split="train",
            streaming=True,
        )
        ds = ds.shuffle(seed=seed, buffer_size=10_000)
        for doc in ds:
            if "text" in doc and doc["text"]:
                yield {"text": doc["text"], "source": "openwebmath"}
    except Exception as e:
        print(f"[WARN] openwebmath stream failed: {e}", file=sys.stderr)
        return


def stream_wikipedia(seed: int) -> Generator[Dict, None, None]:
    """
    Stream English Wikipedia (wikimedia/wikipedia, 20231101.en snapshot).

    Wikipedia provides world-knowledge grounding: factual entities, concepts,
    and structured prose.  We use the 2023-11-01 English snapshot (~21 M docs).

    Field used: "text" (Wikipedia articles; the "title" is prepended to give
    the model context about what the article is about).
    """
    try:
        ds = load_dataset(
            "wikimedia/wikipedia",
            name="20231101.en",
            split="train",
            streaming=True,
        )
        ds = ds.shuffle(seed=seed, buffer_size=10_000)
        for doc in ds:
            # Prepend the article title so the model learns the association
            # between title and content (helpful for entity retrieval tasks).
            title = doc.get("title", "")
            text = doc.get("text", "")
            if text:
                full_text = f"{title}\n\n{text}" if title else text
                yield {"text": full_text, "source": "wikipedia"}
    except Exception as e:
        print(f"[WARN] wikipedia stream failed: {e}", file=sys.stderr)
        return


def stream_numinamath(seed: int) -> Generator[Dict, None, None]:
    """
    Stream NuminaMath-TIR (AI-MO/NuminaMath-TIR).

    NuminaMath-TIR (~860 K examples) contains competition-style math problems
    paired with tool-integrated reasoning (TIR) solutions.  Each example is a
    (problem, solution) pair; we join them with a separator so the model sees
    both the problem statement and the worked solution in one document.

    This is introduced only in Stage 2 (30–100% of tokens) per the curriculum
    spec, because the model needs a solid language foundation first before math
    reasoning fine-grained signal can be absorbed effectively.

    Fields: "problem" + "solution" → joined as one document.
    """
    try:
        ds = load_dataset(
            "AI-MO/NuminaMath-TIR",
            split="train",
            streaming=True,
        )
        ds = ds.shuffle(seed=seed, buffer_size=5_000)
        for doc in ds:
            problem = doc.get("problem", "")
            solution = doc.get("solution", "")
            if problem and solution:
                # The separator is human-readable so that if the tokenizer
                # sees this text during SFT later, the boundary is explicit.
                text = f"Problem:\n{problem}\n\nSolution:\n{solution}"
                yield {"text": text, "source": "numinamath"}
    except Exception as e:
        print(f"[WARN] numinamath stream failed: {e}", file=sys.stderr)
        return


def stream_the_stack(seed: int) -> Generator[Dict, None, None]:
    """
    Stream StarCoderData (bigcode/starcoderdata) — a large public code corpus.

    Original plan was The Stack v2 (bigcode/the-stack-v2-train-smol-ids) but
    that dataset is gated and requires explicit HuggingFace approval.
    StarCoderData is the permissively-licensed deduplicated corpus used to train
    StarCoder; it is publicly accessible without any approval step and covers
    ~80 programming languages.  It serves the same training objective: building
    structured-reasoning "circuits" that transfer to math and multi-step reasoning
    in Phase 2.

    Field used: "content" (the source code text).
    """
    try:
        ds = load_dataset(
            "bigcode/starcoderdata",
            data_dir="python",  # start with Python — highest quality, most reasoning-relevant
            split="train",
            streaming=True,
            trust_remote_code=False,
        )
        ds = ds.shuffle(seed=seed, buffer_size=10_000)
        for doc in ds:
            content = doc.get("content", "")
            if content:
                yield {"text": content, "source": "the_stack"}
    except Exception as e:
        # Gated dataset: warn clearly rather than crash so the rest of the
        # pipeline keeps running.
        print(
            f"[WARN] the_stack (starcoderdata) stream failed: {e}",
            file=sys.stderr,
        )
        return


# Map each slug to its loader function.
# This is used by get_stage_mix() to instantiate the streams we actually need.
SOURCE_LOADERS = {
    "fineweb_edu": stream_fineweb_edu,
    "openwebmath": stream_openwebmath,
    "wikipedia": stream_wikipedia,
    "numinamath": stream_numinamath,
    "the_stack": stream_the_stack,
}


# ---------------------------------------------------------------------------
# Curriculum mixing — proportional round-robin
# ---------------------------------------------------------------------------


def get_stage_mix(
    tokens_written: int,
    target_tokens: int,
    allowed_sources: Optional[Set[str]],
) -> List[Tuple[str, float]]:
    """
    Return the (source, fraction) list appropriate for the current token count.

    The curriculum advances through three stages based on how far through the
    target token budget we are.  allowed_sources filters to the subset the
    user requested via --sources; remaining fractions are re-normalised so
    they still sum to 1.0.

    Args:
        tokens_written:  Tokens emitted to disk so far.
        target_tokens:   Total token budget for this run.
        allowed_sources: If not None, restrict to these source slugs.

    Returns:
        List of (slug, fraction) pairs with fractions summing to 1.0.
    """
    progress = tokens_written / max(target_tokens, 1)

    if progress < STAGE_1_END:
        raw_mix = STAGE_1_MIX
    elif progress >= STAGE_3_START:
        raw_mix = STAGE_3_MIX
    else:
        raw_mix = STAGE_2_MIX

    # Filter to allowed sources
    if allowed_sources is not None:
        raw_mix = [(s, w) for s, w in raw_mix if s in allowed_sources]

    if not raw_mix:
        raise ValueError(
            f"No sources remain after filtering to {allowed_sources}. " "Check --sources flag."
        )

    # Re-normalise so weights sum to 1.0 even after filtering
    total_weight = sum(w for _, w in raw_mix)
    return [(s, w / total_weight) for s, w in raw_mix]


class MixedStreamSampler:
    """
    Proportional sampler over multiple source generators.

    Instead of interleaving at a fixed stride, we draw the next source
    probabilistically from the current curriculum mix.  This means the mix
    proportions are approximate (converge to exact in expectation) but the
    output is never perfectly periodic — which matters because neural nets
    can overfit to periodic patterns in training data.

    Design: we keep one open generator per source.  When a source's generator
    is exhausted, we restart it from the beginning (the dataset loops).  This
    handles the case where a small dataset (e.g. NuminaMath at 860 K docs)
    runs out before a large one (FineWeb-Edu at 10 B tokens).

    Thread safety: this class is single-threaded.  For multi-process data
    loading, spawn separate MixedStreamSampler instances per worker.
    """

    def __init__(
        self,
        active_sources: List[str],
        seed: int,
        allowed_sources: Optional[Set[str]],
        target_tokens: int,
    ):
        self.active_sources = active_sources
        self.seed = seed
        self.allowed_sources = allowed_sources
        self.target_tokens = target_tokens
        self.rng = random.Random(seed)

        # Lazily opened generators, one per source.
        # Sources are removed from this dict when they exhaust or become inaccessible.
        # stream() uses self._generators.keys() as the live allow-list so that
        # get_stage_mix() stage entries are always filtered to reachable sources,
        # even when the user did not specify --sources (allowed_sources=None).
        self._generators: Dict[str, Optional[Generator]] = {s: None for s in active_sources}

    def _get_generator(self, source: str) -> Generator:
        """Return the open generator for source, creating/restarting as needed."""
        gen = self._generators[source]
        if gen is None:
            loader = SOURCE_LOADERS[source]
            gen = loader(self.seed)
            self._generators[source] = gen
        return gen

    def _restart_generator(self, source: str) -> Generator:
        """Restart a generator that has been exhausted."""
        loader = SOURCE_LOADERS[source]
        gen = loader(self.seed)
        self._generators[source] = gen
        return gen

    def stream(self, tokens_written_ref: List[int]) -> Generator[Dict, None, None]:
        """
        Yield documents in curriculum-proportional order.

        tokens_written_ref is a mutable list of length 1 so the caller can
        update it and the sampler sees the current stage without restarting.
        We use a list (not int) because Python ints are immutable; a list is a
        simple mutable container.
        """
        while True:
            # Build the effective allow-set: intersection of the caller's --sources
            # filter and the sources still alive in self._generators.  Using
            # self._generators.keys() as the live filter ensures that when a source
            # is removed due to access failure, it disappears from future stage mixes
            # even when allowed_sources=None (i.e. the user did not specify --sources).
            live = set(self._generators.keys())
            effective_allowed = (
                live if self.allowed_sources is None else (live & self.allowed_sources)
            )
            mix = get_stage_mix(tokens_written_ref[0], self.target_tokens, effective_allowed)

            # Choose a source proportionally
            sources = [s for s, _ in mix]
            weights = [w for _, w in mix]
            source = self.rng.choices(sources, weights=weights, k=1)[0]

            gen = self._get_generator(source)

            # Try to get the next document from this source
            try:
                doc = next(gen)
                yield doc
            except StopIteration:
                # Source exhausted — restart it (loop)
                print(
                    f"[INFO] Source '{source}' exhausted, restarting from beginning.",
                    file=sys.stderr,
                )
                gen = self._restart_generator(source)
                try:
                    doc = next(gen)
                    yield doc
                except StopIteration:
                    # The source has zero documents even after restart (e.g.
                    # access denied, empty dataset).  Remove it from the active
                    # list so we don't spin forever.
                    print(
                        f"[WARN] Source '{source}' yielded zero documents after restart. "
                        "Removing from mix.",
                        file=sys.stderr,
                    )
                    self.active_sources.remove(source)
                    self._generators.pop(source, None)
                    if self.allowed_sources:
                        self.allowed_sources.discard(source)
                    if not self.active_sources:
                        raise RuntimeError("All sources exhausted or inaccessible.")


# ---------------------------------------------------------------------------
# Token count estimation
# ---------------------------------------------------------------------------

# We need to track tokens written to know when to stop and which stage we're
# in.  Loading a full tokenizer just for counting is heavy.  Instead we use a
# character-to-token ratio derived empirically from LLaMA-style BPE tokenizers
# on English text.  The ratio is ~4.0 chars/token for general English, ~3.5
# for code, ~4.2 for math.  We use 4.0 as a conservative estimate.
#
# Why not count exactly?  The actual tokenizer lives in ./tokenizer_output and
# may not be trained yet when this script runs.  The estimate is accurate to
# within ~10%, which is sufficient for stage-transition decisions.
CHARS_PER_TOKEN_ESTIMATE = 4.0


def estimate_tokens(text: str) -> int:
    """Estimate token count from character count."""
    return max(1, int(len(text) / CHARS_PER_TOKEN_ESTIMATE))


# ---------------------------------------------------------------------------
# Manifest tracking
# ---------------------------------------------------------------------------


def load_manifest(manifest_path: Path) -> Dict:
    """Load existing manifest or return a fresh one."""
    if manifest_path.exists():
        with open(manifest_path) as f:
            return json.load(f)
    return {
        "sources": {},
        "total_docs": 0,
        "total_tokens_estimate": 0,
        "total_bytes": 0,
        "dedup_rejected": 0,
        "quality_rejected": 0,
        "start_time": time.time(),
    }


def save_manifest(manifest: Dict, manifest_path: Path) -> None:
    """Atomically write manifest to disk."""
    tmp_path = manifest_path.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(manifest, f, indent=2)
    tmp_path.replace(manifest_path)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    output_dir: Path,
    target_tokens: int,
    limit_gb: Optional[float],
    sources: Optional[List[str]],
    seed: int,
) -> None:
    """
    Run the full data download, filter, dedup, and mix pipeline.

    Args:
        output_dir:    Directory where train.jsonl and manifest.json are written.
        target_tokens: Stop after writing approximately this many tokens.
        limit_gb:      Hard cap on output file size in gigabytes.
        sources:       If not None, restrict to this list of source slugs.
        seed:          Random seed for reproducibility.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    manifest_path = output_dir / "manifest.json"

    # Determine which sources are active
    allowed_sources: Optional[Set[str]] = None
    if sources is not None:
        invalid = set(sources) - set(SOURCE_LOADERS.keys())
        if invalid:
            raise ValueError(f"Unknown sources: {invalid}. Valid: {sorted(SOURCE_LOADERS.keys())}")
        allowed_sources = set(sources)
        active_sources = sources
    else:
        active_sources = list(SOURCE_LOADERS.keys())

    print(f"[INFO] Output dir:      {output_dir}")
    print(f"[INFO] Target tokens:   {target_tokens:,}")
    print(f"[INFO] Limit GB:        {limit_gb}")
    print(f"[INFO] Active sources:  {active_sources}")
    print(f"[INFO] Seed:            {seed}")

    # Byte cap derived from limit_gb
    limit_bytes: Optional[int] = None
    if limit_gb is not None:
        limit_bytes = int(limit_gb * 1024**3)
        print(f"[INFO] Byte cap:        {limit_bytes:,} bytes ({limit_gb} GB)")

    # Load existing progress (allows resuming an interrupted run)
    manifest = load_manifest(manifest_path)
    seen_hashes: Set[str] = set()

    # If resuming, we can't re-hydrate the seen_hashes set (it wasn't saved)
    # so dedup starts fresh.  This means near-duplicate documents written in a
    # previous run can reappear.  Acceptable for a validation run.
    tokens_written = manifest["total_tokens_estimate"]
    bytes_written = manifest["total_bytes"]

    if tokens_written > 0:
        print(
            f"[INFO] Resuming: {tokens_written:,} tokens, {bytes_written:,} bytes already written."
        )

    # Mutable reference so MixedStreamSampler sees live updates
    tokens_written_ref = [tokens_written]

    sampler = MixedStreamSampler(
        active_sources=list(active_sources),
        seed=seed,
        allowed_sources=allowed_sources,
        target_tokens=target_tokens,
    )

    # Open the output file in append mode (safe for resume)
    with open(train_path, "a", encoding="utf-8") as out_f:
        pbar = tqdm(
            total=target_tokens,
            initial=tokens_written,
            unit="tok",
            unit_scale=True,
            desc="tokens",
            dynamic_ncols=True,
        )

        manifest_save_interval = 50_000  # Save manifest every N documents
        docs_since_save = 0

        try:
            for raw_doc in sampler.stream(tokens_written_ref):
                # ── 1. Extract text ──────────────────────────────────────────
                text = raw_doc.get("text", "")
                source_slug = raw_doc.get("source", "unknown")

                # ── 2. Quality filter ────────────────────────────────────────
                if not passes_quality_filter(text):
                    manifest["quality_rejected"] = manifest.get("quality_rejected", 0) + 1
                    continue

                # ── 3. Exact dedup ───────────────────────────────────────────
                doc_hash = make_doc_hash(text)
                if doc_hash in seen_hashes:
                    manifest["dedup_rejected"] = manifest.get("dedup_rejected", 0) + 1
                    continue
                seen_hashes.add(doc_hash)

                # ── 4. Estimate tokens & check budget ────────────────────────
                n_tokens = estimate_tokens(text)

                # ── 5. Serialise to JSONL ────────────────────────────────────
                record = json.dumps({"text": text, "source": source_slug}, ensure_ascii=False)
                line = record + "\n"
                line_bytes = line.encode("utf-8")
                n_bytes = len(line_bytes)

                # ── 6. Check size cap before writing ─────────────────────────
                if limit_bytes is not None and bytes_written + n_bytes > limit_bytes:
                    print(
                        f"\n[INFO] Reached --limit_gb cap ({limit_gb} GB). Stopping.",
                        file=sys.stderr,
                    )
                    break

                # ── 7. Write to disk ─────────────────────────────────────────
                out_f.write(line)

                # ── 8. Update counters ───────────────────────────────────────
                tokens_written += n_tokens
                bytes_written += n_bytes
                tokens_written_ref[0] = tokens_written

                src_stats = manifest["sources"].setdefault(
                    source_slug, {"docs": 0, "tokens_estimate": 0}
                )
                src_stats["docs"] += 1
                src_stats["tokens_estimate"] += n_tokens
                manifest["total_docs"] = manifest.get("total_docs", 0) + 1
                manifest["total_tokens_estimate"] = tokens_written
                manifest["total_bytes"] = bytes_written

                pbar.update(n_tokens)
                docs_since_save += 1

                # ── 9. Periodic manifest flush ───────────────────────────────
                if docs_since_save >= manifest_save_interval:
                    save_manifest(manifest, manifest_path)
                    docs_since_save = 0

                # ── 10. Check token budget ───────────────────────────────────
                if tokens_written >= target_tokens:
                    print(
                        f"\n[INFO] Reached target_tokens ({target_tokens:,}). Done.",
                        file=sys.stderr,
                    )
                    break

        except KeyboardInterrupt:
            print("\n[INFO] Interrupted by user. Saving manifest.", file=sys.stderr)
        finally:
            pbar.close()

    # Final manifest write with elapsed time
    manifest["elapsed_seconds"] = time.time() - manifest.get("start_time", time.time())
    manifest["done"] = tokens_written >= target_tokens
    save_manifest(manifest, manifest_path)

    print("\n[DONE] Pipeline complete.")
    print(f"       Documents written: {manifest['total_docs']:,}")
    print(f"       Tokens (estimate): {manifest['total_tokens_estimate']:,}")
    print(f"       Bytes written:     {manifest['total_bytes']:,}")
    print(f"       Quality rejected:  {manifest.get('quality_rejected', 0):,}")
    print(f"       Dedup rejected:    {manifest.get('dedup_rejected', 0):,}")
    print(f"       Output:            {train_path}")
    print(f"       Manifest:          {manifest_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-training data download and preprocessing pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full run — 10B tokens, 50 GB cap
  python data/preprocess.py \\
    --output_dir ./data/pretrain \\
    --target_tokens 10_000_000_000 \\
    --limit_gb 50 \\
    --seed 42

  # Validation run — subset of sources, 1 GB cap
  python data/preprocess.py \\
    --output_dir ./data/pretrain_val \\
    --target_tokens 10_000_000_000 \\
    --limit_gb 1 \\
    --sources fineweb_edu,openwebmath,wikipedia \\
    --seed 42
""",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/pretrain",
        help="Directory for train.jsonl and manifest.json (default: ./data/pretrain)",
    )
    parser.add_argument(
        "--target_tokens",
        type=lambda x: int(x.replace("_", "")),
        default=10_000_000_000,
        help="Stop after writing approximately this many tokens (default: 10_000_000_000)",
    )
    parser.add_argument(
        "--limit_gb",
        type=float,
        default=None,
        help="Hard cap on output file size in GB (useful for validation runs)",
    )
    parser.add_argument(
        "--sources",
        type=str,
        default=None,
        help=(
            "Comma-separated subset of sources to use.  "
            f"Valid: {', '.join(sorted(SOURCE_LOADERS.keys()))}.  "
            "Default: all sources."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for stream shuffling and source selection (default: 42)",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    sources: Optional[List[str]] = None
    if args.sources:
        sources = [s.strip() for s in args.sources.split(",") if s.strip()]

    run_pipeline(
        output_dir=Path(args.output_dir),
        target_tokens=args.target_tokens,
        limit_gb=args.limit_gb,
        sources=sources,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
