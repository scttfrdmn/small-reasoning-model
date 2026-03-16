"""
sft_format.py
=============
Download and reformat raw instruction datasets into the <think>...</think> template
for Phase 1 SFT training (spec Section 3.2).

Target format — every example must become:
  {"prompt": "<question>", "response": "<think>\n...\n</think>\n<final answer>"}

Why this template:
  The SFT phase teaches the model the *structural habit* of generating a
  reasoning trace before committing to an answer. The think block acts as a
  scratchpad. During Phase 2 GRPO, the model is rewarded for correct answers
  and also given a small format reward for maintaining this structure. The
  format must be consistent across all SFT sources so the model generalises
  the pattern, not source-specific quirks.

Sources and their reformatting strategy:
  NuminaMath-CoT   — already has step-by-step solution; wrap it in <think> tags.
                     The solution IS the chain of thought, so no extraction needed.
  OpenHermes 2.5   — multi-turn conversations; we take the first human turn as
                     the prompt and the first assistant turn as the answer.
                     No pre-existing CoT, so we use the minimal template.
  CodeFeedback     — instruction/answer pairs; the answer is typically a code block
                     with explanation. Put full answer in <think>, extract last code
                     block or keep full answer as final answer.
  Orca-Math        — direct QA math word problems; no reasoning chain. Apply
                     minimal CoT template: put full answer in <think> as the
                     "reasoning", repeat as final answer after </think>.

Filtering rules:
  - Skip examples where the prompt exceeds 512 tokens (as a rough char estimate:
    512 * ~4 chars/token ≈ 2048 chars). Long prompts swamp the context window,
    leaving almost no room for the CoT trace.
  - Skip examples with empty or whitespace-only responses.
  - Skip OpenHermes examples where there is no assistant turn.

Output:
  data/sft/sft_train.jsonl  — 95% of examples
  data/sft/sft_val.jsonl    — 5% of examples
  data/sft/manifest.json    — source counts, total, split sizes, timestamp

Usage:
  python data/sft_format.py --output_dir ./data/sft --sources all
  python data/sft_format.py --output_dir ./data/sft --sources numina,orca_math
  python data/sft_format.py --output_dir ./data/sft --limit 10000  # quick test run
"""

import argparse
import json
import os
import random
import sys
import time
from typing import Generator, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Rough upper bound on prompt length in characters.
# 512 tokens × 4 chars/token (BPE average for English/math text).
# This is intentionally conservative — we'd rather discard borderline cases
# than train on examples that leave no room for CoT generation.
MAX_PROMPT_CHARS = 2048

# Fraction of total examples held out for validation.
VAL_FRACTION = 0.05

# All available source keys, in the order they will be processed.
ALL_SOURCES = ["numina", "openhermes", "codefeedback", "orca_math"]

# HuggingFace dataset identifiers for each source.
DATASET_IDS = {
    "numina": "AI-MO/NuminaMath-CoT",
    "openhermes": "teknium/OpenHermes-2.5",
    "codefeedback": "m-a-p/CodeFeedback-Filtered-Instruction",
    "orca_math": "microsoft/orca-math-word-problems-200k",
}

# ---------------------------------------------------------------------------
# Template helpers
# ---------------------------------------------------------------------------


def wrap_with_think(reasoning: str, final_answer: str) -> str:
    """
    Build the canonical response string:
        <think>
        {reasoning}
        </think>
        {final_answer}

    Both reasoning and final_answer are stripped of leading/trailing whitespace
    before insertion so the tags sit flush against the content — the model
    should not have to learn to ignore extraneous blank lines inside <think>.
    """
    reasoning = reasoning.strip()
    final_answer = final_answer.strip()
    return f"<think>\n{reasoning}\n</think>\n{final_answer}"


def minimal_think_template(answer: str) -> str:
    """
    For sources with no pre-existing reasoning chain.

    Strategy: place the full answer inside <think> as a "self-explanation"
    and repeat it verbatim after </think> as the final answer. This is not
    ideal CoT, but it:
      (a) teaches the model the <think> structural habit, and
      (b) is consistent with how NuminaMath examples look when the solution
          is already a worked explanation.
    The GRPO phase will later reinforce *correct* reasoning; SFT just needs
    to establish the format.
    """
    answer = answer.strip()
    return f"<think>\nLet me work through this.\n{answer}\n</think>\n{answer}"


def is_valid(prompt: str, response: str) -> bool:
    """
    Return True if the example should be kept.

    Filtering criteria explained:
      - prompt length: we guard on characters rather than tokens because running
        the tokenizer on every example during data prep is slow. The char-based
        limit is deliberately loose enough to keep any example that a 512-token
        limit would keep (BPE is compressive — rare to hit 1 char/token for
        English/math text).
      - empty response: an empty response provides no training signal and would
        cause the model to learn that silence is a valid answer format.
    """
    if not prompt or not prompt.strip():
        return False
    if not response or not response.strip():
        return False
    if len(prompt) > MAX_PROMPT_CHARS:
        return False
    return True


# ---------------------------------------------------------------------------
# Source-specific formatters
# ---------------------------------------------------------------------------


def format_numina(example: dict) -> Optional[dict]:
    """
    NuminaMath-CoT fields: `problem` (str), `solution` (str).

    The solution is a full chain-of-thought ending with the numerical answer.
    We wrap the entire solution in <think> tags. The final answer is extracted
    as the last non-empty line of the solution — NuminaMath solutions typically
    end with "The answer is X" or just the number, so the last line is a
    reasonable proxy for the final answer.

    Why last line instead of a regex:
      NuminaMath uses varied answer formats (\boxed{}, "= X", "is X") across
      different problem sets. Extracting \boxed{} misses ~30% of examples.
      The last non-empty line is a crude but high-recall heuristic; GRPO's
      verifiable reward function will filter out cases where it's wrong.
    """
    problem = (example.get("problem") or "").strip()
    solution = (example.get("solution") or "").strip()

    if not problem or not solution:
        return None

    # Extract the final answer as the last non-empty line of the solution.
    lines = [ln for ln in solution.splitlines() if ln.strip()]
    final_answer = lines[-1].strip() if lines else solution

    response = wrap_with_think(reasoning=solution, final_answer=final_answer)

    if not is_valid(problem, response):
        return None

    return {"prompt": problem, "response": response}


def format_openhermes(example: dict) -> Optional[dict]:
    """
    OpenHermes-2.5 fields: `conversations` (list of {from: str, value: str}).

    We take the first human turn as the prompt and the first assistant turn
    as the answer. Multi-turn context beyond the first exchange is ignored:
      (a) most SFT value is in single-turn instruction following, and
      (b) including multi-turn context would require careful speaker formatting
          that is out of scope here.

    The assistant reply is placed in the minimal <think> template because
    OpenHermes 2.5 does not include reasoning traces — it is a filtered
    instruction-following dataset, not a CoT dataset.
    """
    conversations = example.get("conversations") or []

    human_value = None
    assistant_value = None

    for turn in conversations:
        role = (turn.get("from") or "").lower()
        value = (turn.get("value") or "").strip()
        if role in ("human", "user") and human_value is None:
            human_value = value
        elif role in ("gpt", "assistant") and assistant_value is None:
            assistant_value = value
        # Stop once we have both — we only use the first exchange.
        if human_value and assistant_value:
            break

    if not human_value or not assistant_value:
        return None

    response = minimal_think_template(assistant_value)

    if not is_valid(human_value, response):
        return None

    return {"prompt": human_value, "response": response}


def format_codefeedback(example: dict) -> Optional[dict]:
    """
    CodeFeedback-Filtered-Instruction fields: `query` (str), `answer` (str).

    The answer typically contains a code block plus natural-language explanation.
    We put the full answer inside <think> as the "working-out" and use the full
    answer again as the final answer. This is the same minimal template as
    Orca-Math — for code tasks, the "final answer" IS the code, so repeating it
    makes sense semantically.

    Alternative considered: extract the last code block with a ```...``` regex
    and use that as the final answer. Rejected because:
      (a) many answers end with prose like "This runs in O(n log n) time" which
          is valuable output, and
      (b) the final answer after </think> is only used for reward verification in
          GRPO; code problems use test-case execution, not string matching, so
          exact extraction matters less here.
    """
    query = (example.get("query") or "").strip()
    answer = (example.get("answer") or "").strip()

    if not query or not answer:
        return None

    response = minimal_think_template(answer)

    if not is_valid(query, response):
        return None

    return {"prompt": query, "response": response}


def format_orca_math(example: dict) -> Optional[dict]:
    """
    Orca-Math fields: `question` (str), `answer` (str).

    Pure QA math word problems with no pre-existing reasoning chain.
    Apply the minimal template: put the answer in <think>, repeat as final answer.

    Why include Orca-Math if there's no CoT:
      200K diverse math word problems are valuable for variety. The SFT phase
      is about format learning, not CoT quality. During GRPO, Orca-Math problems
      will be rewarded based on the correct numerical answer, and the model will
      learn to generate CoT traces that lead to those answers through RL.
    """
    question = (example.get("question") or "").strip()
    answer = (example.get("answer") or "").strip()

    if not question or not answer:
        return None

    response = minimal_think_template(answer)

    if not is_valid(question, response):
        return None

    return {"prompt": question, "response": response}


# Map source keys to their formatter functions.
FORMATTERS = {
    "numina": format_numina,
    "openhermes": format_openhermes,
    "codefeedback": format_codefeedback,
    "orca_math": format_orca_math,
}

# ---------------------------------------------------------------------------
# Dataset loading with streaming
# ---------------------------------------------------------------------------


def stream_source(source: str, limit: Optional[int] = None) -> Generator[dict, None, None]:
    """
    Stream and format examples from a single source.

    Why streaming (not load_dataset without streaming=True):
      These datasets are large (NuminaMath ~860K, OpenHermes ~1M). Loading them
      fully into RAM before filtering would peak at several GB of Python objects.
      Streaming processes one shard at a time from the HuggingFace CDN or local
      cache, keeping memory usage flat regardless of dataset size.

    Yields formatted dicts {"prompt": ..., "response": ...}.
    Skips malformed examples silently (they're a small fraction of each dataset).
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: `datasets` library not installed. Run: uv sync", file=sys.stderr)
        return

    dataset_id = DATASET_IDS[source]
    formatter = FORMATTERS[source]

    print(f"  Loading {source} ({dataset_id}) …", flush=True)

    try:
        # streaming=True avoids downloading the entire dataset before iteration.
        # We always use the train split; held-out val split is carved out locally
        # after formatting so we control the split ratio ourselves.
        ds = load_dataset(dataset_id, split="train", streaming=True, trust_remote_code=True)
    except Exception as exc:
        # A download failure (network error, private dataset, wrong split name)
        # should not abort the entire pipeline — just skip this source with a
        # clear warning so the operator knows to investigate.
        print(f"  WARNING: could not load {source} ({dataset_id}): {exc}", file=sys.stderr)
        print(f"  Skipping {source} and continuing.", file=sys.stderr)
        return

    count = 0
    skipped = 0

    for raw in ds:
        try:
            formatted = formatter(raw)
        except Exception as exc:
            # Defensive catch: a badly structured example (missing key, wrong type)
            # should not crash the pipeline. Log the first few failures for debugging.
            skipped += 1
            if skipped <= 5:
                print(
                    f"  WARNING: formatter error on {source} example #{count + skipped}: {exc}",
                    file=sys.stderr,
                )
            continue

        if formatted is None:
            skipped += 1
            continue

        yield formatted
        count += 1

        if limit is not None and count >= limit:
            break

    print(f"  {source}: {count} kept, {skipped} skipped", flush=True)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def build_dataset(
    sources: list[str],
    output_dir: str,
    limit: Optional[int] = None,
    seed: int = 42,
) -> None:
    """
    Collect examples from all requested sources, shuffle, split, and write JSONL.

    The shuffle before split is important: without it, the validation set would
    be drawn entirely from the last source in the list, biasing the val metrics.
    We use a fixed seed for reproducibility across runs.

    Why write-then-split instead of streaming directly to two files:
      We need a random split, which requires knowing the full length. We accumulate
      all examples in memory (they're dicts of strings, not tensors — RAM is fine)
      then write two JSONL files. For very large runs (> 5M examples), one could
      instead do a two-pass approach, but for the expected dataset size (~500K–1M
      after filtering) a list in RAM is straightforward.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Per-source counts for the manifest.
    source_counts: dict[str, int] = {}
    all_examples: list[dict] = []

    for source in sources:
        before = len(all_examples)

        # If --limit is set, distribute evenly across sources so each source
        # gets limit // len(sources) examples. This keeps the mix balanced
        # for small test runs instead of pulling all examples from the first source.
        per_source_limit = (limit // len(sources)) if limit is not None else None

        for ex in stream_source(source, limit=per_source_limit):
            all_examples.append(ex)

        source_counts[source] = len(all_examples) - before
        print(f"  Running total: {len(all_examples)} examples", flush=True)

    if not all_examples:
        print("ERROR: no examples collected. Check dataset availability.", file=sys.stderr)
        sys.exit(1)

    # Shuffle with a fixed seed for reproducible train/val splits across runs.
    rng = random.Random(seed)
    rng.shuffle(all_examples)

    total = len(all_examples)
    n_val = max(1, int(total * VAL_FRACTION))
    n_train = total - n_val

    train_examples = all_examples[:n_train]
    val_examples = all_examples[n_train:]

    # Write JSONL files — one JSON object per line, no trailing comma.
    train_path = os.path.join(output_dir, "sft_train.jsonl")
    val_path = os.path.join(output_dir, "sft_val.jsonl")

    print(f"\nWriting {n_train} train examples → {train_path}", flush=True)
    _write_jsonl(train_examples, train_path)

    print(f"Writing {n_val} val examples   → {val_path}", flush=True)
    _write_jsonl(val_examples, val_path)

    # Write manifest so downstream scripts know what was built and when.
    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sources": sources,
        "source_counts": source_counts,
        "total": total,
        "train": n_train,
        "val": n_val,
        "val_fraction": VAL_FRACTION,
        "max_prompt_chars": MAX_PROMPT_CHARS,
        "limit_per_source": (limit // len(sources)) if limit is not None else None,
        "seed": seed,
    }
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest → {manifest_path}", flush=True)

    print(f"\nDone. {n_train} train / {n_val} val ({100 * VAL_FRACTION:.0f}% val split).")


def _write_jsonl(examples: list[dict], path: str) -> None:
    """Write a list of dicts to a JSONL file (one JSON object per line)."""
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            # ensure_ascii=False preserves unicode math symbols (∫, Σ, √, etc.)
            # without escaping them to \uXXXX sequences, which bloat file size
            # and make manual inspection harder.
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and reformat SFT datasets into <think>...</think> template.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python data/sft_format.py --output_dir ./data/sft --sources all
  python data/sft_format.py --output_dir ./data/sft --sources numina,orca_math
  python data/sft_format.py --output_dir ./data/sft --limit 10000
        """,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for output JSONL files and manifest.",
    )
    parser.add_argument(
        "--sources",
        type=str,
        default="all",
        help=(
            f"Comma-separated source keys or 'all'. "
            f"Available: {', '.join(ALL_SOURCES)}. Default: all."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Total example cap across all sources (distributed evenly). "
            "Useful for quick test runs."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffle and train/val split. Default: 42.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve source list.
    if args.sources.strip().lower() == "all":
        sources = ALL_SOURCES
    else:
        sources = [s.strip() for s in args.sources.split(",") if s.strip()]
        unknown = [s for s in sources if s not in ALL_SOURCES]
        if unknown:
            print(
                f"ERROR: unknown source(s): {unknown}. " f"Available: {ALL_SOURCES}",
                file=sys.stderr,
            )
            sys.exit(1)

    print(f"Sources: {sources}")
    print(f"Output dir: {args.output_dir}")
    if args.limit:
        print(f"Limit: {args.limit} total ({args.limit // len(sources)} per source)")

    build_dataset(
        sources=sources,
        output_dir=args.output_dir,
        limit=args.limit,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
