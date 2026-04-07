"""
grpo_dataset.py
===============
Build and difficulty-filter the verifiable problem dataset for Phase 2 GRPO.

Why verifiable problems only:
  GRPO requires a reward signal that is ground-truth correct — not a learned
  reward model, which can be gamed through reward hacking. Every problem in
  this dataset has a deterministic answer that can be checked programmatically.
  See rewards.py for the verification functions.

Two-stage workflow:
  Stage 1 — build_dataset():
    Download problems from each source, extract/normalize ground-truth answers,
    write grpo_raw.jsonl. No model needed. pass_rate is null for all examples.

  Stage 2 — filter_by_difficulty():
    Load the SFT checkpoint, run each problem with G=8 completions, record
    the fraction correct, keep only 20%–80% pass rate.

    Why 20–80% pass rate:
      GRPO's advantage A_i = (r_i - mean(r)) / std(r) is zero when all rewards
      in a group are identical. If pass_rate > 80%, most or all of the G=8
      completions are correct → all rewards ≈ 1 → advantage ≈ 0 → no gradient.
      If pass_rate < 20%, all completions are wrong → rewards ≈ 0 → advantage ≈ 0.
      Only the 20–80% "Goldilocks zone" produces meaningful gradients.
      (This analysis follows Section 3.3 of the spec and the GRPO paper.)

Output schema per example:
  {
    "prompt":            "Solve: x^2 - 5x + 6 = 0",
    "answer":            "x = 2 or x = 3",      # human-readable ground truth
    "answer_normalized": "2,3",                  # machine-checkable; what rewards.py compares against
    "domain":            "math_sympy",           # which verifier to call: math_exact | math_sympy | logic
    "source":            "math",                 # originating dataset key
    "difficulty":        "3",                    # level from source metadata, or null
    "pass_rate":         null                    # filled in by filter_by_difficulty()
  }

Sources:
  MATH (Hendrycks)  — "hendrycks/competition_math" — problem, solution, level
  GSM8K             — "openai/gsm8k" (main)      — question, answer (#### N at end)
  NuminaMath-TIR    — "AI-MO/NuminaMath-TIR"    — problem, solution (\boxed{} answer)
  LogiQA            — "lucasmccabe/logiqa"       — query, options, correct_option

Usage:
  # Stage 1: build raw dataset (no model needed)
  python data/grpo_dataset.py --output_dir ./data/grpo --sources all

  # Stage 2: filter by difficulty (needs SFT checkpoint)
  python data/grpo_dataset.py \\
    --filter \\
    --input ./data/grpo/grpo_raw.jsonl \\
    --checkpoint ./checkpoints/1b_sft/best.pt \\
    --config 1b \\
    --output ./data/grpo/grpo_filtered.jsonl
"""

import argparse
import json
import os
import re
import sys
import time
from typing import Generator, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_SOURCES = ["math", "gsm8k", "numina_tir", "logiqa"]

DATASET_IDS = {
    # EleutherAI/hendrycks_math splits the 12 MATH topic categories into separate
    # HuggingFace configs (algebra, geometry, …). We concatenate all of them.
    # The value is (repo_id, configs) where configs is a list (one load per config)
    # or None (load once without a config name).
    "math": (
        "EleutherAI/hendrycks_math",
        [
            "algebra",
            "counting_and_probability",
            "geometry",
            "intermediate_algebra",
            "number_theory",
            "prealgebra",
            "precalculus",
        ],
    ),
    "gsm8k": ("openai/gsm8k", ["main"]),
    "numina_tir": ("AI-MO/NuminaMath-TIR", None),
    "logiqa": ("lucasmccabe/logiqa", None),
}

# Number of completions per problem during difficulty filtering.
# G=8 is the standard GRPO group size; same value used in grpo.py.
# Smaller values make pass_rate estimates noisier; larger values are slower.
FILTER_GROUP_SIZE = 8

# Pass-rate window to keep after difficulty filtering.
# With group_size=8, the granularity is 1/8=0.125 steps. The canonical
# 20-80% window (keep 2-6/8 correct) produced only ~2% keep rate on the
# SFT checkpoint — the model rarely falls in that exact window; most
# problems are pass_rate=0 (too hard) or pass_rate=1 (too easy).
# Widened to 5-95% so that any problem where the model occasionally
# succeeds (≥1/8) or occasionally fails (≤7/8) provides gradient signal.
# GRPO only requires variance within the group: all-zero or all-one groups
# produce zero advantage (no learning); any mix is useful.
PASS_RATE_MIN = 0.05   # keep if ≥1/8 correct  (was 0.20)
PASS_RATE_MAX = 0.95   # keep if ≤7/8 correct  (was 0.80)

# Max new tokens to generate per completion during difficulty filtering.
MAX_GEN_TOKENS = 512

# ---------------------------------------------------------------------------
# Answer extraction helpers
# ---------------------------------------------------------------------------


def extract_boxed(text: str) -> Optional[str]:
    r"""
    Extract the content of the LAST \boxed{...} in the text.

    Why the last occurrence:
      Math solutions often contain \boxed{} mid-solution for sub-results
      (e.g. "so x = \boxed{3} and y = \boxed{5}"). The final \boxed{} is
      the definitive answer. Using the last one is consistent with how
      NuminaMath and MATH datasets are structured.

    Handles nested braces: \boxed{x^{2} + 1} is extracted as "x^{2} + 1".
    The brace-counting loop is necessary because a regex like \boxed{([^}]*)}
    would stop at the first closing brace, missing nested expressions.
    """
    # Find the rightmost \boxed{ occurrence.
    marker = r"\boxed{"
    idx = text.rfind(marker)
    if idx == -1:
        return None

    start = idx + len(marker)
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1

    if depth != 0:
        # Unmatched braces — malformed LaTeX.
        return None

    return text[start : i - 1].strip()


def extract_gsm8k_answer(answer_field: str) -> Optional[str]:
    """
    GSM8K answer fields end with '#### 42' (the numeric answer after a hash separator).

    The full answer field is long-form reasoning followed by the number.
    We extract only the number — that is what the reward verifier will check.

    Example input:  "She has 3 apples.\n#### 3"
    Example output: "3"
    """
    # Match the pattern: #### followed by optional whitespace and a number
    # (possibly with commas as thousands separators, e.g. "1,234").
    match = re.search(r"####\s*([\d,\.\-]+)", answer_field)
    if match:
        # Remove commas from thousands separators so "1,234" == "1234" numerically.
        return match.group(1).replace(",", "").strip()
    return None


def normalize_logiqa_answer(options: list, correct_option: int) -> Optional[str]:
    """
    LogiQA provides options as a list and correct_option as a 0-based index.

    We return the text of the correct option as the answer, which allows
    rewards.py's verify_math_exact (string match after normalization) to
    check it. We also store the option letter (A/B/C/D) in answer_normalized
    for a shorter comparison string.

    Why text instead of index:
      The model generates text, not indices. Comparing model output to the
      text of the correct option is more robust than trying to parse "A" vs
      "option A" vs "the first option" from a generation.
    """
    if not options or correct_option is None:
        return None
    try:
        idx = int(correct_option)
        if 0 <= idx < len(options):
            return options[idx].strip()
    except (ValueError, TypeError):
        pass
    return None


# ---------------------------------------------------------------------------
# Source-specific formatters
# ---------------------------------------------------------------------------


def format_math(example: dict) -> Optional[dict]:
    """
    MATH (Hendrycks) fields: `problem` (str), `solution` (str), `level` (str).

    The solution ends with \\boxed{answer}. We extract it with extract_boxed().
    Level is "Level 1" through "Level 5" — we store just the digit.

    Domain is math_sympy because MATH answers are often algebraic expressions
    (fractions, polynomials, trig values) where string match is insufficient.
    """
    problem = (example.get("problem") or "").strip()
    solution = (example.get("solution") or "").strip()
    level_str = (example.get("level") or "").strip()

    if not problem or not solution:
        return None

    answer = extract_boxed(solution)
    if not answer:
        # Fall back to the full solution if no \boxed{} found — better than
        # discarding the example entirely.
        answer = solution

    # Extract the digit from "Level 3" → "3".
    level_match = re.search(r"\d+", level_str)
    difficulty = level_match.group(0) if level_match else None

    return {
        "prompt": problem,
        "answer": answer,
        "answer_normalized": answer,
        "domain": "math_sympy",
        "source": "math",
        "difficulty": difficulty,
        "pass_rate": None,
    }


def format_gsm8k(example: dict) -> Optional[dict]:
    """
    GSM8K fields: `question` (str), `answer` (str).

    The answer field is a multi-line explanation ending with '#### N'.
    We extract N as the ground-truth answer.

    Domain is math_exact because GSM8K answers are always integers.
    """
    question = (example.get("question") or "").strip()
    answer_field = (example.get("answer") or "").strip()

    if not question or not answer_field:
        return None

    answer = extract_gsm8k_answer(answer_field)
    if not answer:
        return None

    return {
        "prompt": question,
        "answer": answer,
        "answer_normalized": answer,
        "domain": "math_exact",
        "source": "gsm8k",
        "difficulty": None,  # GSM8K has no difficulty metadata
        "pass_rate": None,
    }


def format_numina_tir(example: dict) -> Optional[dict]:
    """
    NuminaMath-TIR fields: `problem` (str), `solution` (str).

    Like MATH, the answer is the content of the last \\boxed{} in the solution.
    NuminaMath-TIR focuses on tool-integrated reasoning (Python code in the
    solution), but the final boxed answer is still the verifiable ground truth.

    Domain is math_sympy because NuminaMath problems span competition math
    with symbolic answers.
    """
    problem = (example.get("problem") or "").strip()
    solution = (example.get("solution") or "").strip()

    if not problem or not solution:
        return None

    answer = extract_boxed(solution)
    if not answer:
        answer = solution

    return {
        "prompt": problem,
        "answer": answer,
        "answer_normalized": answer,
        "domain": "math_sympy",
        "source": "numina_tir",
        "difficulty": None,
        "pass_rate": None,
    }


def format_logiqa(example: dict) -> Optional[dict]:
    """
    LogiQA fields: `query` (str), `options` (list[str]), `correct_option` (int).

    LogiQA is a multiple-choice logical reasoning dataset. We present the
    problem with its options in the prompt (so the model sees the full
    multiple-choice context), and check whether the model selects the right
    option text in its answer.

    Domain is logic (maps to verify_math_exact in rewards.py after stripping).
    """
    query = (example.get("query") or "").strip()
    options = example.get("options") or []
    correct_option = example.get("correct_option")

    if not query or not options:
        return None

    correct_text = normalize_logiqa_answer(options, correct_option)
    if not correct_text:
        return None

    # Build the option letter labels A, B, C, D … for the prompt.
    option_letters = "ABCDEFGHIJ"
    options_str = "\n".join(
        f"{option_letters[i]}. {opt}" for i, opt in enumerate(options) if i < len(option_letters)
    )

    # The prompt includes the options so the model can reason over them.
    full_prompt = f"{query}\n\n{options_str}"

    # answer_normalized is the option letter (A/B/C/D) for compact comparison.
    idx = int(correct_option) if correct_option is not None else 0
    answer_normalized = option_letters[idx] if 0 <= idx < len(option_letters) else correct_text

    return {
        "prompt": full_prompt,
        "answer": correct_text,
        "answer_normalized": answer_normalized,
        "domain": "logic",
        "source": "logiqa",
        "difficulty": None,
        "pass_rate": None,
    }


FORMATTERS = {
    "math": format_math,
    "gsm8k": format_gsm8k,
    "numina_tir": format_numina_tir,
    "logiqa": format_logiqa,
}

# ---------------------------------------------------------------------------
# Dataset loading with streaming
# ---------------------------------------------------------------------------


def stream_source(source: str, limit: Optional[int] = None) -> Generator[dict, None, None]:
    """
    Stream and format examples from a single source.

    Uses streaming=True to keep memory usage flat — same rationale as in
    sft_format.py. Download failures skip the source with a warning rather
    than aborting the build, since the user may not have network access for
    all datasets simultaneously.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: `datasets` library not installed. Run: uv sync", file=sys.stderr)
        return

    repo_id, configs = DATASET_IDS[source]
    formatter = FORMATTERS[source]

    # configs is either None (no sub-config needed) or a list of config name strings.
    # When it's a list we concatenate all sub-configs into a single stream.
    # gsm8k uses ["main"] so it goes through the same list path.
    if configs is None:
        config_list = [None]
    else:
        config_list = configs

    print(f"  Loading {source} ({repo_id}) …", flush=True)

    # Build a combined iterator across all configs.
    def _iter_all_configs():
        for cfg in config_list:
            try:
                if cfg is not None:
                    ds = load_dataset(
                        repo_id, cfg, split="train", streaming=True, trust_remote_code=True
                    )
                else:
                    ds = load_dataset(
                        repo_id, split="train", streaming=True, trust_remote_code=True
                    )
                yield from ds
            except Exception as exc:
                print(
                    f"  WARNING: could not load {source} config={cfg} ({repo_id}): {exc}",
                    file=sys.stderr,
                )

    count = 0
    skipped = 0

    for raw in _iter_all_configs():
        try:
            formatted = formatter(raw)
        except Exception as exc:
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
# Stage 1: Build raw dataset
# ---------------------------------------------------------------------------


def build_dataset(
    sources: list[str],
    output_dir: str,
    limit: Optional[int] = None,
) -> None:
    """
    Collect problems from all sources and write grpo_raw.jsonl.

    Unlike sft_format.py, we do NOT shuffle here — the raw file retains source
    ordering so that it is easy to inspect by source. Shuffling happens inside
    the GRPO DataLoader during training.
    """
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "grpo_raw.jsonl")
    source_counts: dict[str, int] = {}
    total = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for source in sources:
            per_source_limit = (limit // len(sources)) if limit is not None else None
            count = 0

            for ex in stream_source(source, limit=per_source_limit):
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                count += 1

            source_counts[source] = count
            total += count
            print(f"  Running total: {total} problems", flush=True)

    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sources": sources,
        "source_counts": source_counts,
        "total": total,
        "limit_per_source": (limit // len(sources)) if limit is not None else None,
        "pass_rate_min": PASS_RATE_MIN,
        "pass_rate_max": PASS_RATE_MAX,
        "filtered": False,
    }
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nRaw dataset: {total} problems → {output_path}")
    print(f"Manifest    → {manifest_path}")


# ---------------------------------------------------------------------------
# Stage 2: Difficulty filtering
# ---------------------------------------------------------------------------


def filter_by_difficulty(
    input_path: str,
    checkpoint_path: str,
    config_name: str,
    output_path: str,
    group_size: int = FILTER_GROUP_SIZE,
    pass_rate_min: float = PASS_RATE_MIN,
    pass_rate_max: float = PASS_RATE_MAX,
    tokenizer_path: Optional[str] = None,
) -> None:
    """
    Run the SFT model on each problem, estimate pass_rate, and write the
    difficulty-filtered dataset.

    Algorithm:
      For each problem:
        1. Tokenize the prompt.
        2. Generate group_size completions (temperature=1.0 for diversity).
        3. For each completion, extract the answer (text after </think>)
           and verify against the ground truth using the appropriate domain
           verifier from rewards.py.
        4. pass_rate = num_correct / group_size.
        5. Keep the example iff pass_rate_min <= pass_rate <= pass_rate_max.

    Why temperature=1.0 for filtering:
      We want to estimate the TRUE difficulty of the problem for the current
      model policy, not the greedy-decode difficulty. At temperature=0, the
      model always gives the same answer (either always right or always wrong),
      which gives no pass_rate signal. Temperature=1.0 gives diverse completions
      that reflect the model's uncertainty.

    Why G=8:
      G=8 gives pass_rate in {0, 1/8, 2/8, ..., 8/8} = 9 possible values.
      The 20–80% window corresponds to pass_rate in {2/8, 3/8, 4/8, 5/8, 6/8}.
      More completions give finer-grained estimates but scale linearly with cost.
      G=8 is also the group size used during GRPO training, so difficulty estimates
      are consistent between filtering and training.

    Args:
      input_path:      Path to grpo_raw.jsonl (output of build_dataset).
      checkpoint_path: Path to SFT checkpoint (.pt file).
      config_name:     Model config key ("500m", "1b", "3b").
      output_path:     Where to write grpo_filtered.jsonl.
      group_size:      Number of completions per problem (default 8).
      pass_rate_min:   Lower bound of the difficulty window (default 0.20).
      pass_rate_max:   Upper bound of the difficulty window (default 0.80).
    """
    import torch
    from training.rewards import compute_reward, _extract_answer

    # Import the model architecture from the project package.
    # training/grpo.py uses the same pattern.
    from model.architecture import SmallReasoningModel, CONFIGS

    print(f"Loading model config '{config_name}' …", flush=True)
    if config_name not in CONFIGS:
        print(
            f"ERROR: unknown config '{config_name}'. Available: {list(CONFIGS.keys())}",
            file=sys.stderr,
        )
        sys.exit(1)

    model_cfg = CONFIGS[config_name]
    model = SmallReasoningModel(model_cfg)

    print(f"Loading checkpoint from {checkpoint_path} …", flush=True)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # Checkpoints saved by training/sft.py store the state dict under "model".
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}", flush=True)
    model = model.to(device)

    # Load the tokenizer. We expect a HuggingFace-compatible tokenizer saved
    # by srm-tokenizer (tokenizer/train_tokenizer.py). Resolution order:
    #   1. Explicit --tokenizer path (if provided)
    #   2. tokenizer_output/ relative to CWD (project root convention)
    #   3. tokenizer_output/ relative to checkpoint directory
    try:
        from transformers import PreTrainedTokenizerFast

        if tokenizer_path and os.path.isdir(tokenizer_path):
            tokenizer_dir = tokenizer_path
        elif os.path.isdir("tokenizer_output"):
            # Most common case: running from project root
            tokenizer_dir = "tokenizer_output"
        else:
            # Last resort: look next to the checkpoint
            tokenizer_dir = os.path.join(os.path.dirname(checkpoint_path), "tokenizer_output")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_dir)
        print(f"Tokenizer loaded from {tokenizer_dir}", flush=True)
    except Exception as exc:
        print(f"ERROR: could not load tokenizer: {exc}", file=sys.stderr)
        sys.exit(1)

    # Read all problems from the raw JSONL.
    problems = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                problems.append(json.loads(line))

    print(f"Filtering {len(problems)} problems (group_size={group_size}) …", flush=True)

    kept = 0
    total = len(problems)

    with open(output_path, "w", encoding="utf-8") as out_f:
        for idx, problem in enumerate(problems):
            prompt = problem["prompt"]
            ground_truth = problem["answer_normalized"]
            domain = problem["domain"]

            # Wrap in the instruction format the SFT model was trained on, and
            # seed the assistant turn with <think> to put the model in chain-of-
            # thought mode. Seeding <think> empirically improves answer accuracy
            # even when the model doesn't close with </think> — the model produces
            # more structured reasoning rather than free-form prose.
            formatted_prompt = f"User: {prompt}\nAssistant: <think>"

            # Encode and strip the trailing EOS that the tokenizer post-processor
            # always appends. We want the model to generate the response, not
            # predict tokens after end-of-sequence (which produces garbage output).
            enc = tokenizer(formatted_prompt, return_tensors="pt", add_special_tokens=True)
            ids = enc["input_ids"][0]
            eos_id = tokenizer.eos_token_id or 2
            if ids[-1].item() == eos_id:
                ids = ids[:-1]
            input_ids = ids.unsqueeze(0).to(device)

            correct = 0

            with torch.no_grad():
                for _ in range(group_size):
                    # Generate one completion using our KV-cache autoregressive loop.
                    # SmallReasoningModel.forward() returns (logits, kv_caches) —
                    # not a HuggingFace model, so no .generate() method.
                    #
                    # Prefill: single forward pass over the full prompt.
                    # kv_caches=[] = collect KV but no prior cache to prepend.
                    logits, kv_caches = model(input_ids, kv_caches=[])
                    generated: list[int] = []
                    prompt_len = input_ids.shape[1]

                    for step_i in range(MAX_GEN_TOKENS):
                        # Temperature=1.0: sample from the unscaled distribution
                        # for diverse completions (needed for pass_rate estimation).
                        next_logits = logits[0, -1, :].float()
                        probs = torch.softmax(next_logits, dim=-1)
                        next_id = int(torch.multinomial(probs, num_samples=1).item())
                        generated.append(next_id)

                        # Stop on EOS
                        eos_id = tokenizer.eos_token_id or 2
                        if next_id == eos_id:
                            break

                        # Decode step: one new token, reuse KV cache.
                        next_input = torch.tensor([[next_id]], dtype=torch.long, device=device)
                        logits, kv_caches = model(
                            next_input,
                            kv_caches=kv_caches,
                            position_offset=prompt_len + step_i,
                        )

                    completion = tokenizer.decode(generated, skip_special_tokens=True)

                    # The prompt was seeded with "<think>", so the completion
                    # is everything AFTER "<think>". Prepend it so that
                    # _extract_answer can find </think> if the model generates it,
                    # and so that the format_reward check for <think>...</think>
                    # works correctly. format_weight=0.0 here so format doesn't
                    # influence pass_rate estimation.
                    full_response = "<think>" + completion
                    reward = compute_reward(
                        response=full_response,
                        ground_truth=ground_truth,
                        domain=domain,
                        format_weight=0.0,
                    )
                    if reward >= 0.5:
                        # Threshold at 0.5: for binary rewards (math/logic) this is
                        # exact; for code (fractional) it means at least half the
                        # test cases pass.
                        correct += 1

            pass_rate = correct / group_size

            # Update the example with the measured pass_rate.
            problem["pass_rate"] = pass_rate

            if pass_rate_min <= pass_rate <= pass_rate_max:
                out_f.write(json.dumps(problem, ensure_ascii=False) + "\n")
                kept += 1

            # Progress logging every 100 problems.
            if (idx + 1) % 100 == 0:
                print(
                    f"  {idx + 1}/{total} processed, {kept} kept "
                    f"({100 * kept / (idx + 1):.1f}% kept so far)",
                    flush=True,
                )

    print(f"\nDone. {kept}/{total} problems kept ({100 * kept / max(total, 1):.1f}%)")
    print(f"Filtered dataset → {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and difficulty-filter the GRPO verifiable problem dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build raw dataset (no model needed)
  python data/grpo_dataset.py --output_dir ./data/grpo --sources all

  # Filter by difficulty (needs SFT checkpoint)
  python data/grpo_dataset.py \\
    --filter \\
    --input ./data/grpo/grpo_raw.jsonl \\
    --checkpoint ./checkpoints/1b_sft/best.pt \\
    --config 1b \\
    --output ./data/grpo/grpo_filtered.jsonl
        """,
    )

    # --- Build mode arguments ---
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for grpo_raw.jsonl and manifest.json (build mode).",
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
        help="Total problem cap across all sources (distributed evenly). For testing.",
    )

    # --- Filter mode arguments ---
    parser.add_argument(
        "--filter",
        action="store_true",
        help="Run difficulty filtering (Stage 2). Requires --input, --checkpoint, --config.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to grpo_raw.jsonl (input for filter mode).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to SFT checkpoint .pt file (filter mode).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="1b",
        help="Model config key: 500m | 1b | 3b. Default: 1b.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for grpo_filtered.jsonl (filter mode).",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Path to tokenizer directory (filter mode). Default: auto-detect tokenizer_output/.",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=FILTER_GROUP_SIZE,
        help=f"Completions per problem during filtering. Default: {FILTER_GROUP_SIZE}.",
    )
    parser.add_argument(
        "--pass_rate_min",
        type=float,
        default=PASS_RATE_MIN,
        help=f"Minimum pass rate to keep (inclusive). Default: {PASS_RATE_MIN}.",
    )
    parser.add_argument(
        "--pass_rate_max",
        type=float,
        default=PASS_RATE_MAX,
        help=f"Maximum pass rate to keep (inclusive). Default: {PASS_RATE_MAX}.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.filter:
        # --- Stage 2: difficulty filtering ---
        missing = []
        if not args.input:
            missing.append("--input")
        if not args.checkpoint:
            missing.append("--checkpoint")
        if not args.output:
            missing.append("--output")
        if missing:
            print(
                f"ERROR: filter mode requires: {', '.join(missing)}",
                file=sys.stderr,
            )
            sys.exit(1)

        filter_by_difficulty(
            input_path=args.input,
            checkpoint_path=args.checkpoint,
            config_name=args.config,
            output_path=args.output,
            group_size=args.group_size,
            pass_rate_min=args.pass_rate_min,
            pass_rate_max=args.pass_rate_max,
            tokenizer_path=args.tokenizer,
        )

    else:
        # --- Stage 1: build raw dataset ---
        if not args.output_dir:
            print("ERROR: --output_dir is required in build mode.", file=sys.stderr)
            sys.exit(1)

        if args.sources.strip().lower() == "all":
            sources = ALL_SOURCES
        else:
            sources = [s.strip() for s in args.sources.split(",") if s.strip()]
            unknown = [s for s in sources if s not in ALL_SOURCES]
            if unknown:
                print(
                    f"ERROR: unknown source(s): {unknown}. Available: {ALL_SOURCES}",
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
        )


if __name__ == "__main__":
    main()
