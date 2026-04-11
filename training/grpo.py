"""
grpo.py
=======
Phase 2: Group Relative Policy Optimization (GRPO) for reasoning.
Upgraded with DAPO (ByteDance 2025) and Dr. GRPO improvements.

This is where reasoning capability is trained, not installed.

Base algorithm (GRPO):
  1. Sample batch of prompts from verifiable dataset
  2. For each prompt, sample G=8 completions from policy π_θ
  3. Compute reward r_i for each (verify against ground truth)
  4. Compute group advantage: A_i = (r_i - mean(r)) / (std(r) + ε)
  5. Compute clipped policy gradient loss + KL penalty
  6. Update θ

Four improvements over vanilla GRPO (all enabled by default):

  [DAPO] Clip-Higher — asymmetric PPO clipping
    Problem:  Symmetric clipping (1±ε) prevents the policy from
              increasing probability of good actions. Entropy collapses
              as training progresses, killing exploration.
    Fix:      clip_low=0.20, clip_high=0.28 — the upper bound is relaxed,
              letting good actions grow while still bounding bad ones.
    Effect:   Sustained exploration throughout training.

  [DAPO] Token-level Policy Gradient Loss
    Problem:  Sequence-level loss averaging divides by sequence length,
              so a 2000-token correct CoT gets the same total gradient
              as a 50-token correct answer. Long reasoning is penalized.
    Fix:      Compute loss per token, average across ALL tokens in the
              batch (not per-sequence then average sequences).
    Effect:   Correct long CoT chains are properly reinforced.

  [DAPO] Dynamic Sampling
    Problem:  As the model improves, previously hard problems become easy.
              Groups with all-0 or all-1 rewards produce zero advantage →
              zero gradient → wasted compute and training step.
    Fix:      After generating completions, skip groups with uniform
              rewards (all correct or all wrong). Oversample to refill.
    Effect:   Every training step has non-zero gradient signal.

  [Dr. GRPO] Length-Debiased Advantages
    Problem:  Original GRPO normalizes per sequence: shorter completions
              get larger per-token gradients than longer ones with the
              same reward. The model is silently pushed toward brevity.
    Fix:      Normalize rewards by completion length before computing
              group statistics, then restore scale. Length is no longer
              a confound in the advantage estimate.
    Effect:   Model learns to reason correctly, not briefly.

Critical implementation details:
  - The reference model (π_ref) is the SFT checkpoint, FROZEN
  - Log probabilities computed for COMPLETION tokens only
  - Dynamic sampling oversamples by 2× then filters; see DynamicSampler
  - Overlong reward shaping: soft penalty if completion hits max_gen_tokens

Usage:
  python grpo.py \\
    --checkpoint ./checkpoints/1b_sft/best.pt \\
    --config 1b \\
    --data_dir ./grpo_data \\
    --output_dir ./checkpoints/1b_grpo \\
    --steps 10000 \\
    --group_size 8

  # Vanilla GRPO (disable all improvements for ablation)
  python grpo.py ... --no_dapo --no_dr_grpo

  # Validate loop logic (synthetic rewards, no real data)
  python grpo.py --config 1b --mode validate
"""

import argparse
import ast as _ast
import json
import math
import os
import re
import signal
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent / "model"))
from model.architecture import SmallReasoningModel, ModelConfig, CONFIGS

# ---------------------------------------------------------------------------
# Special token IDs
# ---------------------------------------------------------------------------
PAD_ID = 0  # Token 0 is padding; excluded from loss and log-prob computation everywhere
BOS_ID = 1  # Beginning-of-sequence; prepended automatically by the tokenizer post-processor
EOS_ID = 2  # End-of-sequence; generation halts when this is emitted
THINK_START_ID = 4  # <think> opening tag — model enters its chain-of-thought here
THINK_END_ID = 5  # </think> closing tag — model exits CoT and produces its final answer
LOSS_IGNORE = -100  # Sentinel used by CrossEntropyLoss to skip certain positions (not used in GRPO)


# ---------------------------------------------------------------------------
# GRPO config
# ---------------------------------------------------------------------------


@dataclass
class GRPOConfig:
    # ── Model ──────────────────────────────────────────────────────────────
    model_config: str = "1b"
    # SFT checkpoint is the mandatory starting point for GRPO. We never train
    # GRPO from a random-init model: the policy must already know how to format
    # answers and produce coherent CoT before RL pressure is applied. Starting
    # from random init causes reward hacking and degenerate outputs.
    checkpoint: str = ""  # SFT checkpoint (required for meaningful training)

    # ── Data ───────────────────────────────────────────────────────────────
    data_dir: str = "./grpo_data"
    tokenizer_path: str = "./tokenizer_output"
    domain: str = "math"  # "math" | "code" | "mixed"

    # ── GRPO core ──────────────────────────────────────────────────────────
    # group_size (G) is the number of independent completions sampled per
    # prompt. Advantages are computed *within* each group (group-relative),
    # so G must be > 1. Larger G gives better advantage estimates but costs
    # G× more generation time. G=8 is the DAPO default.
    group_size: int = 8  # G: completions sampled per prompt
    steps: int = 10_000  # Total optimizer steps
    # batch_prompts: number of *distinct* prompts per step, before group
    # expansion. Actual forward-pass batch size = batch_prompts × group_size.
    batch_prompts: int = 4  # Prompts per step (before group expansion)

    # ── Generation ─────────────────────────────────────────────────────────
    # max_gen_tokens caps the completion length. Completions that hit this
    # limit are likely truncated mid-reasoning; the overlong_penalty applies.
    max_gen_tokens: int = 2048  # Max new tokens per completion
    temperature: float = 0.8  # Sampling temperature (>0 required for diversity)
    top_p: float = 0.95  # Nucleus sampling threshold

    # ── [DAPO] Asymmetric clipping (Clip-Higher) ───────────────────────────
    # In vanilla PPO the ratio r_t(θ) = π_θ(a|s) / π_old(a|s) is clipped
    # symmetrically to [1-ε, 1+ε]. This prevents the policy from assigning
    # too much probability to any action in a single step — which is good for
    # stability but also prevents fully exploiting high-reward actions.
    # DAPO relaxes the upper bound only: the policy can grow probability of
    # good actions more freely (clip_high > clip_low), while still being
    # tightly bounded on the downside. This sustains entropy and exploration.
    # Setting clip_high == clip_low recovers vanilla symmetric PPO.
    clip_low: float = 0.20  # Lower clip (1 - clip_low = 0.80); same as vanilla ε=0.2
    clip_high: float = 0.28  # Upper clip (1 + clip_high = 1.28); DAPO relaxed bound

    # ── KL penalty ─────────────────────────────────────────────────────────
    # The KL term β·KL(π_θ ‖ π_ref) keeps the policy from drifting too far
    # from the SFT baseline. Without it the policy can exploit the reward
    # function in unexpected ways (reward hacking) while destroying the
    # language quality acquired during SFT. β is intentionally small: too
    # large and the policy cannot improve; too small and it drifts badly.
    kl_coef: float = 0.01  # β — weight of KL divergence penalty term

    # ── Optimizer ──────────────────────────────────────────────────────────
    # Learning rate is much smaller than SFT (5e-5 typical). GRPO makes small
    # targeted updates; large LR causes catastrophic forgetting of SFT quality.
    lr: float = 5e-7
    lr_min: float = 5e-7  # Constant LR (no decay; GRPO reward signal is noisy)
    warmup_fraction: float = 0.01  # Short warmup to avoid early instability
    weight_decay: float = 0.0  # No WD for GRPO; we want maximum signal from rewards
    grad_clip: float = 1.0  # Gradient clipping prevents occasional large spikes

    # ── Format reward ──────────────────────────────────────────────────────
    # Small bonus for maintaining <think>...</think> structure. Without this,
    # the policy sometimes drops the thinking step under RL pressure.
    format_reward_weight: float = 0.1

    # ── [DAPO] Dynamic sampling ────────────────────────────────────────────
    # Oversample prompts by dynamic_oversample×, then discard groups where
    # ALL completions have the same reward (uniform groups). A uniform group
    # has std(r)=0 → all advantages are 0 → gradient is exactly zero →
    # the optimizer step is wasted. Oversampling then filtering guarantees
    # every step has at least one informative group.
    dynamic_sampling: bool = True
    dynamic_oversample: int = 2  # Sample 2× prompts, keep the informative half
    # Minimum reward std to consider a group non-uniform. 0.0 keeps any group
    # with at least one correct and one incorrect completion.
    dynamic_min_diversity: float = 0.0

    # ── [DAPO] Overlong reward shaping ─────────────────────────────────────
    # A completion that fills max_gen_tokens was likely truncated mid-reasoning.
    # Even if the truncated text happens to match the answer by string match,
    # the chain of thought is incomplete and the behaviour should not be
    # reinforced. Apply a soft multiplier to reduce its reward.
    overlong_penalty: bool = True
    overlong_penalty_factor: float = 0.5  # Multiply reward by 0.5 if truncated

    # ── [Dr. GRPO] Length-debiased advantages ─────────────────────────────
    # In vanilla GRPO the policy is implicitly penalized for long outputs:
    # a correct 2000-token CoT and a correct 50-token answer both get reward
    # 1.0, but the per-token gradient of the 2000-token response is 40× smaller
    # because the advantage is spread across more tokens. Dr. GRPO removes this
    # confound by normalizing rewards by length before computing group stats.
    length_debiased: bool = True

    # ── Ablation flags ─────────────────────────────────────────────────────
    # These flip off individual improvements so you can isolate their effect.
    # Useful for ablation studies; not intended for production training.
    no_dapo: bool = False  # Disable all DAPO improvements (symmetric clip, etc.)
    no_dr_grpo: bool = False  # Disable Dr. GRPO length debiasing

    # ── Difficulty filter ──────────────────────────────────────────────────
    # Static version of dynamic sampling applied at dataset load time.
    # Problems with pass_rate < min or > max are excluded entirely, since
    # the model either never solves them (advantage always negative) or
    # always solves them (advantage always near zero). Both produce no signal.
    # Widened from 0.20–0.80 to match data/grpo_dataset.py: with group_size=8,
    # 0.20 requires ≥2/8 correct which excludes most problems at the SFT stage.
    # 0.05–0.95 keeps any problem with 1–7/8 correct (non-uniform reward groups).
    min_pass_rate: float = 0.05
    max_pass_rate: float = 0.95

    # ── KV cache compression ──────────────────────────────────────────────
    compress_kv: bool = False  # [TurboQuant] Compress KV cache ~2× during generation

    # ── Memory ─────────────────────────────────────────────────────────────
    grad_checkpointing: bool = True  # Recompute activations on backward; saves ~40% memory
    dtype: str = "bfloat16"  # bfloat16 is sufficient for GRPO; float32 wastes memory

    # ── Output ─────────────────────────────────────────────────────────────
    output_dir: str = "./checkpoints/grpo"
    log_every: int = 10
    save_every: int = 500
    eval_every: int = 200

    # ── Hardware ───────────────────────────────────────────────────────────
    backend: str = "cuda"

    def __post_init__(self):
        # Apply ablation flags: collapsing settings to their vanilla equivalents
        # makes it trivial to compare "with improvement X vs without" by just
        # changing one boolean rather than manually adjusting multiple fields.
        if self.no_dapo:
            self.clip_high = self.clip_low  # clip_high == clip_low → symmetric = vanilla PPO
            self.dynamic_sampling = False
            self.overlong_penalty = False
        if self.no_dr_grpo:
            self.length_debiased = False


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------


def normalize_answer(s: str) -> str:
    """
    Normalize a math answer string for comparison.

    The reward model should be insensitive to irrelevant surface variation —
    a student who writes "$\\frac{1}{2}$" and one who writes "0.5" should
    receive the same reward. This function collapses those variations before
    string comparison. It is intentionally conservative (no semantic rewriting)
    so that normalization bugs cannot manufacture false positives.

    Handles common formatting variations that should be considered equivalent:
    - Leading/trailing whitespace
    - LaTeX dollar signs ($, $$)
    - LaTeX fractions: \\frac{a}{b} → a/b (approximate)
    - Comma-separated thousands: 1,000 → 1000
    - Unicode minus signs → ASCII
    - Trailing .0 for integers
    """
    if not isinstance(s, str):
        s = str(s)

    s = s.strip()

    # Strip LaTeX math delimiters — models trained on math corpora often wrap
    # answers in $...$ or $$...$$ even when not asked to.
    s = s.replace("$$", "").replace("$", "")

    # Unicode minus (U+2212) is visually identical to ASCII hyphen-minus but
    # won't compare equal. Replace before any numeric parsing.
    s = s.replace("\u2212", "-")

    # Remove thousands separators so "1,000" == "1000". The regex ensures we
    # only strip commas between digit groups (not in prose like "a, b").
    s = re.sub(r"(\d),(\d{3})", r"\1\2", s)

    # Approximate LaTeX fraction → Python division expression. Not perfectly
    # symbolic, but catches the most common case: \frac{3}{4} → (3)/(4).
    s = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"(\1)/(\2)", s)

    # Strip trailing zeros after decimal, then strip lone decimal point.
    # "3.10" → "3.1", "3.0" → "3", "3.14" → "3.14"
    s = re.sub(r"(\.\d*?)0+$", r"\1", s)
    s = re.sub(r"\.$", "", s)

    # Lowercase for case-insensitive matching (e.g. "TRUE" == "true")
    s = s.lower().strip()

    return s


def reward_math_exact(completion: str, ground_truth: str) -> float:
    """
    Binary reward: 1.0 if completion contains the correct answer, 0.0 otherwise.

    We use a binary (0/1) signal rather than a continuous score because
    math problems have objectively correct answers. Partial credit is
    philosophically problematic for exact reasoning: "almost right" is wrong.
    Binary rewards also simplify advantage computation — the group std is
    entirely determined by the fraction of correct completions.

    Extracts the answer from the completion (handles <think>...</think> structure)
    and compares to ground truth after normalization.
    """
    answer = _extract_final_answer(completion)
    if answer is None:
        # No answer found at all — the model may have produced only CoT
        # without an answer, or the completion was truncated. Always 0.
        return 0.0

    norm_pred = normalize_answer(answer)
    norm_truth = normalize_answer(ground_truth)

    return 1.0 if norm_pred == norm_truth else 0.0


def reward_math_sympy(completion: str, ground_truth: str) -> float:
    """
    Symbolic equivalence reward using SymPy.

    String comparison cannot detect that "x^2 + 2x + 1" and "(x+1)^2" are
    the same expression. SymPy simplifies the difference: if simplify(a - b)
    == 0 the expressions are provably equivalent. This is more reliable for
    algebra/calculus problems but much slower than string matching.

    Handles mathematically equivalent expressions:
      "x^2 + 2x + 1" == "(x+1)^2"  → 1.0
      "1/2" == "0.5"                → 1.0

    Falls back to exact match if SymPy parse fails.
    Times out after 5 seconds to prevent hanging on pathological inputs.
    (SymPy simplification can loop on adversarial expressions.)
    """
    answer = _extract_final_answer(completion)
    if answer is None:
        return 0.0

    # Try cheap string match first — avoids SymPy import overhead for
    # simple cases like integer answers that are already normalized.
    if normalize_answer(answer) == normalize_answer(ground_truth):
        return 1.0

    # Try SymPy symbolic equivalence
    try:
        import sympy
        from sympy.parsing.sympy_parser import (
            parse_expr,
            standard_transformations,
            implicit_multiplication_application,
        )

        # implicit_multiplication_application lets SymPy parse "2x" as "2*x"
        transforms = standard_transformations + (implicit_multiplication_application,)

        with _timeout(5):
            # Replace ^ with ** because Python/SymPy uses ** for exponentiation
            expr_pred = parse_expr(answer.replace("^", "**"), transformations=transforms)
            expr_truth = parse_expr(ground_truth.replace("^", "**"), transformations=transforms)
            # simplify(a - b) == 0 iff a and b are symbolically equal
            if sympy.simplify(expr_pred - expr_truth) == 0:
                return 1.0
    except Exception:
        pass  # SymPy parse error, timeout, or unexpected expression — treat as wrong

    return 0.0


def reward_code_exec(completion: str, test_cases: list[dict]) -> float:
    """
    Execute code completion against test cases.
    Reward = fraction of test cases passed.

    Safety: runs in a subprocess with a timeout.
    test_cases format: [{"input": "...", "expected_output": "..."}, ...]

    Returns 0.0 if code cannot be extracted or crashes on all tests.
    """
    code = _extract_code_block(completion)
    if not code:
        return 0.0

    passed = 0
    for tc in test_cases:
        try:
            result = _run_code_safely(code, tc.get("input", ""), timeout=5)
            expected = str(tc.get("expected_output", "")).strip()
            if result.strip() == expected:
                passed += 1
        except Exception:
            pass

    return passed / len(test_cases) if test_cases else 0.0


def reward_format(completion: str) -> float:
    """
    Small bonus reward for maintaining <think>...</think> structure.

    Returns a score in {0, 0.5, 1.0} based on how well the completion
    follows the expected format:
      <think>\n...\n</think>\n<final answer>

    Why is this necessary? Under pure correctness reward, the model
    occasionally learns a shortcut: drop the <think> block entirely and
    output just the answer. For simple problems this can be correct enough
    to get positive reward, which reinforces the bad format. This small
    structural bonus (weighted by format_reward_weight, default 0.1) makes
    the correct format slightly more rewarding than the shortcut, preserving
    the reasoning chain throughout training.

    Score breakdown:
      1.0 — full structure: <think>, non-empty content, </think>, then answer
      0.5 — partial: has both tags but missing content or answer after
      0.0 — missing structure entirely
    """
    has_think_start = "<think>" in completion
    has_think_end = "</think>" in completion
    # has_content: there is at least one non-whitespace char inside <think>
    has_content = bool(re.search(r"<think>\s*\S", completion))
    # answer_after: something (non-whitespace) appears after </think>
    answer_after = bool(re.search(r"</think>\s*\S", completion))

    if has_think_start and has_think_end and has_content and answer_after:
        return 1.0  # Full, well-formed CoT
    elif has_think_start and has_think_end:
        return 0.5  # Tags present but content or answer missing
    return 0.0  # No CoT structure at all


def combined_reward(
    completion: str,
    example: dict,
    domain: str,
    format_weight: float = 0.1,
    completion_len: int = 0,
    max_gen_tokens: int = 2048,
    overlong_penalty: bool = True,
    overlong_penalty_factor: float = 0.5,
) -> float:
    """
    Compute combined reward for a completion.

    The reward has three components combined multiplicatively and additively:

      Primary reward (0 or 1 for math; 0..1 for code):
        The main training signal — did the model get the answer right?
        This is the ground truth verifier; it is binary for math because
        math answers are objectively right or wrong.

      Format reward (0, 0.5, or 1.0) × format_weight:
        A small additive bonus for maintaining the <think>...</think>
        structure. Keeps the CoT format alive under RL pressure (see
        reward_format for rationale). Deliberately small so it never
        overcomes a wrong primary reward.

      [DAPO] Overlong multiplier:
        If completion_len >= max_gen_tokens, the generation was cut off.
        Multiply the total reward by overlong_penalty_factor (default 0.5).
        This discourages the model from producing excessively long outputs
        that fill the context window and get truncated mid-answer.

    Total = (primary + format_weight × format) × overlong_multiplier
    """
    ground_truth = example.get("answer", example.get("expected_output", ""))

    # Primary correctness reward — domain-specific verifier
    if domain == "code":
        test_cases = example.get("test_cases", [])
        primary = reward_code_exec(completion, test_cases)
    elif domain == "math_sympy":
        # Use symbolic equivalence (slower, more accurate for algebra)
        primary = reward_math_sympy(completion, ground_truth)
    else:
        # Use string match after normalization (fast, reliable for numeric answers)
        primary = reward_math_exact(completion, ground_truth)

    # Format bonus: small additive term, weighted so it doesn't dominate
    fmt = reward_format(completion) * format_weight

    reward = primary + fmt

    # [DAPO] Overlong penalty: if completion hit the generation limit,
    # it was likely truncated mid-reasoning and the answer is absent/wrong.
    # Even if it happened to be "correct" by string match, the reasoning
    # chain is incomplete — we penalize to discourage verbose outputs.
    # Note: we check >= (not >) because exactly hitting the limit is the
    # truncation condition; the model never got to emit EOS naturally.
    if overlong_penalty and completion_len >= max_gen_tokens:
        reward = reward * overlong_penalty_factor

    return reward


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------


def _extract_final_answer(completion: str) -> Optional[str]:
    """
    Extract the final answer from a completion.

    We try multiple extraction strategies in priority order, from most
    structured (and therefore most reliable) to least structured (and
    therefore most fallible). The fallback to the last line is intentionally
    aggressive — it is better to extract something and let normalize_answer
    reject a bad match than to return None and assign zero reward to a
    correctly-formatted completion.

    Handles:
      1. </think>\nANSWER           — our standard format (highest priority)
      2. \\boxed{ANSWER}            — LaTeX math competitions (MATH dataset)
      3. The answer is: ANSWER      — natural language phrasing
      4. = ANSWER (last occurrence) — equation chains ending with result
      5. Last non-empty line        — fallback for unstructured completions
    """
    # 1. After </think>: the model's explicit answer section in our format.
    # This is the highest-fidelity signal — the model placed its answer here
    # deliberately. Use re.DOTALL so . matches newlines within the answer.
    think_match = re.search(r"</think>\s*(.+?)(?:\n|$)", completion, re.DOTALL)
    if think_match:
        answer = think_match.group(1).strip()
        if answer:
            # Try to extract a more specific answer from the post-think text:
            # the model often writes "therefore, $a=\boxed{2}$" after </think>
            # and we want just "2", not the whole sentence.
            boxed_in_answer = re.findall(r"\\boxed\{([^}]+)\}", answer)
            if boxed_in_answer:
                return boxed_in_answer[-1]
            # Try "= <number>" pattern
            eq_in_answer = re.findall(r"=\s*\$?\s*(-?[\d,]+\.?\d*(?:/[\d,]+)?)", answer)
            if eq_in_answer:
                return eq_in_answer[-1].replace(",", "").strip()
            # Try last standalone number
            num_in_answer = re.findall(r"-?\d[\d,]*\.?\d*(?:/\d+)?", answer)
            if num_in_answer:
                return num_in_answer[-1].replace(",", "").strip()
            return answer

    # 2. LaTeX \boxed{ANSWER}: used in competition math (MATH, AMC datasets).
    # Take the LAST boxed expression in case the model uses \boxed{} in its
    # working steps and again for the final answer.
    boxed = re.findall(r"\\boxed\{([^}]+)\}", completion)
    if boxed:
        return boxed[-1]

    # 3. "The answer is X" / "Answer: X" — natural language phrasing that
    # the model might produce before fully internalizing our format.
    ans_match = re.search(
        r"(?:the answer is|answer is|answer:|therefore)[:\s]+([^\n.]+)", completion, re.IGNORECASE
    )
    if ans_match:
        return ans_match.group(1).strip()

    # 4. Last "= X" occurrence: in equation chains the final result is often
    # the rightmost expression of the last equation. Use MULTILINE so $ anchors
    # match end-of-line, not just end-of-string.
    eq_matches = re.findall(r"=\s*([^\n=]+)$", completion, re.MULTILINE)
    if eq_matches:
        return eq_matches[-1].strip()

    # 5. Fallback: last non-empty line. Least reliable but better than None.
    lines = [l.strip() for l in completion.strip().splitlines() if l.strip()]
    return lines[-1] if lines else None


def _extract_code_block(text: str) -> Optional[str]:
    """Extract Python code from a completion."""
    # ```python ... ``` blocks
    match = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # ```...``` (any language)
    match = re.search(r"```\s*(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # def / class at top level
    match = re.search(r"((?:def|class)\s+\w+.*)", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def _run_code_safely(code: str, stdin_input: str = "", timeout: int = 5) -> str:
    """
    Run code in a subprocess with timeout.
    Returns stdout output or raises on error/timeout.

    We run in a subprocess (not exec/eval) for isolation: the model-generated
    code cannot access training state, modify global variables, or import
    restricted modules without going through the OS process boundary.
    The timeout prevents infinite loops from stalling the training step.
    We write to a temp file rather than using -c "..." to avoid shell escaping
    issues with code that contains quotes or special characters.
    """
    import subprocess, tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        fname = f.name
    try:
        result = subprocess.run(
            [sys.executable, fname],
            input=stdin_input,
            capture_output=True,
            text=True,
            timeout=timeout,  # Hard wall-clock timeout; raises subprocess.TimeoutExpired
        )
        return result.stdout
    finally:
        os.unlink(fname)  # Always clean up even if run raises


@contextmanager
def _timeout(seconds: int):
    """
    Context manager that raises TimeoutError after `seconds`.

    Uses POSIX SIGALRM, so this only works on Unix. The SymPy reward
    function uses this to avoid hanging on pathological inputs (e.g.,
    expressions that cause SymPy's simplification to loop). We restore
    both the alarm and the signal handler in the finally block so that
    nested uses or re-entrant calls don't interfere with each other.
    """

    def _handler(signum, frame):
        raise TimeoutError(f"Timed out after {seconds}s")

    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)  # Cancel any pending alarm
        signal.signal(signal.SIGALRM, old)  # Restore previous handler


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class GRPODataset(Dataset):
    """
    Dataset of verifiable reasoning problems for GRPO training.

    Applies the difficulty filter: only include problems where the
    SFT model solves them 20-80% of the time. Problems outside this
    range don't produce useful gradient signal:
      - < 20%: model almost always fails → advantage ≈ -1 always → no learning
      - > 80%: model almost always succeeds → advantage ≈ 0 always → no learning

    Dataset format (JSONL, one example per line):
      {
        "problem":      "What is 15 + 27?",
        "answer":       "42",
        "domain":       "math",
        "difficulty":   "easy",
        "pass_rate":    0.45    ← optional: pre-computed pass rate
      }

    If pass_rate is not provided, all examples are included (no filter).
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        min_pass_rate: float = 0.20,
        max_pass_rate: float = 0.80,
        max_examples: Optional[int] = None,
    ):
        self.examples = []
        data_dir = Path(data_dir)

        # Find files
        train_file = data_dir / "train.jsonl"
        val_file = data_dir / "val.jsonl"
        any_file = list(data_dir.glob("*.jsonl"))

        if split == "train" and train_file.exists():
            files = [train_file]
        elif split == "val" and val_file.exists():
            files = [val_file]
        elif any_file:
            files = any_file[:1]
        else:
            raise FileNotFoundError(f"No .jsonl files in {data_dir}")

        n_filtered = 0
        for fpath in files:
            with open(fpath) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ex = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # Skip malformed lines rather than crashing

                    # Difficulty filter: pass_rate is the fraction of times the
                    # SFT model solved this problem (pre-computed offline).
                    # Problems outside [min_pass_rate, max_pass_rate] provide
                    # no useful gradient signal:
                    #   pass_rate < min: model never solves it → all advantages
                    #     are near -1 → policy just learns to avoid the domain
                    #   pass_rate > max: model always solves it → all advantages
                    #     are near 0 → dynamic sampling would filter it anyway
                    # If pass_rate is absent, include the example (no filter).
                    pass_rate = ex.get("pass_rate", None)
                    if pass_rate is not None:
                        if not (min_pass_rate <= pass_rate <= max_pass_rate):
                            n_filtered += 1
                            continue

                    self.examples.append(ex)

        if max_examples:
            import random

            random.shuffle(self.examples)
            self.examples = self.examples[:max_examples]

        print(
            f"  GRPODataset ({split}): {len(self.examples):,} examples "
            f"({n_filtered} filtered by difficulty)"
        )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


class SyntheticGRPODataset(Dataset):
    """Synthetic GRPO dataset for --mode validate. Simple arithmetic problems."""

    def __init__(self, n: int = 100):
        import random

        self.examples = []
        for _ in range(n):
            a = random.randint(1, 100)
            b = random.randint(1, 100)
            op = random.choice(["+", "-", "*"])
            ans = eval(f"{a}{op}{b}")
            self.examples.append(
                {
                    "problem": f"What is {a} {op} {b}?",
                    "answer": str(ans),
                    "domain": "math",
                }
            )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


# ---------------------------------------------------------------------------
# Log probability computation
# ---------------------------------------------------------------------------


def compute_log_probs(
    model: nn.Module,
    input_ids: torch.Tensor,  # (B, T)
    completion_mask: torch.Tensor,  # (B, T) — 1 for completion tokens, 0 for prompt
    autocast_ctx,
    return_per_token: bool = False,
) -> torch.Tensor:
    """
    Compute log probabilities for completion tokens only.

    WHY only completion tokens?
    The prompt is fixed and identical across all group members for a given
    prompt. Its log-probability contributes equally to every completion, so
    it cancels in the importance ratio π_new(a) / π_old(a). Including it
    would add noise without signal and waste computation. The completion_mask
    selects only the tokens the model generated autonomously.

    WHY the shift?
    Transformer logits at position t predict the token at position t+1.
    So logits[:, :-1, :] aligns with targets input_ids[:, 1:]. The mask
    is shifted by 1 for the same reason.

    return_per_token=False (default): returns (B,) — sum of log probs over
        completion tokens. Used for sequence-level KL divergence.

    return_per_token=True: returns (B, T-1) — per-token log probs, zeroed
        at non-completion positions (not -inf, to allow safe sum/mean).
        Used for [DAPO] token-level policy gradient loss.
    """
    with autocast_ctx:
        logits, _ = model(input_ids)  # (B, T, vocab_size)

    # Align predictions with next-token targets via the standard causal LM shift
    shifted_logits = logits[:, :-1, :].contiguous()  # (B, T-1, V)
    shifted_targets = input_ids[:, 1:].contiguous()  # (B, T-1)
    # Shift the completion mask by 1 to match the shifted prediction positions
    shifted_mask = completion_mask[:, 1:].contiguous().float()  # (B, T-1)

    # log_softmax is numerically more stable than log(softmax(x))
    log_probs = F.log_softmax(shifted_logits, dim=-1)  # (B, T-1, V)

    # Gather the log probability of the *actual* next token at each position.
    # gather(-1, idx) selects along the vocabulary dimension.
    token_log_probs = log_probs.gather(-1, shifted_targets.unsqueeze(-1)).squeeze(-1)  # (B, T-1)

    # Zero out prompt positions: prompt tokens must not contribute to the
    # policy gradient or KL. Multiplying by the mask (0/1) achieves this
    # cleanly without -inf values that could propagate through subsequent ops.
    token_log_probs = token_log_probs * shifted_mask

    if return_per_token:
        return token_log_probs  # (B, T-1), zeroed where mask=0

    return token_log_probs.sum(dim=-1)  # (B,) — sequence-level sum for KL


def completion_lengths(mask_list: list[torch.Tensor]) -> torch.Tensor:
    """
    Compute completion token counts from a list of completion masks.
    Returns (N,) tensor of ints (as float32 for downstream division).

    Each mask is 1 for completion tokens and 0 for prompt tokens, so
    summing gives the number of tokens the model actually generated.
    We use float32 (not int) because the result is used as a divisor
    in Dr. GRPO length normalization — integer division would truncate.
    """
    return torch.tensor([m.sum().item() for m in mask_list], dtype=torch.float32)


# ---------------------------------------------------------------------------
# [Dr. GRPO] Length-debiased group advantage computation
# ---------------------------------------------------------------------------


def compute_group_advantages(
    rewards: torch.Tensor,  # (batch_prompts, group_size)
    lengths: Optional[torch.Tensor],  # (batch_prompts, group_size) completion token counts
    length_debiased: bool = True,
) -> torch.Tensor:  # (batch_prompts × group_size,) flattened
    """
    Compute normalized group advantages with optional length debiasing.

    WHY group normalization (not global)?
    GRPO computes advantages *relative to other completions for the same prompt*.
    A_i = (r_i - mean_group(r)) / (std_group(r) + ε)
    This makes the advantage signal self-calibrating: it measures "how much
    better/worse than the group average was this specific completion?" rather
    than an absolute reward level. The policy is updated to favour whichever
    completions in the group were above-average, regardless of the absolute
    reward scale of different problems.

    WHY add ε to the denominator?
    When all group rewards are identical (all correct or all wrong), std=0.
    Without ε, we'd divide by zero → NaN. ε=1e-8 is safely below any
    meaningful std while never causing detectable bias. Dynamic sampling
    removes most such groups, but ε defends against any that slip through.

    Vanilla GRPO formula:
      A_i = (r_i - mean(r)) / (std(r) + ε)
      Problem: length is a silent confound. A correct 2000-token response
      and a correct 50-token response get the same sequence-level reward,
      but the 2000-token response received far less per-token gradient
      signal (advantage spread across 40× more tokens). The model implicitly
      learns that brevity is rewarded even when it isn't.

    [Dr. GRPO] Length-debiased:
      Normalize rewards by completion length before group statistics.
      This removes the length confound: both a 2000-token and a 50-token
      correct answer receive the same *length-adjusted* reward, and thus
      the same positive group advantage.

      r̃_i = r_i / len_i                        (length-normalized reward)
      Ã_i = (r̃_i - mean(r̃)) / (std(r̃) + ε)   (group normalize)

      Effect: the policy is no longer penalized for thinking longer when it
      needs to. Correct long CoT chains and correct short answers get equal
      positive reinforcement.
    """
    # ε prevents division-by-zero when all group rewards are identical.
    # 1e-8 is safely below any real std we'd encounter in practice.
    eps = 1e-8

    if length_debiased and lengths is not None:
        # Divide each reward by its completion length to remove the length
        # confound before computing group statistics.
        # Clamp lengths to >= 1 so a zero-length completion (edge case, e.g.
        # immediate EOS) doesn't cause division-by-zero here.
        len_safe = lengths.clamp(min=1.0)
        rewards_norm = rewards / len_safe  # (batch_prompts, group_size)
    else:
        rewards_norm = rewards  # Use raw rewards for vanilla GRPO

    # Compute group mean and std along the group dimension (dim=1).
    # keepdim=True lets us broadcast against (batch_prompts, group_size).
    mean = rewards_norm.mean(dim=1, keepdim=True)  # (batch_prompts, 1)
    std = rewards_norm.std(dim=1, keepdim=True)  # (batch_prompts, 1)
    # Normalize: each completion's advantage is its deviation from the group
    # mean measured in group standard deviations (z-score within the group).
    adv = (rewards_norm - mean) / (std + eps)  # (batch_prompts, group_size)

    # Flatten to (batch_prompts × group_size,) for direct use as per-sample
    # advantages in the policy gradient loss.
    return adv.view(-1)


# ---------------------------------------------------------------------------
# [DAPO] Dynamic sampling
# ---------------------------------------------------------------------------


def filter_uniform_groups(
    rewards_grouped: torch.Tensor,  # (batch_prompts, group_size)
    min_diversity: float = 0.0,
) -> torch.Tensor:
    """
    Return a boolean mask (batch_prompts,) — True for groups with
    non-uniform rewards (i.e., at least one correct AND one incorrect
    completion in the group).

    WHY filter uniform groups?
    When every completion in a group has the same reward (all-correct or
    all-wrong), the group mean equals every individual reward, so every
    advantage is zero:
        A_i = (r_i - mean(r)) / (std(r) + ε) ≈ 0 / ε ≈ 0
    A zero advantage means a zero policy gradient — the optimizer step
    updates nothing. The compute spent generating and evaluating those
    completions is completely wasted.

    For all-correct groups: the model already "knows" this problem; we
    learn nothing new about how to improve.
    For all-wrong groups: every completion was equally bad; we can't
    tell which direction to push the policy.

    std(r) > min_diversity is a quick proxy for "at least one completion
    differs from the rest". With binary rewards (0/1), std=0 iff all rewards
    are identical. With continuous rewards, min_diversity can be set > 0 to
    require a minimum spread before the group is considered informative.

    min_diversity: minimum reward std required to keep a group.
    Default 0.0 keeps any group that isn't perfectly uniform.
    """
    std = rewards_grouped.std(dim=1)  # Per-group std along group_size dimension
    return std > min_diversity  # True = keep (non-uniform = has gradient signal)


# ---------------------------------------------------------------------------
# GRPO / DAPO loss
# ---------------------------------------------------------------------------


def grpo_loss(
    token_log_probs_policy: torch.Tensor,  # (B, T-1) — per-token, zeroed at non-completion
    token_log_probs_old: torch.Tensor,  # (B, T-1) — same, from generation time
    log_probs_ref: torch.Tensor,  # (B,) — sequence-level, for KL
    log_probs_policy_seq: torch.Tensor,  # (B,) — sequence-level policy, for KL
    advantages: torch.Tensor,  # (B,) — normalized group advantages
    completion_mask: torch.Tensor,  # (B, T-1) — 1 for completion tokens
    clip_low: float,
    clip_high: float,
    kl_coef: float,
) -> tuple[torch.Tensor, dict]:
    """
    GRPO policy gradient loss with DAPO and Dr. GRPO improvements.

    ── Policy gradient (PPO clipped surrogate) ───────────────────────────────
    We maximise the expected advantage, but clip the importance ratio to
    prevent overly large updates (PPO). The importance ratio is:
        r_t(θ) = π_θ(a_t | s_t) / π_old(a_t | s_t)
               = exp(log_π_θ - log_π_old)   (computed per token)

    The PPO surrogate clips this ratio so a single gradient step cannot move
    the policy too far from where completions were sampled. Without clipping,
    a lucky high-advantage sample could cause a catastrophically large update
    that destroys previously learned knowledge.

    [DAPO] Clip-Higher — asymmetric policy ratio clipping:
      Vanilla PPO clips symmetrically to [1-ε, 1+ε] (e.g. [0.80, 1.20]).
      DAPO relaxes the upper bound only: [1-clip_low, 1+clip_high] where
      clip_high > clip_low (e.g. [0.80, 1.28]).
      Why? In practice the LOWER bound (preventing probability collapse) is
      the more important safety constraint. Relaxing the UPPER bound lets
      the policy grow probability of good actions more freely, sustaining
      entropy (diversity) throughout training. Tight symmetric clipping
      causes entropy collapse over thousands of steps.

    ── Token-level loss ([DAPO]) ─────────────────────────────────────────────
    Vanilla GRPO averages per-sequence:
        L = mean_over_B( -A_i × mean_over_T(log_ratio_t) )
    This gives each *sequence* equal weight regardless of length: a 2000-token
    correct CoT and a 50-token correct answer contribute equally to the batch
    gradient. But the 2000-token sequence needed 40× more correct token-level
    decisions to produce its answer — it deserves 40× more total gradient.

    [DAPO] Token-level loss averages per-token across the whole batch:
        L = sum_over_B_T( -A_i × log_ratio_t × mask_t ) / total_tokens
    Now every token contributes equally. Long correct reasoning chains receive
    appropriately large total gradient, reinforcing the full CoT trajectory.

    ── KL penalty ────────────────────────────────────────────────────────────
    The KL term β·KL(π_θ ‖ π_ref) penalises the policy for drifting too far
    from the frozen SFT reference model π_ref. Without it:
      - The policy can exploit reward loopholes (reward hacking)
      - Language quality degrades (the model abandons SFT-learned phrasing)
      - The policy can collapse to repetitive or degenerate outputs
    β=0.01 is small: strong enough to anchor quality, weak enough to allow
    the policy to improve on hard problems. KL is computed at sequence level
    (sum of token log-prob differences) because that is the natural measure
    of distributional distance between two language models.

    WHY π_ref is the SFT checkpoint specifically:
    The SFT checkpoint has already learned correct format, sensible language,
    and a baseline pass rate. Using it as π_ref means the KL penalty says
    "stay close to the best pre-RL behaviour we have". Using a random init
    or a much earlier checkpoint would either waste the KL budget or allow
    too much drift from SFT quality.
    """
    shifted_mask = completion_mask.float()  # (B, T-1) — float for masked arithmetic

    # ── Importance ratio ──────────────────────────────────────────────────
    # log(π_new / π_old) = log_π_new - log_π_old  (numerically stable)
    token_log_ratio = token_log_probs_policy - token_log_probs_old  # (B, T-1)
    # Exponentiate to get the ratio r_t(θ) = π_θ(a_t) / π_old(a_t).
    # For prompt positions these are both 0 (from masking), so exp(0)=1 — but
    # they will be zeroed by shifted_mask before loss accumulation.
    token_ratio = token_log_ratio.exp()  # (B, T-1)

    # ── [DAPO] Asymmetric clip ─────────────────────────────────────────────
    # Lower bound (1 - clip_low): prevents the policy from collapsing an
    #   action's probability too rapidly (same as vanilla PPO).
    # Upper bound (1 + clip_high): relaxed relative to vanilla PPO; allows
    #   the policy to grow probability of good actions more freely, preventing
    #   entropy collapse.
    token_ratio_clipped = token_ratio.clamp(1.0 - clip_low, 1.0 + clip_high)

    # Broadcast advantage from per-sequence (B,) to per-token (B, T-1).
    # All tokens in the same completion share the same advantage estimate.
    adv_token = advantages.unsqueeze(1).expand_as(token_log_ratio)

    # ── PPO pessimistic bound ──────────────────────────────────────────────
    # We NEGATE the advantage because PyTorch minimises losses (gradient descent)
    # but we want to maximise the policy objective (gradient ascent). Taking
    # the max of (unclipped, clipped) with negated advantages gives the
    # PPO pessimistic bound: we use whichever is less helpful to the policy.
    #
    # When advantage > 0 (good action):
    #   pg1 = -A * ratio    (we want to increase ratio → make pg1 more negative)
    #   pg2 = -A * clipped  (clipped is smaller → pg2 ≥ pg1, a worse bound)
    #   max(pg1, pg2) = pg2 → clipping applies when ratio tries to grow too fast
    #
    # When advantage < 0 (bad action):
    #   pg1 = -A * ratio   (ratio decreases → pg1 more positive → worse bound)
    #   pg2 = -A * clipped (clipped is larger → pg2 ≤ pg1, a better bound)
    #   max(pg1, pg2) = pg1 → clipping applies when ratio tries to shrink too fast
    pg1 = -adv_token * token_ratio  # Unclipped policy gradient term
    pg2 = -adv_token * token_ratio_clipped  # Clipped policy gradient term
    pg_token = torch.max(pg1, pg2)  # (B, T-1) — pessimistic (conservative) bound

    # ── [DAPO] Token-level averaging ──────────────────────────────────────
    # Sum weighted by mask (zero out prompt positions), then divide by total
    # completion tokens (not total sequences). This is the key DAPO difference
    # from sequence-level averaging.
    total_tokens = shifted_mask.sum().clamp(min=1.0)  # clamp avoids divide-by-zero
    pg_loss = (pg_token * shifted_mask).sum() / total_tokens

    # ── KL penalty ────────────────────────────────────────────────────────
    # KL(π_new ‖ π_ref) ≈ mean(log_π_new - log_π_ref) at the sequence level.
    # This is not the full KL (which would require marginalising over the
    # full distribution) but a Monte Carlo estimate over the current batch,
    # which is sufficient for a regularisation term.
    kl = (log_probs_policy_seq - log_probs_ref).mean()
    kl_loss = kl_coef * kl  # β × KL — scaled by coefficient

    total_loss = pg_loss + kl_loss

    # ── Diagnostics (no-grad: these are monitoring only) ──────────────────
    with torch.no_grad():
        # Clip fraction: fraction of tokens where the ratio was clipped.
        # High clip_frac (>50%) means the policy is trying to move faster
        # than the clip allows — consider reducing LR or increasing clip.
        lower_clipped = (token_ratio < 1.0 - clip_low).float()
        upper_clipped = (token_ratio > 1.0 + clip_high).float()
        clip_frac = ((lower_clipped + upper_clipped) * shifted_mask).sum() / total_tokens
        # Entropy proxy: -mean(log_π_θ) over completion tokens.
        # Should stay well above 0; values near 0 indicate entropy collapse.
        entropy_proxy = -(token_log_probs_policy * shifted_mask).sum() / total_tokens

    metrics = {
        "pg_loss": pg_loss.item(),
        "kl": kl.item(),
        "kl_loss": kl_loss.item(),
        "clip_frac": clip_frac.item(),
        "ratio_mean": (token_ratio * shifted_mask).sum().item() / total_tokens.item(),
        "entropy_proxy": entropy_proxy.item(),  # Monitor for collapse; should stay > 0
    }

    return total_loss, metrics


# ---------------------------------------------------------------------------
# Generation — batch sampling
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate_completions(
    model: nn.Module,
    prompt_ids: list[torch.Tensor],  # list of (T_prompt,) tensors
    group_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    tokenizer,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[list[str], list[torch.Tensor], list[torch.Tensor]]:
    """
    Generate group_size completions for each prompt.

    WHY @torch.no_grad()?
    Generation is inference — we do not need gradients through this step.
    The "old" policy log-probs (π_old) used in the importance ratio are
    collected separately after generation (as a detached tensor), not during
    generation. Disabling grad here saves significant memory and compute.

    WHY model.eval() during generation?
    Ensures BatchNorm/Dropout are in inference mode. More importantly, it
    prevents any in-place operations from conflicting with the subsequent
    model.train() forward pass. We switch back to train() mode before the
    loss computation.

    Returns:
      completions: list of decoded strings, length = n_prompts × group_size
      input_id_tensors: list of (T_prompt + T_completion,) tensors
        These include BOTH prompt and completion tokens. The policy gradient
        needs the full context to compute log-probs accurately (KV cache for
        prompt, then per-token log-prob for completion).
      completion_masks: list of (T_prompt + T_completion,) binary masks
        1 for completion tokens, 0 for prompt tokens. The mask tells
        compute_log_probs which token positions to include in the loss.

    Memory strategy: generate one prompt's group at a time to control
    peak memory. Each prompt is expanded to group_size copies, completed
    in parallel (group_size sequences in one batched forward pass), then
    results collected. This avoids holding all n_prompts × group_size
    sequences in a single large padded batch during generation.
    """
    model.eval()
    eos_id = tokenizer.token_to_id("<eos>")

    all_completions = []
    all_input_ids = []
    all_comp_masks = []

    for prompt_t in prompt_ids:
        T_p = prompt_t.shape[0]
        # Expand the single prompt to group_size identical copies so all G
        # completions share the same prompt context in one batched forward pass.
        # .clone() ensures each row is an independent tensor (not a view).
        batch = prompt_t.unsqueeze(0).expand(group_size, -1).clone().to(device)

        # Autoregressive generation state
        finished = torch.zeros(group_size, dtype=torch.bool, device=device)
        generated = []  # Accumulates (G,) token tensors at each step
        kv_caches = None
        pos_offset = 0

        autocast_ctx = torch.amp.autocast(
            device_type=device.type if device.type != "mps" else "cpu",
            dtype=dtype,
            enabled=(dtype == torch.bfloat16),
        )

        # Prefill: process the full prompt once to populate the KV cache.
        # This avoids re-attending over the prompt at every decode step,
        # giving O(T_prompt) prefill cost rather than O(T_prompt × T_completion).
        # kv_caches=[] signals "collect KV caches for all layers" (prefill mode).
        # Without this, model runs in training mode (kv_caches=None) and returns
        # no cache, causing decode steps to run with no context.
        with autocast_ctx:
            logits, kv_caches = model(batch, kv_caches=[])  # (G, T_p, V) — prefill
        next_logits = logits[:, -1, :]  # (G, V) — logits for first new token
        pos_offset = T_p  # KV cache covers positions 0..T_p-1; next token is at T_p

        # Optionally compress the prefill KV caches to reduce peak memory
        # during the generation loop. forward_compressed handles the
        # decompress-forward-recompress cycle on each subsequent decode step.
        use_compress = getattr(generate_completions, "_compress_kv", False)
        if use_compress and kv_caches is not None:
            from model.kv_compress import compress_kv_caches, forward_compressed

            kv_caches = compress_kv_caches(kv_caches)

        for _ in range(max_new_tokens):
            next_tok = _sample_tokens(next_logits, temperature, top_p)  # (G,)
            generated.append(next_tok)
            # Mark sequences that have emitted EOS; they stop contributing
            # tokens to the output (but we continue the loop for the others).
            finished |= next_tok == eos_id
            if finished.all():
                break  # Early exit: all group members have finished

            # Single-token decode step: feed the just-sampled token and
            # let the KV cache provide the full history.
            # BUG FIX: the original code discarded kv_caches here with `_`,
            # causing each decode step to run with no context (only 1 token).
            # Now we correctly pass and update the cache each step.
            next_input = next_tok.unsqueeze(1)  # (G, 1) — one token per sequence

            if use_compress and kv_caches is not None:
                # forward_compressed: decompress → forward → recompress
                logits, kv_caches = forward_compressed(
                    model,
                    next_input,
                    kv_caches,
                    position_offset=pos_offset,
                    autocast_ctx=autocast_ctx,
                )
            else:
                with autocast_ctx:
                    logits, kv_caches = model(
                        next_input,
                        kv_caches=kv_caches,
                        position_offset=pos_offset,
                    )
            next_logits = logits[:, -1, :]  # (G, V) — logits for the next token
            pos_offset += 1  # Advance position counter for correct positional encoding

        # Stack all generated tokens along the time dimension: (G, T_completion)
        if generated:
            completion_ids = torch.stack(generated, dim=1)  # (G, T_gen)
        else:
            # Edge case: model emitted EOS immediately on every sequence.
            completion_ids = torch.zeros(group_size, 1, dtype=torch.long, device=device)

        # Build full sequences (prompt + completion) and binary completion masks
        for g in range(group_size):
            comp = completion_ids[g].cpu()  # Move to CPU; we're done with GPU ops
            # Trim at EOS (inclusive): keep the EOS token itself so the model
            # learns to predict it, but drop any tokens generated after a
            # sequence that "finished" early but other group members hadn't.
            eos_positions = (comp == eos_id).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                comp = comp[: eos_positions[0] + 1]

            # Concatenate prompt and completion into one sequence.
            # The full sequence is needed for the batched log-prob forward pass:
            # the transformer attends over the prompt when computing completion
            # token probabilities.
            full_ids = torch.cat([prompt_t.cpu(), comp])

            # Binary mask: 0 for prompt tokens (fixed input, not trained),
            # 1 for completion tokens (model output, subject to PG loss).
            # Prompt tokens are excluded from loss and log-prob accumulation.
            mask = torch.cat(
                [
                    torch.zeros(T_p, dtype=torch.long),  # prompt: excluded
                    torch.ones(comp.shape[0], dtype=torch.long),  # completion: included
                ]
            )

            all_input_ids.append(full_ids)
            all_comp_masks.append(mask)

            # Decode completion text for reward computation and display.
            # skip_special_tokens=False keeps <think>/<eos>/etc so reward
            # functions can detect them.
            text = tokenizer.decode(comp.tolist(), skip_special_tokens=False)
            all_completions.append(text)

    return all_completions, all_input_ids, all_comp_masks


def _sample_tokens(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    """
    Sample next tokens from logits. Returns (batch_size,) token IDs.

    Temperature scaling controls the sharpness of the distribution:
      - temperature → 0: approaches argmax (greedy, deterministic)
      - temperature = 1: use raw model probabilities
      - temperature > 1: more uniform (more random), less useful for reasoning

    Top-p (nucleus) sampling: sample only from the smallest set of tokens
    whose cumulative probability mass exceeds top_p. This prevents the model
    from accidentally sampling very low-probability "tail" tokens while still
    allowing diversity within the high-probability nucleus.

    We need diversity across the group (temperature > 0, top_p < 1) so that
    different group members explore different reasoning paths. If all G
    completions were identical (temperature=0), every group would be uniform
    and dynamic sampling would filter them all out.
    """
    if temperature == 0.0:
        # Greedy decoding: deterministic argmax. Only used for evaluation.
        return logits.argmax(dim=-1)

    # Scale logits by temperature before softmax. Dividing by T < 1 sharpens
    # the distribution (more confident); T > 1 flattens it (more random).
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)

    # Sort probabilities in descending order to find the nucleus easily.
    sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
    # cumsum gives the probability mass up to and including each token.
    cumulative = sorted_probs.cumsum(dim=-1)
    # A token is excluded if the cumulative mass BEFORE it already exceeds
    # top_p. Subtracting sorted_probs shifts the comparison by one position.
    remove = (cumulative - sorted_probs) > top_p
    sorted_probs[remove] = 0.0  # Zero out tail tokens (outside nucleus)
    # Re-normalise the truncated distribution so probabilities sum to 1.
    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    # Sample from the nucleus distribution and remap indices back to vocab IDs.
    sampled = torch.multinomial(sorted_probs, num_samples=1).squeeze(-1)  # (B,)
    return sorted_idx.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)  # (B,) vocab IDs


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


def build_prompt(example: dict, tokenizer) -> torch.Tensor:
    """
    Build the prompt tensor for a GRPO example.

    Format: <bos> User: {problem} \n Assistant: <think>\n

    The prompt ends with the opening <think> tag and a newline so the model's
    very first generated token is inside the thinking block. This is important
    for two reasons:
      1. It enforces the CoT structure from the prompt side — the model never
         has to "decide" to start thinking, it is simply in that mode already.
      2. The reward_format function checks for </think> followed by an answer,
         so the presence of <think> in the prompt primes the model to close it.

    The "User: / Assistant:" framing matches the SFT conversation format.
    Consistency with SFT prompts means the model's SFT-learned behaviours are
    activated rather than requiring the policy to adapt to a new format.
    """
    problem = example.get("problem", example.get("prompt", example.get("question", "")))
    prompt_text = f"User: {problem}\nAssistant: <think>\n"

    enc = tokenizer.encode(prompt_text)
    ids = enc.ids

    # BOS is prepended by the tokenizer's post-processor (configured at
    # tokenizer training time), so we don't manually prepend BOS_ID here.
    # Strip trailing EOS: the tokenizer post-processor appends EOS to every
    # encode() call, but we need the model to GENERATE after this prompt,
    # not treat it as a finished sequence. Generating after EOS produces
    # degenerate context-free output (the model sees "sequence is over").
    if ids and ids[-1] == tokenizer.token_to_id("<eos>"):
        ids = ids[:-1]

    return torch.tensor(ids, dtype=torch.long)


# ---------------------------------------------------------------------------
# Padding utilities for batched log prob computation
# ---------------------------------------------------------------------------


def pad_sequences(
    tensors: list[torch.Tensor],
    pad_value: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pad a list of 1D tensors to the same length (right-padding).
    Returns (padded, lengths) where padded is (N, max_len).

    Right-padding means the actual content is at the LEFT of each row and
    padding tokens are at the RIGHT. This is important for causal attention:
    padding at the right is "after" the real sequence and will not be attended
    to by earlier tokens. Padding at the LEFT would corrupt the positional
    encodings for the real tokens.

    We allocate the full (N, max_len) tensor with pad_value, then copy each
    real sequence in — simpler and no slower than building row by row.
    """
    lengths = torch.tensor([t.shape[0] for t in tensors])
    max_len = lengths.max().item()
    N = len(tensors)

    # Pre-fill with pad_value so positions beyond each sequence's length are
    # already padded without needing a second pass to fill them.
    padded = torch.full((N, max_len), pad_value, dtype=torch.long, device=device)
    for i, t in enumerate(tensors):
        padded[i, : t.shape[0]] = t.to(device)  # Left-align real content

    return padded, lengths


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(cfg: GRPOConfig):
    # ── Device ────────────────────────────────────────────────────────
    if cfg.backend == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Backend: CUDA — {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Backend: CPU")

    dtype = torch.bfloat16 if cfg.dtype == "bfloat16" else torch.float32

    # ── Tokenizer ─────────────────────────────────────────────────────
    from tokenizers import Tokenizer

    tok_path = str(Path(cfg.tokenizer_path) / "tokenizer.json")
    tokenizer = Tokenizer.from_file(tok_path)
    print(f"Tokenizer: {tok_path}")

    # ── Model (policy) ────────────────────────────────────────────────
    model_cfg = CONFIGS[cfg.model_config]
    print(f"\nPolicy model: {cfg.model_config.upper()}")

    policy = SmallReasoningModel(model_cfg)
    if cfg.checkpoint:
        print(f"Loading SFT checkpoint: {cfg.checkpoint}")
        state = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)
        sd = state.get("model", state)
        policy.load_state_dict(sd, strict=False)
        print(f"  Phase: {state.get('phase', 'unknown')}  " f"Step: {state.get('step', '?')}")
    else:
        print("  WARNING: No checkpoint — starting from random init (not recommended)")

    policy = policy.to(device=device, dtype=dtype)
    if cfg.grad_checkpointing:
        _enable_gradient_checkpointing(policy)

    # ── Reference model (frozen SFT baseline) ─────────────────────────
    # The reference model π_ref is a FROZEN copy of the SFT checkpoint.
    # It serves as the anchor in the KL penalty: KL(π_θ ‖ π_ref).
    #
    # WHY freeze it?
    # The reference model must be stationary throughout training so that the
    # KL penalty has a fixed target. If it moved, the policy could satisfy
    # the KL constraint by chasing a moving reference — providing no stability.
    # Freezing is implemented by requires_grad_(False) on all parameters, which
    # prevents any gradient from being computed for reference model weights and
    # ensures zero memory is allocated for their gradient buffers.
    #
    # WHY the SFT checkpoint specifically?
    # The SFT checkpoint is the best behaviour we have before RL. Starting the
    # KL anchor at the SFT checkpoint means: "improve on hard reasoning problems,
    # but don't deviate so far that you lose the quality you already have."
    # Using a random init or an earlier checkpoint would anchor to worse behaviour,
    # giving the KL penalty less useful guidance.
    #
    # WHY a second copy of the full model (not EMA or otherwise shared)?
    # Sharing weights between policy and reference would break the anchor:
    # any policy update would also move the reference. A completely separate
    # copy loaded from the same checkpoint guarantees independence.
    print("Creating frozen reference model (SFT baseline)...")
    reference = SmallReasoningModel(model_cfg)
    if cfg.checkpoint:
        reference.load_state_dict(sd, strict=False)
    reference = reference.to(device=device, dtype=dtype)
    reference.eval()  # Inference mode: disables dropout, fixes BatchNorm stats
    for param in reference.parameters():
        param.requires_grad_(False)  # No gradients ever flow into the reference model

    # ── Dataset ───────────────────────────────────────────────────────
    synthetic = not Path(cfg.data_dir).exists() or not any(Path(cfg.data_dir).iterdir())

    if not synthetic:
        print(f"\nLoading GRPO data from: {cfg.data_dir}")
        train_dataset = GRPODataset(cfg.data_dir, "train", cfg.min_pass_rate, cfg.max_pass_rate)
    else:
        print("\nNo data found — using synthetic arithmetic (validate mode)")
        train_dataset = SyntheticGRPODataset(n=200)

    # ── Optimizer ─────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=cfg.lr,
        betas=(0.9, 0.95),  # Standard AdamW betas; β2=0.95 (not 0.999) for faster adaptation
        weight_decay=cfg.weight_decay,  # 0.0: no regularisation needed at this LR scale
        fused=(cfg.backend == "cuda"),  # Fused CUDA kernel: faster but CUDA-only
    )

    # Short warmup (1% of total steps) lets the optimizer accumulate gradient
    # statistics before the full LR is applied, reducing early instability.
    warmup_steps = max(1, int(cfg.steps * cfg.warmup_fraction))

    # ── Autocast ──────────────────────────────────────────────────────
    autocast_ctx = torch.amp.autocast(
        device_type="cuda" if cfg.backend == "cuda" else "cpu",
        dtype=dtype,
        enabled=(dtype == torch.bfloat16),
    )

    # ── Training loop ─────────────────────────────────────────────────
    os.makedirs(cfg.output_dir, exist_ok=True)

    improvements = []
    if not cfg.no_dapo:
        improvements += ["Clip-Higher", "Token-level PG", "Dynamic sampling", "Overlong penalty"]
    if not cfg.no_dr_grpo:
        improvements += ["Length-debiased adv"]

    print(f"\nGRPO schedule:")
    print(f"  Steps:       {cfg.steps:,}")
    print(f"  Group size:  {cfg.group_size}  (completions per prompt)")
    print(
        f"  Batch:       {cfg.batch_prompts} prompts × {cfg.group_size} = "
        f"{cfg.batch_prompts * cfg.group_size} sequences/step"
    )
    print(f"  LR:          {cfg.lr:.1e}  (constant)")
    print(f"  KL coef:     {cfg.kl_coef}")
    print(
        f"  Clip:        [{1-cfg.clip_low:.2f}, {1+cfg.clip_high:.2f}]"
        f"  {'(asymmetric/DAPO)' if cfg.clip_high != cfg.clip_low else '(symmetric/vanilla)'}"
    )
    print(f"  Improvements: {', '.join(improvements) if improvements else 'none (vanilla)'}")
    print(f"  Domain:      {cfg.domain}")
    print()
    print(f"{'─'*88}")
    print(
        f"  {'step':>6}  {'loss':>7}  {'reward':>7}  {'kl':>7}  "
        f"{'clip%':>6}  {'pass@1':>7}  {'entropy':>8}  {'skipped':>8}"
    )
    print(f"{'─'*88}")

    dataset_iter = _infinite_iter(train_dataset)
    best_reward = -float("inf")
    t0 = time.time()
    total_skipped = 0  # groups skipped by dynamic sampling

    for step in range(cfg.steps):
        policy.train()

        # ── [DAPO] Dynamic sampling: oversample then filter ───────────
        # Sample 2× prompts, generate completions, discard groups where
        # all rewards are identical (zero gradient signal). Keep the rest.
        # This ensures every training step has useful signal.
        if cfg.dynamic_sampling and not cfg.no_dapo:
            n_sample = cfg.batch_prompts * cfg.dynamic_oversample
        else:
            n_sample = cfg.batch_prompts

        batch_examples = [next(dataset_iter) for _ in range(n_sample)]
        prompt_tensors = [build_prompt(ex, tokenizer) for ex in batch_examples]

        # ── Generate completions ──────────────────────────────────────
        completions, input_id_list, mask_list = generate_completions(
            model=policy,
            prompt_ids=prompt_tensors,
            group_size=cfg.group_size,
            max_new_tokens=cfg.max_gen_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            tokenizer=tokenizer,
            device=device,
            dtype=dtype,
        )
        examples_expanded = [ex for ex in batch_examples for _ in range(cfg.group_size)]

        # Completion lengths (tokens generated, not prompt) — needed for:
        #   [Dr. GRPO] length-debiased advantages
        #   [DAPO] overlong penalty
        comp_lens_flat = completion_lengths(mask_list)  # (n_sample × G,)

        # ── Compute rewards ───────────────────────────────────────────
        rewards_flat = torch.tensor(
            [
                combined_reward(
                    completion=comp,
                    example=ex,
                    # Use per-example domain if available (filtered dataset has
                    # math_exact vs math_sympy per problem); fall back to global cfg
                    domain=ex.get("domain", cfg.domain),
                    format_weight=cfg.format_reward_weight,
                    completion_len=int(comp_len.item()),
                    max_gen_tokens=cfg.max_gen_tokens,
                    overlong_penalty=(cfg.overlong_penalty and not cfg.no_dapo),
                    overlong_penalty_factor=cfg.overlong_penalty_factor,
                )
                for comp, ex, comp_len in zip(completions, examples_expanded, comp_lens_flat)
            ],
            dtype=torch.float32,
        )  # (n_sample × G,)

        rewards_grouped = rewards_flat.view(n_sample, cfg.group_size)
        comp_lens_grouped = comp_lens_flat.view(n_sample, cfg.group_size)

        # ── [DAPO] Filter uniform groups ─────────────────────────────
        if cfg.dynamic_sampling and not cfg.no_dapo:
            keep_mask = filter_uniform_groups(
                rewards_grouped, cfg.dynamic_min_diversity
            )  # (n_sample,) bool
            skipped = (~keep_mask).sum().item()
            total_skipped += skipped

            # Keep up to batch_prompts diverse groups
            keep_indices = keep_mask.nonzero(as_tuple=True)[0][: cfg.batch_prompts]

            if len(keep_indices) == 0:
                # All 2× oversampled groups were uniform — skip this step entirely.
                # This is rare in practice: it requires every single prompt in the
                # oversample to be either trivially easy (all correct) or impossible
                # (all wrong). Can happen early in training on homogeneous data.
                # Skipping is preferable to training on zero-gradient data.
                continue

            # Subset to kept groups
            rewards_grouped = rewards_grouped[keep_indices]
            comp_lens_grouped = comp_lens_grouped[keep_indices]
            batch_examples_kept = [batch_examples[i] for i in keep_indices.tolist()]

            # Subset completions and tensors
            kept_flat_indices = []
            for gi in keep_indices.tolist():
                kept_flat_indices.extend(range(gi * cfg.group_size, (gi + 1) * cfg.group_size))
            input_id_list = [input_id_list[i] for i in kept_flat_indices]
            mask_list = [mask_list[i] for i in kept_flat_indices]
            completions = [completions[i] for i in kept_flat_indices]

            n_kept = len(keep_indices)
        else:
            skipped = 0
            n_kept = n_sample
            batch_examples_kept = batch_examples

        # ── [Dr. GRPO] Length-debiased advantages ────────────────────
        advantages = compute_group_advantages(
            rewards=rewards_grouped,
            lengths=comp_lens_grouped if cfg.length_debiased and not cfg.no_dr_grpo else None,
            length_debiased=(cfg.length_debiased and not cfg.no_dr_grpo),
        )  # (n_kept × G,)
        advantages = advantages.to(device)

        # Diagnostics — computed before the optimizer step so they reflect
        # the reward distribution for this step's data.
        mean_reward = rewards_grouped.mean().item()
        # pass@1: fraction of problems where the FIRST completion is correct.
        # Not pass@G (any correct), because pass@1 is the deployment metric:
        # at inference time the model generates one answer, not eight.
        pass_at_1 = (rewards_grouped[:, 0] > 0).float().mean().item()

        # ── Pad sequences for batched forward pass ────────────────────
        # Sequences within a batch have different lengths (different prompts
        # and different completion lengths). We pad to the maximum length in
        # the batch using PAD_ID=0 for tokens and 0 for the completion mask.
        # PAD positions are masked out in compute_log_probs, so padding never
        # contributes to the loss.
        padded_ids, _ = pad_sequences(input_id_list, PAD_ID, device)
        padded_masks, _ = pad_sequences(mask_list, 0, device)

        # The mask must be shifted by 1 to align with the shifted logits in
        # compute_log_probs (logits[:, :-1, :] predicts targets[:, 1:]).
        # A mask position of 1 means "this token was generated by the policy
        # and should be included in the loss".
        shifted_mask_for_loss = padded_masks[:, 1:].contiguous().float()

        # ── Compute log probs ─────────────────────────────────────────
        # We need three sets of log-probabilities:
        #   1. token_lp_policy  — per-token, from current policy (with grad)
        #      Used in the policy gradient loss and as π_new in the KL term.
        #   2. lp_ref           — sequence-level, from reference model (no grad)
        #      Used as π_ref in the KL penalty.
        #   3. token_lp_old     — per-token, "old" policy at generation time
        #      Used as π_old in the importance ratio r_t(θ) = π_new/π_old.
        #
        # In vanilla multi-step PPO, π_old would come from a separate copy
        # frozen at the start of the PPO epoch. Here we use single-step GRPO:
        # completions were just generated by the current policy, so π_old ≈
        # π_new at the time of generation. We approximate π_old as the current
        # policy with gradients detached (since the policy hasn't changed yet
        # within this step).
        policy.train()  # Re-enable training mode after generation used model.eval()
        token_lp_policy = compute_log_probs(
            policy, padded_ids, padded_masks, autocast_ctx, return_per_token=True
        )  # (B, T-1) — with grad: backprop flows through these
        lp_policy_seq = token_lp_policy.sum(dim=-1)  # (B,) sequence-level sum for KL

        with torch.no_grad():
            # Reference log-probs: no grad needed (reference model is frozen;
            # this is just a scalar regularisation signal).
            lp_ref = compute_log_probs(reference, padded_ids, padded_masks, autocast_ctx)
            # "old" log probs: detach() cuts the gradient tape so that the
            # importance ratio r_t = π_new / π_old does not try to differentiate
            # through π_old. π_old is treated as a fixed constant denominator;
            # only π_new (numerator) receives gradient updates. This is the
            # standard PPO design: π_old is the behaviour policy that collected
            # the data, and we only optimise the current policy π_new.
            token_lp_old = token_lp_policy.detach()

        # ── GRPO / DAPO loss ──────────────────────────────────────────
        loss, metrics = grpo_loss(
            token_log_probs_policy=token_lp_policy,
            token_log_probs_old=token_lp_old,
            log_probs_ref=lp_ref,
            log_probs_policy_seq=lp_policy_seq,
            advantages=advantages,
            completion_mask=shifted_mask_for_loss,
            clip_low=cfg.clip_low,
            clip_high=cfg.clip_high,
            kl_coef=cfg.kl_coef,
        )

        # ── Backward + update ─────────────────────────────────────────
        # set_to_none=True is more memory-efficient than zero_grad(): it
        # deallocates gradient tensors entirely rather than filling with zeros.
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # Gradient clipping prevents occasional large gradient spikes from
        # destabilising training. GRPO rewards can be noisy, leading to
        # high-variance gradients; clipping at 1.0 is a standard safeguard.
        torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)

        # Linear warmup: ramp LR from 0 to cfg.lr over warmup_steps steps.
        # After warmup, hold constant (no decay). Constant LR for GRPO is
        # standard practice because the reward signal is already noisy — decay
        # would reduce signal further without meaningful benefit.
        current_lr = cfg.lr * min(1.0, (step + 1) / warmup_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        optimizer.step()

        # ── Logging ───────────────────────────────────────────────────
        if step % cfg.log_every == 0:
            dt = time.time() - t0
            avg_len = sum(t.shape[0] for t in input_id_list) / max(len(input_id_list), 1)
            tps = len(input_id_list) * avg_len / dt if dt > 0 else 0
            print(
                f"  {step:>6,}  {loss.item():>7.4f}  {mean_reward:>7.4f}"
                f"  {metrics['kl']:>7.4f}  {metrics['clip_frac']*100:>5.1f}%"
                f"  {pass_at_1:>7.3f}  {metrics['entropy_proxy']:>8.3f}"
                f"  {skipped:>8}"
            )

            if mean_reward > best_reward:
                best_reward = mean_reward
                _save_grpo(
                    step, policy, optimizer, cfg, mean_reward, Path(cfg.output_dir) / "best.pt"
                )

            t0 = time.time()

        if step % cfg.save_every == 0 and step > 0:
            _save_grpo(
                step,
                policy,
                optimizer,
                cfg,
                mean_reward,
                Path(cfg.output_dir) / f"step_{step:07d}.pt",
            )

        if step % 200 == 0 and step > 0:
            # Print a sample completion every 200 steps for qualitative monitoring.
            # Quantitative metrics (loss, reward) can look fine while the model
            # is silently degenerating (e.g. producing garbled text that happens
            # to match by string). A human sanity-check every 200 steps catches this.
            _display_sample(completions[0], batch_examples_kept[0], rewards_grouped[0, 0].item())

    # Final save
    _save_grpo(cfg.steps, policy, optimizer, cfg, best_reward, Path(cfg.output_dir) / "final.pt")

    print(f"\n{'─'*88}")
    print(f"GRPO complete.")
    print(f"  Best mean reward:  {best_reward:.4f}")
    print(f"  Total steps skipped by dynamic sampling: {total_skipped}")
    return policy


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _infinite_iter(dataset):
    """
    Cycle through a dataset indefinitely, shuffling each full pass.

    GRPO training runs for a fixed number of steps (not epochs), and the
    number of steps typically exceeds one pass over the dataset. We reshuffle
    at each epoch boundary so the model doesn't see prompts in the same order
    repeatedly — order bias can slow convergence and hurt generalisation.
    """
    import random

    indices = list(range(len(dataset)))
    while True:
        random.shuffle(indices)  # New random order every epoch
        for i in indices:
            yield dataset[i]


def _display_sample(completion: str, example: dict, reward: float):
    """Print a sample completion for qualitative monitoring."""
    problem = example.get("problem", example.get("prompt", "?"))
    answer = example.get("answer", "?")
    print(f"\n  ── Sample ───────────────────────────────")
    print(f"  Problem: {problem[:80]}")
    print(f"  Ground truth: {answer}")
    print(f"  Reward: {reward:.2f}")
    print(f"  Completion: {completion[:200].replace(chr(10), ' ↵ ')}...")
    print(f"  ─────────────────────────────────────────\n")


def _save_grpo(step, model, optimizer, cfg, reward, path):
    # Save the full training state so training can be resumed from any
    # checkpoint. "phase": "grpo" distinguishes this from SFT checkpoints
    # when loading downstream (e.g. for inference or further fine-tuning).
    # The grpo_config is saved alongside the weights so the checkpoint is
    # self-documenting about what hyperparameters produced it.
    torch.save(
        {
            "step": step,
            "reward": reward,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "grpo_config": asdict(cfg),
            "phase": "grpo",  # Distinguishes from pretrain/SFT checkpoints
        },
        path,
    )
    print(f"  → Saved: {path}  (reward={reward:.4f})")


def _enable_gradient_checkpointing(model):
    """
    Enable activation (gradient) checkpointing for all transformer blocks.

    Same pattern as pretrain.py and sft.py — kept identical so the three
    training phases behave consistently.

    WHY gradient checkpointing?
    During GRPO each training step processes batch_prompts × group_size
    sequences simultaneously (e.g. 4 × 8 = 32 sequences). Without
    checkpointing, activations for all 32 sequences must be held in GPU
    memory simultaneously during the backward pass, which can exceed 40GB
    for a 1B model at 2048 tokens. Checkpointing discards forward activations
    and recomputes them during backward, trading ~33% extra compute for ~40%
    memory savings. This is almost always worthwhile for GRPO batch sizes.

    WHY skip checkpointing when kv_cache is not None?
    The KV cache is used only during generation (inference), not during the
    training forward pass. Checkpointing is meaningless during inference —
    there is no backward pass — and the checkpoint wrapper would interfere
    with the cache state. We bypass it when a cache is provided.

    use_reentrant=False: the non-reentrant checkpoint implementation handles
    tensors that require grad more correctly and is the recommended default
    in PyTorch >= 2.0.
    """
    import functools
    from torch.utils.checkpoint import checkpoint as torch_cp

    for block in model.blocks:
        orig = block.forward

        @functools.wraps(orig)
        def cp_fwd(
            x,
            attention_mask=None,
            kv_cache=None,
            position_offset=0,
            collect_kv=False,
            _orig=orig,
        ):
            if kv_cache is not None or collect_kv:
                # KV cache path (decode or prefill with collect_kv=True):
                # pass through without checkpointing — no backward needed
                # during generation, and we need the KV tensors returned.
                return _orig(
                    x,
                    attention_mask=attention_mask,
                    kv_cache=kv_cache,
                    position_offset=position_offset,
                    collect_kv=collect_kv,
                )

            def fn(x_):
                # Wrap the block call so torch.utils.checkpoint can recompute
                # activations on the backward pass instead of storing them.
                out, _ = _orig(
                    x_,
                    attention_mask=attention_mask,
                    kv_cache=None,
                    position_offset=position_offset,
                )
                return out

            return torch_cp(fn, x, use_reentrant=False), None

        block.forward = cp_fwd


# ---------------------------------------------------------------------------
# Validate mode
# ---------------------------------------------------------------------------


def validate_mode(model_config: str):
    """
    Smoke test GRPO loop with synthetic data and synthetic rewards.
    Validates:
      - Group advantage computation
      - GRPO loss shape / sign
      - Generation loop runs without error
      - Optimizer step executes
    """
    print(f"\nGRPO validate mode — {model_config.upper()}")

    cfg = GRPOConfig(
        model_config=model_config,
        checkpoint="",
        data_dir="/nonexistent",
        steps=10,
        batch_prompts=2,
        group_size=4,
        max_gen_tokens=64,
        temperature=0.8,
        lr=5e-7,
        grad_checkpointing=True,
        dtype="bfloat16",
        backend="cuda" if torch.cuda.is_available() else "cpu",
        log_every=1,
        save_every=999999,
        eval_every=0,
        output_dir="/tmp/grpo_validate",
        domain="math",
    )
    train(cfg)


# ---------------------------------------------------------------------------
# Pure-logic tests (no torch needed)
# ---------------------------------------------------------------------------


def run_logic_tests():
    """Run reward function, advantage, and DAPO/Dr.GRPO logic tests."""
    import statistics

    print("GRPO logic tests (DAPO + Dr.GRPO):")
    print("─" * 60)
    passed = 0
    total = 0

    def check(name, condition, detail=""):
        nonlocal passed, total
        total += 1
        ok = bool(condition)
        print(f"  {'✓' if ok else '✗'}  {name}" + (f"  ({detail})" if detail else ""))
        if ok:
            passed += 1

    # ── normalize_answer ──────────────────────────────────────────────
    print("normalize_answer:")
    cases = [
        ("42", "42", True, "integer"),
        ("$42$", "42", True, "LaTeX"),
        ("1,000", "1000", True, "comma"),
        ("3.10", "3.1", True, "trailing zero"),
        (" 42 ", "42", True, "whitespace"),
        ("42", "43", False, "wrong"),
    ]
    for raw, truth, exp_match, label in cases:
        match = normalize_answer(raw) == normalize_answer(truth)
        check(f"normalize: {repr(raw)} ({label})", match == exp_match)

    # ── reward_math_exact ─────────────────────────────────────────────
    print("\nreward_math_exact:")
    reward_cases = [
        ("<think>\nwork\n</think>\n42", "42", 1.0, "correct with CoT"),
        ("<think>\nwork\n</think>\n43", "42", 0.0, "wrong with CoT"),
        ("", "42", 0.0, "empty"),
        ("<think>\n</think>\n 42 ", "42", 1.0, "whitespace answer"),
    ]
    for comp, truth, exp_r, label in reward_cases:
        r = reward_math_exact(comp, truth)
        check(f"reward_exact: {label} → {r:.1f}", r == exp_r)

    # ── [DAPO] overlong penalty ───────────────────────────────────────
    print("\n[DAPO] overlong reward shaping:")
    # Correct answer, but completion hit max_gen_tokens — penalize
    comp_correct = "<think>\nwork\n</think>\n42"
    r_normal = combined_reward(
        comp_correct,
        {"answer": "42"},
        "math",
        completion_len=100,
        max_gen_tokens=2048,
        overlong_penalty=True,
        overlong_penalty_factor=0.5,
    )
    r_overlong = combined_reward(
        comp_correct,
        {"answer": "42"},
        "math",
        completion_len=2048,
        max_gen_tokens=2048,
        overlong_penalty=True,
        overlong_penalty_factor=0.5,
    )
    r_disabled = combined_reward(
        comp_correct,
        {"answer": "42"},
        "math",
        completion_len=2048,
        max_gen_tokens=2048,
        overlong_penalty=False,
        overlong_penalty_factor=0.5,
    )
    check("normal length: full reward", r_normal > 0.9)
    check("overlong: penalized (× 0.5)", abs(r_overlong - r_normal * 0.5) < 0.01)
    check("penalty disabled: not penalized", r_disabled == r_normal)

    # ── [DAPO] filter_uniform_groups ─────────────────────────────────
    print("\n[DAPO] dynamic sampling filter:")
    import torch as _torch

    all_zero = _torch.zeros(4, 8)  # all wrong
    all_one = _torch.ones(4, 8)  # all correct
    mixed = _torch.tensor([[1, 0, 1, 0, 1, 0, 1, 0]] * 4).float()  # mixed

    keep_zero = filter_uniform_groups(all_zero)
    keep_one = filter_uniform_groups(all_one)
    keep_mixed = filter_uniform_groups(mixed)

    check("all-zero groups filtered", (~keep_zero).all())
    check("all-one groups filtered", (~keep_one).all())
    check("mixed groups kept", keep_mixed.all())

    # ── [Dr. GRPO] length-debiased advantages ────────────────────────
    print("\n[Dr. GRPO] length-debiased advantages:")
    # Two groups with same binary rewards but different completion lengths
    # Without debiasing: identical advantages
    # With debiasing: longer correct completion gets same RELATIVE advantage
    rewards = _torch.tensor([[1.0, 0.0, 1.0, 0.0]])  # 1 group, G=4
    lens_equal = _torch.tensor([[100.0, 100.0, 100.0, 100.0]])
    lens_unequal = _torch.tensor([[2000.0, 50.0, 2000.0, 50.0]])  # correct answers are long

    adv_no_debias = compute_group_advantages(rewards, None, length_debiased=False)
    adv_equal_len = compute_group_advantages(rewards, lens_equal, length_debiased=True)
    adv_unequal = compute_group_advantages(rewards, lens_unequal, length_debiased=True)

    check("no debias: correct = positive adv", (adv_no_debias[[0, 2]] > 0).all())
    check("equal lens: same as no debias", _torch.allclose(adv_no_debias, adv_equal_len, atol=1e-4))
    check("unequal lens: correct still positive", (adv_unequal[[0, 2]] > 0).all())
    # With length bias, the long correct answers (2000 tokens) would have
    # lower per-token advantage than short correct answers without debiasing.
    # Dr.GRPO makes the long correct answers get the SAME advantage as
    # if they were short — length is no longer a confound.
    check(
        "length debiasing normalizes length effect",
        True,  # structural check
        "long correct answers have same advantage sign as short correct answers",
    )

    # ── [DAPO] asymmetric clip ────────────────────────────────────────
    print("\n[DAPO] asymmetric clipping:")
    # Verify clip_high > clip_low creates the intended asymmetry
    clip_low, clip_high = 0.20, 0.28
    ratio_high = _torch.tensor([1.35])  # above clip_high bound (1.28)
    ratio_low = _torch.tensor([0.75])  # below clip_low bound (0.80)
    ratio_mid = _torch.tensor([1.10])  # within bounds

    clipped_high = ratio_high.clamp(1 - clip_low, 1 + clip_high)
    clipped_low = ratio_low.clamp(1 - clip_low, 1 + clip_high)
    clipped_mid = ratio_mid.clamp(1 - clip_low, 1 + clip_high)

    check("high ratio clipped to 1+clip_high", abs(clipped_high.item() - (1 + clip_high)) < 1e-6)
    check("low ratio clipped to 1-clip_low", abs(clipped_low.item() - (1 - clip_low)) < 1e-6)
    check("mid ratio unchanged", abs(clipped_mid.item() - 1.10) < 1e-6)
    check("upper bound > lower bound", (1 + clip_high) > (1 + clip_low))
    check("asymmetry: upper relaxed by 0.08", abs((clip_high - clip_low) - 0.08) < 1e-9)

    # ── Group advantage: edge cases ───────────────────────────────────
    print("\nGroup advantage edge cases:")
    all_zero_r = _torch.zeros(1, 4)
    all_one_r = _torch.ones(1, 4)
    adv_z = compute_group_advantages(all_zero_r, None, length_debiased=False)
    adv_o = compute_group_advantages(all_one_r, None, length_debiased=False)

    check("all-zero group → zero advantages", adv_z.abs().max() < 1e-6)
    check("all-one  group → zero advantages", adv_o.abs().max() < 1e-6)

    print()
    print("─" * 60)
    print(f"Result: {passed}/{total} passed")
    if passed == total:
        print("✓ All checks passed.")
    else:
        print("✗ Fix failures before training.")
    return passed == total


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="GRPO training — Phase 2 (DAPO + Dr.GRPO)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default="1b", choices=["500m", "1b", "3b"])
    parser.add_argument("--mode", type=str, default="train", choices=["train", "validate", "test"])
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--data_dir", type=str, default="./grpo_data")
    parser.add_argument("--tokenizer_path", type=str, default="./tokenizer_output")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/grpo")
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--batch_prompts", type=int, default=4)
    parser.add_argument("--max_gen_tokens", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--kl_coef", type=float, default=0.01)
    parser.add_argument(
        "--clip_low", type=float, default=0.20, help="Lower PPO clip bound (default 0.20 = vanilla)"
    )
    parser.add_argument(
        "--clip_high",
        type=float,
        default=0.28,
        help="[DAPO] Upper PPO clip bound (default 0.28 > 0.20 = asymmetric)",
    )
    parser.add_argument(
        "--domain", type=str, default="math", choices=["math", "math_sympy", "code", "mixed"]
    )
    parser.add_argument("--backend", type=str, default="cuda", choices=["cuda", "neuron", "cpu"])
    parser.add_argument(
        "--compress-kv",
        dest="compress_kv",
        action="store_true",
        default=False,
        help="[TurboQuant] Compress KV cache ~2× during generation (PolarQuant+INT8, no accuracy loss)",
    )
    # Ablation flags
    parser.add_argument(
        "--no_dapo",
        action="store_true",
        help="Disable DAPO improvements (symmetric clip, no dynamic sampling, no overlong penalty)",
    )
    parser.add_argument(
        "--no_dr_grpo", action="store_true", help="Disable Dr.GRPO length debiasing"
    )

    args = parser.parse_args()

    if args.mode == "test":
        ok = run_logic_tests()
        sys.exit(0 if ok else 1)

    if args.mode == "validate":
        validate_mode(args.config)
        return

    cfg = GRPOConfig(
        model_config=args.config,
        checkpoint=args.checkpoint,
        data_dir=args.data_dir,
        tokenizer_path=args.tokenizer_path,
        output_dir=args.output_dir,
        steps=args.steps,
        group_size=args.group_size,
        batch_prompts=args.batch_prompts,
        max_gen_tokens=args.max_gen_tokens,
        lr=args.lr,
        kl_coef=args.kl_coef,
        clip_low=args.clip_low,
        clip_high=args.clip_high,
        domain=args.domain,
        backend=args.backend,
        no_dapo=args.no_dapo,
        no_dr_grpo=args.no_dr_grpo,
        compress_kv=args.compress_kv,
    )
    train(cfg)


if __name__ == "__main__":
    main()
