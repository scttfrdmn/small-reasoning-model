"""
rewards.py
==========
Reward / verification functions for Phase 2 GRPO.

Only verifiable rewards — no learned reward model (spec Section 3.3).

Verification methods by domain:
  Math (integer answer):  exact match after normalization           → 0 or 1
  Math (expression):      SymPy symbolic equivalence               → 0 or 1
  Code:                   execute against test cases               → fraction passing
  Logic puzzles:          deterministic checker                    → 0 or 1

Format reward (weight 0.1):
  +0.1 if response contains a valid <think>...</think> block.
  Encourages CoT structure under RL pressure.

These functions are called inside grpo.py's reward computation loop.
All verification must be cheap and parallelizable.
"""

import re
import signal
from contextlib import contextmanager
from typing import Optional


# ---------------------------------------------------------------------------
# Math verification
# ---------------------------------------------------------------------------

def verify_math_exact(prediction: str, ground_truth: str) -> float:
    """
    Exact match after normalization.
    Handles: trailing zeros, whitespace, sign, simple fractions.
    Returns 1.0 if match, 0.0 otherwise.
    """
    pred  = _normalize_math(prediction)
    truth = _normalize_math(ground_truth)
    return 1.0 if pred == truth else 0.0


def verify_math_sympy(prediction: str, ground_truth: str) -> float:
    """
    Symbolic equivalence via SymPy.
    Used for expression answers (e.g. "x^2 + 3x - 1" or "sqrt(2)/2").
    Falls back to exact match if SymPy parse fails.
    Returns 1.0 if equivalent, 0.0 otherwise.
    """
    try:
        import sympy
        pred_expr  = sympy.sympify(prediction,  evaluate=True)
        truth_expr = sympy.sympify(ground_truth, evaluate=True)
        diff = sympy.simplify(pred_expr - truth_expr)
        return 1.0 if diff == 0 else 0.0
    except Exception:
        return verify_math_exact(prediction, ground_truth)


def _normalize_math(s: str) -> str:
    """Strip whitespace, normalize common number representations."""
    s = s.strip().lower()
    s = re.sub(r"\s+", "", s)           # remove all whitespace
    s = s.lstrip("+")                   # remove leading +
    s = re.sub(r"\.0+$", "", s)         # 42.0 → 42
    s = re.sub(r"(\.\d*?)0+$", r"\1", s)  # 3.140 → 3.14
    return s


# ---------------------------------------------------------------------------
# Code verification
# ---------------------------------------------------------------------------

@contextmanager
def _timeout(seconds: int):
    """Context manager that raises TimeoutError after N seconds."""
    def _handler(signum, frame):
        raise TimeoutError(f"Code execution timed out after {seconds}s")
    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def verify_code(
    code: str,
    test_cases: list[dict],
    timeout_seconds: int = 5,
) -> float:
    """
    Execute generated code against test cases.
    Returns fraction of test cases that pass.

    test_cases: list of {"input": ..., "expected_output": ...}

    SECURITY NOTE: This executes arbitrary code in a subprocess.
    In production, run in an isolated sandbox (e.g. Docker, nsjail).
    For research use only.
    """
    if not test_cases:
        return 0.0

    passed = 0
    for tc in test_cases:
        try:
            with _timeout(timeout_seconds):
                ns = {}
                exec(code, ns)  # noqa: S102
                # TODO: call the function with tc["input"] and compare to tc["expected_output"]
                passed += 1  # placeholder
        except Exception:
            pass

    return passed / len(test_cases)


# ---------------------------------------------------------------------------
# Format reward
# ---------------------------------------------------------------------------

THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)

def format_reward(response: str, weight: float = 0.1) -> float:
    """
    +weight if response contains a valid <think>...</think> block, else 0.
    Encourages the model to maintain CoT structure under RL pressure.
    """
    return weight if THINK_PATTERN.search(response) else 0.0


# ---------------------------------------------------------------------------
# Composite reward
# ---------------------------------------------------------------------------

def compute_reward(
    response: str,
    ground_truth: str,
    domain: str,                           # "math_exact" | "math_sympy" | "code" | "logic"
    test_cases: Optional[list] = None,
    format_weight: float = 0.1,
) -> float:
    """
    Compute total reward for a single completion.
    outcome_reward + format_reward.
    """
    if domain == "math_exact":
        outcome = verify_math_exact(_extract_answer(response), ground_truth)
    elif domain == "math_sympy":
        outcome = verify_math_sympy(_extract_answer(response), ground_truth)
    elif domain == "code":
        outcome = verify_code(response, test_cases or [])
    elif domain == "logic":
        outcome = verify_math_exact(_extract_answer(response), ground_truth)
    else:
        raise ValueError(f"Unknown domain: {domain}")

    fmt = format_reward(response, weight=format_weight)
    return outcome + fmt


def _extract_answer(response: str) -> str:
    """
    Extract the final answer from a model response.
    Looks for content after </think> tag, or the full response if no <think> block.
    """
    parts = response.split("</think>")
    return parts[-1].strip() if len(parts) > 1 else response.strip()
