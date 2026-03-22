"""
rewards.py
==========
Reward / verification functions for Phase 2 GRPO.

Only verifiable rewards — no learned reward model (spec Section 3.3).

Why verifiable rewards only:
  A learned reward model introduces a second trained system that can be
  "gamed" by the policy through reward hacking. Verifiable rewards
  (exact match, symbolic equivalence, test-case execution) are ground-truth
  checks that cannot be fooled — the model gets credit only for genuinely
  correct answers. This follows the DeepSeek-R1 and OpenAI o1 approach.

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

import math
import re
import signal
import types
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
    pred = _normalize_math(prediction)
    truth = _normalize_math(ground_truth)
    return 1.0 if pred == truth else 0.0


def verify_math_sympy(prediction: str, ground_truth: str) -> float:
    """
    Symbolic equivalence via SymPy.
    Used for expression answers (e.g. "x^2 + 3x - 1" or "sqrt(2)/2").
    Falls back to exact match if SymPy parse fails.
    Returns 1.0 if equivalent, 0.0 otherwise.

    SymPy's sympify + simplify handles algebraic equivalences that string
    matching misses, e.g. "2*sqrt(2)/2" == "sqrt(2)", "(x+1)^2" == "x^2+2x+1".
    The diff == 0 check uses SymPy's symbolic zero, not floating-point equality.
    """
    try:
        import sympy

        pred_expr = sympy.sympify(prediction, evaluate=True)
        truth_expr = sympy.sympify(ground_truth, evaluate=True)
        # simplify(a - b) == 0 tests symbolic equality; handles trig identities,
        # polynomial expansions, and rational expressions without numeric evaluation.
        diff = sympy.simplify(pred_expr - truth_expr)
        return 1.0 if diff == 0 else 0.0
    except Exception:
        # SymPy parse can fail on ill-formed expressions (e.g. plain integers like "42",
        # LaTeX notation, or prose answers). Fall back to normalized string match.
        return verify_math_exact(prediction, ground_truth)


def _normalize_math(s: str) -> str:
    """
    Strip whitespace and normalize common number representations so that
    semantically identical answers compare equal as strings.

    Normalization steps and the edge cases each one handles:

      strip().lower()            — trim surrounding whitespace; case-insensitive compare
                                   ("Pi" == "pi", "X" == "x")

      re.sub(r"\\s+", "", s)     — remove ALL internal whitespace
                                   ("1 0 0" == "100", "x + 1" == "x+1")

      s.lstrip("+")              — remove a leading plus sign
                                   ("+42" == "42"; sign is conventional, not semantic)

      re.sub(r"\\.0+$", "", s)   — remove trailing ".0" or ".00" etc. from integers
                                   ("42.0" == "42", "7.000" == "7")
                                   Matches only when ALL post-decimal digits are zero,
                                   so "3.10" is handled by the next rule.

      re.sub(r"(\\. \\d*?)0+$", r"\\1", s)
                                 — remove trailing zeros after a decimal point
                                   ("3.140" == "3.14", "1.2300" == "1.23")
                                   The non-greedy \\d*? ensures we keep at least one
                                   digit after the decimal: "3.0" → "3." would then
                                   be caught by a subsequent lstrip/rstrip if needed.
                                   Note: "3.0" hits the previous rule first → "3".
    """
    s = s.strip().lower()
    s = re.sub(r"\s+", "", s)  # collapse/remove all whitespace ("3 . 1 4" → "3.14")
    s = s.lstrip("+")  # "+42" → "42" (leading plus is conventional, not semantic)
    s = re.sub(r"\.0+$", "", s)  # "42.000" → "42" (integer stored as float)
    s = re.sub(r"(\.\d*?)0+$", r"\1", s)  # "3.1400" → "3.14" (trailing decimal zeros)
    return s


# ---------------------------------------------------------------------------
# Code verification
# ---------------------------------------------------------------------------


@contextmanager
def _timeout(seconds: int):
    """
    Context manager that raises TimeoutError after N seconds using POSIX SIGALRM.

    Why signal-based timeout:
      Python's threading.Timer or concurrent.futures cannot forcibly interrupt
      a tight CPU-bound loop (e.g. an infinite while loop in generated code)
      because the GIL prevents the timer thread from running. SIGALRM delivers
      an OS-level interrupt that preempts the running thread regardless of GIL state,
      making it the only reliable way to halt runaway generated code in CPython.

    Alternatives and their drawbacks:
      - subprocess with timeout: cleaner isolation but higher overhead per test case;
        requires serializing inputs/outputs (pickle or JSON).
      - multiprocessing with timeout: similar to subprocess; process startup cost.
      - asyncio with timeout: only works for async/await code, not sync loops.
      - ctypes PyErr_SetInterrupt: brittle, CPython-internal, not portable.

    Limitation: SIGALRM is only available on POSIX systems (Linux, macOS).
    On Windows this context manager would need to be replaced with subprocess-based
    timeout. Also, SIGALRM is process-wide — nested _timeout() calls will conflict.
    """

    def _handler(signum, frame):
        raise TimeoutError(f"Code execution timed out after {seconds}s")

    old = signal.signal(signal.SIGALRM, _handler)  # register our handler, save old one
    signal.alarm(seconds)  # arm the OS timer
    try:
        yield
    finally:
        signal.alarm(0)  # disarm the timer (0 cancels a pending alarm)
        signal.signal(signal.SIGALRM, old)  # restore the previous signal handler


def verify_code(
    code: str,
    test_cases: list[dict],
    timeout_seconds: int = 5,
) -> float:
    """
    Execute generated code against test cases.
    Returns fraction of test cases that pass.

    test_cases: list of {"input": ..., "expected_output": ...}

    SECURITY NOTE: exec() isolation is insufficient for production use.
      exec(code, ns) runs in the same OS process as the training loop.
      Generated code can:
        - Read/write arbitrary files (os.listdir, open, etc.)
        - Import sys and call sys.exit(), killing the training process
        - Use ctypes or cffi to escape the Python sandbox entirely
        - Fork child processes that outlive the timeout
        - Exhaust memory with a large allocation before SIGALRM fires
      For production deployment, run code execution in a separate OS process
      inside a sandbox such as Docker with no-network, seccomp-bpf syscall
      filtering, a read-only filesystem, and a strict memory limit (nsjail or
      gVisor provide this). For research use, the SIGALRM timeout prevents
      infinite loops, which is the most common failure mode.

    For research use only — not production-safe.
    """
    if not test_cases:
        # No test cases means we cannot verify correctness at all; return 0 rather than
        # returning 1 (which would grant reward without evidence of correctness).
        return 0.0

    passed = 0
    for tc in test_cases:
        try:
            with _timeout(timeout_seconds):
                # Execute generated code in a fresh namespace dict.
                # ns isolates top-level names from the training process's globals,
                # but does NOT prevent access to builtins (print, open, __import__).
                # See SECURITY NOTE in the docstring for why this is insufficient
                # for production use.
                ns = {}
                exec(code, ns)  # noqa: S102

                # Find the first callable defined by the generated code.
                # ns starts empty so every key was placed there by exec —
                # no risk of accidentally finding a builtin.  Skip module
                # objects that exec occasionally leaks from import statements.
                fn = None
                for _name, _obj in ns.items():
                    if callable(_obj) and not isinstance(_obj, types.ModuleType):
                        fn = _obj
                        break

                if fn is None:
                    # Code ran without error but defined no callable — fail.
                    continue

                # Call the function with the test-case input (single argument).
                # If a problem needs multiple args, the dataset should store
                # them as a tuple and the function should accept a tuple.
                result = fn(tc["input"])

                # Compare result to expected output.  Use math.isclose for
                # floats to tolerate IEEE-754 rounding; exact equality otherwise.
                expected = tc["expected_output"]
                if isinstance(result, float) or isinstance(expected, float):
                    if math.isclose(float(result), float(expected), rel_tol=1e-6):
                        passed += 1
                else:
                    if result == expected:
                        passed += 1
        except Exception:
            # Any exception (including our TimeoutError, SyntaxError, NameError, etc.)
            # counts as a test-case failure. We intentionally catch broadly here so that
            # one badly-formed completion doesn't crash the entire reward computation loop.
            pass

    return passed / len(test_cases)  # fraction in [0, 1]; partial credit for partial solutions


# ---------------------------------------------------------------------------
# Format reward
# ---------------------------------------------------------------------------

# Pre-compiled regex for detecting a well-formed <think>...</think> block.
# re.DOTALL makes "." match newlines, which is required because the think block
# spans multiple lines in practice (e.g. multi-step reasoning traces).
# The non-greedy .*? avoids matching from the first <think> to the LAST </think>
# when multiple CoT blocks are present — it stops at the nearest </think>.
THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)


def format_reward(response: str, weight: float = 0.1) -> float:
    """
    +weight if response contains a valid <think>...</think> block, else 0.

    Encourages the model to maintain CoT structure under RL pressure.

    Why this is necessary:
      Without a format reward, GRPO may discover that very short responses
      (no CoT, just the final answer) achieve the same outcome reward with
      lower generation cost. The format reward nudges the model to keep
      the <think> block, which is needed for the reasoning capability we care about.
      Weight 0.1 is intentionally small — it should not override a correct answer.
    """
    return weight if THINK_PATTERN.search(response) else 0.0


# ---------------------------------------------------------------------------
# Composite reward
# ---------------------------------------------------------------------------


def compute_reward(
    response: str,
    ground_truth: str,
    domain: str,  # "math_exact" | "math_sympy" | "code" | "logic"
    test_cases: Optional[list] = None,
    format_weight: float = 0.1,
) -> float:
    """
    Compute total reward for a single completion.
    Total = outcome_reward + format_reward.

    outcome_reward is in {0, 1} for math/logic, [0, 1] for code (fraction passing).
    format_reward is 0 or format_weight (default 0.1).
    Maximum total reward = 1.1.
    """
    if domain == "math_exact":
        # Extract the answer portion of the response (after </think>), then exact-match normalize
        outcome = verify_math_exact(_extract_answer(response), ground_truth)
    elif domain == "math_sympy":
        # Use SymPy for algebraic/symbolic answers where string normalization is insufficient
        outcome = verify_math_sympy(_extract_answer(response), ground_truth)
    elif domain == "code":
        # Pass the full response (including any prose); verify_code extracts executable blocks
        outcome = verify_code(response, test_cases or [])
    elif domain == "logic":
        # Logic puzzles have deterministic string answers; reuse exact-match verifier
        outcome = verify_math_exact(_extract_answer(response), ground_truth)
    else:
        raise ValueError(f"Unknown domain: {domain}")

    fmt = format_reward(response, weight=format_weight)
    return outcome + fmt


def _extract_answer(response: str) -> str:
    """
    Extract the final answer from a model response.

    Heuristic: the answer lives AFTER the closing </think> tag.
    The model is trained to produce:
        <think>
        [multi-step reasoning trace]
        </think>
        [final answer here]

    Edge cases handled:
      - No </think> present (response is just the answer, no CoT):
        split("</think>") returns ["response"], parts[-1] == response.
        We return the full stripped response, so exact-match still works.
      - Multiple </think> tags (malformed response with nested or repeated CoT blocks):
        split("</think>") returns more than two parts.
        parts[-1] is the text after the LAST </think>, which is the most likely
        location of the final answer. The intermediate parts are discarded.
      - Trailing whitespace after </think>:
        strip() removes it so the answer string compares cleanly.
      - Empty answer after </think> (model stopped generating too early):
        strip() returns ""; verify_math_exact("", ground_truth) returns 0.0.
    """
    # Split on the closing think tag; take everything after the last occurrence.
    parts = response.split("</think>")
    return parts[-1].strip() if len(parts) > 1 else response.strip()
