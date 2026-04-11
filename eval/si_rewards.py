"""
si_rewards.py
=============
Structured Intent reward functions for evaluating whether the model
can generate valid SI specs (JSON with function/signature/behavior fields).

These are eval-only rewards — they don't depend on actually compiling or
running code. They measure whether the model's output is a well-formed
structured specification that an SI model (si-go-v1) could consume.

Scoring tiers:
  - format:  Can we extract JSON from the completion at all?
  - fields:  Does the JSON have the required top-level fields?
  - quality: Are the fields populated with plausible content?
"""

import json
import re
from typing import Optional

# Required fields for a valid SI spec
REQUIRED_FIELDS = {"function", "signature", "behavior"}
OPTIONAL_FIELDS = {"constraints", "examples"}
ALL_FIELDS = REQUIRED_FIELDS | OPTIONAL_FIELDS


def extract_json_from_completion(completion: str) -> Optional[dict]:
    """
    Extract JSON from a model completion.

    The expected format is:
      <think>...reasoning...</think>
      { ... json spec ... }

    But the model might produce JSON without think tags, or embed it in text.
    We try multiple extraction strategies in order of specificity.
    """
    # Strategy 1: text after </think> tag
    think_end = completion.find("</think>")
    if think_end >= 0:
        after_think = completion[think_end + len("</think>") :].strip()
    else:
        after_think = completion.strip()

    # Strategy 2: find the first { ... } block (greedy match for outermost braces)
    # Use a simple brace-counting parser rather than regex for nested JSON
    json_str = _extract_outermost_json(after_think)
    if json_str is None:
        # Try the full completion as fallback
        json_str = _extract_outermost_json(completion)

    if json_str is None:
        return None

    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        return None


def _extract_outermost_json(text: str) -> Optional[str]:
    """Find the outermost {...} block in text using brace counting."""
    start = text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    return None


def reward_si_format(completion: str) -> float:
    """
    Format reward: can we extract valid JSON from the completion?

    Returns:
      1.0 — valid JSON extracted after </think>
      0.5 — valid JSON found but no </think> tag
      0.0 — no valid JSON found
    """
    has_think_end = "</think>" in completion
    parsed = extract_json_from_completion(completion)

    if parsed is None:
        return 0.0
    if has_think_end:
        return 1.0
    return 0.5


def reward_si_fields(completion: str) -> float:
    """
    Field completeness reward: does the JSON have the required fields?

    Returns fraction of required fields present (0.0 to 1.0).
    Required: function, signature, behavior
    """
    parsed = extract_json_from_completion(completion)
    if parsed is None:
        return 0.0

    present = sum(1 for f in REQUIRED_FIELDS if f in parsed)
    return present / len(REQUIRED_FIELDS)


def reward_si_quality(completion: str) -> float:
    """
    Quality reward: are the fields populated with plausible content?

    Checks:
      - function: is a non-empty string and valid identifier
      - signature: has 'inputs' and 'output' sub-fields
      - behavior: is a non-empty string
      - constraints: is a non-empty list (bonus)
      - examples: is a non-empty list with input/output entries (bonus)

    Returns score from 0.0 to 1.0 (average of checks that apply).
    """
    parsed = extract_json_from_completion(completion)
    if parsed is None:
        return 0.0

    checks = []

    # function: non-empty string, valid identifier
    fn = parsed.get("function", "")
    if isinstance(fn, str) and fn and re.match(r"^[a-zA-Z_][a-zA-Z0-9_]*$", fn):
        checks.append(1.0)
    else:
        checks.append(0.0)

    # signature: has inputs and output
    sig = parsed.get("signature", {})
    if isinstance(sig, dict) and "inputs" in sig and "output" in sig:
        checks.append(1.0)
    elif isinstance(sig, dict) and ("inputs" in sig or "output" in sig):
        checks.append(0.5)
    else:
        checks.append(0.0)

    # behavior: non-empty string
    beh = parsed.get("behavior", "")
    if isinstance(beh, str) and len(beh) > 5:
        checks.append(1.0)
    elif isinstance(beh, str) and beh:
        checks.append(0.5)
    else:
        checks.append(0.0)

    # constraints: non-empty list (bonus, not required)
    con = parsed.get("constraints", [])
    if isinstance(con, list) and len(con) > 0:
        checks.append(1.0)
    else:
        checks.append(0.0)

    # examples: list with at least one entry that has input/output
    exs = parsed.get("examples", [])
    if isinstance(exs, list) and len(exs) > 0:
        # Check if at least one example has input and output
        has_good = any(isinstance(e, dict) and ("input" in e and "output" in e) for e in exs)
        checks.append(1.0 if has_good else 0.5)
    else:
        checks.append(0.0)

    return sum(checks) / len(checks) if checks else 0.0


def reward_si_combined(completion: str, format_weight: float = 0.3) -> float:
    """
    Combined SI reward: weighted average of format, fields, and quality.

    This mirrors combined_reward() in grpo.py but for structured intent.
    """
    fmt = reward_si_format(completion)
    fields = reward_si_fields(completion)
    quality = reward_si_quality(completion)

    # If no JSON at all, everything is 0
    if fmt == 0.0:
        return 0.0

    # Weighted: format matters less once achieved, fields and quality matter more
    return format_weight * fmt + (1 - format_weight) * (0.5 * fields + 0.5 * quality)
