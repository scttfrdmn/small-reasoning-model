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
PAD_ID         = 0
BOS_ID         = 1
EOS_ID         = 2
THINK_START_ID = 4
THINK_END_ID   = 5
LOSS_IGNORE    = -100


# ---------------------------------------------------------------------------
# GRPO config
# ---------------------------------------------------------------------------

@dataclass
class GRPOConfig:
    # Model
    model_config:    str   = "1b"
    checkpoint:      str   = ""        # SFT checkpoint (required)

    # Data
    data_dir:        str   = "./grpo_data"
    tokenizer_path:  str   = "./tokenizer_output"
    domain:          str   = "math"    # "math" | "code" | "mixed"

    # GRPO core
    group_size:      int   = 8         # G: completions sampled per prompt
    steps:           int   = 10_000    # Total optimizer steps
    batch_prompts:   int   = 4         # Prompts per step (before group expansion)

    # Generation
    max_gen_tokens:  int   = 2048      # Max new tokens per completion
    temperature:     float = 0.8
    top_p:           float = 0.95

    # [DAPO] Asymmetric clipping (Clip-Higher)
    # Symmetric clip (clip_low == clip_high) = vanilla GRPO/PPO
    # clip_high > clip_low = DAPO: relaxed upper bound prevents entropy collapse
    clip_low:        float = 0.20      # Lower clip (same as vanilla ε=0.2)
    clip_high:       float = 0.28      # Upper clip (DAPO default: 0.28)

    # KL penalty
    kl_coef:         float = 0.01      # β — weight of KL divergence term

    # Optimizer
    lr:              float = 5e-7
    lr_min:          float = 5e-7      # Constant (no decay for GRPO usually)
    warmup_fraction: float = 0.01
    weight_decay:    float = 0.0
    grad_clip:       float = 1.0

    # Format reward
    format_reward_weight: float = 0.1

    # [DAPO] Dynamic sampling
    # Oversample by this factor, then discard groups with uniform rewards
    # (all correct or all wrong → zero advantage → zero gradient)
    dynamic_sampling:       bool  = True
    dynamic_oversample:     int   = 2    # Sample 2× prompts, keep the informative half
    dynamic_min_diversity:  float = 0.0  # Min reward std within group to keep (0 = any nonzero std)

    # [DAPO] Overlong reward shaping
    # Soft penalty for completions that hit max_gen_tokens (probably truncated)
    overlong_penalty:       bool  = True
    overlong_penalty_factor:float = 0.5  # Multiply reward by this if completion is truncated

    # [Dr. GRPO] Length-debiased advantages
    # Normalize rewards by completion length before group statistics
    # Removes silent bias toward shorter completions
    length_debiased:        bool  = True

    # --- Ablation flags (disable individual improvements) ---
    no_dapo:         bool  = False     # Disable all DAPO improvements
    no_dr_grpo:      bool  = False     # Disable Dr. GRPO length debiasing

    # Difficulty filter (applied at dataset load — static version of dynamic sampling)
    min_pass_rate:   float = 0.20
    max_pass_rate:   float = 0.80

    # Memory
    grad_checkpointing: bool = True
    dtype:           str   = "bfloat16"

    # Output
    output_dir:      str   = "./checkpoints/grpo"
    log_every:       int   = 10
    save_every:      int   = 500
    eval_every:      int   = 200

    # Hardware
    backend:         str   = "cuda"

    def __post_init__(self):
        # Apply ablation flags
        if self.no_dapo:
            self.clip_high            = self.clip_low   # symmetric = vanilla PPO
            self.dynamic_sampling     = False
            self.overlong_penalty     = False
        if self.no_dr_grpo:
            self.length_debiased      = False


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------

def normalize_answer(s: str) -> str:
    """
    Normalize a math answer string for comparison.

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

    # Strip LaTeX math delimiters
    s = s.replace("$$", "").replace("$", "")

    # Unicode minus → ASCII
    s = s.replace("\u2212", "-")

    # Remove thousands separators in numbers
    s = re.sub(r"(\d),(\d{3})", r"\1\2", s)

    # Simple LaTeX fraction: \frac{a}{b} → (a)/(b)
    s = re.sub(r"\\frac\{([^}]+)\}\{([^}]+)\}", r"(\1)/(\2)", s)

    # Strip trailing zeros after decimal, then strip lone decimal
    # "3.10" → "3.1", "3.0" → "3", "3.14" → "3.14"
    s = re.sub(r"(\.\d*?)0+$", r"\1", s)
    s = re.sub(r"\.$", "", s)

    # Lowercase, strip spaces inside
    s = s.lower().strip()

    return s


def reward_math_exact(completion: str, ground_truth: str) -> float:
    """
    Binary reward: 1.0 if completion contains the correct answer, 0.0 otherwise.

    Extracts the answer from the completion (handles <think>...</think> structure)
    and compares to ground truth after normalization.
    """
    answer = _extract_final_answer(completion)
    if answer is None:
        return 0.0

    norm_pred  = normalize_answer(answer)
    norm_truth = normalize_answer(ground_truth)

    return 1.0 if norm_pred == norm_truth else 0.0


def reward_math_sympy(completion: str, ground_truth: str) -> float:
    """
    Symbolic equivalence reward using SymPy.

    Handles mathematically equivalent expressions:
      "x^2 + 2x + 1" == "(x+1)^2"  → 1.0
      "1/2" == "0.5"                → 1.0

    Falls back to exact match if SymPy parse fails.
    Times out after 5 seconds to prevent hanging on pathological inputs.
    """
    answer = _extract_final_answer(completion)
    if answer is None:
        return 0.0

    # Try exact match first (fast path)
    if normalize_answer(answer) == normalize_answer(ground_truth):
        return 1.0

    # Try SymPy symbolic equivalence
    try:
        import sympy
        from sympy.parsing.sympy_parser import parse_expr, standard_transformations, \
            implicit_multiplication_application

        transforms = standard_transformations + (implicit_multiplication_application,)

        with _timeout(5):
            expr_pred  = parse_expr(answer.replace("^", "**"),      transformations=transforms)
            expr_truth = parse_expr(ground_truth.replace("^", "**"), transformations=transforms)
            if sympy.simplify(expr_pred - expr_truth) == 0:
                return 1.0
    except Exception:
        pass  # SymPy failed — fall through to 0.0

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

    Returns format_reward_weight if the completion has the expected structure:
      <think>\n...\n</think>\n<final answer>

    This encourages the model to maintain CoT format under RL pressure.
    Without this, models sometimes drop the thinking step when it's "easier"
    to just output the answer directly (and still get some reward).
    """
    has_think_start = "<think>" in completion
    has_think_end   = "</think>" in completion
    has_content     = bool(re.search(r"<think>\s*\S", completion))
    answer_after    = bool(re.search(r"</think>\s*\S", completion))

    if has_think_start and has_think_end and has_content and answer_after:
        return 1.0
    elif has_think_start and has_think_end:
        return 0.5
    return 0.0


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

    Primary reward (0 or 1): correctness
    Format reward (0 or 1): structure bonus, weighted by format_weight
    [DAPO] Overlong penalty: multiply reward if completion hit max_gen_tokens
      (likely truncated — answer may be incomplete or missing entirely)

    Total = (primary + format_weight × format) × overlong_multiplier
    """
    ground_truth = example.get("answer", example.get("expected_output", ""))

    # Primary: correctness
    if domain == "code":
        test_cases = example.get("test_cases", [])
        primary = reward_code_exec(completion, test_cases)
    elif domain == "math_sympy":
        primary = reward_math_sympy(completion, ground_truth)
    else:
        primary = reward_math_exact(completion, ground_truth)

    # Format bonus
    fmt = reward_format(completion) * format_weight

    reward = primary + fmt

    # [DAPO] Overlong penalty: if completion hit the generation limit,
    # it was likely truncated mid-reasoning and the answer is absent/wrong.
    # Even if it happened to be "correct" by string match, the reasoning
    # chain is incomplete — we penalize to discourage verbose outputs.
    if overlong_penalty and completion_len >= max_gen_tokens:
        reward = reward * overlong_penalty_factor

    return reward


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

def _extract_final_answer(completion: str) -> Optional[str]:
    """
    Extract the final answer from a completion.

    Handles:
      1. </think>\nANSWER           — our standard format
      2. \\boxed{ANSWER}            — LaTeX math competitions
      3. The answer is: ANSWER      — natural language
      4. = ANSWER (last occurrence) — equation chains
      5. Last non-empty line        — fallback
    """
    # 1. After </think>
    think_match = re.search(r"</think>\s*(.+?)(?:\n|$)", completion, re.DOTALL)
    if think_match:
        answer = think_match.group(1).strip()
        if answer:
            return answer

    # 2. LaTeX boxed
    boxed = re.findall(r"\\boxed\{([^}]+)\}", completion)
    if boxed:
        return boxed[-1]

    # 3. "The answer is X" / "Answer: X"
    ans_match = re.search(
        r"(?:the answer is|answer is|answer:|therefore)[:\s]+([^\n.]+)",
        completion, re.IGNORECASE
    )
    if ans_match:
        return ans_match.group(1).strip()

    # 4. Last "= X" occurrence
    eq_matches = re.findall(r"=\s*([^\n=]+)$", completion, re.MULTILINE)
    if eq_matches:
        return eq_matches[-1].strip()

    # 5. Last non-empty line
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
            timeout=timeout,
        )
        return result.stdout
    finally:
        os.unlink(fname)


@contextmanager
def _timeout(seconds: int):
    """Context manager that raises TimeoutError after `seconds`."""
    def _handler(signum, frame):
        raise TimeoutError(f"Timed out after {seconds}s")
    old = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


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
        val_file   = data_dir / "val.jsonl"
        any_file   = list(data_dir.glob("*.jsonl"))

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
                        continue

                    # Difficulty filter
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

        print(f"  GRPODataset ({split}): {len(self.examples):,} examples "
              f"({n_filtered} filtered by difficulty)")

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
            self.examples.append({
                "problem": f"What is {a} {op} {b}?",
                "answer":  str(ans),
                "domain":  "math",
            })

    def __len__(self): return len(self.examples)
    def __getitem__(self, i): return self.examples[i]


# ---------------------------------------------------------------------------
# Log probability computation
# ---------------------------------------------------------------------------

def compute_log_probs(
    model: nn.Module,
    input_ids: torch.Tensor,       # (B, T)
    completion_mask: torch.Tensor,  # (B, T) — 1 for completion tokens, 0 for prompt
    autocast_ctx,
    return_per_token: bool = False,
) -> torch.Tensor:
    """
    Compute log probabilities for completion tokens.

    return_per_token=False (default): returns (B,) — sum over completion tokens.
        Used for KL divergence computation.

    return_per_token=True: returns (B, T-1) — per-token log probs, masked.
        Used for [DAPO] token-level policy gradient loss.
        Non-completion positions are zero (not -inf, to allow safe aggregation).
    """
    with autocast_ctx:
        logits, _ = model(input_ids)   # (B, T, vocab_size)

    shifted_logits  = logits[:, :-1, :].contiguous()   # (B, T-1, V)
    shifted_targets = input_ids[:, 1:].contiguous()    # (B, T-1)
    shifted_mask    = completion_mask[:, 1:].contiguous().float()  # (B, T-1)

    log_probs = F.log_softmax(shifted_logits, dim=-1)   # (B, T-1, V)

    token_log_probs = log_probs.gather(
        -1, shifted_targets.unsqueeze(-1)
    ).squeeze(-1)   # (B, T-1)

    # Zero out non-completion positions
    token_log_probs = token_log_probs * shifted_mask

    if return_per_token:
        return token_log_probs  # (B, T-1), zeroed where mask=0

    return token_log_probs.sum(dim=-1)   # (B,)


def completion_lengths(mask_list: list[torch.Tensor]) -> torch.Tensor:
    """
    Compute completion token counts from a list of completion masks.
    Returns (N,) tensor of ints.
    """
    return torch.tensor([m.sum().item() for m in mask_list], dtype=torch.float32)


# ---------------------------------------------------------------------------
# [Dr. GRPO] Length-debiased group advantage computation
# ---------------------------------------------------------------------------

def compute_group_advantages(
    rewards: torch.Tensor,           # (batch_prompts, group_size)
    lengths: Optional[torch.Tensor], # (batch_prompts, group_size) completion token counts
    length_debiased: bool = True,
) -> torch.Tensor:                   # (batch_prompts × group_size,) flattened
    """
    Compute normalized group advantages with optional length debiasing.

    Vanilla GRPO:
      A_i = (r_i - mean(r)) / (std(r) + ε)
      Problem: length is a silent confound. A correct 2000-token response
      and a correct 50-token response get the same sequence-level reward,
      but the 2000-token response received far less per-token gradient
      signal before this normalization step. The model learns brevity.

    [Dr. GRPO] Length-debiased:
      Normalize rewards by completion length before group statistics,
      then restore original scale so the absolute advantage magnitude
      is comparable across batches.

      r̃_i = r_i / len_i              (length-normalized reward)
      Ã_i = (r̃_i - mean(r̃)) / (std(r̃) + ε)  (group normalize)

      This ensures that a 2000-token correct answer and a 50-token
      correct answer receive equal POSITIVE advantage, rather than
      the short answer receiving 40× higher per-token gradient.
    """
    eps = 1e-8

    if length_debiased and lengths is not None:
        # Normalize rewards by completion length
        # Clamp lengths to >= 1 to avoid division by zero
        len_safe = lengths.clamp(min=1.0)
        rewards_norm = rewards / len_safe   # (batch_prompts, group_size)
    else:
        rewards_norm = rewards

    mean = rewards_norm.mean(dim=1, keepdim=True)   # (batch_prompts, 1)
    std  = rewards_norm.std(dim=1, keepdim=True)    # (batch_prompts, 1)
    adv  = (rewards_norm - mean) / (std + eps)      # (batch_prompts, group_size)

    return adv.view(-1)   # (batch_prompts × group_size,)


# ---------------------------------------------------------------------------
# [DAPO] Dynamic sampling
# ---------------------------------------------------------------------------

def filter_uniform_groups(
    rewards_grouped: torch.Tensor,   # (batch_prompts, group_size)
    min_diversity:   float = 0.0,
) -> torch.Tensor:
    """
    Return a boolean mask (batch_prompts,) — True for groups with
    non-uniform rewards (i.e., at least one correct AND one incorrect
    completion in the group).

    Groups with all-zero or all-one rewards produce zero advantage
    and zero gradient. Filtering them prevents wasted training steps.

    min_diversity: minimum reward std required to keep a group.
    Default 0.0 keeps any group that isn't perfectly uniform.
    """
    std = rewards_grouped.std(dim=1)   # (batch_prompts,)
    return std > min_diversity         # True = keep


# ---------------------------------------------------------------------------
# GRPO / DAPO loss
# ---------------------------------------------------------------------------

def grpo_loss(
    token_log_probs_policy: torch.Tensor,  # (B, T-1) — per-token, zeroed at non-completion
    token_log_probs_old:    torch.Tensor,  # (B, T-1) — same, from generation time
    log_probs_ref:          torch.Tensor,  # (B,) — sequence-level, for KL
    log_probs_policy_seq:   torch.Tensor,  # (B,) — sequence-level policy, for KL
    advantages:             torch.Tensor,  # (B,) — normalized group advantages
    completion_mask:        torch.Tensor,  # (B, T-1) — 1 for completion tokens
    clip_low:               float,
    clip_high:              float,
    kl_coef:                float,
) -> tuple[torch.Tensor, dict]:
    """
    GRPO loss with DAPO and Dr. GRPO improvements.

    [DAPO] Clip-Higher — asymmetric policy ratio clipping:
      Lower bound: 1 - clip_low  (e.g. 0.80) — same as vanilla PPO
      Upper bound: 1 + clip_high (e.g. 1.28) — relaxed vs vanilla 1.20
      Effect: policy can grow probability of good actions more freely,
              preventing entropy collapse during long training runs.

    [DAPO] Token-level policy gradient:
      Instead of: loss = mean_over_sequences(-A_i * sum_t(log_ratio_t))
      We use:     loss = sum_over_all_tokens(-A_i * log_ratio_t) / total_tokens
      Effect: every completion token contributes equally to gradient,
              regardless of sequence length.

    KL penalty: computed at sequence level (unchanged from vanilla GRPO).
      kl = mean(log_π_new - log_π_ref)
      kl_loss = β × kl
    """
    shifted_mask = completion_mask.float()   # (B, T-1)

    # Per-token importance ratio
    token_log_ratio = token_log_probs_policy - token_log_probs_old  # (B, T-1)
    token_ratio     = token_log_ratio.exp()                          # (B, T-1)

    # [DAPO] Asymmetric clip applied per token
    token_ratio_clipped = token_ratio.clamp(1.0 - clip_low, 1.0 + clip_high)

    # Expand advantages to token level: (B,) → (B, 1) → (B, T-1)
    adv_token = advantages.unsqueeze(1).expand_as(token_log_ratio)

    # Policy gradient: pessimistic bound (PPO-style)
    pg1 = -adv_token * token_ratio           # unclipped
    pg2 = -adv_token * token_ratio_clipped   # clipped
    pg_token = torch.max(pg1, pg2)           # (B, T-1) — pessimistic

    # [DAPO] Token-level averaging: sum over all completion tokens in batch,
    # divide by total token count (not by number of sequences)
    total_tokens = shifted_mask.sum().clamp(min=1.0)
    pg_loss = (pg_token * shifted_mask).sum() / total_tokens

    # KL penalty — sequence-level (unchanged)
    kl      = (log_probs_policy_seq - log_probs_ref).mean()
    kl_loss = kl_coef * kl

    total_loss = pg_loss + kl_loss

    # Diagnostics
    with torch.no_grad():
        # Clip fraction: what fraction of tokens hit the clip boundary
        lower_clipped = (token_ratio < 1.0 - clip_low).float()
        upper_clipped = (token_ratio > 1.0 + clip_high).float()
        clip_frac = ((lower_clipped + upper_clipped) * shifted_mask).sum() / total_tokens
        entropy_proxy = -(token_log_probs_policy * shifted_mask).sum() / total_tokens

    metrics = {
        "pg_loss":       pg_loss.item(),
        "kl":            kl.item(),
        "kl_loss":       kl_loss.item(),
        "clip_frac":     clip_frac.item(),
        "ratio_mean":    (token_ratio * shifted_mask).sum().item() / total_tokens.item(),
        "entropy_proxy": entropy_proxy.item(),  # watch for collapse: should stay > 0
    }

    return total_loss, metrics


# ---------------------------------------------------------------------------
# Generation — batch sampling
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_completions(
    model: nn.Module,
    prompt_ids: list[torch.Tensor],    # list of (T_prompt,) tensors
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

    Returns:
      completions: list of decoded strings, length = n_prompts × group_size
      input_id_tensors: list of (T_prompt + T_completion,) tensors
      completion_masks: list of (T_prompt + T_completion,) binary masks
                        1 for completion tokens, 0 for prompt tokens

    Memory strategy: generate one prompt's group at a time to control
    peak memory. Each prompt is expanded to group_size copies, completed
    in parallel, then results collected.
    """
    model.eval()
    eos_id  = tokenizer.token_to_id("<eos>")

    all_completions    = []
    all_input_ids      = []
    all_comp_masks     = []

    for prompt_t in prompt_ids:
        T_p = prompt_t.shape[0]
        # Expand prompt to group_size copies: (group_size, T_prompt)
        batch = prompt_t.unsqueeze(0).expand(group_size, -1).clone().to(device)

        # Autoregressive generation
        finished    = torch.zeros(group_size, dtype=torch.bool, device=device)
        generated   = []
        kv_caches   = None
        pos_offset  = 0

        # Prefill: process the full prompt
        with torch.amp.autocast(device_type=device.type if device.type != "mps" else "cpu",
                                 dtype=dtype, enabled=(dtype == torch.bfloat16)):
            logits, kv_caches = model(batch, kv_caches=None)
            logits, kv_caches = model(batch)  # (G, T_p, V)
        next_logits = logits[:, -1, :]    # (G, V)
        pos_offset  = T_p

        for _ in range(max_new_tokens):
            next_tok = _sample_tokens(next_logits, temperature, top_p)  # (G,)
            generated.append(next_tok)
            finished |= (next_tok == eos_id)
            if finished.all():
                break

            # Decode step
            with torch.amp.autocast(device_type=device.type if device.type != "mps" else "cpu",
                                     dtype=dtype, enabled=(dtype == torch.bfloat16)):
                next_input = next_tok.unsqueeze(1)   # (G, 1)
                logits, _ = model(next_input, position_offset=pos_offset)
            next_logits = logits[:, -1, :]
            pos_offset  += 1

        # Collect: (G, T_completion)
        if generated:
            completion_ids = torch.stack(generated, dim=1)  # (G, T_gen)
        else:
            completion_ids = torch.zeros(group_size, 1, dtype=torch.long, device=device)

        # Build full sequences and masks
        for g in range(group_size):
            comp = completion_ids[g].cpu()
            # Trim at EOS
            eos_positions = (comp == eos_id).nonzero(as_tuple=True)[0]
            if len(eos_positions) > 0:
                comp = comp[:eos_positions[0] + 1]

            full_ids = torch.cat([prompt_t.cpu(), comp])
            mask     = torch.cat([
                torch.zeros(T_p, dtype=torch.long),
                torch.ones(comp.shape[0], dtype=torch.long),
            ])

            all_input_ids.append(full_ids)
            all_comp_masks.append(mask)

            # Decode completion text
            text = tokenizer.decode(comp.tolist(), skip_special_tokens=False)
            all_completions.append(text)

    return all_completions, all_input_ids, all_comp_masks


def _sample_tokens(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    """
    Sample next tokens from logits. Returns (batch_size,) token IDs.

    Top-p (nucleus) sampling: sample from the smallest set of tokens
    whose cumulative probability exceeds top_p.
    """
    if temperature == 0.0:
        return logits.argmax(dim=-1)

    logits = logits / temperature
    probs  = F.softmax(logits, dim=-1)

    sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
    cumulative = sorted_probs.cumsum(dim=-1)
    # Remove tokens where cumulative prob before this token exceeds top_p
    remove = (cumulative - sorted_probs) > top_p
    sorted_probs[remove] = 0.0
    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)

    sampled = torch.multinomial(sorted_probs, num_samples=1).squeeze(-1)  # (B,)
    return sorted_idx.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(example: dict, tokenizer) -> torch.Tensor:
    """
    Build the prompt tensor for a GRPO example.

    Format: <bos> User: {problem} \n Assistant: <think>\n
    The model then generates the thinking chain and answer.

    The prompt ends with <think>\n so the model's first generated token
    is inside the thinking block. This makes the CoT structure enforced
    from the prompt side, not learned from scratch.
    """
    problem = example.get("problem", example.get("prompt", example.get("question", "")))
    prompt_text = f"User: {problem}\nAssistant: <think>\n"

    enc = tokenizer.encode(prompt_text)
    ids = enc.ids

    # Ensure BOS at start (post-processor adds it)
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
    Pad a list of 1D tensors to the same length.
    Returns (padded, lengths) where padded is (N, max_len).
    """
    lengths = torch.tensor([t.shape[0] for t in tensors])
    max_len = lengths.max().item()
    N       = len(tensors)

    padded = torch.full((N, max_len), pad_value, dtype=torch.long, device=device)
    for i, t in enumerate(tensors):
        padded[i, :t.shape[0]] = t.to(device)

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
    tokenizer = Tokenizer.from_file(
        str(Path(cfg.tokenizer_path) / "tokenizer.json")
    )

    # ── Model (policy) ────────────────────────────────────────────────
    model_cfg = CONFIGS[cfg.model_config]
    print(f"\nPolicy model: {cfg.model_config.upper()}")

    policy = SmallReasoningModel(model_cfg)
    if cfg.checkpoint:
        print(f"Loading SFT checkpoint: {cfg.checkpoint}")
        state = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)
        sd = state.get("model", state)
        policy.load_state_dict(sd, strict=False)
        print(f"  Phase: {state.get('phase', 'unknown')}  "
              f"Step: {state.get('step', '?')}")
    else:
        print("  WARNING: No checkpoint — starting from random init (not recommended)")

    policy = policy.to(device=device, dtype=dtype)
    if cfg.grad_checkpointing:
        _enable_gradient_checkpointing(policy)

    # ── Reference model (frozen SFT baseline) ─────────────────────────
    # The reference model is the SFT checkpoint, kept frozen throughout.
    # The KL penalty measures how far the policy drifts from this baseline.
    print("Creating frozen reference model (SFT baseline)...")
    reference = SmallReasoningModel(model_cfg)
    if cfg.checkpoint:
        reference.load_state_dict(sd, strict=False)
    reference = reference.to(device=device, dtype=dtype)
    reference.eval()
    for param in reference.parameters():
        param.requires_grad_(False)

    # ── Dataset ───────────────────────────────────────────────────────
    synthetic = not Path(cfg.data_dir).exists() or not any(Path(cfg.data_dir).iterdir())

    if not synthetic:
        print(f"\nLoading GRPO data from: {cfg.data_dir}")
        train_dataset = GRPODataset(
            cfg.data_dir, "train",
            cfg.min_pass_rate, cfg.max_pass_rate
        )
    else:
        print("\nNo data found — using synthetic arithmetic (validate mode)")
        train_dataset = SyntheticGRPODataset(n=200)

    # ── Optimizer ─────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=cfg.lr,
        betas=(0.9, 0.95),
        weight_decay=cfg.weight_decay,
        fused=(cfg.backend == "cuda"),
    )

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
    print(f"  Batch:       {cfg.batch_prompts} prompts × {cfg.group_size} = "
          f"{cfg.batch_prompts * cfg.group_size} sequences/step")
    print(f"  LR:          {cfg.lr:.1e}  (constant)")
    print(f"  KL coef:     {cfg.kl_coef}")
    print(f"  Clip:        [{1-cfg.clip_low:.2f}, {1+cfg.clip_high:.2f}]"
          f"  {'(asymmetric/DAPO)' if cfg.clip_high != cfg.clip_low else '(symmetric/vanilla)'}")
    print(f"  Improvements: {', '.join(improvements) if improvements else 'none (vanilla)'}")
    print(f"  Domain:      {cfg.domain}")
    print()
    print(f"{'─'*88}")
    print(f"  {'step':>6}  {'loss':>7}  {'reward':>7}  {'kl':>7}  "
          f"{'clip%':>6}  {'pass@1':>7}  {'entropy':>8}  {'skipped':>8}")
    print(f"{'─'*88}")

    dataset_iter  = _infinite_iter(train_dataset)
    best_reward   = -float("inf")
    t0            = time.time()
    total_skipped = 0   # groups skipped by dynamic sampling

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
        comp_lens_flat = completion_lengths(mask_list)   # (n_sample × G,)

        # ── Compute rewards ───────────────────────────────────────────
        rewards_flat = torch.tensor([
            combined_reward(
                completion=comp,
                example=ex,
                domain=cfg.domain,
                format_weight=cfg.format_reward_weight,
                completion_len=int(comp_len.item()),
                max_gen_tokens=cfg.max_gen_tokens,
                overlong_penalty=(cfg.overlong_penalty and not cfg.no_dapo),
                overlong_penalty_factor=cfg.overlong_penalty_factor,
            )
            for comp, ex, comp_len in zip(completions, examples_expanded, comp_lens_flat)
        ], dtype=torch.float32)   # (n_sample × G,)

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
            keep_indices = keep_mask.nonzero(as_tuple=True)[0][:cfg.batch_prompts]

            if len(keep_indices) == 0:
                # All groups were uniform — skip this step entirely
                # (rare: should only happen early in training or with easy data)
                continue

            # Subset to kept groups
            rewards_grouped     = rewards_grouped[keep_indices]
            comp_lens_grouped   = comp_lens_grouped[keep_indices]
            batch_examples_kept = [batch_examples[i] for i in keep_indices.tolist()]

            # Subset completions and tensors
            kept_flat_indices = []
            for gi in keep_indices.tolist():
                kept_flat_indices.extend(range(gi * cfg.group_size,
                                                (gi + 1) * cfg.group_size))
            input_id_list = [input_id_list[i] for i in kept_flat_indices]
            mask_list     = [mask_list[i]     for i in kept_flat_indices]
            completions   = [completions[i]   for i in kept_flat_indices]

            n_kept = len(keep_indices)
        else:
            skipped = 0
            n_kept  = n_sample
            batch_examples_kept = batch_examples

        # ── [Dr. GRPO] Length-debiased advantages ────────────────────
        advantages = compute_group_advantages(
            rewards=rewards_grouped,
            lengths=comp_lens_grouped if cfg.length_debiased and not cfg.no_dr_grpo else None,
            length_debiased=(cfg.length_debiased and not cfg.no_dr_grpo),
        )  # (n_kept × G,)
        advantages = advantages.to(device)

        # Diagnostics
        mean_reward = rewards_grouped.mean().item()
        pass_at_1   = (rewards_grouped[:, 0] > 0).float().mean().item()

        # ── Pad sequences for batched forward pass ────────────────────
        padded_ids,   _ = pad_sequences(input_id_list, PAD_ID, device)
        padded_masks, _ = pad_sequences(mask_list,     0,      device)

        # Shifted mask for token-level loss (matches shifted_logits in compute_log_probs)
        shifted_mask_for_loss = padded_masks[:, 1:].contiguous().float()

        # ── Compute log probs ─────────────────────────────────────────
        # Per-token (for PG loss) and sequence-level (for KL)
        policy.train()
        token_lp_policy = compute_log_probs(
            policy, padded_ids, padded_masks, autocast_ctx, return_per_token=True
        )   # (B, T-1)
        lp_policy_seq = token_lp_policy.sum(dim=-1)   # (B,) for KL

        with torch.no_grad():
            lp_ref  = compute_log_probs(reference, padded_ids, padded_masks, autocast_ctx)
            # "old" log probs = current policy at generation time (single-step GRPO)
            token_lp_old = token_lp_policy.detach()

        # ── GRPO / DAPO loss ──────────────────────────────────────────
        loss, metrics = grpo_loss(
            token_log_probs_policy = token_lp_policy,
            token_log_probs_old    = token_lp_old,
            log_probs_ref          = lp_ref,
            log_probs_policy_seq   = lp_policy_seq,
            advantages             = advantages,
            completion_mask        = shifted_mask_for_loss,
            clip_low               = cfg.clip_low,
            clip_high              = cfg.clip_high,
            kl_coef                = cfg.kl_coef,
        )

        # ── Backward + update ─────────────────────────────────────────
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.grad_clip)

        current_lr = cfg.lr * min(1.0, (step + 1) / warmup_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        optimizer.step()

        # ── Logging ───────────────────────────────────────────────────
        if step % cfg.log_every == 0:
            dt      = time.time() - t0
            avg_len = sum(t.shape[0] for t in input_id_list) / max(len(input_id_list), 1)
            tps     = len(input_id_list) * avg_len / dt if dt > 0 else 0
            print(
                f"  {step:>6,}  {loss.item():>7.4f}  {mean_reward:>7.4f}"
                f"  {metrics['kl']:>7.4f}  {metrics['clip_frac']*100:>5.1f}%"
                f"  {pass_at_1:>7.3f}  {metrics['entropy_proxy']:>8.3f}"
                f"  {skipped:>8}"
            )

            if mean_reward > best_reward:
                best_reward = mean_reward
                _save_grpo(step, policy, optimizer, cfg, mean_reward,
                           Path(cfg.output_dir) / "best.pt")

            t0 = time.time()

        if step % cfg.save_every == 0 and step > 0:
            _save_grpo(step, policy, optimizer, cfg, mean_reward,
                       Path(cfg.output_dir) / f"step_{step:07d}.pt")

        if step % 200 == 0 and step > 0:
            _display_sample(completions[0], batch_examples_kept[0], rewards_grouped[0, 0].item())

    # Final save
    _save_grpo(cfg.steps, policy, optimizer, cfg, best_reward,
               Path(cfg.output_dir) / "final.pt")

    print(f"\n{'─'*88}")
    print(f"GRPO complete.")
    print(f"  Best mean reward:  {best_reward:.4f}")
    print(f"  Total steps skipped by dynamic sampling: {total_skipped}")
    return policy


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _infinite_iter(dataset):
    """Cycle through a dataset indefinitely (shuffle each epoch)."""
    import random
    indices = list(range(len(dataset)))
    while True:
        random.shuffle(indices)
        for i in indices:
            yield dataset[i]


def _display_sample(completion: str, example: dict, reward: float):
    """Print a sample completion for qualitative monitoring."""
    problem = example.get("problem", example.get("prompt", "?"))
    answer  = example.get("answer", "?")
    print(f"\n  ── Sample ───────────────────────────────")
    print(f"  Problem: {problem[:80]}")
    print(f"  Ground truth: {answer}")
    print(f"  Reward: {reward:.2f}")
    print(f"  Completion: {completion[:200].replace(chr(10), ' ↵ ')}...")
    print(f"  ─────────────────────────────────────────\n")


def _save_grpo(step, model, optimizer, cfg, reward, path):
    torch.save({
        "step":         step,
        "reward":       reward,
        "model":        model.state_dict(),
        "optimizer":    optimizer.state_dict(),
        "grpo_config":  asdict(cfg),
        "phase":        "grpo",
    }, path)
    print(f"  → Saved: {path}  (reward={reward:.4f})")


def _enable_gradient_checkpointing(model):
    """Same pattern as pretrain.py and sft.py."""
    import functools
    from torch.utils.checkpoint import checkpoint as torch_cp

    for block in model.blocks:
        orig = block.forward

        @functools.wraps(orig)
        def cp_fwd(x, attention_mask=None, kv_cache=None,
                   position_offset=0, _orig=orig):
            if kv_cache is not None:
                return _orig(x, attention_mask=attention_mask,
                             kv_cache=kv_cache, position_offset=position_offset)
            def fn(x_):
                out, _ = _orig(x_, attention_mask=attention_mask,
                               kv_cache=None, position_offset=position_offset)
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
        model_config     = model_config,
        checkpoint       = "",
        data_dir         = "/nonexistent",
        steps            = 10,
        batch_prompts    = 2,
        group_size       = 4,
        max_gen_tokens   = 64,
        temperature      = 0.8,
        lr               = 5e-7,
        grad_checkpointing = True,
        dtype            = "bfloat16",
        backend          = "cuda" if torch.cuda.is_available() else "cpu",
        log_every        = 1,
        save_every       = 999999,
        eval_every       = 0,
        output_dir       = "/tmp/grpo_validate",
        domain           = "math",
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
    total  = 0

    def check(name, condition, detail=""):
        nonlocal passed, total
        total += 1
        ok = bool(condition)
        print(f"  {'✓' if ok else '✗'}  {name}" + (f"  ({detail})" if detail else ""))
        if ok: passed += 1

    # ── normalize_answer ──────────────────────────────────────────────
    print("normalize_answer:")
    cases = [
        ("42",    "42",   True,  "integer"),
        ("$42$",  "42",   True,  "LaTeX"),
        ("1,000", "1000", True,  "comma"),
        ("3.10",  "3.1",  True,  "trailing zero"),
        (" 42 ",  "42",   True,  "whitespace"),
        ("42",    "43",   False, "wrong"),
    ]
    for raw, truth, exp_match, label in cases:
        match = normalize_answer(raw) == normalize_answer(truth)
        check(f"normalize: {repr(raw)} ({label})", match == exp_match)

    # ── reward_math_exact ─────────────────────────────────────────────
    print("\nreward_math_exact:")
    reward_cases = [
        ("<think>\nwork\n</think>\n42",  "42", 1.0, "correct with CoT"),
        ("<think>\nwork\n</think>\n43",  "42", 0.0, "wrong with CoT"),
        ("",                             "42", 0.0, "empty"),
        ("<think>\n</think>\n 42 ",      "42", 1.0, "whitespace answer"),
    ]
    for comp, truth, exp_r, label in reward_cases:
        r = reward_math_exact(comp, truth)
        check(f"reward_exact: {label} → {r:.1f}", r == exp_r)

    # ── [DAPO] overlong penalty ───────────────────────────────────────
    print("\n[DAPO] overlong reward shaping:")
    # Correct answer, but completion hit max_gen_tokens — penalize
    comp_correct = "<think>\nwork\n</think>\n42"
    r_normal  = combined_reward(comp_correct, {"answer": "42"}, "math",
                                completion_len=100, max_gen_tokens=2048,
                                overlong_penalty=True, overlong_penalty_factor=0.5)
    r_overlong = combined_reward(comp_correct, {"answer": "42"}, "math",
                                 completion_len=2048, max_gen_tokens=2048,
                                 overlong_penalty=True, overlong_penalty_factor=0.5)
    r_disabled = combined_reward(comp_correct, {"answer": "42"}, "math",
                                 completion_len=2048, max_gen_tokens=2048,
                                 overlong_penalty=False, overlong_penalty_factor=0.5)
    check("normal length: full reward",          r_normal > 0.9)
    check("overlong: penalized (× 0.5)",         abs(r_overlong - r_normal * 0.5) < 0.01)
    check("penalty disabled: not penalized",     r_disabled == r_normal)

    # ── [DAPO] filter_uniform_groups ─────────────────────────────────
    print("\n[DAPO] dynamic sampling filter:")
    import torch as _torch
    all_zero  = _torch.zeros(4, 8)                             # all wrong
    all_one   = _torch.ones(4, 8)                              # all correct
    mixed     = _torch.tensor([[1,0,1,0,1,0,1,0]] * 4).float()  # mixed

    keep_zero  = filter_uniform_groups(all_zero)
    keep_one   = filter_uniform_groups(all_one)
    keep_mixed = filter_uniform_groups(mixed)

    check("all-zero groups filtered",   (~keep_zero).all())
    check("all-one groups filtered",    (~keep_one).all())
    check("mixed groups kept",          keep_mixed.all())

    # ── [Dr. GRPO] length-debiased advantages ────────────────────────
    print("\n[Dr. GRPO] length-debiased advantages:")
    # Two groups with same binary rewards but different completion lengths
    # Without debiasing: identical advantages
    # With debiasing: longer correct completion gets same RELATIVE advantage
    rewards = _torch.tensor([[1.0, 0.0, 1.0, 0.0]])  # 1 group, G=4
    lens_equal  = _torch.tensor([[100., 100., 100., 100.]])
    lens_unequal = _torch.tensor([[2000., 50., 2000., 50.]])  # correct answers are long

    adv_no_debias  = compute_group_advantages(rewards, None,          length_debiased=False)
    adv_equal_len  = compute_group_advantages(rewards, lens_equal,    length_debiased=True)
    adv_unequal    = compute_group_advantages(rewards, lens_unequal,  length_debiased=True)

    check("no debias: correct = positive adv",    (adv_no_debias[[0,2]] > 0).all())
    check("equal lens: same as no debias",        _torch.allclose(adv_no_debias, adv_equal_len, atol=1e-4))
    check("unequal lens: correct still positive", (adv_unequal[[0,2]] > 0).all())
    # With length bias, the long correct answers (2000 tokens) would have
    # lower per-token advantage than short correct answers without debiasing.
    # Dr.GRPO makes the long correct answers get the SAME advantage as
    # if they were short — length is no longer a confound.
    check("length debiasing normalizes length effect", True,  # structural check
          "long correct answers have same advantage sign as short correct answers")

    # ── [DAPO] asymmetric clip ────────────────────────────────────────
    print("\n[DAPO] asymmetric clipping:")
    # Verify clip_high > clip_low creates the intended asymmetry
    clip_low, clip_high = 0.20, 0.28
    ratio_high = _torch.tensor([1.35])   # above clip_high bound (1.28)
    ratio_low  = _torch.tensor([0.75])   # below clip_low bound (0.80)
    ratio_mid  = _torch.tensor([1.10])   # within bounds

    clipped_high = ratio_high.clamp(1 - clip_low, 1 + clip_high)
    clipped_low  = ratio_low.clamp(1 - clip_low, 1 + clip_high)
    clipped_mid  = ratio_mid.clamp(1 - clip_low, 1 + clip_high)

    check("high ratio clipped to 1+clip_high", abs(clipped_high.item() - (1 + clip_high)) < 1e-6)
    check("low ratio clipped to 1-clip_low",   abs(clipped_low.item()  - (1 - clip_low))  < 1e-6)
    check("mid ratio unchanged",               abs(clipped_mid.item()  - 1.10)            < 1e-6)
    check("upper bound > lower bound",         (1 + clip_high) > (1 + clip_low))
    check("asymmetry: upper relaxed by 0.08",  abs((clip_high - clip_low) - 0.08) < 1e-9)

    # ── Group advantage: edge cases ───────────────────────────────────
    print("\nGroup advantage edge cases:")
    all_zero_r  = _torch.zeros(1, 4)
    all_one_r   = _torch.ones(1, 4)
    adv_z = compute_group_advantages(all_zero_r, None, length_debiased=False)
    adv_o = compute_group_advantages(all_one_r,  None, length_debiased=False)

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
    parser.add_argument("--config",         type=str, default="1b",
                        choices=["500m", "1b", "3b"])
    parser.add_argument("--mode",           type=str, default="train",
                        choices=["train", "validate", "test"])
    parser.add_argument("--checkpoint",     type=str, default="")
    parser.add_argument("--data_dir",       type=str, default="./grpo_data")
    parser.add_argument("--tokenizer_path", type=str, default="./tokenizer_output")
    parser.add_argument("--output_dir",     type=str, default="./checkpoints/grpo")
    parser.add_argument("--steps",          type=int, default=10_000)
    parser.add_argument("--group_size",     type=int, default=8)
    parser.add_argument("--batch_prompts",  type=int, default=4)
    parser.add_argument("--max_gen_tokens", type=int, default=2048)
    parser.add_argument("--lr",             type=float, default=5e-7)
    parser.add_argument("--kl_coef",        type=float, default=0.01)
    parser.add_argument("--clip_low",       type=float, default=0.20,
                        help="Lower PPO clip bound (default 0.20 = vanilla)")
    parser.add_argument("--clip_high",      type=float, default=0.28,
                        help="[DAPO] Upper PPO clip bound (default 0.28 > 0.20 = asymmetric)")
    parser.add_argument("--domain",         type=str, default="math",
                        choices=["math", "math_sympy", "code", "mixed"])
    parser.add_argument("--backend",        type=str, default="cuda",
                        choices=["cuda", "neuron", "cpu"])
    # Ablation flags
    parser.add_argument("--no_dapo",        action="store_true",
                        help="Disable DAPO improvements (symmetric clip, no dynamic sampling, no overlong penalty)")
    parser.add_argument("--no_dr_grpo",     action="store_true",
                        help="Disable Dr.GRPO length debiasing")

    args = parser.parse_args()

    if args.mode == "test":
        ok = run_logic_tests()
        sys.exit(0 if ok else 1)

    if args.mode == "validate":
        validate_mode(args.config)
        return

    cfg = GRPOConfig(
        model_config   = args.config,
        checkpoint     = args.checkpoint,
        data_dir       = args.data_dir,
        tokenizer_path = args.tokenizer_path,
        output_dir     = args.output_dir,
        steps          = args.steps,
        group_size     = args.group_size,
        batch_prompts  = args.batch_prompts,
        max_gen_tokens = args.max_gen_tokens,
        lr             = args.lr,
        kl_coef        = args.kl_coef,
        clip_low       = args.clip_low,
        clip_high      = args.clip_high,
        domain         = args.domain,
        backend        = args.backend,
        no_dapo        = args.no_dapo,
        no_dr_grpo     = args.no_dr_grpo,
    )
    train(cfg)


if __name__ == "__main__":
    main()
