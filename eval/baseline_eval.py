"""
baseline_eval.py
================
Run math and SI evaluations against peer HuggingFace small models so we
can compare apples-to-apples against our own SRM checkpoints.

Why this exists
---------------
Our `math_eval.py` and `si_eval.py` are tightly coupled to SmallReasoningModel
(custom architecture, custom prompt format, custom generation loop). To measure
whether SRM's pass@1 of 4% is good or bad, we need the same metrics computed
on the same problems with other 0.5B-1.5B models.

This file is a HuggingFace adapter that reuses the *reward functions* but
replaces the generation pipeline with transformers.AutoModelForCausalLM.

Measurements are strictly comparable only where the reward function is
domain-pure: correctness on math (answer extraction + normalization)
and structural validity on SI (JSON + required fields). Format rewards
that depend on our `<think>` token are zeroed out so peer models aren't
penalized for not using our specific structure.

Usage
-----
    python eval/baseline_eval.py \
        --model Qwen/Qwen2.5-0.5B-Instruct \
        --suite math \
        --n_problems 100 \
        --output results/baseline_qwen25_0.5b_math.json

    python eval/baseline_eval.py \
        --model microsoft/phi-1_5 \
        --suite si \
        --output results/baseline_phi_1.5_si.json

Both suites in one run:
    python eval/baseline_eval.py \
        --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --suite both \
        --output_dir results/baseline_tinyllama
"""

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "training"))
sys.path.insert(0, str(_REPO_ROOT / "eval"))

# Reuse SRM reward functions — the scoring must be identical to make
# the comparison meaningful.
from grpo import (
    combined_reward as math_combined_reward,
    normalize_answer,
    _extract_final_answer,
)
from si_rewards import (
    extract_json_from_completion,
    reward_si_format,
    reward_si_fields,
    reward_si_quality,
    reward_si_combined,
)

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def build_math_prompt(example: dict, tokenizer, model_name: str) -> str:
    """Build a math prompt using the peer model's chat template.

    We ask the model to solve the problem step by step and put the final
    answer after the chain-of-thought. No <think> tags — peer models
    weren't trained with them. We rely on `_extract_final_answer` which
    falls back to finding \\boxed{} or trailing numbers.
    """
    problem = example.get("prompt", example.get("problem", example.get("question", "")))
    user_msg = (
        f"Solve the following problem step by step. "
        f"Put the final answer in \\boxed{{}}.\n\n"
        f"Problem: {problem}"
    )

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [{"role": "user", "content": user_msg}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # Fallback for models without chat templates
    return f"User: {user_msg}\nAssistant:"


def build_si_prompt(example: dict, tokenizer, model_name: str) -> str:
    """Build an SI prompt using the peer model's chat template."""
    task = example["prompt"]
    user_msg = (
        f"Generate a structured specification for the following task as JSON "
        f"with fields: function, signature, behavior, constraints, examples.\n\n"
        f"Task: {task}\n\n"
        f"Respond with only the JSON object, no surrounding prose."
    )

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [{"role": "user", "content": user_msg}]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"User: {user_msg}\nAssistant:"


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


@torch.inference_mode()
def generate_samples(
    model,
    tokenizer,
    prompt_text: str,
    n_samples: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> list[str]:
    """Generate n_samples completions from a HuggingFace model.

    We use model.generate with num_return_sequences to get all samples in
    one forward pass where possible. Some models don't support it well,
    so we fall back to a loop.
    """
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    try:
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=n_samples,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        # outputs shape: (n_samples, prompt_len + gen_len)
        gen_tokens = outputs[:, prompt_len:]
        return [tokenizer.decode(ids, skip_special_tokens=True) for ids in gen_tokens]
    except Exception:
        # Fallback: generate one at a time
        completions = []
        for _ in range(n_samples):
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
            completions.append(tokenizer.decode(out[0, prompt_len:], skip_special_tokens=True))
        return completions


# ---------------------------------------------------------------------------
# Math eval (peer-model version)
# ---------------------------------------------------------------------------


def run_math_eval(
    model,
    tokenizer,
    model_name: str,
    problems: list[dict],
    group_size: int,
    max_gen_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> dict:
    """Same metrics as math_eval.py but with HF-model generation."""
    results = {"per_problem": [], "n_problems": len(problems), "group_size": group_size}
    total_correct_greedy = total_correct_any = total_correct_voted = 0
    total_reward = 0.0
    domain_stats: dict = {}

    for i, example in enumerate(problems):
        domain = example.get("domain", "math_exact")
        source = example.get("source", "unknown")

        prompt_text = build_math_prompt(example, tokenizer, model_name)
        completions = generate_samples(
            model,
            tokenizer,
            prompt_text,
            group_size,
            max_gen_tokens,
            temperature,
            top_p,
            device,
        )

        # Score each with the same reward function SRM uses.
        # format_weight=0 so peer models aren't penalized for not using <think>.
        rewards = [
            math_combined_reward(
                completion=c,
                example=example,
                domain=domain,
                format_weight=0.0,
                completion_len=len(c),
                max_gen_tokens=max_gen_tokens,
                overlong_penalty=False,
            )
            for c in completions
        ]
        n_correct = sum(1 for r in rewards if r >= 1.0)
        mean_reward = sum(rewards) / len(rewards)
        greedy_correct = rewards[0] >= 1.0
        any_correct = n_correct > 0

        # Majority vote
        ground_truth = example.get("answer", example.get("expected_output", ""))
        answers = [_extract_final_answer(c) for c in completions]
        normalized = [normalize_answer(a) if a else "" for a in answers]
        non_empty = [a for a in normalized if a]
        voted_correct = False
        if non_empty:
            voted_answer = Counter(non_empty).most_common(1)[0][0]
            voted_correct = voted_answer == normalize_answer(ground_truth)

        total_correct_greedy += int(greedy_correct)
        total_correct_any += int(any_correct)
        total_correct_voted += int(voted_correct)
        total_reward += mean_reward

        if source not in domain_stats:
            domain_stats[source] = {
                "correct_greedy": 0,
                "correct_any": 0,
                "correct_voted": 0,
                "total": 0,
                "reward_sum": 0.0,
            }
        domain_stats[source]["correct_greedy"] += int(greedy_correct)
        domain_stats[source]["correct_any"] += int(any_correct)
        domain_stats[source]["correct_voted"] += int(voted_correct)
        domain_stats[source]["total"] += 1
        domain_stats[source]["reward_sum"] += mean_reward

        results["per_problem"].append(
            {
                "prompt": example.get("prompt", "")[:100],
                "answer": ground_truth,
                "source": source,
                "n_correct": n_correct,
                "rewards": rewards,
                "sample_completion": completions[0][:200],
            }
        )

        if (i + 1) % 10 == 0 or i == 0:
            n = i + 1
            print(
                f"  [{n}/{len(problems)}] pass@1={total_correct_greedy/n:.3f} "
                f"pass@1_voted={total_correct_voted/n:.3f} "
                f"pass@{group_size}={total_correct_any/n:.3f} "
                f"reward={total_reward/n:.3f} this={n_correct}/{group_size}",
                flush=True,
            )

    n = len(problems)
    results["pass_at_1"] = total_correct_greedy / n
    results["pass_at_1_voted"] = total_correct_voted / n
    results["pass_at_k"] = total_correct_any / n
    results["mean_reward"] = total_reward / n
    results["per_domain"] = {
        src: {
            "n": s["total"],
            "pass_at_1": s["correct_greedy"] / s["total"],
            "pass_at_1_voted": s["correct_voted"] / s["total"],
            "pass_at_k": s["correct_any"] / s["total"],
            "mean_reward": s["reward_sum"] / s["total"],
        }
        for src, s in domain_stats.items()
    }
    return results


# ---------------------------------------------------------------------------
# SI eval (peer-model version)
# ---------------------------------------------------------------------------


def run_si_eval(
    model,
    tokenizer,
    model_name: str,
    test_cases: list[dict],
    group_size: int,
    max_gen_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
) -> dict:
    results = {"per_problem": [], "n_problems": len(test_cases), "group_size": group_size}
    total_format = total_fields = total_quality = total_combined = 0.0
    json_any = json_first = fields_any = fields_first = 0
    category_stats: dict = {}
    difficulty_stats: dict = {}

    for i, case in enumerate(test_cases):
        prompt_text = build_si_prompt(case, tokenizer, model_name)
        completions = generate_samples(
            model,
            tokenizer,
            prompt_text,
            group_size,
            max_gen_tokens,
            temperature,
            top_p,
            device,
        )

        comp_results = [
            {
                "format": reward_si_format(c),
                "fields": reward_si_fields(c),
                "quality": reward_si_quality(c),
                "combined": reward_si_combined(c),
                "has_json": extract_json_from_completion(c) is not None,
                "has_all_fields": reward_si_fields(c) == 1.0,
            }
            for c in completions
        ]
        mean_format = sum(r["format"] for r in comp_results) / len(comp_results)
        mean_fields = sum(r["fields"] for r in comp_results) / len(comp_results)
        mean_quality = sum(r["quality"] for r in comp_results) / len(comp_results)
        mean_combined = sum(r["combined"] for r in comp_results) / len(comp_results)
        any_json = any(r["has_json"] for r in comp_results)
        first_json = comp_results[0]["has_json"]
        any_fields = any(r["has_all_fields"] for r in comp_results)
        first_fields = comp_results[0]["has_all_fields"]

        total_format += mean_format
        total_fields += mean_fields
        total_quality += mean_quality
        total_combined += mean_combined
        json_any += int(any_json)
        json_first += int(first_json)
        fields_any += int(any_fields)
        fields_first += int(first_fields)

        cat = case.get("category", "unknown")
        diff = case.get("difficulty", "unknown")
        for key, d in [(cat, category_stats), (diff, difficulty_stats)]:
            if key not in d:
                d[key] = {
                    "n": 0,
                    "format_sum": 0,
                    "fields_sum": 0,
                    "quality_sum": 0,
                    "json_any": 0,
                    "fields_any": 0,
                }
            d[key]["n"] += 1
            d[key]["format_sum"] += mean_format
            d[key]["fields_sum"] += mean_fields
            d[key]["quality_sum"] += mean_quality
            d[key]["json_any"] += int(any_json)
            d[key]["fields_any"] += int(any_fields)

        results["per_problem"].append(
            {
                "prompt": case["prompt"][:80],
                "category": cat,
                "difficulty": diff,
                "mean_format": mean_format,
                "mean_fields": mean_fields,
                "any_json": any_json,
                "any_all_fields": any_fields,
                "sample_completion": completions[0][:300],
            }
        )

        if (i + 1) % 5 == 0 or i == 0:
            n = i + 1
            print(
                f"  [{n}/{len(test_cases)}] format={total_format/n:.3f} "
                f"fields={total_fields/n:.3f} json_any={json_any/n:.3f} "
                f"fields_any={fields_any/n:.3f}",
                flush=True,
            )

    n = len(test_cases)
    results["summary"] = {
        "mean_format": total_format / n,
        "mean_fields": total_fields / n,
        "mean_quality": total_quality / n,
        "mean_combined": total_combined / n,
        "json_pass_at_1": json_first / n,
        "json_pass_at_k": json_any / n,
        "fields_pass_at_1": fields_first / n,
        "fields_pass_at_k": fields_any / n,
    }
    for label, stats_dict in [
        ("per_category", category_stats),
        ("per_difficulty", difficulty_stats),
    ]:
        results[label] = {
            k: {
                "n": s["n"],
                "mean_format": s["format_sum"] / s["n"],
                "mean_fields": s["fields_sum"] / s["n"],
                "mean_quality": s["quality_sum"] / s["n"],
                "json_pass_at_k": s["json_any"] / s["n"],
                "fields_pass_at_k": s["fields_any"] / s["n"],
            }
            for k, s in sorted(stats_dict.items())
        }
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def load_math_problems(path: str, n: int, seed: int) -> list[dict]:
    with open(path) as f:
        all_p = [json.loads(l) for l in f if l.strip()]
    rng = random.Random(seed)
    if len(all_p) > n:
        all_p = rng.sample(all_p, n)
    return all_p


def load_si_cases(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    parser = argparse.ArgumentParser(description="Baseline comparison against HF models")
    parser.add_argument(
        "--model", required=True, help="HF model ID (e.g., Qwen/Qwen2.5-0.5B-Instruct)"
    )
    parser.add_argument("--suite", choices=["math", "si", "both"], default="both")
    parser.add_argument("--math_data", default="data/grpo/grpo_filtered_v5.jsonl")
    parser.add_argument("--si_data", default="eval/si_test_cases.jsonl")
    parser.add_argument("--n_problems", type=int, default=100, help="Math problems to eval")
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--max_gen_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map[args.dtype]

    print(f"Loading {args.model} ...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Loaded in {time.time()-t0:.1f}s: {n_params:.1f}M params on {device} ({dtype})")

    os.makedirs(args.output_dir, exist_ok=True)
    model_slug = args.model.replace("/", "_")

    all_results = {"model": args.model, "n_params_M": n_params, "dtype": args.dtype}

    if args.suite in ("math", "both"):
        print(f"\n=== MATH EVAL ({args.n_problems} problems) ===")
        problems = load_math_problems(args.math_data, args.n_problems, args.seed)
        t0 = time.time()
        math_results = run_math_eval(
            model,
            tokenizer,
            args.model,
            problems,
            args.group_size,
            args.max_gen_tokens,
            args.temperature,
            args.top_p,
            device,
        )
        math_results["elapsed_seconds"] = time.time() - t0
        all_results["math"] = math_results
        print(f"\n  pass@1:        {math_results['pass_at_1']:.4f}")
        print(f"  pass@1 voted:  {math_results['pass_at_1_voted']:.4f}")
        print(f"  pass@{args.group_size}:        {math_results['pass_at_k']:.4f}")
        print(f"  mean reward:   {math_results['mean_reward']:.4f}")

    if args.suite in ("si", "both"):
        print(f"\n=== SI EVAL ===")
        cases = load_si_cases(args.si_data)
        t0 = time.time()
        si_results = run_si_eval(
            model,
            tokenizer,
            args.model,
            cases,
            args.group_size,
            args.max_gen_tokens,
            args.temperature,
            args.top_p,
            device,
        )
        si_results["elapsed_seconds"] = time.time() - t0
        all_results["si"] = si_results
        s = si_results["summary"]
        print(f"\n  mean format:   {s['mean_format']:.4f}")
        print(f"  mean fields:   {s['mean_fields']:.4f}")
        print(f"  JSON pass@1:   {s['json_pass_at_1']:.4f}")
        print(f"  JSON pass@{args.group_size}:   {s['json_pass_at_k']:.4f}")
        print(f"  Fields pass@1: {s['fields_pass_at_1']:.4f}")
        print(f"  Fields pass@{args.group_size}: {s['fields_pass_at_k']:.4f}")

    output_path = os.path.join(args.output_dir, f"baseline_{model_slug}.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
