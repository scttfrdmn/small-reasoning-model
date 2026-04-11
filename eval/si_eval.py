"""
si_eval.py
==========
Structured Intent evaluation: tests whether the model can generate
valid SI specifications (JSON with function/signature/behavior fields)
given natural language task descriptions.

Uses the same generation pipeline as GRPO training (build_prompt,
generate_completions) so results are directly comparable to math eval.

Usage:
  python eval/si_eval.py \
    --checkpoint checkpoints/500m_grpo/best.pt \
    --config 500m \
    --tokenizer_path tokenizer_output_prod \
    --test_cases eval/si_test_cases.jsonl \
    --output results/si_eval_grpo.json
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent / "model"))
sys.path.insert(0, str(Path(__file__).parent.parent / "training"))
sys.path.insert(0, str(Path(__file__).parent))

from model.architecture import SmallReasoningModel, ModelConfig, CONFIGS
from grpo import build_prompt, generate_completions
from si_rewards import (
    extract_json_from_completion,
    reward_si_format,
    reward_si_fields,
    reward_si_quality,
    reward_si_combined,
)


def load_test_cases(path: str) -> list[dict]:
    """Load SI test cases from JSONL."""
    with open(path) as f:
        cases = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(cases)} SI test cases from {path}")
    return cases


def build_si_prompt(example: dict, tokenizer) -> torch.Tensor:
    """
    Build prompt for SI evaluation.

    Wraps the task description in the same User/Assistant format as GRPO,
    but adds an instruction to output a JSON specification.
    """
    task = example["prompt"]
    # The prompt instructs the model to think through the problem then
    # output a structured JSON spec. This matches the format described
    # in blog post 11.
    si_prompt = (
        f"Generate a structured specification for the following task as JSON "
        f"with fields: function, signature, behavior, constraints, examples.\n\n"
        f"Task: {task}"
    )
    # Reuse build_prompt by wrapping in the expected format
    wrapped = {"prompt": si_prompt}
    return build_prompt(wrapped, tokenizer)


def evaluate(
    model: nn.Module,
    tokenizer,
    test_cases: list[dict],
    group_size: int,
    max_gen_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    """Run SI evaluation and return metrics."""
    results = {
        "per_problem": [],
        "n_problems": len(test_cases),
        "group_size": group_size,
    }

    # Aggregate metrics
    total_format = 0.0
    total_fields = 0.0
    total_quality = 0.0
    total_combined = 0.0
    json_extracted_any = 0  # pass@k: at least one completion had valid JSON
    json_extracted_first = 0  # pass@1: first completion had valid JSON
    fields_complete_any = 0
    fields_complete_first = 0

    category_stats = {}
    difficulty_stats = {}

    for i, case in enumerate(test_cases):
        prompt_ids = build_si_prompt(case, tokenizer)
        completions, _, _ = generate_completions(
            model=model,
            prompt_ids=[prompt_ids],
            group_size=group_size,
            max_new_tokens=max_gen_tokens,
            temperature=temperature,
            top_p=top_p,
            tokenizer=tokenizer,
            device=device,
            dtype=dtype,
        )

        # Score each completion
        comp_results = []
        for comp in completions:
            fmt = reward_si_format(comp)
            fld = reward_si_fields(comp)
            qual = reward_si_quality(comp)
            combined = reward_si_combined(comp)
            parsed = extract_json_from_completion(comp)
            comp_results.append(
                {
                    "format": fmt,
                    "fields": fld,
                    "quality": qual,
                    "combined": combined,
                    "has_json": parsed is not None,
                    "has_all_fields": fld == 1.0,
                    "parsed_function": parsed.get("function", "") if parsed else "",
                }
            )

        # Aggregate per-problem
        formats = [c["format"] for c in comp_results]
        fields = [c["fields"] for c in comp_results]
        qualities = [c["quality"] for c in comp_results]
        combineds = [c["combined"] for c in comp_results]

        best_format = max(formats)
        best_fields = max(fields)
        best_quality = max(qualities)
        best_combined = max(combineds)
        mean_format = sum(formats) / len(formats)
        mean_fields = sum(fields) / len(fields)
        mean_quality = sum(qualities) / len(qualities)
        mean_combined = sum(combineds) / len(combineds)

        any_json = any(c["has_json"] for c in comp_results)
        first_json = comp_results[0]["has_json"]
        any_fields = any(c["has_all_fields"] for c in comp_results)
        first_fields = comp_results[0]["has_all_fields"]

        total_format += mean_format
        total_fields += mean_fields
        total_quality += mean_quality
        total_combined += mean_combined
        json_extracted_any += int(any_json)
        json_extracted_first += int(first_json)
        fields_complete_any += int(any_fields)
        fields_complete_first += int(first_fields)

        # Track by category and difficulty
        cat = case.get("category", "unknown")
        diff = case.get("difficulty", "unknown")
        for group_key, group_dict in [(cat, category_stats), (diff, difficulty_stats)]:
            if group_key not in group_dict:
                group_dict[group_key] = {
                    "n": 0,
                    "format_sum": 0,
                    "fields_sum": 0,
                    "quality_sum": 0,
                    "json_any": 0,
                    "fields_any": 0,
                }
            group_dict[group_key]["n"] += 1
            group_dict[group_key]["format_sum"] += mean_format
            group_dict[group_key]["fields_sum"] += mean_fields
            group_dict[group_key]["quality_sum"] += mean_quality
            group_dict[group_key]["json_any"] += int(any_json)
            group_dict[group_key]["fields_any"] += int(any_fields)

        results["per_problem"].append(
            {
                "prompt": case["prompt"][:80],
                "category": cat,
                "difficulty": diff,
                "expected_function": case.get("expected_function", ""),
                "mean_format": mean_format,
                "mean_fields": mean_fields,
                "mean_quality": mean_quality,
                "mean_combined": mean_combined,
                "best_combined": best_combined,
                "any_json": any_json,
                "any_all_fields": any_fields,
                "first_json": first_json,
                "parsed_functions": [c["parsed_function"] for c in comp_results],
                "sample_completion": completions[0][:300] + "..." if completions else "",
            }
        )

        # Progress
        if (i + 1) % 5 == 0 or i == 0:
            n = i + 1
            print(
                f"  [{n}/{len(test_cases)}] "
                f"format={total_format/n:.3f} "
                f"fields={total_fields/n:.3f} "
                f"quality={total_quality/n:.3f} "
                f"json_any={json_extracted_any/n:.3f} "
                f"fields_any={fields_complete_any/n:.3f}"
            )

    n = len(test_cases)
    results["summary"] = {
        "mean_format": total_format / n,
        "mean_fields": total_fields / n,
        "mean_quality": total_quality / n,
        "mean_combined": total_combined / n,
        "json_pass_at_1": json_extracted_first / n,
        "json_pass_at_k": json_extracted_any / n,
        "fields_pass_at_1": fields_complete_first / n,
        "fields_pass_at_k": fields_complete_any / n,
    }

    # Per-category and per-difficulty breakdowns
    for label, stats_dict in [
        ("per_category", category_stats),
        ("per_difficulty", difficulty_stats),
    ]:
        results[label] = {}
        for key, stats in sorted(stats_dict.items()):
            t = stats["n"]
            results[label][key] = {
                "n": t,
                "mean_format": stats["format_sum"] / t,
                "mean_fields": stats["fields_sum"] / t,
                "mean_quality": stats["quality_sum"] / t,
                "json_pass_at_k": stats["json_any"] / t,
                "fields_pass_at_k": stats["fields_any"] / t,
            }

    return results


def main():
    parser = argparse.ArgumentParser(description="Structured Intent evaluation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizer_path", default="./tokenizer_output_prod")
    parser.add_argument("--test_cases", default="eval/si_test_cases.jsonl")
    parser.add_argument("--group_size", type=int, default=8)
    parser.add_argument("--max_gen_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    # Load tokenizer
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(os.path.join(args.tokenizer_path, "tokenizer.json"))
    print(f"Tokenizer: {args.tokenizer_path} (vocab={tokenizer.get_vocab_size()})")

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    model_cfg = CONFIGS[args.config]
    model = SmallReasoningModel(model_cfg).to(device=device, dtype=dtype)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt
    model.load_state_dict(state, strict=False)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {args.config} from {args.checkpoint} ({n_params:.1f}M params)")

    # Load test cases
    test_cases = load_test_cases(args.test_cases)

    # Run evaluation
    t0 = time.time()
    results = evaluate(
        model=model,
        tokenizer=tokenizer,
        test_cases=test_cases,
        group_size=args.group_size,
        max_gen_tokens=args.max_gen_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=device,
        dtype=dtype,
    )
    elapsed = time.time() - t0

    results["meta"] = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "tokenizer": args.tokenizer_path,
        "test_cases": args.test_cases,
        "temperature": args.temperature,
        "max_gen_tokens": args.max_gen_tokens,
        "group_size": args.group_size,
        "elapsed_seconds": elapsed,
    }

    # Print summary
    s = results["summary"]
    print(f"\n{'='*60}")
    print(f"Structured Intent Evaluation Results")
    print(f"{'='*60}")
    print(f"  Checkpoint:     {args.checkpoint}")
    print(f"  Problems:       {results['n_problems']}")
    print(f"  Group size:     {results['group_size']}")
    print(f"  Time:           {elapsed:.1f}s ({elapsed/len(test_cases):.1f}s/problem)")
    print(f"\n  JSON format (mean):   {s['mean_format']:.4f}")
    print(f"  Field completeness:   {s['mean_fields']:.4f}")
    print(f"  Quality score:        {s['mean_quality']:.4f}")
    print(f"  Combined score:       {s['mean_combined']:.4f}")
    print(f"\n  JSON pass@1:          {s['json_pass_at_1']:.4f}")
    print(f"  JSON pass@{args.group_size}:          {s['json_pass_at_k']:.4f}")
    print(f"  Fields pass@1:        {s['fields_pass_at_1']:.4f}")
    print(f"  Fields pass@{args.group_size}:        {s['fields_pass_at_k']:.4f}")

    print(f"\n  Per-difficulty:")
    for diff, stats in results.get("per_difficulty", {}).items():
        print(
            f"    {diff:8s}  n={stats['n']:2d}  "
            f"format={stats['mean_format']:.3f}  "
            f"fields={stats['mean_fields']:.3f}  "
            f"json@k={stats['json_pass_at_k']:.3f}"
        )

    print(f"\n  Per-category:")
    for cat, stats in results.get("per_category", {}).items():
        print(
            f"    {cat:16s}  n={stats['n']:2d}  "
            f"format={stats['mean_format']:.3f}  "
            f"fields={stats['mean_fields']:.3f}  "
            f"json@k={stats['json_pass_at_k']:.3f}"
        )

    # Save results
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
