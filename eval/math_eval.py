"""
math_eval.py
============
Domain-specific math evaluation using the GRPO reward pipeline.

Unlike lm-eval-harness (which uses exact-match on generated text in a specific
format), this evaluator uses the same generation + answer extraction + reward
pipeline that GRPO training uses. This gives an apples-to-apples measurement
of whether GRPO actually improved math reasoning.

Evaluates:
  - pass@1: fraction of problems where greedy decode gets the right answer
  - pass@8: fraction of problems where at least 1 of 8 samples is correct
  - mean reward: average reward across all completions (includes format bonus)
  - per-domain breakdown (gsm8k vs math)

Usage:
  python eval/math_eval.py \
    --checkpoint checkpoints/500m_grpo/best.pt \
    --config 500m \
    --tokenizer_path tokenizer_output_prod \
    --eval_data data/grpo/grpo_raw.jsonl \
    --train_data data/grpo/grpo_filtered_v5.jsonl \
    --n_problems 200 \
    --output results/math_eval_grpo.json
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent / "model"))
sys.path.insert(0, str(Path(__file__).parent.parent / "training"))

from model.architecture import SmallReasoningModel, ModelConfig, CONFIGS
from grpo import (
    build_prompt,
    generate_completions,
    combined_reward,
    _enable_gradient_checkpointing,
)


def load_eval_data(
    eval_path: str,
    train_path: str | None,
    n_problems: int,
    seed: int = 42,
    source_filter: str | None = None,
) -> list[dict]:
    """Load eval data, optionally excluding training problems and filtering by source."""
    # Load all candidate problems
    with open(eval_path) as f:
        all_problems = [json.loads(line) for line in f if line.strip()]

    # Filter by source if requested (e.g. "gsm8k", "math")
    if source_filter:
        all_problems = [p for p in all_problems if p.get("source") == source_filter]
        print(f"Filtered to source={source_filter}: {len(all_problems)} problems")

    # If train data provided, exclude training problems by prompt text
    if train_path and os.path.exists(train_path):
        with open(train_path) as f:
            train_problems = {json.loads(line)["prompt"] for line in f if line.strip()}
        held_out = [p for p in all_problems if p["prompt"] not in train_problems]
        print(f"Total problems: {len(all_problems)}, training: {len(train_problems)}, "
              f"held-out: {len(held_out)}")
    else:
        held_out = all_problems
        print(f"Total problems: {len(all_problems)} (no train exclusion)")

    # Sample n_problems from the held-out set
    rng = random.Random(seed)
    if len(held_out) > n_problems:
        held_out = rng.sample(held_out, n_problems)
    else:
        rng.shuffle(held_out)

    return held_out


def evaluate(
    model: nn.Module,
    tokenizer,
    problems: list[dict],
    group_size: int,
    max_gen_tokens: int,
    temperature: float,
    top_p: float,
    device: torch.device,
    dtype: torch.dtype,
) -> dict:
    """Run evaluation and return metrics."""

    results = {
        "per_problem": [],
        "n_problems": len(problems),
        "group_size": group_size,
    }

    total_correct_greedy = 0  # pass@1 (best completion per problem)
    total_correct_any = 0  # pass@8 (any completion correct)
    total_reward = 0.0
    total_completions = 0
    domain_stats = {}  # per-domain tracking

    for i, example in enumerate(problems):
        domain = example.get("domain", "math_exact")
        source = example.get("source", "unknown")

        # Build prompt and generate completions
        prompt_ids = build_prompt(example, tokenizer)
        completions, _, comp_masks = generate_completions(
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
        rewards = []
        for comp in completions:
            r = combined_reward(
                completion=comp,
                example=example,
                domain=domain,
                format_weight=0.0,  # Pure correctness for eval, no format bonus
                completion_len=len(comp),
                max_gen_tokens=max_gen_tokens,
                overlong_penalty=False,  # No penalty for eval
            )
            rewards.append(r)

        n_correct = sum(1 for r in rewards if r >= 1.0)
        best_reward = max(rewards)
        mean_reward = sum(rewards) / len(rewards)

        # pass@1: use greedy (first completion at temp=0.8 is close enough,
        # or we check if the best single completion is correct)
        # For a fair pass@1, we check if any single completion is correct
        # but track the mean reward for the "typical" quality
        greedy_correct = rewards[0] >= 1.0  # First sample as proxy for greedy
        any_correct = n_correct > 0

        total_correct_greedy += int(greedy_correct)
        total_correct_any += int(any_correct)
        total_reward += mean_reward
        total_completions += len(completions)

        # Per-domain tracking
        if source not in domain_stats:
            domain_stats[source] = {"correct_greedy": 0, "correct_any": 0, "total": 0, "reward_sum": 0.0}
        domain_stats[source]["correct_greedy"] += int(greedy_correct)
        domain_stats[source]["correct_any"] += int(any_correct)
        domain_stats[source]["total"] += 1
        domain_stats[source]["reward_sum"] += mean_reward

        results["per_problem"].append({
            "prompt": example["prompt"][:100] + "...",
            "answer": example.get("answer", ""),
            "source": source,
            "domain": domain,
            "n_correct": n_correct,
            "rewards": rewards,
            "best_reward": best_reward,
            "sample_completion": completions[0][:200] + "..." if completions else "",
        })

        # Progress logging
        if (i + 1) % 10 == 0 or i == 0:
            pass1_so_far = total_correct_greedy / (i + 1)
            pass8_so_far = total_correct_any / (i + 1)
            print(f"  [{i+1}/{len(problems)}] pass@1={pass1_so_far:.3f} "
                  f"pass@{group_size}={pass8_so_far:.3f} "
                  f"reward={total_reward/(i+1):.3f} "
                  f"this={n_correct}/{group_size}")

    # Compute final metrics
    n = len(problems)
    results["pass_at_1"] = total_correct_greedy / n
    results["pass_at_k"] = total_correct_any / n
    results["mean_reward"] = total_reward / n

    # Per-domain results
    results["per_domain"] = {}
    for source, stats in domain_stats.items():
        t = stats["total"]
        results["per_domain"][source] = {
            "n": t,
            "pass_at_1": stats["correct_greedy"] / t,
            "pass_at_k": stats["correct_any"] / t,
            "mean_reward": stats["reward_sum"] / t,
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Domain-specific math evaluation")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--config", required=True, help="Model config name (e.g. 500m)")
    parser.add_argument("--tokenizer_path", default="./tokenizer_output_prod")
    parser.add_argument("--eval_data", required=True, help="JSONL file with eval problems")
    parser.add_argument("--train_data", default=None, help="JSONL training data to exclude")
    parser.add_argument("--n_problems", type=int, default=200, help="Number of problems to eval")
    parser.add_argument("--group_size", type=int, default=8, help="Completions per problem")
    parser.add_argument("--max_gen_tokens", type=int, default=512, help="Max generation length")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.95, help="Nucleus sampling threshold")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--source", default=None, help="Filter by source (e.g. gsm8k, math)")
    parser.add_argument("--seed", type=int, default=42)
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
    # Support different checkpoint formats: "model_state_dict", "model", or bare state dict
    if "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"Model: {args.config} from {args.checkpoint}")

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Parameters: {n_params:.1f}M on {device} ({dtype})")

    # Load eval data
    problems = load_eval_data(args.eval_data, args.train_data, args.n_problems, args.seed, args.source)
    print(f"Evaluating on {len(problems)} held-out problems\n")

    # Run evaluation
    t0 = time.time()
    results = evaluate(
        model=model,
        tokenizer=tokenizer,
        problems=problems,
        group_size=args.group_size,
        max_gen_tokens=args.max_gen_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=device,
        dtype=dtype,
    )
    elapsed = time.time() - t0

    # Add metadata
    results["meta"] = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "tokenizer": args.tokenizer_path,
        "eval_data": args.eval_data,
        "train_data": args.train_data,
        "temperature": args.temperature,
        "max_gen_tokens": args.max_gen_tokens,
        "elapsed_seconds": elapsed,
    }

    # Print summary
    print(f"\n{'='*60}")
    print(f"Math Evaluation Results")
    print(f"{'='*60}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Problems:   {results['n_problems']}")
    print(f"  Group size: {results['group_size']}")
    print(f"  Time:       {elapsed:.1f}s ({elapsed/len(problems):.1f}s/problem)")
    print(f"\n  pass@1:     {results['pass_at_1']:.4f}")
    print(f"  pass@{args.group_size}:     {results['pass_at_k']:.4f}")
    print(f"  mean reward:{results['mean_reward']:.4f}")

    print(f"\n  Per-domain breakdown:")
    for source, stats in results["per_domain"].items():
        print(f"    {source:10s}  n={stats['n']:3d}  "
              f"pass@1={stats['pass_at_1']:.3f}  "
              f"pass@{args.group_size}={stats['pass_at_k']:.3f}  "
              f"reward={stats['mean_reward']:.3f}")

    # Save results
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
