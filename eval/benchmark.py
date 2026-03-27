"""
benchmark.py
============
Run the standard evaluation suite across a checkpoint and write a JSON
results file with per-task accuracy and key diagnostic ratios.

Benchmarks tracked (spec Section 7):
  MATH (Hendrycks)     — primary GRPO target
  GSM8K                — baseline reasoning sanity check
  ARC-Challenge        — science QA, out-of-domain
  HellaSwag            — commonsense regression check
  MMLU (5-shot)        — broad capability regression

Key ratios written to results JSON:
  math_vs_hellaswag    — GRPO should improve MATH without collapsing HellaSwag
  (pass@1 ratios handled separately via --num_samples flag)

Usage:
  uv run srm-benchmark --checkpoint checkpoints/500m_sft/best.pt
  uv run srm-benchmark --checkpoint best.pt --suite quick       # arc+gsm8k only
  uv run srm-benchmark --checkpoint best.pt --suite full        # all tasks
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Benchmark suites
# ---------------------------------------------------------------------------

# "quick" is a fast smoke test; "full" is the complete evaluation suite.
SUITES: dict[str, list[tuple[str, int]]] = {
    "quick": [
        ("arc_challenge", 0),
        ("gsm8k", 5),
    ],
    "standard": [
        ("arc_challenge", 0),
        ("gsm8k", 5),
        ("hellaswag", 0),
        ("mmlu", 5),
    ],
    "full": [
        ("arc_challenge", 0),
        ("gsm8k", 5),
        ("hellaswag", 0),
        ("mmlu", 5),
        ("mathqa", 0),
        ("hendrycks_math", 4),  # available if lm_eval has it installed
    ],
}

# Default suite if --suite is omitted
DEFAULT_SUITE = "standard"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(
    checkpoint: str,
    config: str,
    tokenizer_path: str,
    suite: str,
    batch_size: int,
    limit: int | None,
    device: str | None,
    output_dir: Path,
) -> dict:
    """
    Run the specified benchmark suite and return the results dict.
    """
    # Import here so a missing lm_eval gives a clear error at runtime, not
    # at import time (the eval extras may not be installed).
    try:
        from lm_eval.evaluator import simple_evaluate  # lm_eval 0.4.x API

        # Importing harness registers the "small_reasoning" model class
        import eval.harness  # noqa: F401
    except ImportError as e:
        sys.exit(f"ERROR: {e}\nInstall with: uv sync --extra eval")

    task_specs = SUITES[suite]
    all_results: dict = {"suite": suite, "tasks": {}, "ratios": {}, "meta": {}}

    model_args = (
        f"checkpoint={checkpoint},"
        f"config={config},"
        f"tokenizer_path={tokenizer_path},"
        f"batch_size={batch_size}"
    )
    if device:
        model_args += f",device={device}"

    for task_name, num_fewshot in task_specs:
        print(f"\n{'='*60}")
        print(f"Task: {task_name}  ({num_fewshot}-shot)")
        print("=" * 60)
        try:
            result = simple_evaluate(
                model="small_reasoning",
                model_args=model_args,
                tasks=[task_name],
                num_fewshot=num_fewshot,
                limit=limit,
                log_samples=False,
            )
            task_result = result.get("results", {}).get(task_name, {})
            all_results["tasks"][task_name] = task_result
        except Exception as e:
            print(f"WARNING: task {task_name} failed: {e}")
            all_results["tasks"][task_name] = {"error": str(e)}

    # Compute key diagnostic ratios
    ratios = {}
    math_acc = _get_accuracy(all_results["tasks"], "hendrycks_math")
    hellaswag_acc = _get_accuracy(all_results["tasks"], "hellaswag")
    gsm8k_acc = _get_accuracy(all_results["tasks"], "gsm8k")
    arc_acc = _get_accuracy(all_results["tasks"], "arc_challenge")

    if math_acc is not None and hellaswag_acc is not None and hellaswag_acc > 0:
        ratios["math_vs_hellaswag"] = round(math_acc / hellaswag_acc, 4)

    if gsm8k_acc is not None:
        ratios["gsm8k_acc"] = round(gsm8k_acc, 4)
    if arc_acc is not None:
        ratios["arc_challenge_acc"] = round(arc_acc, 4)

    all_results["ratios"] = ratios

    # Metadata for reproducibility
    ckpt_name = Path(checkpoint).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results["meta"] = {
        "checkpoint": checkpoint,
        "config": config,
        "suite": suite,
        "timestamp": timestamp,
        "limit": limit,
    }

    # Write results file
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{ckpt_name}_{timestamp}.json"
    out_path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nResults written to {out_path}")

    return all_results


def _get_accuracy(tasks: dict, task_name: str) -> float | None:
    """Extract primary accuracy metric from a task result dict."""
    result = tasks.get(task_name)
    if result is None or "error" in result:
        return None
    # lm_eval uses various metric names; try common ones in order
    for key in ("acc,none", "acc_norm,none", "acc", "exact_match,none"):
        if key in result:
            return float(result[key])
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run standard SRM benchmark suite and write results JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=str,
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--config",
        default="500m",
        choices=["500m", "1b", "3b"],
        help="Model config (default: 500m)",
    )
    parser.add_argument(
        "--tokenizer",
        default="tokenizer_output",
        type=str,
        help="Path to tokenizer directory (default: tokenizer_output)",
    )
    parser.add_argument(
        "--suite",
        default=DEFAULT_SUITE,
        choices=list(SUITES.keys()),
        help=f"Benchmark suite to run (default: {DEFAULT_SUITE})",
    )
    parser.add_argument(
        "--batch_size",
        default=8,
        type=int,
        help="Evaluation batch size (default: 8)",
    )
    parser.add_argument(
        "--limit",
        default=None,
        type=int,
        help="Limit to N examples per task (default: all; use 50 for smoke test)",
    )
    parser.add_argument(
        "--device",
        default=None,
        type=str,
        help="Device override (default: auto-detect cuda/mps/cpu)",
    )
    parser.add_argument(
        "--output_dir",
        default="results",
        type=Path,
        help="Directory for results JSON files (default: results/)",
    )
    args = parser.parse_args()

    results = run(
        checkpoint=args.checkpoint,
        config=args.config,
        tokenizer_path=args.tokenizer,
        suite=args.suite,
        batch_size=args.batch_size,
        limit=args.limit,
        device=args.device,
        output_dir=args.output_dir,
    )

    # Print summary
    print("\n=== Summary ===")
    for task, result in results["tasks"].items():
        if "error" in result:
            print(f"  {task}: ERROR — {result['error']}")
        else:
            acc = _get_accuracy(results["tasks"], task)
            print(f"  {task}: {acc:.4f}" if acc is not None else f"  {task}: (no acc metric)")

    if results["ratios"]:
        print("\n=== Ratios ===")
        for k, v in results["ratios"].items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
