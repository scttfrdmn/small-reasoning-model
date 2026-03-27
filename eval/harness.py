"""
harness.py
==========
lm-evaluation-harness integration for SmallReasoningModel.

Wraps SmallReasoningModel in the lm_eval.api.model.LM interface so standard
benchmarks can be run with:

  lm_eval --model small_reasoning \\
          --model_args checkpoint=./checkpoints/500m_sft/best.pt \\
          --tasks mathqa,gsm8k,hellaswag,arc_challenge,mmlu \\
          --num_fewshot 5

Or via the project CLI:

  uv run srm-eval --checkpoint checkpoints/500m_sft/best.pt --tasks gsm8k

Three methods required by lm_eval.api.model.LM:
  loglikelihood        — log P(continuation | context), used for MC benchmarks
  loglikelihood_rolling — perplexity on full text (sliding window)
  generate_until       — free generation with stop strings (GSM8K, MATH)
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

try:
    import lm_eval
    import lm_eval.api.model
    import lm_eval.api.registry
    import lm_eval.api.instance
    from lm_eval.evaluator import simple_evaluate  # lm_eval 0.4.x API
except ImportError:
    sys.exit(
        "ERROR: lm-eval not found.\n"
        "Install with:  pip install lm-eval>=0.4\n"
        "or:            uv sync --extra eval"
    )

from model.architecture import SmallReasoningModel, get_config
from tokenizer.train_tokenizer import load_tokenizer


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------


@lm_eval.api.registry.register_model("small_reasoning")
class SmallReasoningLM(lm_eval.api.model.LM):
    """
    lm_eval adapter for SmallReasoningModel.

    model_args (pass via --model_args key=value):
      checkpoint      Path to .pt checkpoint file
      config          Model config name: 500m | 1b | 3b  (default: 500m)
      tokenizer_path  Path to tokenizer directory         (default: tokenizer_output)
      batch_size      Forward-pass batch size             (default: 8)
      device          torch device string                 (default: cuda / mps / cpu)
      dtype           bfloat16 | float32                  (default: bfloat16)
    """

    def __init__(
        self,
        checkpoint: str,
        config: str = "500m",
        tokenizer_path: str = "tokenizer_output",
        batch_size: int = 8,
        device: Optional[str] = None,
        dtype: str = "bfloat16",
    ) -> None:
        super().__init__()

        self._batch_size = int(batch_size)
        self._dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32

        # Auto-select device: CUDA > MPS > CPU
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self._device = torch.device(device)

        # Load tokenizer
        self._tokenizer = load_tokenizer(tokenizer_path)
        self._vocab_size = self._tokenizer.get_vocab_size()
        # Retrieve special token ids from the tokenizer vocabulary
        self._bos_id: int = self._tokenizer.token_to_id("<bos>") or 1
        self._eos_id: int = self._tokenizer.token_to_id("<eos>") or 2

        # Build model and load weights
        cfg = get_config(config)
        self._model = SmallReasoningModel(cfg)
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        self._model.load_state_dict(state_dict)
        self._model.to(dtype=self._dtype, device=self._device)
        self._model.eval()

        self._max_seq_len: int = cfg.max_seq_len
        print(
            f"[SmallReasoningLM] {config} on {self._device} ({dtype})  "
            f"vocab={self._vocab_size}  max_seq={self._max_seq_len}"
        )

    # ------------------------------------------------------------------
    # lm_eval required interface
    # ------------------------------------------------------------------

    @property
    def eot_token_id(self) -> int:
        return self._eos_id

    @property
    def max_length(self) -> int:
        return self._max_seq_len

    @property
    def max_gen_toks(self) -> int:
        # Upper bound on tokens we'll generate in generate_until.
        return 512

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def device(self) -> torch.device:
        return self._device

    def tok_encode(self, string: str) -> list[int]:
        return self._tokenizer.encode(string).ids

    def tok_decode(self, tokens: list[int]) -> str:
        return self._tokenizer.decode(tokens)

    # ------------------------------------------------------------------
    # Core computation helpers
    # ------------------------------------------------------------------

    def _encode_pair(self, context: str, continuation: str) -> tuple[list[int], int]:
        """
        Tokenize context + continuation and return (full_ids, cont_len).
        cont_len is the number of continuation tokens so we can mask the loss
        to count only those positions.
        """
        ctx_ids = self.tok_encode(context)
        cont_ids = self.tok_encode(continuation)
        full_ids = (ctx_ids + cont_ids)[-(self._max_seq_len) :]
        # How many of the final tokens belong to the continuation?
        cont_len = min(len(cont_ids), len(full_ids))
        return full_ids, cont_len

    @torch.inference_mode()
    def _forward_logprobs(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Run a forward pass and return log-probabilities tensor.

        Args:
            input_ids: (B, T) int64

        Returns:
            log_probs: (B, T, V) float32 — log-softmax over vocabulary
        """
        with torch.amp.autocast(
            device_type=self._device.type if self._device.type != "mps" else "cpu",
            dtype=self._dtype,
            enabled=(self._dtype == torch.bfloat16),
        ):
            # SmallReasoningModel.forward returns (logits, kv_caches) — unpack.
            # For loglikelihood we only need logits; no KV cache is maintained.
            logits, _kv = self._model(input_ids)
        return F.log_softmax(logits.float(), dim=-1)

    # ------------------------------------------------------------------
    # loglikelihood — log P(continuation | context)
    # ------------------------------------------------------------------

    def loglikelihood(
        self, requests: list[lm_eval.api.instance.Instance]
    ) -> list[tuple[float, bool]]:
        """
        Compute log P(continuation | context) for a batch of requests.

        Each request is (context_str, continuation_str).
        Returns (log_prob, is_greedy) for each request.
        is_greedy = True if the argmax at every continuation position matches
        the actual continuation token (i.e. greedy decoding would produce it).
        """
        results: list[tuple[float, bool]] = []

        # Process in batches to avoid OOM
        batch_args = [req.args for req in requests]
        for start in range(0, len(batch_args), self._batch_size):
            chunk = batch_args[start : start + self._batch_size]
            chunk_results = self._loglikelihood_batch(chunk)
            results.extend(chunk_results)

        return results

    def _loglikelihood_batch(
        self, pairs: list[tuple[str, str]]
    ) -> list[tuple[float, bool]]:
        """Process a single mini-batch of (context, continuation) pairs."""
        encoded = [self._encode_pair(ctx, cont) for ctx, cont in pairs]
        max_len = max(len(ids) for ids, _ in encoded)

        # Left-pad to max_len so all sequences have the same length.
        # We pad with eos_id (arbitrary, won't be in the loss mask).
        input_ids = torch.full(
            (len(encoded), max_len), self._eos_id, dtype=torch.long, device=self._device
        )
        for i, (ids, _) in enumerate(encoded):
            input_ids[i, max_len - len(ids) :] = torch.tensor(ids, dtype=torch.long)

        log_probs = self._forward_logprobs(input_ids)  # (B, T, V)

        results: list[tuple[float, bool]] = []
        for i, (ids, cont_len) in enumerate(encoded):
            seq_len = len(ids)
            # Positions of continuation tokens in the (left-padded) sequence.
            # The continuation occupies the last cont_len positions; we look at
            # the logits one step before each target token (causal LM convention).
            offset = max_len - seq_len  # left-padding offset
            # Target token positions: [seq_len-cont_len .. seq_len-1] in orig coords
            # → [max_len-cont_len .. max_len-1] after padding
            target_pos = slice(max_len - cont_len, max_len)
            # Logit positions: one step earlier (the model predicts token t from pos t-1)
            logit_pos = slice(max_len - cont_len - 1, max_len - 1)

            targets = input_ids[i, target_pos]  # (cont_len,)
            lp = log_probs[i, logit_pos, :]  # (cont_len, V)

            # Sum log-probs over continuation tokens
            log_prob_sum = lp[range(cont_len), targets].sum().item()

            # Greedy: argmax at each position == actual token
            greedy_ids = lp.argmax(dim=-1)  # (cont_len,)
            is_greedy = bool((greedy_ids == targets).all().item())

            results.append((log_prob_sum, is_greedy))

        return results

    # ------------------------------------------------------------------
    # loglikelihood_rolling — perplexity on full text
    # ------------------------------------------------------------------

    def loglikelihood_rolling(
        self, requests: list[lm_eval.api.instance.Instance]
    ) -> list[float]:
        """
        Compute total log P(text) over sliding windows for perplexity evaluation.

        Splits long texts into overlapping windows of max_seq_len tokens,
        using the first half of each window as context to avoid edge effects.
        Returns total log-likelihood (sum over all tokens).
        """
        results: list[float] = []

        for req in requests:
            (text,) = req.args
            ids = self.tok_encode(text)
            total_lp = 0.0

            stride = self._max_seq_len // 2  # 50% overlap reduces boundary effects
            for start in range(0, max(1, len(ids) - 1), stride):
                end = min(start + self._max_seq_len, len(ids))
                chunk = ids[start:end]
                if len(chunk) < 2:
                    break

                input_ids = torch.tensor(
                    [chunk], dtype=torch.long, device=self._device
                )
                log_probs = self._forward_logprobs(input_ids)  # (1, T, V)

                # How many tokens in this window to actually score?
                # For the first window, score everything after the first token.
                # For subsequent windows, score only the new (non-overlapping) tokens.
                score_start = 0 if start == 0 else stride
                for t in range(score_start, len(chunk) - 1):
                    total_lp += log_probs[0, t, chunk[t + 1]].item()

            results.append(total_lp)

        return results

    # ------------------------------------------------------------------
    # generate_until — free generation (MATH, GSM8K)
    # ------------------------------------------------------------------

    def generate_until(
        self, requests: list[lm_eval.api.instance.Instance]
    ) -> list[str]:
        """
        Autoregressive generation until stop string or max_tokens.

        Each request is (context_str, gen_kwargs dict).
        gen_kwargs may contain:
          until      list of stop strings (generation halts on first match)
          max_gen_toks  maximum new tokens to generate
        """
        results: list[str] = []

        for req in requests:
            context, gen_kwargs = req.args
            stop_strings: list[str] = gen_kwargs.get("until", [self.tok_decode([self._eos_id])])
            max_new: int = int(gen_kwargs.get("max_gen_toks", self.max_gen_toks))

            generated = self._generate_single(context, stop_strings, max_new)
            results.append(generated)

        return results

    @torch.inference_mode()
    def _generate_single(
        self, context: str, stop_strings: list[str], max_new_tokens: int
    ) -> str:
        """
        Generate a single completion using greedy decoding.

        Uses KV-cache for efficient autoregressive decoding: one prefill pass
        to populate the cache, then one token at a time for the decode phase.
        """
        ctx_ids = self.tok_encode(context)
        # Truncate context to leave room for new tokens
        ctx_ids = ctx_ids[-(self._max_seq_len - max_new_tokens) :]

        input_ids = torch.tensor([ctx_ids], dtype=torch.long, device=self._device)
        generated_ids: list[int] = []

        with torch.amp.autocast(
            device_type=self._device.type if self._device.type != "mps" else "cpu",
            dtype=self._dtype,
            enabled=(self._dtype == torch.bfloat16),
        ):
            # Prefill: process entire context in one shot to build KV cache
            logits, kv_caches = self._model(input_ids)  # (1, T_ctx, V)
            next_logits = logits[0, -1, :]  # (V,)
            position_offset = len(ctx_ids)  # RoPE position for next generated token

            for step_i in range(max_new_tokens):
                # Greedy: pick the most probable token
                next_id = int(next_logits.argmax(dim=-1).item())
                generated_ids.append(next_id)

                if next_id == self._eos_id:
                    break

                # Decode step: extend with single new token.
                # position_offset ensures RoPE assigns the correct absolute position
                # to each generated token rather than always treating it as position 0.
                next_input = torch.tensor(
                    [[next_id]], dtype=torch.long, device=self._device
                )
                logits, kv_caches = self._model(
                    next_input,
                    kv_caches=kv_caches,
                    position_offset=position_offset + step_i,
                )
                next_logits = logits[0, -1, :]

        generated_text = self.tok_decode(generated_ids)

        # Truncate at the first stop string
        for stop in stop_strings:
            idx = generated_text.find(stop)
            if idx != -1:
                generated_text = generated_text[:idx]

        return generated_text


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run lm-evaluation-harness benchmarks on SmallReasoningModel.",
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
        "--tasks",
        default="gsm8k",
        type=str,
        help="Comma-separated list of lm_eval task names",
    )
    parser.add_argument(
        "--num_fewshot",
        default=0,
        type=int,
        help="Number of few-shot examples (default: 0)",
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
        help="Limit evaluation to N examples per task (useful for smoke tests)",
    )
    parser.add_argument(
        "--device",
        default=None,
        type=str,
        help="Device override (default: auto-detect cuda/mps/cpu)",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        type=str,
        help="Write results JSON to this path (optional)",
    )
    args = parser.parse_args()

    # Build model_args string that lm_eval passes to SmallReasoningLM.__init__
    model_args = (
        f"checkpoint={args.checkpoint},"
        f"config={args.config},"
        f"tokenizer_path={args.tokenizer},"
        f"batch_size={args.batch_size}"
    )
    if args.device:
        model_args += f",device={args.device}"

    results = simple_evaluate(
        model="small_reasoning",
        model_args=model_args,
        tasks=args.tasks.split(","),
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        write_out=True,
        log_samples=False,
    )

    # Print summary table
    print("\n=== Results ===")
    for task_name, task_results in results["results"].items():
        metrics = {k: v for k, v in task_results.items() if k != "alias"}
        print(f"  {task_name}:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"    {metric}: {value:.4f}")
            else:
                print(f"    {metric}: {value}")

    # Optionally write JSON
    if args.output_path:
        import json

        out = Path(args.output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2, default=str))
        print(f"\nResults written to {out}")


if __name__ == "__main__":
    main()
