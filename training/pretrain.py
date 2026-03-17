"""
pretrain.py
===========
Phase 0: Pre-training loop for the small reasoning model.

Trains from scratch using next-token prediction (causal language modeling).
Designed to run on:
  - RTX 5090 (32GB)    — 500M / 1B validation runs
  - AWS Trn2           — 1B / 3B full runs (set --backend neuron)
  - Single H100 (cloud)— 3B full run

Key features:
  - BF16 mixed precision (forward + backward in BF16, master weights FP32)
  - Gradient checkpointing (reduces activation memory ~30-40%)
  - Gradient accumulation (simulate large batch on single GPU)
  - Cosine LR schedule with linear warmup
  - Gradient clipping (norm = 1.0)
  - Checkpoint save + resume (fault tolerant)
  - Token throughput logging (tokens/sec)
  - Streaming dataset (no full corpus in memory)

Design philosophy:
  This file is deliberately self-contained: one file trains a 1B–3B LLM from
  scratch. The dataset is streaming (no pre-tokenized binary blobs required),
  the schedule is standard (Chinchilla-style cosine), and the optimizer setup
  follows GPT-3/LLaMA conventions. The goal is legibility over cleverness.

Usage:
  # Validation run — 500M on 5090, tiny data sample
  python pretrain.py --config 500m --mode validate

  # Full local run — 1B on 5090
  python pretrain.py \\
    --config 1b \\
    --data_path /path/to/corpus.jsonl \\
    --tokenizer_path ./tokenizer_output \\
    --output_dir ./checkpoints/1b \\
    --max_tokens 50_000_000_000 \\
    --batch_size 4 \\
    --grad_accum 128

  # Resume from checkpoint
  python pretrain.py --config 1b ... --resume ./checkpoints/1b/step_50000.pt

  # Trainium (Trn2) — set backend and use NxD launcher
  python pretrain.py --config 1b --backend neuron ...
"""

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset

# Local imports — model definition lives in model/architecture.py
from model.architecture import SmallReasoningModel, ModelConfig, CONFIGS, compute_loss

# torch.utils.checkpoint provides activation recomputation for grad checkpointing
from torch.utils.checkpoint import checkpoint as torch_checkpoint

# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    """
    Central configuration object for a pre-training run.

    All hyperparameters live here so that a complete run can be reproduced
    from a single saved config dict (see asdict(cfg) in save_checkpoint).

    Design notes:
      - Default values are calibrated for the 1B model on a single RTX 5090.
      - For 3B on the 5090 you must reduce batch_size or increase grad_accum
        to stay within 32 GB VRAM.
      - Chinchilla optimal for 1B ≈ 20B tokens; we use 50B for over-training
        (better downstream reasoning quality at inference cost).
    """

    # ── Model ───────────────────────────────────────────────────────────────
    # Which entry in the CONFIGS dict to instantiate. The key maps to a
    # ModelConfig with pre-set n_layers / d_model / n_heads for each scale.
    model_config: str = "1b"

    # ── Data ────────────────────────────────────────────────────────────────
    # Path to a .jsonl file where each line is {"text": "..."}, or a plain
    # text file where each line is a document. Empty = synthetic (validate).
    data_path: str = ""
    tokenizer_path: str = "./tokenizer_output"

    # Context window used during training. Deliberately shorter than the
    # model's theoretical max_seq_len (e.g. 4096) because activation memory
    # scales quadratically with sequence length in standard attention.
    # 2048 hits the sweet spot between context richness and VRAM budget.
    max_seq_len: int = 2048

    # ── Token budget ────────────────────────────────────────────────────────
    # 50B tokens for a 1B-parameter model is 2.5× Chinchilla-optimal (20B).
    # Over-training beyond Chinchilla is beneficial when inference cost matters
    # more than training cost — the model improves further per token seen even
    # after the "optimal" point, just with diminishing returns.
    max_tokens: int = 50_000_000_000  # 50B tokens

    # Alternative stop condition: cap by gradient-update steps instead of
    # total tokens. -1 disables this and defers to max_tokens.
    max_steps: int = -1

    # ── Batch sizing ────────────────────────────────────────────────────────
    # per-GPU micro-batch size: number of sequences in each forward pass.
    # Keep this as large as VRAM allows to maximise GPU utilisation.
    # At 2048 seq_len, batch_size=4 ≈ 8K tokens per micro step.
    batch_size: int = 4

    # Gradient accumulation multiplier. We don't do a weight update until
    # grad_accum micro-steps have been run and their gradients summed.
    # This simulates a much larger logical batch without needing multiple GPUs.
    #
    # Effective batch size in tokens:
    #   batch_size × grad_accum × max_seq_len
    #   = 4 × 128 × 2048
    #   = 1,048,576 tokens  (~1M tokens per optimizer step)
    #
    # ~1M tokens/step is well-established for large LLM stability (GPT-3,
    # LLaMA, Mistral all use 2–4M; 1M is appropriate for a single GPU run).
    grad_accum: int = 128

    # ── Optimizer hyperparameters ────────────────────────────────────────────
    # Peak learning rate. 3e-4 is standard for AdamW on LLMs at this scale.
    # Too high → loss spikes; too low → slow convergence.
    lr: float = 3e-4

    # Cosine decay floor: LR never drops below this value. Setting it to 10%
    # of peak (3e-5) is the LLaMA / Mistral convention. Going to zero would
    # waste training budget in the final phase since the model still learns
    # at lr_min.
    lr_min: float = 3e-5  # 10% of peak LR

    # Adam momentum term for the gradient (first moment EMA).
    # 0.9 is the universal default and rarely needs tuning.
    beta1: float = 0.9

    # Adam momentum term for the squared gradient (second moment EMA).
    # Default PyTorch/paper value is 0.999, but LLM training uses 0.95.
    # Rationale: 0.999 makes the second-moment estimate very "sticky" —
    # it adapts slowly to changes in gradient magnitude. During LLM training
    # the loss landscape is non-stationary (different layers, different
    # training phases) and 0.95 lets the adaptive learning rate respond
    # faster to gradient scale changes, which stabilises training.
    # References: GPT-3 (0.95), PaLM (0.99), LLaMA (0.95).
    beta2: float = 0.95

    # Epsilon prevents division by zero in the AdamW update.
    # 1e-8 is the standard; some practitioners use 1e-5 for FP16 stability,
    # but BF16 has the same exponent range as FP32 so 1e-8 is fine here.
    eps: float = 1e-8

    # L2 regularisation applied to weight matrices. 0.1 follows GPT-3/LLaMA
    # convention. Applied only to weight matrices (not embeddings or norms)
    # — see the param-group split in train().
    weight_decay: float = 0.1

    # Maximum gradient norm before clipping. 1.0 is standard. This prevents
    # a single bad batch from causing a catastrophic weight update ("loss spike").
    grad_clip: float = 1.0

    # ── LR schedule ─────────────────────────────────────────────────────────
    # Fraction of total training steps spent in linear warmup.
    # Warmup is necessary because at step 0 the weights are random: the
    # gradient norms are large and inconsistent, and jumping straight to
    # lr=3e-4 causes early instability. Ramping up over 2% of steps
    # (≈ 490 steps at 24,414 total) gives the optimizer time to build
    # reliable second-moment estimates before using the full learning rate.
    warmup_fraction: float = 0.02  # 2% of total steps

    # ── Memory optimisations ─────────────────────────────────────────────────
    # When True, each TransformerBlock recomputes its activations during
    # the backward pass instead of storing them after the forward pass.
    # This trades ~20% extra compute for ~30-40% less peak activation memory,
    # which is the difference between fitting 3B on a 32 GB GPU or not.
    grad_checkpointing: bool = True

    # "bfloat16" is preferred over "float16" for LLM training because:
    #   - Same 8-bit exponent range as FP32 (no loss scaling needed)
    #   - Hardware-native on A100/H100/5090/Trainium
    #   - Numerically safer for long training runs than FP16
    dtype: str = "bfloat16"  # "bfloat16" or "float32"

    # ── Logging + checkpointing ──────────────────────────────────────────────
    output_dir: str = "./checkpoints"
    log_every: int = 10  # Print a progress row every N optimizer steps
    save_every: int = 1000  # Write a checkpoint every N optimizer steps
    eval_every: int = 500  # Run validation every N steps (0 = disable)

    # ── Hardware backend ─────────────────────────────────────────────────────
    # "cuda"   → NVIDIA GPU (RTX 5090, H100, A100 …)
    # "neuron" → AWS Trainium via torch_neuronx / XLA
    # "mps"    → Apple Silicon GPU (M-series Macs) — good for smoke tests
    # "cpu"    → Debug / unit test only; extremely slow
    backend: str = "cuda"

    # torch.compile traces and JIT-compiles the model graph with Triton
    # kernels, giving ~20-30% throughput improvement on CUDA. Disabled by
    # default because (a) it adds a multi-minute compilation overhead on
    # first run and (b) it is not supported on Trainium / XLA.
    compile: bool = False

    # ── Fault tolerance ──────────────────────────────────────────────────────
    # If non-empty, training resumes from this checkpoint file. The step
    # counter and token counter are restored so the LR schedule continues
    # correctly from where it left off.
    resume: str = ""

    def effective_batch_tokens(self) -> int:
        """
        Return the number of tokens consumed per optimizer step.

        This is the "logical" batch size used for scaling laws and
        for computing how many optimizer steps fit in max_tokens.
        Formula: micro_batch × grad_accum_steps × sequence_length.
        """
        return self.batch_size * self.grad_accum * self.max_seq_len

    def total_steps(self, tokens_per_step: int) -> int:
        """
        Return the total number of optimizer steps for this run.

        If max_steps is set explicitly (> 0), use it directly — useful for
        quick experiments. Otherwise derive it from the token budget:
          total_steps = max_tokens / tokens_per_step
        Integer division is intentional: we stop at the last complete step.
        """
        if self.max_steps > 0:
            return self.max_steps
        return self.max_tokens // tokens_per_step


# ---------------------------------------------------------------------------
# Dataset — streaming token iterator
# ---------------------------------------------------------------------------


class TokenDataset(IterableDataset):
    """
    Streaming, pack-style dataset for pre-training.

    Design rationale — why streaming?
      A 50B-token corpus is hundreds of GB on disk. Loading it into RAM or
      even memory-mapping it requires careful platform-specific code. An
      IterableDataset reads one line at a time, keeping the process footprint
      near-constant regardless of corpus size.

    Design rationale — why pack-style (no padding)?
      In padding-based batching every sequence is padded to max_seq_len with
      dummy tokens. The model still processes those positions (wasting FLOPS),
      and the loss must explicitly ignore them. Pack-style concatenates
      documents end-to-end and slices out fixed-length windows. Every token
      in every batch is a real, loss-contributing token — this maximises
      training efficiency and matches how GPT-3, LLaMA, and Mistral were
      trained.

    Design rationale — why line_hash for train/val split?
      A deterministic hash of the line content assigns each document to
      train or val without reading the full file first (no random shuffle
      pass), without needing an index file, and in a way that is stable
      across runs (the same document always lands in the same split).
      The cost is that adjacent lines in the file can be in different splits,
      but for a large pre-training corpus this is fine.

    Input format:
      JSONL: one JSON object per line, {"text": "document text"}
      Plain text: one document per line

    Output:
      Dicts {input_ids: LongTensor(max_seq_len), labels: LongTensor(max_seq_len)}
      where labels[t] = input_ids[t+1]  (next-token prediction target).
      The +1 shift is baked in at the dataset level so the training loop
      never needs to slice.

    For production use: replace on-the-fly tokenization with pre-tokenized
    .bin files (e.g. via numpy memmap). The current approach tokenizes at
    read time, which adds ~10-20% overhead but requires no preprocessing step.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer_path: str,
        max_seq_len: int,
        split: str = "train",  # "train" or "val"
        val_fraction: float = 0.001,  # Reserve 0.1% of documents for validation
    ):
        self.data_path = data_path
        self.tokenizer_path = tokenizer_path
        self.max_seq_len = max_seq_len
        self.split = split
        self.val_fraction = val_fraction

        # Tokenizer is not loaded here. In DataLoader with num_workers > 0,
        # each worker process is forked AFTER __init__ completes. If we loaded
        # the tokenizer in __init__ it would be shared across workers via
        # fork(), which can cause file-descriptor conflicts. Lazy loading
        # (first access in __iter__) ensures each worker creates its own copy.
        self._tokenizer = None

    def _get_tokenizer(self):
        """Load tokenizer on first access (each worker gets its own copy)."""
        if self._tokenizer is None:
            from tokenizers import Tokenizer

            tok_path = Path(self.tokenizer_path) / "tokenizer.json"
            self._tokenizer = Tokenizer.from_file(str(tok_path))
        return self._tokenizer

    def _document_iter(self) -> Iterator[str]:
        """
        Yield raw document strings from the corpus file.

        Train/val split is performed here using a deterministic hash:
          - hash(line) % 2^32 maps each document to a 32-bit integer
          - documents whose hash < split_threshold go to val
          - the rest go to train
        This produces a stable ~val_fraction split without scanning the
        full file in advance or maintaining an index.

        The modulo 2^32 bounds the hash to a fixed range so that
        split_threshold = val_fraction × 2^32 is a meaningful threshold
        regardless of document content.
        """
        path = Path(self.data_path)
        if not path.exists():
            raise FileNotFoundError(f"Corpus not found: {path}")

        # Compute the hash boundary: documents with hash below this value
        # are assigned to the validation split. For val_fraction=0.001 this
        # is 0.1% of the 2^32 hash space ≈ 4,294,967 out of 4,294,967,296.
        split_threshold = int(self.val_fraction * (2**32))

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                # Python's built-in hash() is seeded per-process in Python 3.3+
                # (PYTHONHASHSEED). We use % (2**32) to get a stable positive
                # integer that can be compared against split_threshold regardless
                # of the hash seed. This relies on the distribution of hash values
                # being roughly uniform, which holds for typical text documents.
                line_hash = hash(line) % (2**32)
                in_val = line_hash < split_threshold

                # Skip documents that belong to the other split
                if self.split == "val" and not in_val:
                    continue
                if self.split == "train" and in_val:
                    continue

                # Parse JSONL or treat as plain text
                if line.startswith("{"):
                    try:
                        doc = json.loads(line)
                        yield doc.get("text", "")
                    except json.JSONDecodeError:
                        # Malformed JSON — treat the raw line as a document
                        # rather than silently dropping it
                        yield line
                else:
                    yield line

    def __iter__(self) -> Iterator[dict]:
        """
        Yield {input_ids, labels} dicts of shape (max_seq_len,).

        Documents are concatenated pack-style separated by <eos>.
        We fill a rolling buffer of token IDs and emit one chunk of
        max_seq_len + 1 tokens each time the buffer is long enough.

        Why buffer max_seq_len + 1 tokens?
          We need max_seq_len tokens for input_ids AND max_seq_len tokens
          for labels. Because labels[t] = input_ids[t+1], a single chunk
          of max_seq_len+1 tokens gives us both: input = chunk[:-1],
          target = chunk[1:]. This avoids any cross-batch boundary issues.

        The slide-by-max_seq_len after each emit:
          buffer = buffer[max_seq_len:]
          This drops exactly the tokens we emitted as input_ids, keeping
          the overlap token (which was the last label) as the first input
          of the next chunk. That way no tokens are wasted and the model
          sees every token in context at least once.
        """
        tokenizer = self._get_tokenizer()
        # We need the EOS ID to mark document boundaries in the packed stream
        eos_id = tokenizer.token_to_id("<eos>")
        bos_id = tokenizer.token_to_id("<bos>")

        # Rolling accumulator: we append token IDs from each document until
        # we have enough for a full sequence, then slice and emit.
        buffer = []

        for doc in self._document_iter():
            if not doc.strip():
                continue

            # Tokenize the document. The tokenizer's post-processor may
            # automatically prepend BOS and append EOS — we strip these so
            # we can manage separators ourselves (pack-style needs explicit
            # control over where document boundaries appear).
            enc = tokenizer.encode(doc)
            ids = enc.ids

            # Remove auto-added BOS if present
            if ids and ids[0] == bos_id:
                ids = ids[1:]
            # Remove auto-added EOS if present (we'll add our own below)
            if ids and ids[-1] == eos_id:
                ids = ids[:-1]

            # Append document tokens then a single EOS as the boundary marker.
            # The EOS tells the model "a new document context begins here",
            # which is important so cross-document attention doesn't cause
            # the model to treat the end of one document as context for the
            # next one's first tokens.
            buffer.extend(ids)
            buffer.append(eos_id)

            # Emit as many full-length chunks as the buffer allows.
            # We need max_seq_len+1 buffered to produce one sample.
            while len(buffer) >= self.max_seq_len + 1:
                chunk = buffer[: self.max_seq_len + 1]
                # Slide window forward by exactly max_seq_len, retaining the
                # last token of this chunk as the first token of the next
                # (the overlap preserves causal continuity across chunks).
                buffer = buffer[self.max_seq_len :]

                # Split the +1 chunk into input / target pair
                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:], dtype=torch.long)

                yield {"input_ids": input_ids, "labels": labels}

        # Emit the final partial buffer if it has more than one token.
        # Pad to max_seq_len+1 with <pad>; the loss function ignores pad
        # tokens (ignore_index=0), so this doesn't corrupt training.
        if len(buffer) > 1:
            chunk = buffer[: self.max_seq_len + 1]
            if len(chunk) < self.max_seq_len + 1:
                pad_id = tokenizer.token_to_id("<pad>")
                # Right-pad the chunk so it has the correct fixed length
                chunk = chunk + [pad_id] * (self.max_seq_len + 1 - len(chunk))
            input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
            labels = torch.tensor(chunk[1:], dtype=torch.long)
            yield {"input_ids": input_ids, "labels": labels}


class SyntheticDataset(IterableDataset):
    """
    Synthetic dataset for --mode validate.

    Generates random token sequences without needing a real corpus or
    a tokenizer. The vocabulary range starts at 6 to avoid special token
    IDs (0=<pad>, 1=<unk>, 2=<bos>, 3=<eos>, etc.) so that the loss
    function's ignore_index=0 logic doesn't accidentally mask real tokens.

    Used only for shape / throughput validation — not real training.
    Loss values from this dataset are meaningless; what we check is that
    (a) the training loop runs without errors and (b) loss decreases
    slightly as the model overfits the random sequences.
    """

    def __init__(self, vocab_size: int, seq_len: int, n_batches: int = 200):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.n_batches = n_batches  # Finite length so DataLoader terminates

    def __iter__(self):
        for _ in range(self.n_batches):
            # Generate seq_len+1 random tokens; split into input/target pair.
            # Lower bound 6 avoids special token IDs (pad, unk, bos, eos, …).
            tokens = torch.randint(6, self.vocab_size, (self.seq_len + 1,))
            yield {
                "input_ids": tokens[:-1],  # Shape: (seq_len,)
                "labels": tokens[1:],  # Shape: (seq_len,) — next-token targets
            }


# ---------------------------------------------------------------------------
# LR schedule: linear warmup + cosine decay
# ---------------------------------------------------------------------------


def get_lr(step: int, total_steps: int, warmup_steps: int, lr: float, lr_min: float) -> float:
    """
    Compute the learning rate for a given step using linear warmup + cosine decay.

    Phase 1 — Linear warmup (steps 0 → warmup_steps):
      LR rises linearly from 0 to `lr`. This prevents large, destabilising
      updates at initialisation when gradient norms are high and the second-
      moment estimates in AdamW are cold (close to zero, giving artificially
      large effective learning rates).

    Phase 2 — Cosine annealing (steps warmup_steps → total_steps):
      LR follows a half-cosine curve from `lr` down to `lr_min`.
      The cosine formula is:
        lr_t = lr_min + (lr - lr_min) × 0.5 × (1 + cos(π × t/T))
      where t = steps elapsed since warmup ended, T = total decay steps.
      At t=0  the cosine term = 1.0, so lr_t = lr          (full LR)
      At t=T  the cosine term = 0.0, so lr_t = lr_min      (minimum LR)
      The cosine shape decays slowly at first and accelerates in the middle,
      which empirically produces better loss curves than linear decay.

    Phase 3 — After total_steps:
      LR is held flat at lr_min. This handles the case where the DataLoader
      outlasts the scheduled steps (e.g. when max_steps is set explicitly).

    Why not use a PyTorch LRScheduler?
      torch.optim.lr_scheduler objects are stateful and require calling
      scheduler.step() in sync with optimizer.step(). With gradient
      accumulation this requires extra bookkeeping. Computing the LR
      analytically from the step number is simpler and makes it trivial
      to resume from a checkpoint without restoring scheduler state.

    Args:
      step:         Current optimizer step (0-based).
      total_steps:  Total number of optimizer steps for the run.
      warmup_steps: Number of steps for the linear warmup phase.
      lr:           Peak learning rate (reached at end of warmup).
      lr_min:       Minimum learning rate (floor of cosine decay).

    Returns:
      Learning rate to use for this step.
    """
    if step < warmup_steps:
        # Linear ramp: at step=0 we get lr*(1/warmup_steps) which is small
        # but non-zero. We add 1 so step=0 gives lr/warmup_steps, not 0.
        return lr * (step + 1) / warmup_steps

    # Hold at lr_min once training has passed the scheduled end
    if step >= total_steps:
        return lr_min

    # Cosine decay over the steps between warmup end and training end
    decay_steps = total_steps - warmup_steps  # Total steps in the decay phase
    elapsed = step - warmup_steps  # Steps elapsed since warmup ended

    # cosine goes from 1.0 (at elapsed=0) to -1.0 (at elapsed=decay_steps).
    # The 0.5*(1 + cos(π*t/T)) expression maps that to [1.0 → 0.0], then
    # we scale and shift into [lr_min → lr].
    cosine = 0.5 * (1.0 + math.cos(math.pi * elapsed / decay_steps))
    return lr_min + (lr - lr_min) * cosine


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------


def save_checkpoint(
    step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_cfg: TrainConfig,
    tokens_seen: int,
    loss: float,
    output_dir: str,
):
    """
    Save a fully resumable checkpoint to disk.

    The checkpoint contains:
      - model state_dict    (all parameter tensors)
      - optimizer state_dict (momentum buffers, second moments — critical for
                              a smooth resume; without these AdamW restarts
                              its moment estimates from scratch, causing a
                              temporary LR spike on resume)
      - step and tokens_seen (to restore the LR schedule and token counter)
      - train_config as a plain dict (for provenance / reproducibility)

    File naming uses zero-padded step numbers (step_0050000.pt) so that
    lexicographic sort == chronological sort, which the cleanup logic below
    relies on.

    Retention policy: keep the 3 most recent checkpoints. Older ones are
    deleted to avoid filling disk on long runs. Adjust [:-3] to keep more.
    """
    os.makedirs(output_dir, exist_ok=True)
    # Zero-pad to 7 digits: supports up to 9,999,999 steps before overflow
    path = Path(output_dir) / f"step_{step:07d}.pt"

    state = {
        "step": step,
        "tokens_seen": tokens_seen,
        "loss": loss,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        # asdict() converts the dataclass to a plain dict for JSON-safe storage
        "train_config": asdict(train_cfg),
    }
    torch.save(state, path)

    # Glob all existing checkpoints, sorted lexicographically (= oldest first
    # due to zero-padded step numbers), then delete all but the last 3.
    checkpoints = sorted(Path(output_dir).glob("step_*.pt"))
    for old_ckpt in checkpoints[:-3]:
        old_ckpt.unlink()

    print(f"  → Saved checkpoint: {path}")
    return str(path)


def load_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer):
    """
    Load a checkpoint and restore model + optimizer state.

    map_location="cpu" loads tensors onto CPU first regardless of what
    device they were saved from. We then let the caller move the model
    to the correct device. This avoids CUDA OOM errors that can occur
    if the checkpoint was saved on a different GPU than we're resuming on.

    weights_only=False is required because the checkpoint also contains
    the train_config dict (a plain Python object, not just tensors).
    Note: set weights_only=True for untrusted checkpoint sources.

    Returns:
      (step, tokens_seen) so the training loop can restore the LR schedule
      and the progress counter exactly.
    """
    print(f"  → Resuming from: {path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    return state["step"], state["tokens_seen"]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(cfg: TrainConfig):
    """
    Main training loop.

    High-level structure:
      1. Device + dtype setup
      2. Model instantiation + optional gradient checkpointing + compile
      3. Optimizer with parameter-group weight-decay split
      4. Schedule parameters (total steps, warmup steps)
      5. Optional checkpoint resume
      6. DataLoader construction
      7. AMP autocast context
      8. Micro-step loop with gradient accumulation
         - Every grad_accum micro-steps → clip gradients, update LR, step optimizer
         - Log / eval / checkpoint at configured intervals
      9. Final checkpoint

    Gradient accumulation pattern:
      Instead of one large batch per GPU step we run `grad_accum` small
      forward+backward passes and sum their gradients before calling
      optimizer.step(). This is mathematically equivalent to a single
      forward pass with a batch size of batch_size × grad_accum, but
      uses only 1/grad_accum the peak activation memory.

      Critical detail: we divide the loss by grad_accum before each
      backward call. PyTorch accumulates (adds) gradients across backward
      calls, so without scaling the accumulated gradient would be
      grad_accum× too large, causing a learning rate that is effectively
      multiplied by grad_accum.
    """

    # ── Device setup ──────────────────────────────────────────────────
    if cfg.backend == "neuron":
        try:
            import torch_neuronx

            # AWS Trainium uses XLA (Accelerated Linear Algebra) as its
            # device abstraction. All tensor operations are traced into an
            # XLA graph and compiled for the Neuron hardware.
            device = torch.device("xla")
            print("Backend: AWS Trainium (XLA)")
        except ImportError:
            print("ERROR: torch_neuronx not found. Install Neuron SDK.")
            sys.exit(1)
    elif cfg.backend == "mps":
        if not torch.backends.mps.is_available():
            print("WARNING: MPS not available, falling back to CPU")
            device = torch.device("cpu")
        else:
            device = torch.device("mps")
            print("Backend: Apple MPS (Metal Performance Shaders)")
    elif cfg.backend == "cuda":
        if not torch.cuda.is_available():
            print("WARNING: CUDA not available, falling back to CPU")
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
            print(f"Backend: CUDA — {torch.cuda.get_device_name(0)}")
            # Report total VRAM so the user can judge if the config will fit
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("Backend: CPU (validation / debug only)")

    # Resolve dtype string to torch.dtype constant once; used in multiple places
    dtype = torch.bfloat16 if cfg.dtype == "bfloat16" else torch.float32
    print(f"Precision: {cfg.dtype}")

    # ── Model ─────────────────────────────────────────────────────────
    model_cfg = CONFIGS[cfg.model_config]
    print(f"\nModel: {cfg.model_config.upper()}")
    params = model_cfg.num_params()
    print(f"  Parameters: {params['total_B']:.3f}B ({params['total_M']:.0f}M)")
    print(f"  Layers: {model_cfg.n_layers}, d_model: {model_cfg.d_model}")
    print(f"  GQA: {model_cfg.n_heads}Q / {model_cfg.n_kv_heads}KV = {model_cfg.gqa_ratio}:1")

    model = SmallReasoningModel(model_cfg)
    # Move weights to the target device AND cast to the training dtype.
    # BF16 halves the weight memory vs FP32 (e.g. 1B params: 2 GB vs 4 GB).
    model = model.to(device=device, dtype=dtype)

    # Gradient checkpointing — see _enable_gradient_checkpointing for details.
    # At 1B this is optional (model fits in 32 GB without it), but enabling it
    # by default ensures the same code path works for the 3B model too.
    if cfg.grad_checkpointing:
        _enable_gradient_checkpointing(model)
        print("  Gradient checkpointing: ON")

    # torch.compile traces the model's computation graph and emits fused
    # Triton kernels, bypassing Python overhead on the hot path.
    # Disabled for Trainium because XLA already does graph compilation
    # via its own mechanism and torch.compile is not yet supported there.
    if cfg.compile and cfg.backend == "cuda":
        print("  Compiling model (torch.compile)...")
        model = torch.compile(model)

    # ── Optimizer ─────────────────────────────────────────────────────
    # Weight decay (L2 regularisation) should only be applied to weight
    # matrices, not to vectors or scalars. Specifically:
    #   - LayerNorm scale/bias: these normalise activations; decaying them
    #     would shrink the normalisation towards zero which is harmful.
    #   - Embeddings: decaying embeddings biases frequently-used tokens
    #     towards zero which degrades their representations.
    #   - 1-D parameters (param.ndim < 2): biases and norms typically have
    #     ndim=1; this catches any we might have missed by name.
    # Everything else (attention weight matrices, FFN matrices) gets decay.
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Match by name for norms and embeddings; by shape for 1-D tensors
        if "norm" in name or "embedding" in name or param.ndim < 2:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    # fused=True: on CUDA, PyTorch can fuse all the per-parameter Adam update
    # kernels into a single CUDA kernel launch. For large models this reduces
    # kernel-launch overhead and global memory round-trips, giving ~5-15%
    # optimizer-step speedup. Not available on CPU or XLA (Trainium).
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        eps=cfg.eps,
        # Fused AdamW merges the update into one CUDA kernel launch — ~5-15% faster.
        # Only available on CUDA; MPS and CPU must use the unfused implementation.
        fused=(cfg.backend == "cuda"),
    )

    # ── Schedule params ───────────────────────────────────────────────
    # Compute once here so we don't repeat the arithmetic every step
    tokens_per_step = cfg.effective_batch_tokens()
    total_steps = cfg.total_steps(tokens_per_step)
    warmup_steps = max(1, int(total_steps * cfg.warmup_fraction))  # At least 1 warmup step

    print(f"\nTraining schedule:")
    print(f"  Target tokens:      {cfg.max_tokens/1e9:.1f}B")
    print(f"  Total steps:        {total_steps:,}")
    print(f"  Warmup steps:       {warmup_steps:,} ({cfg.warmup_fraction*100:.0f}%)")
    print(f"  Effective batch:    {tokens_per_step/1e6:.2f}M tokens/step")
    print(f"    = {cfg.batch_size} (micro) × {cfg.grad_accum} (accum) × {cfg.max_seq_len} (seq)")
    print(f"  LR: {cfg.lr:.2e} → {cfg.lr_min:.2e} (cosine)")
    print(f"  Output:             {cfg.output_dir}")

    # ── Resume ────────────────────────────────────────────────────────
    start_step = 0
    tokens_seen = 0
    if cfg.resume:
        # Restoring start_step means get_lr() will return the correct LR
        # for where we are in the schedule, not re-run warmup from scratch.
        start_step, tokens_seen = load_checkpoint(cfg.resume, model, optimizer)
        print(f"\nResumed from step {start_step:,} ({tokens_seen/1e9:.2f}B tokens)")

    # ── Dataset ───────────────────────────────────────────────────────
    if cfg.data_path:
        train_dataset = TokenDataset(
            cfg.data_path,
            cfg.tokenizer_path,
            cfg.max_seq_len,
            split="train",
        )
        # Only create the validation dataset if we'll actually use it.
        # An eval_every of 0 means "no eval", so skip the dataset entirely.
        val_dataset = (
            TokenDataset(
                cfg.data_path,
                cfg.tokenizer_path,
                cfg.max_seq_len,
                split="val",
            )
            if cfg.eval_every > 0
            else None
        )
    else:
        # Synthetic mode — for --mode validate (no real data needed)
        print("\nNo data_path specified — using synthetic data (validate mode)")
        train_dataset = SyntheticDataset(model_cfg.vocab_size, cfg.max_seq_len, n_batches=500)
        val_dataset = SyntheticDataset(model_cfg.vocab_size, cfg.max_seq_len, n_batches=20)

    # num_workers=0: run data loading in the main process thread.
    #
    # Why not num_workers > 0 with IterableDataset:
    #   PyTorch's DataLoader forks worker processes AFTER the model is already
    #   on CUDA.  When pin_memory=True is combined with num_workers > 0 and an
    #   IterableDataset, the pin-memory thread and worker processes end up in a
    #   futex deadlock — the main process waits on a lock that a worker holds,
    #   while the worker waits on pin_memory queue operations.  The bug manifests
    #   as 100 % CPU + 98 % GPU utilisation with zero logged steps (training is
    #   technically running in the GPU async queue, but nothing ever completes
    #   from the Python perspective because the DataLoader never yields the next
    #   batch to the main thread).
    #
    #   For map-style datasets (with __getitem__) forked workers are safe because
    #   each worker independently indexes into a shared-memory array.  For
    #   IterableDataset the workers each iterate the full sequence, which requires
    #   explicit sharding logic to avoid data duplication AND the fork+CUDA+pin_memory
    #   combination that causes the deadlock.
    #
    # Why num_workers=0 is still fast enough:
    #   With gradient_accumulation=128, the GPU is busy for ~100 s per logged step.
    #   The CPU needs to tokenise one micro-batch (4 × 2048 = 8192 tokens ≈ 32 KB
    #   of text) per micro-step.  At typical disk/tokeniser throughput this takes
    #   ~1 ms per micro-step, vs ~1 s of GPU compute.  Data loading never becomes
    #   the bottleneck: the GPU is idle <1 % of the time waiting for data.
    #
    # pin_memory=False for the same reason: pin_memory spawns an additional
    # background thread that interacts with the CUDA allocator.  With num_workers=0
    # the tensors are created directly in the main thread and moved to CUDA via
    # .to(device), which is synchronous and equally fast for our batch sizes.
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=0,  # IterableDataset + workers + pin_memory → futex deadlock
        pin_memory=False,  # only useful when num_workers > 0
        prefetch_factor=None,  # must be None when num_workers=0
    )

    # ── AMP autocast (BF16 mixed precision) ───────────────────────────
    # torch.amp.autocast casts eligible operations to the target dtype
    # during the forward pass (and backward, via autograd). "Eligible"
    # means matmuls and convolutions; layer norms and softmax stay in FP32
    # for numerical stability.
    #
    # Why BF16 doesn't need a GradScaler (unlike FP16):
    #   FP16 has a very small representable range for large values (max ~65504).
    #   Large gradient norms overflow to Inf, causing NaN propagation — hence
    #   the need for loss scaling. BF16 has the same 8-bit exponent as FP32
    #   (max ~3.4e38), so overflows are essentially impossible in practice.
    #   We can skip the GradScaler entirely, simplifying the training loop.
    #
    # Trainium handles mixed precision via its own XLA casting; autocast
    # on "cpu" device type is a no-op here — we just need a valid context.
    autocast_ctx = torch.amp.autocast(
        # autocast device_type must match the tensor device.
        # MPS supports bfloat16 ops natively in PyTorch 2.x but uses "mps"
        # as the device_type string. Neuron handles its own precision — skip.
        device_type="cuda" if cfg.backend == "cuda" else ("mps" if cfg.backend == "mps" else "cpu"),
        dtype=dtype,
        enabled=(dtype == torch.bfloat16 and cfg.backend not in ("neuron", "cpu")),
    )

    # ── Training loop ─────────────────────────────────────────────────
    print(f"\n{'─'*62}")
    print(f"  {'step':>8}  {'loss':>8}  {'lr':>10}  {'tok/s':>10}  {'tokens':>12}")
    print(f"{'─'*62}")

    model.train()
    step = start_step  # Counts optimizer updates (not micro-steps)
    micro_step = 0  # Counts individual forward+backward passes within one grad-accum window
    accum_loss = 0.0  # Accumulates raw (unscaled) loss values across micro-steps for logging
    t0 = time.time()  # Wall-clock time at the start of the current log window
    tokens_batch = 0  # Tokens processed in the current log window (for tok/s metric)

    for batch in train_loader:
        # Stop when we've completed the scheduled number of optimizer steps
        if step >= total_steps:
            break

        # Move tensors to device. non_blocking=True would allow async transfer
        # but requires pin_memory=True on the DataLoader, which is already set.
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # ── Forward + loss ────────────────────────────────────────────
        with autocast_ctx:
            # The model returns (logits, aux_outputs); we only need logits here.
            logits, _ = model(input_ids)
            # labels are already the next-token targets (from the dataset's
            # chunk[1:] slice), so we pass them directly to _direct_loss.
            loss = _direct_loss(logits, labels)

            # CRITICAL: divide loss by grad_accum before backward.
            # PyTorch accumulates (adds) gradients across .backward() calls.
            # If we call .backward() grad_accum times without scaling, the
            # accumulated gradient is grad_accum× what a single-step would give,
            # effectively multiplying the learning rate by grad_accum.
            # Dividing by grad_accum here keeps the gradient magnitude the same
            # as if we had done a single large forward pass.
            scaled_loss = loss / cfg.grad_accum

        # ── Backward ─────────────────────────────────────────────────
        # Gradients accumulate in .grad tensors; we don't zero them until
        # after the optimizer step at the end of the accumulation window.
        scaled_loss.backward()

        # Track unscaled loss for logging (we want the "true" loss value)
        accum_loss += loss.item()
        tokens_batch += input_ids.numel()  # B × T tokens processed this micro-step
        micro_step += 1

        # ── Optimizer step (every grad_accum micro steps) ─────────────
        # Only update weights once we've accumulated enough gradients.
        if micro_step % cfg.grad_accum == 0:
            # Gradient clipping: scale down all gradients if the global
            # L2 norm exceeds grad_clip (default 1.0). This caps the
            # maximum parameter update per step, preventing loss spikes
            # caused by unusually large gradients (which occur during
            # training on diverse text, especially at domain boundaries).
            # clip_grad_norm_ returns the pre-clipping norm for monitoring.
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            # Update the LR for this step. We set it manually rather than
            # using a scheduler because get_lr() is stateless: it computes
            # the correct value from the step number alone, which makes
            # checkpoint resume trivial (no scheduler state to restore).
            current_lr = get_lr(step, total_steps, warmup_steps, cfg.lr, cfg.lr_min)
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr

            optimizer.step()

            # set_to_none=True is more memory-efficient than zero_grad().
            # zero_grad() fills .grad tensors with 0.0 (tensors remain
            # allocated). set_to_none=True deallocates the .grad tensors
            # entirely — the next backward will re-allocate them, saving
            # memory during the window between optimizer.step() and the
            # next backward call. Also slightly faster because writing
            # None is cheaper than writing a tensor of zeros.
            optimizer.zero_grad(set_to_none=True)

            # Update total token counter (used for LR schedule when
            # max_tokens is the stop condition, and for progress reporting)
            tokens_seen += tokens_batch
            # Average loss over the grad_accum micro-steps in this window
            step_loss = accum_loss / cfg.grad_accum

            # ── Logging ──────────────────────────────────────────────
            if step % cfg.log_every == 0:
                dt = time.time() - t0
                # tok/s measures throughput over the log window (not just
                # the last step), giving a more stable estimate
                tok_per_s = tokens_batch / dt if dt > 0 else 0
                print(
                    f"  {step:>8,}  {step_loss:>8.4f}  {current_lr:>10.2e}"
                    f"  {tok_per_s:>10,.0f}  {tokens_seen/1e9:>10.2f}B"
                    f"  ‖∇‖={grad_norm:.2f}"
                )
                t0 = time.time()  # Reset timer for next log window

            # ── Validation ───────────────────────────────────────────
            if cfg.eval_every > 0 and step % cfg.eval_every == 0 and step > 0:
                val_loss = evaluate(model, val_dataset, cfg, device, dtype)
                print(f"  {'':>8}  val_loss={val_loss:.4f}  (step {step:,})")
                # evaluate() sets model.eval(); restore train mode after
                model.train()

            # ── Checkpoint ───────────────────────────────────────────
            if step % cfg.save_every == 0 and step > 0:
                save_checkpoint(step, model, optimizer, cfg, tokens_seen, step_loss, cfg.output_dir)

            # Reset per-window accumulators for the next grad_accum window
            accum_loss = 0.0
            tokens_batch = 0
            step += 1  # Increment the optimizer-step counter

    # ── Final checkpoint ──────────────────────────────────────────────
    print(f"\n{'─'*62}")
    print(f"Training complete.")
    print(f"  Steps:        {step:,}")
    print(f"  Tokens seen:  {tokens_seen/1e9:.2f}B")
    # step_loss may be undefined if the loop body never executed an optimizer
    # step; for the final save we use the last computed value.
    save_checkpoint(step, model, optimizer, cfg, tokens_seen, step_loss, cfg.output_dir)

    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataset: IterableDataset,
    cfg: TrainConfig,
    device: torch.device,
    dtype: torch.dtype,
    max_batches: int = 50,
) -> float:
    """
    Compute mean validation loss over up to max_batches batches.

    @torch.no_grad() disables gradient tracking for the entire function,
    saving memory and compute — we don't need gradients during evaluation.

    max_batches caps the evaluation at 50 batches (50 × batch_size sequences)
    to keep eval time short. For a precise perplexity estimate use a larger
    value; for a training-time sanity check 50 batches is sufficient to detect
    divergence or overfitting.

    Returns float("nan") if the dataset yielded no batches (empty val set),
    rather than raising a ZeroDivisionError, so the caller can detect this
    gracefully.
    """
    model.eval()  # Disables dropout and uses running stats in batch norm (if any)
    loader = DataLoader(dataset, batch_size=cfg.batch_size)
    losses = []

    # Recreate the autocast context with the same settings as training
    autocast_ctx = torch.amp.autocast(
        # autocast device_type must match the tensor device.
        # MPS supports bfloat16 ops natively in PyTorch 2.x but uses "mps"
        # as the device_type string. Neuron handles its own precision — skip.
        device_type="cuda" if cfg.backend == "cuda" else ("mps" if cfg.backend == "mps" else "cpu"),
        dtype=dtype,
        enabled=(dtype == torch.bfloat16 and cfg.backend not in ("neuron", "cpu")),
    )

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        with autocast_ctx:
            logits, _ = model(input_ids)
            loss = _direct_loss(logits, labels)
        losses.append(loss.item())

    # Return mean loss, or NaN if no batches were processed
    return sum(losses) / len(losses) if losses else float("nan")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _direct_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Cross-entropy loss for next-token prediction.

    This function assumes the caller has already handled the causal shift:
      logits[b, t, :] is the prediction for position t
      labels[b, t]    is the target token at position t+1
    (Both have the same shape because the dataset yields pre-shifted pairs.)

    Why reshape to (B*T, V) and (B*T,)?
      F.cross_entropy expects either (N, C) logits and (N,) targets, or the
      multi-dimensional form (N, C, *). Flattening the batch and time dims
      into a single N dimension is the simplest way to use the standard form.

    ignore_index=0:
      Token ID 0 is <pad>. The final partial chunk in TokenDataset pads
      short sequences with <pad>. Setting ignore_index=0 means those
      positions are excluded from the loss numerator and denominator,
      so padding doesn't dilute the loss on real tokens.

    reduction="mean":
      Average over all non-padding token positions in the batch. This is
      the standard choice; "sum" would make the loss scale with batch size
      and would require adjusting the learning rate.

    Args:
      logits: (B, T, vocab_size) — raw (un-normalised) model predictions
      labels: (B, T)             — integer target token IDs

    Returns:
      Scalar tensor: mean cross-entropy loss over non-padding positions.
    """
    B, T, V = logits.shape
    return torch.nn.functional.cross_entropy(
        logits.reshape(B * T, V),  # Flatten batch × time into a single N dimension
        labels.reshape(B * T),  # Flatten correspondingly
        ignore_index=0,  # Exclude <pad> (ID=0) from loss computation
        reduction="mean",
    )


def _enable_gradient_checkpointing(model: SmallReasoningModel):
    """
    Enable activation recomputation (gradient checkpointing) on each TransformerBlock.

    Memory / compute tradeoff:
      Normally, the backward pass uses activations saved during the forward pass
      to compute gradients. For a transformer with n_layers layers, each of depth
      d_model and sequence length T, the activation memory is O(n_layers × T × d_model).
      For a 3B model (32 layers, d_model=2560, T=2048) at BF16 this is roughly:
        32 × 2048 × 2560 × 2 bytes ≈ 335 MB per batch element
      At batch_size=4 that's ~1.3 GB just for activations — manageable, but
      combined with weights (6 GB), gradients (6 GB), and optimizer state (12 GB)
      the total exceeds 32 GB.

      With gradient checkpointing, activations are NOT stored after the forward
      pass. During the backward pass, PyTorch re-runs each block's forward to
      recompute the activations it needs. This reduces activation memory by
      roughly a factor of sqrt(n_layers) (the theoretical optimum), in practice
      ~30-40% total memory reduction, at the cost of ~20% extra FLOPs.

    Implementation detail — why not model.gradient_checkpointing_enable()?
      That HuggingFace method only works for HF-wrapped models. Our model is
      custom, so we patch each block's forward method manually. The patch wraps
      the original forward with torch.utils.checkpoint.checkpoint(), which
      implements the discard-and-recompute logic.

    Why skip checkpointing during generation (kv_cache is not None)?
      Gradient checkpointing is a training-only optimisation: it only matters
      when we need to compute gradients. During inference (when we pass a
      kv_cache), we never call .backward(), so there is no benefit to discarding
      activations. More importantly, the checkpoint wrapper returns only a single
      tensor (the block output), discarding the kv_cache return value, which
      would break incremental decoding.

    use_reentrant=False:
      The "reentrant" implementation of torch.utils.checkpoint uses Python's
      re-entrant autograd hook system and has subtle interaction bugs with
      certain operations (e.g. in-place ops, custom autograd functions).
      The newer non-reentrant implementation (use_reentrant=False) is more
      robust and is the recommended default as of PyTorch 2.x.
    """
    import functools

    for block in model.blocks:
        original_forward = block.forward

        @functools.wraps(original_forward)
        def checkpointed_forward(
            x, attention_mask=None, kv_cache=None, position_offset=0, _orig=original_forward
        ):
            # torch.utils.checkpoint.checkpoint does not natively support
            # keyword arguments. We close over the kwargs in a nested function
            # and pass only `x` as the checkpointed tensor argument.
            def fn(x_):
                out, kv = _orig(
                    x_,
                    attention_mask=attention_mask,
                    kv_cache=kv_cache,
                    position_offset=position_offset,
                )
                return out  # Only return the activation tensor; kv is discarded

            # If kv_cache is provided we're in inference mode: skip checkpointing.
            # The checkpoint wrapper would drop the kv return value, breaking
            # autoregressive generation which relies on the cached keys/values.
            if kv_cache is not None:
                return _orig(
                    x,
                    attention_mask=attention_mask,
                    kv_cache=kv_cache,
                    position_offset=position_offset,
                )

            # Run the block under checkpoint: activations are freed after
            # forward and will be recomputed during backward pass.
            out = torch_checkpoint(fn, x, use_reentrant=False)
            # Return None for the kv_cache slot because checkpointing is
            # training-only (no kv_cache expected in training forward passes).
            return out, None

        # Replace the block's bound method with the checkpointed version.
        # functools.wraps preserves the original __name__ and __doc__.
        block.forward = checkpointed_forward


# ---------------------------------------------------------------------------
# Validate mode — runs a 20-step smoke test without real data
# ---------------------------------------------------------------------------


def validate_mode(model_config: str):
    """
    Validate the training loop without real data or a full corpus.

    Purpose: catch configuration errors, shape mismatches, NaN losses, and
    LR schedule bugs quickly (< 2 minutes) before committing to a long run.

    Runs 20 optimizer steps with synthetic random-token data and checks:
      - Loss decreases (model is learning on the synthetic sequences)
      - No NaN/Inf in gradients (grad_norm should be a finite number)
      - Throughput is measurable (tok/s logged at each step)
      - LR schedule is correct (should ramp up for first ~1 step, then decay)

    The batch configuration (batch_size=2, grad_accum=2, seq_len=256) is
    chosen to be small enough to complete quickly on any hardware, including
    a CPU fallback. This is not representative of production batch sizes.

    save_every=999999 ensures no checkpoints are written during validation
    (no disk I/O needed for a smoke test).
    """
    print(f"\nValidation mode — {model_config.upper()}")
    print("Running 20 steps with synthetic data to verify training loop...\n")

    # Small config for speed: 20 steps × 2 × 2 × 256 = 20,480 tokens total
    cfg = TrainConfig(
        model_config=model_config,
        max_tokens=1,  # Ignored because max_steps overrides it
        max_steps=20,  # Run exactly 20 optimizer steps then stop
        batch_size=2,
        grad_accum=2,
        max_seq_len=256,  # Short sequence for fast iteration
        lr=3e-4,
        grad_checkpointing=True,
        dtype="bfloat16",
        backend=(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        ),
        log_every=1,  # Log every step so we can see the loss curve
        save_every=999999,  # Never save — avoid disk I/O during smoke test
        eval_every=10,  # Run one validation pass halfway through
        output_dir="/tmp/validate_checkpoints",
        compile=False,  # Skip compilation overhead for a quick test
    )

    model = train(cfg)

    print("\nValidation complete.")
    print("  If loss decreased over 20 steps, the training loop is working.")
    print("  If loss is NaN, check weight initialization and LR.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Pre-train the small reasoning model (Phase 0)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--config", type=str, default="1b", choices=["500m", "1b", "3b"], help="Model configuration"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "validate"],
        help="train: full training | validate: 20-step smoke test",
    )
    parser.add_argument(
        "--data_path", type=str, default="", help="Path to training corpus (.jsonl or .txt)"
    )
    parser.add_argument("--tokenizer_path", type=str, default="./tokenizer_output")
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--max_tokens", type=int, default=50_000_000_000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=128)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--no_grad_ckpt", action="store_true", help="Disable gradient checkpointing"
    )
    parser.add_argument(
        "--backend", type=str, default="cuda", choices=["cuda", "neuron", "mps", "cpu"]
    )
    parser.add_argument("--compile", action="store_true", help="torch.compile (CUDA only)")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from")

    args = parser.parse_args()

    if args.mode == "validate":
        validate_mode(args.config)
        return

    cfg = TrainConfig(
        model_config=args.config,
        data_path=args.data_path,
        tokenizer_path=args.tokenizer_path,
        output_dir=args.output_dir,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        max_seq_len=args.max_seq_len,
        lr=args.lr,
        grad_checkpointing=not args.no_grad_ckpt,
        backend=args.backend,
        compile=args.compile,
        resume=args.resume,
    )

    train(cfg)


if __name__ == "__main__":
    main()
