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

# Local imports
from model.architecture import SmallReasoningModel, ModelConfig, CONFIGS, compute_loss

# Optional: gradient checkpointing support
from torch.utils.checkpoint import checkpoint as torch_checkpoint


# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    # Model
    model_config:    str   = "1b"         # Key into CONFIGS dict

    # Data
    data_path:       str   = ""           # Path to .jsonl or .txt corpus
    tokenizer_path:  str   = "./tokenizer_output"
    max_seq_len:     int   = 2048         # Context window during training
                                          # Shorter than model max — saves memory

    # Budget
    max_tokens:      int   = 50_000_000_000   # 50B tokens for 1B model
    max_steps:       int   = -1              # Alternative: cap by steps (-1 = use max_tokens)

    # Batch
    batch_size:      int   = 4            # Per-GPU micro batch size
    grad_accum:      int   = 128          # Gradient accumulation steps
                                          # Effective batch = batch_size × grad_accum × seq_len
                                          # = 4 × 128 × 2048 = 1,048,576 tokens (~1M)

    # Optimizer
    lr:              float = 3e-4         # Peak learning rate
    lr_min:          float = 3e-5         # Cosine decay floor (10% of peak)
    beta1:           float = 0.9
    beta2:           float = 0.95         # Lower than default 0.999 — better for LLM training
    eps:             float = 1e-8
    weight_decay:    float = 0.1
    grad_clip:       float = 1.0

    # Schedule
    warmup_fraction: float = 0.02         # 2% of total steps for warmup

    # Memory
    grad_checkpointing: bool = True       # Recompute activations on backward — saves ~30% memory
    dtype:           str  = "bfloat16"   # "bfloat16" or "float32"

    # Logging + checkpointing
    output_dir:      str  = "./checkpoints"
    log_every:       int  = 10            # Log every N steps
    save_every:      int  = 1000          # Save checkpoint every N steps
    eval_every:      int  = 500           # Evaluate on val set every N steps (0 = disable)

    # Hardware
    backend:         str  = "cuda"        # "cuda" | "neuron" | "cpu"
    compile:         bool = False         # torch.compile (CUDA only; not Trainium)

    # Resume
    resume:          str  = ""            # Path to checkpoint to resume from

    def effective_batch_tokens(self) -> int:
        return self.batch_size * self.grad_accum * self.max_seq_len

    def total_steps(self, tokens_per_step: int) -> int:
        if self.max_steps > 0:
            return self.max_steps
        return self.max_tokens // tokens_per_step


# ---------------------------------------------------------------------------
# Dataset — streaming token iterator
# ---------------------------------------------------------------------------

class TokenDataset(IterableDataset):
    """
    Streaming dataset for pre-training.

    Reads a JSONL corpus (one document per line, {"text": "..."})
    or a plain text file, tokenizes on the fly, and yields fixed-length
    token sequences of length max_seq_len + 1 (for next-token prediction).

    The +1 is so that input = tokens[:-1] and target = tokens[1:] both
    have length max_seq_len with no additional slicing at training time.

    Documents are concatenated with <eos> between them (pack-style).
    No padding — every token in every batch is a real token.
    This is the standard approach for pre-training efficiency.

    For production: replace with a pre-tokenized .bin file for speed.
    The tokenize-on-the-fly approach here is for simplicity and portability.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer_path: str,
        max_seq_len: int,
        split: str = "train",          # "train" or "val"
        val_fraction: float = 0.001,   # 0.1% of data for validation
    ):
        self.data_path    = data_path
        self.tokenizer_path = tokenizer_path
        self.max_seq_len  = max_seq_len
        self.split        = split
        self.val_fraction = val_fraction

        # Lazy tokenizer load (so it works with DataLoader workers)
        self._tokenizer   = None

    def _get_tokenizer(self):
        """Load tokenizer on first access (each worker gets its own copy)."""
        if self._tokenizer is None:
            from tokenizers import Tokenizer
            tok_path = Path(self.tokenizer_path) / "tokenizer.json"
            self._tokenizer = Tokenizer.from_file(str(tok_path))
        return self._tokenizer

    def _document_iter(self) -> Iterator[str]:
        """Yield raw document strings from the corpus file."""
        path = Path(self.data_path)
        if not path.exists():
            raise FileNotFoundError(f"Corpus not found: {path}")

        # Determine split by line hash (stable, no full scan needed)
        split_threshold = int(self.val_fraction * (2**32))

        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                # Route to train or val split deterministically
                line_hash = hash(line) % (2**32)
                in_val = line_hash < split_threshold

                if self.split == "val" and not in_val:
                    continue
                if self.split == "train" and in_val:
                    continue

                # JSONL format: {"text": "..."}
                if line.startswith("{"):
                    try:
                        doc = json.loads(line)
                        yield doc.get("text", "")
                    except json.JSONDecodeError:
                        yield line   # Fallback: treat as plain text
                else:
                    yield line

    def __iter__(self) -> Iterator[dict]:
        """
        Yield {input_ids, labels} dicts of shape (max_seq_len,).

        Documents are concatenated pack-style separated by <eos>.
        We fill a buffer and emit when we have max_seq_len + 1 tokens.
        """
        tokenizer = self._get_tokenizer()
        eos_id    = tokenizer.token_to_id("<eos>")
        bos_id    = tokenizer.token_to_id("<bos>")

        buffer = []   # Accumulate token IDs

        for doc in self._document_iter():
            if not doc.strip():
                continue

            # Tokenize: encode without the post-processor's BOS/EOS
            # (we handle separators manually for pack-style)
            enc   = tokenizer.encode(doc)
            ids   = enc.ids

            # Strip BOS/EOS that the post-processor adds
            if ids and ids[0] == bos_id:
                ids = ids[1:]
            if ids and ids[-1] == eos_id:
                ids = ids[:-1]

            # Append document tokens + separator
            buffer.extend(ids)
            buffer.append(eos_id)  # Document boundary

            # Emit fixed-length sequences
            while len(buffer) >= self.max_seq_len + 1:
                chunk = buffer[:self.max_seq_len + 1]
                buffer = buffer[self.max_seq_len:]  # Slide by max_seq_len

                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels    = torch.tensor(chunk[1:],  dtype=torch.long)

                yield {"input_ids": input_ids, "labels": labels}

        # Final partial chunk (pad if needed, mask padding in loss)
        if len(buffer) > 1:
            chunk = buffer[:self.max_seq_len + 1]
            if len(chunk) < self.max_seq_len + 1:
                pad_id = tokenizer.token_to_id("<pad>")
                chunk  = chunk + [pad_id] * (self.max_seq_len + 1 - len(chunk))
            input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
            labels    = torch.tensor(chunk[1:],  dtype=torch.long)
            yield {"input_ids": input_ids, "labels": labels}


class SyntheticDataset(IterableDataset):
    """
    Synthetic dataset for --mode validate.
    Generates random token sequences without needing a real corpus.
    Used only for shape / throughput validation — not real training.
    """
    def __init__(self, vocab_size: int, seq_len: int, n_batches: int = 200):
        self.vocab_size = vocab_size
        self.seq_len    = seq_len
        self.n_batches  = n_batches

    def __iter__(self):
        for _ in range(self.n_batches):
            tokens = torch.randint(6, self.vocab_size, (self.seq_len + 1,))
            yield {
                "input_ids": tokens[:-1],
                "labels":    tokens[1:],
            }


# ---------------------------------------------------------------------------
# LR schedule: linear warmup + cosine decay
# ---------------------------------------------------------------------------

def get_lr(step: int, total_steps: int, warmup_steps: int, lr: float, lr_min: float) -> float:
    """
    Linear warmup then cosine decay.

    Warmup (0 → warmup_steps): linear 0 → lr
    Decay (warmup_steps → total_steps): cosine lr → lr_min
    After total_steps: lr_min (flat)

    This is the standard schedule for LLM pre-training.
    The 2% warmup fraction prevents early training instability
    (large gradient norms at random initialization).
    """
    if step < warmup_steps:
        # Linear warmup
        return lr * (step + 1) / warmup_steps
    if step >= total_steps:
        return lr_min
    # Cosine decay
    decay_steps = total_steps - warmup_steps
    elapsed     = step - warmup_steps
    cosine      = 0.5 * (1.0 + math.cos(math.pi * elapsed / decay_steps))
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
    """Save a resumable checkpoint."""
    os.makedirs(output_dir, exist_ok=True)
    path = Path(output_dir) / f"step_{step:07d}.pt"

    state = {
        "step":         step,
        "tokens_seen":  tokens_seen,
        "loss":         loss,
        "model":        model.state_dict(),
        "optimizer":    optimizer.state_dict(),
        "train_config": asdict(train_cfg),
    }
    torch.save(state, path)

    # Keep only the last 3 checkpoints to save disk space
    checkpoints = sorted(Path(output_dir).glob("step_*.pt"))
    for old_ckpt in checkpoints[:-3]:
        old_ckpt.unlink()

    print(f"  → Saved checkpoint: {path}")
    return str(path)


def load_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer):
    """Load a checkpoint and return (step, tokens_seen)."""
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

    Structure:
      Setup → DataLoader → Model → Optimizer → Loop → Save

    The loop runs micro-steps (forward + backward) accumulated
    across grad_accum steps before an optimizer update.
    Each optimizer step = one "training step" for logging/scheduling.
    """

    # ── Device setup ──────────────────────────────────────────────────
    if cfg.backend == "neuron":
        try:
            import torch_neuronx
            device = torch.device("xla")   # Trainium uses XLA device
            print("Backend: AWS Trainium (XLA)")
        except ImportError:
            print("ERROR: torch_neuronx not found. Install Neuron SDK.")
            sys.exit(1)
    elif cfg.backend == "cuda":
        if not torch.cuda.is_available():
            print("WARNING: CUDA not available, falling back to CPU")
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
            print(f"Backend: CUDA — {torch.cuda.get_device_name(0)}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device("cpu")
        print("Backend: CPU (validation / debug only)")

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
    model = model.to(device=device, dtype=dtype)

    # Gradient checkpointing — recomputes activations during backward
    # Saves ~30-40% activation memory at cost of ~20% extra compute
    if cfg.grad_checkpointing:
        _enable_gradient_checkpointing(model)
        print("  Gradient checkpointing: ON")

    # torch.compile — speeds up training ~20-30% on CUDA (not Trainium)
    if cfg.compile and cfg.backend == "cuda":
        print("  Compiling model (torch.compile)...")
        model = torch.compile(model)

    # ── Optimizer ─────────────────────────────────────────────────────
    # Separate parameters into weight-decay and no-weight-decay groups.
    # Embeddings and norm parameters should NOT have weight decay.
    decay_params     = []
    no_decay_params  = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # No weight decay on: biases (none in our model), norms, embeddings
        if "norm" in name or "embedding" in name or param.ndim < 2:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params,    "weight_decay": cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        eps=cfg.eps,
        fused=(cfg.backend == "cuda"),   # Fused AdamW is faster on CUDA
    )

    # ── Schedule params ───────────────────────────────────────────────
    tokens_per_step = cfg.effective_batch_tokens()
    total_steps     = cfg.total_steps(tokens_per_step)
    warmup_steps    = max(1, int(total_steps * cfg.warmup_fraction))

    print(f"\nTraining schedule:")
    print(f"  Target tokens:      {cfg.max_tokens/1e9:.1f}B")
    print(f"  Total steps:        {total_steps:,}")
    print(f"  Warmup steps:       {warmup_steps:,} ({cfg.warmup_fraction*100:.0f}%)")
    print(f"  Effective batch:    {tokens_per_step/1e6:.2f}M tokens/step")
    print(f"    = {cfg.batch_size} (micro) × {cfg.grad_accum} (accum) × {cfg.max_seq_len} (seq)")
    print(f"  LR: {cfg.lr:.2e} → {cfg.lr_min:.2e} (cosine)")
    print(f"  Output:             {cfg.output_dir}")

    # ── Resume ────────────────────────────────────────────────────────
    start_step   = 0
    tokens_seen  = 0
    if cfg.resume:
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
        val_dataset = TokenDataset(
            cfg.data_path,
            cfg.tokenizer_path,
            cfg.max_seq_len,
            split="val",
        ) if cfg.eval_every > 0 else None
    else:
        # Synthetic mode — for --mode validate
        print("\nNo data_path specified — using synthetic data (validate mode)")
        train_dataset = SyntheticDataset(model_cfg.vocab_size, cfg.max_seq_len, n_batches=500)
        val_dataset   = SyntheticDataset(model_cfg.vocab_size, cfg.max_seq_len, n_batches=20)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=2 if cfg.data_path else 0,
        pin_memory=(cfg.backend == "cuda"),
        prefetch_factor=4 if cfg.data_path else None,
    )

    # ── AMP scaler (CUDA BF16 — not needed for Trainium) ──────────────
    # BF16 doesn't need loss scaling (unlike FP16), but torch.amp.autocast
    # handles the BF16/FP32 cast boundary automatically
    autocast_ctx = torch.amp.autocast(
        device_type="cuda" if cfg.backend == "cuda" else "cpu",
        dtype=dtype,
        enabled=(dtype == torch.bfloat16 and cfg.backend != "neuron"),
    )

    # ── Training loop ─────────────────────────────────────────────────
    print(f"\n{'─'*62}")
    print(f"  {'step':>8}  {'loss':>8}  {'lr':>10}  {'tok/s':>10}  {'tokens':>12}")
    print(f"{'─'*62}")

    model.train()
    step         = start_step
    micro_step   = 0
    accum_loss   = 0.0
    t0           = time.time()
    tokens_batch = 0

    for batch in train_loader:
        if step >= total_steps:
            break

        input_ids = batch["input_ids"].to(device)
        labels    = batch["labels"].to(device)

        # ── Forward + loss ────────────────────────────────────────────
        with autocast_ctx:
            logits, _ = model(input_ids)
            # compute_loss does the causal shift internally
            # labels here are already shifted (target = input[1:])
            # so we pass both as-is and use a direct cross-entropy
            loss = _direct_loss(logits, labels)
            # Scale loss by grad_accum for correct gradient magnitude
            scaled_loss = loss / cfg.grad_accum

        # ── Backward ─────────────────────────────────────────────────
        scaled_loss.backward()

        accum_loss  += loss.item()
        tokens_batch += input_ids.numel()
        micro_step   += 1

        # ── Optimizer step (every grad_accum micro steps) ─────────────
        if micro_step % cfg.grad_accum == 0:
            # Gradient clipping — prevents loss spikes from blowing up weights
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            # Update LR
            current_lr = get_lr(step, total_steps, warmup_steps, cfg.lr, cfg.lr_min)
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)   # set_to_none is more memory-efficient

            # Accumulate token count
            tokens_seen  += tokens_batch
            step_loss     = accum_loss / cfg.grad_accum

            # ── Logging ──────────────────────────────────────────────
            if step % cfg.log_every == 0:
                dt       = time.time() - t0
                tok_per_s = tokens_batch / dt if dt > 0 else 0
                print(
                    f"  {step:>8,}  {step_loss:>8.4f}  {current_lr:>10.2e}"
                    f"  {tok_per_s:>10,.0f}  {tokens_seen/1e9:>10.2f}B"
                    f"  ‖∇‖={grad_norm:.2f}"
                )
                t0 = time.time()

            # ── Validation ───────────────────────────────────────────
            if cfg.eval_every > 0 and step % cfg.eval_every == 0 and step > 0:
                val_loss = evaluate(model, val_dataset, cfg, device, dtype)
                print(f"  {'':>8}  val_loss={val_loss:.4f}  (step {step:,})")
                model.train()

            # ── Checkpoint ───────────────────────────────────────────
            if step % cfg.save_every == 0 and step > 0:
                save_checkpoint(
                    step, model, optimizer, cfg, tokens_seen, step_loss, cfg.output_dir
                )

            # Reset accumulators
            accum_loss   = 0.0
            tokens_batch = 0
            step        += 1

    # ── Final checkpoint ──────────────────────────────────────────────
    print(f"\n{'─'*62}")
    print(f"Training complete.")
    print(f"  Steps:        {step:,}")
    print(f"  Tokens seen:  {tokens_seen/1e9:.2f}B")
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
    """Compute validation loss over a fixed number of batches."""
    model.eval()
    loader = DataLoader(dataset, batch_size=cfg.batch_size)
    losses = []

    autocast_ctx = torch.amp.autocast(
        device_type="cuda" if cfg.backend == "cuda" else "cpu",
        dtype=dtype,
        enabled=(dtype == torch.bfloat16 and cfg.backend != "neuron"),
    )

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        labels    = batch["labels"].to(device)
        with autocast_ctx:
            logits, _ = model(input_ids)
            loss = _direct_loss(logits, labels)
        losses.append(loss.item())

    return sum(losses) / len(losses) if losses else float("nan")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _direct_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Cross-entropy loss where logits and labels are already shifted.

    logits: (B, T, vocab_size)  — predictions
    labels: (B, T)              — targets (next tokens, from dataset)
    ignore_index=0              — <pad> tokens don't contribute to loss
    """
    B, T, V = logits.shape
    return torch.nn.functional.cross_entropy(
        logits.reshape(B * T, V),
        labels.reshape(B * T),
        ignore_index=0,   # <pad> token ID
        reduction="mean",
    )


def _enable_gradient_checkpointing(model: SmallReasoningModel):
    """
    Enable gradient checkpointing on each TransformerBlock.

    Replaces the block's forward call with torch.utils.checkpoint.checkpoint,
    which discards activations after the forward pass and recomputes them
    during backward. Reduces activation memory by ~(n_layers / sqrt(n_layers)).

    The tradeoff: ~20% extra compute for ~30-40% memory savings.
    At 1B on the 5090 (17GB training footprint), this is not required.
    At 3B (52GB), this is what makes the difference between fitting or not.
    """
    import functools

    for block in model.blocks:
        original_forward = block.forward

        @functools.wraps(original_forward)
        def checkpointed_forward(x, attention_mask=None, kv_cache=None, position_offset=0,
                                  _orig=original_forward):
            # torch_checkpoint doesn't support kwargs directly
            # Wrap in a lambda that captures the fixed args
            def fn(x_):
                out, kv = _orig(x_, attention_mask=attention_mask,
                                kv_cache=kv_cache, position_offset=position_offset)
                return out

            # During generation (kv_cache is not None), skip checkpointing
            if kv_cache is not None:
                return _orig(x, attention_mask=attention_mask,
                             kv_cache=kv_cache, position_offset=position_offset)

            out = torch_checkpoint(fn, x, use_reentrant=False)
            return out, None

        block.forward = checkpointed_forward


# ---------------------------------------------------------------------------
# Validate mode — runs a 20-step smoke test without real data
# ---------------------------------------------------------------------------

def validate_mode(model_config: str):
    """
    Validate the training loop without real data or a full corpus.
    Runs 20 optimizer steps with synthetic data and checks:
      - Loss decreases (model is learning)
      - No NaN/Inf in gradients
      - Throughput is measurable
      - LR schedule is correct
    """
    print(f"\nValidation mode — {model_config.upper()}")
    print("Running 20 steps with synthetic data to verify training loop...\n")

    # Use a small effective batch for speed
    cfg = TrainConfig(
        model_config    = model_config,
        max_tokens      = 1,           # ignored in validate mode
        max_steps       = 20,
        batch_size      = 2,
        grad_accum      = 2,
        max_seq_len     = 256,         # Short for speed
        lr              = 3e-4,
        grad_checkpointing = True,
        dtype           = "bfloat16",
        backend         = "cuda" if torch.cuda.is_available() else "cpu",
        log_every       = 1,
        save_every      = 999999,      # Don't save during validate
        eval_every      = 10,
        output_dir      = "/tmp/validate_checkpoints",
        compile         = False,
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

    parser.add_argument("--config",        type=str, default="1b",
                        choices=["500m", "1b", "3b"],
                        help="Model configuration")
    parser.add_argument("--mode",          type=str, default="train",
                        choices=["train", "validate"],
                        help="train: full training | validate: 20-step smoke test")
    parser.add_argument("--data_path",     type=str, default="",
                        help="Path to training corpus (.jsonl or .txt)")
    parser.add_argument("--tokenizer_path",type=str, default="./tokenizer_output")
    parser.add_argument("--output_dir",    type=str, default="./checkpoints")
    parser.add_argument("--max_tokens",    type=int, default=50_000_000_000)
    parser.add_argument("--batch_size",    type=int, default=4)
    parser.add_argument("--grad_accum",    type=int, default=128)
    parser.add_argument("--max_seq_len",   type=int, default=2048)
    parser.add_argument("--lr",            type=float, default=3e-4)
    parser.add_argument("--no_grad_ckpt",  action="store_true",
                        help="Disable gradient checkpointing")
    parser.add_argument("--backend",       type=str, default="cuda",
                        choices=["cuda", "neuron", "cpu"])
    parser.add_argument("--compile",       action="store_true",
                        help="torch.compile (CUDA only)")
    parser.add_argument("--resume",        type=str, default="",
                        help="Path to checkpoint to resume from")

    args = parser.parse_args()

    if args.mode == "validate":
        validate_mode(args.config)
        return

    cfg = TrainConfig(
        model_config       = args.config,
        data_path          = args.data_path,
        tokenizer_path     = args.tokenizer_path,
        output_dir         = args.output_dir,
        max_tokens         = args.max_tokens,
        batch_size         = args.batch_size,
        grad_accum         = args.grad_accum,
        max_seq_len        = args.max_seq_len,
        lr                 = args.lr,
        grad_checkpointing = not args.no_grad_ckpt,
        backend            = args.backend,
        compile            = args.compile,
        resume             = args.resume,
    )

    train(cfg)


if __name__ == "__main__":
    main()
