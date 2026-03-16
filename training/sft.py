"""
sft.py
======
Phase 1: Supervised Fine-Tuning (SFT) for the small reasoning model.

Takes the Phase 0 pre-trained checkpoint and teaches:
  1. Instruction following (user/assistant turn format)
  2. The <think>...</think> chain-of-thought structure
  3. Step-by-step reasoning before final answers

The critical difference from pre-training:
  Loss is computed ONLY on assistant turns.
  User prompts, system tokens, and <think> content structure
  are all in the input — but only the assistant's output tokens
  (including the reasoning chain) contribute to gradients.

  This is not a minor detail. Computing loss on the prompt causes
  the model to "overfit" to formatting rather than learning to
  generate reasoning. Get this wrong and Phase 2 GRPO will
  fail silently — the model will produce fluent but non-reasoning
  outputs because it never learned to own the generation.

Dataset format (all examples converted to this before training):
  {
    "prompt": "What is 15 + 27?",
    "response": "<think>\n15 + 27\n= 10 + 20 + 5 + 7\n= 30 + 12\n= 42\n</think>\n42"
  }

Token layout during training:
  [<bos>] [user tokens...] [<sep>] [<think>] [...] [</think>] [answer] [<eos>]
   ← no loss ──────────────────────────────────────────────────────────────────→ loss computed here

Usage:
  # Fine-tune 1B pre-trained model
  python sft.py \\
    --checkpoint ./checkpoints/1b/step_0025000.pt \\
    --config 1b \\
    --output_dir ./checkpoints/1b_sft \\
    --data_dir ./sft_data

  # Dry-run with synthetic data (no checkpoint needed)
  python sft.py --config 1b --mode validate
"""

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent / "model"))
from model.architecture import SmallReasoningModel, ModelConfig, CONFIGS


# ---------------------------------------------------------------------------
# Special token IDs (must match tokenizer — see tokenizer_config.json)
# ---------------------------------------------------------------------------
PAD_ID         = 0
BOS_ID         = 1
EOS_ID         = 2
UNK_ID         = 3
THINK_START_ID = 4
THINK_END_ID   = 5
LOSS_IGNORE    = -100   # PyTorch cross_entropy ignores this index


# ---------------------------------------------------------------------------
# SFT config
# ---------------------------------------------------------------------------

@dataclass
class SFTConfig:
    # Model
    model_config:  str   = "1b"
    checkpoint:    str   = ""          # Pre-trained checkpoint to start from (required for real runs)

    # Data
    data_dir:      str   = "./sft_data"
    tokenizer_path:str   = "./tokenizer_output"
    max_seq_len:   int   = 4096        # Longer than pre-training — CoT sequences are long

    # Training
    epochs:        int   = 2
    batch_size:    int   = 4
    grad_accum:    int   = 8           # Smaller than pre-training (smaller dataset)
    lr:            float = 2e-5        # Much lower than pre-train — we're fine-tuning
    lr_min:        float = 2e-6
    warmup_fraction: float = 0.03      # 3% warmup
    weight_decay:  float = 0.01
    grad_clip:     float = 1.0
    dropout:       float = 0.0

    # Memory
    grad_checkpointing: bool = True
    dtype:         str   = "bfloat16"

    # Output
    output_dir:    str   = "./checkpoints/sft"
    save_every:    int   = 500
    log_every:     int   = 10
    eval_every:    int   = 200

    # Hardware
    backend:       str   = "cuda"

    def effective_batch_tokens(self) -> int:
        return self.batch_size * self.grad_accum * self.max_seq_len


# ---------------------------------------------------------------------------
# Data formatting — the <think> template
# ---------------------------------------------------------------------------

CHAT_TEMPLATE = """\
User: {prompt}
Assistant: <think>
{thinking}
</think>
{answer}"""

CHAT_TEMPLATE_NO_THINK = """\
User: {prompt}
Assistant: {answer}"""


def format_example(example: dict) -> str:
    """
    Convert a dataset example to the training text format.

    Accepts multiple source formats and normalizes to our template.
    The <think> block is always present — if the source doesn't have
    step-by-step reasoning, we create a minimal one.

    Input formats handled:
      {"prompt": "...", "response": "..."}             — already formatted
      {"prompt": "...", "thinking": "...", "answer": "..."} — split reasoning
      {"problem": "...", "solution": "..."}            — math dataset style
      {"instruction": "...", "output": "..."}          — Alpaca style
      {"messages": [{"role": "user", ...}, ...]}       — ChatML style
    """
    # Already in our format
    if "prompt" in example and "response" in example:
        resp = example["response"]
        # Ensure <think> wrapper exists
        if "<think>" not in resp:
            resp = f"<think>\nLet me work through this.\n</think>\n{resp}"
        return f"User: {example['prompt']}\nAssistant: {resp}"

    # Split thinking/answer
    if "thinking" in example and "answer" in example:
        return CHAT_TEMPLATE.format(
            prompt=example.get("prompt", example.get("problem", "")),
            thinking=example["thinking"].strip(),
            answer=example["answer"].strip(),
        )

    # Math dataset (problem + solution)
    if "problem" in example and "solution" in example:
        return CHAT_TEMPLATE.format(
            prompt=example["problem"].strip(),
            thinking=example["solution"].strip(),
            answer=_extract_answer(example["solution"]),
        )

    # Alpaca-style
    if "instruction" in example and "output" in example:
        prompt = example["instruction"]
        if example.get("input", "").strip():
            prompt = f"{prompt}\n\n{example['input']}"
        output = example["output"]
        if "<think>" not in output:
            output = f"<think>\nLet me think about this.\n</think>\n{output}"
        return f"User: {prompt}\nAssistant: {output}"

    # ChatML messages
    if "messages" in example:
        msgs = example["messages"]
        user_content = next(
            (m["content"] for m in msgs if m["role"] == "user"), ""
        )
        asst_content = next(
            (m["content"] for m in msgs if m["role"] == "assistant"), ""
        )
        if "<think>" not in asst_content:
            asst_content = f"<think>\nLet me think step by step.\n</think>\n{asst_content}"
        return f"User: {user_content}\nAssistant: {asst_content}"

    raise ValueError(f"Unknown example format: {list(example.keys())}")


def _extract_answer(solution: str) -> str:
    """Extract final answer from a math solution string."""
    # Try common patterns: "= X", "answer is X", "\\boxed{X}"
    import re
    boxed = re.search(r"\\boxed\{([^}]+)\}", solution)
    if boxed:
        return boxed.group(1)
    # Last line that looks like an answer
    lines = [l.strip() for l in solution.strip().splitlines() if l.strip()]
    return lines[-1] if lines else solution


# ---------------------------------------------------------------------------
# Tokenization with loss masking
# ---------------------------------------------------------------------------

def tokenize_with_mask(
    text: str,
    tokenizer,
    max_seq_len: int,
) -> Optional[dict]:
    """
    Tokenize a formatted example and produce a loss mask.

    The loss mask is 1 for assistant output tokens, 0 for everything else.
    Loss is ONLY computed where mask == 1.

    Strategy: find the "Assistant:" boundary in the token sequence,
    then mask everything before it (including the boundary itself).

    Returns None if the sequence is too long after truncation or
    if no assistant boundary is found.

    Token layout:
      [<bos>] [User: ...] [Assistant: ] [<think>] [...] [</think>] [...] [<eos>]
        mask=0  mask=0        mask=0       mask=1   mask=1  mask=1   mask=1  mask=1
    """
    # Encode the full text
    enc = tokenizer.encode(text)
    ids = enc.ids   # includes <bos> and <eos> from post-processor

    # Truncate to max_seq_len (keep from the end — preserve the answer)
    if len(ids) > max_seq_len:
        ids = ids[:max_seq_len]
        # Ensure last token is <eos>
        ids[-1] = EOS_ID

    if len(ids) < 4:
        return None   # Too short to be useful

    # Find the assistant turn boundary
    # Encode "Assistant:" alone to get its token IDs for boundary detection
    asst_marker = tokenizer.encode("Assistant:").ids
    # Strip BOS/EOS that post-processor adds
    if asst_marker and asst_marker[0] == BOS_ID:
        asst_marker = asst_marker[1:]
    if asst_marker and asst_marker[-1] == EOS_ID:
        asst_marker = asst_marker[:-1]

    # Find last occurrence of the assistant marker in ids
    # (last, because there might be multi-turn examples)
    boundary = _find_subsequence(ids, asst_marker)

    if boundary is None:
        # Fallback: mask only the last 80% of the sequence
        # (crude but prevents total loss masking)
        boundary = max(1, len(ids) // 5)

    # Build mask: 0 before boundary+len(marker), 1 after
    mask_start = boundary + len(asst_marker)
    labels = [LOSS_IGNORE] * mask_start + ids[mask_start:]
    labels = labels[:len(ids)]  # ensure same length

    # Sanity: at least 10 tokens in the loss region
    active = sum(1 for l in labels if l != LOSS_IGNORE)
    if active < 10:
        return None

    input_ids = torch.tensor(ids,    dtype=torch.long)
    label_ids = torch.tensor(labels, dtype=torch.long)

    return {"input_ids": input_ids, "labels": label_ids}


def _find_subsequence(seq: list, subseq: list) -> Optional[int]:
    """Find the last occurrence of subseq in seq. Returns start index or None."""
    if not subseq or len(subseq) > len(seq):
        return None
    last = None
    for i in range(len(seq) - len(subseq) + 1):
        if seq[i:i+len(subseq)] == subseq:
            last = i
    return last


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SFTDataset(Dataset):
    """
    SFT dataset — loads, formats, and tokenizes examples at init.

    In-memory: SFT datasets are typically 1-3M examples at 500-2000 tokens
    each, which fits comfortably in RAM. We tokenize upfront so DataLoader
    workers don't need to re-tokenize on every batch.

    Data directory structure:
      sft_data/
        train.jsonl    — training examples (one JSON per line)
        val.jsonl      — validation examples

    Or single file:
      sft_data/data.jsonl  — split 95/5 at load time
    """

    def __init__(
        self,
        data_dir: str,
        tokenizer_path: str,
        max_seq_len: int,
        split: str = "train",
        max_examples: Optional[int] = None,
    ):
        from tokenizers import Tokenizer
        tok_path = Path(tokenizer_path) / "tokenizer.json"
        tokenizer = Tokenizer.from_file(str(tok_path))

        self.examples = []

        data_dir = Path(data_dir)

        # Find data files
        if (data_dir / f"{split}.jsonl").exists():
            files = [data_dir / f"{split}.jsonl"]
        elif (data_dir / "data.jsonl").exists():
            files = [data_dir / "data.jsonl"]
            # Will split below
        else:
            files = list(data_dir.glob("*.jsonl")) + list(data_dir.glob("*.json"))

        if not files:
            raise FileNotFoundError(f"No data files found in {data_dir}")

        raw_examples = []
        for fpath in sorted(files):
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        raw_examples.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        # Split if no separate val file
        if len(files) == 1 and not (data_dir / f"{split}.jsonl").exists():
            random.seed(42)
            random.shuffle(raw_examples)
            split_idx = max(1, int(len(raw_examples) * 0.95))
            if split == "train":
                raw_examples = raw_examples[:split_idx]
            else:
                raw_examples = raw_examples[split_idx:]

        if max_examples:
            raw_examples = raw_examples[:max_examples]

        # Tokenize
        skipped = 0
        for ex in raw_examples:
            try:
                text = format_example(ex)
                item = tokenize_with_mask(text, tokenizer, max_seq_len)
                if item is not None:
                    self.examples.append(item)
                else:
                    skipped += 1
            except Exception:
                skipped += 1
                continue

        print(f"  SFTDataset ({split}): {len(self.examples):,} examples "
              f"({skipped} skipped — too long or no assistant turn)")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict:
        return self.examples[idx]


class SyntheticSFTDataset(Dataset):
    """Synthetic SFT dataset for --mode validate. No real data needed."""

    def __init__(self, vocab_size: int, seq_len: int, n: int = 200):
        self.examples = []
        for _ in range(n):
            length = random.randint(seq_len // 2, seq_len)
            ids    = torch.randint(6, vocab_size, (length,))
            ids[0] = BOS_ID
            ids[-1] = EOS_ID
            # Mask first 30% as prompt
            labels = ids.clone()
            mask_end = max(1, length // 3)
            labels[:mask_end] = LOSS_IGNORE
            self.examples.append({"input_ids": ids, "labels": labels})

    def __len__(self): return len(self.examples)
    def __getitem__(self, i): return self.examples[i]


# ---------------------------------------------------------------------------
# Collator — variable length → padded batch
# ---------------------------------------------------------------------------

def sft_collate(batch: list[dict]) -> dict:
    """
    Pad a batch of variable-length sequences to the same length.

    Padding:
      input_ids: pad with PAD_ID (0)
      labels:    pad with LOSS_IGNORE (-100) — padding never contributes to loss

    Sequences are left-aligned, padded on the right.
    """
    max_len = max(item["input_ids"].shape[0] for item in batch)

    padded_ids    = []
    padded_labels = []
    attn_masks    = []

    for item in batch:
        ids = item["input_ids"]
        lbl = item["labels"]
        pad_len = max_len - ids.shape[0]

        padded_ids.append(
            torch.cat([ids, torch.full((pad_len,), PAD_ID, dtype=torch.long)])
        )
        padded_labels.append(
            torch.cat([lbl, torch.full((pad_len,), LOSS_IGNORE, dtype=torch.long)])
        )
        # Attention mask: 1 for real tokens, 0 for padding
        attn_masks.append(
            torch.cat([torch.ones(ids.shape[0]), torch.zeros(pad_len)]).long()
        )

    return {
        "input_ids":      torch.stack(padded_ids),
        "labels":         torch.stack(padded_labels),
        "attention_mask": torch.stack(attn_masks),
    }


# ---------------------------------------------------------------------------
# LR schedule (same formula as pretrain.py)
# ---------------------------------------------------------------------------

def get_lr(step: int, total_steps: int, warmup_steps: int,
           lr: float, lr_min: float) -> float:
    if step < warmup_steps:
        return lr * (step + 1) / warmup_steps
    if step >= total_steps:
        return lr_min
    elapsed  = step - warmup_steps
    progress = elapsed / (total_steps - warmup_steps)
    cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_min + (lr - lr_min) * cosine


# ---------------------------------------------------------------------------
# Loss — masked cross-entropy
# ---------------------------------------------------------------------------

def sft_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Cross-entropy loss with LOSS_IGNORE masking.

    logits: (B, T, vocab_size)
    labels: (B, T)  — LOSS_IGNORE (-100) where no loss should be computed

    The standard PyTorch cross_entropy handles ignore_index=-100 natively.
    No manual masking needed.

    Note: we do NOT shift here — the dataset already provides shifted labels.
    input_ids[t] predicts labels[t], where labels[t] = original token[t+1].
    """
    B, T, V = logits.shape
    loss = F.cross_entropy(
        logits.reshape(B * T, V),
        labels.reshape(B * T),
        ignore_index=LOSS_IGNORE,
        reduction="mean",
    )
    return loss


def mask_fraction(labels: torch.Tensor) -> float:
    """What fraction of tokens are in the loss region (diagnostic)."""
    active = (labels != LOSS_IGNORE).sum().item()
    total  = labels.numel()
    return active / total if total > 0 else 0.0


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: SFTConfig):
    # ── Device ────────────────────────────────────────────────────────
    if cfg.backend == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Backend: CUDA — {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Backend: CPU")

    dtype = torch.bfloat16 if cfg.dtype == "bfloat16" else torch.float32

    # ── Model ─────────────────────────────────────────────────────────
    model_cfg = CONFIGS[cfg.model_config]
    print(f"\nModel: {cfg.model_config.upper()}")
    model = SmallReasoningModel(model_cfg)

    # Load pre-trained checkpoint
    if cfg.checkpoint:
        print(f"Loading checkpoint: {cfg.checkpoint}")
        state = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)
        # Handle both raw state_dict and our checkpoint format
        sd = state.get("model", state)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            print(f"  WARNING: {len(missing)} missing keys")
        if unexpected:
            print(f"  WARNING: {len(unexpected)} unexpected keys")
        print(f"  Loaded from step {state.get('step', '?')}, "
              f"tokens seen: {state.get('tokens_seen', 0)/1e9:.1f}B")
    else:
        print("  WARNING: No checkpoint provided. Starting from random init.")
        print("  For real SFT, always start from a pre-trained base.")

    model = model.to(device=device, dtype=dtype)

    # Gradient checkpointing
    if cfg.grad_checkpointing:
        _enable_gradient_checkpointing(model)

    # ── Dataset ───────────────────────────────────────────────────────
    if Path(cfg.data_dir).exists() and any(Path(cfg.data_dir).iterdir()):
        print(f"\nLoading SFT data from: {cfg.data_dir}")
        train_dataset = SFTDataset(cfg.data_dir, cfg.tokenizer_path, cfg.max_seq_len, "train")
        val_dataset   = SFTDataset(cfg.data_dir, cfg.tokenizer_path, cfg.max_seq_len, "val")
    else:
        print(f"\nNo data found at {cfg.data_dir} — using synthetic data (validate mode)")
        train_dataset = SyntheticSFTDataset(model_cfg.vocab_size, cfg.max_seq_len // 4, n=400)
        val_dataset   = SyntheticSFTDataset(model_cfg.vocab_size, cfg.max_seq_len // 4, n=40)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=sft_collate,
        num_workers=2 if cfg.checkpoint else 0,
        pin_memory=(cfg.backend == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=sft_collate,
    )

    # ── Optimizer ─────────────────────────────────────────────────────
    # SFT uses a lower LR than pre-training and typically lower weight decay.
    # We apply weight decay only to 2D+ parameters (weights, not norms/biases).
    decay_params    = [p for n, p in model.named_parameters()
                       if p.requires_grad and p.ndim >= 2
                       and "norm" not in n and "embedding" not in n]
    no_decay_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and (p.ndim < 2
                       or "norm" in n or "embedding" in n)]

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params,    "weight_decay": cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
        betas=(0.9, 0.95),
        fused=(cfg.backend == "cuda"),
    )

    # ── Schedule ──────────────────────────────────────────────────────
    steps_per_epoch = math.ceil(len(train_dataset) / (cfg.batch_size * cfg.grad_accum))
    total_steps     = steps_per_epoch * cfg.epochs
    warmup_steps    = max(1, int(total_steps * cfg.warmup_fraction))

    print(f"\nSFT schedule:")
    print(f"  Examples:     {len(train_dataset):,} train / {len(val_dataset):,} val")
    print(f"  Epochs:       {cfg.epochs}")
    print(f"  Steps:        {total_steps:,} ({steps_per_epoch:,}/epoch)")
    print(f"  Warmup:       {warmup_steps:,} steps")
    print(f"  LR:           {cfg.lr:.1e} → {cfg.lr_min:.1e}")
    print(f"  Batch:        {cfg.batch_size} × {cfg.grad_accum} accum = "
          f"{cfg.batch_size * cfg.grad_accum} sequences/step")

    autocast_ctx = torch.amp.autocast(
        device_type="cuda" if cfg.backend == "cuda" else "cpu",
        dtype=dtype,
        enabled=(dtype == torch.bfloat16),
    )

    # ── Training loop ─────────────────────────────────────────────────
    print(f"\n{'─'*70}")
    print(f"  {'step':>7}  {'loss':>8}  {'val_loss':>9}  {'lr':>9}  {'mask%':>7}  {'tok/s':>8}")
    print(f"{'─'*70}")

    os.makedirs(cfg.output_dir, exist_ok=True)
    step        = 0
    best_val    = float("inf")
    t0          = time.time()

    for epoch in range(cfg.epochs):
        model.train()
        accum_loss = 0.0
        accum_mask = 0.0
        micro      = 0

        for batch in train_loader:
            if step >= total_steps:
                break

            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)
            # attention_mask unused here — we rely on label masking
            # (LOSS_IGNORE handles padding; causal mask is always applied)

            with autocast_ctx:
                logits, _ = model(input_ids)
                loss = sft_loss(logits, labels)
                scaled = loss / cfg.grad_accum

            scaled.backward()
            accum_loss += loss.item()
            accum_mask += mask_fraction(labels)
            micro      += 1

            if micro % cfg.grad_accum == 0:
                # Gradient clip + optimizer step
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

                current_lr = get_lr(step, total_steps, warmup_steps, cfg.lr, cfg.lr_min)
                for pg in optimizer.param_groups:
                    pg["lr"] = current_lr

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if step % cfg.log_every == 0:
                    dt      = time.time() - t0
                    tps     = (cfg.batch_size * cfg.grad_accum * cfg.max_seq_len) / dt
                    avg_msk = accum_mask / cfg.grad_accum * 100
                    print(
                        f"  {step:>7,}  {accum_loss/cfg.grad_accum:>8.4f}"
                        f"  {'':>9}  {current_lr:>9.2e}"
                        f"  {avg_msk:>6.1f}%  {tps:>8,.0f}"
                        f"  ep={epoch+1}"
                    )
                    t0 = time.time()

                if cfg.eval_every > 0 and step % cfg.eval_every == 0 and step > 0:
                    val_loss = evaluate(model, val_loader, cfg, device, dtype, autocast_ctx)
                    is_best  = val_loss < best_val
                    if is_best:
                        best_val = val_loss
                        _save(step, model, optimizer, cfg, val_loss,
                              Path(cfg.output_dir) / "best.pt")
                    print(
                        f"  {step:>7,}  {'':>8}  {val_loss:>9.4f}  {'':>9}"
                        f"  {'':>7}  {'':>8}  {'★ best' if is_best else ''}"
                    )
                    model.train()

                if step % cfg.save_every == 0 and step > 0:
                    _save(step, model, optimizer, cfg, accum_loss / cfg.grad_accum,
                          Path(cfg.output_dir) / f"step_{step:07d}.pt")

                accum_loss = 0.0
                accum_mask = 0.0
                step      += 1

        print(f"  Epoch {epoch+1} complete.")

    # Final save
    _save(step, model, optimizer, cfg, 0.0,
          Path(cfg.output_dir) / "final.pt")
    print(f"\n{'─'*70}")
    print(f"SFT complete. Best val loss: {best_val:.4f}")
    print(f"Checkpoints: {cfg.output_dir}")

    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, cfg, device, dtype, autocast_ctx, max_batches=30):
    model.eval()
    losses = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        labels    = batch["labels"].to(device)
        with autocast_ctx:
            logits, _ = model(input_ids)
            loss = sft_loss(logits, labels)
        losses.append(loss.item())
    return sum(losses) / len(losses) if losses else float("nan")


# ---------------------------------------------------------------------------
# Checkpoint save
# ---------------------------------------------------------------------------

def _save(step, model, optimizer, cfg, loss, path):
    torch.save({
        "step":         step,
        "loss":         loss,
        "model":        model.state_dict(),
        "optimizer":    optimizer.state_dict(),
        "sft_config":   asdict(cfg),
        "phase":        "sft",
    }, path)
    print(f"  → Saved: {path}")


# ---------------------------------------------------------------------------
# Gradient checkpointing (same pattern as pretrain.py)
# ---------------------------------------------------------------------------

def _enable_gradient_checkpointing(model):
    import functools
    from torch.utils.checkpoint import checkpoint as torch_cp

    for block in model.blocks:
        orig = block.forward

        @functools.wraps(orig)
        def cp_forward(x, attention_mask=None, kv_cache=None,
                       position_offset=0, _orig=orig):
            if kv_cache is not None:
                return _orig(x, attention_mask=attention_mask,
                             kv_cache=kv_cache, position_offset=position_offset)
            def fn(x_):
                out, _ = _orig(x_, attention_mask=attention_mask,
                               kv_cache=None, position_offset=position_offset)
                return out
            return torch_cp(fn, x, use_reentrant=False), None

        block.forward = cp_forward


# ---------------------------------------------------------------------------
# Validate mode
# ---------------------------------------------------------------------------

def validate_mode(model_config: str):
    """Smoke test: 10 optimizer steps with synthetic data."""
    print(f"\nSFT validate mode — {model_config.upper()}")
    print("Running 10 steps with synthetic data...\n")

    cfg = SFTConfig(
        model_config       = model_config,
        checkpoint         = "",
        data_dir           = "/nonexistent",   # triggers synthetic path
        epochs             = 1,
        batch_size         = 2,
        grad_accum         = 2,
        max_seq_len        = 128,
        lr                 = 2e-5,
        grad_checkpointing = True,
        dtype              = "bfloat16",
        backend            = "cuda" if torch.cuda.is_available() else "cpu",
        log_every          = 1,
        save_every         = 999999,
        eval_every         = 5,
        output_dir         = "/tmp/sft_validate",
    )
    train(cfg)


# ---------------------------------------------------------------------------
# Data format validation utility
# ---------------------------------------------------------------------------

def validate_data(data_dir: str, tokenizer_path: str, max_seq_len: int = 4096, n: int = 5):
    """
    Inspect how your data will be tokenized before running training.

    Shows token counts, mask fractions, and a snippet of the formatted text
    for the first N examples. Run this before starting SFT.

    Usage:
      python sft.py --mode inspect --data_dir ./sft_data --tokenizer_path ./tokenizer_output
    """
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(Path(tokenizer_path) / "tokenizer.json"))

    data_dir = Path(data_dir)
    files = list(data_dir.glob("*.jsonl"))[:1]
    if not files:
        print("No .jsonl files found.")
        return

    print(f"\nInspecting: {files[0]}")
    print("─" * 60)

    with open(files[0]) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            try:
                ex   = json.loads(line.strip())
                text = format_example(ex)
                item = tokenize_with_mask(text, tokenizer, max_seq_len)

                if item is None:
                    print(f"Example {i}: SKIPPED (too short or no assistant turn)")
                    continue

                ids    = item["input_ids"]
                labels = item["labels"]
                active = (labels != LOSS_IGNORE).sum().item()
                total  = len(ids)

                print(f"\nExample {i}:")
                print(f"  Tokens:  {total}  |  Loss tokens: {active} ({active/total*100:.0f}%)")
                print(f"  Text preview: {text[:120].replace(chr(10), '↵')}...")

                # Show where the loss boundary is
                boundary = next((j for j, l in enumerate(labels.tolist()) if l != LOSS_IGNORE), total)
                print(f"  Loss starts at token {boundary}: "
                      f"'{tokenizer.decode([ids[boundary].item()])}'")

            except Exception as e:
                print(f"Example {i}: ERROR — {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SFT fine-tuning — Phase 1",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config",         type=str, default="1b",
                        choices=["500m", "1b", "3b"])
    parser.add_argument("--mode",           type=str, default="train",
                        choices=["train", "validate", "inspect"])
    parser.add_argument("--checkpoint",     type=str, default="",
                        help="Pre-trained checkpoint (.pt)")
    parser.add_argument("--data_dir",       type=str, default="./sft_data")
    parser.add_argument("--tokenizer_path", type=str, default="./tokenizer_output")
    parser.add_argument("--output_dir",     type=str, default="./checkpoints/sft")
    parser.add_argument("--epochs",         type=int, default=2)
    parser.add_argument("--batch_size",     type=int, default=4)
    parser.add_argument("--grad_accum",     type=int, default=8)
    parser.add_argument("--max_seq_len",    type=int, default=4096)
    parser.add_argument("--lr",             type=float, default=2e-5)
    parser.add_argument("--backend",        type=str, default="cuda",
                        choices=["cuda", "neuron", "cpu"])
    parser.add_argument("--no_grad_ckpt",   action="store_true")

    args = parser.parse_args()

    if args.mode == "validate":
        validate_mode(args.config)
        return

    if args.mode == "inspect":
        validate_data(args.data_dir, args.tokenizer_path)
        return

    cfg = SFTConfig(
        model_config       = args.config,
        checkpoint         = args.checkpoint,
        data_dir           = args.data_dir,
        tokenizer_path     = args.tokenizer_path,
        output_dir         = args.output_dir,
        epochs             = args.epochs,
        batch_size         = args.batch_size,
        grad_accum         = args.grad_accum,
        max_seq_len        = args.max_seq_len,
        lr                 = args.lr,
        grad_checkpointing = not args.no_grad_ckpt,
        backend            = args.backend,
    )
    train(cfg)


if __name__ == "__main__":
    main()
