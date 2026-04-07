"""
architecture.py
===============
Small reasoning model — core transformer architecture.

Implements the consensus stack from the spec:
  Pre-norm RMSNorm → GQA (with QK-Norm + RoPE) → SwiGLU FFN
  Tied input/output embeddings. No bias terms anywhere.

Design philosophy
-----------------
Every choice here prioritises training stability and hardware efficiency
over novelty.  The goal is a model that trains cleanly on AWS Trainium2
from scratch, not a research prototype.

Key design decisions and their rationale:
  - Pre-norm (vs post-norm): avoids early-layer gradient explosion, which
    is especially severe at the small scale where signal-to-noise in
    activations is lower.
  - RMSNorm (vs LayerNorm): skips the mean-centering step, which is
    empirically unnecessary and ~15% slower on systolic-array hardware.
  - GQA (vs MHA): reduces KV-cache memory by gqa_ratio× at inference,
    making 16k context feasible without custom memory management.
  - QK-Norm: prevents attention logit explosion at init — without it,
    dot products of random projections grow as O(sqrt(d)), which causes
    attention to collapse to near-one-hot distributions in early training.
  - SwiGLU (vs ReLU/GELU): consistently +0.2-0.5 perplexity at the same
    parameter budget; the extra gate matrix pays for itself.
  - Tied embeddings: saves vocab_size × d_model parameters (≈67M at 1B
    scale); also regularises — the model cannot have two inconsistent
    representations of the same token.
  - No bias anywhere: simplifies weight decay (can apply uniformly),
    removes a class of quantisation bugs, and has no empirical downside.
  - rope_base=500_000: with base=10_000 (original RoPE), the effective
    context degrades past ~4k tokens because high-frequency dimensions
    complete many full rotations and become uninformative.  A higher base
    spreads the frequencies out, keeping all dimensions informative up to
    the target sequence length.

Three configs (all tile-aligned for Trainium2):
  Config A — 500M  (validation / RTX 5090)
  Config B —   1B  (primary / Trn2)
  Config C —   3B  (full experiment / Trn2 or cloud H100)

Design invariants enforced in __post_init__:
  - d_model % 128 == 0          (tile alignment)
  - head_dim == 128             (maps to Trn2 SBUF partition dimension exactly)
  - n_heads % n_kv_heads == 0   (integer GQA ratio)
  - ffn_intermediate % 128 == 0 (tile alignment)
  - vocab_size % 128 == 0       (tile alignment)

Usage:
  from architecture import ModelConfig, SmallReasoningModel, CONFIGS

  model = SmallReasoningModel(CONFIGS["1b"])
  logits = model(input_ids)          # (B, T, vocab_size)

  # Parameter count
  print(model.num_params())

  # For Trainium: export as TorchScript before neuronx compilation
  # traced = torch.jit.trace(model, example_inputs)
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    # Core dimensions
    d_model: int = 2048  # Hidden dimension — must be multiple of 128
    n_layers: int = 20  # Number of transformer blocks
    n_heads: int = 16  # Number of query heads
    n_kv_heads: int = 4  # Number of key/value heads (GQA)
    head_dim: int = 128  # Dimension per head — FIXED at 128 for tile alignment
    ffn_intermediate: int = 5504  # SwiGLU intermediate dimension — must be multiple of 128

    # Vocabulary
    vocab_size: int = 32768  # Tile-aligned: 32768 / 128 = 256

    # Sequence
    max_seq_len: int = 16384  # Maximum sequence length

    # RoPE
    rope_base: float = 500_000.0  # Base frequency — high value for long CoT sequences

    # Regularization
    dropout: float = 0.0  # Set > 0 only for ablation studies; 0 is standard

    # Misc
    tie_embeddings: bool = True  # Tie input embedding ↔ LM head (saves vocab×d_model params)
    norm_eps: float = 1e-5  # RMSNorm epsilon

    def __post_init__(self):
        """
        Enforce all spec invariants at construction time.  Fail loudly — silent
        misalignment would produce a model that compiles but runs incorrectly or
        inefficiently on Trainium2 without any obvious error message.

        Each assertion has a specific hardware or correctness reason:
        """

        # --- Tile alignment assertions ---
        # Trainium2's NeuronCore-v3 systolic array operates on 128-element tiles.
        # Any tensor dimension that does not divide evenly by 128 will be padded
        # internally by the XLA/Neuron compiler, wasting SBUF capacity and causing
        # uneven workload distribution across the 32 NeuronCores.  We enforce
        # alignment here so that padding is zero — every byte of SRAM is useful.
        assert self.d_model % 128 == 0, f"d_model={self.d_model} must be ÷128 for tile alignment"
        assert (
            self.ffn_intermediate % 128 == 0
        ), f"ffn_intermediate={self.ffn_intermediate} must be ÷128 for tile alignment"
        assert (
            self.vocab_size % 128 == 0
        ), f"vocab_size={self.vocab_size} must be ÷128 for tile alignment"

        # --- head_dim must be exactly 128 ---
        # The Trn2 SBUF (Shared Buffer) is partitioned into 128-element lanes.
        # A head_dim of 128 means one attention head maps perfectly to one lane,
        # allowing the compiler to schedule the QKV computation without splits or
        # padding.  Any other value (e.g. 64 or 256) breaks the mapping and
        # degrades hardware utilisation.
        assert (
            self.head_dim == 128
        ), f"head_dim must be 128 (maps to Trn2 SBUF partition dim); got {self.head_dim}"

        # --- Consistency: d_model == n_heads * head_dim ---
        # The model dimension is simply the concatenation of all Q-head outputs.
        # If this does not hold, the output projection o_proj would have a wrong
        # input size and the forward pass would silently produce garbled shapes.
        assert (
            self.n_heads * self.head_dim == self.d_model
        ), f"n_heads({self.n_heads}) * head_dim({self.head_dim}) != d_model({self.d_model})"

        # --- GQA ratio must be a whole number ---
        # In GQA each Q head is assigned to exactly one KV head, so the ratio
        # n_heads / n_kv_heads must be an integer.  A fractional ratio would mean
        # some Q heads share a KV head while others do not, making the expand
        # trick in GroupedQueryAttention.forward impossible with a uniform reshape.
        assert (
            self.n_heads % self.n_kv_heads == 0
        ), f"n_heads({self.n_heads}) must be divisible by n_kv_heads({self.n_kv_heads})"

        # --- KV projection width is also tile-aligned ---
        # The K and V projections output (n_kv_heads * head_dim) features.
        # This width must also be a multiple of 128 for the same systolic-array
        # reasons as d_model above.  With head_dim=128 this is automatically
        # satisfied whenever n_kv_heads >= 1, but we check explicitly to guard
        # against future changes to head_dim.
        assert (
            self.n_kv_heads * self.head_dim
        ) % 128 == 0, f"n_kv_heads({self.n_kv_heads}) * head_dim({self.head_dim}) must be ÷128"

    @property
    def gqa_ratio(self) -> int:
        """Number of Q heads per KV head."""
        return self.n_heads // self.n_kv_heads

    def num_params(self) -> dict:
        """
        Compute parameter count analytically.
        Returns per-component breakdown and total.
        No model instantiation needed — useful for budget planning.
        """
        d = self.d_model
        h = self.head_dim
        nq = self.n_heads
        nkv = self.n_kv_heads
        fi = self.ffn_intermediate
        v = self.vocab_size
        L = self.n_layers

        # Attention projections per layer
        # Q: d_model → n_heads * head_dim = d_model
        # K: d_model → n_kv_heads * head_dim
        # V: d_model → n_kv_heads * head_dim
        # O: d_model → d_model
        q_proj = d * (nq * h)
        k_proj = d * (nkv * h)
        v_proj = d * (nkv * h)
        o_proj = (nq * h) * d
        # QK-Norm: learned scale γ per head (shape: [n_heads, head_dim] for Q, [n_kv_heads, head_dim] for K)
        qk_norm = (nq * h) + (nkv * h)
        attn_per_layer = q_proj + k_proj + v_proj + o_proj + qk_norm

        # SwiGLU FFN per layer: gate + up + down (no bias)
        # SwiGLU: output = (gate(x) * silu(up(x))) @ down
        # gate: d_model → ffn_intermediate
        # up:   d_model → ffn_intermediate
        # down: ffn_intermediate → d_model
        gate_proj = d * fi
        up_proj = d * fi
        down_proj = fi * d
        ffn_per_layer = gate_proj + up_proj + down_proj

        # RMSNorm per layer: 2 norms (pre-attn, pre-ffn), each has d_model scale params
        norm_per_layer = 2 * d

        per_layer = attn_per_layer + ffn_per_layer + norm_per_layer
        all_layers = per_layer * L

        # Final RMSNorm
        final_norm = d

        # Embedding table (not tied: counts twice; tied: counts once)
        embedding = v * d
        lm_head = 0 if self.tie_embeddings else v * d

        total = all_layers + final_norm + embedding + lm_head

        return {
            "embedding": embedding,
            "lm_head": lm_head,
            "per_layer": {
                "attention": attn_per_layer,
                "ffn": ffn_per_layer,
                "norms": norm_per_layer,
                "total": per_layer,
            },
            "all_layers": all_layers,
            "final_norm": final_norm,
            "total": total,
            "total_B": total / 1e9,
            "total_M": total / 1e6,
        }


# ---------------------------------------------------------------------------
# Canonical configs (from spec Section 1.2)
# ---------------------------------------------------------------------------
# All three configs obey the same invariants (tile alignment, head_dim=128).
# Scaling follows the Chinchilla depth-vs-width tradeoff: wider models are
# more sample-efficient but slower per token; deeper models are the reverse.
# These sizes were chosen to fit specific hardware targets (see module docstring).
#
# ffn_intermediate sizing:
#   Target ≈ (8/3) * d_model, rounded DOWN to the nearest multiple of 128.
#   500M: (8/3)*1280 ≈ 3413 → 3456 (next multiple of 128 above 3413 that fits budget)
#   1B:   (8/3)*2048 ≈ 5461 → 5504
#   3B:   (8/3)*3072 ≈ 8192 → 8192 (exact)
#
# gqa_ratio (n_heads / n_kv_heads):
#   500M: 10/2  = 5×  — aggressive KV reduction for the small GPU target
#   1B:   16/4  = 4×  — standard ratio for this scale
#   3B:   24/6  = 4×  — same ratio, larger absolute KV cache

CONFIGS = {
    # 500M: validation config, designed to fit on an RTX 5090 (24 GB).
    # d_model=1280 with 10 heads gives head_dim=128. GQA ratio=5 for aggressive
    # KV cache saving.  Shorter max_seq_len=8192 keeps activation memory manageable.
    "500m": ModelConfig(
        d_model=1280,
        n_layers=26,
        n_heads=10,
        n_kv_heads=2,
        head_dim=128,
        ffn_intermediate=3456,
        vocab_size=32768,
        max_seq_len=8192,
    ),
    # 1B: primary training config for AWS Trainium2.
    # 20 layers × 2048 hidden gives a compact but capable reasoning model.
    # max_seq_len=16384 accommodates long chain-of-thought traces.
    "1b": ModelConfig(
        d_model=2048,
        n_layers=20,
        n_heads=16,
        n_kv_heads=4,
        head_dim=128,
        ffn_intermediate=5504,
        vocab_size=32768,
        max_seq_len=16384,
    ),
    # 3B: full-scale experiment config for Trn2 multi-node or cloud H100.
    # ffn_intermediate=8192 is exactly (8/3)*3072, which is a lucky coincidence
    # of 3072 being cleanly divisible by 3 — no rounding needed.
    # max_seq_len=32768 for extended reasoning chains.
    "3b": ModelConfig(
        d_model=3072,
        n_layers=28,
        n_heads=24,
        n_kv_heads=6,
        head_dim=128,
        ffn_intermediate=8192,
        vocab_size=32768,
        max_seq_len=32768,
    ),
}


def get_config(name: str) -> ModelConfig:
    """Return a ModelConfig by name ('500m', '1b', '3b'). Raises KeyError on unknown names."""
    if name not in CONFIGS:
        raise KeyError(f"Unknown config '{name}'. Valid options: {list(CONFIGS.keys())}")
    return CONFIGS[name]


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (Zhang & Sennrich, 2019).

    Standard LayerNorm subtracts the mean before scaling (re-centering + re-scaling).
    RMSNorm skips the mean subtraction entirely, keeping only the RMS scaling step.
    This is valid because empirical work shows the re-centering step contributes
    little to training stability while adding compute overhead.

    Formula:
        RMSNorm(x) = x / RMS(x) * γ
        where RMS(x) = sqrt( mean(x²) + ε )
        and γ is a learned per-dimension scale parameter initialised to 1.

    Why RMS instead of standard deviation?
        std(x) = sqrt( mean((x - mean(x))²) )  — requires two passes over x
        RMS(x) = sqrt( mean(x²) )               — single pass, no centering
    The epsilon ε is added inside the sqrt (before taking the reciprocal) to
    prevent division by zero when x is very close to the zero vector.

    This is used for: pre-attention norm, pre-FFN norm, final norm, and (with a
    smaller dim) the QK-Norm applied per attention head.
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        # γ: learned per-dimension scale, initialised to 1 (identity transform at init)
        # Shape (dim,) — broadcast over all batch and sequence dimensions
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: any tensor whose last dimension is `dim`; typically
               (B, T, d_model) for token-level norms or
               (B, n_heads, T, head_dim) for QK-Norm.
        Returns:
            Normalised tensor of the same shape as x.
        """
        # Step 1: compute x² then average across the last dimension (the feature dim).
        #   mean(dim=-1, keepdim=True) keeps the shape (..., 1) so we can broadcast
        #   back against x without an explicit unsqueeze.  We MUST keepdim here
        #   because x has shape (..., dim) and the normaliser has shape (..., 1).
        # Step 2: add ε for numerical safety before taking the reciprocal square root.
        # Step 3: rsqrt(v) = 1/sqrt(v) — a fused CUDA instruction that is faster
        #   than separate sqrt + reciprocal; also avoids an intermediate allocation.
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()

        # Multiply x by the reciprocal RMS to normalise, then by γ to re-scale.
        # self.weight has shape (dim,) which broadcasts over all leading dims of x.
        return x * rms * self.weight

    def extra_repr(self) -> str:
        return f"dim={self.weight.shape[0]}, eps={self.eps}"


# ---------------------------------------------------------------------------
# Rotary Position Embedding (RoPE)
# ---------------------------------------------------------------------------


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) — Su et al., 2021 (RoFormer).

    Core idea
    ---------
    Rather than adding a position vector to token embeddings (absolute PE),
    RoPE *rotates* the query and key vectors in paired 2D subspaces of the
    head dimension.  The rotation angle of pair i at position p is:

        angle = p * θ_i,   where θ_i = 1 / (base ^ (2i / head_dim))

    Because of the rotation identity:
        dot( R(p) * q,  R(s) * k ) = dot( R(p-s) * q, k )

    the dot product between any Q at position p and any K at position s
    depends ONLY on the relative offset (p - s).  This gives the model
    relative position sensitivity without ever explicitly computing p - s.

    Why base=500_000 instead of the original 10_000?
    -------------------------------------------------
    The θ_i frequencies span a geometric sequence from θ_0 = 1/base^0 = 1
    (highest frequency, completes one full rotation per token) down to
    θ_{D/2-1} = 1/base (lowest frequency, completes one rotation over
    `base` tokens).  With base=10_000 the lowest-frequency dimension
    is still completing ~1.6 rotations over a 16k-token sequence, making
    it hard to distinguish positions that are far apart.  Raising the base
    to 500_000 (as in Llama 3) keeps all dimensions in a useful
    discriminative range for sequences up to ~32k.

    Implementation notes
    --------------------
    We precompute and cache cos/sin tables at init rather than recomputing
    them on every forward pass.  The tables are registered as non-persistent
    buffers so they move with the model to the right device but are NOT
    saved in checkpoints (they can always be recomputed from inv_freq).

    The rotation is implemented with real arithmetic (_rotate_half) rather
    than complex arithmetic (torch.view_as_complex) because:
      1. torch.view_as_complex requires contiguous float32 — fragile under
         AMP and bf16 training.
      2. TorchScript / torch.jit.trace exports cleanly without complex ops.
      3. The einsum-free formulation is trivially fused by XLA on Trainium.
    """

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 500_000.0):
        super().__init__()
        # RoPE works by splitting head_dim into D/2 pairs; requires even head_dim
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"

        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base

        # --- Precompute inverse frequencies ---
        # θ_i = 1 / (base ^ (2i / head_dim))  for i in 0, 1, ..., head_dim/2 - 1
        # torch.arange(0, head_dim, 2) produces [0, 2, 4, ..., head_dim-2],
        # i.e. the even indices, one per 2D subspace.
        # Dividing by head_dim normalises so the exponent runs from 0 to 1.
        # Non-persistent: will not be included in state_dict; recomputed on load.
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build and cache cos/sin tables immediately so they are on the right
        # device at construction time (avoids a device mismatch on first forward)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        """
        Precompute cos and sin rotation tables for positions [0, seq_len).

        Math
        ----
        For each position p and frequency index i:
            freqs[p, i] = p * θ_i                       (outer product)

        We then concatenate [freqs, freqs] along the last dimension to produce
        a (seq_len, head_dim) table.  The duplication is intentional: the full
        rotation formula for pair (x1, x2) is:

            x1' = x1 * cos(angle) - x2 * sin(angle)
            x2' = x1 * sin(angle) + x2 * cos(angle)

        Written in the _rotate_half form:
            x_rot = x * cos + rotate_half(x) * sin

        where rotate_half maps [x1...x_{D/2}, x_{D/2+1}...x_D] to
        [-x_{D/2+1}...-x_D, x1...x_{D/2}].  For this to broadcast correctly,
        both cos and sin must have the full head_dim width (i.e. D, not D/2),
        hence the cat([freqs, freqs]) duplication.

        The result is stored with a leading (1, 1, ...) batch so it broadcasts
        directly against (B, n_heads, T, head_dim) without manual unsqueezing.
        """
        positions = torch.arange(seq_len, device=self.inv_freq.device).float()

        # Outer product: (seq_len,) x (head_dim/2,) → (seq_len, head_dim/2)
        # freqs[p, i] = p * θ_i
        freqs = torch.outer(positions, self.inv_freq)

        # Duplicate across the feature dimension: (seq_len, head_dim/2) → (seq_len, head_dim)
        # The first half corresponds to the "x1" elements of each pair,
        # the second half to the "x2" elements — they share the same angle.
        emb = torch.cat([freqs, freqs], dim=-1)

        # Add leading (1, 1) dims so shape is (1, 1, seq_len, head_dim).
        # This broadcasts cleanly against (B, n_heads, T, head_dim).
        # Non-persistent: large tensors we do not want in checkpoints.
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :], persistent=False
        )  # (1, 1, seq_len, head_dim)
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :], persistent=False
        )  # (1, 1, seq_len, head_dim)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """
        Implement the 90° rotation needed for the real-valued RoPE formula.

        For a vector x = [x1, x2, x3, x4, ...] split into two halves:
            first  half: x1 = x[..., :D//2]   — "real" components
            second half: x2 = x[..., D//2:]    — "imaginary" components

        This function returns [-x2, x1], which is the same as rotating each
        2D pair (x1_i, x2_i) by 90° (i.e. multiplying by j in complex notation).

        Combined with the main rotation:
            x_rot = x * cos(θ) + rotate_half(x) * sin(θ)
                  = [x1*cos - x2*sin,   x2*cos + x1*sin]

        which is exactly the standard 2D rotation matrix applied to (x1, x2).
        """
        x1 = x[..., : x.shape[-1] // 2]  # first half of each head's features
        x2 = x[..., x.shape[-1] // 2 :]  # second half of each head's features
        # Swap and negate to achieve the 90° rotation: (-x2, x1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(
        self,
        q: torch.Tensor,  # (B, n_heads, T, head_dim)
        k: torch.Tensor,  # (B, n_kv_heads, T, head_dim)
        offset: int = 0,  # For KV-cache: absolute position of the first token in this batch
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary embeddings to Q and K.

        During training, offset=0 and T = full sequence length.
        During generation (one token at a time with KV cache), offset is the
        number of tokens already decoded, and T=1.  The offset ensures that
        the newly generated token is rotated by the correct absolute position
        angle rather than always treating itself as position 0.
        """
        T = q.shape[2]

        # Safety: extend the cache if a sequence longer than max_seq_len arrives
        # (rare, but can happen during evaluation on longer prompts)
        if offset + T > self.cos_cached.shape[2]:
            self._build_cache(offset + T)

        # Slice out only the positions we need: [offset, offset+T)
        cos = self.cos_cached[:, :, offset : offset + T, :]  # (1, 1, T, head_dim)
        sin = self.sin_cached[:, :, offset : offset + T, :]  # (1, 1, T, head_dim)

        # Apply rotation: x_rot = x * cos + rotate_half(x) * sin
        # Both cos and sin broadcast over the batch and head dimensions.
        q_rot = (q * cos) + (self._rotate_half(q) * sin)
        k_rot = (k * cos) + (self._rotate_half(k) * sin)

        return q_rot, k_rot


# ---------------------------------------------------------------------------
# Grouped Query Attention with QK-Norm
# ---------------------------------------------------------------------------


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA) with QK-Norm and RoPE.

    GQA (Ainslie et al., 2023)
    --------------------------
    Standard Multi-Head Attention (MHA) gives every head its own K and V
    matrices, so the KV cache grows as:
        cache_bytes = 2 * n_heads * head_dim * seq_len * layers * bytes_per_elem

    GQA partitions the n_heads Q heads into n_kv_heads groups.  All Q heads
    within a group share a single K head and a single V head.  The KV cache
    shrinks by a factor of gqa_ratio = n_heads / n_kv_heads.

    At 1B scale with gqa_ratio=4 and seq_len=16k:
        MHA cache ≈ 4 GB (bf16)     GQA cache ≈ 1 GB (bf16)
    This is the difference between fitting inference in GPU memory and not.

    Q quality is not significantly harmed because Q heads do not share weights —
    only the K/V projections are shared.

    QK-Norm
    -------
    Without normalisation, the raw Q and K projections have variance that grows
    with d_model.  The pre-softmax attention scores are Q @ K^T / sqrt(head_dim).
    If the magnitudes of Q and K are large (common early in training), many
    scores become very large positive or negative values, the softmax saturates
    to near-one-hot, and gradients through the attention weights vanish.

    QK-Norm applies RMSNorm to each head's Q and K slice *independently* and
    *before* RoPE.  This bounds the magnitudes to a small range at init and
    lets the learned γ parameters gradually increase them only as needed.

    Applying QK-Norm BEFORE RoPE is critical: RoPE is a norm-preserving
    rotation, so it does not undo the normalisation.  Applying QK-Norm after
    RoPE would discard the rotational position information.

    Ordering (strictly enforced):
        project → reshape to heads → QK-Norm → RoPE → GQA expand →
        scaled dot-product attention → merge heads → output project

    No bias in any projection: simplifies weight decay, avoids quantisation
    offset errors, and has no empirical downside for transformers of this size.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.gqa_ratio = config.gqa_ratio
        self.d_model = config.d_model

        # --- Input projections (no bias anywhere) ---
        # Q: maps d_model → n_heads * head_dim (= d_model, since n_heads * head_dim = d_model)
        self.q_proj = nn.Linear(config.d_model, config.n_heads * config.head_dim, bias=False)
        # K, V: maps d_model → n_kv_heads * head_dim (smaller than Q because GQA)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        # Output: maps concatenated head outputs back to d_model
        self.o_proj = nn.Linear(config.n_heads * config.head_dim, config.d_model, bias=False)

        # --- QK-Norm ---
        # Each norm operates on a single head slice (head_dim features), not
        # the full projection output.  After the view/transpose reshaping below,
        # the last dimension is head_dim, so RMSNorm(head_dim) applies correctly.
        # Separate γ weights for Q and K allow asymmetric scaling.
        self.q_norm = RMSNorm(config.head_dim, eps=config.norm_eps)
        self.k_norm = RMSNorm(config.head_dim, eps=config.norm_eps)

        # --- Rotary position embedding ---
        self.rope = RotaryEmbedding(
            head_dim=config.head_dim,
            max_seq_len=config.max_seq_len,
            base=config.rope_base,
        )

        self.dropout = config.dropout

        # Attention scale: 1 / sqrt(head_dim)
        # Without this factor the dot products grow in magnitude as O(sqrt(head_dim)),
        # pushing the softmax into saturation.  The 1/sqrt(d) normalisation keeps
        # the pre-softmax logits O(1) regardless of head_dim.
        self.scale = config.head_dim**-0.5

    def forward(
        self,
        x: torch.Tensor,  # (B, T, d_model)
        attention_mask: Optional[torch.Tensor],  # (B, 1, T, T) or None
        kv_cache: Optional[tuple] = None,  # (k_cache, v_cache) for generation decode steps
        position_offset: int = 0,
        collect_kv: bool = False,  # True during prefill to capture KV for the decode cache
    ) -> tuple[torch.Tensor, Optional[tuple]]:

        B, T, _ = x.shape

        # --- Project ---
        q = self.q_proj(x)  # (B, T, n_heads * head_dim)
        k = self.k_proj(x)  # (B, T, n_kv_heads * head_dim)
        v = self.v_proj(x)  # (B, T, n_kv_heads * head_dim)

        # Reshape flat projections into per-head slices, then move the head
        # dimension before the sequence dimension.
        # view splits the last dim into (n_heads, head_dim) pairs.
        # transpose(1, 2) swaps dim-1 (T) and dim-2 (n_heads) so the layout
        # is (B, n_heads, T, head_dim) — the canonical shape for batched GEMM.
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # --- QK-Norm (applied per head, before RoPE) ---
        # After the reshape above the last dimension is head_dim, so
        # RMSNorm(head_dim) applies independently to each (batch, head, token)
        # position without needing a further reshape.
        q = self.q_norm(q)  # (B, n_heads,    T, head_dim)
        k = self.k_norm(k)  # (B, n_kv_heads, T, head_dim)

        # --- RoPE (after QK-Norm, before attention) ---
        # RoPE is a norm-preserving rotation so it does not interfere with
        # QK-Norm's magnitude bounds.  position_offset is 0 during training
        # and equal to the number of cached tokens during generation.
        q, k = self.rope(q, k, offset=position_offset)

        # --- KV cache: prepend cached keys/values (decode steps only) ---
        # During generation we process one new token at a time (T=1), but the
        # attention must see all past tokens.  The KV cache stores the already-
        # computed keys and values from previous steps.  We simply concatenate
        # them along the sequence dimension (dim=2) to reconstruct the full
        # context before computing attention.
        # The new (k, v) pair — including the current token — is returned so
        # the caller can store it for the next decode step.
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)  # (B, n_kv_heads, past+T, head_dim)
            v = torch.cat([v_cache, v], dim=2)  # (B, n_kv_heads, past+T, head_dim)
        # Return KV when (a) updating an existing decode cache, or (b) collect_kv=True
        # (prefill mode — saves the full prompt KV so decode step 1 has context).
        new_kv_cache = (k, v) if (kv_cache is not None or collect_kv) else None

        # --- GQA: expand KV heads to match Q heads ---
        # After prepending the cache, k and v have n_kv_heads heads but q has
        # n_heads heads.  We need to "broadcast" each KV head to gqa_ratio Q heads.
        #
        # Step 1: unsqueeze(2) inserts a new dim after the head dim:
        #   (B, n_kv_heads, S, head_dim) → (B, n_kv_heads, 1, S, head_dim)
        #
        # Step 2: expand(..., gqa_ratio, ...) tiles that new dim gqa_ratio times:
        #   → (B, n_kv_heads, gqa_ratio, S, head_dim)
        #   IMPORTANT: expand is zero-copy — it creates a view with a stride of 0
        #   along the new dimension, so no memory is duplicated.  The data for each
        #   KV head is physically shared across all gqa_ratio Q heads that use it.
        #
        # Step 3: reshape merges dims 1 and 2 to get the standard (B, n_heads, S, D):
        #   → (B, n_kv_heads * gqa_ratio, S, head_dim) = (B, n_heads, S, head_dim)
        #   reshape makes the tensor contiguous (allocates new memory) which is
        #   required because scaled_dot_product_attention needs contiguous input.
        k = k.unsqueeze(2).expand(-1, -1, self.gqa_ratio, -1, -1)
        k = k.reshape(B, self.n_heads, -1, self.head_dim)
        v = v.unsqueeze(2).expand(-1, -1, self.gqa_ratio, -1, -1)
        v = v.reshape(B, self.n_heads, -1, self.head_dim)

        # --- Scaled dot-product attention ---
        # Prefer F.scaled_dot_product_attention (PyTorch ≥ 2.0) which dispatches
        # to FlashAttention-2 on CUDA and avoids materialising the full (T, T)
        # score matrix — O(T) memory instead of O(T²).
        # On Trainium the NKI kernel replaces this call at compile time; the
        # manual fallback is a reference implementation only.
        if hasattr(F, "scaled_dot_product_attention"):
            # When attention_mask is None AND is_causal=True, SDPA generates the
            # causal mask internally and fuses it with the softmax — no explicit
            # (T, T) mask tensor is allocated.  When a mask IS provided (training
            # with padding), is_causal must be False to avoid double-masking.
            #
            # CRITICAL: is_causal must be False during KV-cache decode steps.
            # When T_q=1 and T_k=T_cache+1, PyTorch's is_causal=True creates a
            # (1, T_k) lower-triangular mask that only unmasks column 0 — the query
            # can only attend to the very first token, destroying all context.
            # In decode mode every cached K/V is already from a prior position
            # (causality is structurally guaranteed), so no mask is needed.
            dropout_p = self.dropout if self.training else 0.0
            is_decode = kv_cache is not None  # True only during KV-cache decode steps
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attention_mask,
                dropout_p=dropout_p,
                is_causal=(attention_mask is None and not is_decode),
                scale=self.scale,
            )
        else:
            # Manual fallback: explicit O(T²) attention matrix.
            # Used on Trainium before the NKI attention kernel is installed,
            # and on CPU for debugging.
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, T, S)
            if attention_mask is not None:
                # Additive mask: 0 for valid positions, -inf for masked positions.
                # Adding -inf causes those positions to become 0 after softmax.
                scores = scores + attention_mask
            elif kv_cache is None:
                # Training / prefill: build a (T_q, T_q) causal lower-triangular mask.
                # In decode mode (kv_cache is not None), all cached positions are already
                # causally prior — no mask needed and T_q != T_k so .tril() would be wrong.
                causal = torch.ones(T, T, device=x.device, dtype=torch.bool).tril()
                scores = scores.masked_fill(~causal, float("-inf"))
            scores = F.softmax(scores, dim=-1)
            if self.dropout > 0 and self.training:
                scores = F.dropout(scores, p=self.dropout)
            out = torch.matmul(scores, v)  # (B, H, T, head_dim)

        # --- Merge heads and project back to d_model ---
        # transpose(1, 2) restores (B, T, n_heads, head_dim).
        # contiguous() is required before view because transpose creates a
        # non-contiguous tensor and view needs contiguous memory layout.
        # view merges the head dimensions: n_heads * head_dim = d_model.
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        # Final linear mix of head outputs — allows cross-head information to
        # combine before the residual addition.
        out = self.o_proj(out)

        return out, new_kv_cache


# ---------------------------------------------------------------------------
# SwiGLU Feed-Forward Network
# ---------------------------------------------------------------------------


class SwiGLUFFN(nn.Module):
    """
    SwiGLU Feed-Forward Network (Noam Shazeer, 2020).

    Formula:
        gate  = SiLU( W_gate * x )   — soft gating signal
        up    =       W_up   * x     — content signal (values to pass through)
        FFN(x) = W_down * (gate ⊙ up)

    where ⊙ is element-wise multiplication and SiLU(z) = z * sigmoid(z).

    Why SwiGLU vs a plain two-layer FFN?
    -------------------------------------
    A standard FFN is:  FFN(x) = W2 * GeLU(W1 * x)
    SwiGLU introduces a third matrix W_gate that learns WHICH dimensions of
    the intermediate representation to let through.  This is a "conditional
    computation" mechanism: the gate can suppress noisy or irrelevant features
    before the down projection, giving the network more expressive power per
    parameter at training cost of one extra matmul.

    Why 2/3 * 4 * d_model intermediate size?
    ------------------------------------------
    A two-matrix FFN (W1, W2) uses 2 * d_model * ffn_intermediate parameters.
    A three-matrix SwiGLU uses 3 * d_model * ffn_intermediate parameters.
    To keep total FFN parameters equal, we shrink the intermediate dimension
    by 2/3, giving ffn_intermediate ≈ (2/3) * 4 * d_model = 8/3 * d_model.
    The config's ffn_intermediate is pre-rounded to the nearest multiple of
    128 for tile alignment, which is why the exact values look irregular
    (e.g. 5504 instead of 5461.3 for d_model=2048).

    No bias. No dropout (regularisation is handled upstream if needed).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        # gate and up have identical shape — both project d_model → ffn_intermediate.
        # They are kept as separate modules (not a single fused matrix) to make
        # the computation graph explicit and to allow separate weight init.
        self.gate_proj = nn.Linear(config.d_model, config.ffn_intermediate, bias=False)
        self.up_proj = nn.Linear(config.d_model, config.ffn_intermediate, bias=False)
        # down_proj projects the gated intermediate back to d_model to close the
        # residual loop.  It receives special (smaller) weight initialisation —
        # see _init_weights for the rationale.
        self.down_proj = nn.Linear(config.ffn_intermediate, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        # gate_proj and up_proj are independent linear transforms of the same input.
        # SiLU (Sigmoid Linear Unit): silu(z) = z * sigmoid(z)
        #   — smooth, non-monotonic gating function; empirically outperforms ReLU
        #     and GeLU in this gated architecture.
        gate = F.silu(self.gate_proj(x))  # (B, T, ffn_intermediate): soft gate in [0, ~1]
        up = self.up_proj(x)  # (B, T, ffn_intermediate): content to gate
        # Element-wise multiply: gate zeroes out "unwanted" dimensions of `up`
        # before the down projection collapses back to d_model.
        return self.down_proj(gate * up)  # (B, T, d_model)


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------


class TransformerBlock(nn.Module):
    """
    Single transformer decoder block.

    Layout (pre-norm):
      x = x + Attention(RMSNorm(x))
      x = x + FFN(RMSNorm(x))

    Pre-norm places normalization *before* each sublayer, inside the residual.
    This keeps gradient magnitudes stable through the depth of the network —
    the residual stream carries the full signal and gradients flow cleanly.

    Post-norm (OLMo style) normalizes *after* adding the residual.
    It can achieve slightly better final quality but is harder to train
    at small scale due to early-layer gradient explosion risk.
    We use pre-norm as the safer choice for a new training run.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.attn = GroupedQueryAttention(config)
        self.ffn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.ffn = SwiGLUFFN(config)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[tuple] = None,
        position_offset: int = 0,
        collect_kv: bool = False,
    ) -> tuple[torch.Tensor, Optional[tuple]]:

        # --- Attention sub-block (pre-norm + residual) ---
        # self.attn_norm(x) normalises x before passing into attention;
        # the *original* x is added back as the residual.  This is the
        # defining characteristic of pre-norm: the normalised version goes
        # through the sublayer, but the unnormalised residual stream continues.
        # Gradient flow: d(loss)/d(x_in) = d(loss)/d(x_out) * (1 + d(attn)/d(x_in))
        # The "+1" ensures gradients flow directly to earlier layers without
        # being gated by the sublayer Jacobian.
        attn_out, new_kv_cache = self.attn(
            self.attn_norm(x),
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            position_offset=position_offset,
            collect_kv=collect_kv,
        )
        x = x + attn_out  # residual addition: maintains the input signal

        # --- FFN sub-block (pre-norm + residual) ---
        # Same pre-norm + residual pattern as attention.
        # The FFN norm is independent of attn_norm — separate learned γ weights.
        x = x + self.ffn(self.ffn_norm(x))  # residual addition

        return x, new_kv_cache


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------


class SmallReasoningModel(nn.Module):
    """
    Small reasoning model — full decoder-only transformer.

    Architecture:
      Embedding → L × TransformerBlock → RMSNorm → LM Head

    Tied embeddings: the LM head reuses the embedding weight matrix transposed.
    This halves the embedding parameter count (vocab_size × d_model) and
    has been shown to improve sample efficiency, especially at small scale.

    Special token IDs (must match tokenizer):
      <pad>     = 0   (padding_idx — excluded from loss)
      <bos>     = 1
      <eos>     = 2
      <think>   = 4
      </think>  = 5
    """

    PAD_TOKEN_ID = 0

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.embedding = nn.Embedding(
            config.vocab_size,
            config.d_model,
            padding_idx=self.PAD_TOKEN_ID,
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])

        # Final norm
        self.norm = RMSNorm(config.d_model, eps=config.norm_eps)

        # LM head — optionally tied to embedding
        if config.tie_embeddings:
            # Tied: LM head IS the embedding matrix (transposed)
            # No new parameters. Saves vocab_size * d_model params.
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight initialization
        self._init_weights()

    def _init_weights(self):
        """
        Initialise weights following the GPT-2 / Llama convention with
        residual-stream variance control.

        General principle
        -----------------
        We want activations to have roughly unit variance at init so that
        neither gradients nor activations explode or vanish through depth.
        A standard choice is N(0, 0.02) for all weights — empirically this
        keeps activations in a reasonable range for d_model ≈ 1k-4k.

        Residual scaling (the "GPT-2 trick", Radford et al. 2019)
        ----------------------------------------------------------
        Each transformer block adds its attention output and FFN output into
        the residual stream:
            x_{l+1} = x_l + f_attn(x_l) + f_ffn(x_l)

        If every layer adds an independent N(0, σ²) contribution, after L
        layers the variance of the residual stream grows as L * σ².  For a
        20-layer model this is 20× the single-layer variance — enough to
        cause numerical issues in bf16.

        To neutralise this growth, the *output* projections of each sublayer
        (the ones that write INTO the residual stream) are initialised with
        a smaller standard deviation:

            residual_std = 0.02 / sqrt(2 * n_layers)

        The factor of 2 accounts for two sublayers per block (attention and FFN).
        Each block then contributes variance σ² / (2L), and the sum over 2L
        contributions gives total variance σ² ≈ 0.02² — constant regardless
        of depth.  Critically, this is an INIT-time trick; the optimizer will
        move weights away from these values quickly, but it prevents divergence
        in the first few hundred steps.

        The output projections are:
          - o_proj:   the final linear in GroupedQueryAttention (writes to residual)
          - down_proj: the final linear in SwiGLUFFN (writes to residual)
        All other linear weights (q, k, v, gate, up projections) use std=0.02.

        RMSNorm γ weights: initialised to 1 so the norm is identity at init.
        The optimizer will learn the appropriate scale.

        Padding embedding row: zeroed out so the model never "learns" a
        representation for the special PAD token that leaks into computations.
        """
        # Base standard deviation — chosen to match GPT-2 / Llama; small enough
        # that random projections don't saturate activations at init
        std = 0.02

        # Reduced std for projections that write directly into the residual stream.
        # Formula: std / sqrt(2 * n_layers)
        #   - dividing by sqrt(2 * n_layers) compensates for the variance accumulation
        #     described above; the "2" covers both the attention and FFN sublayers.
        residual_scale = std / math.sqrt(2 * self.config.n_layers)

        for name, module in self.named_modules():
            if isinstance(module, nn.Embedding):
                # Embedding rows are token representations at position 0 (before any
                # layers).  N(0, 0.02) keeps initial token norms modest.
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.padding_idx is not None:
                    # PAD token must produce a zero vector so that padded positions
                    # contribute nothing to attention keys or values.
                    module.weight.data[module.padding_idx].zero_()

            elif isinstance(module, nn.Linear):
                # Identify output projections by their name suffix.
                # o_proj ends every GroupedQueryAttention; down_proj ends every SwiGLUFFN.
                # Both feed directly into the residual addition — apply the smaller std.
                if name.endswith(("o_proj", "down_proj")):
                    nn.init.normal_(module.weight, mean=0.0, std=residual_scale)
                else:
                    # All other projections: q, k, v, gate_proj, up_proj, lm_head
                    nn.init.normal_(module.weight, mean=0.0, std=std)
                # We have no bias parameters (bias=False everywhere), but handle
                # the case defensively in case a future layer adds bias.
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, RMSNorm):
                # γ = 1: identity transform at init; the network can scale up from here.
                # Initialising to 0 would zero out gradients; >1 would pre-amplify noise.
                nn.init.ones_(module.weight)

    def forward(
        self,
        input_ids: torch.Tensor,  # (B, T) — token IDs
        attention_mask: Optional[torch.Tensor] = None,  # (B, T) — 1=attend, 0=mask (padding)
        kv_caches: Optional[list] = None,  # list of (k, v) per layer, for generation
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, Optional[list]]:
        """
        Forward pass.

        Returns:
          logits: (B, T, vocab_size) — raw unnormalized scores
          new_kv_caches: updated KV caches (None during training)
        """
        B, T = input_ids.shape

        # --- Embedding ---
        # Look up the d_model-dimensional vector for each token ID.
        # The padding row (token 0) was zeroed in _init_weights so pad tokens
        # enter the network as the zero vector rather than a random embedding.
        x = self.embedding(input_ids)  # (B, T, d_model)

        # --- Build attention mask (training only) ---
        # During training with variable-length sequences the batch is padded to
        # the same length.  We need a mask that is BOTH causal (no future leakage)
        # AND excludes padded key positions.  _build_additive_mask fuses both.
        #
        # During generation we do NOT build a mask here because:
        #   (a) kv_caches is not None, so we're in decode mode (T=1, no padding)
        #   (b) F.scaled_dot_product_attention handles causality via is_causal=True
        #       when attn_mask=None
        attn_mask = None
        if attention_mask is not None and kv_caches is None:
            # attention_mask: (B, T), value 1 for real tokens, 0 for pad
            # Result: (B, 1, T, T) additive float mask; 0 = attend, -inf = block
            attn_mask = _build_additive_mask(attention_mask, x.dtype, x.device)

        # --- Transformer blocks ---
        # Accumulate updated KV caches when kv_caches is not None (generation mode).
        # kv_caches=[]  → prefill: collect fresh KV caches, no prior values to prepend.
        # kv_caches=[…] → decode: prepend prior cached K/V then append new token's K/V.
        # kv_caches=None → training: skip all caching overhead.
        new_kv_caches = [] if kv_caches is not None else None
        # collect_kv=True tells each attention layer to return its (k, v) tensors even
        # when there is no prior cache to prepend (i.e., during prefill with kv_caches=[]).
        collect_kv = kv_caches is not None and len(kv_caches) == 0

        for i, block in enumerate(self.blocks):
            # Retrieve the per-layer cache (None during prefill or training)
            kv = kv_caches[i] if (kv_caches is not None and len(kv_caches) > i) else None
            x, new_kv = block(
                x,
                attention_mask=attn_mask,
                kv_cache=kv,
                position_offset=position_offset,
                collect_kv=collect_kv,
            )
            # Store the updated cache for this layer so it can be returned to the caller
            if new_kv_caches is not None:
                new_kv_caches.append(new_kv)

        # --- Final RMSNorm ---
        # Applied once after the last transformer block.  Pre-norm architectures
        # do not normalise the residual stream after the last block's residual
        # addition, so this final norm brings the output into a stable range
        # before the vocabulary projection.
        x = self.norm(x)  # (B, T, d_model)

        # --- LM head: project to vocabulary logits ---
        if self.config.tie_embeddings:
            # Tied embeddings: the LM head IS the embedding matrix, transposed.
            # embedding.weight has shape (vocab_size, d_model).
            # F.linear(x, W) computes x @ W^T, effectively treating each row of
            # W as a "template" vector and scoring how much x matches each token.
            # Using F.linear avoids allocating a separate weight tensor — the
            # embedding and LM head share the EXACT same storage in memory.
            logits = F.linear(x, self.embedding.weight)  # (B, T, vocab_size)
        else:
            # Untied: separate weight matrix for the LM head
            logits = self.lm_head(x)  # (B, T, vocab_size)

        return logits, new_kv_caches

    def num_params(self) -> int:
        """Total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_params_breakdown(self) -> dict:
        """Parameter count by component."""
        return self.config.num_params()

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,  # (B, T_prompt)
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.9,
        eos_token_id: int = 2,
    ) -> torch.Tensor:
        """
        Autoregressive token generation with KV caching and nucleus sampling.

        Two-phase structure
        -------------------
        Phase 1 — Prefill:
            Process the entire prompt in a single forward pass.  All T_prompt
            tokens are computed in parallel (the attention mask handles causality).
            The KV cache is populated for all prompt positions in one shot.
            This is the expensive phase: O(T_prompt²) attention cost.

        Phase 2 — Decode:
            For each new token, run a forward pass with only the ONE new token
            as input (T=1).  The KV cache provides the keys and values for all
            preceding tokens, so attention still sees the full context without
            recomputing anything.  Cost per step: O(T_current) instead of
            O(T_current²).

        KV cache memory layout
        ----------------------
        kv_caches is a list of length n_layers.  Each element is a tuple
        (k_cache, v_cache) where:
            k_cache: (B, n_kv_heads, current_len, head_dim)
            v_cache: (B, n_kv_heads, current_len, head_dim)
        On each decode step, the new K and V for that token are concatenated
        onto the cache tensors inside GroupedQueryAttention.forward, and the
        updated cache is returned and stored here.

        position_offset
        ---------------
        Tracks how many tokens have been processed so that RoPE applies the
        correct absolute position angles.  After prefill it equals T_prompt;
        after each decode step it increments by 1.

        Stopping condition
        ------------------
        Generation stops when ALL sequences in the batch have produced an EOS
        token (or when max_new_tokens is reached).  Checking .all() rather
        than .any() ensures the returned tensor has the same length for all
        batch elements — simplifying downstream processing at the cost of
        generating a few extra tokens for already-finished sequences.

        For production serving use a dedicated inference engine (llama.cpp,
        vLLM, TGI) which handles more advanced batching and stopping logic.
        """
        self.eval()  # disable dropout for deterministic generation
        B = input_ids.shape[0]
        device = input_ids.device

        # --- Phase 1: Prefill ---
        # Process the full prompt in one forward pass to populate the KV cache.
        # kv_caches=[] signals "collect KV but no prior cache to prepend" — the
        # returned kv_caches will hold the per-layer (K, V) tensors for the prompt,
        # ready to be prepended during decode step 1.
        logits, kv_caches = self.forward(input_ids, kv_caches=[])
        # We only need the logits for the LAST prompt token — that prediction
        # gives the distribution for the first generated token.
        next_token_logits = logits[:, -1, :]  # (B, vocab_size)

        generated = input_ids  # will accumulate all tokens (prompt + generated)
        # After processing T_prompt tokens, the next token is at position T_prompt.
        position_offset = input_ids.shape[1]

        # --- Phase 2: Decode loop ---
        for _ in range(max_new_tokens):
            # Sample the next token from the current logit distribution.
            # Temperature scales the logits; top_p truncates the vocabulary to the
            # smallest set whose cumulative probability exceeds top_p.
            next_token = _sample(next_token_logits, temperature, top_p)  # (B, 1)

            # Append the new token to the running sequence
            generated = torch.cat([generated, next_token], dim=1)  # (B, T_so_far+1)

            # Early exit: if every sequence in the batch has produced EOS, stop.
            # Using .all() rather than .any() keeps the batch synchronised.
            if (next_token == eos_token_id).all():
                break

            # Single-token decode step: T=1, but the KV cache carries the full history.
            # position_offset tells RoPE to treat this token as being at the correct
            # absolute position in the sequence.
            logits, kv_caches = self.forward(
                next_token,  # (B, 1) — only the new token
                kv_caches=kv_caches,  # full accumulated KV cache from all prior steps
                position_offset=position_offset,
            )
            # Again take only the last (and only) time-step's logits
            next_token_logits = logits[:, -1, :]  # (B, vocab_size)
            # Advance position for the next token
            position_offset += 1

        return generated  # (B, T_prompt + n_generated)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_additive_mask(
    padding_mask: torch.Tensor,  # (B, T), 1=real token, 0=pad
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """
    Build a combined causal + padding additive attention mask for training.

    Why additive instead of boolean?
    ---------------------------------
    PyTorch's scaled_dot_product_attention and the manual fallback both accept
    an *additive* mask that is added directly to the pre-softmax attention
    scores.  Positions that should be masked receive -inf so that after
    softmax they become exactly 0 (exp(-inf) = 0).  Using -inf rather than
    a large negative number (e.g. -1e9) is numerically important: -1e9 in
    float16/bfloat16 can still produce a non-zero softmax value because the
    dynamic range of those formats is limited, whereas -inf reliably underflows
    to 0 in all floating-point formats.

    Two types of masking are combined here:
    1. Causal mask: token at position i may only attend to positions j ≤ i.
       Represented by the lower-triangular boolean matrix.
    2. Padding mask: real tokens (value=1) may only attend to other real tokens.
       A padded position (value=0) should not be attended to from ANY position,
       including from itself.  The mask blocks padding columns by broadcasting
       the (B, 1, 1, T) padding vector across all query positions.

    The combined rule: attend iff (j ≤ i) AND (token j is not padding).

    Output:
        (B, 1, T, T) float tensor with dtype matching the model activations.
        The leading (1) head dimension broadcasts over all attention heads.
        Value 0.0 at positions that are allowed; -inf at positions that are masked.
    """
    B, T = padding_mask.shape

    # --- Causal mask ---
    # torch.ones(T, T).tril() creates a T×T lower-triangular matrix of 1s.
    # As a bool tensor: True where attention is PERMITTED (j ≤ i).
    # Shape: (T, T)
    causal = torch.ones(T, T, device=device, dtype=torch.bool).tril()

    # --- Padding mask ---
    # padding_mask is (B, T): value 1 for real tokens, 0 for pad tokens.
    # We want to block attention TO padded columns (the KEY side), not from them,
    # so we broadcast as (B, 1, 1, T) — the last dim indexes the KEY position.
    # Unsqueezing twice: (B, T) → (B, 1, 1, T) for broadcasting against (B, 1, T, T).
    pad = padding_mask[:, None, None, :].bool()  # (B, 1, 1, T)

    # --- Combine: logical AND ---
    # causal (T, T) broadcasts over B; pad (B, 1, 1, T) broadcasts over the query dim.
    # combined[b, 0, i, j] = True iff (j ≤ i) AND (token j is a real token in batch b)
    combined = causal.unsqueeze(0) & pad  # (B, 1, T, T)

    # --- Convert boolean mask to additive float mask ---
    # Start from all-zeros (allow everything), then write -inf where combined is False.
    # masked_fill_ is an in-place operation — avoids allocating a second tensor.
    # dtype must match the model's activation dtype (float32 / bfloat16) to prevent
    # an implicit cast inside the attention kernel.
    additive = torch.zeros(B, 1, T, T, dtype=dtype, device=device)
    additive.masked_fill_(~combined, float("-inf"))  # -inf at masked positions
    return additive


def _sample(
    logits: torch.Tensor,  # (B, vocab_size)
    temperature: float,
    top_p: float,
) -> torch.Tensor:
    """
    Nucleus (top-p) sampling with temperature scaling.

    Temperature
    -----------
    Dividing logits by temperature T before softmax reshapes the distribution:
      - T < 1: distribution sharpens (lower-probability tokens suppressed further)
      - T = 1: distribution unchanged (standard sampling)
      - T > 1: distribution flattens (more uniform, more random)
    T = 0 is the degenerate greedy case — return the argmax directly without
    calling softmax (avoids division-by-zero and is exact).

    Nucleus / top-p sampling (Holtzman et al., 2020)
    -------------------------------------------------
    Instead of always sampling from all vocab_size tokens (which can draw from
    the long tail of very unlikely tokens and produce incoherent text), we
    truncate the distribution to the smallest "nucleus" of tokens whose
    cumulative probability mass meets or exceeds top_p.

    Algorithm:
      1. Sort tokens by probability (descending).
      2. Compute the running cumulative sum.
      3. Keep tokens up to (and including) the first token that pushes the
         cumulative sum past top_p.
      4. Zero out the remaining tokens, renormalise, and sample.

    The shift trick `(cumulative - sorted_probs) > top_p`:
      Without the shift, we would remove the token that FIRST crosses top_p,
      which could exclude the highest-probability token in an extreme case.
      Subtracting sorted_probs before comparing effectively checks whether the
      cumulative sum *before adding this token* already exceeds top_p.  This
      guarantees that at least the single most-probable token is always included,
      preventing a divide-by-zero when renormalising.
    """
    # Greedy decoding shortcut: avoids softmax overflow for temperature=0
    if temperature == 0.0:
        return logits.argmax(dim=-1, keepdim=True)  # (B, 1)

    # Temperature scaling: adjust sharpness of the distribution
    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)  # (B, vocab_size), sums to 1

    # --- Top-p nucleus construction ---
    # Sort tokens from most to least probable within each batch element
    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)

    # Running cumulative probability: cumulative[b, i] = sum of top-i probs for batch b
    cumulative = sorted_probs.cumsum(dim=-1)  # (B, vocab_size)

    # Mark positions to REMOVE: those where the cumulative sum (before this token)
    # already exceeds top_p.  Shifting by subtracting sorted_probs ensures the
    # token that first reaches top_p is kept (not discarded).
    remove = (cumulative - sorted_probs) > top_p  # (B, vocab_size) bool

    # Zero out removed tokens, then renormalise to a valid probability distribution
    sorted_probs[remove] = 0.0
    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)  # keepdim for broadcast

    # Draw one sample per batch element from the filtered distribution
    sampled = torch.multinomial(sorted_probs, num_samples=1)  # (B, 1) index in SORTED order

    # Map back from sorted indices to the original vocabulary indices
    return sorted_idx.gather(-1, sampled)  # (B, 1) original vocab token ID


# ---------------------------------------------------------------------------
# Cross-entropy loss (with ignore_index for padding)
# ---------------------------------------------------------------------------


def compute_loss(
    logits: torch.Tensor,  # (B, T, vocab_size)
    targets: torch.Tensor,  # (B, T)
    ignore_index: int = 0,  # PAD_TOKEN_ID — these positions are excluded from the loss
) -> torch.Tensor:
    """
    Causal language modelling loss: cross-entropy with next-token prediction.

    The shift
    ---------
    At each position t the model produces logits for the distribution over the
    NEXT token.  So logits[:, t, :] should predict targets[:, t+1].
    We implement this by:
      - dropping the LAST logit position (it has no corresponding next token)
      - dropping the FIRST target token (it is the BOS / first context token,
        not a prediction target)
    After the shift both tensors have length T-1.

    Padding exclusion
    -----------------
    Padding tokens (token ID 0) are present in targets to make batches
    rectangular.  Including them in the loss would cause the model to learn to
    predict padding, wasting capacity and distorting the loss scale.
    F.cross_entropy with ignore_index=0 skips those positions entirely and
    computes the mean only over real token positions.

    Why mean reduction?
    -------------------
    "mean" divides the summed cross-entropy by the number of NON-ignored tokens,
    so the loss scale is independent of batch size, sequence length, and padding
    ratio.  This makes learning rate schedules transferable across different
    sequence lengths and batching strategies.

    Returns:
        Scalar tensor: mean NLL loss over all non-padding next-token positions.
    """
    # --- Shift logits and targets to align predictions with targets ---
    # shift_logits[b, t, :] = distribution predicted from context tokens 0..t
    # shift_targets[b, t]   = the actual token at position t+1 (what we want to predict)
    shift_logits = logits[:, :-1, :].contiguous()  # (B, T-1, vocab_size)
    shift_targets = targets[:, 1:].contiguous()  # (B, T-1)

    # .contiguous() ensures the tensors are laid out sequentially in memory after the
    # slice, which is required by view() below (non-contiguous tensors cannot be viewed).

    # --- Flatten batch and sequence dims for cross_entropy ---
    # F.cross_entropy expects (N, C) logits and (N,) targets.
    # We flatten B and T-1 into a single N = B*(T-1) dimension.
    B, T, V = shift_logits.shape
    loss = F.cross_entropy(
        shift_logits.view(B * T, V),  # (B*(T-1), vocab_size)
        shift_targets.view(B * T),  # (B*(T-1),)
        ignore_index=ignore_index,  # skip PAD tokens (token ID 0)
        reduction="mean",  # mean over non-ignored positions
    )
    return loss  # scalar
