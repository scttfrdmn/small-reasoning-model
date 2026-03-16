"""
architecture.py
===============
Small reasoning model — core transformer architecture.

Implements the consensus stack from the spec:
  Pre-norm RMSNorm → GQA (with QK-Norm + RoPE) → SwiGLU FFN
  Tied input/output embeddings. No bias terms anywhere.

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
    d_model:          int = 2048   # Hidden dimension — must be multiple of 128
    n_layers:         int = 20     # Number of transformer blocks
    n_heads:          int = 16     # Number of query heads
    n_kv_heads:       int = 4      # Number of key/value heads (GQA)
    head_dim:         int = 128    # Dimension per head — FIXED at 128 for tile alignment
    ffn_intermediate: int = 5504   # SwiGLU intermediate dimension — must be multiple of 128

    # Vocabulary
    vocab_size:       int = 32768  # Tile-aligned: 32768 / 128 = 256

    # Sequence
    max_seq_len:      int = 16384  # Maximum sequence length

    # RoPE
    rope_base:        float = 500_000.0  # Base frequency — high value for long CoT sequences

    # Regularization
    dropout:          float = 0.0  # Set > 0 only for ablation studies; 0 is standard

    # Misc
    tie_embeddings:   bool  = True   # Tie input embedding ↔ LM head (saves vocab×d_model params)
    norm_eps:         float = 1e-5   # RMSNorm epsilon

    def __post_init__(self):
        """Enforce all spec invariants. Fail loudly — silent misalignment is worse."""

        # Tile alignment — Trainium2 NeuronCore systolic array
        assert self.d_model % 128 == 0, \
            f"d_model={self.d_model} must be ÷128 for tile alignment"
        assert self.ffn_intermediate % 128 == 0, \
            f"ffn_intermediate={self.ffn_intermediate} must be ÷128 for tile alignment"
        assert self.vocab_size % 128 == 0, \
            f"vocab_size={self.vocab_size} must be ÷128 for tile alignment"

        # Head dimension must be exactly 128
        # This maps the attention tile directly to the Trn2 SBUF partition dimension
        assert self.head_dim == 128, \
            f"head_dim must be 128 (maps to Trn2 SBUF partition dim); got {self.head_dim}"

        # Consistency: d_model == n_heads * head_dim
        assert self.n_heads * self.head_dim == self.d_model, \
            f"n_heads({self.n_heads}) * head_dim({self.head_dim}) != d_model({self.d_model})"

        # GQA ratio must be integer
        assert self.n_heads % self.n_kv_heads == 0, \
            f"n_heads({self.n_heads}) must be divisible by n_kv_heads({self.n_kv_heads})"

        # KV heads also tile-aligned
        assert (self.n_kv_heads * self.head_dim) % 128 == 0, \
            f"n_kv_heads({self.n_kv_heads}) * head_dim({self.head_dim}) must be ÷128"

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
        q_proj    = d * (nq * h)
        k_proj    = d * (nkv * h)
        v_proj    = d * (nkv * h)
        o_proj    = (nq * h) * d
        # QK-Norm: learned scale γ per head (shape: [n_heads, head_dim] for Q, [n_kv_heads, head_dim] for K)
        qk_norm   = (nq * h) + (nkv * h)
        attn_per_layer = q_proj + k_proj + v_proj + o_proj + qk_norm

        # SwiGLU FFN per layer: gate + up + down (no bias)
        # SwiGLU: output = (gate(x) * silu(up(x))) @ down
        # gate: d_model → ffn_intermediate
        # up:   d_model → ffn_intermediate
        # down: ffn_intermediate → d_model
        gate_proj = d * fi
        up_proj   = d * fi
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
            "embedding":    embedding,
            "lm_head":      lm_head,
            "per_layer": {
                "attention": attn_per_layer,
                "ffn":       ffn_per_layer,
                "norms":     norm_per_layer,
                "total":     per_layer,
            },
            "all_layers":   all_layers,
            "final_norm":   final_norm,
            "total":        total,
            "total_B":      total / 1e9,
            "total_M":      total / 1e6,
        }


# ---------------------------------------------------------------------------
# Canonical configs (from spec Section 1.2)
# ---------------------------------------------------------------------------

CONFIGS = {
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


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    No mean-centering — only RMS scaling. Faster than LayerNorm and
    standard in all modern LLMs (Llama, Qwen, OLMo).

    Formula: x / RMS(x) * γ
    where RMS(x) = sqrt(mean(x²) + ε)
    """

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # γ, learned scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model) or (B, n_heads, T, head_dim)
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight

    def extra_repr(self) -> str:
        return f"dim={self.weight.shape[0]}, eps={self.eps}"


# ---------------------------------------------------------------------------
# Rotary Position Embedding (RoPE)
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).

    Encodes position as rotation in 2D subspaces of the head dimension.
    Applied to Q and K *after* QK-Norm, *before* the dot product.

    Key properties:
    - Relative position only: <q_i, k_j> depends only on (i - j)
    - Generalizes to unseen sequence lengths (unlike learned absolute PE)
    - base=500_000 extends context gracefully to 32k+ tokens
      (standard base=10_000 degrades past ~4k)

    Implementation: precompute cos/sin cache up to max_seq_len,
    apply via complex-number rotation (einsum-free for export compat).
    """

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 500_000.0):
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"

        self.head_dim   = head_dim
        self.max_seq_len = max_seq_len
        self.base       = base

        # Precompute inverse frequencies
        # θ_i = 1 / (base ^ (2i / head_dim))  for i in [0, head_dim/2)
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build and cache cos/sin tables at init
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        positions = torch.arange(seq_len, device=self.inv_freq.device).float()
        # freqs: (seq_len, head_dim/2)
        freqs = torch.outer(positions, self.inv_freq)
        # emb: (seq_len, head_dim) — duplicate for cos and sin
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)  # (1,1,T,D)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotate each pair (x1, x2) → (-x2, x1)."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def forward(
        self,
        q: torch.Tensor,   # (B, n_heads, T, head_dim)
        k: torch.Tensor,   # (B, n_kv_heads, T, head_dim)
        offset: int = 0,   # For KV-cache: position offset during generation
    ) -> tuple[torch.Tensor, torch.Tensor]:
        T = q.shape[2]

        # Extend cache if needed (e.g. unexpected long sequence)
        if offset + T > self.cos_cached.shape[2]:
            self._build_cache(offset + T)

        cos = self.cos_cached[:, :, offset : offset + T, :]
        sin = self.sin_cached[:, :, offset : offset + T, :]

        q_rot = (q * cos) + (self._rotate_half(q) * sin)
        k_rot = (k * cos) + (self._rotate_half(k) * sin)

        return q_rot, k_rot


# ---------------------------------------------------------------------------
# Grouped Query Attention with QK-Norm
# ---------------------------------------------------------------------------

class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA) with QK-Norm and RoPE.

    GQA: n_heads Q heads share n_kv_heads K/V heads.
    GQA ratio = n_heads / n_kv_heads (must be integer).
    Reduces KV cache size by factor of gqa_ratio — critical for inference.

    QK-Norm: RMSNorm applied independently to Q and K projections,
    per head, *before* RoPE. Prevents attention logit explosion.
    Critical at small scale where weight initialization variance is
    proportionally larger relative to the signal.

    Ordering (enforced):
      project → QK-Norm → RoPE → attention → output project

    No bias in any projection (cleaner quantization).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads    = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim   = config.head_dim
        self.gqa_ratio  = config.gqa_ratio
        self.d_model    = config.d_model

        # Projections — no bias
        self.q_proj = nn.Linear(config.d_model, config.n_heads    * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads  * config.head_dim, config.d_model,   bias=False)

        # QK-Norm: independent RMSNorm per head for Q and K
        # Shape: (n_heads * head_dim,) — applied after reshape, before RoPE
        self.q_norm = RMSNorm(config.head_dim, eps=config.norm_eps)
        self.k_norm = RMSNorm(config.head_dim, eps=config.norm_eps)

        # RoPE
        self.rope = RotaryEmbedding(
            head_dim=config.head_dim,
            max_seq_len=config.max_seq_len,
            base=config.rope_base,
        )

        self.dropout = config.dropout

        # Scaling factor: 1/sqrt(head_dim)
        self.scale = config.head_dim ** -0.5

    def forward(
        self,
        x: torch.Tensor,                          # (B, T, d_model)
        attention_mask: Optional[torch.Tensor],   # (B, 1, T, T) or None
        kv_cache: Optional[tuple] = None,         # (k_cache, v_cache) for generation
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, Optional[tuple]]:

        B, T, _ = x.shape

        # --- Project ---
        q = self.q_proj(x)   # (B, T, n_heads * head_dim)
        k = self.k_proj(x)   # (B, T, n_kv_heads * head_dim)
        v = self.v_proj(x)   # (B, T, n_kv_heads * head_dim)

        # Reshape to (B, n_heads, T, head_dim)
        q = q.view(B, T, self.n_heads,    self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # --- QK-Norm (applied per head, before RoPE) ---
        # Norm operates on the last dim (head_dim); reshape is not needed
        q = self.q_norm(q)   # (B, n_heads, T, head_dim)
        k = self.k_norm(k)   # (B, n_kv_heads, T, head_dim)

        # --- RoPE ---
        q, k = self.rope(q, k, offset=position_offset)

        # --- KV cache (generation) ---
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)
        new_kv_cache = (k, v) if kv_cache is not None else None

        # --- GQA: expand KV heads to match Q heads ---
        # (B, n_kv_heads, T, head_dim) → (B, n_heads, T, head_dim)
        # expand is zero-copy — no data duplication
        k = k.unsqueeze(2).expand(-1, -1, self.gqa_ratio, -1, -1)
        k = k.reshape(B, self.n_heads, -1, self.head_dim)
        v = v.unsqueeze(2).expand(-1, -1, self.gqa_ratio, -1, -1)
        v = v.reshape(B, self.n_heads, -1, self.head_dim)

        # --- Scaled dot-product attention ---
        # Use F.scaled_dot_product_attention when available (FlashAttention-2 path on CUDA)
        # Falls back to manual implementation otherwise (CPU, Trainium NKI replaces this)
        if hasattr(F, "scaled_dot_product_attention"):
            # attn_mask: None triggers causal masking automatically with is_causal=True
            dropout_p = self.dropout if self.training else 0.0
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=dropout_p,
                is_causal=(attention_mask is None),
                scale=self.scale,
            )
        else:
            # Manual fallback (used on Trainium before NKI kernel is installed)
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, T, T)
            if attention_mask is not None:
                scores = scores + attention_mask
            else:
                # Causal mask
                causal = torch.ones(T, T, device=x.device, dtype=torch.bool).tril()
                scores = scores.masked_fill(~causal, float("-inf"))
            scores = F.softmax(scores, dim=-1)
            if self.dropout > 0 and self.training:
                scores = F.dropout(scores, p=self.dropout)
            out = torch.matmul(scores, v)  # (B, H, T, D)

        # --- Merge heads and project ---
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        out = self.o_proj(out)

        return out, new_kv_cache


# ---------------------------------------------------------------------------
# SwiGLU Feed-Forward Network
# ---------------------------------------------------------------------------

class SwiGLUFFN(nn.Module):
    """
    SwiGLU Feed-Forward Network.

    SwiGLU(x) = SiLU(gate(x)) * up(x)
    output     = down(SwiGLU(x))

    Two parallel projections (gate + up) followed by element-wise gate,
    then a down projection. Uses 3 matrices instead of the standard 2,
    but the intermediate width is typically set to 2/3 of the standard
    4*d_model to keep parameter count equivalent.

    Our spec uses a pre-computed ffn_intermediate that is already
    optimally sized AND tile-aligned (÷128).

    No bias. No dropout (applied at block level if needed).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.d_model, config.ffn_intermediate, bias=False)
        self.up_proj   = nn.Linear(config.d_model, config.ffn_intermediate, bias=False)
        self.down_proj = nn.Linear(config.ffn_intermediate, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        gate = F.silu(self.gate_proj(x))  # SiLU(Wg * x) — gating signal
        up   = self.up_proj(x)            # Wu * x       — content signal
        return self.down_proj(gate * up)  # element-wise gate then project down


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
        self.attn      = GroupedQueryAttention(config)
        self.ffn_norm  = RMSNorm(config.d_model, eps=config.norm_eps)
        self.ffn       = SwiGLUFFN(config)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[tuple] = None,
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, Optional[tuple]]:

        # Attention sub-block (pre-norm + residual)
        attn_out, new_kv_cache = self.attn(
            self.attn_norm(x),
            attention_mask=attention_mask,
            kv_cache=kv_cache,
            position_offset=position_offset,
        )
        x = x + attn_out

        # FFN sub-block (pre-norm + residual)
        x = x + self.ffn(self.ffn_norm(x))

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
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])

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
        Initialize weights following the GPT-2 / Llama convention.

        Embedding: N(0, 0.02)
        Linear weights: N(0, 0.02)
        Output projections (o_proj, down_proj): scaled by 1/sqrt(2 * n_layers)
          This is the "GPT-2 residual scaling" trick — prevents the residual
          stream from growing in variance with depth.
        RMSNorm weights: 1.0 (identity at init)
        """
        std = 0.02
        residual_scale = std / math.sqrt(2 * self.config.n_layers)

        for name, module in self.named_modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()

            elif isinstance(module, nn.Linear):
                # Output projections get the residual scaling
                if name.endswith(("o_proj", "down_proj")):
                    nn.init.normal_(module.weight, mean=0.0, std=residual_scale)
                else:
                    nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)

    def forward(
        self,
        input_ids: torch.Tensor,                  # (B, T) — token IDs
        attention_mask: Optional[torch.Tensor] = None,  # (B, T) — 1=attend, 0=mask (padding)
        kv_caches: Optional[list] = None,         # list of (k, v) per layer, for generation
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
        x = self.embedding(input_ids)  # (B, T, d_model)

        # --- Build causal attention mask from padding mask ---
        # During training with padding: mask combines causal + padding
        # During generation: handled by kv_cache + is_causal in SDPA
        attn_mask = None
        if attention_mask is not None and kv_caches is None:
            # attention_mask: (B, T) — 0 for pad tokens
            # Build additive mask: 0 for attended positions, -inf for masked
            attn_mask = _build_additive_mask(attention_mask, x.dtype, x.device)

        # --- Transformer blocks ---
        new_kv_caches = [] if kv_caches is not None else None

        for i, block in enumerate(self.blocks):
            kv = kv_caches[i] if kv_caches is not None else None
            x, new_kv = block(
                x,
                attention_mask=attn_mask,
                kv_cache=kv,
                position_offset=position_offset,
            )
            if new_kv_caches is not None:
                new_kv_caches.append(new_kv)

        # --- Final norm ---
        x = self.norm(x)  # (B, T, d_model)

        # --- LM head ---
        if self.config.tie_embeddings:
            # Reuse embedding weight matrix: (vocab_size, d_model) → transpose to (d_model, vocab_size)
            logits = F.linear(x, self.embedding.weight)  # (B, T, vocab_size)
        else:
            logits = self.lm_head(x)

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
        input_ids: torch.Tensor,   # (B, T_prompt)
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.9,
        eos_token_id: int = 2,
    ) -> torch.Tensor:
        """
        Simple autoregressive generation with KV cache.
        Nucleus (top-p) sampling.

        For research / evaluation use. Production serving would use
        llama.cpp or a dedicated inference engine.
        """
        self.eval()
        B = input_ids.shape[0]
        device = input_ids.device

        # Prefill: process the prompt
        logits, kv_caches = self.forward(input_ids)
        next_token_logits = logits[:, -1, :]  # (B, vocab_size)

        generated = input_ids
        position_offset = input_ids.shape[1]

        for _ in range(max_new_tokens):
            # Sample
            next_token = _sample(next_token_logits, temperature, top_p)  # (B, 1)
            generated = torch.cat([generated, next_token], dim=1)

            # Stop if all sequences hit EOS
            if (next_token == eos_token_id).all():
                break

            # Decode step: single token with KV cache
            logits, kv_caches = self.forward(
                next_token,
                kv_caches=kv_caches,
                position_offset=position_offset,
            )
            next_token_logits = logits[:, -1, :]
            position_offset += 1

        return generated


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_additive_mask(
    padding_mask: torch.Tensor,  # (B, T), 1=real token, 0=pad
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """
    Build a combined causal + padding additive attention mask.
    Output shape: (B, 1, T, T)
    Value: 0 for attend, -inf for mask.
    """
    B, T = padding_mask.shape
    # Causal mask: lower triangular
    causal = torch.ones(T, T, device=device, dtype=torch.bool).tril()
    # Padding mask broadcast
    pad = padding_mask[:, None, None, :].bool()  # (B, 1, 1, T)
    # Combined: must be causal AND not padding
    combined = causal.unsqueeze(0) & pad  # (B, 1, T, T)
    # Convert to additive float mask
    additive = torch.zeros(B, 1, T, T, dtype=dtype, device=device)
    additive.masked_fill_(~combined, float("-inf"))
    return additive


def _sample(
    logits: torch.Tensor,  # (B, vocab_size)
    temperature: float,
    top_p: float,
) -> torch.Tensor:
    """Nucleus (top-p) sampling."""
    if temperature == 0.0:
        return logits.argmax(dim=-1, keepdim=True)

    logits = logits / temperature
    probs = F.softmax(logits, dim=-1)

    # Top-p filtering
    sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
    cumulative = sorted_probs.cumsum(dim=-1)
    # Remove tokens with cumulative prob above top_p (shift right to keep first token above)
    remove = (cumulative - sorted_probs) > top_p
    sorted_probs[remove] = 0.0
    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)

    # Sample and map back to original indices
    sampled = torch.multinomial(sorted_probs, num_samples=1)  # (B, 1) index into sorted
    return sorted_idx.gather(-1, sampled)                      # (B, 1) original vocab index


# ---------------------------------------------------------------------------
# Cross-entropy loss (with ignore_index for padding)
# ---------------------------------------------------------------------------

def compute_loss(
    logits: torch.Tensor,    # (B, T, vocab_size)
    targets: torch.Tensor,   # (B, T)
    ignore_index: int = 0,   # PAD_TOKEN_ID
) -> torch.Tensor:
    """
    Causal language modeling loss.

    Shift by 1: predict token[t+1] from context token[0..t].
    Ignore padding tokens in loss computation.

    Returns scalar loss (mean over non-padding positions).
    """
    # Shift: logits predict the next token
    shift_logits  = logits[:, :-1, :].contiguous()   # (B, T-1, vocab_size)
    shift_targets = targets[:, 1:].contiguous()       # (B, T-1)

    # Flatten for cross-entropy
    B, T, V = shift_logits.shape
    loss = F.cross_entropy(
        shift_logits.view(B * T, V),
        shift_targets.view(B * T),
        ignore_index=ignore_index,
        reduction="mean",
    )
    return loss
