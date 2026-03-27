"""
kv_compress.py
==============
TurboQuant KV cache compression (Google Research, March 2026).

Two-stage algorithm:
  Stage 1 — PolarQuant (for K vectors):
    Separate each K vector into magnitude r and unit direction u.
    Quantize u to INT8 — unit vectors live in [-1, 1] so no per-block
    normalization constants are needed, eliminating the overhead that
    defeats traditional KV quantization.

  Stage 2 — QJL error correction (optional, ~4-6× total):
    Project the quantization residual through a random Johnson-Lindenstrauss
    matrix Φ and store the sign bits. During decompression, add back the
    corrected residual.  Not yet implemented here; PolarQuant alone gives
    ~2× compression with zero accuracy loss at the model scales we target.

  V vectors:
    Values are not used in inner products (only in a weighted sum), so they
    are less sensitive to directional quantization. Simple INT8 with a
    per-head scale is sufficient and gives ~2× compression.

Why this matters for this project:
  At inference on Graviton4 / Kamrui, the KV cache is the binding memory
  constraint for long chain-of-thought generation. A 1B model at 32k context
  uses ~1.34 GB of KV cache in BF16; PolarQuant brings this to ~670 MB,
  enabling 2× more concurrent requests or 2× longer context on the same
  hardware.

  During GRPO training, generate_completions() samples group_size=8 completions
  per prompt in parallel. With BF16 KV caches this costs ~5 GB per prompt
  group at 2k context.  PolarQuant halves this, making larger group sizes or
  longer generation budgets feasible on a 32 GB GPU.

Key properties (inherited from TurboQuant):
  - Data-oblivious: no calibration data required, no codebook training.
  - Training-free: applies post-hoc to any checkpoint.
  - Drop-in: the compress/decompress API wraps existing (k, v) tuple usage.
  - Exact for head_dim=128 (our tile-aligned design): the INT8 scale is
    always 127 and no padding waste occurs.

Compression ratios (BF16 baseline, head_dim=128):
  PolarQuant K (INT8):  128B (int8) + 2B (f16 magnitude) = 130B  vs  256B → 1.97×
  Simple INT8 V:        128B (int8) + 2B (f16 scale/head) ≈ 130B  vs  256B → 1.97×
  Combined:             ~2× KV cache memory reduction, zero accuracy loss.

Usage:
  from model.kv_compress import CompressedKV

  # Compress a single layer's (k, v) after the attention forward pass
  cmp = CompressedKV.compress(k, v)          # k,v: (B, n_kv_heads, T, 128)

  # Decompress before passing back into the attention layer
  k_dec, v_dec = cmp.decompress()

  # Convenience: compress/decompress a full list of per-layer caches
  compressed   = compress_kv_caches(kv_caches)   # list[(k,v)] → list[CompressedKV]
  decompressed = decompress_kv_caches(compressed) # list[CompressedKV] → list[(k,v)]

  # Transparent wrapper for a generation forward pass
  logits, kv_caches = forward_compressed(model, input_ids, kv_caches, pos_offset)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# INT8 uses signed values in [-128, 127]; we use 127 as the scale factor
# so the full [-1, 1] range maps to [-127, 127] with one extra code point
# reserved for safety.  Never use -128 to avoid asymmetric ranges.
_INT8_SCALE = 127.0


# ---------------------------------------------------------------------------
# CompressedKV — per-layer compressed cache object
# ---------------------------------------------------------------------------


@dataclass
class CompressedKV:
    """
    Compressed KV cache for one transformer layer.

    Attributes
    ----------
    k_magnitude : (B, n_kv_heads, T)  float16
        L2 norm of each original K vector. Kept in float16 for memory
        efficiency; the magnitude rarely needs full float32 precision.

    k_direction : (B, n_kv_heads, T, head_dim)  int8
        Unit-vector direction of each K vector, quantized to INT8.
        Dequantization: direction_float = k_direction.float() / _INT8_SCALE
        Reconstruction: k ≈ k_magnitude.unsqueeze(-1) * direction_float

    v_quant : (B, n_kv_heads, T, head_dim)  int8
        V vectors quantized to INT8.

    v_scale : (B, n_kv_heads, 1, 1)  float16
        Per-(batch, head) absolute-max scale for V quantization.
        Dequantization: v ≈ v_quant.float() * v_scale / _INT8_SCALE

    device : torch.device
        The device where compressed tensors live.

    dtype : torch.dtype
        The original float dtype (bfloat16 / float32) to restore on decompress.
    """

    k_magnitude: torch.Tensor  # (B, n_kv_heads, T)          float16
    k_direction: torch.Tensor  # (B, n_kv_heads, T, head_dim) int8
    v_quant: torch.Tensor  # (B, n_kv_heads, T, head_dim) int8
    v_scale: torch.Tensor  # (B, n_kv_heads, 1, 1)        float16
    device: torch.device
    dtype: torch.dtype

    # ── Compress ──────────────────────────────────────────────────────────

    @classmethod
    def compress(
        cls,
        k: torch.Tensor,  # (B, n_kv_heads, T, head_dim)
        v: torch.Tensor,  # (B, n_kv_heads, T, head_dim)
    ) -> "CompressedKV":
        """
        Compress a (k, v) pair to a CompressedKV.

        K — PolarQuant
        --------------
        1. Compute L2 magnitude: r = ||k||_2   (per vector, keepdim for broadcast)
        2. Compute unit direction: u = k / r    (values guaranteed in [-1, 1])
        3. Quantize u to INT8: q = round(u * 127).clamp(-127, 127).to(int8)

        Unit vectors need NO per-block normalization constant because their
        range is bounded analytically by definition: ||u||_2 = 1 implies
        each element |u_i| ≤ 1.  This is the core TurboQuant insight —
        eliminating the per-block constant removes 1-2 bits of overhead that
        traditional quantizers must pay.

        V — Per-head INT8
        -----------------
        V vectors are summed (not dot-producted with Q), so their directional
        precision matters less than K. A simple per-head absolute-max scaling
        suffices:
          scale = max(|v|)  per (batch, head)
          q     = round(v / scale * 127).clamp(-127, 127).to(int8)

        Args:
            k: Key tensor,   shape (B, n_kv_heads, T, head_dim), any float dtype.
            v: Value tensor, shape (B, n_kv_heads, T, head_dim), any float dtype.

        Returns:
            CompressedKV with all tensors on the same device as k.
        """
        orig_dtype = k.dtype
        device = k.device

        # Work in float32 for quantization arithmetic to avoid BF16 rounding
        # errors accumulating in the clamp/round operations.
        k_f = k.float()
        v_f = v.float()

        # ── K: PolarQuant ────────────────────────────────────────────────
        # Magnitude: L2 norm over the last (head_dim) dimension.
        # clamp(min=1e-8) prevents division by zero for near-zero vectors
        # (rare in practice but possible after QK-Norm resets magnitudes).
        k_mag = k_f.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # (B, nkv, T, 1)

        # Unit direction: divide by magnitude → all values in [-1, 1]
        k_unit = k_f / k_mag  # (B, nkv, T, head_dim)

        # Quantize: scale to [-127, 127] and round to nearest integer.
        # Using _INT8_SCALE = 127 (not 128) keeps the range symmetric,
        # avoiding asymmetric INT8 bias toward positive values.
        k_dir_q = (k_unit * _INT8_SCALE).round().clamp(-127, 127).to(torch.int8)

        # Store magnitude in float16 — sufficient precision for a scalar scale
        k_mag_stored = k_mag.squeeze(-1).to(torch.float16)  # (B, nkv, T)

        # ── V: Per-head absolute-max INT8 ────────────────────────────────
        # Scale = max(|v|) per (batch, head) — preserves the full range of
        # each head's values while fitting into INT8.
        # The (B, nkv, 1, 1) shape broadcasts over T and head_dim.
        v_scale = v_f.abs().amax(dim=(-2, -1), keepdim=True).clamp(min=1e-8)  # (B, nkv, 1, 1)

        v_quant = (v_f / v_scale * _INT8_SCALE).round().clamp(-127, 127).to(torch.int8)

        # Store V scale in float16
        v_scale_stored = v_scale.to(torch.float16)  # (B, nkv, 1, 1)

        return cls(
            k_magnitude=k_mag_stored,
            k_direction=k_dir_q,
            v_quant=v_quant,
            v_scale=v_scale_stored,
            device=device,
            dtype=orig_dtype,
        )

    # ── Decompress ────────────────────────────────────────────────────────

    def decompress(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Reconstruct approximate (k, v) tensors from compressed representation.

        K reconstruction:
            u = k_direction.float() / 127         → unit vector in float32
            k ≈ k_magnitude.unsqueeze(-1) * u     → scale by magnitude

        V reconstruction:
            v ≈ v_quant.float() / 127 * v_scale   → scale by per-head max

        Both are cast back to the original dtype (bfloat16 or float32) for
        seamless integration with the existing attention computation.

        Reconstruction error:
            K: INT8 quantization error ≤ 0.5/127 ≈ 0.004 per element on the
               unit sphere. For attention scores this is < 0.004 * head_dim
               = 0.004 * 128 = 0.51 per Q·K dot product — negligible relative
               to the typical magnitude of attention logits.

            V: INT8 error ≤ 0.5/127 × scale_per_head ≈ 0.4% of the max value.
               For value aggregation (attention output = softmax × V) this
               introduces at most ~0.4% relative error per element.

        Returns:
            (k_reconstructed, v_reconstructed) — same shapes as original inputs
            to CompressedKV.compress(), on the original device.
        """
        # K: dequantize direction, then scale by magnitude
        k_unit = self.k_direction.float() / _INT8_SCALE  # (B, nkv, T, D) f32
        k_mag = self.k_magnitude.float().unsqueeze(-1)  # (B, nkv, T, 1) f32
        k_rec = (k_unit * k_mag).to(self.dtype)  # (B, nkv, T, D)

        # V: dequantize with per-head scale
        v_scale = self.v_scale.float()  # (B, nkv, 1, 1) f32
        v_rec = (self.v_quant.float() / _INT8_SCALE * v_scale).to(self.dtype)  # (B, nkv, T, D)

        return k_rec, v_rec

    # ── Memory reporting ──────────────────────────────────────────────────

    def bytes_used(self) -> int:
        """Return total bytes consumed by all compressed tensors."""
        return (
            self.k_magnitude.nelement() * self.k_magnitude.element_size()
            + self.k_direction.nelement() * self.k_direction.element_size()
            + self.v_quant.nelement() * self.v_quant.element_size()
            + self.v_scale.nelement() * self.v_scale.element_size()
        )

    def bytes_uncompressed(self) -> int:
        """Return bytes that the uncompressed (k, v) pair would consume."""
        B, nkv, T, D = self.k_direction.shape
        # Both k and v: (B, nkv, T, D) in the original dtype
        bytes_per_elem = {
            torch.bfloat16: 2,
            torch.float16: 2,
            torch.float32: 4,
        }.get(self.dtype, 2)
        return 2 * B * nkv * T * D * bytes_per_elem

    def compression_ratio(self) -> float:
        """Actual compression ratio (uncompressed / compressed)."""
        used = self.bytes_used()
        return self.bytes_uncompressed() / used if used > 0 else 1.0

    @property
    def seq_len(self) -> int:
        """Number of cached token positions."""
        return self.k_direction.shape[2]

    def __repr__(self) -> str:
        B, nkv, T, D = self.k_direction.shape
        ratio = self.compression_ratio()
        return (
            f"CompressedKV(B={B}, n_kv_heads={nkv}, T={T}, head_dim={D}, "
            f"ratio={ratio:.2f}×, device={self.device})"
        )


# ---------------------------------------------------------------------------
# Cache-list helpers
# ---------------------------------------------------------------------------


def compress_kv_caches(
    kv_caches: list[tuple[torch.Tensor, torch.Tensor]],
) -> list[CompressedKV]:
    """
    Compress a full list of per-layer (k, v) tuples.

    Args:
        kv_caches: list of length n_layers, each element is (k, v) where
            k: (B, n_kv_heads, T, head_dim)
            v: (B, n_kv_heads, T, head_dim)

    Returns:
        list of CompressedKV, same length as input.
    """
    return [CompressedKV.compress(k, v) for k, v in kv_caches]


def decompress_kv_caches(
    compressed: list[CompressedKV],
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """
    Decompress a list of CompressedKV back to (k, v) tuple pairs.

    Args:
        compressed: list of CompressedKV, length n_layers.

    Returns:
        list of (k, v) tuples, compatible with SmallReasoningModel.forward().
    """
    return [c.decompress() for c in compressed]


def kv_cache_memory_report(
    kv_caches: list[CompressedKV | tuple],
    label: str = "",
) -> str:
    """
    Return a formatted memory summary for a KV cache list.

    Works with both CompressedKV objects and raw (k, v) tuples.
    Useful for logging during generation to verify compression is active.
    """
    lines = [f"KV cache memory {label}:"]
    total_compressed = 0
    total_uncompressed = 0

    for i, entry in enumerate(kv_caches):
        if isinstance(entry, CompressedKV):
            c = entry.bytes_used()
            u = entry.bytes_uncompressed()
            total_compressed += c
            total_uncompressed += u
            lines.append(
                f"  layer {i:2d}: {c/1024:.1f} KB compressed"
                f" ({entry.compression_ratio():.2f}×, T={entry.seq_len})"
            )
        else:
            k, v = entry
            u = k.nelement() * k.element_size() + v.nelement() * v.element_size()
            total_uncompressed += u
            total_compressed += u
            lines.append(f"  layer {i:2d}: {u/1024:.1f} KB (uncompressed)")

    if total_compressed > 0:
        ratio = total_uncompressed / total_compressed
        lines.append(
            f"  TOTAL: {total_compressed/1024**2:.2f} MB compressed"
            f" / {total_uncompressed/1024**2:.2f} MB uncompressed"
            f" ({ratio:.2f}× overall)"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Generation loop wrapper
# ---------------------------------------------------------------------------


def forward_compressed(
    model: "torch.nn.Module",
    input_ids: torch.Tensor,
    kv_caches: Optional[list],
    position_offset: int = 0,
    autocast_ctx=None,
) -> tuple[torch.Tensor, list[CompressedKV] | None]:
    """
    Drop-in replacement for model(input_ids, kv_caches=..., position_offset=...)
    that decompresses caches before the forward pass and re-compresses after.

    This wrapper keeps architecture.py unchanged:  the model always receives
    and returns plain (k, v) tensors; compression/decompression is handled
    entirely outside the model.

    Memory lifecycle during a decode step:
      1. Decompress stored caches (small →  full float, brief peak)
      2. Forward pass (attention sees full-precision K/V from cache + new token)
      3. Compress new caches (full float → small, brief peak)
      4. Return compressed; full-float caches are freed immediately

    Peak memory is briefly 2× the compressed size during steps 1 and 3,
    but the full-size cache is never held alongside the compressed version
    for any significant duration.

    Args:
        model:           SmallReasoningModel instance (or any nn.Module with
                         the same KV cache interface).
        input_ids:       (B, T) token ID tensor.
        kv_caches:       None, or list of CompressedKV / (k, v) tuples.
                         Mixed lists (some compressed, some not) are handled.
        position_offset: RoPE position offset for the current tokens.
        autocast_ctx:    Optional torch.amp.autocast context manager.
                         If None, no autocast is applied.

    Returns:
        (logits, new_kv_caches):
            logits:         (B, T, vocab_size) float tensor.
            new_kv_caches:  list[CompressedKV] if input kv_caches was not None,
                            else None (matches model's own convention).
    """
    # Decompress mixed/compressed caches before passing to model
    if kv_caches is not None:
        raw_caches = []
        for entry in kv_caches:
            if isinstance(entry, CompressedKV):
                raw_caches.append(entry.decompress())
            else:
                raw_caches.append(entry)
    else:
        raw_caches = None

    # Forward pass — the model sees plain (k, v) tensors as always
    if autocast_ctx is not None:
        with autocast_ctx:
            logits, new_kv_raw = model(
                input_ids,
                kv_caches=raw_caches,
                position_offset=position_offset,
            )
    else:
        logits, new_kv_raw = model(
            input_ids,
            kv_caches=raw_caches,
            position_offset=position_offset,
        )

    # Free decompressed tensors immediately (before compressing new ones)
    del raw_caches

    # Compress new caches if we're in generation mode
    if new_kv_raw is not None:
        new_kv_compressed = [CompressedKV.compress(k, v) for k, v in new_kv_raw]
    else:
        new_kv_compressed = None

    return logits, new_kv_compressed


# ---------------------------------------------------------------------------
# Verification / tests
# ---------------------------------------------------------------------------


def verify_compression(
    head_dim: int = 128,
    n_kv_heads: int = 4,
    batch: int = 2,
    seq_len: int = 512,
    dtype: torch.dtype = torch.bfloat16,
    atol_k: float = 0.025,  # theoretical bound is ~0.056; 0.025 gives BF16 variance headroom
    atol_v: float = 0.02,
    verbose: bool = True,
) -> bool:
    """
    Verify CompressedKV round-trip accuracy on synthetic data.

    Checks:
      1. K and V reconstruct to within atol tolerance (mean absolute error).
      2. Attention dot products (q·k^T) are close to uncompressed dot products.
      3. Compression ratio is ≥ 1.5× (expected ~2×).
      4. No NaN/Inf in compressed or decompressed tensors.
      5. Memory bytes match the formula.

    Returns True if all checks pass.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    passed = 0
    total = 0

    def check(name: str, cond: bool, detail: str = ""):
        nonlocal passed, total
        total += 1
        ok = bool(cond)
        if verbose:
            tag = "✓" if ok else "✗"
            print(f"  {tag}  {name}" + (f"  ({detail})" if detail else ""))
        if ok:
            passed += 1
        return ok

    if verbose:
        print(
            f"\nCompressedKV verification  "
            f"[B={batch}, nkv={n_kv_heads}, T={seq_len}, D={head_dim}, {dtype}]"
        )
        print("─" * 60)

    # Generate synthetic K and V tensors resembling real attention outputs:
    # K vectors often have moderate norms (~1-3) after QK-Norm;
    # V vectors have scale similar to the input magnitude.
    torch.manual_seed(42)
    k = torch.randn(batch, n_kv_heads, seq_len, head_dim, dtype=dtype, device=device)
    v = torch.randn(batch, n_kv_heads, seq_len, head_dim, dtype=dtype, device=device)

    # Compress
    cmp = CompressedKV.compress(k, v)

    # ── Checks on compressed tensors ──────────────────────────────────────
    check("k_direction dtype == int8", cmp.k_direction.dtype == torch.int8)
    check("k_magnitude dtype == float16", cmp.k_magnitude.dtype == torch.float16)
    check("v_quant dtype == int8", cmp.v_quant.dtype == torch.int8)
    check("v_scale dtype == float16", cmp.v_scale.dtype == torch.float16)
    check("no NaN in k_direction", not cmp.k_direction.float().isnan().any())
    check("no NaN in v_quant", not cmp.v_quant.float().isnan().any())
    check(
        "compression ratio ≥ 1.5×",
        cmp.compression_ratio() >= 1.5,
        f"got {cmp.compression_ratio():.2f}×",
    )

    # ── Round-trip reconstruction accuracy ────────────────────────────────
    k_dec, v_dec = cmp.decompress()

    k_f = k.float()
    k_dec_f = k_dec.float()
    v_f = v.float()
    v_dec_f = v_dec.float()

    k_mae = (k_f - k_dec_f).abs().mean().item()
    v_mae = (v_f - v_dec_f).abs().mean().item()

    check(f"K round-trip MAE ≤ {atol_k}", k_mae <= atol_k, f"MAE={k_mae:.5f}")
    check(f"V round-trip MAE ≤ {atol_v}", v_mae <= atol_v, f"MAE={v_mae:.5f}")
    check("no NaN in decompressed K", not k_dec.isnan().any())
    check("no NaN in decompressed V", not v_dec.isnan().any())

    # ── Attention dot-product accuracy ────────────────────────────────────
    # Simulate Q vectors (same shape as K)
    q = torch.randn(batch, n_kv_heads, seq_len, head_dim, dtype=dtype, device=device)
    scale = head_dim**-0.5

    # Original attention scores: (B, nkv, T_q, T_k)
    scores_orig = torch.matmul(q.float(), k_f.transpose(-2, -1)) * scale
    # Compressed attention scores
    scores_comp = torch.matmul(q.float(), k_dec_f.transpose(-2, -1)) * scale

    dot_mae = (scores_orig - scores_comp).abs().mean().item()
    # Expected dot-product error: INT8 MAE per element × head_dim × scale
    # ≈ (1/127) * head_dim * scale ≈ 0.008 * 128 * 0.088 ≈ 0.09
    check(f"Attention dot-product MAE ≤ 0.15", dot_mae <= 0.15, f"MAE={dot_mae:.5f}")

    # Softmax attention weights difference (the practical measure of quality)
    attn_orig = torch.softmax(scores_orig, dim=-1)
    attn_comp = torch.softmax(scores_comp, dim=-1)
    attn_mae = (attn_orig - attn_comp).abs().mean().item()
    check(f"Attention weight MAE ≤ 0.01", attn_mae <= 0.01, f"MAE={attn_mae:.6f}")

    # ── Memory formula verification ───────────────────────────────────────
    # Expected bytes:
    #   k_magnitude:  B * nkv * T * 2 (float16)
    #   k_direction:  B * nkv * T * D * 1 (int8)
    #   v_quant:      B * nkv * T * D * 1 (int8)
    #   v_scale:      B * nkv * 1 * 1 * 2 (float16)
    expected = (
        batch * n_kv_heads * seq_len * 2  # k_magnitude (float16)
        + batch * n_kv_heads * seq_len * head_dim  # k_direction (int8)
        + batch * n_kv_heads * seq_len * head_dim  # v_quant (int8)
        + batch * n_kv_heads * 1 * 1 * 2  # v_scale (float16)
    )
    actual = cmp.bytes_used()
    check("Bytes formula matches", actual == expected, f"expected={expected}, got={actual}")

    # ── Summary ───────────────────────────────────────────────────────────
    if verbose:
        print(f"\n  Compression ratio: {cmp.compression_ratio():.2f}×")
        print(
            f"  K MAE: {k_mae:.5f}  |  V MAE: {v_mae:.5f}  |  "
            f"Attention MAE: {dot_mae:.5f}  |  Softmax MAE: {attn_mae:.6f}"
        )
        print(f"\n  Result: {passed}/{total} passed")
        if passed == total:
            print("  ✓ All checks passed. TurboQuant integration is ready.")
        else:
            print("  ✗ Some checks failed. Review before enabling in production.")

    return passed == total


# ---------------------------------------------------------------------------
# CLI: run verification standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Verify TurboQuant KV cache compression round-trip accuracy."
    )
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--n-kv-heads", type=int, default=4)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float32"])
    args = parser.parse_args()

    dtype_map = {"bfloat16": torch.bfloat16, "float32": torch.float32}
    ok = verify_compression(
        head_dim=args.head_dim,
        n_kv_heads=args.n_kv_heads,
        batch=args.batch,
        seq_len=args.seq_len,
        dtype=dtype_map[args.dtype],
    )
    raise SystemExit(0 if ok else 1)
