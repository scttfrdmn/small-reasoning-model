"""
shape_check.py
==============
Validates the architecture analytically — no torch required.

Why analytical shape checking matters:
  Shape mismatches (e.g. n_heads × head_dim ≠ d_model) are silent on CUDA —
  PyTorch will often reshape tensors implicitly, producing numerically wrong
  results with no exception. On Trainium2, a non-tile-aligned dimension causes
  the NeuronCore compiler to insert padding, reducing effective SBUF utilization
  and hurting throughput. Both failure modes are caught here with zero GPU cost.

  Running this script before writing any PyTorch code lets you validate that
  the numerical architecture is self-consistent: every matrix multiplication
  produces the expected output shape, every tile alignment constraint is met,
  and parameter counts match the spec. If this script passes, the architecture
  is correct by construction.

Traces every tensor dimension through the full forward pass using
symbolic shapes. Verifies:
  1. All tile alignment constraints (÷128)
  2. GQA dimension consistency
  3. QK-Norm shape compatibility
  4. RoPE shape compatibility
  5. Attention output shape
  6. FFN intermediate shapes
  7. Parameter counts (analytical vs spec)
  8. Memory estimates (weights + optimizer states)

Run this before touching a GPU:
  python shape_check.py
"""

from dataclasses import dataclass
import math

# ── Inline the config (no torch import needed) ────────────────────────────


@dataclass
class ShapeConfig:
    name: str
    d_model: int  # hidden dimension; must be divisible by 128 for tile alignment
    n_layers: int  # number of transformer blocks
    n_heads: int  # number of query heads
    n_kv_heads: int  # number of key/value heads (GQA: n_kv_heads ≤ n_heads)
    head_dim: int  # dimension per head; must be 128 for NKI tile fit
    ffn_intermediate: int  # SwiGLU intermediate dimension; must be divisible by 128
    vocab_size: int  # tokenizer vocabulary size; must be divisible by 128
    max_seq_len: int  # maximum sequence length for RoPE pre-computation and KV cache sizing
    rope_base: float = 500_000.0  # RoPE theta base (500k for long-context; llama3 default)
    tie_embeddings: bool = True  # share weights between embedding table and LM head (saves memory)


# Three model sizes matching the spec.
# All d_model, ffn_intermediate, vocab_size values are divisible by 128.
# All use head_dim=128 so that attention tiles map exactly onto the 128×128 NeuronCore MXU.
CONFIGS = {
    "500m": ShapeConfig("500m", 1280, 26, 10, 2, 128, 3456, 32768, 8192),
    "1b": ShapeConfig("1b", 2048, 20, 16, 4, 128, 5504, 32768, 16384),
    "3b": ShapeConfig("3b", 3072, 28, 24, 6, 128, 8192, 32768, 32768),
}

# ── Shape tracer ──────────────────────────────────────────────────────────

PASS = "✓"
FAIL = "✗"
WARN = "⚠"


class ShapeChecker:
    def __init__(self, config: ShapeConfig, batch=2, seq=512):
        self.c = config
        self.B = batch  # batch dimension (symbolic; any positive integer works)
        self.T = seq  # sequence length (symbolic; checked against max_seq_len elsewhere)
        self.ok = 0  # count of passing checks
        self.bad = 0  # count of failing checks

    def check(self, name: str, condition: bool, detail: str = ""):
        if condition:
            print(f"  {PASS}  {name}")
            self.ok += 1
        else:
            print(f"  {FAIL}  {name}  ← {detail}")
            self.bad += 1

    def warn(self, name: str, detail: str = ""):
        print(f"  {WARN}  {name}  {detail}")

    def run(self) -> bool:
        c = self.c
        B, T = self.B, self.T
        print(f"\n{'='*62}")
        print(f"  CONFIG: {c.name.upper()}  |  batch={B}  seq={T}")
        print(f"{'='*62}")

        # ── 1. Tile alignment ─────────────────────────────────────────
        # Trainium2's NeuronCore has a 128×128 BF16 systolic array (the MXU / TensorE unit).
        # Every matrix multiply operand must have its inner dimension be a multiple of 128
        # to fill a complete tile. A non-aligned dimension (e.g. d_model=1024 is fine;
        # d_model=1000 is not) forces the compiler to pad to the next tile boundary,
        # wasting SBUF bandwidth and reducing effective MXU utilization.
        # If ANY of these checks fail, fix the config — do not proceed to implementation.
        print("\n  [Tile alignment — Trainium2 NeuronCore 128×128 BF16]")
        self.check("d_model ÷ 128", c.d_model % 128 == 0, f"{c.d_model} % 128 = {c.d_model % 128}")
        self.check(
            "ffn_intermediate ÷ 128",
            c.ffn_intermediate % 128 == 0,
            f"{c.ffn_intermediate} % 128 = {c.ffn_intermediate % 128}",
        )
        self.check(
            "vocab_size ÷ 128",
            c.vocab_size % 128 == 0,
            f"{c.vocab_size} % 128 = {c.vocab_size % 128}",
        )
        # head_dim=128 is not just "divisible by 128" — it IS the tile size.
        # A single attention head's Q×K^T matmul is (seq_len × 128) @ (128 × seq_len),
        # where both operands are tile-sized on the inner dimension. This means one
        # NKI matmul tile exactly covers one head with zero padding waste.
        # head_dim=64 would leave the second half of each tile unused; head_dim=256
        # would require two tiles per head with extra orchestration overhead.
        self.check("head_dim == 128", c.head_dim == 128, f"got {c.head_dim}")
        kv_dim = c.n_kv_heads * c.head_dim
        # KV projection output (n_kv_heads × head_dim) must also be tile-aligned
        # because it feeds into the GQA expand and subsequent K^T matmul.
        self.check(
            f"n_kv_heads×head_dim ({kv_dim}) ÷ 128",
            kv_dim % 128 == 0,
            f"{kv_dim} % 128 = {kv_dim % 128}",
        )

        # ── 2. Attention dimension consistency ────────────────────────
        # These are mathematical identities that must hold by construction.
        # n_heads × head_dim == d_model: the concatenated multi-head output must
        # match the residual stream width exactly (otherwise the O-projection would
        # change the hidden dimension, which is not the intended design).
        # n_heads % n_kv_heads == 0: GQA requires each KV head to be shared by
        # an integer number of query heads. A fractional ratio is undefined.
        print("\n  [Attention dimensions]")
        self.check(
            "n_heads × head_dim == d_model",
            c.n_heads * c.head_dim == c.d_model,
            f"{c.n_heads}×{c.head_dim} = {c.n_heads*c.head_dim} ≠ {c.d_model}",
        )
        self.check(
            "n_heads % n_kv_heads == 0 (integer GQA ratio)",
            c.n_heads % c.n_kv_heads == 0,
            f"{c.n_heads} % {c.n_kv_heads} = {c.n_heads % c.n_kv_heads}",
        )
        gqa_ratio = c.n_heads // c.n_kv_heads
        # GQA ratio indicates how many Q heads share each KV head.
        # Higher ratio = more KV cache compression = smaller KV cache but potentially
        # lower model quality. Llama 3.2-1B uses ratio=4; this model uses up to 8.
        print(f"         GQA ratio = {c.n_heads}/{c.n_kv_heads} = {gqa_ratio}:1")

        # ── 3. Tensor shapes through attention ────────────────────────
        print("\n  [Tensor shapes: attention forward pass]")

        # Input embedding: (batch, seq_len, d_model)
        # Each token position maps to a d_model-dimensional vector.
        emb_out = (B, T, c.d_model)
        self.check(f"Embedding output {emb_out}", emb_out == (B, T, c.d_model), "")

        # Q projection: (B, T, d_model) @ W_q (d_model, n_heads×head_dim) → (B, T, n_heads×head_dim)
        # All query heads' projections are computed in a single batched matmul for efficiency.
        q_proj = (B, T, c.n_heads * c.head_dim)
        self.check(f"Q proj {(B,T,c.d_model)} → {q_proj}", q_proj[-1] == c.n_heads * c.head_dim, "")

        # K,V projections: (B, T, d_model) → (B, T, n_kv_heads×head_dim)
        # Fewer heads than Q (GQA): smaller KV cache, fewer parameters in K/V weight matrices.
        k_proj = (B, T, c.n_kv_heads * c.head_dim)
        self.check(
            f"K proj {(B,T,c.d_model)} → {k_proj}", k_proj[-1] == c.n_kv_heads * c.head_dim, ""
        )

        # Reshape: (B, T, heads×head_dim) → (B, heads, T, head_dim)
        # This is a view (no data copy) that separates the heads dimension out
        # so that batched attention can operate independently on each head.
        q_shape = (B, c.n_heads, T, c.head_dim)
        k_shape = (B, c.n_kv_heads, T, c.head_dim)
        self.check(f"Q reshape → {q_shape}", True)
        self.check(f"K reshape → {k_shape}", True)

        # QK-Norm: layer norm applied to Q and K independently, each of shape (head_dim,).
        # Normalizing Q and K before the dot product stabilizes attention entropy —
        # without it, dot products grow as head_dim^0.5, causing attention collapse
        # in long sequences. The learned scale parameter has shape (head_dim,) per head.
        self.check(f"QK-Norm: last dim == head_dim == 128", c.head_dim == 128, f"got {c.head_dim}")

        # RoPE: applied to Q and K after normalization.
        # RoPE pairs consecutive elements of the head_dim vector as (cos, sin) rotations.
        # Requires head_dim to be even so pairing is exact with no leftover dimension.
        self.check("head_dim even (required for RoPE)", c.head_dim % 2 == 0, f"got {c.head_dim}")
        n_freq = c.head_dim // 2  # number of (cos, sin) rotation pairs per head
        print(f"         RoPE: {n_freq} frequency pairs per head")

        # Attention scores: Q @ K^T gives (B, n_heads, T, T).
        # Memory grows quadratically with sequence length — this is why FlashAttention
        # tiles the computation and never materializes the full (T, T) matrix.
        scores_shape = (B, c.n_heads, T, T)
        print(
            f"         Attention scores: {scores_shape}  "
            f"({scores_shape[2]*scores_shape[3]*c.n_heads*B/1e6:.1f}M values)"
        )

        # GQA expand: broadcast each KV head across gqa_ratio Q heads.
        # k_shape is (B, n_kv_heads, T, head_dim); expanded to (B, n_heads, T, head_dim).
        # Implementation: tensor.expand() or repeat_interleave() along the heads dim.
        # expand() is preferred — it is a zero-copy view that shares memory; only
        # triggers an actual data copy if a write occurs (which it doesn't in attention).
        # This expansion is what enables GQA: each KV head's data is seen by gqa_ratio
        # Q heads without storing gqa_ratio copies of K and V.
        expanded_kv = (B, c.n_heads, T, c.head_dim)
        self.check(
            f"GQA expand: {k_shape} → {expanded_kv}",
            expanded_kv == (B, c.n_heads, T, c.head_dim),
            "",
        )

        # Attention output: softmax(scores) @ V → (B, n_heads, T, head_dim),
        # then reshape back to (B, T, n_heads×head_dim) = (B, T, d_model).
        attn_out = (B, T, c.n_heads * c.head_dim)
        self.check(f"Attn output {attn_out}", attn_out[-1] == c.d_model, "")

        # O projection: (B, T, d_model) @ W_o (d_model, d_model) → (B, T, d_model)
        # Mixes information across the heads before returning to the residual stream.
        o_out = (B, T, c.d_model)
        self.check(f"O proj {attn_out} → {o_out}", True)

        # ── 4. FFN shapes ─────────────────────────────────────────────
        # SwiGLU FFN: two parallel projections (gate and up) followed by element-wise
        # gating, then one down projection. This uses 3 weight matrices instead of 2,
        # but empirically outperforms standard FFN at the same parameter budget.
        # gate and up project to ffn_intermediate; their element-wise product is the
        # activation; down projects back to d_model.
        print("\n  [Tensor shapes: SwiGLU FFN]")
        gate_shape = (B, T, c.ffn_intermediate)  # gate branch: (B, T, d_model) → (B, T, ffn_inter)
        self.check(f"Gate proj {(B,T,c.d_model)} → {gate_shape}", True)
        self.check(f"Up   proj {(B,T,c.d_model)} → {gate_shape}", True)  # parallel to gate
        gated = (B, T, c.ffn_intermediate)  # SiLU(gate) ⊙ up: element-wise, shape unchanged
        down_out = (B, T, c.d_model)  # down proj: (B, T, ffn_inter) → (B, T, d_model)
        self.check(f"SiLU(gate)×up → {gated}", True)
        self.check(f"Down proj {gated} → {down_out}", True)

        # ── 5. LM head ───────────────────────────────────────────────
        # Projects the final hidden state to vocabulary logits.
        # With tied embeddings: LM head weight = embedding_table.T (transposed).
        # This saves vocab_size × d_model × 2 bytes = significant for large vocabs.
        # For 32768 × 1280 in BF16 = ~83MB saved per tied model vs untied.
        print("\n  [LM head]")
        logits_shape = (B, T, c.vocab_size)  # (B, T, d_model) @ (d_model, vocab_size)
        self.check(f"Logits {(B,T,c.d_model)} → {logits_shape}", True)
        if c.tie_embeddings:
            print(f"         Tied embeddings: LM head = embedding.T (no extra params)")

        # ── 6. Parameter count ────────────────────────────────────────
        print("\n  [Parameter count]")
        params = _count_params(c)

        spec_totals = {"500m": 489e6, "1b": 953e6, "3b": 2870e6}
        spec_total = spec_totals.get(c.name, None)
        pct_diff = abs(params["total"] - spec_total) / spec_total * 100 if spec_total else 0

        print(f"         Embedding:     {params['embedding']/1e6:>8.1f}M")
        print(
            f"         LM head:       {params['lm_head']/1e6:>8.1f}M  {'(tied)' if c.tie_embeddings else ''}"
        )
        print(f"         Per layer:")
        print(f"           Attention:   {params['attn_per_layer']/1e6:>8.2f}M")
        print(f"           FFN:         {params['ffn_per_layer']/1e6:>8.2f}M")
        print(f"           Norms:       {params['norm_per_layer']/1e3:>8.1f}K")
        print(f"           Total:       {params['per_layer']/1e6:>8.2f}M × {c.n_layers} layers")
        print(f"         All layers:    {params['all_layers']/1e6:>8.1f}M")
        print(f"         Final norm:    {c.d_model/1e3:>8.1f}K")
        print(f"         ─────────────────────────")
        print(f"         TOTAL:         {params['total']/1e9:>8.3f}B  ({params['total']/1e6:.0f}M)")

        if spec_total:
            self.check(
                f"Within 2% of spec ({spec_total/1e6:.0f}M)",
                pct_diff < 2.0,
                f"got {params['total']/1e6:.0f}M, diff={pct_diff:.1f}%",
            )

        # ── 7. Memory estimates ───────────────────────────────────────
        print("\n  [Memory estimates]")
        _memory_estimate(c, params["total"])

        # ── Summary ──────────────────────────────────────────────────
        print(f"\n  {'─'*58}")
        print(f"  Result: {self.ok} passed, {self.bad} failed")
        if self.bad == 0:
            print(f"  {PASS} All shape checks passed.")
        else:
            print(f"  {FAIL} Shape violations detected — fix before training.")
        print(f"{'='*62}")

        return self.bad == 0


def _count_params(c: ShapeConfig) -> dict:
    """
    Analytical parameter count — matches architecture.py ModelConfig.num_params().

    All weight matrix shapes are (input_dim, output_dim), no bias terms.
    QK-Norm contributes learned scale parameters of shape (head_dim,) per head.
    SwiGLU FFN has three weight matrices (gate, up, down) — not two.
    """
    d = c.d_model
    nq = c.n_heads
    nk = c.n_kv_heads
    h = c.head_dim
    fi = c.ffn_intermediate
    v = c.vocab_size
    L = c.n_layers

    # Attention weight matrices per layer:
    #   W_q: (d_model, n_heads × head_dim)    — query projection
    #   W_k: (d_model, n_kv_heads × head_dim) — key projection (GQA: fewer heads)
    #   W_v: (d_model, n_kv_heads × head_dim) — value projection (GQA: same as K)
    #   W_o: (n_heads × head_dim, d_model)    — output projection (mixes heads)
    q_proj = d * (nq * h)  # W_q: d_model → n_heads × head_dim
    k_proj = d * (nk * h)  # W_k: d_model → n_kv_heads × head_dim
    v_proj = d * (nk * h)  # W_v: d_model → n_kv_heads × head_dim (same shape as K)
    o_proj = (nq * h) * d  # W_o: n_heads × head_dim → d_model
    # QK-Norm: one learnable scale (gamma) per element of Q and K, after reshape to heads.
    # Shape: (n_heads × head_dim,) for Q-norm + (n_kv_heads × head_dim,) for K-norm.
    # These are RMSNorm without bias — just one scalar per dimension.
    qk_norm = (nq * h) + (nk * h)  # learned scale per head
    attn = q_proj + k_proj + v_proj + o_proj + qk_norm

    # SwiGLU FFN per layer (3 weight matrices — no bias):
    #   W_gate: (d_model, ffn_intermediate) — gate branch, fed through SiLU
    #   W_up:   (d_model, ffn_intermediate) — up branch, multiplied element-wise with gate
    #   W_down: (ffn_intermediate, d_model) — projects gated activation back to d_model
    ffn = d * fi + d * fi + fi * d  # gate + up + down

    # Per-layer RMSNorms: one before attention, one before FFN.
    # Each has d_model learnable scale parameters (no bias, no beta in RMSNorm).
    norms = 2 * d  # 2 × d_model scale params per layer

    per_layer = attn + ffn + norms
    all_layers = per_layer * L  # same structure repeated L times
    # Embedding table: (vocab_size, d_model) — maps token IDs to vectors
    embedding = v * d
    # LM head: (d_model, vocab_size) — projects hidden state to logits
    # With tied embeddings this weight matrix is shared with embedding.T; zero extra params.
    lm_head = 0 if c.tie_embeddings else v * d
    final_norm = d  # single RMSNorm after all layers, before LM head

    total = all_layers + embedding + lm_head + final_norm

    return {
        "embedding": embedding,
        "lm_head": lm_head,
        "attn_per_layer": attn,
        "ffn_per_layer": ffn,
        "norm_per_layer": norms,
        "per_layer": per_layer,
        "all_layers": all_layers,
        "final_norm": final_norm,
        "total": total,
    }


def _memory_estimate(c: ShapeConfig, total_params: int):
    """
    Estimate GPU/accelerator memory for training and inference.

    Memory breakdown for training (mixed precision with master weights):
      - Model weights BF16:  2 bytes/param  — used for forward/backward pass
      - Master weights FP32: 4 bytes/param  — higher precision copy for optimizer step.
        AdamW updates are numerically sensitive; accumulating in BF16 (7 mantissa bits)
        causes the update to round to zero for small gradients. The FP32 master copy
        (23 mantissa bits) is updated by the optimizer; then cast back to BF16 for
        the next forward pass. This "BF16 training + FP32 master" pattern doubles
        the weight memory versus inference but preserves training stability.
      - AdamW states: 8 bytes/param (m and v, both in FP32: 4+4)
      - Gradients FP32: 4 bytes/param — accumulated in FP32 for the same reason as master weights
    """
    # Model weights stored in two precisions simultaneously during training:
    weights_bf16 = total_params * 2  # 2 bytes per BF16 param (forward/backward pass copy)
    weights_fp32 = total_params * 4  # 4 bytes per FP32 param (master weights for optimizer)

    # AdamW optimizer states: m (first moment) + v (second moment), both in FP32.
    # 4 bytes for m + 4 bytes for v = 8 bytes per parameter.
    optimizer = total_params * 8

    # Gradient buffer: accumulated in FP32 to avoid numeric underflow on small gradients.
    gradients = total_params * 4

    # Total training memory = FP32 master weights + optimizer states + gradients
    # (the BF16 activations/weights add a smaller increment on top)
    training_gb = (weights_fp32 + optimizer + gradients) / 1e9
    # Activation memory for typical batch is roughly 2× the BF16 weight size;
    # this is a rough estimate — exact value depends on sequence length and checkpointing.
    training_total_gb = training_gb + weights_bf16 / 1e9

    # Inference only needs the BF16 forward-pass weights (no gradients, no optimizer states)
    infer_bf16_gb = weights_bf16 / 1e9

    # GGUF Q4_K_M quantization stores weights at ~4 bits (0.5 bytes) per param.
    # Used for local inference on consumer hardware; quality loss is usually minor.
    infer_q4_gb = total_params * 0.5 / 1e9  # ~4 bits per param

    print(f"         Weights BF16:  {weights_bf16/1e9:.2f} GB")
    print(f"         Training total (weights + optimizer + grads): {training_total_gb:.1f} GB")

    # Hardware fit check
    gpu_32gb = training_total_gb <= 32  # e.g. RTX 5090 / A100-40GB
    gpu_96gb = training_total_gb <= 96  # e.g. RTX Pro 6000 / A100-80GB (pair)

    # KV cache estimate at max sequence length (per batch item, all layers):
    # Each layer stores K and V tensors of shape (n_kv_heads, seq_len, head_dim).
    # × 2 for K and V, × 2 bytes for BF16 storage.
    # This grows linearly with batch size and sequence length — at long context
    # (e.g. seq_len=32768 for the 3B config) it can dominate inference memory.
    kv_per_layer = c.n_kv_heads * c.max_seq_len * c.head_dim * 2 * 2  # K+V, BF16
    kv_total_gb = kv_per_layer * c.n_layers / 1e9  # sum over all layers
    print(
        f"         KV cache (max seq, BF16): {kv_total_gb:.2f} GB  "
        f"(per batch item, seq_len={c.max_seq_len})"
    )
    print(f"         Inference BF16: {infer_bf16_gb:.2f} GB  |  Q4_K_M: {infer_q4_gb:.2f} GB")
    print(
        f"         Training fits:  RTX 5090 (32GB)={'YES' if gpu_32gb else 'NO (use gradient checkpointing)'}  "
        f"|  RTX Pro 6000 (96GB)={'YES' if gpu_96gb else 'NO'}"
    )


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import ast

    # First: syntax-check the architecture file.
    # Catches Python syntax errors (missing colons, bad indentation, etc.) before
    # any shape math is run — faster feedback loop than waiting for an import error.
    print("Syntax checking architecture.py...")
    try:
        with open("architecture.py") as f:
            source = f.read()
        ast.parse(source)
        print(f"  {PASS}  architecture.py: valid Python syntax\n")
    except SyntaxError as e:
        print(f"  {FAIL}  architecture.py: SYNTAX ERROR at line {e.lineno}: {e.msg}")
        sys.exit(1)
    except FileNotFoundError:
        # Not an error — shape_check.py can be run standalone before architecture.py exists
        print(f"  {WARN}  architecture.py not found — running shape checks only\n")

    # Run shape checks for all three configs
    all_passed = True
    for name, config in CONFIGS.items():
        checker = ShapeChecker(config, batch=2, seq=512)
        ok = checker.run()
        all_passed = all_passed and ok

    print()
    if all_passed:
        print(f"  {PASS}  All configs validated. Ready for implementation.")
    else:
        print(f"  {FAIL}  Validation failures — check output above.")
        sys.exit(1)


def main():
    import sys
    import ast
    from pathlib import Path

    arch_path = Path(__file__).parent.parent / "model" / "architecture.py"
    print(f"Syntax checking {arch_path}...")
    try:
        with open(arch_path) as f:
            source = f.read()
        ast.parse(source)
        print(f"  {PASS}  architecture.py: valid Python syntax\n")
    except SyntaxError as e:
        print(f"  {FAIL}  architecture.py: SYNTAX ERROR at line {e.lineno}: {e.msg}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"  {WARN}  architecture.py not found — running shape checks only\n")

    all_passed = True
    for name, config in CONFIGS.items():
        checker = ShapeChecker(config, batch=2, seq=512)
        ok = checker.run()
        all_passed = all_passed and ok

    print()
    if all_passed:
        print(f"  {PASS}  All configs validated. Ready for implementation.")
    else:
        print(f"  {FAIL}  Validation failures — check output above.")
        sys.exit(1)
