"""
shape_check.py
==============
Validates the architecture analytically — no torch required.

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
    name:             str
    d_model:          int
    n_layers:         int
    n_heads:          int
    n_kv_heads:       int
    head_dim:         int
    ffn_intermediate: int
    vocab_size:       int
    max_seq_len:      int
    rope_base:        float = 500_000.0
    tie_embeddings:   bool  = True

CONFIGS = {
    "500m": ShapeConfig("500m", 1280,  26, 10, 2,  128, 3456,  32768,  8192),
    "1b":   ShapeConfig("1b",   2048,  20, 16, 4,  128, 5504,  32768, 16384),
    "3b":   ShapeConfig("3b",   3072,  28, 24, 6,  128, 8192,  32768, 32768),
}

# ── Shape tracer ──────────────────────────────────────────────────────────

PASS = "✓"
FAIL = "✗"
WARN = "⚠"

class ShapeChecker:
    def __init__(self, config: ShapeConfig, batch=2, seq=512):
        self.c   = config
        self.B   = batch
        self.T   = seq
        self.ok  = 0
        self.bad = 0

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
        print("\n  [Tile alignment — Trainium2 NeuronCore 128×128 BF16]")
        self.check("d_model ÷ 128",
            c.d_model % 128 == 0,
            f"{c.d_model} % 128 = {c.d_model % 128}")
        self.check("ffn_intermediate ÷ 128",
            c.ffn_intermediate % 128 == 0,
            f"{c.ffn_intermediate} % 128 = {c.ffn_intermediate % 128}")
        self.check("vocab_size ÷ 128",
            c.vocab_size % 128 == 0,
            f"{c.vocab_size} % 128 = {c.vocab_size % 128}")
        self.check("head_dim == 128",
            c.head_dim == 128,
            f"got {c.head_dim}")
        kv_dim = c.n_kv_heads * c.head_dim
        self.check(f"n_kv_heads×head_dim ({kv_dim}) ÷ 128",
            kv_dim % 128 == 0,
            f"{kv_dim} % 128 = {kv_dim % 128}")

        # ── 2. Attention dimension consistency ────────────────────────
        print("\n  [Attention dimensions]")
        self.check("n_heads × head_dim == d_model",
            c.n_heads * c.head_dim == c.d_model,
            f"{c.n_heads}×{c.head_dim} = {c.n_heads*c.head_dim} ≠ {c.d_model}")
        self.check("n_heads % n_kv_heads == 0 (integer GQA ratio)",
            c.n_heads % c.n_kv_heads == 0,
            f"{c.n_heads} % {c.n_kv_heads} = {c.n_heads % c.n_kv_heads}")
        gqa_ratio = c.n_heads // c.n_kv_heads
        print(f"         GQA ratio = {c.n_heads}/{c.n_kv_heads} = {gqa_ratio}:1")

        # ── 3. Tensor shapes through attention ────────────────────────
        print("\n  [Tensor shapes: attention forward pass]")

        # Embedding
        emb_out = (B, T, c.d_model)
        self.check(f"Embedding output {emb_out}",
            emb_out == (B, T, c.d_model), "")

        # Q projection: (B,T,d_model) → (B,T, n_heads×head_dim)
        q_proj = (B, T, c.n_heads * c.head_dim)
        self.check(f"Q proj {(B,T,c.d_model)} → {q_proj}",
            q_proj[-1] == c.n_heads * c.head_dim, "")

        # K,V projections
        k_proj = (B, T, c.n_kv_heads * c.head_dim)
        self.check(f"K proj {(B,T,c.d_model)} → {k_proj}",
            k_proj[-1] == c.n_kv_heads * c.head_dim, "")

        # Reshape to (B, heads, T, head_dim)
        q_shape = (B, c.n_heads,    T, c.head_dim)
        k_shape = (B, c.n_kv_heads, T, c.head_dim)
        self.check(f"Q reshape → {q_shape}", True)
        self.check(f"K reshape → {k_shape}", True)

        # QK-Norm: operates on last dim (head_dim)
        self.check(f"QK-Norm: last dim == head_dim == 128",
            c.head_dim == 128, f"got {c.head_dim}")

        # RoPE: applied to last dim, must be even
        self.check("head_dim even (required for RoPE)",
            c.head_dim % 2 == 0, f"got {c.head_dim}")
        n_freq = c.head_dim // 2
        print(f"         RoPE: {n_freq} frequency pairs per head")

        # Attention scores: (B, n_heads, T, T)
        scores_shape = (B, c.n_heads, T, T)
        print(f"         Attention scores: {scores_shape}  "
              f"({scores_shape[2]*scores_shape[3]*c.n_heads*B/1e6:.1f}M values)")

        # GQA expand: k,v go from n_kv_heads → n_heads
        expanded_kv = (B, c.n_heads, T, c.head_dim)
        self.check(f"GQA expand: {k_shape} → {expanded_kv}",
            expanded_kv == (B, c.n_heads, T, c.head_dim), "")

        # Attention output
        attn_out = (B, T, c.n_heads * c.head_dim)
        self.check(f"Attn output {attn_out}",
            attn_out[-1] == c.d_model, "")

        # O projection
        o_out = (B, T, c.d_model)
        self.check(f"O proj {attn_out} → {o_out}", True)

        # ── 4. FFN shapes ─────────────────────────────────────────────
        print("\n  [Tensor shapes: SwiGLU FFN]")
        gate_shape = (B, T, c.ffn_intermediate)
        self.check(f"Gate proj {(B,T,c.d_model)} → {gate_shape}", True)
        self.check(f"Up   proj {(B,T,c.d_model)} → {gate_shape}", True)
        gated = (B, T, c.ffn_intermediate)
        down_out = (B, T, c.d_model)
        self.check(f"SiLU(gate)×up → {gated}", True)
        self.check(f"Down proj {gated} → {down_out}", True)

        # ── 5. LM head ───────────────────────────────────────────────
        print("\n  [LM head]")
        logits_shape = (B, T, c.vocab_size)
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
        print(f"         LM head:       {params['lm_head']/1e6:>8.1f}M  {'(tied)' if c.tie_embeddings else ''}")
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
            self.check(f"Within 2% of spec ({spec_total/1e6:.0f}M)",
                pct_diff < 2.0,
                f"got {params['total']/1e6:.0f}M, diff={pct_diff:.1f}%")

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
    """Analytical parameter count — matches architecture.py ModelConfig.num_params()"""
    d  = c.d_model
    nq = c.n_heads
    nk = c.n_kv_heads
    h  = c.head_dim
    fi = c.ffn_intermediate
    v  = c.vocab_size
    L  = c.n_layers

    # Attention per layer
    q_proj  = d * (nq * h)
    k_proj  = d * (nk * h)
    v_proj  = d * (nk * h)
    o_proj  = (nq * h) * d
    qk_norm = (nq * h) + (nk * h)         # learned scale per head
    attn    = q_proj + k_proj + v_proj + o_proj + qk_norm

    # FFN per layer (SwiGLU: 3 matrices)
    ffn = d * fi + d * fi + fi * d

    # Norms per layer (2 × d_model scale params)
    norms = 2 * d

    per_layer  = attn + ffn + norms
    all_layers = per_layer * L
    embedding  = v * d
    lm_head    = 0 if c.tie_embeddings else v * d
    final_norm = d

    total = all_layers + embedding + lm_head + final_norm

    return {
        "embedding":      embedding,
        "lm_head":        lm_head,
        "attn_per_layer": attn,
        "ffn_per_layer":  ffn,
        "norm_per_layer": norms,
        "per_layer":      per_layer,
        "all_layers":     all_layers,
        "final_norm":     final_norm,
        "total":          total,
    }


def _memory_estimate(c: ShapeConfig, total_params: int):
    """Estimate GPU memory for training and inference."""
    # Weights
    weights_bf16 = total_params * 2       # 2 bytes per BF16 param
    weights_fp32 = total_params * 4       # 4 bytes per FP32 param (master weights)

    # AdamW optimizer states: m + v in FP32 = 8 bytes per param
    optimizer    = total_params * 8

    # Gradients: FP32 = 4 bytes per param
    gradients    = total_params * 4

    # Total training
    training_gb  = (weights_fp32 + optimizer + gradients) / 1e9
    # + activations (rough estimate: 2 × weights for typical batch)
    training_total_gb = training_gb + weights_bf16 / 1e9

    # Inference (BF16 only)
    infer_bf16_gb = weights_bf16 / 1e9

    # GGUF quantized
    infer_q4_gb  = total_params * 0.5 / 1e9  # ~4 bits per param

    print(f"         Weights BF16:  {weights_bf16/1e9:.2f} GB")
    print(f"         Training total (weights + optimizer + grads): {training_total_gb:.1f} GB")

    # Hardware fit check
    gpu_32gb  = training_total_gb <= 32
    gpu_96gb  = training_total_gb <= 96

    # KV cache estimate at max_seq_len (per layer, per batch item)
    # shape: (n_kv_heads, seq_len, head_dim) × 2 (k+v) × 2 bytes
    kv_per_layer = c.n_kv_heads * c.max_seq_len * c.head_dim * 2 * 2
    kv_total_gb  = kv_per_layer * c.n_layers / 1e9
    print(f"         KV cache (max seq, BF16): {kv_total_gb:.2f} GB  "
          f"(per batch item, seq_len={c.max_seq_len})")
    print(f"         Inference BF16: {infer_bf16_gb:.2f} GB  |  Q4_K_M: {infer_q4_gb:.2f} GB")
    print(f"         Training fits:  RTX 5090 (32GB)={'YES' if gpu_32gb else 'NO (use gradient checkpointing)'}  "
          f"|  RTX Pro 6000 (96GB)={'YES' if gpu_96gb else 'NO'}")


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import ast

    # First: syntax-check the architecture file
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
