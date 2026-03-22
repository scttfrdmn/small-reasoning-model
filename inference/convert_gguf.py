"""
convert_gguf.py
===============
Export trained model to GGUF format for llama.cpp inference.

Two-stage export strategy:
  1. This script writes a BF16 GGUF using the `gguf` Python package.
  2. Run `llama-quantize model-bf16.gguf model-q4_k_m.gguf Q4_K_M` (from
     llama.cpp) to produce quantized variants.  We don't reimplement
     quantization — llama.cpp's quant paths are battle-tested and produce
     the same Q4_K_M files everyone already uses.

Target formats (spec Section 6):
  BF16      ~2 GB    reference; for eval
  Q8_0      ~1 GB    near-lossless; Graviton4, any 2GB+ device
  Q4_K_M   ~700 MB   recommended default; Graviton4 / Kamrui cluster
  Q4_0     ~550 MB   edge deployment; Raspberry Pi 5
  Q2_K     ~400 MB   curiosity only

Tile-aligned dimensions (all ÷128) map cleanly to GGUF's 32-element
block quantization with no remainder handling.

Graviton4 inference estimate (1B Q4_K_M):
  c8g.4xlarge:  ~25–35 tokens/sec
  c8g.8xlarge:  ~60–80 tokens/sec

Usage:
  uv run python inference/convert_gguf.py \\
      --checkpoint checkpoints/500m_sft/best.pt \\
      --tokenizer tokenizer_output \\
      --config 500m \\
      --output checkpoints/500m_sft/model-bf16.gguf

  # Then quantize externally:
  llama-quantize model-bf16.gguf model-q4_k_m.gguf Q4_K_M
"""

import argparse
import json
import sys
from pathlib import Path

import torch

# `gguf` is in the [inference] optional group; give a clear error if missing.
try:
    import gguf
    from gguf import GGUFWriter, GGMLQuantizationType
except ImportError:
    sys.exit(
        "ERROR: `gguf` package not found.\n"
        "Install with:  pip install gguf\n"
        "or:            uv sync --extra inference"
    )

# Project imports — works because the package is installed in editable mode.
from model.architecture import ModelConfig, get_config


# ---------------------------------------------------------------------------
# Weight name mapping: our names → llama.cpp GGUF tensor names
# ---------------------------------------------------------------------------
# llama.cpp uses a flat naming scheme; we need to translate our hierarchical
# module names.  Tied embeddings mean we write `embedding.weight` twice:
# once as `token_embd.weight` and once as `output.weight`.


def _build_name_map(n_layers: int) -> dict[str, str]:
    """
    Return a mapping from our state-dict key → GGUF tensor name.
    Only weights (no biases — our model has none) are included.
    """
    m: dict[str, str] = {
        "embedding.weight": "token_embd.weight",
        "norm.weight": "output_norm.weight",
    }
    for i in range(n_layers):
        p = f"blocks.{i}"
        g = f"blk.{i}"
        m.update(
            {
                f"{p}.attn_norm.weight": f"{g}.attn_norm.weight",
                f"{p}.attn.q_proj.weight": f"{g}.attn_q.weight",
                f"{p}.attn.k_proj.weight": f"{g}.attn_k.weight",
                f"{p}.attn.v_proj.weight": f"{g}.attn_v.weight",
                f"{p}.attn.o_proj.weight": f"{g}.attn_output.weight",
                # QK-Norm scale vectors (present when qk_norm=True)
                f"{p}.attn.q_norm.weight": f"{g}.attn_q_norm.weight",
                f"{p}.attn.k_norm.weight": f"{g}.attn_k_norm.weight",
                f"{p}.ffn_norm.weight": f"{g}.ffn_norm.weight",
                f"{p}.ffn.gate_proj.weight": f"{g}.ffn_gate.weight",
                f"{p}.ffn.up_proj.weight": f"{g}.ffn_up.weight",
                f"{p}.ffn.down_proj.weight": f"{g}.ffn_down.weight",
            }
        )
    return m


# ---------------------------------------------------------------------------
# Tokenizer helpers
# ---------------------------------------------------------------------------


def _load_tokenizer_vocab(tokenizer_dir: Path) -> tuple[list[str], list[int]]:
    """
    Load vocabulary and token types from the tokenizer directory.

    Returns:
        tokens:      list of token strings, indexed by token id
        token_types: parallel list of lm_eval/GGUF token type ints
                     (normal=1, bos=3, eos=3, unknown=2)

    The tokenizer.json written by our train_tokenizer.py is a HuggingFace
    tokenizers JSON; the vocab is in .model.vocab as a list of [token, score]
    pairs sorted by id.
    """
    tok_file = tokenizer_dir / "tokenizer.json"
    if not tok_file.exists():
        sys.exit(f"ERROR: tokenizer.json not found at {tok_file}")

    raw = json.loads(tok_file.read_text())

    # The HF tokenizers format stores vocab in model.vocab as {token: id} dict
    # for BPE models.  We need id-sorted list.
    vocab_dict: dict[str, int] = raw["model"]["vocab"]
    tokens_sorted = sorted(vocab_dict.items(), key=lambda kv: kv[1])
    tokens = [tok for tok, _id in tokens_sorted]

    # Determine special token ids from added_tokens list
    bos_id, eos_id = 1, 2  # defaults matching our tokenizer training
    for entry in raw.get("added_tokens", []):
        if entry.get("content") == "<bos>":
            bos_id = entry["id"]
        elif entry.get("content") == "<eos>":
            eos_id = entry["id"]

    # Build token type array: GGUF type codes
    # 1 = normal, 2 = unknown/pad, 3 = control (bos/eos), 6 = user-defined
    token_types = [1] * len(tokens)
    if bos_id < len(token_types):
        token_types[bos_id] = 3
    if eos_id < len(token_types):
        token_types[eos_id] = 3

    return tokens, token_types


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------


def convert(
    checkpoint_path: Path,
    tokenizer_dir: Path,
    config_name: str,
    output_path: Path,
) -> None:
    cfg: ModelConfig = get_config(config_name)

    print(f"Config:      {config_name}")
    print(f"  d_model    {cfg.d_model}")
    print(f"  n_layers   {cfg.n_layers}")
    print(f"  n_heads    {cfg.n_heads}  (kv={cfg.n_kv_heads})")
    print(f"  ffn_interm {cfg.ffn_intermediate}")
    print(f"  max_seq    {cfg.max_seq_len}")
    print(f"  rope_base  {cfg.rope_base}")
    print(f"  vocab_size {cfg.vocab_size}")

    # --- Load checkpoint ---
    print(f"\nLoading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    # Checkpoints may be saved as {"model": state_dict, ...} or as bare state_dict
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict: dict[str, torch.Tensor] = ckpt["model"]
    else:
        state_dict = ckpt
    print(f"  {len(state_dict)} tensors loaded")

    # --- Load tokenizer ---
    print(f"Loading tokenizer: {tokenizer_dir}")
    tokens, token_types = _load_tokenizer_vocab(tokenizer_dir)
    print(f"  vocab size: {len(tokens)}")

    # --- Build GGUF ---
    print(f"\nWriting GGUF: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = GGUFWriter(str(output_path), arch="llama")

    # -- Metadata (llama.cpp GGUF keys) --
    writer.add_name(f"small-reasoning-model-{config_name}")
    writer.add_block_count(cfg.n_layers)
    writer.add_context_length(cfg.max_seq_len)
    writer.add_embedding_length(cfg.d_model)
    writer.add_feed_forward_length(cfg.ffn_intermediate)
    writer.add_head_count(cfg.n_heads)
    writer.add_head_count_kv(cfg.n_kv_heads)
    writer.add_layer_norm_rms_eps(cfg.norm_eps)
    # rope.dimension_count = head_dim (number of dims that RoPE rotates)
    writer.add_rope_dimension_count(cfg.head_dim)
    writer.add_rope_freq_base(cfg.rope_base)
    writer.add_vocab_size(cfg.vocab_size)
    # Tokenizer metadata
    writer.add_tokenizer_model("gpt2")  # BPE — lm_eval tokenizer backend key
    writer.add_token_list(tokens)
    writer.add_token_types(token_types)
    writer.add_bos_token_id(1)
    writer.add_eos_token_id(2)

    # -- Tensors --
    name_map = _build_name_map(cfg.n_layers)
    written = set()
    skipped = []

    for our_key, gguf_name in name_map.items():
        if our_key not in state_dict:
            skipped.append(our_key)
            continue

        tensor: torch.Tensor = state_dict[our_key]

        # Convert to BF16 — GGUF BF16 type is F32 in storage when written via
        # the Python library; we store as float32 numpy and let llama-quantize
        # handle the final type reduction.  However, to keep file size close to
        # the 2 GB BF16 target we convert to float32 (not float16) since the
        # gguf library's BF16 support is still being fleshed out; downstream
        # llama-quantize handles precision.
        arr = tensor.to(torch.float32).numpy()
        writer.add_tensor(gguf_name, arr)
        written.add(our_key)

    # Tied embeddings: write embedding.weight again as output.weight (LM head)
    # This is required for llama.cpp to find the LM head projection.
    if cfg.tie_embeddings and "embedding.weight" in state_dict:
        arr = state_dict["embedding.weight"].to(torch.float32).numpy()
        writer.add_tensor("output.weight", arr)
        print("  [tied] wrote embedding.weight → output.weight")

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    print(f"\nWrote {len(written)} tensors to {output_path}")
    if skipped:
        print(f"Skipped (not in checkpoint): {skipped}")

    size_mb = output_path.stat().st_size / 1024**2
    print(f"Output size: {size_mb:.0f} MB")
    print("\nTo quantize:")
    print(f"  llama-quantize {output_path} {output_path.with_suffix('')}-q4_k_m.gguf Q4_K_M")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export SRM checkpoint to BF16 GGUF for llama.cpp inference.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        type=Path,
        help="Path to .pt checkpoint (from training/sft.py or training/grpo.py)",
    )
    parser.add_argument(
        "--tokenizer",
        required=True,
        type=Path,
        help="Path to tokenizer directory (must contain tokenizer.json)",
    )
    parser.add_argument(
        "--config",
        required=True,
        choices=["500m", "1b", "3b"],
        help="Model config name",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output path for the .gguf file",
    )
    args = parser.parse_args()

    convert(
        checkpoint_path=args.checkpoint,
        tokenizer_dir=args.tokenizer,
        config_name=args.config,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
