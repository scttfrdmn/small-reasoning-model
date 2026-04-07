"""
serve.py
========
Simple inference server for the small reasoning model.

Loads a checkpoint at startup and serves a minimal HTTP API for single-turn
generation.

Endpoints:
  POST /generate   {"prompt": "...", "max_tokens": 512, "temperature": 0.8}
  GET  /health

Usage:
  uv run srm-serve \\
    --checkpoint checkpoints/500m_sft/best.pt \\
    --config 500m \\
    --tokenizer tokenizer_output \\
    --host 0.0.0.0 \\
    --port 8080
"""

import argparse
from contextlib import asynccontextmanager
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from model.architecture import SmallReasoningModel, get_config
from model.kv_compress import forward_compressed, kv_cache_memory_report
from tokenizer.train_tokenizer import load_tokenizer

# ---------------------------------------------------------------------------
# Global server state (populated in lifespan, read by endpoint handlers)
# ---------------------------------------------------------------------------

# Module-level globals rather than a singleton class — FastAPI's DI system
# is heavier than needed for a single model.  Lifespan is the only writer;
# request handlers only read.
_model: Optional[SmallReasoningModel] = None
_tokenizer = None
_device: Optional[torch.device] = None
_dtype: torch.dtype = torch.bfloat16
_max_seq_len: int = 4096
_eos_id: int = 2
_compress_kv: bool = False  # TurboQuant KV cache compression


# ---------------------------------------------------------------------------
# Lifespan: load model once at startup, release at shutdown
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager (replaces deprecated on_startup events).

    Everything before `yield` runs at startup; everything after at shutdown.
    """
    global _model, _tokenizer, _device, _dtype, _max_seq_len, _eos_id, _compress_kv

    args = app.state.args  # stashed by main() before calling uvicorn.run

    # Auto-select device: CUDA > MPS > CPU (same heuristic as harness.py)
    if args.device:
        _device = torch.device(args.device)
    elif torch.cuda.is_available():
        _device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        _device = torch.device("mps")
    else:
        _device = torch.device("cpu")

    _dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    print(f"[srm-serve] Loading tokenizer from {args.tokenizer} …")
    _tokenizer = load_tokenizer(args.tokenizer)
    _eos_id = _tokenizer.token_to_id("<eos>") or 2

    print(f"[srm-serve] Loading {args.config} model from {args.checkpoint} …")
    cfg = get_config(args.config)
    _max_seq_len = cfg.max_seq_len

    _model = SmallReasoningModel(cfg)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    _model.load_state_dict(state_dict)
    _model.to(dtype=_dtype, device=_device)
    _model.eval()

    _compress_kv = args.compress_kv
    compress_label = "TurboQuant 2×" if _compress_kv else "off"
    print(
        f"[srm-serve] Ready on {_device} ({args.dtype})  "
        f"vocab={_tokenizer.get_vocab_size()}  max_seq={_max_seq_len}  "
        f"kv_compress={compress_label}"
    )

    yield  # server runs here

    print("[srm-serve] Shutting down.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="SmallReasoningModel inference server", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)


class GenerateResponse(BaseModel):
    text: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    """Liveness probe — returns 200 once the model is loaded."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok"}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    """
    Autoregressive text generation with optional temperature sampling.

    temperature=0 → greedy argmax (deterministic, good for benchmarks).
    temperature>0 → divide logits by temperature then multinomial sample.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    text = _generate(req.prompt, req.max_tokens, req.temperature)
    return GenerateResponse(text=text)


# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------


@torch.inference_mode()
def _generate(prompt: str, max_new_tokens: int, temperature: float) -> str:
    """
    KV-cache autoregressive generation with optional TurboQuant compression.

    Prefill runs once over the full prompt to populate the KV cache, then
    each decode step processes a single token — keeping per-step cost
    O(cache size) rather than O(sequence_length²).

    When _compress_kv=True (--compress-kv flag), each layer's KV cache is
    compressed to ~2× smaller using PolarQuant (K) + INT8 (V) between decode
    steps. This halves the memory occupied by the KV cache during generation,
    enabling longer contexts or more concurrent sessions on the same hardware.
    """
    ctx_ids = _tokenizer.encode(prompt).ids
    # The tokenizer post-processor appends EOS to every encode() call.
    # Strip it here so the model generates a response rather than predicting
    # tokens after an end-of-sequence marker (which produces garbage).
    if ctx_ids and ctx_ids[-1] == _eos_id:
        ctx_ids = ctx_ids[:-1]
    # Truncate context to leave room for generated tokens within the model window
    ctx_ids = ctx_ids[-(_max_seq_len - max_new_tokens) :]

    input_ids = torch.tensor([ctx_ids], dtype=torch.long, device=_device)
    generated_ids: list[int] = []

    autocast_ctx = torch.amp.autocast(
        # MPS does not support bfloat16 autocast natively — use CPU dtype path
        device_type=_device.type if _device.type != "mps" else "cpu",
        dtype=_dtype,
        enabled=(_dtype == torch.bfloat16),
    )

    with autocast_ctx:
        # Prefill: single forward pass over full context to populate KV cache.
        # kv_caches=[] signals collect-but-no-prior-cache; the returned list
        # holds per-layer (K, V) tensors that decode step 1 will prepend to.
        logits, kv_caches = _model(input_ids, kv_caches=[])  # (1, T_ctx, V)

        # Optionally compress the prefill KV caches immediately after generation
        if _compress_kv and kv_caches is not None:
            from model.kv_compress import compress_kv_caches
            kv_caches = compress_kv_caches(kv_caches)

        next_logits = logits[0, -1, :]  # (V,) — logits for the first new token

        for step_i in range(max_new_tokens):
            if temperature == 0.0:
                # Greedy: deterministic argmax
                next_id = int(next_logits.argmax(dim=-1).item())
            else:
                # Temperature scaling then multinomial sample.
                # Cast to float32 before softmax for numerical stability on BF16.
                probs = torch.softmax(next_logits.float() / temperature, dim=-1)
                next_id = int(torch.multinomial(probs, num_samples=1).item())

            generated_ids.append(next_id)

            if next_id == _eos_id:
                break

            # Decode step: single new token, reusing the populated KV cache.
            # When kv_compress is enabled, forward_compressed handles the
            # decompress-forward-recompress cycle transparently.
            next_input = torch.tensor([[next_id]], dtype=torch.long, device=_device)
            if _compress_kv:
                logits, kv_caches = forward_compressed(
                    _model, next_input, kv_caches, position_offset=len(ctx_ids) + step_i
                )
            else:
                logits, kv_caches = _model(next_input, kv_caches=kv_caches)
            next_logits = logits[0, -1, :]

    return _tokenizer.decode(generated_ids)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve SmallReasoningModel via HTTP.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint file")
    parser.add_argument(
        "--config", default="500m", choices=["500m", "1b", "3b"], help="Model config"
    )
    parser.add_argument(
        "--tokenizer", default="tokenizer_output", help="Path to tokenizer directory"
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", default=8080, type=int, help="Bind port (default: 8080)")
    parser.add_argument("--device", default=None, help="Device override (default: auto)")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float32"],
        help="Weight dtype (default: bfloat16)",
    )
    parser.add_argument(
        "--compress-kv",
        dest="compress_kv",
        action="store_true",
        default=False,
        help="[TurboQuant] Compress KV cache ~2× using PolarQuant+INT8 (no accuracy loss)",
    )
    args = parser.parse_args()

    # Stash args on app.state so the lifespan function can read them.
    # This is the standard FastAPI pattern for passing startup config.
    app.state.args = args
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
