"""
serve.py
========
Simple inference server for the small reasoning model.

Loads a checkpoint (or GGUF via llama-cpp-python) and serves
a minimal HTTP API for single-turn generation.

Endpoints:
  POST /generate   {"prompt": "...", "max_tokens": 512, "temperature": 0.8}
  GET  /health

Status: STUB
"""

# TODO: implement inference server (FastAPI or plain http.server)
